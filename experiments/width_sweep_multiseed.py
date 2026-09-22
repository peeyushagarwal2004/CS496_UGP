"""Multi-seed width sweep: does F21's capacity ordering survive model-to-model variance?

F22 found the single-seed width curves badly non-monotone -- ``e4m3`` at w=32 fell to
0.0024, ``e3m2`` at w=24 spiked to 0.5366 -- and traced it to having one trained model per
width.  This pools several independently trained models per width: seed 0 is the original
sweep, seeds 1.. are retrained with a different initialisation and data order.

Paired design
-------------
Fault sites are drawn with a fixed RNG, and tensor shapes depend only on width, so every
seed of a given width is probed at the *same* weight positions and bits.  All three formats
are also evaluated on the *same* trained model.  So:

* differences between seeds come from the trained weights, not from which faults were drawn
* differences between formats within a seed are paired, which cancels model-to-model
  variance -- the variance that made the single-seed curves unusable

Tests
-----
1. Endpoint ordering per seed: does SDC(w48)/SDC(w8) rank e3m2 < e4m3 < e5m2?
2. Capacity slopes: OLS slope of log(element SDC) on log(params) per seed and format, and
   the *paired* between-format slope differences that F18 predicts are negative.
3. Variance decomposition: between-seed spread vs injection-sampling uncertainty.
4. The F18 control: non-finite rate flat in width, now with seed error bars.

Run::

    .venv/Scripts/python.exe -m experiments.width_sweep_multiseed
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.models import build_model, count_parameters
from mxfi.train import CHECKPOINT_DIR, checkpoint_stem

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIDTHS = [8, 16, 24, 32, 48]
FORMATS = ["e3m2", "e4m3", "e5m2"]          # ascending special-code count
SPECIAL_CODES = {"e3m2": 0, "e4m3": 2, "e5m2": 8}
N_INJECTIONS = 3000
MAX_SEED = 10


def _tag(model: str, seed: int) -> str:
    return model if seed == 0 else f"{model}_s{seed}"


def _section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def _mean_sd(v: pd.Series) -> str:
    return f"{v.mean():.4f}+-{v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"


def collect() -> pd.DataFrame:
    rows = []
    for w in WIDTHS:
        model = f"resnet8_w{w}"
        params = count_parameters(build_model(model))
        for seed in range(MAX_SEED):
            log = CHECKPOINT_DIR / f"{checkpoint_stem(model, seed)}.log.json"
            acc = json.loads(log.read_text())[-1]["test_acc"] if log.exists() else np.nan
            for fmt in FORMATS:
                p = RESULTS / f"e01_{_tag(model, seed)}_{fmt}-K32-ocp-w-n{N_INJECTIONS}.csv"
                if not p.exists():
                    continue
                d = pd.read_csv(p)
                if len(d) < N_INJECTIONS:           # campaign still flushing
                    continue
                e = d[d.site == "element"]
                failed = e[e.sdc]
                rows.append({
                    "width": w, "params": params, "seed": seed, "fmt": fmt,
                    "accuracy": acc,
                    "element_sdc": float(e.sdc.mean()),
                    "perturbation_sdc": float((e.sdc & ~e.nonfinite).mean()),
                    "nonfinite_rate": float(e.nonfinite.mean()),
                    "nonfinite_share": (float(failed.nonfinite.mean())
                                        if len(failed) else np.nan),
                    "n_element": len(e),
                })
    return pd.DataFrame(rows)


def main() -> None:
    df = collect()
    if df.empty:
        print("no width-sweep results yet")
        return

    need = len(WIDTHS) * len(FORMATS)
    counts = df.groupby("seed").size()
    seeds = sorted(counts[counts == need].index)
    partial = sorted(set(df.seed) - set(seeds))
    print(f"complete seeds: {seeds}"
          + (f"   (partial, excluded: {partial})" if partial else ""))
    if len(seeds) < 2:
        print("need at least 2 complete seeds for a multi-seed analysis")
        return

    df = df[df.seed.isin(seeds)].copy()
    df.to_csv(RESULTS / "width_sweep_multiseed.csv", index=False)
    k = len(seeds)
    pw = df.drop_duplicates("width").set_index("width").params.sort_index()

    _section(f"accuracy by width -- mean +- std over {k} seeds")
    print(df.drop_duplicates(["width", "seed"]).groupby("width").accuracy
            .apply(_mean_sd).to_string())

    for col, title in (("element_sdc", "element SDC"),
                       ("perturbation_sdc", "perturbation-driven SDC (excl. non-finite)"),
                       ("nonfinite_rate", "non-finite rate"),
                       ("nonfinite_share", "share of failures that were non-finite")):
        _section(f"{title} -- mean +- std over {k} seeds")
        print(df.groupby(["fmt", "width"])[col].apply(_mean_sd)
                .unstack("width").reindex(FORMATS).to_string())

    # ---- 1. endpoint ordering, per seed
    lo, hi = min(WIDTHS), max(WIDTHS)
    _section(f"TEST 1  endpoint ordering per seed: SDC(w{hi}) / SDC(w{lo})")
    held, ratios = 0, []
    for s in seeds:
        g = df[df.seed == s].set_index(["fmt", "width"]).element_sdc
        r = {f: g[(f, hi)] / max(g[(f, lo)], 1e-9) for f in FORMATS}
        ok = r["e3m2"] < r["e4m3"] < r["e5m2"]
        held += ok
        ratios.append(r)
        print(f"  seed {s}: " + "  ".join(f"{f} {r[f]:.3f}x" for f in FORMATS)
              + ("   ordered" if ok else "   NOT ordered"))
    rt = pd.DataFrame(ratios)
    print("  mean   : " + "  ".join(f"{f} {rt[f].mean():.3f}+-{rt[f].std(ddof=1):.3f}x"
                                    for f in FORMATS))
    p_chance = sum(comb(k, j) * (1 / 6) ** j * (5 / 6) ** (k - j)
                   for j in range(held, k + 1))
    print(f"\n  predicted ordering held in {held}/{k} seeds; a random ordering matches "
          f"with p=1/6, so P(>= {held} of {k} by chance) = {p_chance:.1e}")

    # ---- 2. capacity slopes, paired within seed
    _section("TEST 2  capacity slope: d log(element SDC) / d log(params)")
    slopes = {}
    for s in seeds:
        for f in FORMATS:
            g = df[(df.seed == s) & (df.fmt == f)].sort_values("params")
            y = np.log(g.element_sdc + 0.5 / g.n_element)   # continuity for zeros
            slopes[(s, f)] = np.polyfit(np.log(g.params), y, 1)[0]
    sl = pd.Series(slopes).unstack()[FORMATS]
    for f in FORMATS:
        v = sl[f]
        print(f"  {f} ({SPECIAL_CODES[f]} special codes): slope {v.mean():+.3f} "
              f"+- {v.std(ddof=1) / np.sqrt(k):.3f} se   per seed: "
              + " ".join(f"{x:+.2f}" for x in v))
    print("\n  F18 predicts steeper (more negative) slopes for fewer special codes.")
    print("  Differences are paired within a seed (same trained model), so model variance cancels:")
    for a, b in (("e3m2", "e4m3"), ("e4m3", "e5m2"), ("e3m2", "e5m2")):
        d = sl[a] - sl[b]
        se = d.std(ddof=1) / np.sqrt(k)
        t = d.mean() / se if se > 0 else float("nan")
        print(f"    slope[{a}] - slope[{b}] = {d.mean():+.3f} +- {se:.3f} se, "
              f"t = {t:+.2f}, negative in {int((d < 0).sum())}/{k} seeds")
    sl.to_csv(RESULTS / "width_sweep_multiseed_slopes.csv")

    # ---- 3. variance decomposition
    _section("TEST 3  where the single-seed scatter came from")
    g = df.groupby(["fmt", "width"]).agg(p=("element_sdc", "mean"),
                                         between=("element_sdc", lambda v: v.std(ddof=1)),
                                         n=("n_element", "mean"))
    p = np.clip(g.p, 0.5 / g.n, 1 - 0.5 / g.n)
    g["ratio"] = g.between / np.sqrt(p * (1 - p) / g.n)
    print("  between-seed std / injection-sampling std:")
    print(g.ratio.unstack("width").reindex(FORMATS).round(1).to_string())
    print(f"\n  median {g.ratio.median():.1f}x -- values well above 1 mean the scatter is "
          "model-to-model, which more injections cannot remove and more seeds can.")

    # ---- 4. the F18 control
    _section("TEST 4  F18 control: non-finite rate should be flat in width")
    m = df.groupby(["fmt", "width"]).nonfinite_rate.mean().unstack("width")
    for f in ("e4m3", "e5m2"):
        v = m.loc[f, pw.index]
        r = np.corrcoef(np.log(pw.values), v.values)[0, 1]
        print(f"  {f}: {v.min():.4f} .. {v.max():.4f} over a {pw.max() / pw.min():.0f}x "
              f"capacity range (max/min {v.max() / max(v.min(), 1e-9):.2f}x), "
              f"r vs log(params) {r:+.2f}")
    zero = (df[df.fmt == "e3m2"].nonfinite_rate == 0).all()
    print("  e3m2: " + ("0 at every width and seed, as its lack of special codes requires"
                        if zero else "NONZERO -- contradicts the format definition, investigate"))

    print(f"\nwrote {RESULTS / 'width_sweep_multiseed.csv'} and "
          f"{RESULTS / 'width_sweep_multiseed_slopes.csv'}")


if __name__ == "__main__":
    main()
