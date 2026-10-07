"""E24 -- do the headline results survive a harder task? CIFAR-100 as the ImageNet stand-in.

The plan's transformer study was DeiT on ImageNet, which no machine this project
can reach holds. CIFAR-100 is the nearest affordable substitute: ten times the
classes, a tenth of the images per class, and so networks that sit much closer to
their decision boundaries -- the property F48-F49 found to govern sensitivity. The
same three architectures are retrained on it with the CIFAR-10 recipes unchanged
(`scripts/c100_run.sh`), and every campaign scores 200 images, 2 per class, so per-inference
rates have the same resolution as on CIFAR-10.

This script sets each CIFAR-100 result beside its CIFAR-10 counterpart:

1. quantised accuracy per format (E00) -- are formats still matched in accuracy?
2. per-inference element and scale rates per format, per seed (E01) -- does the
   format ordering hold, and does the scale still dominate?
3. how much element damage is non-finite -- is the NaN pathway still the mechanism?
4. the NaN read guard (E23), and weights against activations (E04), in both
   NaN-coded formats, written to `results/e04_corrected_summary.csv`.

Run::

    PYTHONPATH=. python -m experiments.e24_cifar100
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMAGES = 200
PAIRS = [("vit_small", "vit_small_c100"), ("repvgg_a0", "repvgg_a0_c100"),
         ("resnet8_w16", "resnet8_c100")]
E01 = re.compile(r"^e01_(?P<model>[a-z0-9_]+?)(?:_s(?P<seed>\d+))?_"
                 r"(?P<fmt>e\dm\d)-K32-ocp-w-n3000\.csv$")


def campaigns() -> pd.DataFrame:
    """One row per (model, seed, format, site) from the uniform campaigns."""
    rows = []
    for path in RESULTS.glob("e01_*-K32-ocp-w-n3000.csv"):
        m = E01.match(path.name)
        if not m or m["model"] not in {x for pair in PAIRS for x in pair}:
            continue
        d = pd.read_csv(path)
        d["r"] = d.changed / IMAGES
        for site, g in d.groupby("site"):
            dmg = g.r.sum()
            rows.append({"model": m["model"], "seed": int(m["seed"] or 0),
                         "fmt": m["fmt"], "site": site, "n": len(g),
                         "per_inference": g.r.mean(),
                         "se": g.r.std(ddof=1) / np.sqrt(len(g)),
                         "nonfinite_share": (g.r[g.nonfinite].sum() / dmg) if dmg else 0.0})
    return pd.DataFrame(rows)


def accuracy() -> pd.DataFrame:
    rows = []
    for _, m in PAIRS:
        path = RESULTS / f"e00_{m}_quantized_accuracy.csv"
        if path.exists():
            d = pd.read_csv(path)
            d = d[(d.block_size == 32) & (d.scale_mode == "ocp")]
            rows += [{"model": m, "fmt": r.fmt, "accuracy": r.accuracy,
                      "drop_vs_fp32": r.drop_vs_fp32} for r in d.itertuples()]
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 200)
    acc = accuracy()
    if len(acc):
        print("=== quantised accuracy, K=32, OCP rule (CIFAR-100, seed 0) ===")
        print(acc.pivot(index="model", columns="fmt", values="accuracy").round(4).to_string())

    c = campaigns()
    if c.empty:
        raise SystemExit("no campaigns found")
    el = c[c.site == "element"]
    sc = c[c.site == "scale"]

    print("\n=== element per-inference rate, mean over seeds [seed range] ===")
    rows = []
    for c10, c100 in PAIRS:
        for model in (c10, c100):
            g = el[el.model == model]
            for fmt, h in g.groupby("fmt"):
                s = sc[(sc.model == model) & (sc.fmt == fmt)]
                rows.append({"pair": c10, "dataset": "CIFAR-100" if model == c100 else "CIFAR-10",
                             "fmt": fmt, "seeds": len(h),
                             "element": h.per_inference.mean(),
                             "el_min": h.per_inference.min(), "el_max": h.per_inference.max(),
                             "scale": s.per_inference.mean(),
                             "scale_over_element": s.per_inference.mean() / max(h.per_inference.mean(), 1e-9),
                             "nonfinite_share": h.nonfinite_share.mean()})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    t.to_csv(RESULTS / "e24_cifar100_vs_cifar10.csv", index=False)

    print("\n=== does e3m2 < e4m3 < e5m2 hold per seed (element, per inference)? ===")
    for model in [m for pair in PAIRS for m in pair]:
        g = el[el.model == model].pivot(index="seed", columns="fmt", values="per_inference")
        if not {"e3m2", "e4m3", "e5m2"} <= set(g.columns):
            continue
        ok = ((g.e3m2 < g.e4m3) & (g.e4m3 < g.e5m2))
        print(f"  {model:>16}: {int(ok.sum())}/{len(ok)} seeds")

    guard = RESULTS / "e23_nan_guard_summary.csv"
    if guard.exists():
        gd = pd.read_csv(guard)
        gd = gd[gd.run.str.contains("_c100")]
        if len(gd):
            print("\n=== NaN read guard on CIFAR-100 (E23; reduction is a 95% lower bound) ===")
            print(gd[["run", "special_fraction", "special_share_of_damage",
                      "element_rate_unguarded", "element_rate_guarded",
                      "reduction_at_least"]].to_string(index=False,
                                                       float_format=lambda v: f"{v:.4g}"))

    print(f"\nwrote {RESULTS / 'e24_cifar100_vs_cifar10.csv'}")
    weights_vs_activations()


E04 = re.compile(r"^e04_(?P<model>[a-z0-9_]+)_activations_(?P<fmt>e\dm\d)-K32-ocp-wa\.csv$")


def weights_vs_activations(boot: int = 2000) -> None:
    """Per-inference damage of a weight fault against an activation fault (F69).

    Each E04 campaign is set against the seed-0 E01 weight campaign of the same
    model and format. The activation interval is a bootstrap over images, since a
    few near-tie images carry much of the activation damage.
    """
    rows = []
    for path in sorted(RESULTS.glob("e04_*_activations_*-K32-ocp-wa.csv")):
        m = E04.match(path.name)
        wpath = RESULTS / f"e01_{m['model']}_{m['fmt']}-K32-ocp-w-n3000.csv" if m else None
        if not m or not wpath.exists():
            continue
        a, w = pd.read_csv(path), pd.read_csv(wpath)
        w["r"] = w.changed / IMAGES
        rng = np.random.default_rng(0)
        for site in ("element", "scale"):
            g, ws = a[a.site == site], w[w.site == site]
            per = g.groupby("image").corrupted.agg(["sum", "size"]).to_numpy()
            idx = rng.integers(0, len(per), (boot, len(per)))
            bs = per[idx, 0].sum(1) / per[idx, 1].sum(1)
            act = g.corrupted.mean()
            rows.append({"model": m["model"], "fmt": m["fmt"], "site": site, "n_act": len(g),
                         "act": act, "act_lo": np.percentile(bs, 2.5),
                         "act_hi": np.percentile(bs, 97.5), "act_nonfinite": g.nonfinite.mean(),
                         "weight": ws.r.mean(),
                         "weight_over_act": ws.r.mean() / act if act > 0 else np.inf})
    t = pd.DataFrame(rows)
    out = RESULTS / "e04_corrected_summary.csv"
    t.to_csv(out, index=False)
    print("\n=== weights vs activations, per inference (E04 vs E01, seed 0) ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
