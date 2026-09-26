"""E19 -- every campaign rescored on the per-inference rate.

F52 showed that the SDC definition used throughout this study -- a fault counts as
failing if *any* of the $n$ evaluated inferences changes its prediction -- saturates
in $n$ and mostly reports whether the evaluation set happens to contain an input
near a decision boundary. The per-inference rate, the fraction of the $n$
inferences a fault actually corrupts, does not have that defect: on the fold study
it moved $1.15$--$1.27\\times$ where the any-image rate moved sixfold, and it
recovered the format effect the any-image rate had erased.

No new injections are needed to apply that lesson retrospectively. ``run_campaign``
has always recorded ``changed`` and ``change_rate`` per fault, so every campaign
already carries its own per-inference rate; only the summaries drew on the
any-image column. This rescores all of them and re-tests the headline claims under
both metrics, so each one can be labelled as surviving, strengthened, or lost.

Confidence intervals differ between the two metrics by necessity. The any-image
rate is a proportion, so it takes a Wilson interval; the per-inference rate is a
mean of per-fault proportions, so it takes a bootstrap over faults.

Run::

    PYTHONPATH=. python -m experiments.e19_per_inference
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.stats import wilson_interval

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IMAGES = 200          # every campaign in this study scored 200 images

# e01_resnet8_w16_s3_e4m3-K32-ocp-w-n3000.csv -> model, seed, fmt, K, mode
PATTERN = re.compile(
    r"^(?P<exp>e0[136]b?)_(?P<model>[a-z0-9_]+?)(?:_s(?P<seed>\d+))?_"
    r"(?P<fmt>e\dm\d)-K(?P<K>\d+)-(?P<mode>ocp|fit)-w"
    r"(?:-n\d+)?(?:_exhaustive)?\.csv(?:\.gz)?$")


def mean_ci(x: np.ndarray) -> tuple[float, float]:
    """95% interval for the mean per-inference rate, from its standard error.

    The per-inference rate of one fault is a proportion over the images, so a
    campaign's figure is a mean of bounded per-fault values and the central limit
    theorem applies at these sample sizes (3000 faults, or 600k exhaustively). A
    bootstrap would answer the same question at a cost that does not pay: for the
    exhaustive campaigns the resample matrix alone would need 10 GB.
    """
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    se = float(x.std(ddof=1)) / np.sqrt(n)
    m = float(x.mean())
    return (max(m - 1.96 * se, 0.0), min(m + 1.96 * se, 1.0))


def load(path: Path) -> pd.DataFrame | None:
    """Return a frame with site, bit, tensor, sdc and the per-inference rate."""
    d = pd.read_csv(path)
    if "sdc" not in d.columns:
        return None
    if "change_rate" in d.columns:
        d["pir"] = d.change_rate.astype(float)
    elif "changed" in d.columns:
        d["pir"] = d.changed.astype(float) / IMAGES
    else:
        return None
    return d


def describe(path: Path) -> dict:
    """Identify a campaign from its filename; e06.csv.gz predates the convention."""
    if path.name == "e06.csv.gz":
        return {"exp": "e06", "model": "resnet8_w16", "seed": 0, "fmt": "e4m3",
                "K": 32, "scale_mode": "ocp"}
    m = PATTERN.match(path.name)
    if not m:
        return {}
    g = m.groupdict()
    return {"exp": g["exp"], "model": g["model"], "seed": int(g["seed"] or 0),
            "fmt": g["fmt"], "K": int(g["K"]), "scale_mode": g["mode"]}


def rescore() -> pd.DataFrame:
    rows = []
    files = sorted(list(RESULTS.glob("e0[136]*.csv")) + list(RESULTS.glob("e06.csv.gz")))
    for f in files:
        meta = describe(f)
        if not meta:
            continue
        d = load(f)
        if d is None:
            continue
        for site in ("element", "element_pert", "scale", "all"):
            if site == "all":
                s = d
            elif site == "element_pert":
                # F45's headline quantity: element faults that perturb a value rather
                # than turning it into NaN/Inf, so the two mechanisms stay separate
                if "to_nonfinite" not in d.columns:
                    continue
                s = d[(d.site == "element") & (~d.to_nonfinite.astype(bool))]
            else:
                s = d[d.site == site]
            if not len(s):
                continue
            any_img = float(s.sdc.mean())
            lo, hi = wilson_interval(int(s.sdc.sum()), len(s))
            pir = s.pir.to_numpy()
            blo, bhi = mean_ci(pir)
            rows.append({**meta, "site": site, "n": len(s),
                         "any_image": any_img, "any_lo": lo, "any_hi": hi,
                         "per_inference": float(pir.mean()),
                         "pi_lo": blo, "pi_hi": bhi,
                         "nonfinite": float(s.nonfinite.mean())
                         if "nonfinite" in s else np.nan})
    return pd.DataFrame(rows)


def _fmt_pair(r) -> str:
    return (f"{r.any_image:.4f} [{r.any_lo:.4f},{r.any_hi:.4f}]   "
            f"{r.per_inference:.5f} [{r.pi_lo:.5f},{r.pi_hi:.5f}]")


def claim_formats(df: pd.DataFrame) -> None:
    """F1/F17/F45: does the format ordering hold under the per-inference rate?"""
    print("\n" + "=" * 78)
    print("CLAIM 1 -- the format ordering (F1, F17, F45)")
    print("=" * 78)
    d = df[(df.exp == "e01") & (df.site == "element") & (df.K == 32) & (df.scale_mode == "ocp")]
    for model, g in d.groupby("model"):
        seeds = sorted(g.seed.unique())
        print(f"\n{model} ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}), "
              f"element faults")
        print(f"  {'fmt':>6} {'any-image':>28} {'per-inference':>26}")
        for fmt, gg in g.groupby("fmt"):
            a, p = gg.any_image.mean(), gg.per_inference.mean()
            print(f"  {fmt:>6} {a:>12.4f} {'':>15} {p:>13.5f}")
        # the pair the report leans on
        for x, y in (("e3m2", "e5m2"), ("e3m2", "e4m3")):
            gx, gy = g[g.fmt == x], g[g.fmt == y]
            if len(gx) and len(gy):
                ra = gy.any_image.mean() / max(gx.any_image.mean(), 1e-12)
                rp = gy.per_inference.mean() / max(gx.per_inference.mean(), 1e-12)
                print(f"    {y} / {x}: any-image {ra:>6.2f}x   "
                      f"per-inference {rp:>7.2f}x")


def claim_scale(df: pd.DataFrame) -> None:
    """F4: shared scale against element."""
    print("\n" + "=" * 78)
    print("CLAIM 2 -- shared scale dominates element (F4)")
    print("=" * 78)
    d = df[(df.exp.isin(("e01", "e06"))) & (df.K == 32) & (df.scale_mode == "ocp")]
    print(f"\n{'model':>12} {'fmt':>6} {'exp':>5} {'any-image':>20} {'per-inference':>22}")
    for (model, fmt, exp), g in d.groupby(["model", "fmt", "exp"]):
        e = g[g.site == "element"]
        s = g[g.site == "scale"]
        if not (len(e) and len(s)):
            continue
        ra = s.any_image.mean() / max(e.any_image.mean(), 1e-12)
        rp = s.per_inference.mean() / max(e.per_inference.mean(), 1e-12)
        print(f"{model:>12} {fmt:>6} {exp:>5} {ra:>18.1f}x {rp:>20.1f}x")


def claim_block(df: pd.DataFrame) -> None:
    """F15: OCP clipping fabricates a block-size trend; fit is flat."""
    print("\n" + "=" * 78)
    print("CLAIM 3 -- the block-size trend is an artefact of the OCP scale rule (F15)")
    print("=" * 78)
    d = df[(df.exp.isin(("e03", "e03b"))) & (df.site == "element") & (df.fmt == "e4m3")]
    for (model, mode, exp), g in d.groupby(["model", "scale_mode", "exp"]):
        g = g.sort_values("K")
        if len(g) < 3:
            continue
        sa = g.any_image.max() / max(g.any_image.min(), 1e-12)
        sp = g.per_inference.max() / max(g.per_inference.min(), 1e-12)
        ks = "  ".join(f"K{int(k)}" for k in g.K)
        print(f"\n{model}, scale rule {mode}   ({ks})")
        print("  any-image     " + "  ".join(f"{v:.4f}" for v in g.any_image)
              + f"   spread {sa:.2f}x")
        print("  per-inference " + "  ".join(f"{v:.5f}" for v in g.per_inference)
              + f"  spread {sp:.2f}x")


def claim_capacity(df: pd.DataFrame) -> None:
    """The width sweep: does capacity still help, per inference?"""
    print("\n" + "=" * 78)
    print("CLAIM 4 -- capacity helps, in inverse proportion to special codes (F29)")
    print("=" * 78)
    d = df[(df.exp == "e01") & (df.site == "element") & (df.K == 32)
           & (df.scale_mode == "ocp") & df.model.str.startswith("resnet8_w")]
    order = {f"resnet8_w{w}": w for w in (8, 16, 24, 32, 48, 64)}
    d = d[d.model.isin(order)].copy()
    d["width"] = d.model.map(order)
    for fmt, g in d.groupby("fmt"):
        m = g.groupby("width").agg(a=("any_image", "mean"), p=("per_inference", "mean"),
                                   s=("seed", "nunique")).sort_index()
        if len(m) < 3:
            continue
        ra = m.a.iloc[-1] / max(m.a.iloc[0], 1e-12)
        rp = m.p.iloc[-1] / max(m.p.iloc[0], 1e-12)
        widths = "  ".join(f"w{w}" for w in m.index)
        print(f"\n{fmt}  ({widths};  {int(m.s.max())} seeds max)")
        print("  any-image     " + "  ".join(f"{v:.4f}" for v in m.a)
              + f"   widest/narrowest {ra:.2f}x")
        print("  per-inference " + "  ".join(f"{v:.5f}" for v in m.p)
              + f"  widest/narrowest {rp:.2f}x")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()

    df = rescore()
    out = RESULTS / "e19_per_inference_rates.csv"
    df.to_csv(out, index=False)
    print(f"rescored {df.model.nunique()} models across "
          f"{len(df[df.site == 'all']):,} campaigns, {int(df[df.site == 'all'].n.sum()):,} "
          f"recorded faults, without a single new injection")

    claim_formats(df)
    claim_scale(df)
    claim_block(df)
    claim_capacity(df)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
