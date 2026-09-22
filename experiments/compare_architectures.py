"""Cross-architecture replication check: ResNet8 vs RepVGG-A0.

A finding measured on one model is an observation about that model.  The same
finding on a residual net *and* a fused plain net is a claim about
microscaling.  This script re-derives the three findings that need a second
architecture and reports, for each, whether it replicated.

    F1   matched-accuracy formats differ in vulnerability (e5m2 vs e3m2)
    F4   shared-scale faults are far worse than element faults
    F15  the OCP scale rule fabricates a block-size trend

Run::

    .venv/Scripts/python.exe -m experiments.compare_architectures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.stats import binomial_rate

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = ["resnet8", "repvgg_a0"]


def _load(pattern: str) -> pd.DataFrame | None:
    p = RESULTS / pattern
    return pd.read_csv(p) if p.exists() else None


def _fmt_est(rate: float, hw: float) -> str:
    return f"{rate:.4f} +-{hw:.4f}"


def f1_matched_accuracy() -> pd.DataFrame:
    """e5m2 (8b) vs e3m2 (6b): near-identical accuracy, different SDC?"""
    rows = []
    for m in MODELS:
        acc = _load(f"e00_{m}_quantized_accuracy.csv")
        for fmt in ("e5m2", "e3m2", "e4m3"):
            fi = _load(f"e01_{m}_{fmt}-K32-ocp-w-n3000.csv")
            if fi is None:
                continue
            est = binomial_rate(fi["sdc"].to_numpy())
            a = np.nan
            if acc is not None:
                sel = acc[(acc.fmt == fmt) & (acc.block_size == 32)
                          & (acc.scale_mode == "ocp")]
                if len(sel):
                    a = float(sel.accuracy.iloc[0])
            rows.append({"model": m, "fmt": fmt, "accuracy": a,
                         "sdc": est.rate, "hw": est.half_width, "n": est.n})
    return pd.DataFrame(rows)


def f4_site_asymmetry() -> pd.DataFrame:
    """Scale vs element SDC, per model and format."""
    rows = []
    for m in MODELS:
        for fmt in ("e4m3", "e5m2", "e3m2"):
            fi = _load(f"e01_{m}_{fmt}-K32-ocp-w-n3000.csv")
            if fi is None:
                continue
            e = fi[fi.site == "element"]
            s = fi[fi.site == "scale"]
            if not len(e) or not len(s):
                continue
            rows.append({
                "model": m, "fmt": fmt,
                "element_sdc": e.sdc.mean(), "scale_sdc": s.sdc.mean(),
                "ratio": s.sdc.mean() / max(e.sdc.mean(), 1e-9),
                "scale_imgs": s.changed.mean(),
                "element_imgs": e.changed.mean(),
                "n_scale": len(s),
            })
    return pd.DataFrame(rows)


def f15_clipping_artifact() -> pd.DataFrame:
    """Element SDC across K under ocp vs fit -- flat under fit?"""
    rows = []
    for m in MODELS:
        for mode in ("ocp", "fit"):
            d = _load(f"e03b_{m}_block_size_matched_{mode}.csv")
            if d is None:
                continue
            lo, hi = d.element_sdc.min(), d.element_sdc.max()
            rows.append({
                "model": m, "scale_mode": mode,
                **{f"K{int(k)}": v for k, v in zip(d.K, d.element_sdc)},
                "spread": hi / max(lo, 1e-9),
                "max_hw": d.hw.max(),
                "flat": bool((hi - lo) < 2 * d.hw.max()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 74)
    print("F1  matched-accuracy formats -> different vulnerability?")
    print("=" * 74)
    f1 = f1_matched_accuracy()
    if f1.empty:
        print("  (no data yet)")
    else:
        print(f1.round(4).to_string(index=False))
        for m in MODELS:
            g = f1[f1.model == m].set_index("fmt")
            if {"e5m2", "e3m2"} <= set(g.index):
                da = abs(g.loc["e5m2", "accuracy"] - g.loc["e3m2", "accuracy"])
                r = g.loc["e3m2", "sdc"] / max(g.loc["e5m2", "sdc"], 1e-9)
                sep = (abs(g.loc["e3m2", "sdc"] - g.loc["e5m2", "sdc"])
                       > g.loc["e3m2", "hw"] + g.loc["e5m2", "hw"])
                print(f"\n  {m}: accuracy gap {da*100:.2f} pp, SDC ratio {r:.2f}x, "
                      f"CIs {'disjoint' if sep else 'OVERLAP'}")

    print("\n" + "=" * 74)
    print("F4  shared-scale vs element asymmetry")
    print("=" * 74)
    f4 = f4_site_asymmetry()
    print(f4.round(4).to_string(index=False) if not f4.empty else "  (no data yet)")

    print("\n" + "=" * 74)
    print("F15 does OCP clipping fabricate a block-size trend?")
    print("=" * 74)
    f15 = f15_clipping_artifact()
    if f15.empty:
        print("  (no data yet)")
    else:
        print(f15.round(4).to_string(index=False))
        for m in MODELS:
            g = f15[f15.model == m].set_index("scale_mode")
            if {"ocp", "fit"} <= set(g.index):
                print(f"\n  {m}: ocp spread {g.loc['ocp','spread']:.2f}x "
                      f"({'flat' if g.loc['ocp','flat'] else 'NOT flat'}), "
                      f"fit spread {g.loc['fit','spread']:.2f}x "
                      f"({'flat' if g.loc['fit','flat'] else 'NOT flat'})")
                if not g.loc["ocp", "flat"] and g.loc["fit", "flat"]:
                    print("        -> REPLICATES: clipping fabricates the trend")
                elif g.loc["ocp", "flat"] and g.loc["fit", "flat"]:
                    print("        -> no trend under either rule")
                else:
                    print("        -> DOES NOT replicate; inspect before reporting")

    out = RESULTS / "cross_architecture_summary.csv"
    pd.concat([f1.assign(finding="F1"), f4.assign(finding="F4"),
               f15.assign(finding="F15")], ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
