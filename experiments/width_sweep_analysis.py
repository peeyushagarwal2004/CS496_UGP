"""Width-sweep analysis: does model capacity drive the F18 mechanism?

F18 claims that as capacity grows, the dominant failure mode shifts from
numeric perturbation to NaN/Inf propagation -- so vulnerability should fall
with width for formats that *cannot* produce a non-finite value, and fall much
less for formats that can.

The evidence for it so far compares two models differing in depth, width,
topology, training budget and accuracy all at once.  This sweep varies only
ResNet8's base width, at a fixed 30-epoch budget, across a 35x parameter
range.  The three formats form a gradient in special-code availability:

    e5m2  Inf + NaN (8 codes)
    e4m3  NaN only  (2 codes)
    e3m2  none

so F18 predicts a *dose-response*: e5m2 should degrade most with capacity,
e4m3 less, e3m2 not at all.  A difference between two formats is suggestive;
an ordering across three that matches special-code count is much harder to
explain any other way.

Run::

    .venv/Scripts/python.exe -m experiments.width_sweep_analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.models import build_model, count_parameters
from mxfi.stats import binomial_rate

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIDTHS = [8, 16, 24, 32, 48]
FORMATS = ["e5m2", "e4m3", "e3m2"]
SPECIAL_CODES = {"e5m2": 8, "e4m3": 2, "e3m2": 0}


def collect() -> pd.DataFrame:
    rows = []
    for w in WIDTHS:
        model = f"resnet8_w{w}"
        try:
            params = count_parameters(build_model(model))
        except KeyError:
            continue
        for fmt in FORMATS:
            p = RESULTS / f"e01_{model}_{fmt}-K32-ocp-w-n3000.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            e = d[d.site == "element"]
            s = d[d.site == "scale"]
            if not len(e):
                continue
            est = binomial_rate(e.sdc.to_numpy())
            sdc_rows = e[e.sdc]
            rows.append({
                "width": w, "params": params, "fmt": fmt,
                "special_codes": SPECIAL_CODES[fmt],
                "element_sdc": est.rate, "hw": est.half_width,
                "nonfinite_rate": float(e.nonfinite.mean()),
                "nonfinite_share_of_sdc": (float(sdc_rows.nonfinite.mean())
                                           if len(sdc_rows) else np.nan),
                "perturbation_sdc": float((e.sdc & ~e.nonfinite).mean()),
                "scale_sdc": float(s.sdc.mean()) if len(s) else np.nan,
                "n_element": len(e), "n_scale": len(s),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = collect()
    if df.empty:
        print("no width-sweep results yet")
        return
    df.to_csv(RESULTS / "width_sweep.csv", index=False)

    print("=" * 78)
    print("element SDC rate vs width")
    print("=" * 78)
    print(df.pivot(index="fmt", columns="width", values="element_sdc")
            .reindex(FORMATS).round(4).to_string())

    print("\n" + "=" * 78)
    print("PERTURBATION-driven SDC (excludes non-finite) -- F18 says this")
    print("should fall with capacity for EVERY format")
    print("=" * 78)
    print(df.pivot(index="fmt", columns="width", values="perturbation_sdc")
            .reindex(FORMATS).round(4).to_string())

    print("\n" + "=" * 78)
    print("NON-FINITE rate -- only formats with special codes can be non-zero")
    print("=" * 78)
    print(df.pivot(index="fmt", columns="width", values="nonfinite_rate")
            .reindex(FORMATS).round(4).to_string())

    print("\n" + "=" * 78)
    print("share of failures that were non-finite -- F18's core claim:")
    print("this should RISE with capacity wherever it can be non-zero")
    print("=" * 78)
    print(df.pivot(index="fmt", columns="width", values="nonfinite_share_of_sdc")
            .reindex(FORMATS).round(4).to_string())

    # ---- the dose-response test
    print("\n" + "=" * 78)
    print("F18 VERDICT")
    print("=" * 78)
    for fmt in FORMATS:
        g = df[df.fmt == fmt].sort_values("params")
        if len(g) < 3:
            print(f"  {fmt}: too few widths to judge")
            continue
        lp = np.log(g.params.to_numpy())
        r_pert = np.corrcoef(lp, g.perturbation_sdc.to_numpy())[0, 1]
        share = g.nonfinite_share_of_sdc.to_numpy()
        r_share = (np.corrcoef(lp, share)[0, 1]
                   if np.isfinite(share).all() and share.std() > 0 else np.nan)
        print(f"  {fmt} ({SPECIAL_CODES[fmt]} special codes): "
              f"perturbation-SDC vs log(params) r={r_pert:+.3f} | "
              f"non-finite share vs log(params) r={r_share:+.3f}"
              if np.isfinite(r_share) else
              f"  {fmt} ({SPECIAL_CODES[fmt]} special codes): "
              f"perturbation-SDC vs log(params) r={r_pert:+.3f} | "
              f"non-finite share n/a (format cannot produce one)")

    first, last = df.width.min(), df.width.max()
    piv = df.pivot(index="fmt", columns="width", values="element_sdc").reindex(FORMATS)
    if {first, last} <= set(piv.columns):
        print(f"\n  element SDC change from width {first} to {last}:")
        for fmt in FORMATS:
            if fmt in piv.index and np.isfinite(piv.loc[fmt, [first, last]]).all():
                a, b = piv.loc[fmt, first], piv.loc[fmt, last]
                print(f"    {fmt} ({SPECIAL_CODES[fmt]} special codes): "
                      f"{a:.4f} -> {b:.4f}  ({b/max(a,1e-9):.2f}x)")
        print("\n  F18 predicts the ratio should be ordered "
              "e3m2 < e4m3 < e5m2 (fewer special codes -> more benefit "
              "from capacity).")

    print(f"\nwrote {RESULTS / 'width_sweep.csv'}")


if __name__ == "__main__":
    main()
