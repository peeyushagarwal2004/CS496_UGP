"""E10 -- does quantisation set vulnerability by compressing prediction margins?

E09/E09b ruled out everything about the fault itself: `e2m3` and `e3m2` deliver
the same distribution of absolute perturbation, hit weights of the same size,
and still differ ~6x in failure rate at matched perturbation, whether the
perturbation is measured against the layer or against the output neuron.

That leaves the *model*, not the fault. Quantising to a format produces a
different network, and two networks can have identical accuracy while sitting
at very different distances from their decision boundaries. Accuracy only asks
whether the top logit is correct; SDC asks whether a small push changes which
logit is on top. The quantity that governs the second is the margin

    m(x) = z_(1)(x) - z_(2)(x)

between the largest and second-largest logit. If a format compresses margins,
every perturbation becomes likelier to flip a prediction, with no change in
accuracy and no change in per-fault severity.

This measures the golden margin distribution of each MX-quantised model on the
same fixed evaluation subset the campaigns used, then checks it against the
element SDC rates already measured.

Run::

    PYTHONPATH=. python3 -m experiments.e10_margins --model vit_small
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained, setup_device

RESULTS = Path(__file__).resolve().parent.parent / "results"
FORMATS = ["e5m2", "e4m3", "e3m2", "e2m3", "e2m1"]


def margins(model, xs, device) -> np.ndarray:
    """Top-1 minus top-2 logit, scaled by the spread of that example's logits.

    The raw gap is not comparable across models whose logits have different
    overall scale, so each gap is divided by the standard deviation of the ten
    logits for that image. What matters is how far the decision is from a tie,
    relative to how spread out the scores are.
    """
    with torch.no_grad():
        z = model(xs.to(device)).float().cpu()
    top2 = z.topk(2, dim=1).values
    gap = (top2[:, 0] - top2[:, 1]).numpy()
    return gap / np.maximum(z.std(dim=1).numpy(), 1e-9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_small")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--formats", nargs="*", default=FORMATS)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    dev = setup_device(a.device)
    xs = campaign_subset(n_per_class=20, download=False).tensors[0]

    rows = []
    for seed in a.seeds:
        net, _ = load_trained(a.model, seed=seed)
        deployed = to_deploy(net).to(dev)
        for fmt in a.formats:
            mx = MXModel(deployed, MXConfig(fmt=fmt, block_size=a.block_size))
            mx.quantize_weights()
            m = margins(mx.module, xs, dev)
            mx.restore_weights()

            tag = a.model if seed == 0 else f"{a.model}_s{seed}"
            csv = RESULTS / f"e01_{tag}_{fmt}-K{a.block_size}-ocp-w-n3000.csv"
            sdc = np.nan
            if csv.exists():
                d = pd.read_csv(csv)
                e = d[d.site == "element"]
                sdc = float((e.sdc & ~e.nonfinite).mean())   # perturbation-driven only
            rows.append({"seed": seed, "fmt": fmt, "median_margin": float(np.median(m)),
                         "mean_margin": float(m.mean()),
                         "frac_below_0.5": float((m < 0.5).mean()),
                         "frac_below_1.0": float((m < 1.0).mean()),
                         "element_sdc_perturbation": sdc})

    df = pd.DataFrame(rows)
    g = df.groupby("fmt").mean(numeric_only=True).reindex(a.formats)
    out = RESULTS / f"e10_{a.model}_margins.csv"
    df.to_csv(out, index=False)

    print(f"model {a.model}, {len(a.seeds)} seeds, golden margins on the campaign subset")
    print("(SDC here counts perturbation-driven failures only, excluding NaN/Inf)\n")
    print(f"{'fmt':>6} {'median margin':>14} {'mean':>8} {'frac<0.5':>9} {'frac<1.0':>9} "
          f"{'element SDC':>12}")
    for fmt, r in g.iterrows():
        print(f"{fmt:>6} {r.median_margin:>14.3f} {r.mean_margin:>8.3f} "
              f"{r['frac_below_0.5']:>9.3f} {r['frac_below_1.0']:>9.3f} "
              f"{r.element_sdc_perturbation:>12.4f}")

    ok = g.dropna(subset=["element_sdc_perturbation"])
    if len(ok) >= 3:
        r_med = np.corrcoef(ok.median_margin, ok.element_sdc_perturbation)[0, 1]
        r_frac = np.corrcoef(ok["frac_below_1.0"], ok.element_sdc_perturbation)[0, 1]
        print(f"\n  correlation of element SDC with median margin : {r_med:+.3f}")
        print(f"  correlation of element SDC with frac(margin<1) : {r_frac:+.3f}")
        if {"e3m2", "e2m3"} <= set(ok.index):
            a3, a2 = ok.loc["e3m2"], ok.loc["e2m3"]
            print(f"\n  e3m2 median margin {a3.median_margin:.3f} vs e2m3 "
                  f"{a2.median_margin:.3f}  ({a3.median_margin / a2.median_margin:.2f}x wider)")
            print(f"  their SDC rates: {a3.element_sdc_perturbation:.4f} vs "
                  f"{a2.element_sdc_perturbation:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
