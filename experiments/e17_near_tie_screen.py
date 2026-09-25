"""E17 -- if near-ties are the mechanism, a forward pass should predict vulnerability.

E16 decomposed the sensitivity $s=|\\partial m/\\partial w|\\,\\mathrm{rms}/m$ that
predicts vulnerability and found the gradient factor format-invariant to within
$1.08\\times$ while the margin of the hardest evaluation image varies by $33\\times$.
If that is the mechanism and not a coincidence of five formats, then a statistic
of the margin alone -- no gradients, no injections, one forward pass per image --
should predict the measured failure rate of an independently trained network.

That is a real prediction, and this tests it out of sample: the margins are
measured on 2000 images, the failure rates come from campaigns already run on
different images, and the test spans three seeds, so the fifteen points are
fifteen different networks rather than five formats of one.

It is also the practically useful form of the result. Sensitivity needs a
backward pass per image and the fault space scored; near-tie mass needs a forward
pass, which any deployment already does.

Run::

    PYTHONPATH=. python3 -m experiments.e17_near_tie_screen --device cuda
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
FORMATS = ["e5m2", "e3m2", "e4m3", "e2m3", "e2m1"]


def margins(mx: MXModel, xs: torch.Tensor, device: str, batch: int) -> np.ndarray:
    """Top-1 minus top-2 logit for every image, forward passes only."""
    out = []
    with torch.no_grad():
        for i in range(0, xs.shape[0], batch):
            z = mx.module(xs[i : i + batch].to(device))
            t2 = torch.topk(z, 2, dim=1).values
            out.append((t2[:, 0] - t2[:, 1]).cpu().numpy())
    return np.concatenate(out)


def near_tie_stats(m: np.ndarray) -> dict:
    """Several summaries of how much near-tie mass the network carries."""
    m = np.maximum(m, 1e-6)
    return {
        "margin_min": float(m.min()),
        "margin_p01": float(np.quantile(m, 0.001)),
        "margin_p1": float(np.quantile(m, 0.01)),
        "margin_median": float(np.median(m)),
        "frac_below_0.05": float((m < 0.05).mean()),
        "frac_below_0.20": float((m < 0.20).mean()),
        # the factor that actually enters s: 1/m, averaged over the images
        "mean_inv_margin": float((1.0 / m).mean()),
        "inv_min_margin": float(1.0 / m.min()),
    }


def measured_sdc(model: str, seed: int, fmt: str, block: int) -> dict:
    """Perturbation-driven element SDC from the campaign already run for this cell."""
    stem = model if seed == 0 else f"{model}_s{seed}"
    path = RESULTS / f"e01_{stem}_{fmt}-K{block}-ocp-w-n3000.csv"
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    el = d[d.site == "element"]
    pert = el[~el.to_nonfinite.astype(bool)] if "to_nonfinite" in el else el
    return {"sdc_element": float(el.sdc.mean()),
            "sdc_perturbation": float(pert.sdc.mean()),
            "n_element": int(len(el)),
            "nonfinite_rate": float(el.to_nonfinite.astype(bool).mean())
            if "to_nonfinite" in el else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_small")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--images", type=int, default=200, help="per class, so x10")
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    dev = setup_device(a.device)
    xs = campaign_subset(n_per_class=a.images, download=False).tensors[0]
    print(f"{a.model}: margins over {xs.shape[0]} images, "
          f"seeds {a.seeds}, {len(FORMATS)} formats")

    rows = []
    for seed in a.seeds:
        for fmt in FORMATS:
            try:
                net, _ = load_trained(a.model, seed=seed)
            except FileNotFoundError:
                print(f"  seed {seed}: no checkpoint, skipped")
                break
            mx = MXModel(to_deploy(net).to(dev),
                         MXConfig(fmt=fmt, block_size=a.block_size)).quantize_weights()
            st = near_tie_stats(margins(mx, xs, dev, a.batch))
            mx.restore_weights()
            rows.append({"seed": seed, "fmt": fmt, **st,
                         **measured_sdc(a.model, seed, fmt, a.block_size)})

    df = pd.DataFrame(rows)
    # the image count is the whole point of this experiment, so it names the file
    out = RESULTS / f"e17_{a.model}_near_tie_screen_n{xs.shape[0]}.csv"
    df.to_csv(out, index=False)

    have = df.dropna(subset=["sdc_perturbation"]) if "sdc_perturbation" in df else df
    print(f"\n{'seed':>5} {'fmt':>6} {'min margin':>11} {'0.1st pct':>10} "
          f"{'frac < 0.05':>12} {'mean 1/m':>9} {'element SDC':>12} {'pert. SDC':>10}")
    for _, r in df.iterrows():
        e = r.get("sdc_element", float("nan"))
        s = r.get("sdc_perturbation", float("nan"))
        print(f"{int(r.seed):>5} {r.fmt:>6} {r.margin_min:>11.4f} {r.margin_p01:>10.4f} "
              f"{r['frac_below_0.05']:>12.4f} {r.mean_inv_margin:>9.2f} "
              f"{e:>12.4f} {s:>10.4f}")

    if len(have) < 3:
        print("\nno campaign results to compare against; predictors written to the csv")
        return

    from scipy import stats as st
    print(f"\nagainst the independently measured failure rate, over "
          f"{len(have)} networks ({have.seed.nunique()} seeds x {have.fmt.nunique()} formats)")
    print(f"\n{'predictor':>22} {'pearson':>9} {'spearman':>9}")
    for col in ("inv_min_margin", "mean_inv_margin", "frac_below_0.05",
                "frac_below_0.20", "margin_p01", "margin_median"):
        y = have.sdc_perturbation
        print(f"{col:>22} {st.pearsonr(have[col], y)[0]:>+9.3f} "
              f"{st.spearmanr(have[col], y)[0]:>+9.3f}")

    # within a seed the formats vary; within a format the seeds vary. Both matter,
    # because a predictor that only separates formats is not predicting a network.
    print("\n  rank correlation within each seed, formats only:")
    for seed, g in have.groupby("seed"):
        if len(g) > 2:
            print(f"    seed {int(seed)}: "
                  f"{st.spearmanr(g.mean_inv_margin, g.sdc_perturbation)[0]:+.3f} "
                  f"(n={len(g)})")
    print("\n  rank correlation within each format, seeds only:")
    for fmt, g in have.groupby("fmt"):
        if len(g) > 2:
            print(f"    {fmt}: "
                  f"{st.spearmanr(g.mean_inv_margin, g.sdc_perturbation)[0]:+.3f} "
                  f"(n={len(g)})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
