"""E11 -- the mechanism behind F36/F37: how hard does a weight push the decision?

F36 found that quantised networks of equal accuracy and equal margins differ by
more than an order of magnitude in how often a fixed-size weight perturbation
flips a prediction, and F37 tied that to mantissa width rather than exponent
width. Margins were ruled out, so what remains is the other half of the
question: not how far the decision sits from a tie, but how fast it moves when
a weight moves.

For image $x$ with margin $m = z_{(1)} - z_{(2)}$, perturbing weight $w_i$ by
$\\delta$ changes the margin by about $\\delta \\, \\partial m / \\partial w_i$ to first
order. The prediction flips when that exceeds $m$, so the natural dimensionless
sensitivity of weight $i$ on image $x$ is

    s_i(x) = |dm/dw_i| * rms(layer) / m(x)

and a perturbation of `delta * rms` flips image $x$ when `s_i(x) * delta > 1`.
A weight fault is exposed to every inference at once, so the quantity that
predicts the measured probe is the maximum over images.

This computes s exactly by backpropagation, turns it into a predicted flip
rate, and checks it against the rates E10's format-agnostic probe measured.

Run::

    PYTHONPATH=. python3 -m experiments.e11_sensitivity --device cuda
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
MANTISSA = {"e5m2": 2, "e3m2": 2, "e4m3": 3, "e2m3": 3, "e2m1": 1}
# measured by the format-agnostic probe (one weight moved by delta * layer RMS)
MEASURED = {"e5m2": (0.0000, 0.0000, 0.0013), "e3m2": (0.0000, 0.0000, 0.0013),
            "e4m3": (0.0080, 0.0493, 0.1380), "e2m3": (0.0013, 0.0187, 0.0513),
            "e2m1": (0.0160, 0.0740, 0.1407)}
DELTAS = (0.5, 1.5, 3.0)


def sensitivity(mx: MXModel, xs: torch.Tensor, device: str) -> np.ndarray:
    """max over images of |dm/dw| * rms(layer) / m, for every quantised weight."""
    layers = {n: l.module for n, l in mx.layers.items()}
    for p in mx.module.parameters():
        p.requires_grad_(False)
    for m in layers.values():
        m.weight.requires_grad_(True)

    rms = {n: float(torch.sqrt((m.weight.detach() ** 2).mean()))
           for n, m in layers.items()}
    best = {n: torch.zeros_like(m.weight) for n, m in layers.items()}

    for i in range(xs.shape[0]):
        mx.module.zero_grad(set_to_none=True)
        z = mx.module(xs[i : i + 1].to(device))[0]
        top2 = torch.topk(z, 2)
        margin = top2.values[0] - top2.values[1]
        margin.backward()
        with torch.no_grad():
            for n, m in layers.items():
                if m.weight.grad is None:
                    continue
                s = m.weight.grad.abs() * (rms[n] / float(margin.clamp(min=1e-6)))
                torch.maximum(best[n], s, out=best[n])

    for m in layers.values():
        m.weight.requires_grad_(False)
    return torch.cat([b.detach().flatten().cpu() for b in best.values()]).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    dev = setup_device(a.device)
    xs = campaign_subset(n_per_class=a.images, download=False).tensors[0]

    rows = []
    for fmt in FORMATS:
        net, _ = load_trained(a.model, seed=a.seed)
        mx = MXModel(to_deploy(net).to(dev),
                     MXConfig(fmt=fmt, block_size=a.block_size)).quantize_weights()
        s = sensitivity(mx, xs, dev)
        # a random weight, random sign: flips when s * delta > 1, half the time
        pred = {d: 0.5 * float((s * d > 1.0).mean()) for d in DELTAS}
        rows.append({"fmt": fmt, "mantissa": MANTISSA[fmt],
                     "median_s": float(np.median(s)), "p99_s": float(np.quantile(s, 0.99)),
                     "max_s": float(s.max()),
                     **{f"pred_{d}": pred[d] for d in DELTAS},
                     **{f"meas_{d}": MEASURED[fmt][i] for i, d in enumerate(DELTAS)}})
        mx.restore_weights()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"e11_{a.model}_sensitivity.csv", index=False)

    print(f"model {a.model} seed {a.seed}: sensitivity s = |dm/dw| * rms / margin,")
    print("maximised over the 200 evaluation images, computed for every weight\n")
    print(f"{'fmt':>6} {'man':>4} {'median s':>10} {'99th pct':>10} {'max s':>9}")
    for _, r in df.iterrows():
        print(f"{r.fmt:>6} {int(r.mantissa):>4} {r.median_s:>10.4f} "
              f"{r.p99_s:>10.3f} {r.max_s:>9.2f}")

    print(f"\npredicted vs measured flip rate for one weight moved by delta * rms")
    print(f"{'fmt':>6} " + "  ".join(f"{'d=' + str(d):>19}" for d in DELTAS))
    for _, r in df.iterrows():
        cells = [f"{r['pred_' + str(d)]:.4f} vs {r['meas_' + str(d)]:.4f}" for d in DELTAS]
        print(f"{r.fmt:>6} " + "  ".join(f"{c:>19}" for c in cells))

    pred = np.concatenate([df[f"pred_{d}"].to_numpy() for d in DELTAS])
    meas = np.concatenate([df[f"meas_{d}"].to_numpy() for d in DELTAS])
    if pred.std() > 0:
        print(f"\n  correlation between predicted and measured: "
              f"{np.corrcoef(pred, meas)[0, 1]:+.3f} over {len(pred)} cells")
    print(f"\n  2-mantissa networks, median s: "
          f"{df[df.mantissa == 2].median_s.mean():.4f}")
    print(f"  3-mantissa networks, median s: "
          f"{df[df.mantissa == 3].median_s.mean():.4f}")


if __name__ == "__main__":
    main()
