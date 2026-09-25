"""E14 -- where the sensitivity tail comes from.

F38 established that vulnerability tracks the *tail* of
$s_i=|\\partial m/\\partial w_i|\\,\\mathrm{rms}/m$ and that coarse mantissas suppress
that tail, but not why. For a linear layer the gradient factorises,

    dm/dW[o,i] = sum_p a_i^(p) * g_o^(p),

the input activation times the gradient of the margin with respect to that
neuron's output, summed over positions. A heavy tail in $s$ must therefore come
from a heavy tail in the activations, in the output gradients, or in their
alignment. Each is measurable separately, so the question is decidable rather
than a matter of opinion.

The working hypothesis is the activation side: three mantissa bits preserve
outlier weights that two bits round away, and outlier weights produce outlier
activations, which transformers are known to carry. If instead the gradients
carry the tail, the cause is downstream of the representation and the story
has to change.

Run::

    PYTHONPATH=. python3 -m experiments.e14_sensitivity_origin --device cuda
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


class Tap:
    """Accumulates tail statistics of a layer's inputs and output gradients."""

    def __init__(self):
        self.a_sq = self.a_n = 0.0
        self.g_sq = self.g_n = 0.0
        self.a_max = self.g_max = 0.0

    def see_input(self, a: torch.Tensor) -> None:
        a = a.detach().float()
        self.a_sq += float((a ** 2).sum()); self.a_n += a.numel()
        self.a_max = max(self.a_max, float(a.abs().max()))

    def see_grad(self, g: torch.Tensor) -> None:
        g = g.detach().float()
        self.g_sq += float((g ** 2).sum()); self.g_n += g.numel()
        self.g_max = max(self.g_max, float(g.abs().max()))

    def ratios(self):
        a_rms = (self.a_sq / max(self.a_n, 1)) ** 0.5
        g_rms = (self.g_sq / max(self.g_n, 1)) ** 0.5
        return (self.a_max / max(a_rms, 1e-30), self.g_max / max(g_rms, 1e-30))


def measure(mx: MXModel, xs: torch.Tensor, device: str) -> dict:
    taps = {n: Tap() for n in mx.layers}
    handles = []

    def make(name):
        def hook(_mod, inputs, output):
            taps[name].see_input(inputs[0])
            if output.requires_grad:
                output.register_hook(lambda g, n=name: taps[n].see_grad(g))
        return hook

    for n, l in mx.layers.items():
        l.module.weight.requires_grad_(True)
        handles.append(l.module.register_forward_hook(make(n)))

    for i in range(xs.shape[0]):
        mx.module.zero_grad(set_to_none=True)
        z = mx.module(xs[i : i + 1].to(device))[0]
        t2 = torch.topk(z, 2)
        (t2.values[0] - t2.values[1]).backward()

    for h in handles:
        h.remove()
    for l in mx.layers.values():
        l.module.weight.requires_grad_(False)

    sizes = np.array([mx.layers[n].module.weight.numel() for n in taps], float)
    a_r = np.array([taps[n].ratios()[0] for n in taps])
    g_r = np.array([taps[n].ratios()[1] for n in taps])
    w = sizes / sizes.sum()
    return {"act_outlier": float((a_r * w).sum()), "act_outlier_max": float(a_r.max()),
            "grad_outlier": float((g_r * w).sum()), "grad_outlier_max": float(g_r.max())}


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

        # how peaked is the weight distribution the format actually stores?
        peaks = []
        for t in mx.weight_tensors().values():
            w = t.fmt.table()[t.codes][t.valid_mask()]
            rms = float(np.sqrt(np.mean(w ** 2)))
            peaks.append(float(np.abs(w).max() / max(rms, 1e-30)))
        rows.append({"fmt": fmt, "mantissa": MANTISSA[fmt],
                     "weight_outlier": float(np.mean(peaks)), **measure(mx, xs, dev)})
        mx.restore_weights()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"e14_{a.model}_sensitivity_origin.csv", index=False)

    print(f"model {a.model}: outlier ratio = max / rms, so a larger number means a "
          f"heavier tail\n")
    print(f"{'fmt':>6} {'man':>4} {'weights':>9} {'activations':>13} {'grad(out)':>11} "
          f"{'act max layer':>14}")
    for _, r in df.iterrows():
        print(f"{r.fmt:>6} {int(r.mantissa):>4} {r.weight_outlier:>9.2f} "
              f"{r.act_outlier:>13.2f} {r.grad_outlier:>11.2f} {r.act_outlier_max:>14.1f}")

    two = df[df.mantissa == 2].mean(numeric_only=True)
    three = df[df.mantissa == 3].mean(numeric_only=True)
    print(f"\n  2-mantissa vs 3-mantissa, ratio of tails:")
    for k, label in (("weight_outlier", "stored weights"),
                     ("act_outlier", "activations"), ("grad_outlier", "output gradients")):
        print(f"    {label:>18}: {three[k] / max(two[k], 1e-30):.2f}x heavier in 3-mantissa")
    print("\n  (F38 measured the sensitivity tail itself as ~14x heavier at the 99th pct)")


if __name__ == "__main__":
    main()
