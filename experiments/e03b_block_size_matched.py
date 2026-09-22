"""E03b -- block size with *matched* faults (controlled version of E03).

E03 sampled faults independently per block size, so its element-side trend
across K mixed the effect of K with the effect of sampling different weights.
The perturbation physics says the element side should be nearly flat: for a
fixed weight and bit, mean |dw| moves only ~9% across K=8..64.  E03 showed a
2x swing, which the physics cannot produce.

This version removes the confound.  One fixed set of (layer, weight position,
bit) triples is chosen once and injected into *every* block size, so the only
thing that varies is how weights are grouped into scale-sharing blocks.  The
weight position is expressed as a flat index into the (out, reduction) view
and re-mapped per K::

    row, col = divmod(flat, reduction)
    index    = (row, col // K, col % K)

Run::

    .venv/Scripts/python.exe -m experiments.e03b_block_size_matched --n 1500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.campaign import Evaluator, run_campaign
from mxfi.data import campaign_subset
from mxfi.faults import Fault
from mxfi.models import to_deploy
from mxfi.sampling import FaultSite
from mxfi.stats import binomial_rate
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8",
                   choices=sorted(__import__("mxfi.models", fromlist=["MODELS"]).MODELS))
    p.add_argument("--fmt", default="e4m3")
    p.add_argument("--blocks", nargs="*", type=int, default=[8, 16, 32, 64])
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--scale-mode", default="ocp", choices=("ocp", "fit"))
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    net, _ = load_trained(a.model)
    ev = Evaluator(campaign_subset(n_per_class=a.images, download=False))
    rng = np.random.default_rng(a.seed)

    # ---- choose the fault set once, in (layer, flat position, bit) terms
    probe = MXModel(to_deploy(net), MXConfig(a.fmt, 32)).quantize_weights()
    shapes = {n: l.mx.orig_shape for n, l in probe.layers.items()}
    names = list(shapes)
    sizes = np.array([shapes[n][0] * shapes[n][1] for n in names], float)
    per_layer = rng.multinomial(a.n, sizes / sizes.sum())

    chosen: list[tuple[str, int, int]] = []
    for name, k in zip(names, per_layer):
        out, red = shapes[name]
        flats = rng.integers(0, out * red, size=int(k))
        bits = rng.integers(0, probe.layers[name].mx.fmt.width, size=int(k))
        chosen += [(name, int(f), int(b)) for f, b in zip(flats, bits)]
    print(f"{len(chosen)} matched faults across {len(names)} layers, "
          f"injected identically into every K\n")

    rows = []
    for K in a.blocks:
        cfg = MXConfig(fmt=a.fmt, block_size=K, scale_mode=a.scale_mode)
        mx = MXModel(to_deploy(net), cfg).quantize_weights()
        golden = ev.golden(mx.module)

        faults = []
        for name, flat, bit in chosen:
            red = shapes[name][1]
            row, col = divmod(flat, red)
            faults.append(FaultSite(name, Fault("element", (row, col // K, col % K), bit)))

        t0 = time.time()
        df = run_campaign(mx, faults, ev, golden,
                          out_csv=RESULTS / f"e03b_{a.model}_{cfg.label()}.csv", progress=False)
        est = binomial_rate(df.sdc.to_numpy())
        rows.append({"K": K, "element_sdc": est.rate, "hw": est.half_width,
                     "mean_changed": float(df.changed.mean()),
                     "median_changed": float(df.changed.median()),
                     "nonfinite": float(df.nonfinite.mean()),
                     "golden_acc": golden.accuracy, "n": len(df)})
        print(f"K={K:2d}  element SDC {est.rate:.4f} +-{est.half_width:.4f}  "
              f"mean_changed {df.changed.mean():5.2f}  nonfinite {df.nonfinite.mean():.2%}  "
              f"golden {golden.accuracy:.4f}  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"e03b_{a.model}_block_size_matched_{a.scale_mode}.csv", index=False)

    lo, hi = out.element_sdc.min(), out.element_sdc.max()
    print(f"\nspread across K: {lo:.4f} .. {hi:.4f}  ({hi/max(lo,1e-9):.2f}x)")
    print("CIs overlap:" if (out.element_sdc.max() - out.element_sdc.min())
          < 2 * out.hw.max() else "CIs disjoint:", "->",
          "element side is flat in K" if (out.element_sdc.max() - out.element_sdc.min())
          < 2 * out.hw.max() else "element side genuinely depends on K")
    print(f"\nwrote {RESULTS / 'e03b_block_size_matched.csv'}")


if __name__ == "__main__":
    main()
