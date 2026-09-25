"""Check that a recorded campaign's outcomes belong to the checkpoint in hand.

The same model name can refer to different trained weights on different machines
-- training is not bit-reproducible across GPUs and library versions -- so an
analysis that scores faults with one checkpoint against outcomes recorded from
another will silently measure the wrong thing. This replays a random sample of a
recorded campaign and compares the measured outcome with the recorded one. A
matched pair agrees on essentially every fault; a mismatched pair does not.

Run::

    PYTHONPATH=. python3 -m tools.verify_ground_truth \\
        results/e06_resnet8_w16_e3m2-K32-ocp-w_exhaustive.csv --train-seed 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.campaign import Evaluator
from mxfi.data import campaign_subset
from mxfi.faults import Fault
from mxfi.models import to_deploy
from mxfi.sampling import FaultSite
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained, setup_device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="a recorded campaign with row/blk/lane/bit columns")
    p.add_argument("--model", default="resnet8_w16")
    p.add_argument("--fmt", default="e3m2")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    d = pd.read_csv(a.csv)
    rng = np.random.default_rng(0)
    pick = d.iloc[rng.choice(len(d), min(a.n, len(d)), replace=False)]

    dev = setup_device(a.device)
    net, state = load_trained(a.model, seed=a.train_seed)
    mx = MXModel(to_deploy(net).to(dev),
                 MXConfig(fmt=a.fmt, block_size=a.block_size)).quantize_weights()
    ev = Evaluator(campaign_subset(n_per_class=a.images, download=False), device=dev)
    golden = ev.golden(mx.module)
    print(f"{a.model} train-seed {a.train_seed}: checkpoint acc {state['acc']:.4f}, "
          f"golden acc on the subset {golden.accuracy:.4f}")
    print(f"replaying {len(pick)} of {len(d):,} recorded faults from {Path(a.csv).name}")

    agree = 0
    for _, r in pick.iterrows():
        idx = ((int(r.row), int(r.blk), int(r.lane)) if r.site == "element"
               else (int(r.row), int(r.blk)))
        site = FaultSite(r.tensor, Fault(r.site, idx, int(r.bit)))
        with mx.fault(site):
            preds, _ = ev.predict(mx.module)
        agree += int(bool((preds != golden.predictions).sum() > 0) == bool(r.sdc))

    frac = agree / len(pick)
    print(f"\nagreement with the recorded outcome: {agree}/{len(pick)} = {frac:.3f}")
    print("  a matched checkpoint agrees on essentially every fault;" if frac > 0.98
          else "  MISMATCH: these outcomes were not recorded from this checkpoint;")
    print(f"  recorded rate {pick.sdc.mean():.4f} over this sample")


if __name__ == "__main__":
    main()
