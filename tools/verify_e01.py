"""Check that a recorded uniform campaign (E01) belongs to the checkpoint in hand.

`verify_ground_truth` does this for exhaustive campaigns, whose rows carry
row/blk/lane columns; E01 rows carry the fault index as a tuple string instead.
Same test: replay a sample of recorded faults and compare the number of images
each corrupted with what was recorded. A matched pair agrees on essentially
every fault.

Run::

    PYTHONPATH=. python -m tools.verify_e01 results/e01_repvgg_a0_e4m3-K32-ocp-w-n3000.csv \\
        --model repvgg_a0 --fmt e4m3 --device cuda
"""

from __future__ import annotations

import argparse
import ast

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--model", required=True)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--fmt", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    from mxfi.campaign import Evaluator
    from mxfi.data import campaign_images
    from mxfi.faults import Fault
    from mxfi.models import to_deploy
    from mxfi.sampling import FaultSite
    from mxfi.torch_mx import MXConfig, MXModel
    from mxfi.train import load_trained, setup_device

    dev = setup_device(a.device)
    net, _ = load_trained(a.model, seed=a.train_seed)
    mx = MXModel(to_deploy(net).to(dev), MXConfig(fmt=a.fmt)).quantize_weights()
    ev = Evaluator(campaign_images(a.model, 200), device=dev)
    golden = ev.golden(mx.module)

    d = pd.read_csv(a.csv)
    pick = d.iloc[np.random.default_rng(0).choice(len(d), min(a.n, len(d)), replace=False)]
    exact = 0
    for r in pick.itertuples():
        site = FaultSite(r.tensor, Fault(r.site, tuple(ast.literal_eval(r.index)), int(r.bit)))
        with mx.fault(site):
            preds, _ = ev.predict(mx.module)
        exact += int((preds != golden.predictions).sum()) == int(r.changed)
    print(f"{exact}/{len(pick)} replayed faults corrupt exactly the recorded number of images")
    print("MATCHED" if exact >= 0.98 * len(pick) else "MISMATCH: not recorded from this checkpoint")


if __name__ == "__main__":
    main()
