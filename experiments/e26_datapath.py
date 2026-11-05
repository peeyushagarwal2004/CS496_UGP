"""E26 -- how the datapath's semantics change the catastrophic flip cases.

The flip-case tree (`mxfi.cases`) calls three cases catastrophic: an element code
flipped into NaN/Inf (E1), a scale flipped to 255 (S1), and a +128 scale flip that
overflows (S2). That verdict assumes one particular datapath: IEEE arithmetic in
which NaN survives every operation. Two common alternatives are tested here, on
the exhaustive ResNet8 campaigns, by re-injecting every fault of those cases:

* NaN-masking ReLU. IEEE 754-2008 maxNum(NaN, 0) returns 0, so a ReLU built on it
  (or written as ``x > 0 ? x : 0``) turns NaN into 0 instead of passing it on.
  The PyTorch ReLU used everywhere else in this study propagates NaN.
* Read-side guard. A special element code is decoded as 0 (E23's guard), so an
  E1 fault becomes "this one weight is zeroed".

Every E1 fault of the e4m3 network and every bit-7 (+128) scale fault of both
networks is injected; the unmodified outcome is already in the E06 campaign.

Run::

    PYTHONPATH=. python -m experiments.e26_datapath
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mxfi.campaign import Evaluator
from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

from .e25_flip_cases import CAMPAIGNS, N_IMAGES

RESULTS = Path(__file__).resolve().parent.parent / "results"


@contextmanager
def nan_masking_relu():
    """ReLU as ``x > 0 ? x : 0``: NaN and -Inf become 0, +Inf survives."""
    original = F.relu

    def relu(x, inplace=False):
        return torch.where(x > 0, x, torch.zeros_like(x))

    F.relu = relu
    try:
        yield
    finally:
        F.relu = original


def run(fmt: str, kinds: list[str], out: Path, n_e1: int, n_s2: int,
        limit: int = 0) -> None:
    csv, ckpt = CAMPAIGNS[fmt]
    labelled = pd.read_csv(RESULTS / f"e25_{fmt}_labelled.csv.gz", low_memory=False)
    net, _ = load_trained("resnet8_w16", path=ckpt)
    mx = MXModel(to_deploy(net), MXConfig(fmt=fmt, block_size=32)).quantize_weights()
    ev = Evaluator(campaign_subset(n_per_class=20, download=False))
    golden = ev.golden(mx.module)

    e1 = labelled[labelled.case.str.endswith("to_special")]
    s2 = labelled[(labelled.site == "scale") & (labelled.bit == 7)]
    # a uniform sample of each case; its mean damage is what the tree needs
    e1 = e1.sample(min(n_e1, len(e1)), random_state=26)
    s2 = s2.sample(min(n_s2, len(s2)), random_state=26)
    jobs = []
    if "guard" in kinds:
        jobs += [("guard", r) for r in e1.itertuples(index=False)]
    if "relu" in kinds:
        jobs += [("relu", r) for r in e1.itertuples(index=False)]
        jobs += [("relu", r) for r in s2.itertuples(index=False)]
    if limit:
        jobs = [jobs[i] for i in np.linspace(0, len(jobs) - 1, limit).astype(int)]
    print(f"{fmt}: {len(jobs):,} injections", flush=True)

    rows = []
    for i, (kind, r) in enumerate(jobs):
        lay = mx.layers[r.tensor]
        if r.site == "element":
            idx = (int(r.row), int(r.blk), int(r.lane))
        else:
            idx = (int(r.row), int(r.blk))
        if kind == "guard":
            col = idx[1] * lay.mx.block_size + idx[2]
            ctx = mx._patched(lay, (idx[0], col, np.zeros(1, np.float32)))
            with ctx:
                preds, finite = ev.predict(mx.module)
        else:
            with nan_masking_relu(), mx.word_fault(r.tensor, r.site, idx, 1 << int(r.bit)):
                preds, finite = ev.predict(mx.module)
        rows.append({"fmt": fmt, "variant": kind, "tensor": r.tensor, "site": r.site,
                     "bit": int(r.bit), "row": idx[0], "blk": idx[1],
                     "lane": idx[2] if r.site == "element" else -1, "case": r.case,
                     "damage_ieee": r.damage, "nonfinite_ieee": r.nonfinite,
                     "damage": int((preds != golden.predictions).sum()) / N_IMAGES,
                     "nonfinite": not finite})
        if (i + 1) % 500 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  {i + 1:,}/{len(jobs):,}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fmt", nargs="*", default=["e4m3", "e3m2"])
    p.add_argument("--n-e1", type=int, default=2000, help="E1 faults per variant")
    p.add_argument("--n-s2", type=int, default=1000, help="bit-7 scale faults")
    p.add_argument("--limit", type=int, default=0, help="evenly spaced subset (smoke test)")
    a = p.parse_args()
    torch.set_num_threads(4)
    for fmt in a.fmt:
        kinds = ["guard", "relu"] if fmt == "e4m3" else ["relu"]
        tag = f"_smoke" if a.limit else ""
        run(fmt, kinds, RESULTS / f"e26_{fmt}_datapath{tag}.csv", a.n_e1, a.n_s2, a.limit)
    print("done", flush=True)


if __name__ == "__main__":
    main()
