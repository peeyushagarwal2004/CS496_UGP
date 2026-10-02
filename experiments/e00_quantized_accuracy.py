"""E00 -- MX quantised accuracy baseline across the study grid.

Every failure rate the study reports is relative to a *quantised* baseline, so
this has to come first.  It also settles a question the earlier synthetic-data
analysis raised: which formats are genuinely matched in accuracy, and are
therefore usable for the "same accuracy, different reliability" comparison at
the heart of the drafts.

Run::

    .venv/Scripts/python.exe -m experiments.e00_quantized_accuracy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from mxfi.codec import headroom
from mxfi.data import dataset_of, loaders
from mxfi.formats import get_format
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import evaluate_accuracy, load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"
FORMATS = ["e4m3", "e5m2", "e3m2", "e2m3", "e2m1"]
BLOCKS = [8, 16, 32, 64]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8",
                   choices=sorted(__import__("mxfi.models", fromlist=["MODELS"]).MODELS))
    p.add_argument("--formats", nargs="*", default=FORMATS)
    p.add_argument("--blocks", nargs="*", type=int, default=BLOCKS)
    p.add_argument("--modes", nargs="*", default=["ocp", "fit"])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    from mxfi.train import setup_device
    dev = setup_device(a.device)
    net, state = load_trained(a.model)
    print(f"checkpoint: epoch {state['epoch']}, fp32 test acc {state['acc']:.4f}")

    folded = to_deploy(net).to(dev)
    _, test = loaders(dataset_of(a.model), download=False)

    fp32 = evaluate_accuracy(folded, test, dev)
    print(f"fp32 (BN-folded) accuracy: {fp32:.4f}\n")

    rows = []
    for fmt in a.formats:
        f = get_format(fmt)
        for K in a.blocks:
            for mode in a.modes:
                cfg = MXConfig(fmt=fmt, block_size=K, scale_mode=mode)
                mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
                t0 = time.time()
                acc = evaluate_accuracy(mx.module, test, dev)
                rows.append({
                    "fmt": fmt, "bits": f.width, "block_size": K,
                    "scale_mode": mode, "accuracy": acc,
                    "drop_vs_fp32": fp32 - acc,
                    "headroom": headroom(f),
                    "secs": time.time() - t0,
                })
                print(f"  {fmt:5} {f.width}b K={K:2d} {mode:4} -> {acc:.4f} "
                      f"({fp32 - acc:+.4f})")

    df = pd.DataFrame(rows)
    out = RESULTS / f"e00_{a.model}_quantized_accuracy.csv"
    df.to_csv(out, index=False)

    print("\n=== accuracy by format (K=32) ===")
    piv = df[df.block_size == 32].pivot(index="fmt", columns="scale_mode",
                                        values="accuracy")
    piv["bits"] = [get_format(i).width for i in piv.index]
    print(piv.sort_values("bits").to_string())

    print("\n=== block-size effect (ocp) ===")
    print(df[df.scale_mode == "ocp"]
          .pivot(index="fmt", columns="block_size", values="accuracy")
          .to_string())

    print(f"\nfp32 baseline {fp32:.4f}   wrote {out}")


if __name__ == "__main__":
    main()
