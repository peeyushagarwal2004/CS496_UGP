"""E06 -- exhaustive fault injection: the ground truth the drafts ask for.

Every campaign so far *sampled* the fault space and reported a confidence
interval.  This injects **every single bit** of a model's MX weight storage,
so the resulting failure rate is not an estimate at all.  That makes it the
reference against which the sampled campaigns, their intervals, and the
stratified estimators can be checked -- the validation step the drafts call
ground truth and assume is only affordable on a small model.

At 8 ms per injection on an A100 the full ResNet8 MXFP8 space (638,624 bits)
takes about 1.4 h, so it is affordable for every width up to w48.

The run streams to CSV in chunks and is resumable: rerunning continues from
the number of rows already written, since the enumeration order is fixed.

Run::

    PYTHONPATH=. python3 -m experiments.e06_exhaustive --model resnet8_w16 \
        --fmt e4m3 --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from mxfi.campaign import Evaluator
from mxfi.data import campaign_subset
from mxfi.faults import Fault
from mxfi.models import to_deploy
from mxfi.sampling import FaultSite
from mxfi.stats import binomial_rate
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained, setup_device

RESULTS = Path(__file__).resolve().parent.parent / "results"


def enumerate_faults(tensors: dict) -> Iterator[FaultSite]:
    """Every addressable bit, in a fixed order so a resume is unambiguous.

    Padding lanes are skipped: they are real storage but carry no model value,
    so including them would dilute the true failure rate.
    """
    for name in sorted(tensors):
        mx = tensors[name]
        valid = mx.valid_mask()
        out, nblk, K = mx.codes.shape
        for r in range(out):
            for b in range(nblk):
                for l in range(K):
                    if valid[r, b, l]:
                        for bit in range(mx.fmt.width):
                            yield FaultSite(name, Fault("element", (r, b, l), bit))
        for r in range(out):
            for b in range(nblk):
                for bit in range(8):
                    yield FaultSite(name, Fault("scale", (r, b), bit))


def total_faults(tensors: dict) -> int:
    n = 0
    for mx in tensors.values():
        n += int(mx.valid_mask().sum()) * mx.fmt.width + mx.scales.size * 8
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8_w16")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--fmt", default="e4m3")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--scale-mode", default="ocp", choices=("ocp", "fit"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--limit", type=int, default=0, help="stop early (for testing)")
    a = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    dev = setup_device(a.device)
    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size, scale_mode=a.scale_mode)
    tag = a.model if a.train_seed == 0 else f"{a.model}_s{a.train_seed}"
    out_csv = RESULTS / f"e06_{tag}_{cfg.label()}_exhaustive.csv"

    net, state = load_trained(a.model, seed=a.train_seed)
    mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
    ev = Evaluator(campaign_subset(n_per_class=a.images, download=False), device=dev)
    golden = ev.golden(mx.module)

    tensors = mx.weight_tensors()
    total = total_faults(tensors)
    if a.limit:
        total = min(total, a.limit)

    done = 0
    if out_csv.exists():
        done = sum(1 for _ in open(out_csv)) - 1          # minus header
        done = max(done, 0)
        print(f"resuming: {done:,} of {total:,} already recorded")

    print(f"{a.model} seed {a.train_seed} | {cfg.label()} | device {dev} | "
          f"subset {golden.n} images | golden {golden.accuracy:.4f}")
    print(f"exhaustive fault space: {total:,} bits")

    rows, t0, n = [], time.time(), 0
    for i, site in enumerate(enumerate_faults(tensors)):
        if i < done:
            continue
        if a.limit and i >= a.limit:
            break
        with mx.fault(site):
            preds, finite = ev.predict(mx.module)
        changed = int((preds != golden.predictions).sum())
        f = site.fault
        rows.append({
            "i": i, "tensor": site.tensor, "site": f.site, "bit": f.bit,
            "row": f.index[0], "blk": f.index[1],
            "lane": f.index[2] if f.site == "element" else -1,
            "changed": changed, "sdc": changed > 0, "nonfinite": not finite,
        })
        n += 1
        if len(rows) >= a.chunk:
            _flush(rows, out_csv)
            rate = n / (time.time() - t0)
            left = (total - i - 1) / max(rate, 1e-9) / 3600
            print(f"  {i + 1:,}/{total:,} ({(i + 1) / total:.1%})  "
                  f"{rate:.1f} inj/s  ~{left:.2f} h left", flush=True)
            rows = []
    if rows:
        _flush(rows, out_csv)

    _report(out_csv, golden, cfg, a, complete=not a.limit)


def _flush(rows: list, path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a" if path.exists() else "w",
              header=not path.exists(), index=False)


def _report(path: Path, golden, cfg, a, complete: bool = True) -> None:
    d = pd.read_csv(path)
    print(f"\n=== EXHAUSTIVE TRUTH ({len(d):,} injections, no sampling) ===")
    print(f"model-wide SDC rate: {d.sdc.mean():.6f}")
    print(f"non-finite rate:     {d.nonfinite.mean():.6f}")
    print("\n--- by site ---")
    print(d.groupby("site").agg(n=("sdc", "size"), sdc=("sdc", "mean"),
                                imgs=("changed", "mean"),
                                nonfinite=("nonfinite", "mean")).to_string())
    print("\n--- by site x bit ---")
    print(d.groupby(["site", "bit"]).sdc.agg(["size", "mean"]).to_string())
    print("\n--- by layer ---")
    print(d.groupby("tensor").sdc.agg(["size", "mean"])
           .sort_values("mean", ascending=False).to_string())

    # ---- how well did sampling do?
    tag = a.model if a.train_seed == 0 else f"{a.model}_s{a.train_seed}"
    sampled = RESULTS / f"e01_{tag}_{cfg.label()}-n3000.csv"
    if not complete:
        print("\n(truncated run -- skipping the comparison against sampling, "
              "which is only meaningful over the whole fault space)")
    elif sampled.exists():
        s = pd.read_csv(sampled)
        est = binomial_rate(s["sdc"].to_numpy())
        truth = d.sdc.mean()
        covered = est.lo <= truth <= est.hi
        print(f"\n=== was the n=3000 sampled estimate right? ===")
        print(f"  truth      {truth:.6f}")
        print(f"  estimate   {est.rate:.6f}  95% CI [{est.lo:.6f}, {est.hi:.6f}]")
        print(f"  error      {est.rate - truth:+.6f} "
              f"({abs(est.rate - truth) / max(truth, 1e-9):.2%} relative)")
        print(f"  CI covers the truth: {'YES' if covered else 'NO'}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
