"""Prove the fast injection path is byte-identical to the original.

Runs both paths over the same faults and compares the resulting weight tensors
exactly, including the cases most likely to break a shortcut: padding lanes,
the scale MSB (which drives a block past the float32 range), and flips that
land on the reserved NaN scale code.  Also checks that the weights are
restored exactly on exit.

Standalone (no pytest needed, so it runs on the cluster too)::

    python tools/verify_fast_inject.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from mxfi.faults import Fault
from mxfi.models import build_model, to_deploy
from mxfi.sampling import FaultSite
from mxfi.torch_mx import MXConfig, MXModel


def snapshot(mx: MXModel) -> dict:
    return {n: l.module.weight.detach().clone() for n, l in mx.layers.items()}


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact equality, treating NaN in the same position as equal."""
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    failures = 0
    checked = 0

    for fmt, K in (("e4m3", 32), ("e2m1", 8), ("e5m2", 64)):
        net = to_deploy(build_model("resnet8_w16").eval())
        mx = MXModel(net, MXConfig(fmt=fmt, block_size=K)).quantize_weights()
        clean = snapshot(mx)
        names = list(mx.layers)

        faults = []
        for name in names:
            q = mx.layers[name].mx
            out, nblk, _ = q.codes.shape
            for _ in range(25):                       # random element faults
                idx = (int(rng.integers(out)), int(rng.integers(nblk)),
                       int(rng.integers(K)))
                faults.append(FaultSite(name, Fault("element", idx,
                                                    int(rng.integers(q.fmt.width)))))
            for _ in range(15):                       # random scale faults
                idx = (int(rng.integers(out)), int(rng.integers(nblk)))
                faults.append(FaultSite(name, Fault("scale", idx,
                                                    int(rng.integers(8)))))
            # the pathological ones, explicitly
            faults.append(FaultSite(name, Fault("scale", (0, 0), 7)))     # MSB
            faults.append(FaultSite(name, Fault("element", (0, nblk - 1, K - 1),
                                                q.fmt.width - 1)))        # last lane
            faults.append(FaultSite(name, Fault("element", (0, 0, 0), 0)))  # LSB

        for site in faults:
            with mx.fault(site):
                fast = snapshot(mx)
            restored_fast = snapshot(mx)
            with mx._fault_reference(site):
                ref = snapshot(mx)
            restored_ref = snapshot(mx)
            checked += 1

            for n in names:
                if not same(fast[n], ref[n]):
                    d = int((~((fast[n] == ref[n]) |
                               (torch.isnan(fast[n]) & torch.isnan(ref[n])))).sum())
                    print(f"MISMATCH {fmt} K={K} {site.tensor} {site.fault} "
                          f"-> layer {n}: {d} weights differ")
                    failures += 1
                if not same(restored_fast[n], clean[n]) or \
                        not same(restored_ref[n], clean[n]):
                    print(f"NOT RESTORED {fmt} K={K} {site.tensor} {site.fault} "
                          f"-> layer {n}")
                    failures += 1

    print(f"checked {checked} faults across 3 formats/block sizes: "
          f"{'ALL IDENTICAL' if failures == 0 else str(failures) + ' FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
