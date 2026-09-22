"""E08 -- why do two formats of equal width differ in vulnerability?

F34: `e3m2` and `e2m3` are both 6-bit, both free of NaN/Inf codes, and match
within 0.2 accuracy points on the ViT, yet their element SDC rates differ by
5.8x (0.0041 vs 0.0236). Severity arguments predict the opposite ordering:
`e3m2` has coarser mantissa steps (2 bits vs 3) and a wider exponent reach
(up to 16x per flip vs 4x), so per flip it should be the *more* damaging one.

The hypothesis tested here is that per-flip severity is the wrong quantity.
What reaches the network is the **absolute** weight perturbation, and that
depends on where a format places real weights inside its code space:

* a format with wide exponent reach (`e3m2`, range ~448x) lets most weights sit
  far below their block maximum, so a flip moves them by a small absolute
  amount relative to the layer's weight scale;
* a format with narrow reach (`e2m3`, range ~60x) compresses the same weights
  towards the top of its range, where the same relative change is a larger
  absolute one -- and pushes the smallest weights into flush-to-zero.

So this computes, for a *trained* model and each format, the exact distribution
of absolute perturbation per possible fault, normalised by the layer's weight
RMS. Nothing is sampled: every (element, bit) pair is enumerated using the
exact code tables, so the answer is a property of the format and the weights
rather than of an injection campaign.

Run::

    PYTHONPATH=. python3 -m experiments.e08_format_geometry --model vit_small
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.codec import quantize
from mxfi.formats import decode_e8m0, get_format
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"
FORMATS = ["e5m2", "e4m3", "e3m2", "e2m3", "e2m1"]


def layer_stats(mx_tensor, fmt) -> dict:
    """Exact per-fault perturbation statistics for one quantised tensor."""
    table = fmt.table()
    codes = mx_tensor.codes.astype(np.int64)             # (out, nblk, K)
    scale = decode_e8m0(mx_tensor.scales)[..., None]     # (out, nblk, 1)
    valid = mx_tensor.valid_mask()

    before_u = table[codes]                              # unit-scale values
    w = before_u * scale                                 # decoded weights
    rms = float(np.sqrt(np.mean(w[valid] ** 2)))

    bits = np.arange(fmt.width)
    flipped = table[codes[..., None] ^ (1 << bits)]      # (out, nblk, K, width)
    delta = np.abs(flipped - before_u[..., None]) * scale[..., None]
    delta = delta[valid]                                 # drop padding lanes

    finite = np.isfinite(delta)
    d = delta[finite] / max(rms, 1e-30)

    # how far below its block maximum does a typical weight sit?
    blk_max = np.max(np.abs(before_u), axis=-1, keepdims=True)
    nz = (np.abs(before_u) > 0) & valid
    headroom = np.log2(blk_max / np.maximum(np.abs(before_u), 1e-30))[nz]

    return {
        "n_faults": int(finite.sum()),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "p_gt_1rms": float((d > 1.0).mean()),
        "p_gt_3rms": float((d > 3.0).mean()),
        "zero_fraction": float((np.abs(before_u)[valid] == 0).mean()),
        "mean_log2_headroom": float(headroom.mean()),
        "weight_bits": int(valid.sum()) * fmt.width,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_small")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--formats", nargs="*", default=FORMATS)
    a = p.parse_args()

    net, _ = load_trained(a.model, seed=a.train_seed)
    deployed = to_deploy(net)

    rows = []
    for name in a.formats:
        fmt = get_format(name)
        mx = MXModel(deployed, MXConfig(fmt=name, block_size=a.block_size))
        mx.quantize_weights()
        per_layer = [(layer_stats(t, fmt), t) for t in mx.weight_tensors().values()]
        tot_bits = sum(s["weight_bits"] for s, _ in per_layer)

        # weight each layer by its share of the fault space: that is what a
        # uniform campaign actually samples
        agg = {k: sum(s[k] * s["weight_bits"] for s, _ in per_layer) / tot_bits
               for k in ("mean_delta", "median_delta", "p_gt_1rms", "p_gt_3rms",
                         "zero_fraction", "mean_log2_headroom")}
        rows.append({"fmt": name, "bits": fmt.width,
                     "range": fmt.max_normal / fmt.min_subnormal, **agg})
        mx.restore_weights()

    df = pd.DataFrame(rows)
    out = RESULTS / f"e08_{a.model}_format_geometry.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"model {a.model} seed {a.train_seed}, K={a.block_size}, "
          f"every (element, bit) pair enumerated exactly\n")
    print("  fmt   bits   range  mean|dw|/rms  median  P(>1rms)  P(>3rms)  zeros  "
          "mean log2(blockmax/|w|)")
    for _, r in df.iterrows():
        print(f"  {r.fmt:5} {int(r.bits):4d} {r['range']:7.0f} "
              f"{r.mean_delta:13.4f} {r.median_delta:7.4f} "
              f"{r.p_gt_1rms:9.4f} {r.p_gt_3rms:9.4f} {r.zero_fraction:6.3f} "
              f"{r.mean_log2_headroom:10.2f}")

    if {"e3m2", "e2m3"} <= set(df.fmt):
        a3 = df[df.fmt == "e3m2"].iloc[0]
        a2 = df[df.fmt == "e2m3"].iloc[0]
        print(f"\n  e2m3 / e3m2:  mean perturbation {a2.mean_delta / a3.mean_delta:.2f}x, "
              f"P(>1rms) {a2.p_gt_1rms / max(a3.p_gt_1rms, 1e-12):.2f}x, "
              f"P(>3rms) {a2.p_gt_3rms / max(a3.p_gt_3rms, 1e-12):.2f}x")
        print(f"  measured element-SDC ratio on this model was 5.8x "
              f"(0.0236 vs 0.0041)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
