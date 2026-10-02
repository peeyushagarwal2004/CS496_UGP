"""E23 -- does a read-side NaN guard remove the NaN pathway?

F18, F58, F66 and F67 all point at one mechanism: in formats that own NaN/Inf
codes, a small fraction of element faults flip a code into a special value, the
special value reaches the output, and every inference is lost. On ResNet8 under
e4m3 those faults are 1.27% of the element fault space and carry 78% of its damage.

They are also exactly enumerable, which suggests a cheap defence: decode any
special element code as zero on read. A fault that would have produced NaN then
merely zeroes one weight. This measures what that buys, on the special-code faults
themselves and on the element fault space as a whole.

Every element fault that lands on a special code is found by enumeration (no
injection needed); a sample of them is injected twice, unguarded and guarded.
The rest of the element space is unaffected by the guard, so the whole-space
effect follows from the exhaustive campaign without further injections.

Run::

    PYTHONPATH=. python -m experiments.e23_nan_guard --fmt e4m3 --sample 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMAGES = 200


def special_faults(mx) -> pd.DataFrame:
    """Every (weight, bit) whose flip lands an element on a NaN/Inf code."""
    rows = []
    for name, lay in mx.layers.items():
        t = lay.mx
        table = t.fmt.table()
        valid = t.valid_mask()
        for bit in range(t.fmt.width):
            after = table[t.codes ^ np.uint8(1 << bit)]
            hit = valid & ~np.isfinite(after) & np.isfinite(table[t.codes])
            for r, b, l in zip(*np.nonzero(hit)):
                rows.append({"tensor": name, "row": int(r), "blk": int(b),
                             "lane": int(l), "bit": bit})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet8_w16")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--fmt", default="e4m3")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--sample", type=int, default=1000)
    p.add_argument("--exhaustive", default="",
                   help="the exhaustive campaign of this network, for the whole-space effect")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    from mxfi.campaign import Evaluator
    from mxfi.data import campaign_subset
    from mxfi.models import to_deploy
    from mxfi.torch_mx import MXConfig, MXModel
    from mxfi.train import load_trained, setup_device

    dev = setup_device(a.device)
    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size)
    net, _ = load_trained(a.model, seed=a.train_seed)
    mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
    ev = Evaluator(campaign_subset(n_per_class=IMAGES // 10, download=False), device=dev)
    golden = ev.golden(mx.module)

    sp = special_faults(mx)
    n_el = sum(int(l.mx.valid_mask().sum()) * l.mx.fmt.width for l in mx.layers.values())
    print(f"{a.model} s{a.train_seed} {cfg.label()}: golden {golden.accuracy:.4f}")
    print(f"element faults landing on a special code: {len(sp):,} of {n_el:,} "
          f"({len(sp) / n_el:.3%})")
    print("  by bit:", sp.bit.value_counts().sort_index().to_dict())

    pick = sp.sample(min(a.sample, len(sp)), random_state=23).reset_index(drop=True)
    out = []
    for _, f in pick.iterrows():
        lay = mx.layers[f.tensor]
        col = int(f.blk) * lay.mx.block_size + int(f.lane)
        mask = 1 << int(f.bit)
        with mx.word_fault(f.tensor, "element", (f.row, f.blk, f.lane), mask):
            preds, finite = ev.predict(mx.module)
        raw = int((preds != golden.predictions).sum())
        # the guard: a special code decodes as zero, so the weight is simply zeroed
        with mx._patched(lay, (int(f.row), col, np.zeros(1, np.float32))):
            preds_g, finite_g = ev.predict(mx.module)
        guarded = int((preds_g != golden.predictions).sum())
        out.append({**f.to_dict(), "r_raw": raw / IMAGES, "nonfinite_raw": not finite,
                    "r_guarded": guarded / IMAGES, "nonfinite_guarded": not finite_g})
    df = pd.DataFrame(out)
    tag = a.model if a.train_seed == 0 else f"{a.model}_s{a.train_seed}"
    path = RESULTS / f"e23_{tag}_{cfg.label()}_nan_guard.csv"
    df.to_csv(path, index=False)

    print(f"\n{len(df)} special-code faults injected, unguarded vs guarded:")
    print(f"  per-inference rate  {df.r_raw.mean():.4f} -> {df.r_guarded.mean():.6f} "
          f"({df.r_raw.mean() / max(df.r_guarded.mean(), 1e-9):,.0f}x lower)")
    print(f"  any image changed   {(df.r_raw > 0).mean():.4f} -> {(df.r_guarded > 0).mean():.4f}")
    print(f"  non-finite output   {df.nonfinite_raw.mean():.4f} -> "
          f"{df.nonfinite_guarded.mean():.4f}")

    if a.exhaustive:
        d = pd.read_csv(a.exhaustive)
        e = d[d.site == "element"]
        key = ["tensor", "row", "blk", "lane", "bit"]
        m = e.merge(sp.assign(special=True), on=key, how="left")
        special = m.special.fillna(False).to_numpy(bool)
        r = m.changed.to_numpy() / IMAGES
        before = r.mean()
        after = (r[~special].sum() + special.sum() * df.r_guarded.mean()) / len(r)
        tot_before = d.changed.mean() / IMAGES
        tot_after = (d.changed.sum() / IMAGES - r[special].sum()
                     + special.sum() * df.r_guarded.mean()) / len(d)
        print(f"\nwhole space, from the exhaustive campaign ({special.sum():,} special faults matched):")
        print(f"  element per-inference rate {before:.6f} -> {after:.6f} "
              f"({before / after:.2f}x lower)")
        print(f"  all faults (element + scale) {tot_before:.6f} -> {tot_after:.6f} "
              f"({tot_before / tot_after:.2f}x lower)")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
