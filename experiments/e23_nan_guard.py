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
injection needed), so its share f of the element space is exact. A sample of them
is injected twice, unguarded and guarded. The guard changes nothing for any other
fault, so a uniform sample of the *non-special* faults, injected once, completes the
picture:

    element rate, unguarded = f * r_special      + (1 - f) * r_other
    element rate, guarded   = f * r_special_guarded + (1 - f) * r_other

which needs no earlier campaign, and so cannot be scored against the wrong network.
When an exhaustive campaign of the same network exists, `--exhaustive` also reports
the whole-space figure from it, as a check on the sampled one.

Run::

    PYTHONPATH=. python -m experiments.e23_nan_guard --fmt e4m3 --sample 1000
    PYTHONPATH=. python3 -m experiments.e23_nan_guard --model repvgg_a0 \\
        --train-seed 1 --fmt e5m2 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMAGES = 200


def special_faults(mx) -> pd.DataFrame:
    """Every (weight, bit) whose flip lands a finite element on a NaN/Inf code."""
    cols = {"tensor": [], "row": [], "blk": [], "lane": [], "bit": []}
    for name, lay in mx.layers.items():
        t = lay.mx
        table = t.fmt.table()
        base = t.valid_mask() & np.isfinite(table[t.codes])
        for bit in range(t.fmt.width):
            hit = base & ~np.isfinite(table[t.codes ^ np.uint8(1 << bit)])
            r, b, l = np.nonzero(hit)
            cols["tensor"].append(np.full(len(r), name, dtype=object))
            cols["row"].append(r); cols["blk"].append(b); cols["lane"].append(l)
            cols["bit"].append(np.full(len(r), bit))
    return pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})


def is_special(mx, name: str, row: int, blk: int, lane: int, bit: int) -> bool:
    t = mx.layers[name].mx
    table = t.fmt.table()
    c = int(t.codes[row, blk, lane])
    return bool(np.isfinite(table[c]) and not np.isfinite(table[c ^ (1 << bit)]))


def sample_other(mx, n: int, rng: np.random.Generator) -> list[tuple]:
    """Uniform element faults over valid lanes and bits, special ones rejected."""
    names = list(mx.layers)
    sizes = np.array([int(mx.layers[k].mx.valid_mask().sum()) * mx.layers[k].mx.fmt.width
                      for k in names], dtype=float)
    out = []
    while len(out) < n:
        name = names[rng.choice(len(names), p=sizes / sizes.sum())]
        t = mx.layers[name].mx
        r, b, l = (int(rng.integers(0, s)) for s in t.codes.shape)
        if not t.valid_mask()[r, b, l]:
            continue
        bit = int(rng.integers(0, t.fmt.width))
        if not is_special(mx, name, r, b, l, bit):
            out.append((name, r, b, l, bit))
    return out


def upper(y: np.ndarray) -> float:
    """95% upper bound on the mean of per-fault rates in [0, 1].

    A normal interval when anything was corrupted; when nothing was, the rule of
    three, since a mean of mu forces P(y > 0) >= mu.
    """
    if y.max() == 0:
        return 3.0 / len(y)
    return float(y.mean() + 1.96 * y.std(ddof=1) / np.sqrt(len(y)))


def summary() -> None:
    """Every E23 run, with the guarded rate as an upper bound rather than a point."""
    rows = []
    for path in sorted(RESULTS.glob("e23_*_nan_guard.csv")):
        d = pd.read_csv(path)
        f = float(d.special_fraction.iloc[0])
        s, o = d[d.kind == "special"], d[d.kind == "other"]
        before = f * s.r_raw.mean() + (1 - f) * o.r_raw.mean()
        after = f * s.r_guarded.mean() + (1 - f) * o.r_raw.mean()
        after_hi = f * upper(s.r_guarded.to_numpy()) + (1 - f) * upper(o.r_raw.to_numpy())
        rows.append({
            "run": path.stem.replace("e23_", "").replace("_nan_guard", ""),
            "special_fraction": f, "special_share_of_damage": f * s.r_raw.mean() / before,
            "element_rate_unguarded": before, "element_rate_guarded": after,
            "guarded_upper95": after_hi, "reduction_at_least": before / after_hi,
        })
    t = pd.DataFrame(rows)
    out = RESULTS / "e23_nan_guard_summary.csv"
    t.to_csv(out, index=False)
    with pd.option_context("display.width", 200):
        print(t.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print(f"\nwrote {out}")


def main() -> None:
    if "--summary" in __import__("sys").argv:
        summary()
        return
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet8_w16")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--fmt", default="e4m3")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--sample", type=int, default=1000, help="special-code faults to inject")
    p.add_argument("--others", type=int, default=2000,
                   help="non-special element faults to inject, for the whole-space rate")
    p.add_argument("--exhaustive", default="",
                   help="the exhaustive campaign of this network, as a cross-check")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    from mxfi.campaign import Evaluator
    from mxfi.data import campaign_images
    from mxfi.models import to_deploy
    from mxfi.torch_mx import MXConfig, MXModel
    from mxfi.train import load_trained, setup_device

    dev = setup_device(a.device)
    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size)
    net, _ = load_trained(a.model, seed=a.train_seed)
    mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
    ev = Evaluator(campaign_images(a.model, IMAGES), device=dev)
    golden = ev.golden(mx.module)

    def changed(preds) -> int:
        return int((preds != golden.predictions).sum())

    sp = special_faults(mx)
    n_el = sum(int(l.mx.valid_mask().sum()) * l.mx.fmt.width for l in mx.layers.values())
    f = len(sp) / n_el
    print(f"{a.model} s{a.train_seed} {cfg.label()}: golden {golden.accuracy:.4f}", flush=True)
    print(f"element faults landing on a special code: {len(sp):,} of {n_el:,} ({f:.3%})")
    print("  by bit:", sp.bit.value_counts().sort_index().to_dict(), flush=True)

    rows = []
    pick = sp.sample(min(a.sample, len(sp)), random_state=23)
    for r in pick.itertuples(index=False):
        lay = mx.layers[r.tensor]
        col = int(r.blk) * lay.mx.block_size + int(r.lane)
        idx = (int(r.row), int(r.blk), int(r.lane))
        with mx.word_fault(r.tensor, "element", idx, 1 << int(r.bit)):
            preds, finite = ev.predict(mx.module)
        # the guard: a special code decodes as zero, so the weight is simply zeroed
        with mx._patched(lay, (int(r.row), col, np.zeros(1, np.float32))):
            preds_g, finite_g = ev.predict(mx.module)
        rows.append({"kind": "special", "tensor": r.tensor, "row": idx[0], "blk": idx[1],
                     "lane": idx[2], "bit": int(r.bit),
                     "r_raw": changed(preds) / IMAGES, "nonfinite_raw": not finite,
                     "r_guarded": changed(preds_g) / IMAGES, "nonfinite_guarded": not finite_g})
    for name, r, b, l, bit in sample_other(mx, a.others, np.random.default_rng(23)):
        with mx.word_fault(name, "element", (r, b, l), 1 << bit):
            preds, finite = ev.predict(mx.module)
        y = changed(preds) / IMAGES
        rows.append({"kind": "other", "tensor": name, "row": r, "blk": b, "lane": l,
                     "bit": bit, "r_raw": y, "nonfinite_raw": not finite,
                     "r_guarded": y, "nonfinite_guarded": not finite})
    df = pd.DataFrame(rows)
    tag = a.model if a.train_seed == 0 else f"{a.model}_s{a.train_seed}"
    path = RESULTS / f"e23_{tag}_{cfg.label()}_nan_guard.csv"
    df.assign(special_fraction=f).to_csv(path, index=False)

    s, o = df[df.kind == "special"], df[df.kind == "other"]
    print(f"\n{len(s)} special-code faults injected, unguarded vs guarded:")
    print(f"  per-inference rate  {s.r_raw.mean():.4f} -> {s.r_guarded.mean():.6f} "
          f"({s.r_raw.mean() / max(s.r_guarded.mean(), 1e-9):,.0f}x lower)")
    print(f"  non-finite output   {s.nonfinite_raw.mean():.4f} -> "
          f"{s.nonfinite_guarded.mean():.4f}")
    before = f * s.r_raw.mean() + (1 - f) * o.r_raw.mean()
    after = f * s.r_guarded.mean() + (1 - f) * o.r_raw.mean()
    print(f"\n{len(o)} other element faults: rate {o.r_raw.mean():.6f}, "
          f"non-finite {o.nonfinite_raw.mean():.4f}")
    print(f"whole element space, sampled: {before:.6f} -> {after:.6f} "
          f"({before / max(after, 1e-12):.2f}x lower); special share of damage "
          f"{f * s.r_raw.mean() / max(before, 1e-12):.1%}")

    if a.exhaustive:
        d = pd.read_csv(a.exhaustive)
        e = d[d.site == "element"]
        key = ["tensor", "row", "blk", "lane", "bit"]
        m = e.merge(sp.assign(special=True), on=key, how="left")
        special = m.special.fillna(False).to_numpy(bool)
        y = m.changed.to_numpy() / IMAGES
        ex_before = y.mean()
        ex_after = (y[~special].sum() + special.sum() * s.r_guarded.mean()) / len(y)
        print(f"whole element space, exhaustive: {ex_before:.6f} -> {ex_after:.6f} "
              f"({ex_before / ex_after:.2f}x lower)")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
