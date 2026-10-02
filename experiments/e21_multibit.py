"""E21 -- single-bit versus multi-bit (H-model) fault occurrence.

The plan left one fault-model choice open: whether the single-bit flip it adopts
is enough, or whether the multi-bit tail of its H-model has to be injected too.
Every campaign so far flipped exactly one bit. This measures what a second flip
in the *same stored word* does, which is the case the H-model's Poisson-binomial
multiplicity term describes: a double upset in one element code or one E8M0
scale byte.

Design. Sample words uniformly -- element codes and scale bytes separately --
and inject, for every sampled word, *every* single-bit mask and *every* two-bit
mask. Each injection records which of the evaluation images changed prediction
(packed into a hex string), not just how many, so a two-bit fault can be compared
exactly with the union of the images its two constituent single-bit faults
corrupt. That comparison is the whole question: if a double flip does what its
two single flips do between them, a multi-bit campaign can be synthesised from
single-bit data without a single new injection; if it does something else, the
single-bit model is missing a mechanism.

Run::

    PYTHONPATH=. python3 -m experiments.e21_multibit run --fmt e4m3 \\
        --train-seed 0 --device cuda
    PYTHONPATH=. python3 -m experiments.e21_multibit analyse
"""

from __future__ import annotations

import argparse
import itertools
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMAGES = 200


def masks(width: int) -> list[int]:
    """Every one- and two-bit XOR mask of a `width`-bit word."""
    one = [1 << b for b in range(width)]
    two = [(1 << a) | (1 << b) for a, b in itertools.combinations(range(width), 2)]
    return one + two


def sample_words(tensors: dict, n_el: int, n_sc: int, rng: np.random.Generator):
    """Uniformly sampled element words (valid lanes only) and scale words."""
    el, sc = [], []
    for name in sorted(tensors):
        mx = tensors[name]
        for r, b, l in zip(*np.nonzero(mx.valid_mask())):
            el.append((name, (int(r), int(b), int(l))))
        for r, b in zip(*np.nonzero(np.ones(mx.scales.shape, bool))):
            sc.append((name, (int(r), int(b))))
    pick_el = rng.choice(len(el), n_el, replace=False)
    pick_sc = rng.choice(len(sc), n_sc, replace=False)
    return ([("element",) + el[i] for i in pick_el]
            + [("scale",) + sc[i] for i in pick_sc])


def run(a) -> None:
    from mxfi.campaign import Evaluator
    from mxfi.data import campaign_subset
    from mxfi.models import to_deploy
    from mxfi.torch_mx import MXConfig, MXModel
    from mxfi.train import load_trained, setup_device

    dev = setup_device(a.device)
    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size)
    out = RESULTS / f"e21_{a.model}_s{a.train_seed}_{cfg.label()}_multibit.csv"

    net, _ = load_trained(a.model, seed=a.train_seed)
    mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
    ev = Evaluator(campaign_subset(n_per_class=IMAGES // 10, download=False), device=dev)
    golden = ev.golden(mx.module)
    tensors = mx.weight_tensors()
    words = sample_words(tensors, a.element_words, a.scale_words,
                         np.random.default_rng(a.seed))
    print(f"{a.model} s{a.train_seed} | {cfg.label()} | golden {golden.accuracy:.4f} "
          f"| {len(words)} words", flush=True)

    rows, t0 = [], time.time()
    for w, (site, name, index) in enumerate(words):
        width = mx.layers[name].mx.fmt.width if site == "element" else 8
        for m in masks(width):
            with mx.word_fault(name, site, index, m):
                preds, finite = ev.predict(mx.module)
            hit = preds != golden.predictions
            rows.append({
                "word": w, "site": site, "tensor": name,
                "index": "/".join(map(str, index)), "mask": m,
                "nbits": bin(m).count("1"), "changed": int(hit.sum()),
                "nonfinite": not finite,
                "images": np.packbits(hit).tobytes().hex(),
            })
        if (w + 1) % 100 == 0:
            print(f"  {w + 1}/{len(words)} words, {len(rows):,} injections, "
                  f"{len(rows) / (time.time() - t0):.0f} inj/s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


# ------------------------------------------------------------------ analysis

def _bits(hexes: pd.Series) -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex("".join(hexes)), dtype=np.uint8)
    return np.unpackbits(raw.reshape(len(hexes), -1), axis=1)[:, :IMAGES].astype(bool)


def pairs(d: pd.DataFrame) -> pd.DataFrame:
    """One row per two-bit fault, beside its two constituent single-bit faults."""
    d = d.reset_index(drop=True)
    hit = _bits(d.images)
    single = {(int(r.word), int(r["mask"])): i for i, r in d[d.nbits == 1].iterrows()}
    out = []
    for i, r in d[d.nbits == 2].iterrows():
        m = int(r["mask"])
        lo = m & -m
        hi = m ^ lo
        a, b = single[(int(r.word), lo)], single[(int(r.word), hi)]
        out.append({
            "word": r.word, "site": r.site, "tensor": r.tensor,
            "bit_lo": lo.bit_length() - 1, "bit_hi": hi.bit_length() - 1,
            "r_pair": r.changed / IMAGES,
            "r_lo": d.changed[a] / IMAGES, "r_hi": d.changed[b] / IMAGES,
            "r_union": (hit[a] | hit[b]).sum() / IMAGES,
            "nf_pair": r.nonfinite, "nf_lo": d.nonfinite[a], "nf_hi": d.nonfinite[b],
            "same_set": bool((hit[i] == (hit[a] | hit[b])).all()),
        })
    p = pd.DataFrame(out)
    p["r_add"] = np.minimum(p.r_lo + p.r_hi, 1.0)
    p["adjacent"] = p.bit_hi - p.bit_lo == 1
    return p


def multibit_share(r1_sum: float, r2_sum: float, width: int, p: float) -> float:
    """Fraction of a word's expected damage carried by double flips at bit rate p.

    Per word, E[damage] = sum_b p(1-p)^(W-1) r_b + sum_{a<b} p^2 (1-p)^(W-2) r_ab
    + O(p^3); the triple-and-higher term is bounded by P(m>=3), reported apart.
    """
    one = p * (1 - p) ** (width - 1) * r1_sum
    two = p ** 2 * (1 - p) ** (width - 2) * r2_sum
    return two / max(one + two, 1e-300)


def analyse(a) -> None:
    files = sorted(RESULTS.glob("e21_*_multibit.csv*"))     # plain or gzipped
    if not files:
        raise SystemExit("no E21 campaigns found")
    summary = []
    for f in files:
        tag = f.name.split(".csv")[0].replace("e21_", "").replace("_multibit", "")
        d = pd.read_csv(f, dtype={"images": str})
        fmt = tag.split("_")[-1].split("-")[0]
        p = pairs(d)
        print(f"\n=== {tag}: {d.word.nunique()} words, {len(d):,} injections ===")
        for site, g in p.groupby("site"):
            s1 = d[(d.site == site) & (d.nbits == 1)]
            width = 8 if site == "scale" else int(round(len(s1) / s1.word.nunique()))
            r1 = s1.changed.mean() / IMAGES
            r2 = g.r_pair.mean()
            adj = g[g.adjacent].r_pair.mean()
            nw = s1.word.nunique()
            r1_sum = s1.changed.sum() / IMAGES / nw          # per-word sum over bits
            r2_sum = g.r_pair.sum() / nw                     # per-word sum over pairs
            emergent = ((g.r_pair > 0) & (g.r_lo == 0) & (g.r_hi == 0)).mean()
            nf_new = (g.nf_pair & ~g.nf_lo & ~g.nf_hi).mean()
            row = {
                "campaign": tag, "fmt": fmt, "site": site, "words": nw, "width": width,
                "r1": r1, "r2": r2, "r2_adjacent": adj, "ratio_2_to_1": r2 / max(r1, 1e-12),
                "ratio_adj_to_1": adj / max(r1, 1e-12),
                "union_mean": g.r_union.mean(), "additive_mean": g.r_add.mean(),
                "pair_over_union": r2 / max(g.r_union.mean(), 1e-12),
                "exact_set_match": g.same_set.mean(),
                "abs_err_union": (g.r_pair - g.r_union).abs().mean(),
                "abs_err_additive": (g.r_pair - g.r_add).abs().mean(),
                "pairs_emergent": emergent, "pairs_new_nonfinite": nf_new,
                "nonfinite_1": s1.nonfinite.mean(), "nonfinite_2": g.nf_pair.mean(),
                "pairs_below_max_single": (g.r_pair < np.maximum(g.r_lo, g.r_hi)).mean(),
            }
            for pr in (1e-6, 1e-4, 1e-3, 1e-2):
                row[f"share_p{pr:g}"] = multibit_share(r1_sum, r2_sum, width, pr)
            # bit rate at which double flips carry 1% / 10% of the expected damage
            k = r2_sum / max(r1_sum, 1e-300)
            row["p_at_1pct"] = 0.01 / 0.99 / max(k, 1e-300)
            row["p_at_10pct"] = 0.1 / 0.9 / max(k, 1e-300)
            summary.append(row)
            print(f"  {site:>7} (W={width}, {nw} words): r1 {r1:.5f}  r2 {r2:.5f} "
                  f"({r2 / max(r1, 1e-12):.2f}x)  adjacent {adj:.5f} "
                  f"({adj / max(r1, 1e-12):.2f}x)")
            print(f"          pair vs union of its singles: mean {r2:.5f} vs "
                  f"{g.r_union.mean():.5f}; identical image set in {g.same_set.mean():.1%}"
                  f"; |err| union {row['abs_err_union']:.5f}, additive "
                  f"{row['abs_err_additive']:.5f}")
            print(f"          emergent (both singles harmless): {emergent:.2%}; "
                  f"new non-finite: {nf_new:.2%}; non-finite 1-bit "
                  f"{row['nonfinite_1']:.3%} -> 2-bit {row['nonfinite_2']:.3%}")
            print(f"          double-flip share of damage at p=1e-4: "
                  f"{row['share_p0.0001']:.4%}; reaches 1% at p={row['p_at_1pct']:.2e}, "
                  f"10% at p={row['p_at_10pct']:.2e}")
    s = pd.DataFrame(summary)
    out = RESULTS / "e21_multibit_summary.csv"
    s.to_csv(out, index=False)
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="resnet8_w16")
    r.add_argument("--train-seed", type=int, default=0)
    r.add_argument("--fmt", default="e4m3")
    r.add_argument("--block-size", type=int, default=32)
    r.add_argument("--element-words", type=int, default=1500)
    r.add_argument("--scale-words", type=int, default=500)
    r.add_argument("--seed", type=int, default=21)
    r.add_argument("--device", default="cpu")
    sub.add_parser("analyse")
    a = p.parse_args()
    run(a) if a.cmd == "run" else analyse(a)


if __name__ == "__main__":
    main()
