"""E27 -- first-principles extensions of the flip-case tree, from the exhaustive data.

The case tree of `mxfi.cases` (E25) sorts a flip by what it does to the stored
number. A fault's damage also depends on what happens before that (which way
the bit flips, and whether the fault is a transient upset or a stuck bit), after
it (what the computation does with the changed value), and on whether anything
notices. Everything here is computed from the two exhaustive ResNet8 campaigns
plus one fault-free forward pass; no new injection is needed.

1. direction   damage from 0->1 against 1->0 flips; stuck-at-1 and stuck-at-0 faults
2. outcome     no prediction changed / silent corruption / non-finite output
3. checks      free detectors from invariants of a clean MX tensor, and the guards
               that correct what they detect
4. prediction  held-out variance of per-fault damage explained by the tree, and by
               each added first-principles factor
5. spread      how much damage varies inside each case
6. sign        whether an error that raises a pre-activation is worse than one
               that lowers it by the same amount (ReLU asymmetry)
7. zero input  weights whose input channel is always zero: finite errors vanish,
               NaN does not (NaN * 0 = NaN)

Run (after E25)::

    PYTHONPATH=. python -m experiments.e27_fault_model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mxfi.campaign import Evaluator
from mxfi.cases import classify_elements, classify_scales
from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

from .e25_flip_cases import CAMPAIGNS

RESULTS = Path(__file__).resolve().parent.parent / "results"
KEY = ["tensor", "row", "blk", "lane"]


# ------------------------------------------------------------------ context

def input_activity(mx: MXModel, ev: Evaluator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per quantised layer: mean |input| and max |input| of every input channel."""
    stats, hooks = {}, []

    def make(name):
        def hook(_m, inputs, _out):
            x = inputs[0].detach().abs()
            dims = (0, 2, 3) if x.dim() == 4 else (0,)
            s = stats.setdefault(name, [0.0, 0.0, 0])
            s[0] = s[0] + x.sum(dims).double()
            s[1] = torch.maximum(torch.as_tensor(s[1]), x.amax(dims).double())
            s[2] += x.numel() // x.shape[1]
        return hook

    for name, lay in mx.layers.items():
        hooks.append(lay.module.register_forward_hook(make(name)))
    try:
        ev.predict(mx.module)
    finally:
        for h in hooks:
            h.remove()
    return {k: ((s[0] / s[2]).numpy(), s[1].numpy()) for k, s in stats.items()}


def augment(fmt: str) -> pd.DataFrame:
    """The E25 labelled campaign plus block, input and invariant context per fault."""
    _, ckpt = CAMPAIGNS[fmt]
    d = pd.read_csv(RESULTS / f"e25_{fmt}_labelled.csv.gz", low_memory=False)
    net, _ = load_trained("resnet8_w16", path=ckpt)
    mx = MXModel(to_deploy(net), MXConfig(fmt=fmt, block_size=32)).quantize_weights()
    act = input_activity(mx, Evaluator(campaign_subset(n_per_class=20, download=False)))

    parts = []
    for (name, site), g in d.groupby(["tensor", "site"], sort=False):
        lay = mx.layers[name]
        t, f = lay.mx, lay.mx.fmt
        K = t.block_size
        taps = int(np.prod(lay.torch_shape[2:])) if len(lay.torch_shape) == 4 else 1
        red = int(np.prod(lay.torch_shape[1:]))
        mean_in, max_in = act[name]

        # per-lane context: decoded weight and the activity of the input it reads
        col = np.arange(t.codes.shape[1] * K).reshape(t.codes.shape[1], K)
        cin = np.minimum(col, red - 1) // taps
        lane_act = mean_in[cin]                                   # (nblk, K)
        lane_dead = (max_in[cin] == 0) | (col >= red)
        vals = np.abs(f.table()[t.codes].astype(np.float64)) * \
            np.exp2(t.scales.astype(np.float64) - 127)[..., None]
        vals[:, col >= red] = 0
        block_l1 = vals.sum(-1)
        block_out = (vals * lane_act[None]).sum(-1)               # |partial sum| proxy

        r, b = g["row"].to_numpy(), g["blk"].to_numpy()
        bit = g["bit"].to_numpy()
        g = g.assign(block_out=block_out[r, b], block_l1=block_l1[r, b],
                     block_dead=lane_dead[b].all(-1))
        if site == "element":
            ln = g["lane"].to_numpy().astype(int)
            c = classify_elements(t, (r, b, ln), bit)
            with np.errstate(invalid="ignore"):
                dsign = np.sign(c["after"] - c["before"])
            g = g.assign(x=t.scales[r, b].astype(np.int64), bit_was=c["bit_was"],
                         dsign=np.nan_to_num(dsign), abs_error=c["abs_error"],
                         out_error=c["abs_error"] * lane_act[b, ln],
                         dead_input=lane_dead[b, ln], demote=c["demotes_max"])
        else:
            c = classify_scales(t, (r, b), bit)
            g = g.assign(x=c["x"], x_new=c["x_new"], bit_was=c["bit_was"], dsign=0.0,
                         out_error=block_out[r, b], dead_input=lane_dead[b].all(-1),
                         demote=False, above_band=c["above_band"],
                         out_of_band=c["out_of_band"])
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    truth = {True: True, False: False, "True": True, "False": False}
    for c in ["near_special", "is_block_max", "demote", "dead_input", "nonfinite", "sdc",
              "above_band", "out_of_band"]:
        d[c] = d[c].map(truth).fillna(False).astype(bool)
    return d


# ---------------------------------------------------------------- analyses

def direction(d: pd.DataFrame) -> str:
    N, tot = len(d), d.damage.sum()
    rows = []
    for site in ["element", "scale", "all"]:
        s = d if site == "all" else d[d.site == site]
        r01 = s.loc[s.bit_was == 0, "damage"].sum() / N
        r10 = s.loc[s.bit_was == 1, "damage"].sum() / N
        rows.append({"site": site, "0->1 rate (= stuck-at-1)": r01,
                     "1->0 rate (= stuck-at-0)": r10,
                     "share of damage from 0->1": r01 / max(r01 + r10, 1e-12),
                     "bits holding 0": (s.bit_was == 0).mean()})
    cat = d[d.case.isin(["scale:up_overflow", "scale:to_nan"]) | d.case.str.endswith("to_special")]
    out = pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5f}")
    return (out + f"\ncatastrophic cases that are 0->1 flips: "
            f"{(cat.bit_was == 0).mean():.3f} of {len(cat):,} faults")


def outcomes(d: pd.DataFrame) -> str:
    cls = np.select([d.damage == 0, d.nonfinite], ["no prediction changed", "non-finite output"],
                    "silent corruption")
    g = d.assign(outcome=cls).groupby("outcome").agg(n=("damage", "size"), dmg=("damage", "sum"))
    g["fault_share"] = g.n / len(d)
    g["damage_share"] = g.dmg / d.damage.sum()
    g["rate"] = g.dmg / len(d)
    return g.drop(columns="dmg").to_string(float_format=lambda v: f"{v:.5f}")


def checks(d: pd.DataFrame, guard: pd.DataFrame | None) -> str:
    sc, el = d.site == "scale", d.site == "element"
    flags = {
        "D1 special element code on read": el & d.case.str.endswith("to_special"),
        "D2a scale bit 7 set (all clean scales < 128)": sc & (d.x_new >= 128),
        "D2b scale outside its tensor's clean band": sc & d.out_of_band,
        "D2c scale above its tensor's clean band": sc & d.above_band,
        "D3 no top-binade element left in the block": el & d.demote,
        "D4 non-finite output": d.nonfinite,
    }
    tot = d.damage.sum()
    rows = [{"check": k, "faults flagged": m.mean(), "damage flagged": d.damage[m].sum() / tot}
            for k, m in flags.items()]
    mem = flags["D1 special element code on read"] | flags["D2b scale outside its tensor's clean band"] | \
        flags["D3 no top-binade element left in the block"]
    rows.append({"check": "D1 + D2b + D3 (at read)", "faults flagged": mem.mean(),
                 "damage flagged": d.damage[mem].sum() / tot})
    allc = mem | flags["D4 non-finite output"]
    rows.append({"check": "all of the above", "faults flagged": allc.mean(),
                 "damage flagged": d.damage[allc].sum() / tot})
    out = [pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5f}")]
    out.append(f"silent damage left after every check: rate {d.damage[~allc].sum() / len(d):.6f} "
               f"(of {d.damage.mean():.6f})")

    # guards: correct what D1 and D2c detect
    erase = d[sc & (d.bit == 6)].set_index(["tensor", "row", "blk"]).damage   # x/2^64 = erased
    signflip = d[el & (d.field == "sign")].set_index(KEY).damage
    y = d.damage.copy()
    none = pd.Series(False, index=d.index)
    up = flags["D2c scale above its tensor's clean band"]
    y_g2 = y.copy()
    y_g2[up] = erase.reindex(pd.MultiIndex.from_frame(d.loc[up, ["tensor", "row", "blk"]])).to_numpy()
    e1 = flags["D1 special element code on read"]
    y_g1_bound = y.copy()
    if e1.any():
        y_g1_bound[e1] = signflip.reindex(pd.MultiIndex.from_frame(
            d.loc[e1, KEY].astype({"lane": int}))).to_numpy()
    rows = [("none", y, none), ("G2 scale above band -> erase block", y_g2, up)]
    if e1.any():
        rows.append(("G1 special code -> 0 (bound: sign-flip damage)", y_g1_bound, e1))
        y12 = y_g2.copy(); y12[e1] = y_g1_bound[e1]
        rows.append(("G1 + G2 (bound)", y12, e1 | up))
        if guard is not None and len(guard):
            m = guard.damage.mean()
            y1m = y.copy(); y1m[e1] = m
            rows.append((f"G1 measured (E26 sample mean {m:.4f})", y1m, e1))
            y12m = y_g2.copy(); y12m[e1] = m
            rows.append(("G1 measured + G2", y12m, e1 | up))
    out.append("\nguards: per-inference rate, and its silent part (outputs stay finite, "
               "so no output check sees it)")
    for name, yy, fixed in rows:
        assert not yy.isna().any(), name
        silent = yy[~d.nonfinite | fixed].sum() / len(d)
        out.append(f"  {name:50s} rate {yy.mean():.6f} ({d.damage.mean() / yy.mean():6.2f}x lower)"
                   f"   silent {silent:.6f}")
    return "\n".join(out)


def magnitude_bucket(v: pd.Series) -> pd.Series:
    b = np.floor(np.log2(v.where(v > 0)))
    b = b.where(np.isfinite(v), 99)           # non-finite: its own bucket
    return b.fillna(-99).astype(int)          # exactly zero


def heldout_r2(d: pd.DataFrame, levels: list[list[str]], seed: int = 0) -> tuple[float, float]:
    """R^2 of group-mean prediction on held-out faults, finest available level wins."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(d))
    y = d.damage.to_numpy()
    r2_all, r2_fin = [], []
    fin = ((d.site == "element") & ~d.case.str.endswith("to_special")).to_numpy()
    for k in (0, 1):
        tr, te = fold != k, fold == k
        pred = np.full(te.sum(), y[tr].mean())
        dte = d[te]
        for cols in levels:
            m = d[tr].groupby(cols).damage.mean()
            hit = m.reindex(pd.MultiIndex.from_frame(dte[cols]) if len(cols) > 1 else dte[cols[0]])
            v = hit.to_numpy()
            pred = np.where(np.isnan(v), pred, v)
        yt = y[te]
        r2_all.append(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        f = fin[te]
        r2_fin.append(1 - ((yt[f] - pred[f]) ** 2).sum() / ((yt[f] - yt[f].mean()) ** 2).sum())
    return float(np.mean(r2_all)), float(np.mean(r2_fin))


def prediction(d: pd.DataFrame) -> str:
    d = d.assign(mag_abs=magnitude_bucket(d.abs_error.where(d.site == "element", d.block_l1)),
                 mag_out=magnitude_bucket(d.out_error))
    ladder = [
        ("site only", [["site"]]),
        ("+ case (the E25 tree)", [["site"], ["case"]]),
        ("+ bit position", [["site"], ["case"], ["case", "bit"]]),
        ("+ layer", [["site"], ["case"], ["case", "bit"], ["case", "bit", "tensor"]]),
        ("+ size of the weight error", [["site"], ["case"], ["case", "bit"],
                                        ["case", "bit", "tensor"],
                                        ["case", "bit", "tensor", "mag_abs"]]),
        ("  (size x input activity instead)", [["site"], ["case"], ["case", "bit"],
                                               ["case", "bit", "tensor"],
                                               ["case", "bit", "tensor", "mag_out"]]),
        ("+ sign of the error", [["site"], ["case"], ["case", "bit"],
                                 ["case", "bit", "tensor"],
                                 ["case", "bit", "tensor", "mag_out"],
                                 ["case", "bit", "tensor", "mag_out", "dsign"]]),
    ]
    rows = []
    for name, levels in ladder:
        a, f = heldout_r2(d, levels)
        rows.append({"model": name, "R2 all faults": a, "R2 finite element faults": f})
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}")


def spread(d: pd.DataFrame) -> str:
    g = d.groupby("case").damage
    t = pd.DataFrame({"n": g.size(), "mean": g.mean(), "zero": g.apply(lambda s: (s == 0).mean()),
                      "p50": g.median(), "p90": g.quantile(0.9), "max": g.max()})
    return t.sort_values("mean", ascending=False).to_string(float_format=lambda v: f"{v:.4f}")


def sign_asymmetry(d: pd.DataFrame) -> str:
    e = d[(d.site == "element") & np.isfinite(d.out_error) & (d.out_error > 0) & (d.dsign != 0)]
    e = e.assign(kind=np.select([e.tensor == "conv1", e.tensor == "fc"],
                                ["stem (signed input)", "fc (logits)"], "hidden conv (input >= 0)"),
                 mag=magnitude_bucket(e.out_error))
    out = []
    for kind, g in e.groupby("kind"):
        t = g.pivot_table(index="mag", columns="dsign", values="damage", aggfunc="mean")
        n = g.pivot_table(index="mag", columns="dsign", values="damage", aggfunc="size")
        t.columns = ["error < 0", "error > 0"]
        t["ratio +/-"] = t["error > 0"] / t["error < 0"]
        t["n"] = n.sum(axis=1)
        t = t[t.n >= 200]
        # damage-weighted summary over the buckets where errors matter
        big = g[g.mag >= g.mag.quantile(0.5)]
        pos, neg = big[big.dsign > 0].damage.mean(), big[big.dsign < 0].damage.mean()
        out.append(f"{kind}: upper half of error sizes, error>0 {pos:.5f} vs error<0 {neg:.5f} "
                   f"({pos / neg:.2f}x)\n" + t.to_string(float_format=lambda v: f"{v:.5f}"))
    return "\n\n".join(out)


def zero_input(d: pd.DataFrame) -> str:
    e = d[d.site == "element"]
    dead = e[e.dead_input]
    out = [f"weights whose input channel is zero on all 200 images: "
           f"{dead.drop_duplicates(KEY).shape[0]:,} of {e.drop_duplicates(KEY).shape[0]:,}"]
    if len(dead):
        fin = dead[~dead.case.str.endswith("to_special")]
        nan = dead[dead.case.str.endswith("to_special")]
        out.append(f"  finite flips on them: {len(fin):,}, mean damage {fin.damage.mean():.6f}, "
                   f"any damage {(fin.damage > 0).mean():.4f}")
        if len(nan):
            out.append(f"  NaN flips on them:    {len(nan):,}, mean damage {nan.damage.mean():.4f}")
    s = d[(d.site == "scale") & d.block_dead]
    if len(s):
        out.append(f"scale flips of blocks that read only dead inputs: {len(s):,}, "
                   f"mean damage {s.damage.mean():.6f} (non-finite {s.nonfinite.mean():.3f})")
    return "\n".join(out)


def protection(d: pd.DataFrame) -> str:
    """Bit positions ordered by damage per stored bit: what to protect first."""
    out = []
    for keys in (["site", "bit"], ["site", "bit", "tensor"]):
        g = d.groupby(keys).agg(n=("damage", "size"), dmg=("damage", "sum"))
        g["storage"] = g.n / len(d)
        g["damage"] = g.dmg / d.damage.sum()
        g = g.assign(per_bit=g.dmg / g.n).sort_values("per_bit", ascending=False)
        g["cum storage"] = g.storage.cumsum()
        g["cum damage"] = g.damage.cumsum()
        marks = [g[g["cum damage"] >= q].iloc[0]["cum storage"] for q in (0.5, 0.8, 0.9, 0.95)]
        out.append(f"by {' x '.join(keys)}: storage to protect for 50/80/90/95% of the damage: "
                   + ", ".join(f"{100 * m:.2f}%" for m in marks))
        if keys == ["site", "bit"]:
            out.append(g[["storage", "damage", "cum storage", "cum damage"]].head(8)
                       .to_string(float_format=lambda v: f"{v:.4f}"))
    return "\n".join(out)


def main() -> None:
    text = []
    for fmt in CAMPAIGNS:
        d = augment(fmt)
        gpath = RESULTS / f"e26_{fmt}_datapath.csv"
        guard = None
        if gpath.exists():
            gd = pd.read_csv(gpath)
            guard = gd[gd.variant == "guard"]
        sections = [
            ("1. fault direction and stuck-at faults", direction(d)),
            ("2. outcome classes", outcomes(d)),
            ("3. free checks and guards", checks(d, guard)),
            ("4. held-out variance explained", prediction(d)),
            ("5. spread of damage inside each case", spread(d)),
            ("6. sign of the error (ReLU asymmetry), finite element flips", sign_asymmetry(d)),
            ("7. zero-input masking", zero_input(d)),
            ("8. protection order", protection(d)),
        ]
        text.append(f"## {fmt}: {len(d):,} faults, rate {d.damage.mean():.5f}\n\n" +
                    "\n\n".join(f"### {h}\n{body}" for h, body in sections))
    out = RESULTS / "e27_fault_model.txt"
    out.write_text("\n\n\n".join(text))
    print("\n\n\n".join(text))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
