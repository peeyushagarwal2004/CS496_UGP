"""E25 -- every single-bit flip, sorted into its case, with measured damage.

`mxfi.cases` puts every possible flip into one case of a decision tree: for a
scale flip, the direction and size of the exponent move and whether it crosses
a boundary (NaN code, float32 overflow, all-zero block); for an element flip,
the field the bit belongs to, the direction, the value class of the code and
the transition it causes. This joins that taxonomy with the two exhaustive
ResNet8 campaigns (every stored bit injected) so each case gets its measured
per-inference damage, not an estimate.

Each exhaustive campaign was recorded on a cluster checkpoint, so the matching
checkpoint is loaded from `cluster/checkpoints` and checked by replay first.

Run::

    PYTHONPATH=. python -m experiments.e25_flip_cases
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.cases import classify_elements, classify_scales, element_cases
from mxfi.formats import get_format
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
N_IMAGES = 200

# exhaustive campaign -> the checkpoint it was recorded on (see cluster/README.md)
CAMPAIGNS = {
    "e3m2": (ROOT / "cluster/results/e06_resnet8_w16_e3m2-K32-ocp-w_exhaustive.csv",
             ROOT / "cluster/checkpoints/resnet8_w16_cifar10.pt"),
    "e4m3": (ROOT / "cluster/results/e06_resnet8_w16_e4m3-K32-ocp-w_exhaustive.csv",
             ROOT / "cluster/checkpoints/resnet8_w16_s9_cifar10.pt"),
}


def label(fmt: str, csv: Path, ckpt: Path) -> pd.DataFrame:
    """The recorded campaign with every row's case attached."""
    net, _ = load_trained("resnet8_w16", path=ckpt)
    mx = MXModel(to_deploy(net), MXConfig(fmt=fmt, block_size=32)).quantize_weights()
    tensors = mx.weight_tensors()

    d = pd.read_csv(csv)
    d["damage"] = d["changed"] / N_IMAGES
    parts = []
    for (name, site), g in d.groupby(["tensor", "site"], sort=False):
        t = tensors[name]
        bit = g["bit"].to_numpy()
        if site == "element":
            idx = (g["row"].to_numpy(), g["blk"].to_numpy(), g["lane"].to_numpy())
            c = classify_elements(t, idx, bit)
            g = g.assign(case=c["case"], field=c["field"], direction=c["direction"],
                         from_class=c["from_class"], rel_error=c["rel_error"],
                         near_special=c["near_special"], is_block_max=c["is_block_max"],
                         code=c["code"])
        else:
            idx = (g["row"].to_numpy(), g["blk"].to_numpy())
            c = classify_scales(t, idx, bit)
            g = g.assign(case="scale:" + c["case"], field="scale",
                         direction=np.where(c["delta"] > 0, "up", "down"),
                         delta=c["delta"], x=c["x"])
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def table(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    total = d["damage"].sum()
    g = d.groupby(by, dropna=False).agg(
        n=("damage", "size"), mean_damage=("damage", "mean"),
        nonfinite=("nonfinite", "mean"), damage_sum=("damage", "sum"))
    g["fault_share"] = g["n"] / len(d)
    g["damage_share"] = g["damage_sum"] / total
    return g.drop(columns="damage_sum").sort_values("damage_share", ascending=False)


def report(fmt: str, d: pd.DataFrame) -> str:
    f = get_format(fmt)
    el, sc = d[d.site == "element"], d[d.site == "scale"]
    out = [f"## {fmt}: {len(d):,} faults, per-inference rate {d.damage.mean():.5f}",
           f"element faults {len(el):,} (mean {el.damage.mean():.5f}), "
           f"scale faults {len(sc):,} (mean {sc.damage.mean():.5f})", ""]

    out += ["### all cases", table(d, ["case"]).to_string(), ""]
    out += ["### scale flips by bit and direction",
            table(sc, ["bit", "direction"]).sort_index().to_string(), ""]
    xs = sc.drop_duplicates(["tensor", "row", "blk"])["x"]
    out += ["### scale bytes in the clean model",
            f"range {xs.min()}..{xs.max()}, median {int(xs.median())}; "
            f"bytes with bit 7 set: {(xs >= 128).mean():.3f}",
            xs.value_counts().sort_index().to_string(), ""]

    out += ["### element flips by field and value class of the original code",
            table(el, ["field", "from_class"]).sort_index().to_string(), ""]
    out += ["### element flips by bit",
            table(el, ["bit", "field"]).sort_index().to_string(), ""]
    out += ["### element flips: is the flipped element its block's maximum?",
            table(el, ["is_block_max"]).to_string(), ""]

    codes = el.drop_duplicates(["tensor", "row", "blk", "lane"])["code"].astype(int)
    mag = pd.Series(f.table()[codes & ((1 << f.sign_bit) - 1)], name="|value|")
    out += ["### element magnitudes in the clean model (unit scale, share of weights)",
            (mag.value_counts(normalize=True).sort_index().map("{:.4f}".format)
             .to_string()), ""]

    fin = el[np.isfinite(el.rel_error) & (el.rel_error > 0)]
    bins = [0, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1, 2, 4]
    out += ["### finite element flips by error size (in units of the top binade)",
            table(fin.assign(err=pd.cut(fin.rel_error, bins)), ["err"])
            .sort_index().to_string(), ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fmt", nargs="*", default=list(CAMPAIGNS))
    a = p.parse_args()

    text = []
    for fmt in a.fmt:
        csv, ckpt = CAMPAIGNS[fmt]
        d = label(fmt, csv, ckpt)
        d.to_csv(RESULTS / f"e25_{fmt}_labelled.csv.gz", index=False)
        text.append(report(fmt, d))
    out = RESULTS / "e25_flip_cases.txt"
    out.write_text("\n\n".join(text))
    print("\n\n".join(text))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
