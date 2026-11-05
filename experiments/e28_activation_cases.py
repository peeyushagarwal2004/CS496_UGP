"""E28 -- the flip-case tree on activations, and how re-quantisation spreads an error.

Activations differ from weights in ways that change which flip cases occur:
after a ReLU they are never negative and often exactly zero, their scales are
recomputed for every input, and in an MX pipeline every layer's input is
re-quantised, so an error that survives one layer is encoded again into a block
with 31 neighbours. Three questions, answered by exact counting and by the codec
alone (no network injection):

1. case mix   which cases an activation bit flip falls into, against weights; how
              many activation values are zero; what an FNUZ-style format (NaN at the
              negative-zero pattern) would do to them
2. scale band how often a clean activation scale falls outside a band calibrated on
              other images (false alarms of a plausibility check), and how many scale
              flips leave it (coverage)
3. requant    when one value of a block grows by r before re-quantisation, how many
              of its 31 neighbours are flushed to zero or lose precision, per format

Run::

    PYTHONPATH=. python -m experiments.e28_activation_cases
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mxfi.cases import element_cases, scale_cases, value_class
from mxfi.codec import quantize
from mxfi.data import campaign_subset
from mxfi.formats import ELEMENT_FORMATS, get_format
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = ["resnet8", "repvgg_a0"]
FORMATS = list(ELEMENT_FORMATS)
GROWTH = [2, 4, 16, 256, 65536]


def activations(model: str, fmt: str, images: torch.Tensor):
    net, _ = load_trained(model)
    mx = MXModel(to_deploy(net), MXConfig(fmt=fmt, block_size=32,
                                          quantise_activations=True))
    return mx.activation_space(images), mx


def case_mix(tensors) -> dict:
    """Exact shares of (code, bit) element flips and (byte, bit) scale flips by case."""
    el, sc, n_vals, n_zero = {}, {}, 0, 0
    for t in tensors.values():
        f = t.fmt
        tab = element_cases(f)
        codes = t.codes[t.valid_mask()].astype(np.int64)
        n_vals += codes.size
        n_zero += int((value_class(f)[codes] == "zero").sum())
        for b in range(f.width):
            for k, v in pd.Series(tab["case"][codes, b]).value_counts().items():
                el[k] = el.get(k, 0) + v
        vals = np.abs(f.table()[t.codes])
        bmax = vals.max(-1).ravel()
        zero = (vals == 0).all(-1).ravel()
        x = t.scales.ravel()
        for b in range(8):
            c = scale_cases(x, np.full(x.size, b), bmax, zero)["case"]
            for k, v in pd.Series(c).value_counts().items():
                sc[k] = sc.get(k, 0) + v
    return {"element": el, "scale": sc, "zero_share": n_zero / n_vals}


def scale_band(model: str, fmt: str, imgs: torch.Tensor) -> dict:
    """Band from the first half of the images, tested on the second half."""
    half = len(imgs) // 2
    cal, _ = activations(model, fmt, imgs[:half])
    test, _ = activations(model, fmt, imgs[half:])
    clean_out = n_blocks = 0
    flips_out = np.zeros(8)
    for name, t in test.items():
        lo, hi = int(cal[name].scales.min()), int(cal[name].scales.max())
        x = t.scales.ravel().astype(np.int64)
        clean_out += int(((x < lo) | (x > hi)).sum())
        n_blocks += x.size
        for b in range(8):
            xn = x ^ (1 << b)
            flips_out[b] += ((xn < lo) | (xn > hi)).sum()
    return {"false_alarm": clean_out / n_blocks,
            **{f"bit{b}_flagged": flips_out[b] / n_blocks for b in range(8)}}


def requant(tensors, fmt: str, rng: np.random.Generator, n_blocks: int = 20000) -> list[dict]:
    """Grow one value of real activation blocks by r, re-quantise, inspect the neighbours."""
    f = get_format(fmt)
    blocks = []
    for t in tensors.values():
        v = f.table()[t.codes].astype(np.float64) * \
            np.exp2(t.scales.astype(np.float64) - 127)[..., None]
        v = v.reshape(-1, t.block_size)
        blocks.append(v[(v != 0).sum(1) >= 2])
    v = np.concatenate(blocks)
    v = v[rng.choice(len(v), min(n_blocks, len(v)), replace=False)]
    rows = []
    for which in ["block maximum", "random non-zero value"]:
        if which == "block maximum":
            idx = np.abs(v).argmax(1)
        else:
            idx = np.array([rng.choice(np.flatnonzero(r)) for r in v])
        others = np.ones_like(v, bool)
        others[np.arange(len(v)), idx] = False
        nz = others & (v != 0)
        for r in GROWTH:
            w = v.copy()
            w[np.arange(len(v)), idx] *= r
            q = quantize(w.astype(np.float32), fmt, v.shape[1]).decode().astype(np.float64)
            flushed = nz & (q == 0)
            rel = np.abs(q - v)[nz] / np.abs(v)[nz]
            rows.append({"fmt": fmt, "grown value": which, "growth": r,
                         "neighbours flushed to 0": flushed.sum() / nz.sum(),
                         "blocks losing any neighbour": flushed.any(1).mean(),
                         "neighbour mean rel. error": rel.mean()})
    return rows


def main() -> None:
    rng = np.random.default_rng(28)
    ds = campaign_subset(n_per_class=2, download=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    out = []

    rows, scale_rows, band_rows, crush = [], [], [], []
    for model in MODELS:
        for fmt in FORMATS:
            tensors, mx = activations(model, fmt, imgs)
            m = case_mix(tensors)
            el, sc = m["element"], m["scale"]
            n_el, n_sc = sum(el.values()), sum(sc.values())
            special = sum(v for k, v in el.items() if k.endswith("to_special"))
            masked = el.get("sign:sign:masked", 0)
            zero_tr = sum(v for k, v in el.items() if "from_zero" in k)
            rows.append({"model": model, "fmt": fmt, "values that are 0": m["zero_share"],
                         "E0 masked": masked / n_el, "from_zero": zero_tr / n_el,
                         "E1 to NaN/Inf": special / n_el,
                         "FNUZ-style E1 (+0 sign flip)": (m["zero_share"] / f_width(fmt)
                                                          if fmt in ("e4m3", "e5m2") else np.nan)})
            scale_rows.append({"model": model, "fmt": fmt,
                               **{k: v / n_sc for k, v in sorted(sc.items())}})
            if fmt in ("e4m3", "e3m2"):
                band_rows.append({"model": model, "fmt": fmt, **scale_band(model, fmt, imgs)})
            if model == "resnet8":
                crush += requant(tensors, fmt, rng)
            mx.remove_hooks()

    fl = lambda v: f"{v:.4f}"
    out.append("### 1. element case mix of activation flips\n" +
               pd.DataFrame(rows).to_string(index=False, float_format=fl))
    out.append("### scale case mix of activation flips\n" +
               pd.DataFrame(scale_rows).fillna(0).to_string(index=False, float_format=fl))
    out.append("### 2. activation scale band: calibrated on 10 images, tested on 10 others\n" +
               pd.DataFrame(band_rows).to_string(index=False, float_format=fl))
    out.append("### 3. re-quantisation: one value of a real activation block grows by r\n" +
               pd.DataFrame(crush).to_string(index=False, float_format=fl))
    text = "\n\n".join(out)
    path = RESULTS / "e28_activation_cases.txt"
    path.write_text(text)
    print(text)
    print(f"\nwritten {path}")


def f_width(fmt: str) -> int:
    return get_format(fmt).width


if __name__ == "__main__":
    main()
