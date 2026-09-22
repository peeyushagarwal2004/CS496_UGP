"""E09b -- if not size, then what? Which weights absorb the perturbation.

E09 showed `e2m3` fails 7x more often than `e3m2` at *matched* perturbation
size, so the cause is not how much a fault moves a weight. The remaining
candidate is which weight it moves.

The arithmetic gives a reason to expect exactly that. A flip multiplies a
value by a factor bounded by the format's exponent reach: `e3m2` has 3 exponent
bits, so a single flip can scale a value by up to 16x, while `e2m3` has 2 and
can manage at most 4x. To deliver a perturbation of a given absolute size,
`e3m2` can therefore use a *small* weight and blow it up, whereas `e2m3` must
hit a weight that was already large. If large weights matter more to the
output than small ones, the same absolute perturbation is worse in `e2m3`.

Two checks:

1. In matched perturbation bins, is the original weight larger under `e2m3`?
2. A neuron's output scale is set by its own row of weights, not by the layer
   average. Renormalising the perturbation per output neuron should therefore
   collapse the two P(SDC | perturbation) curves if this is the mechanism.

Run::

    PYTHONPATH=. python3 -m experiments.e09b_where_it_lands
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.formats import decode_e8m0, get_format
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"


def annotate(model: str, fmt: str, K: int, seeds) -> pd.DataFrame:
    f = get_format(fmt)
    table = f.table()
    frames = []
    for s in seeds:
        tag = model if s == 0 else f"{model}_s{s}"
        csv = RESULTS / f"e01_{tag}_{fmt}-K{K}-ocp-w-n3000.csv"
        if not csv.exists():
            continue
        net, _ = load_trained(model, seed=s)
        mx = MXModel(to_deploy(net), MXConfig(fmt=fmt, block_size=K)).quantize_weights()
        tensors = mx.weight_tensors()

        rms, row_rms = {}, {}
        for name, t in tensors.items():
            w = table[t.codes] * decode_e8m0(t.scales)[..., None]   # (out, nblk, K)
            valid = t.valid_mask()
            rms[name] = float(np.sqrt(np.mean(w[valid] ** 2)))
            per_row = []
            for r in range(w.shape[0]):                             # output neuron
                vals = w[r][valid[r]]
                per_row.append(float(np.sqrt(np.mean(vals ** 2))) if vals.size else np.nan)
            row_rms[name] = np.array(per_row)

        d = pd.read_csv(csv)
        d = d[d.site == "element"].copy()
        dl, dr, vm = [], [], []
        for name, idx_s, bit in zip(d.tensor, d["index"], d.bit):
            t = tensors[name]
            idx = ast.literal_eval(idx_s)
            c = int(t.codes[idx])
            scale = float(decode_e8m0(t.scales[idx[:-1]]))
            delta = abs(float(table[c ^ (1 << bit)]) - float(table[c])) * scale
            v = abs(float(table[c])) * scale
            dl.append(delta / max(rms[name], 1e-30))
            dr.append(delta / max(row_rms[name][idx[0]], 1e-30))
            vm.append(v / max(rms[name], 1e-30))
        d["delta_rms"], d["delta_row"], d["w_rms"] = dl, dr, vm
        d["seed"] = s
        frames.append(d)
        mx.restore_weights()
    return pd.concat(frames, ignore_index=True)


def curve(data, col, formats, label):
    allx = np.concatenate([d[col].to_numpy() for d in data.values()])
    edges = np.unique(np.quantile(allx[allx > 0], np.linspace(0, 1, 8)))
    print(f"\n--- P(SDC | {label}), shared bins ---")
    print(f"{'bin':>22}  " + "  ".join(f"{f:>18}" for f in formats))
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        cells, rates = [], {}
        for f in formats:
            d = data[f]
            m = (d[col] > lo) & (d[col] <= hi)
            rates[f] = d[m].sdc.mean() if m.sum() else np.nan
            cells.append(f"{rates[f]:.4f} (n={int(m.sum())})" if m.sum() else "      -       ")
        print(f"{lo:>10.3f} - {hi:<9.3f}  " + "  ".join(f"{c:>18}" for c in cells))
        a, b = rates[formats[0]], rates[formats[1]]
        if a and a > 0 and np.isfinite(b):
            ratios.append(b / a)
    if ratios:
        print(f"  mean ratio across bins: {np.mean(ratios):.1f}x")
    return np.mean(ratios) if ratios else np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_small")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--formats", nargs=2, default=["e3m2", "e2m3"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    a = p.parse_args()

    data = {f: annotate(a.model, f, a.block_size, a.seeds) for f in a.formats}
    f1, f2 = a.formats

    print(f"model {a.model}, element faults, {len(data[f1])} per format\n")
    print("--- check 1: in matched perturbation bins, which weights get hit? ---")
    allx = np.concatenate([d.delta_rms.to_numpy() for d in data.values()])
    edges = np.unique(np.quantile(allx[allx > 0], np.linspace(0.2, 1, 6)))
    print(f"{'|dw|/rms bin':>22}  {'mean |w|/rms ' + f1:>22}  {'mean |w|/rms ' + f2:>22}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        cells = []
        for f in a.formats:
            d = data[f]
            m = (d.delta_rms > lo) & (d.delta_rms <= hi)
            cells.append(f"{d[m].w_rms.mean():.3f} (n={int(m.sum())})" if m.sum() else "   -   ")
        print(f"{lo:>10.3f} - {hi:<9.3f}  " + "  ".join(f"{c:>22}" for c in cells))

    r_layer = curve(data, "delta_rms", a.formats, "perturbation / layer RMS")
    r_row = curve(data, "delta_row", a.formats, "perturbation / output-neuron RMS")

    print(f"\n  gap at matched size, layer-normalised:  {r_layer:.1f}x")
    print(f"  gap at matched size, neuron-normalised: {r_row:.1f}x")
    if np.isfinite(r_row) and np.isfinite(r_layer) and r_row < r_layer / 2:
        print("  -> renormalising per output neuron explains much of the gap")
    else:
        print("  -> neuron normalisation does not explain it either")

    out = RESULTS / f"e09b_{a.model}_where_it_lands.csv"
    pd.concat([d.assign(fmt=f) for f, d in data.items()]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
