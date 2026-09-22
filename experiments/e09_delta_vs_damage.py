"""E09 -- does a perturbation of a given size do the same damage in two formats?

E08 ruled out the obvious explanation for F34: `e2m3` perturbs weights slightly
*less* than `e3m2` (0.91x mean, 0.92x in the tail) yet fails 5.8x more often.
Two possibilities remain, and they are distinguishable:

1. **Different perturbation distributions.** E08 averaged over the whole code
   space; the campaigns sampled actual faults. If the sampled perturbations
   differ from the enumerated ones, magnitude may still be the explanation.
2. **Different damage per unit perturbation.** If both formats deliver the same
   distribution of |dw| but one converts it into failures more often, then the
   cause is not the perturbation at all but where it lands -- which weights,
   and how the network uses them.

This recomputes the exact perturbation for every fault that was actually
injected, then compares P(SDC | perturbation size) between the two formats. If
those curves coincide, (1) holds and the marginal distributions explain the
gap. If they separate, (2) holds.

Run::

    PYTHONPATH=. python3 -m experiments.e09_delta_vs_damage
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
    """Attach the exact |dw| (in layer-RMS units) to every injected element fault."""
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

        tensors, rms = mx.weight_tensors(), {}
        for name, t in tensors.items():
            w = table[t.codes] * decode_e8m0(t.scales)[..., None]
            rms[name] = float(np.sqrt(np.mean(w[t.valid_mask()] ** 2)))

        d = pd.read_csv(csv)
        d = d[d.site == "element"].copy()
        deltas = []
        for name, idx_s, bit in zip(d.tensor, d["index"], d.bit):
            t = tensors[name]
            idx = ast.literal_eval(idx_s)
            c = int(t.codes[idx])
            scale = float(decode_e8m0(t.scales[idx[:-1]]))
            deltas.append(abs(float(table[c ^ (1 << bit)]) - float(table[c]))
                          * scale / max(rms[name], 1e-30))
        d["delta_rms"] = deltas
        d["seed"] = s
        frames.append(d)
        mx.restore_weights()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_small")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--formats", nargs=2, default=["e3m2", "e2m3"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    a = p.parse_args()

    data = {f: annotate(a.model, f, a.block_size, a.seeds) for f in a.formats}

    print(f"model {a.model}, element faults only, |dw| in units of layer weight RMS\n")
    print("--- marginal perturbation distribution of the faults actually injected ---")
    print(f"{'fmt':>6} {'n':>6} {'mean':>8} {'median':>8} {'p90':>8} {'P(>1)':>8} {'SDC':>8}")
    for f, d in data.items():
        x = d.delta_rms.to_numpy()
        print(f"{f:>6} {len(d):>6} {x.mean():>8.4f} {np.median(x):>8.4f} "
              f"{np.quantile(x, 0.9):>8.4f} {(x > 1).mean():>8.4f} {d.sdc.mean():>8.4f}")

    # shared bins so the two curves are directly comparable
    allx = np.concatenate([d.delta_rms.to_numpy() for d in data.values()])
    edges = np.unique(np.quantile(allx[allx > 0], np.linspace(0, 1, 9)))
    print("\n--- P(SDC | perturbation size), same bins for both formats ---")
    hdr = "  ".join(f"{f:>16}" for f in a.formats)
    print(f"{'|dw| / rms bin':>22}  {hdr}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        cells = []
        for f in a.formats:
            d = data[f]
            m = (d.delta_rms > lo) & (d.delta_rms <= hi)
            cells.append(f"{d[m].sdc.mean():.4f} (n={int(m.sum())})" if m.sum() else "     -      ")
        print(f"{lo:>10.3f} - {hi:<9.3f}  " + "  ".join(f"{c:>16}" for c in cells))

    # one-number summary: damage per unit perturbation, matched on size
    f1, f2 = a.formats
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m1 = (data[f1].delta_rms > lo) & (data[f1].delta_rms <= hi)
        m2 = (data[f2].delta_rms > lo) & (data[f2].delta_rms <= hi)
        if m1.sum() >= 50 and m2.sum() >= 50:
            r1, r2 = data[f1][m1].sdc.mean(), data[f2][m2].sdc.mean()
            if r1 > 0:
                ratios.append(r2 / r1)
    if ratios:
        print(f"\n  within matched perturbation bins, {f2} fails "
              f"{np.mean(ratios):.1f}x more often than {f1} on average")
        print("  -> the gap is NOT explained by perturbation size" if np.mean(ratios) > 2
              else "  -> perturbation size accounts for most of the gap")

    out = RESULTS / f"e09_{a.model}_delta_vs_damage.csv"
    pd.concat([d.assign(fmt=f) for f, d in data.items()]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
