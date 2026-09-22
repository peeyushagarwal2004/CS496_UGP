"""E12 -- a severity measure that actually predicts failure.

F3 showed the measure proposed in the project plan, $d=|\\log_2|v'|-\\log_2|v||$,
has essentially no relationship with observed failures (rank correlation 0.026)
and scores a sign flip -- the most damaging element bit -- as zero. That was a
negative result with no replacement. This supplies one and measures it.

Four candidates, each computed exactly for every injected element fault:

``log_severity``  the plan's measure, $|\\log_2|v'| - \\log_2|v||$
``rel_error``     $|v'-v|/|v|$, the obvious repair for the sign-bit blind spot
``delta_rms``     $|v'-v|$ in units of the layer's weight RMS -- absolute, not
                  relative, since what reaches the next layer is an absolute
                  change
``margin_shift``  $s_i \\cdot |v'-v|/\\mathrm{rms}$, where $s_i$ is the weight's
                  sensitivity $|\\partial m/\\partial w_i|\\,\\mathrm{rms}/m$ from E11.
                  This estimates the fraction of the decision margin the fault
                  consumes, so a value above 1 predicts a flipped prediction.

The first three are properties of the number system alone; only the last knows
anything about the network. E11 found that networks of equal accuracy differ
7x in sensitivity, so the comparison also shows how much of severity is a
property of the format and how much of the model.

Run::

    PYTHONPATH=. python3 -m experiments.e12_severity_metrics --device cuda
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from mxfi.data import campaign_subset
from mxfi.formats import decode_e8m0, get_format
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained, setup_device

RESULTS = Path(__file__).resolve().parent.parent / "results"


def weight_sensitivity(mx: MXModel, xs, device) -> dict:
    """Per-weight |dm/dw| * rms / m, maximised over the evaluation images."""
    layers = {n: l.module for n, l in mx.layers.items()}
    for p in mx.module.parameters():
        p.requires_grad_(False)
    for m in layers.values():
        m.weight.requires_grad_(True)
    rms = {n: float(torch.sqrt((m.weight.detach() ** 2).mean())) for n, m in layers.items()}
    best = {n: torch.zeros_like(m.weight) for n, m in layers.items()}
    for i in range(xs.shape[0]):
        mx.module.zero_grad(set_to_none=True)
        z = mx.module(xs[i : i + 1].to(device))[0]
        t2 = torch.topk(z, 2)
        margin = t2.values[0] - t2.values[1]
        margin.backward()
        with torch.no_grad():
            for n, m in layers.items():
                if m.weight.grad is not None:
                    s = m.weight.grad.abs() * (rms[n] / float(margin.clamp(min=1e-6)))
                    torch.maximum(best[n], s, out=best[n])
    for m in layers.values():
        m.weight.requires_grad_(False)
    return {n: b.detach().cpu().numpy() for n, b in best.items()}, rms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_small")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--formats", nargs="*", default=["e3m2", "e2m3", "e4m3", "e5m2", "e2m1"])
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    dev = setup_device(a.device)
    xs = campaign_subset(n_per_class=20, download=False).tensors[0]
    K = a.block_size
    frames = []

    for fmt in a.formats:
        f = get_format(fmt)
        table = f.table()
        for seed in a.seeds:
            tag = a.model if seed == 0 else f"{a.model}_s{seed}"
            csv = RESULTS / f"e01_{tag}_{fmt}-K{K}-ocp-w-n3000.csv"
            if not csv.exists():
                continue
            net, _ = load_trained(a.model, seed=seed)
            mx = MXModel(to_deploy(net).to(dev),
                         MXConfig(fmt=fmt, block_size=K)).quantize_weights()
            sens, rms = weight_sensitivity(mx, xs, dev)
            tensors = mx.weight_tensors()

            d = pd.read_csv(csv)
            d = d[d.site == "element"].copy()
            drms, mshift = [], []
            for name, idx_s, bit in zip(d.tensor, d["index"], d.bit):
                t = tensors[name]
                idx = ast.literal_eval(idx_s)
                c = int(t.codes[idx])
                scale = float(decode_e8m0(t.scales[idx[:-1]]))
                delta = abs(float(table[c ^ (1 << bit)]) - float(table[c])) * scale
                dr = delta / max(rms[name], 1e-30)
                col = idx[1] * K + idx[2]
                s_i = float(sens[name].reshape(sens[name].shape[0], -1)[idx[0], col])
                drms.append(dr)
                mshift.append(s_i * dr)
            d["delta_rms"], d["margin_shift"] = drms, mshift
            d["fmt"], d["seed"] = fmt, seed
            frames.append(d)
            mx.restore_weights()

    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df.log_severity) & ~df.to_nonfinite]   # perturbation faults only
    df.to_csv(RESULTS / f"e12_{a.model}_severity_metrics.csv", index=False)

    metrics = ["log_severity", "rel_error", "delta_rms", "margin_shift"]
    print(f"model {a.model}, {len(df)} element faults that stayed finite\n")
    print(f"{'measure':>14} {'Spearman':>10} {'AUC':>8}   what it knows")
    knows = {"log_severity": "number system only (the plan's measure)",
             "rel_error": "number system only",
             "delta_rms": "number system + layer scale",
             "margin_shift": "number system + network sensitivity"}
    for m in metrics:
        x, y = df[m].to_numpy(), df.sdc.to_numpy().astype(float)
        ok = np.isfinite(x)
        rho = stats.spearmanr(x[ok], y[ok])[0]
        pos, neg = x[ok][y[ok] == 1], x[ok][y[ok] == 0]
        auc = (stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
               if len(pos) and len(neg) else np.nan)
        print(f"{m:>14} {rho:>10.3f} {auc:>8.3f}   {knows[m]}")

    print(f"\nthe margin_shift > 1 rule (predicting a flip when the fault consumes "
          f"the whole margin):")
    pred = df.margin_shift > 1.0
    tp = int((pred & df.sdc).sum()); fp = int((pred & ~df.sdc).sum())
    fn = int((~pred & df.sdc).sum())
    print(f"  precision {tp / max(tp + fp, 1):.3f}   recall {tp / max(tp + fn, 1):.3f}   "
          f"(flagged {int(pred.sum())} of {len(df)} faults)")
    print(f"\nwrote {RESULTS / f'e12_{a.model}_severity_metrics.csv'}")


if __name__ == "__main__":
    main()
