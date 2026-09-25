"""E15 -- what the severity measure actually buys: a value-aware campaign.

E12 showed `margin_shift` ranks faults by whether they will fail (AUC 0.990).
Ranking is not the point in itself; the point is spending fewer injections for
the same answer, which is what the project plan wanted from value-aware
sampling. This measures that.

The comparison can be made exactly rather than approximately. The exhaustive
campaigns recorded the outcome of *every* fault, so any sampling strategy can
be replayed against known ground truth, thousands of times, without running a
single further inference. What a strategy costs is the number of injections it
draws; what it earns is the error of its estimate of the true failure rate.

The predictor itself is nearly free, which is the whole economic argument:
`margin_shift` = $s_i|v'-v|/\\mathrm{rms}$ needs one backpropagation pass per
evaluation image to get the sensitivities $s_i$, and then closed-form table
lookups per fault. Measuring a fault needs a forward pass over the whole
evaluation set. So the predictor can be computed for the entire fault space
for less than the cost of a hundred measured injections.

Scale faults are scored by the same rule summed over the block they rescale,
since a scale fault moves all $K$ weights at once.

Run::

    PYTHONPATH=. python3 -m experiments.e15_value_aware --fmt e3m2 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mxfi.data import campaign_subset
from mxfi.formats import decode_e8m0, get_format
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained, setup_device

RESULTS = Path(__file__).resolve().parent.parent / "results"
HUGE = 1e30          # rank-only sentinel for faults whose perturbation is unbounded


def sensitivities(mx: MXModel, xs: torch.Tensor, device: str):
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
        m_val = max(float(margin), 1e-6)
        with torch.no_grad():
            for n, m in layers.items():
                if m.weight.grad is not None:
                    s = m.weight.grad.abs() * (rms[n] / m_val)
                    torch.maximum(best[n], s, out=best[n])
    for m in layers.values():
        m.weight.requires_grad_(False)
    return {n: b.detach().cpu().numpy() for n, b in best.items()}, rms


def score_faults(d: pd.DataFrame, mx: MXModel, sens: dict, rms: dict, K: int) -> np.ndarray:
    """margin_shift for every recorded fault, vectorised per layer."""
    out = np.zeros(len(d), dtype=np.float64)
    tensors = mx.weight_tensors()
    for name, g in d.groupby("tensor", sort=False):
        t = tensors[name]
        table = t.fmt.table()
        s2d = sens[name].reshape(sens[name].shape[0], -1)
        scale = decode_e8m0(t.scales)
        r = max(rms[name], 1e-30)
        idx = g.index.to_numpy()

        el = g.site.to_numpy() == "element"
        row = g.row.to_numpy(); blk = g.blk.to_numpy()
        lane = g.lane.to_numpy(); bit = g.bit.to_numpy()

        if el.any():
            rr, bb, ll, kk = row[el], blk[el], lane[el], bit[el]
            c = t.codes[rr, bb, ll].astype(np.int64)
            after = table[c ^ (1 << kk)].astype(np.float64)
            delta = np.abs(after - table[c].astype(np.float64)) * scale[rr, bb]
            # a flip into NaN/Inf is maximally severe, not unscoreable: rank it top
            delta = np.where(np.isfinite(delta), delta, HUGE)
            col = bb * K + ll
            out[idx[el]] = s2d[rr, col] * delta / r

        sc = ~el
        if sc.any():
            rr, bb, kk = row[sc], blk[sc], bit[sc]
            byte = t.scales[rr, bb].astype(np.int64)
            mult = decode_e8m0((byte ^ (1 << kk)).astype(np.uint8)).astype(np.float64)
            vals = table[t.codes[rr, bb]].astype(np.float64)    # (n, K)
            # float64 throughout: a scale flip can multiply by 2**127, which
            # overflows float32 and would otherwise score as zero severity
            delta = np.abs(vals * mult[:, None] - vals * scale[rr, bb][:, None].astype(np.float64))
            cols = bb[:, None] * K + np.arange(K)[None, :]
            # the final block of a layer may be part padding: those lanes hold no
            # model value, so they must not contribute to the fault's severity
            real = cols < s2d.shape[1]
            s_blk = np.where(real, s2d[rr[:, None], np.clip(cols, 0, s2d.shape[1] - 1)], 0.0)
            delta = np.where(np.isfinite(delta), delta, HUGE)   # NaN scale code, or overflow
            delta = np.where(real, delta, 0.0)
            out[idx[sc]] = np.minimum((s_blk * delta).sum(axis=1) / r, HUGE)
    return out


def replay(truth: np.ndarray, score: np.ndarray, budgets, reps: int, rng) -> pd.DataFrame:
    """Uniform vs value-aware stratified sampling, replayed against ground truth."""
    N = len(truth)
    p_true = truth.mean()
    # three strata by predicted severity; the boundaries are the natural ones
    edges = [0.0, 0.1, 1.0, np.inf]
    stratum = np.digitize(score, edges[1:-1])
    w = np.array([(stratum == k).mean() for k in range(3)])
    members = [np.flatnonzero(stratum == k) for k in range(3)]

    rows = []
    for n in budgets:
        unif, strat = np.empty(reps), np.empty(reps)
        for r in range(reps):
            unif[r] = truth[rng.integers(0, N, n)].mean()

            # 20% pilot to estimate each stratum, then Neyman-allocate the rest
            pilot = max(int(0.2 * n) // 3, 5)
            p_hat, var = np.empty(3), np.empty(3)
            used = 0
            for k in range(3):
                if len(members[k]) == 0:
                    p_hat[k] = var[k] = 0.0
                    continue
                s = truth[rng.choice(members[k], pilot, replace=True)]
                p_hat[k] = s.mean(); var[k] = max(p_hat[k] * (1 - p_hat[k]), 1e-4)
                used += pilot
            alloc = w * np.sqrt(var)
            alloc = alloc / max(alloc.sum(), 1e-12) * max(n - used, 0)
            est, tot_w = 0.0, 0.0
            for k in range(3):
                nk = int(round(alloc[k])) + pilot
                if len(members[k]) == 0 or nk <= 0:
                    continue
                s = truth[rng.choice(members[k], nk, replace=True)]
                est += w[k] * s.mean(); tot_w += w[k]
            strat[r] = est / max(tot_w, 1e-12)

        rmse_u = float(np.sqrt(((unif - p_true) ** 2).mean()))
        rmse_s = float(np.sqrt(((strat - p_true) ** 2).mean()))
        rows.append({"budget": n, "rmse_uniform": rmse_u, "rmse_valueaware": rmse_s,
                     "variance_ratio": (rmse_u / max(rmse_s, 1e-12)) ** 2,
                     "bias_uniform": float(unif.mean() - p_true),
                     "bias_valueaware": float(strat.mean() - p_true)})
    df = pd.DataFrame(rows)
    df.attrs["strata"] = (w, [truth[m].mean() if len(m) else np.nan for m in members])
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8_w16")
    p.add_argument("--fmt", default="e3m2")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reps", type=int, default=400)
    p.add_argument("--budgets", nargs="*", type=int, default=[500, 1000, 3000])
    a = p.parse_args()

    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size)
    csv = RESULTS / f"e06_{a.model}_{cfg.label()}_exhaustive.csv"
    if not csv.exists():
        alt = RESULTS / "e06.csv.gz"
        if a.fmt == "e4m3" and alt.exists():
            csv = alt
        else:
            raise SystemExit(f"no exhaustive campaign at {csv}")
    d = pd.read_csv(csv)
    print(f"{a.model} {cfg.label()}: {len(d):,} exhaustively measured faults, "
          f"true rate {d.sdc.mean():.6f}")

    dev = setup_device(a.device)
    net, _ = load_trained(a.model)
    mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
    xs = campaign_subset(n_per_class=20, download=False).tensors[0]
    sens, rms = sensitivities(mx, xs, dev)

    score = score_faults(d.reset_index(drop=True), mx, sens, rms, a.block_size)
    d = d.reset_index(drop=True)
    d["margin_shift"] = score

    from scipy import stats as st
    ok = np.isfinite(score)
    auc = st.mannwhitneyu(score[ok & d.sdc.to_numpy()],
                          score[ok & ~d.sdc.to_numpy()]).statistic / \
        (int((ok & d.sdc.to_numpy()).sum()) * int((ok & ~d.sdc.to_numpy()).sum()))
    print(f"predictor AUC over the whole fault space: {auc:.4f}")

    df = replay(d.sdc.to_numpy().astype(float), score, a.budgets, a.reps,
                np.random.default_rng(0))
    w, pk = df.attrs["strata"]
    print(f"\nstrata by predicted severity (margin_shift):")
    for k, (name, lo, hi) in enumerate((("low", 0, 0.1), ("middle", 0.1, 1.0),
                                        ("high", 1.0, "inf"))):
        print(f"  {name:>7} [{lo}, {hi}): {w[k]:6.2%} of the fault space, "
              f"true failure rate {pk[k]:.4f}")

    print(f"\n{'budget':>7} {'RMSE uniform':>14} {'RMSE value-aware':>18} "
          f"{'variance ratio':>15} {'equivalent uniform budget':>26}")
    for _, r in df.iterrows():
        print(f"{int(r.budget):>7} {r.rmse_uniform:>14.5f} {r.rmse_valueaware:>18.5f} "
              f"{r.variance_ratio:>14.2f}x {int(r.budget * r.variance_ratio):>25,}")
    print(f"\n  bias at the largest budget: uniform {df.bias_uniform.iloc[-1]:+.5f}, "
          f"value-aware {df.bias_valueaware.iloc[-1]:+.5f} (both should be ~0)")

    out = RESULTS / f"e15_{a.model}_{a.fmt}_value_aware.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
