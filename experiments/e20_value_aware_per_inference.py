"""E20 -- value-aware sampling re-derived for the per-inference rate.

F46 and F47 measured what `margin_shift` stratification buys, but they bought it
for the *any-image* rate, the one quantity F52 concluded should not be quoted. The
exhaustive campaigns recorded ``changed`` per fault, so per-inference ground truth
already exists for all 1.1M enumerated faults and the re-derivation needs no new
injections -- only a different estimator.

The change is not cosmetic. The any-image rate is a proportion, so a stratum's
variance is $p(1-p)$ and Neyman allocation follows from the stratum means alone.
The per-inference rate is a *mean of per-fault proportions*, so each stratum has a
variance of its own that the pilot has to estimate:

    theta = sum_k w_k mu_k,    Var = sum_k w_k^2 sigma_k^2 / n_k,
    Neyman:  n_k proportional to w_k sigma_k.

There is reason to expect a larger gain than the binary case gave. A severity
measure is trying to predict how much damage a fault does, which is what the
per-inference rate records; the any-image rate saturates as soon as one near-tie
image flips, discarding exactly the gradation `margin_shift` provides. AUC is no
longer the right diagnostic for a continuous target, so Spearman's rho replaces it.

Two stratifications are compared: the fixed cuts F46 used, for continuity, and cuts
at quantiles of the score, which cost nothing extra because the score is computed
over the whole fault space anyway.

Run::

    PYTHONPATH=. python3 -m experiments.e20_value_aware_per_inference \\
        --fmt e3m2 --train-seed 0 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e15_value_aware import score_faults, sensitivities
from mxfi.campaign import Evaluator
from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained, setup_device
from tools.verify_ground_truth import AGREEMENT_FLOOR, replay_agreement

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMAGES = 200
SD_FLOOR = 1e-4     # keeps a token allocation where a small pilot reads zero variance

SCHEMES = {
    "fixed (F46's cuts)": ("fixed", [0.1, 1.0]),
    "score quantiles": ("quantile", [0.50, 0.75, 0.90, 0.97, 0.995]),
}


def strata(score: np.ndarray, kind: str, cuts: list[float]) -> np.ndarray:
    edges = (np.asarray(cuts) if kind == "fixed"
             else np.unique(np.quantile(score, cuts)))
    return np.digitize(score, edges)


def replay(y: np.ndarray, score: np.ndarray, budgets, reps: int,
           rng: np.random.Generator) -> pd.DataFrame:
    """Uniform against value-aware stratified sampling, for a continuous target."""
    N = len(y)
    theta = y.mean()
    rows = []
    for label, (kind, cuts) in SCHEMES.items():
        sid = strata(score, kind, cuts)
        ks = np.unique(sid)
        w = np.array([(sid == k).mean() for k in ks])
        members = [np.flatnonzero(sid == k) for k in ks]
        K = len(ks)

        for n in budgets:
            unif, strat = np.empty(reps), np.empty(reps)
            for r in range(reps):
                unif[r] = y[rng.integers(0, N, n)].mean()

                pilot = max(int(0.2 * n) // K, 5)
                sd = np.empty(K)
                for j in range(K):
                    s = y[rng.choice(members[j], pilot, replace=True)]
                    sd[j] = max(s.std(ddof=1) if pilot > 1 else 0.0, SD_FLOOR)
                alloc = w * sd
                alloc = alloc / max(alloc.sum(), 1e-12) * max(n - pilot * K, 0)
                est = 0.0
                for j in range(K):
                    nj = int(round(alloc[j])) + pilot
                    est += w[j] * y[rng.choice(members[j], nj, replace=True)].mean()
                strat[r] = est

            ru = float(np.sqrt(((unif - theta) ** 2).mean()))
            rs = float(np.sqrt(((strat - theta) ** 2).mean()))
            rows.append({"scheme": label, "strata": K, "budget": n,
                         "rmse_uniform": ru, "rmse_valueaware": rs,
                         "variance_ratio": (ru / max(rs, 1e-15)) ** 2,
                         "bias_uniform": float(unif.mean() - theta),
                         "bias_valueaware": float(strat.mean() - theta)})

        if label.startswith("fixed"):
            rows[-1]["_detail"] = [(float(w[j]), float(y[members[j]].mean()),
                                    float(y[members[j]].std())) for j in range(K)]
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8_w16")
    p.add_argument("--fmt", default="e3m2")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--verify", type=int, default=200)
    p.add_argument("--reps", type=int, default=400)
    p.add_argument("--budgets", nargs="*", type=int, default=[500, 1000, 3000])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size)
    csv = RESULTS / f"e06_{a.model}_{cfg.label()}_exhaustive.csv"
    if not csv.exists():
        alt = RESULTS / "e06.csv.gz"
        if a.fmt == "e4m3" and alt.exists():
            csv = alt
        else:
            raise SystemExit(f"no exhaustive campaign at {csv}")
    d = pd.read_csv(csv).reset_index(drop=True)
    y = d.changed.to_numpy(dtype=np.float64) / IMAGES
    print(f"{a.model} {cfg.label()}: {len(d):,} exhaustively measured faults")
    print(f"  true any-image rate    {d.sdc.mean():.6f}")
    print(f"  true per-inference rate {y.mean():.6f}  (sd across faults {y.std():.6f})")

    dev = setup_device(a.device)
    net, _ = load_trained(a.model, seed=a.train_seed)
    mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
    ds = campaign_subset(n_per_class=20, download=False)

    if a.verify:
        ev = Evaluator(ds, device=dev)
        frac = replay_agreement(mx, ev, ev.golden(mx.module), d, a.verify,
                                np.random.default_rng(0))
        print(f"ground-truth check: {frac:.3f} of {a.verify} replayed faults agree")
        if frac < AGREEMENT_FLOOR:
            raise SystemExit(f"this campaign was not recorded from {a.model} "
                             f"train-seed {a.train_seed}")

    sens, rms = sensitivities(mx, ds.tensors[0], dev)
    score = score_faults(d, mx, sens, rms, a.block_size)

    from scipy import stats as st
    ok = np.isfinite(score)
    rho = st.spearmanr(score[ok], y[ok])[0]
    print(f"\npredictor against the per-inference rate: spearman {rho:+.4f}")
    print(f"  (against the any-image rate it was an AUC, not a correlation)")

    df = replay(y, score, a.budgets, a.reps, np.random.default_rng(0))

    detail = df._detail.dropna().iloc[-1] if "_detail" in df else None
    if detail is not None:
        print("\nstrata under F46's fixed cuts, now describing a continuous target:")
        names = ("low [0, 0.1)", "middle [0.1, 1)", "high [1, inf)")
        for nm, (wk, mu, sd) in zip(names, detail):
            print(f"  {nm:>16}: {wk:6.2%} of the space, mean {mu:.6f}, sd {sd:.6f}")

    for label in SCHEMES:
        g = df[df.scheme == label]
        print(f"\n{label}  ({int(g.strata.iloc[0])} strata)")
        print(f"{'budget':>7} {'RMSE uniform':>14} {'RMSE value-aware':>18} "
              f"{'variance ratio':>15} {'equivalent uniform':>20}")
        for _, r in g.iterrows():
            print(f"{int(r.budget):>7} {r.rmse_uniform:>14.6f} "
                  f"{r.rmse_valueaware:>18.6f} {r.variance_ratio:>14.2f}x "
                  f"{int(r.budget * r.variance_ratio):>19,}")
        print(f"  bias at the largest budget: uniform "
              f"{g.bias_uniform.iloc[-1]:+.6f}, value-aware "
              f"{g.bias_valueaware.iloc[-1]:+.6f}")

    out = RESULTS / f"e20_{a.model}_{a.fmt}_value_aware_per_inference.csv"
    df.drop(columns=[c for c in ("_detail",) if c in df]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
