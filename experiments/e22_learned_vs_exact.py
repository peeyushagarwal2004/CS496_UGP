"""E22 -- learned value intervals versus exact per-code enumeration.

The plan's other open design choice. TreeFI characterises an FP32 fault space by
*learning* value intervals: inject a pilot sample, fit a regression tree from
each fault's value and bit position to its measured impact, and stratify the
campaign on the tree's leaves. In MX the element alphabet is tiny (64 codes for
e3m2, 256 for e4m3) and the scale is a pure power of two, so the decoded change
a flip causes, |v' - v| = |table[c ^ 2^b] - table[c]| * 2^(X-127), can instead
be *enumerated exactly* for every fault in the space at no injection cost.

Which is the better abstraction is an empirical question with a precise form: at
a fixed injection budget, which stratification estimates the per-inference rate
with the smaller error? The two exhaustive campaigns answer it offline, the same
way E15 and E20 did, by replaying every strategy against known ground truth.

Arms (all estimate the per-inference rate; all unbiased by construction):

* uniform                      -- the reference.
* learned tree (value)         -- TreeFI-style: a 20% uniform pilot is injected,
                                  a regression tree on (site, bit, log2|value|,
                                  sign) is fitted to it, its leaves are the strata,
                                  and the rest of the budget is Neyman-allocated.
* learned tree (value + layer) -- the same, also given the layer, a feature
                                  TreeFI's value intervals do not use. The pilot
                                  only chooses the strata; the estimate uses fresh
                                  draws. A variant that also reuses the pilot in
                                  the estimate is reported to show why not.
* oracle tree (value + layer)  -- the same tree fitted to the *whole* exhaustive
                                  truth: an unattainable upper bound on what the
                                  learned abstraction can express.
* exact |dv| enumeration       -- quantile strata of sum |v' - v| / rms_layer,
                                  computed in closed form over every fault.
* exact margin_shift           -- the same enumeration weighted by the gradient
                                  sensitivity (E12/E20), for reference.

Run::

    PYTHONPATH=. python3 -m experiments.e22_learned_vs_exact --fmt e3m2 \\
        --train-seed 0 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMAGES = 200
SD_FLOOR = 1e-4
QUANTILES = [0.50, 0.75, 0.90, 0.97, 0.995]     # E20's cuts
PILOT = 0.2
LEAVES = 6                                       # as many strata as E20's score cuts;
                                                 # --leaves overrides it


def value_features(d: pd.DataFrame, mx, with_layer: bool) -> np.ndarray:
    """What TreeFI's interval learner sees: where the bit is and what value it holds.

    Element faults carry their decoded weight; scale faults carry the block's
    scale exponent. The site flag lets the tree keep the two apart.
    """
    from mxfi.formats import decode_e8m0

    tensors = mx.weight_tensors()
    names = sorted(tensors)
    f = np.zeros((len(d), 5 if with_layer else 4))
    for name, g in d.groupby("tensor", sort=False):
        t = tensors[name]
        idx = g.index.to_numpy()
        el = g.site.to_numpy() == "element"
        row, blk = g.row.to_numpy(), g.blk.to_numpy()
        lane = g.lane.to_numpy()
        v = np.zeros(len(g))
        scale = decode_e8m0(t.scales).astype(np.float64)
        table = t.fmt.table().astype(np.float64)
        v[el] = table[t.codes[row[el], blk[el], lane[el]]] * scale[row[el], blk[el]]
        v[~el] = scale[row[~el], blk[~el]]
        f[idx, 0] = el
        f[idx, 1] = g.bit.to_numpy()
        f[idx, 2] = np.log2(np.maximum(np.abs(v), 1e-30))
        f[idx, 3] = np.sign(v)
        if with_layer:
            f[idx, 4] = names.index(name)
    return f


def _tree(X: np.ndarray, y: np.ndarray, min_leaf: int):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_leaf_nodes=LEAVES, min_samples_leaf=min_leaf,
                                 random_state=0).fit(X, y)


def _neyman(y, sid, w, members, n_left, seen, rng) -> float:
    """Spend `n_left` by Neyman allocation; `seen[k]` = pilot draws already in k."""
    K = len(members)
    sd = np.array([max(np.std(s, ddof=1), SD_FLOOR) if len(s) > 1 else SD_FLOOR
                   for s in seen])
    alloc = w * sd
    alloc = alloc / max(alloc.sum(), 1e-12) * max(n_left, 0)
    est = 0.0
    for k in range(K):
        extra = y[rng.choice(members[k], int(round(alloc[k])), replace=True)] \
            if len(members[k]) and round(alloc[k]) > 0 else np.empty(0)
        both = np.concatenate([seen[k], extra])
        if len(both) == 0:                    # never leave a stratum unestimated
            both = y[rng.choice(members[k], 1)]
        est += w[k] * both.mean()
    return est


def fixed_strata(y, sid, n, rng) -> float:
    """Strata known before any injection: stratified pilot, then Neyman (as E20)."""
    ks, members, w = _members(sid)
    pilot = max(int(PILOT * n) // len(ks), 5)
    seen = [y[rng.choice(m, pilot, replace=True)] for m in members]
    return _neyman(y, sid, w, members, n - pilot * len(ks), seen, rng)


def _members(sid: np.ndarray):
    ks = np.unique(sid)
    members = [np.flatnonzero(sid == k) for k in ks]
    return ks, members, np.array([len(m) for m in members]) / len(sid)


def proportional(y, members, w, n, rng) -> float:
    """Allocate in proportion to stratum size: no pilot, so nothing to mislead it."""
    left = n - 2 * len(members)
    return sum(w[k] * y[rng.choice(m, 2 + int(round(w[k] * max(left, 0))))].mean()
               for k, m in enumerate(members))


def learned_strata(y, X, n, rng, alloc: str = "neyman") -> float:
    """TreeFI-style: a uniform pilot is injected first, and the strata are learned from it.

    The pilot fixes the leaves. Then, by `alloc`:

    * ``neyman``       -- the pilot also sets each leaf's spread, the rest of the
                          budget is Neyman-allocated, and the estimate uses only
                          fresh draws (at least two per leaf), so it is unbiased;
    * ``proportional`` -- the rest is spread in proportion to leaf size, which
                          isolates what the learned partition itself is worth;
    * ``reuse``        -- as ``neyman`` but the pilot is folded back into the
                          estimate: cheaper, but the same draws then both choose
                          the strata and measure them.
    """
    n0 = int(PILOT * n)
    pick = rng.integers(0, len(y), n0)
    tree = _tree(X[pick], y[pick], min_leaf=max(n0 // 20, 5))
    sid = tree.apply(X)
    ks, members, w = _members(sid)
    if alloc == "proportional":
        return proportional(y, members, w, n - n0, rng)
    pid = sid[pick]
    seen = [y[pick[pid == k]] for k in ks]
    if alloc == "reuse":
        return _neyman(y, sid, w, members, n - n0, seen, rng)
    sd = np.array([max(np.std(s, ddof=1), SD_FLOOR) if len(s) > 1 else SD_FLOOR
                   for s in seen])
    left = n - n0 - 2 * len(ks)
    share = w * sd / max((w * sd).sum(), 1e-12) * max(left, 0)
    return sum(w[k] * y[rng.choice(members[k], 2 + int(round(share[k])))].mean()
               for k in range(len(ks)))


def quantile_ids(score: np.ndarray) -> np.ndarray:
    return np.digitize(score, np.unique(np.quantile(score, QUANTILES)))


def main() -> None:
    global LEAVES
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet8_w16")
    p.add_argument("--fmt", default="e3m2")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--verify", type=int, default=200)
    p.add_argument("--reps", type=int, default=400)
    p.add_argument("--budgets", nargs="*", type=int, default=[500, 1000, 3000])
    p.add_argument("--leaves", type=int, default=LEAVES)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    LEAVES = a.leaves

    from scipy import stats as st

    from experiments.e15_value_aware import score_faults, sensitivities
    from mxfi.campaign import Evaluator
    from mxfi.data import campaign_subset
    from mxfi.models import to_deploy
    from mxfi.torch_mx import MXConfig, MXModel
    from mxfi.train import load_trained, setup_device
    from tools.verify_ground_truth import AGREEMENT_FLOOR, replay_agreement

    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size)
    csv = RESULTS / f"e06_{a.model}_{cfg.label()}_exhaustive.csv"
    d = pd.read_csv(csv).reset_index(drop=True)
    y = d.changed.to_numpy(dtype=np.float64) / IMAGES
    print(f"{a.model} {cfg.label()}: {len(d):,} faults, per-inference rate {y.mean():.6f}")

    dev = setup_device(a.device)
    net, _ = load_trained(a.model, seed=a.train_seed)
    mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
    ds = campaign_subset(n_per_class=IMAGES // 10, download=False)
    if a.verify:
        ev = Evaluator(ds, device=dev)
        frac = replay_agreement(mx, ev, ev.golden(mx.module), d, a.verify,
                                np.random.default_rng(0))
        print(f"ground-truth check: {frac:.3f} of {a.verify} replayed faults agree")
        if frac < AGREEMENT_FLOOR:
            raise SystemExit(f"campaign not recorded from train-seed {a.train_seed}")

    sens, rms = sensitivities(mx, ds.tensors[0], dev)
    ones = {k: np.ones_like(v) for k, v in sens.items()}
    exact_dv = score_faults(d, mx, ones, rms, a.block_size)
    margin = score_faults(d, mx, sens, rms, a.block_size)
    Xv = value_features(d, mx, with_layer=False)
    Xl = value_features(d, mx, with_layer=True)
    oracle = _tree(Xl, y, min_leaf=len(y) // 200).apply(Xl)

    def rho(s):
        ok = np.isfinite(s)
        return float(st.spearmanr(s[ok], y[ok])[0])

    print("\nhow much of the target each abstraction can see (spearman with truth):")
    print(f"  exact |dv| / rms         {rho(exact_dv):+.3f}")
    print(f"  exact margin_shift       {rho(margin):+.3f}")
    oracle_pred = pd.Series(y).groupby(oracle).transform("mean").to_numpy()
    print(f"  oracle tree, value+layer {rho(oracle_pred):+.3f}")
    for nm, X in (("value", Xv), ("value+layer", Xl)):
        r = []
        for seed in range(20):
            g = np.random.default_rng(1000 + seed)
            pick = g.integers(0, len(y), 600)
            r.append(rho(_tree(X[pick], y[pick], 30).predict(X)))
        print(f"  tree on a 600-fault pilot, {nm:<11} {np.mean(r):+.3f} "
              f"(sd {np.std(r):.3f} over 20 pilots)")

    oracle_parts = _members(oracle)
    arms = {
        "learned tree (value)": lambda n, g: learned_strata(y, Xv, n, g),
        "learned tree (value+layer)": lambda n, g: learned_strata(y, Xl, n, g),
        "learned tree (value+layer), pilot reused":
            lambda n, g: learned_strata(y, Xl, n, g, alloc="reuse"),
        "learned tree (value+layer), proportional":
            lambda n, g: learned_strata(y, Xl, n, g, alloc="proportional"),
        "oracle tree (value+layer)": lambda n, g: fixed_strata(y, oracle, n, g),
        "oracle tree (value+layer), proportional":
            lambda n, g: proportional(y, oracle_parts[1], oracle_parts[2], n, g),
        "exact |dv| enumeration": lambda n, g: fixed_strata(y, quantile_ids(exact_dv), n, g),
        "exact margin_shift": lambda n, g: fixed_strata(y, quantile_ids(margin), n, g),
    }
    theta = y.mean()
    rows = []
    for n in a.budgets:
        rng = np.random.default_rng(n)
        unif = np.array([y[rng.integers(0, len(y), n)].mean() for _ in range(a.reps)])
        mse_u = ((unif - theta) ** 2).mean()
        for label, fn in arms.items():
            g = np.random.default_rng(7 * n)
            est = np.array([fn(n, g) for _ in range(a.reps)])
            mse = ((est - theta) ** 2).mean()
            rows.append({"fmt": a.fmt, "leaves": LEAVES, "arm": label, "budget": n,
                         "rmse_uniform": np.sqrt(mse_u), "rmse": np.sqrt(mse),
                         "variance_ratio": mse_u / max(mse, 1e-30),
                         "bias": est.mean() - theta})
    df = pd.DataFrame(rows)
    print(f"\nvariance ratio over uniform sampling ({a.reps} replays per cell)")
    print(df.pivot(index="arm", columns="budget", values="variance_ratio")
            .reindex(list(arms)).round(2).to_string())
    print("\nbias / truth at the largest budget")
    last = df[df.budget == max(a.budgets)].set_index("arm")
    print((last.bias / theta).reindex(list(arms)).map("{:+.4f}".format).to_string())
    suffix = "" if LEAVES == 6 else f"_leaves{LEAVES}"
    out = RESULTS / f"e22_{a.model}_{a.fmt}_learned_vs_exact{suffix}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
