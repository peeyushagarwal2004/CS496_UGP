"""E02 -- stratified vs uniform allocation at equal injection budget.

Phase I of the drafts' method is an offline characterisation pass; phase II
spends the budget where that pass says it matters.  Here the E01 uniform
campaign *is* the pilot, supplying per-(layer, site) failure rates, and three
allocations are compared at the same total cost:

``uniform``       every stored bit equally likely -- scale bits get ~3%
``neyman``        n proportional to W*sqrt(r(1-r)), minimising the variance
                  of the model-wide rate
``neyman_phi``    the same, times the drafts' blast-radius weight Phi = K,
                  which buys precision on the scale stratum specifically

The two Neyman arms optimise *different* things and the comparison says so:
``neyman`` should tighten the model-wide estimate, ``neyman_phi`` should
tighten the scale stratum at some cost to the model-wide interval.

Run::

    .venv/Scripts/python.exe -m experiments.e02_stratified --n 3000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.campaign import Evaluator, run_campaign, weighted_rate
from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.sampling import (bit_population, default_blast_weight,
                           neyman_allocation, sample_stratified, sample_uniform)
from mxfi.stats import StratumCount, binomial_rate, stratified_rate
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"


def pilot_rates(csv: Path, pop: dict) -> dict[tuple, float]:
    """Per-(layer, site) SDC rates from the E01 uniform campaign.

    Strata that E01 never sampled fall back to that site's pooled rate rather
    than to 0.5, so an unmeasured layer does not hoover up the whole budget.
    """
    df = pd.read_csv(csv)
    per_site = df.groupby("site")["sdc"].mean().to_dict()
    seen = df.groupby(["tensor", "site"])["sdc"].agg(["mean", "size"])

    rates = {}
    for key in pop:
        tensor, site, _ = key
        if (tensor, site) in seen.index and seen.loc[(tensor, site), "size"] >= 5:
            rates[key] = float(seen.loc[(tensor, site), "mean"])
        else:
            rates[key] = float(per_site.get(site, 0.5))
    return rates


def stratum_counts(df: pd.DataFrame, pop: dict, total: int) -> dict:
    """Per-stratum tallies with population shares, for the combined estimate."""
    out = {}
    for (tensor, site), g in df.groupby(["tensor", "site"]):
        key = (tensor, site, None)
        if key in pop:
            out[key] = StratumCount(int(g["sdc"].sum()), len(g), pop[key] / total)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8",
                   choices=sorted(__import__("mxfi.models", fromlist=["MODELS"]).MODELS))
    p.add_argument("--fmt", default="e4m3")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size)
    RESULTS.mkdir(parents=True, exist_ok=True)

    net, state = load_trained(a.model)
    mx = MXModel(to_deploy(net), cfg).quantize_weights()
    ev = Evaluator(campaign_subset(n_per_class=a.images, download=False))
    golden = ev.golden(mx.module)
    print(f"{cfg.label()} | {golden.n} images | golden acc {golden.accuracy:.4f}")

    tensors = mx.weight_tensors()
    pop = bit_population(tensors)
    total = sum(pop.values())
    scale_share = sum(v for k, v in pop.items() if k[1] == "scale") / total
    print(f"fault space {total:,} bits | scale share {scale_share:.2%}\n")

    pilot_csv = RESULTS / f"e01_{a.model}_{cfg.label()}-n3000.csv"
    rates = pilot_rates(pilot_csv, pop)
    print(f"pilot rates from {pilot_csv.name}")

    arms = {
        "uniform": None,
        "neyman": neyman_allocation(tensors, a.n, rates),
        "neyman_phi": neyman_allocation(tensors, a.n, rates,
                                        blast_weight=default_blast_weight(tensors)),
    }
    for name, alloc in arms.items():
        if alloc:
            n_scale = sum(v for k, v in alloc.items() if k[1] == "scale")
            print(f"  {name:11} allocates {n_scale}/{a.n} ({n_scale/a.n:.1%}) to scale")
    print()

    rows, summary = {}, []
    for name, alloc in arms.items():
        rng = np.random.default_rng(a.seed)
        if alloc is None:
            faults, w = sample_uniform(tensors, a.n, rng), None
        else:
            faults, w = sample_stratified(tensors, alloc, rng)

        t0 = time.time()
        df = run_campaign(mx, faults, ev, golden, weights=w,
                          out_csv=RESULTS / f"e02_{a.model}_{cfg.label()}_{name}.csv",
                          progress=False)
        secs = time.time() - t0
        rows[name] = df

        counts = stratum_counts(df, pop, total)
        overall = stratified_rate(counts)
        sc = df[df.site == "scale"]
        el = df[df.site == "element"]
        scale_est = binomial_rate(sc["sdc"].to_numpy()) if len(sc) else None

        summary.append({
            "arm": name, "n": len(df), "secs": round(secs),
            "n_scale": len(sc), "n_element": len(el),
            "model_rate": overall.rate,
            "model_hw": overall.half_width,
            "scale_rate": scale_est.rate if scale_est else np.nan,
            "scale_hw": scale_est.half_width if scale_est else np.nan,
        })
        print(f"{name:11} n_scale={len(sc):5d}  model {overall.rate:.4f} "
              f"+-{overall.half_width:.4f}   scale {scale_est.rate:.4f} "
              f"+-{scale_est.half_width:.4f}  ({secs:.0f}s)")

    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / f"e02_summary_{a.model}_{cfg.label()}.csv", index=False)

    base = s[s.arm == "uniform"].iloc[0]
    print("\n=== precision vs uniform at equal budget ===")
    print(f"{'arm':11} {'model CI':>12} {'vs unif':>9} {'scale CI':>12} {'vs unif':>9}"
          f" {'equiv injections saved':>24}")
    for _, r in s.iterrows():
        m_gain = (base.model_hw / r.model_hw) ** 2
        s_gain = (base.scale_hw / r.scale_hw) ** 2
        print(f"{r.arm:11} +-{r.model_hw:.4f} {m_gain:8.2f}x  "
              f"+-{r.scale_hw:.4f} {s_gain:8.2f}x  "
              f"{'scale stratum ' + format(s_gain, '.1f') + 'x cheaper':>24}")

    print("\n(gain = variance ratio = how many times more uniform injections "
          "would be needed\n for the same interval width)")
    print(f"\nwrote {RESULTS / f'e02_summary_{a.model}_{cfg.label()}.csv'}")


if __name__ == "__main__":
    main()
