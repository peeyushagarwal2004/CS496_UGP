"""E03 -- block size as an efficiency/reliability trade (objective O3).

Block size K sets the scale-sharing granularity, and it pulls two ways:

* larger K -> fewer scale bytes per tensor, so scale bits are a *smaller*
  share of storage and are hit less often
* larger K -> each scale fault rescales *more* values, so each fault is worse

E00 showed accuracy is nearly flat across K (<1 pp), so any reliability
difference found here is not confounded by an accuracy difference.

Budget is split 50/50 between element and scale strata rather than sampled
uniformly: F9 showed the scale stratum is where stratification pays, and this
experiment is specifically about the scale stratum's behaviour.

Run::

    .venv/Scripts/python.exe -m experiments.e03_block_size --n 2000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.campaign import Evaluator, run_campaign
from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.sampling import bit_population, sample_stratified
from mxfi.stats import StratumCount, binomial_rate, stratified_rate
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"


def half_and_half(tensors, n: int) -> dict[tuple, int]:
    """Split `n` evenly between the element and scale sites.

    Within each site the budget is shared out in proportion to bit population,
    so layers stay comparable; only the site split is forced.
    """
    pop = bit_population(tensors)
    alloc: dict[tuple, int] = {}
    for site in ("element", "scale"):
        keys = [k for k in pop if k[1] == site]
        tot = sum(pop[k] for k in keys)
        budget = n // 2
        got = 0
        for k in keys[:-1]:
            alloc[k] = int(round(budget * pop[k] / tot))
            got += alloc[k]
        alloc[keys[-1]] = budget - got
    return alloc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8",
                   choices=sorted(__import__("mxfi.models", fromlist=["MODELS"]).MODELS))
    p.add_argument("--formats", nargs="*", default=["e4m3", "e2m1"])
    p.add_argument("--blocks", nargs="*", type=int, default=[8, 16, 32, 64])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--seed", type=int, default=2)
    a = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    net, _ = load_trained(a.model)
    ev = Evaluator(campaign_subset(n_per_class=a.images, download=False))

    rows = []
    for fmt in a.formats:
        for K in a.blocks:
            cfg = MXConfig(fmt=fmt, block_size=K)
            mx = MXModel(to_deploy(net), cfg).quantize_weights()
            golden = ev.golden(mx.module)

            tensors = mx.weight_tensors()
            pop = bit_population(tensors)
            total = sum(pop.values())
            scale_share = sum(v for k, v in pop.items() if k[1] == "scale") / total

            rng = np.random.default_rng(a.seed)
            faults, w = sample_stratified(tensors, half_and_half(tensors, a.n), rng)
            t0 = time.time()
            df = run_campaign(mx, faults, ev, golden, weights=w,
                              out_csv=RESULTS / f"e03_{a.model}_{cfg.label()}.csv",
                              progress=False)

            counts = {}
            for (tname, site), g in df.groupby(["tensor", "site"]):
                key = (tname, site, None)
                if key in pop:
                    counts[key] = StratumCount(int(g.sdc.sum()), len(g),
                                               pop[key] / total)
            model = stratified_rate(counts)
            e = df[df.site == "element"]
            s = df[df.site == "scale"]
            e_est, s_est = binomial_rate(e.sdc.to_numpy()), binomial_rate(s.sdc.to_numpy())

            rows.append({
                "fmt": fmt, "K": K,
                "scale_bit_share": scale_share,
                "model_sdc": model.rate,
                "element_sdc": e_est.rate, "element_hw": e_est.half_width,
                "scale_sdc": s_est.rate, "scale_hw": s_est.half_width,
                "scale_imgs": float(s.changed.mean()),
                "element_imgs": float(e.changed.mean()),
                "scale_nonfinite": float(s.nonfinite.mean()),
                "n_scale": len(s), "n_element": len(e),
                "golden_acc": golden.accuracy,
                "secs": round(time.time() - t0),
            })
            print(f"{fmt} K={K:2d}  scale_bits={scale_share:6.2%}  "
                  f"elem_sdc={e_est.rate:.3f}  scale_sdc={s_est.rate:.3f}"
                  f"+-{s_est.half_width:.3f}  scale_imgs={s.changed.mean():5.1f}  "
                  f"model={model.rate:.4f}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"e03_{a.model}_block_size.csv", index=False)

    print("\n=== scale-fault SDC rate vs K ===")
    print(df.pivot(index="fmt", columns="K", values="scale_sdc").to_string())
    print("\n=== images corrupted per scale fault vs K (of 200) ===")
    print(df.pivot(index="fmt", columns="K", values="scale_imgs").to_string())
    print("\n=== element-fault SDC rate vs K (expect ~flat) ===")
    print(df.pivot(index="fmt", columns="K", values="element_sdc").to_string())
    print("\n=== model-wide SDC rate vs K ===")
    print(df.pivot(index="fmt", columns="K", values="model_sdc").to_string())
    print("\n=== scale bits as share of storage vs K ===")
    print(df.pivot(index="fmt", columns="K", values="scale_bit_share").to_string())
    print(f"\nwrote {RESULTS / f'e03_{a.model}_block_size.csv'}")


if __name__ == "__main__":
    main()
