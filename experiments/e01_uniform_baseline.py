"""E01 -- uniform fault-injection baseline on the trained ResNet8.

The first real measurement: per-site SDC rates on a trained network under
physically faithful sampling, where every stored bit is equally likely to
flip.  Under that model scale bits are only ~3% of storage, so this run also
shows *why* stratification is needed -- the rare stratum is exactly the one
with the K-value blast radius.

Run::

    .venv/Scripts/python.exe -m experiments.e01_uniform_baseline --n 3000

Outputs ``results/e01_<label>.csv`` plus a printed summary.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mxfi.campaign import Evaluator, run_campaign, summarise
from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.sampling import sample_uniform
from mxfi.stats import binomial_rate
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import evaluate_accuracy, load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8",
                   choices=sorted(__import__("mxfi.models", fromlist=["MODELS"]).MODELS))
    p.add_argument("--fmt", default="e4m3")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--scale-mode", default="ocp", choices=("ocp", "fit"))
    p.add_argument("--n", type=int, default=3000, help="injections")
    p.add_argument("--images", type=int, default=20, help="per class (10 classes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-seed", type=int, default=0,
                   help="which independently trained copy of the model to load")
    a = p.parse_args()
    tag = a.model if a.train_seed == 0 else f"{a.model}_s{a.train_seed}"

    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size, scale_mode=a.scale_mode)
    label = f"{cfg.label()}-n{a.n}"
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- model: trained -> BN-folded -> MX-quantised
    net, state = load_trained(a.model, seed=a.train_seed)
    print(f"checkpoint: epoch {state['epoch']}, test acc {state['acc']:.4f} "
          f"(best {state['best_acc']:.4f})")

    from mxfi.train import setup_device
    dev = setup_device(a.device)
    folded = to_deploy(net).to(dev)
    mx = MXModel(folded, cfg).quantize_weights()

    ds = campaign_subset(n_per_class=a.images, download=False)
    ev = Evaluator(ds, device=dev)
    golden = ev.golden(mx.module)
    print(f"config {cfg.label()} | subset {golden.n} images | "
          f"golden acc on subset {golden.accuracy:.4f}")

    # ---- fault space
    tensors = mx.weight_tensors()
    from mxfi.sampling import bit_population
    pop = bit_population(tensors)
    scale_share = sum(v for k, v in pop.items() if k[1] == "scale") / sum(pop.values())
    print(f"fault space {sum(pop.values()):,} bits | scale share {scale_share:.2%}")

    # ---- campaign
    rng = np.random.default_rng(a.seed)
    faults = sample_uniform(tensors, a.n, rng)
    t0 = time.time()
    df = run_campaign(mx, faults, ev, golden, out_csv=RESULTS / f"e01_{tag}_{label}.csv")
    print(f"\n{len(df)} injections in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(df)*1000:.0f} ms each)")

    # ---- report
    overall = binomial_rate(df["sdc"].to_numpy())
    print(f"\nmodel-wide SDC rate: {overall}")

    print("\n--- by site ---")
    print(summarise(df, by=("site",)).to_string(index=False))

    print("\n--- by site x bit ---")
    print(summarise(df, by=("site", "bit")).to_string(index=False))

    print("\n--- by layer ---")
    print(summarise(df, by=("tensor",)).to_string(index=False))

    # the headline comparison
    e = df[df.site == "element"]
    s = df[df.site == "scale"]
    if len(s) and len(e):
        re_, rs = e["sdc"].mean(), s["sdc"].mean()
        print(f"\nscale-vs-element SDC rate: {rs:.4f} vs {re_:.4f} "
              f"({rs / max(re_, 1e-9):.1f}x)")
        print(f"non-finite outputs: scale {s['nonfinite'].mean():.2%}, "
              f"element {e['nonfinite'].mean():.2%}")
        print(f"mean images corrupted per fault: scale {s['changed'].mean():.1f}, "
              f"element {e['changed'].mean():.1f} (of {golden.n})")
    print(f"\nwrote {RESULTS / f'e01_{tag}_{label}.csv'}")


if __name__ == "__main__":
    main()
