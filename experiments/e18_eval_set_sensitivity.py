"""E18 -- how much of a measured failure rate belongs to the evaluation set?

E17 found that the statistic predicting a campaign's failure rate -- how close
its closest evaluation image sits to a decision boundary -- predicts at
Spearman $+0.98$ when measured on the campaign's own images and $+0.21$ when
measured on fresh ones. That makes the *evaluation set* a suspect in every
cross-format comparison in this study, including the headline one: ResNet8 under
e3m2 fails at $0.079$ against e4m3 at $0.400$, a factor of five, measured
exhaustively over the fault space but on a single set of $200$ images.

Exhaustive over faults is not exhaustive over inputs. This measures the input
half directly: the same fault list is replayed against several disjoint,
class-balanced image sets, so the only thing that changes between cells is which
images the campaign happens to look at. What comes out is how far a quoted rate
and a quoted ratio move when nothing about the network or the fault model
changes at all.

The design is paired within a format -- one fault list, every fold -- so the
fold-to-fold spread contains no sampling noise from the fault side.

Run::

    PYTHONPATH=. python3 -m experiments.e18_eval_set_sensitivity --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from mxfi.campaign import Evaluator, run_campaign
from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.sampling import sample_uniform
from mxfi.stats import binomial_rate
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained, setup_device

RESULTS = Path(__file__).resolve().parent.parent / "results"


def disjoint_folds(n_folds: int, per_class: int) -> list[TensorDataset]:
    """Split one large class-balanced draw into disjoint, class-balanced folds."""
    big = campaign_subset(n_per_class=n_folds * per_class, download=False)
    xs, ys = big.tensors
    folds = [[] for _ in range(n_folds)]
    for c in ys.unique():
        idx = torch.nonzero(ys == c, as_tuple=True)[0]
        for f in range(n_folds):
            folds[f].append(idx[f * per_class : (f + 1) * per_class])
    out = []
    for f in folds:
        sel = torch.cat(f)
        out.append(TensorDataset(xs[sel], ys[sel]))
    return out


def min_margin(module: torch.nn.Module, ds: TensorDataset, device: str) -> float:
    with torch.no_grad():
        z = module(ds.tensors[0].to(device))
        t2 = torch.topk(z, 2, dim=1).values
        return float((t2[:, 0] - t2[:, 1]).min())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8_w16")
    p.add_argument("--formats", nargs="*", default=["e3m2", "e4m3"])
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--images", type=int, default=20, help="per class per fold")
    p.add_argument("--n", type=int, default=1500, help="element faults per format")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--with-original", action="store_true",
                   help="prepend the 200-image set every earlier campaign used")
    a = p.parse_args()

    dev = setup_device(a.device)
    folds = disjoint_folds(a.folds, a.images) if a.folds else []
    if a.with_original:
        # the exhaustive campaigns, E11, E12 and every cross-format comparison in
        # this study were all scored on this one draw
        folds.insert(0, campaign_subset(n_per_class=a.images, download=False))
    print(f"{a.model}: {a.folds} disjoint folds of {len(folds[0])} images, "
          f"{a.n} element faults per format, the same list for every fold")

    rows = []
    for fmt in a.formats:
        cfg = MXConfig(fmt=fmt, block_size=a.block_size)
        net, _ = load_trained(a.model, seed=a.train_seed)
        mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()

        # one fault list per format, reused across every fold: the paired design
        rng = np.random.default_rng(a.seed)
        faults = [f for f in sample_uniform(mx.weight_tensors(), a.n * 2, rng)
                  if f.site == "element"][:a.n]
        print(f"\n{fmt}: {len(faults)} element faults")

        for k, ds in enumerate(folds):
            ev = Evaluator(ds, device=dev)
            golden = ev.golden(mx.module)
            df = run_campaign(mx, faults, ev, golden, progress=False)
            est = binomial_rate(df.sdc.to_numpy())
            rows.append({
                "fmt": fmt, "fold": k, "images": len(ds),
                "golden_acc": golden.accuracy,
                "min_margin": min_margin(mx.module, ds, dev),
                "sdc": float(df.sdc.mean()),
                # sdc counts a fault as failing if ANY of the n images flips, so it
                # grows with n. change_rate is the per-inference probability, which
                # ought not to depend on the image set at all.
                "change_rate": float(df.change_rate.mean()),
                "ci_lo": est.lo, "ci_hi": est.hi,
                "nonfinite": float(df.nonfinite.mean()),
            })
            r = rows[-1]
            print(f"  fold {k}: golden acc {r['golden_acc']:.4f}, "
                  f"min margin {r['min_margin']:.4f}, SDC {r['sdc']:.4f} "
                  f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
        mx.restore_weights()

    df = pd.DataFrame(rows)
    out = RESULTS / f"e18_{a.model}_eval_set_sensitivity.csv"
    df.to_csv(out, index=False)

    print(f"\n{'fmt':>6} {'mean SDC':>9} {'min':>8} {'max':>8} {'spread':>8} "
          f"{'CI width':>9} {'spread / CI':>12}")
    for fmt, g in df.groupby("fmt", sort=False):
        ciw = float((g.ci_hi - g.ci_lo).mean())
        span = float(g.sdc.max() - g.sdc.min())
        print(f"{fmt:>6} {g.sdc.mean():>9.4f} {g.sdc.min():>8.4f} {g.sdc.max():>8.4f} "
              f"{span:>8.4f} {ciw:>9.4f} {span / max(ciw, 1e-12):>11.2f}x")

    print("\nthe same folds under the per-inference rate, which unlike SDC does not "
          "grow with the size of the image set:")
    print(f"{'fmt':>6} {'mean':>9} {'min':>8} {'max':>8} {'max/min':>9}")
    for fmt, g in df.groupby("fmt", sort=False):
        print(f"{fmt:>6} {g.change_rate.mean():>9.5f} {g.change_rate.min():>8.5f} "
              f"{g.change_rate.max():>8.5f} "
              f"{g.change_rate.max() / max(g.change_rate.min(), 1e-12):>8.2f}x")

    if len(a.formats) == 2:
        p0 = df[df.fmt == a.formats[0]].set_index("fold").sdc
        p1 = df[df.fmt == a.formats[1]].set_index("fold").sdc
        c0 = df[df.fmt == a.formats[0]].set_index("fold").change_rate
        c1 = df[df.fmt == a.formats[1]].set_index("fold").change_rate
        print(f"\npooled over every fold: {a.formats[0]} SDC {p0.mean():.4f} against "
              f"{a.formats[1]} {p1.mean():.4f} ({p1.mean() / max(p0.mean(), 1e-9):.2f}x); "
              f"per-inference {c0.mean():.5f} against {c1.mean():.5f} "
              f"({c1.mean() / max(c0.mean(), 1e-12):.2f}x)")
        ratio = (p1 / p0.clip(lower=1e-9)).sort_index()
        print(f"\nthe headline ratio, {a.formats[1]} over {a.formats[0]}, "
              f"recomputed per fold:")
        print("  " + "  ".join(f"{v:.2f}x" for v in ratio))
        print(f"  from {ratio.min():.2f}x to {ratio.max():.2f}x, "
              f"median {ratio.median():.2f}x")
        # does the fold's closest-to-the-boundary image explain the fold's rate?
        from scipy import stats as st
        for fmt, g in df.groupby("fmt", sort=False):
            inv = 1.0 / g.min_margin.clip(lower=1e-9)
            print(f"  {fmt}: fold SDC against 1/min margin, "
                  f"spearman {st.spearmanr(inv, g.sdc)[0]:+.3f} (n={len(g)})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
