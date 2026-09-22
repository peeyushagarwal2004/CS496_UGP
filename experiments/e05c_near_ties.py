"""E05c -- is "any image changes" SDC measuring near-tie images? (F26)

F25 left one cell unexplained: ``e4m3`` at w=32 has near-zero perturbation SDC in *both*
seeds, while the same trained models are clearly vulnerable under ``e3m2`` and ``e5m2``.
This module holds the three analyses that explain it.

1. ``signflip`` -- the same random element sign flips injected into one model under all
   three formats.  Reports SDC and the size of the logit disturbance.  Equal disturbance
   with very different SDC places the difference in the golden model, not the fault.
2. ``margins`` -- golden top1-top2 logit margin on the 200-image campaign subset for every
   (width, seed, format) with a finished campaign, correlated with perturbation SDC.  A
   near-tie image flips under almost any nudge, so one of them can dominate a campaign.
3. ``robust`` -- the width sweep rescored from the existing ``e01_*`` CSVs under metrics
   that a single knife-edge image cannot dominate (>=3 images change, >=1 pp accuracy
   drop, mean change rate), with seed max/min spread and endpoint ratios.

``robust`` only reads CSVs.  ``signflip`` and ``margins`` re-quantise checkpoints and run
the model on one thread, so they can share the machine with a running sweep.

Run::

    .venv/Scripts/python.exe -m experiments.e05c_near_ties robust --seeds 0 1
    .venv/Scripts/python.exe -m experiments.e05c_near_ties margins
    .venv/Scripts/python.exe -m experiments.e05c_near_ties signflip --width 32 --train-seed 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
FORMATS = ["e3m2", "e4m3", "e5m2"]
N_INJECTIONS = 3000
_CSV = re.compile(r"e01_resnet8_w(\d+)(?:_s(\d+))?_(e\dm\d)-K32-ocp-w-n(\d+)\.csv$")


def _section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def campaigns() -> dict[tuple[int, int, str], pd.DataFrame]:
    """Finished width-sweep campaigns, keyed by (width, train seed, format).

    Only perturbation-driven element faults are kept: non-finite outputs are SDC whatever
    the margins, so they cannot be affected by near-ties (F24).
    """
    out = {}
    for p in sorted(RESULTS.glob("e01_resnet8_w*-K32-ocp-w-n*.csv")):
        m = _CSV.search(p.name)
        if not m or int(m[4]) != N_INJECTIONS:
            continue
        d = pd.read_csv(p)
        if len(d) < N_INJECTIONS:
            continue                       # campaign still running
        out[(int(m[1]), int(m[2] or 0), m[3])] = d[(d.site == "element") & ~d.nonfinite]
    return out


def _mx_model(width: int | None, seed: int, fmt: str, model: str | None = None):
    from mxfi.models import to_deploy
    from mxfi.torch_mx import MXConfig, MXModel
    from mxfi.train import load_trained

    net, _ = load_trained(model or f"resnet8_w{width}", seed=seed)
    cfg = MXConfig(fmt=fmt, block_size=32, scale_mode="ocp")
    return MXModel(to_deploy(net), cfg).quantize_weights()


def _subset_images():
    import torch
    from mxfi.data import campaign_subset

    torch.set_num_threads(1)               # leave the CPU to any running sweep
    return campaign_subset(n_per_class=20, download=False).tensors[0]


def _margins(logits) -> np.ndarray:
    top2 = logits.topk(2, dim=1).values
    return np.sort((top2[:, 0] - top2[:, 1]).numpy())


# ---------------------------------------------------------------- 1. signflip

def signflip(width: int, train_seed: int, n: int, rng_seed: int) -> None:
    import torch
    from mxfi.faults import Fault
    from mxfi.formats import get_format
    from mxfi.sampling import FaultSite

    X = _subset_images()
    models = {f: _mx_model(width, train_seed, f) for f in FORMATS}

    # hidden-layer element positions, drawn once and shared by every format
    tensors = models["e4m3"].weight_tensors()
    names = [k for k in tensors if k.startswith("layer")]
    rng = np.random.default_rng(rng_seed)
    picks = []
    for _ in range(n):
        name = names[rng.integers(len(names))]
        picks.append((name, tuple(int(rng.integers(k)) for k in tensors[name].codes.shape)))

    _section(f"sign flips, w={width} seed {train_seed}: {n} shared element positions")
    for fmt, mx in models.items():
        sign_bit = get_format(fmt).width - 1
        with torch.no_grad():
            L0 = mx.module(X)
        p0 = L0.argmax(1)
        hits, dl = 0, []
        for name, idx in picks:
            with mx.fault(FaultSite(name, Fault("element", idx, sign_bit))):
                with torch.no_grad():
                    L = mx.module(X)
            hits += int((L.argmax(1) != p0).any())
            dl.append(float((L - L0).abs().max()))
        mg = _margins(L0)
        print(f"  {fmt}: SDC {hits}/{n}   median max|dlogit| {np.median(dl):.4f}   "
              f"max {np.max(dl):.3f}   smallest golden margin {mg[0]:.4f}")


# ----------------------------------------------------------------- 2. margins

def margins() -> pd.DataFrame:
    import torch
    from scipy.stats import spearmanr

    X = _subset_images()
    rows = []
    for (w, s, fmt), e in sorted(campaigns().items()):
        with torch.no_grad():
            mg = _margins(_mx_model(w, s, fmt).module(X))
        rows.append(dict(width=w, seed=s, fmt=fmt, pert_sdc=e.sdc.mean(),
                         min_margin=mg[0], margin_2=mg[1], margin_3=mg[2],
                         n_below_0_05=int((mg < 0.05).sum()),
                         n_below_0_2=int((mg < 0.2).sum())))
    t = pd.DataFrame(rows)

    _section(f"golden margins vs perturbation SDC ({len(t)} model-format cells)")
    print(t.round(4).to_string(index=False))
    r1 = spearmanr(t.pert_sdc, t.n_below_0_2)
    r2 = spearmanr(t.pert_sdc, t.min_margin)
    print(f"\n  Spearman(SDC, images with margin < 0.2) = {r1.statistic:+.3f}  (p = {r1.pvalue:.1e})")
    print(f"  Spearman(SDC, smallest margin)          = {r2.statistic:+.3f}  (p = {r2.pvalue:.1e})")
    tie_free = t[t.n_below_0_2 == 0]
    print(f"  cells with no image below 0.2: "
          f"{', '.join(f'w{r.width} s{r.seed} {r.fmt}' for r in tie_free.itertuples()) or 'none'}")

    out = RESULTS / "e05c_golden_margins.csv"
    t.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return t


# ------------------------------------------------------------------ 3. robust

METRICS = {
    "sdc_any": ("any image changes (current metric)", lambda e: e.sdc.mean()),
    "sdc_ge3": (">= 3 of 200 images change", lambda e: (e.changed >= 3).mean()),
    "sdc_acc1pp": ("accuracy drops >= 1 pp", lambda e: (e.acc_drop >= 0.01).mean()),
    "mean_change": ("mean fraction of images changed", lambda e: e.change_rate.mean()),
}


def robust(seeds: list[int]) -> pd.DataFrame:
    rows = []
    for (w, s, fmt), e in campaigns().items():
        if s in seeds:
            rows.append(dict(width=w, seed=s, fmt=fmt,
                             **{k: fn(e) for k, (_, fn) in METRICS.items()}))
    t = pd.DataFrame(rows)
    have = sorted(int(s) for s in t.seed.unique())
    if have != sorted(seeds):
        print(f"note: finished campaigns found for seeds {have} only")

    for col, (desc, _) in METRICS.items():
        _section(f"{col}: {desc} -- mean over seeds {have} (seed max/min)")
        g = t.groupby(["fmt", "width"])[col]
        spread = g.agg(lambda x: x.max() / max(x.min(), 1e-4))
        cell = g.mean().map("{:.4f}".format) + " (" + spread.map("{:.1f}x".format) + ")"
        print(cell.unstack("width").to_string())
        lo, hi = t.width.min(), t.width.max()
        print(f"  endpoint w{hi}/w{lo} per seed:")
        for fmt in FORMATS:
            x = t[t.fmt == fmt].set_index(["seed", "width"])[col]
            ratios = [x[(s, hi)] / max(x[(s, lo)], 1e-9)
                      for s in have if (s, hi) in x and (s, lo) in x]
            print(f"    {fmt}: " + "  ".join(f"{r:.3f}x" for r in ratios))

    out = RESULTS / "e05c_tie_robust_metrics.csv"
    t.sort_values(["fmt", "width", "seed"]).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return t


# ------------------------------------------------------------------ 4. models

def models(names: list[str]) -> pd.DataFrame:
    """Rescore the single-model E01 campaigns behind F1 (ResNet8) and F17 (RepVGG-A0).

    Unlike ``robust`` these are model-wide, all sites and non-finite included, because
    F1 quotes model-wide rates.
    """
    import torch
    from mxfi.stats import binomial_rate

    X = _subset_images()
    rows = []
    for name in names:
        for fmt in FORMATS:
            p = RESULTS / f"e01_{name}_{fmt}-K32-ocp-w-n{N_INJECTIONS}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            with torch.no_grad():
                mg = _margins(_mx_model(None, 0, fmt, model=name).module(X))
            pert = d[~d.nonfinite]
            fails = d[d.sdc]
            rows.append(dict(
                model=name, fmt=fmt, n=len(d),
                min_margin=mg[0], n_below_0_05=int((mg < 0.05).sum()),
                n_below_0_2=int((mg < 0.2).sum()),
                sdc_any=binomial_rate(d.sdc), sdc_ge3=binomial_rate(d.changed >= 3),
                sdc_acc1pp=binomial_rate(d.acc_drop >= 0.01),
                nonfinite=binomial_rate(d.nonfinite),
                pert_any=binomial_rate(pert.sdc), pert_ge3=binomial_rate(pert.changed >= 3),
                single_image_share=(fails.changed == 1).mean() if len(fails) else np.nan,
                mean_change=d.change_rate.mean()))
    t = pd.DataFrame(rows)

    def fmt_est(e) -> str:
        return f"{e.rate:.4f} [{e.lo:.4f}, {e.hi:.4f}]"

    rates = ["sdc_any", "sdc_ge3", "sdc_acc1pp", "nonfinite", "pert_any", "pert_ge3"]
    for name, g in t.groupby("model", sort=False):
        _section(f"{name}: model-wide, n={int(g.n.iloc[0])} per format")
        print(g[["fmt", "min_margin", "n_below_0_05", "n_below_0_2",
                 "single_image_share", "mean_change"]].round(4).to_string(index=False))
        print()
        show = g.set_index("fmt")[rates].map(fmt_est).T
        print(show.to_string())
        gi = g.set_index("fmt")
        if {"e3m2", "e5m2"} <= set(gi.index):
            print("\n  e3m2 / e5m2 ratio (disjoint = 95% CIs do not overlap):")
            for c in rates:
                a, b = gi.at["e3m2", c], gi.at["e5m2", c]
                disj = "disjoint" if a.lo > b.hi or b.lo > a.hi else "overlap"
                print(f"    {c:<11} {a.rate / max(b.rate, 1e-9):6.2f}x   {disj}")

    out = RESULTS / "e05c_single_model_recheck.csv"
    flat = t.copy()
    for c in rates:
        flat[c] = t[c].map(lambda e: e.rate)
        flat[c + "_lo"] = t[c].map(lambda e: e.lo)
        flat[c + "_hi"] = t[c].map(lambda e: e.hi)
    flat.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("signflip", help="same sign flips under all formats on one model")
    a.add_argument("--width", type=int, default=32)
    a.add_argument("--train-seed", type=int, default=0)
    a.add_argument("--n", type=int, default=60)
    a.add_argument("--seed", type=int, default=1, help="RNG seed for fault positions")
    sub.add_parser("margins", help="golden near-tie margins vs SDC, every finished cell")
    r = sub.add_parser("robust", help="rescore CSVs under tie-robust metrics")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    m = sub.add_parser("models", help="rescore the single-model F1/F17 campaigns")
    m.add_argument("--names", nargs="+", default=["resnet8", "repvgg_a0"])
    args = p.parse_args()

    if args.cmd == "signflip":
        signflip(args.width, args.train_seed, args.n, args.seed)
    elif args.cmd == "margins":
        margins()
    elif args.cmd == "models":
        models(args.names)
    else:
        robust(args.seeds)


if __name__ == "__main__":
    main()
