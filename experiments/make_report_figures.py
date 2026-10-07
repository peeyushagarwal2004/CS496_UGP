"""Extended figures (fig5 onwards), used in the README and docs/figures.

`make_figures.py` draws the four report figures; this script draws the rest from
the same raw results, with the same style, palette and conventions:

* rates are per inference unless a panel says otherwise (the any-image rate only
  appears where the point of the panel is to show what is wrong with it);
* formats keep one colour everywhere -- e3m2 blue, e4m3 orange, e5m2 aqua, e2m3
  yellow, e2m1 magenta (palette slots 1-5, validated in that order); three of those
  sit below 3:1 on white, so every series carries a direct label or a legend;
* text is never drawn in a series colour.

Run::

    PYTHONPATH=. python -m experiments.make_report_figures
"""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from experiments.make_figures import (AQUA, BLUE, FIGS, INK, INK2, MUTED, ORANGE,
                                      RESULTS, ROOT, VIOLET, _clean)
from mxfi.formats import ELEMENT_FORMATS

YELLOW, MAGENTA = "#eda100", "#e87ba4"
FMT = {"e3m2": BLUE, "e4m3": ORANGE, "e5m2": AQUA, "e2m3": YELLOW, "e2m1": MAGENTA}
FMT_ORDER = ["e3m2", "e2m3", "e2m1", "e4m3", "e5m2"]
MODEL_NAME = {"resnet8": "ResNet8", "resnet8_w16": "ResNet8", "repvgg_a0": "RepVGG-A0",
              "vit_small": "ViT"}
GRID = "#e6e5e2"


def save(fig, name: str) -> str:
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "png" / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return name


def label_end(ax, x, y, text, dx=4, dy=0, ha="left"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=7, color=INK2, va="center", ha=ha)


# ------------------------------------------------------------------ background

def fig_alphabets():
    """Every finite positive value each element format can hold, on a log axis."""
    fig, ax = plt.subplots(figsize=(6.6, 2.1))
    rows = ["e2m1", "e2m3", "e3m2", "e4m3", "e5m2"]
    for y, name in enumerate(rows):
        f = ELEMENT_FORMATS[name]
        t = f.table()[: f.n_codes // 2]
        vals = t[np.isfinite(t) & (t > 0)]
        ax.scatter(vals, np.full(vals.size, y), marker="|", s=60, lw=1.1,
                   color=FMT[name], zorder=3)
        n_special = int((~np.isfinite(f.table())).sum())
        note = f"{f.n_codes} codes, {n_special} NaN/Inf" if n_special else f"{f.n_codes} codes, none special"
        ax.annotate(note, (vals.max(), y), textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=7, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yticks(range(len(rows)), [f"{r}  ({ELEMENT_FORMATS[r].width} bit)" for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(2 ** -18, 2 ** 22)
    ax.set_xlabel("representable magnitude at unit scale (log$_2$ axis)")
    ax.set_title("Element alphabets: range comes from the shared scale, so wide "
                 "exponents buy little", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    _clean(ax)
    return save(fig, "fig5_alphabets")


def fig_training():
    """Test accuracy per epoch of the seed-0 network of each architecture."""
    runs = {
        "CIFAR-10": {"ResNet8": "checkpoints/resnet8_cifar10.log.json",
                     "RepVGG-A0": "cluster/checkpoints/repvgg_a0_cifar10.log.json",
                     "ViT": "cluster/checkpoints/vit_small_cifar10.log.json"},
        "CIFAR-100": {"ResNet8": "cluster/checkpoints/resnet8_cifar100.log.json",
                      "RepVGG-A0": "cluster/checkpoints/repvgg_a0_cifar100.log.json",
                      "ViT": "cluster/checkpoints/vit_small_cifar100.log.json"},
    }
    colors = {"ResNet8": BLUE, "RepVGG-A0": ORANGE, "ViT": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=False)
    for ax, (ds, models) in zip(axes, runs.items()):
        for name, path in models.items():
            p = ROOT / path
            if not p.exists():
                continue
            log = pd.DataFrame(json.loads(p.read_text()))
            ax.plot(log.epoch + 1, log.test_acc * 100, color=colors[name], lw=1.6)
            label_end(ax, log.epoch.iloc[-1] + 1, log.test_acc.iloc[-1] * 100,
                      f"{name} {log.test_acc.iloc[-1] * 100:.1f}%")
        ax.set_title(ds, loc="left", color=INK)
        ax.set_xlabel("epoch")
        ax.set_xlim(right=ax.get_xlim()[1] * 1.45)
        _clean(ax)
    axes[0].set_ylabel("FP32 test accuracy (%)")
    fig.tight_layout()
    return save(fig, "fig6_training")


def _accuracy(model: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / f"e00_{model}_quantized_accuracy.csv")
    d = d[(d.block_size == 32) & (d.scale_mode == "ocp")].copy()
    d["fp32"] = d.accuracy + d.drop_vs_fp32
    return d


def fig_accuracy():
    """Quantised accuracy per format against the FP32 baseline, both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharex=False)
    for ax, suffix, ds in zip(axes, ("", "_c100"), ("CIFAR-10", "CIFAR-100")):
        for y, model in enumerate(("resnet8", "repvgg_a0", "vit_small")):
            d = _accuracy(model + suffix)
            fp = d.fp32.iloc[0] * 100
            ax.plot([fp, fp], [y - 0.3, y + 0.3], color=INK, lw=1.2, zorder=2)
            for _, r in d.iterrows():
                ax.scatter(r.accuracy * 100, y, s=30, color=FMT[r.fmt], zorder=3,
                           edgecolor="white", linewidth=0.8)
        ax.set_yticks(range(3), ["ResNet8", "RepVGG-A0", "ViT"])
        ax.set_ylim(-0.6, 2.6)
        ax.invert_yaxis()
        ax.set_title(ds, loc="left", color=INK)
        ax.set_xlabel("top-1 accuracy (%), K = 32, OCP scale")
        ax.grid(axis="y", visible=False)
        _clean(ax)
    handles = [Line2D([], [], marker="o", ls="", color=FMT[f], label=f) for f in FMT_ORDER]
    handles.append(Line2D([], [], marker="|", ls="", color=INK, markersize=9, label="FP32"))
    axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return save(fig, "fig7_accuracy")


# --------------------------------------------------------------- O1/O2 results

def fig_acc_vs_vuln():
    """Same accuracy, orders of magnitude apart: accuracy against per-inference rate."""
    r = pd.read_csv(RESULTS / "e19_per_inference_rates.csv")
    r = r[(r.exp == "e01") & (r.seed == 0) & (r.site == "element") & (r.K == 32)
          & (r.scale_mode == "ocp")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
    for ax, model in zip(axes, ("resnet8", "repvgg_a0", "vit_small")):
        acc = _accuracy(model).set_index("fmt").accuracy * 100
        g = r[r.model == model].set_index("fmt")
        acc = acc[acc.index.isin(g.index)]
        for fmt in [f for f in FMT_ORDER if f in g.index]:
            rate = max(g.loc[fmt, "per_inference"], 5e-6)
            ax.scatter(acc[fmt], rate, s=34, color=FMT[fmt], zorder=3,
                       edgecolor="white", linewidth=0.8)
            label_end(ax, acc[fmt], rate, fmt, dx=6)
        lo, hi = acc.min(), acc.max()
        ax.set_xlim(lo - 1.2, hi + 1.8)
        ax.set_yscale("log")
        ax.set_title(MODEL_NAME[model], loc="left", color=INK)
        ax.set_xlabel("accuracy (%)")
        _clean(ax)
    axes[0].set_ylabel("inferences corrupted\nper element fault")
    fig.suptitle("Within about a point of accuracy, fault vulnerability spans up to "
                 "three orders of magnitude", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    return save(fig, "fig8_acc_vs_vuln")


def fig_format_order():
    """Element per-inference rate by format, every architecture, both datasets."""
    d = pd.read_csv(RESULTS / "e24_cifar100_vs_cifar10.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharex=True)
    models = ["resnet8_w16", "repvgg_a0", "vit_small"]
    offs = {"e3m2": -0.2, "e4m3": 0.0, "e5m2": 0.2}
    for ax, ds in zip(axes, ("CIFAR-10", "CIFAR-100")):
        g = d[d.dataset == ds]
        for y, m in enumerate(models):
            for fmt, dy in offs.items():
                row = g[(g.pair == m) & (g.fmt == fmt)].iloc[0]
                lo = max(row.el_min, 1e-6)
                ax.plot([lo, row.el_max], [y + dy] * 2, color=FMT[fmt], lw=1.2, zorder=2)
                ax.scatter(max(row.element, 1e-6), y + dy, s=26, color=FMT[fmt], zorder=3,
                           edgecolor="white", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_yticks(range(3), [MODEL_NAME[m] for m in models])
        ax.set_ylim(-0.5, 2.5)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(ds, loc="left", color=INK)
        ax.set_xlabel("inferences corrupted per element fault")
        _clean(ax)
    handles = [Line2D([], [], marker="o", ls="", color=FMT[f], label=f) for f in offs]
    axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return save(fig, "fig9_format_order")


def fig_scale_vs_element():
    """Element against scale damage per inference, every network and format."""
    d = pd.read_csv(RESULTS / "e24_cifar100_vs_cifar10.csv")
    d = d[d.fmt.isin(["e3m2", "e4m3", "e5m2"])]
    rows = []
    for ds in ("CIFAR-10", "CIFAR-100"):
        for m in ("resnet8_w16", "repvgg_a0", "vit_small"):
            for fmt in ("e3m2", "e4m3", "e5m2"):
                r = d[(d.dataset == ds) & (d.pair == m) & (d.fmt == fmt)].iloc[0]
                rows.append((f"{MODEL_NAME[m]} {fmt}", ds, r.element, r.scale))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    for ax, ds in zip(axes, ("CIFAR-10", "CIFAR-100")):
        sub = [r for r in rows if r[1] == ds]
        for y, (name, _, el, sc) in enumerate(sub):
            el = max(el, 1e-6)
            ax.plot([el, sc], [y, y], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
            ax.scatter(el, y, s=26, color=BLUE, zorder=3, edgecolor="white", linewidth=0.7)
            ax.scatter(sc, y, s=26, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.7)
            label_end(ax, sc, y, f"{sc / el:,.0f}x" if sc / el >= 10 else f"{sc / el:.1f}x", dx=5)
        ax.set_yticks(range(len(sub)), [s[0] for s in sub])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(right=3)
        ax.grid(axis="y", visible=False)
        ax.set_title(ds, loc="left", color=INK)
        ax.set_xlabel("inferences corrupted per fault")
        _clean(ax)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, label="element fault"),
               Line2D([], [], marker="o", ls="", color=ORANGE, label="scale fault")]
    axes[0].legend(handles=handles, loc="lower left")
    fig.tight_layout()
    return save(fig, "fig10_scale_vs_element")


# ----------------------------------------------------------------- O3 results

def fig_block_size():
    r = pd.read_csv(RESULTS / "e19_per_inference_rates.csv")
    blk = pd.read_csv(RESULTS / "e03_resnet8_block_size.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    Ks = [8, 16, 32, 64]

    ax = axes[0]
    share = blk[blk.fmt == "e4m3"].set_index("K").scale_bit_share * 100
    ax.plot(Ks, share.loc[Ks], color=INK2, lw=1.6, marker="o", ms=4)
    for k in Ks:
        ax.annotate(f"{share[k]:.1f}%", (k, share[k]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color=INK2)
    ax.set_ylabel("scale bits, share of storage (%)")
    ax.set_title("(a) scale storage", loc="left", color=INK)
    ax.set_ylim(0, 14)

    e03 = r[r.exp == "e03"]
    for ax, site, title in ((axes[1], "scale", "(b) per scale fault"),
                            (axes[2], "element", "(c) per element fault")):
        for fmt in ("e4m3", "e2m1"):
            g = e03[(e03.fmt == fmt) & (e03.site == site)].set_index("K").per_inference.loc[Ks]
            ax.plot(Ks, g.values, color=FMT[fmt], lw=1.8, marker="o", ms=4)
            label_end(ax, 64, g.loc[64], fmt, dx=6)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(bottom=0)
    axes[1].set_ylabel("inferences corrupted per fault")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ks, [str(k) for k in Ks])
        ax.set_xlim(6.5, 100)
        ax.set_xlabel("block size K")
        _clean(ax)
    fig.tight_layout()
    return save(fig, "fig11_block_size")


def fig_block_rule():
    """Matched element faults across K under the OCP and the non-clipping rule."""
    r = pd.read_csv(RESULTS / "e19_per_inference_rates.csv")
    r = r[(r.exp == "e03b") & (r.site == "element")]
    Ks = [8, 16, 32, 64]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for ax, model in zip(axes, ("resnet8", "repvgg_a0")):
        for mode, color, label in (("ocp", ORANGE, "OCP rule"), ("fit", BLUE, "non-clipping")):
            g = r[(r.model == model) & (r.scale_mode == mode)].set_index("K").per_inference.loc[Ks]
            ax.plot(Ks, g.values, color=color, lw=1.8, marker="o", ms=4, label=label)
        ax.legend(loc="upper right")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ks, [str(k) for k in Ks])
        ax.set_xlim(6.5, 80)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{MODEL_NAME[model]}, e4m3", loc="left", color=INK)
        ax.set_xlabel("block size K")
        _clean(ax)
    axes[0].set_ylabel("inferences corrupted\nper element fault")
    fig.tight_layout()
    return save(fig, "fig12_block_rule")


# ------------------------------------------------------------ metric findings

def fig_eval_sets():
    """One fault list, nine image sets: the two metrics disagree about the formats."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.8), sharex=True)
    for col, (seed, net) in enumerate(((0, "network A"), (9, "network B"))):
        d = pd.read_csv(RESULTS / f"e18_resnet8_w16_s{seed}_eval_set_sensitivity.csv")
        for row, (metric, title) in enumerate((("sdc", "any-image SDC rate"),
                                              ("change_rate", "per-inference rate"))):
            ax = axes[row, col]
            for fmt in ("e3m2", "e4m3"):
                g = d[d.fmt == fmt].sort_values("fold")
                ax.plot(g.fold, g[metric], color=FMT[fmt], lw=1.6, marker="o", ms=4)
                label_end(ax, g.fold.iloc[-1], g[metric].iloc[-1], fmt, dx=6)
            ax.set_title(f"{title}, {net}", loc="left", color=INK)
            ax.set_ylim(bottom=0)
            ax.set_xlim(-0.4, 9.2)
            _clean(ax)
    for ax in axes[1]:
        ax.set_xlabel("evaluation set (200 images each)")
        ax.set_xticks(range(9))
    axes[0, 0].set_ylabel("faults that change\nany prediction")
    axes[1, 0].set_ylabel("inferences corrupted\nper fault")
    fig.tight_layout()
    return save(fig, "fig13_eval_sets")


def fig_near_ties():
    d = pd.read_csv(RESULTS / "e05c_golden_margins.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    rng = np.random.default_rng(0)
    for fmt in ("e3m2", "e4m3", "e5m2"):
        g = d[d.fmt == fmt]
        x = g.n_below_0_2 + rng.uniform(-0.18, 0.18, len(g))
        ax.scatter(x, g.pert_sdc, s=20, color=FMT[fmt], label=fmt, zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.set_xlabel("evaluation images with golden margin < 0.2")
    ax.set_ylabel("any-image SDC rate\n(perturbation faults)")
    ax.set_title("The any-image rate counts near-ties\n(39 networks, Spearman +0.89)",
                 loc="left", color=INK)
    ax.legend(loc="lower right")
    _clean(ax)
    fig.tight_layout()
    return save(fig, "fig14_near_ties")


def fig_zero_inflation():
    """Distribution of per-fault damage over both exhaustive campaigns (element faults)."""
    sets = (("e3m2", RESULTS / "e06_resnet8_w16_e3m2-K32-ocp-w_exhaustive.csv"),
            ("e4m3", RESULTS / "e06.csv.gz"))
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    edges = np.array([0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 200, 201])
    centers = np.arange(len(edges) - 1)
    width = 0.38
    for i, (fmt, path) in enumerate(sets):
        if not path.exists():
            continue
        d = pd.read_csv(path, usecols=["site", "changed", "nonfinite"])
        d = d[d.site == "element"]
        h, _ = np.histogram(d.changed, bins=edges)
        share = h / len(d)
        ax.bar(centers + (i - 0.5) * width, share, width=width * 0.92, color=FMT[fmt],
               label=f"{fmt}: {share[0] * 100:.1f}% of faults corrupt nothing", linewidth=0)
    labels = ["0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-199", "200"]
    ax.set_xticks(centers, labels)
    ax.set_yscale("log")
    ax.set_xlabel("predictions changed by one element fault (of 200)")
    ax.set_ylabel("share of all element faults")
    ax.set_title("Exhaustive campaigns: damage is zero-inflated, with a spike at 'every "
                 "inference' from NaN faults", loc="left", color=INK)
    ax.legend(loc="upper center")
    ax.grid(axis="x", visible=False)
    _clean(ax)
    fig.tight_layout()
    return save(fig, "fig15_zero_inflation")


# ------------------------------------------------------------------ sampling

def fig_sampling():
    arms = [("learned tree (value)", "learned tree"),
            ("learned tree (value+layer), proportional", "learned tree, proportional"),
            ("fixed (F46's cuts)", "margin_shift, fixed cuts"),
            ("exact |dv| enumeration", "exact |dv| enumeration"),
            ("exact margin_shift", "exact margin_shift"),
            ("score quantiles", "margin_shift, quantile strata")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, fmt in zip(axes, ("e3m2", "e4m3")):
        e22 = pd.read_csv(RESULTS / f"e22_resnet8_w16_{fmt}_learned_vs_exact.csv")
        e20 = pd.read_csv(RESULTS / f"e20_resnet8_w16_{fmt}_value_aware_per_inference.csv")
        vals = []
        for key, label in arms:
            if key in set(e22.arm):
                g = e22[e22.arm == key]
            else:
                g = e20[e20.scheme == key].rename(columns={"scheme": "arm"})
            v = g.set_index("budget").variance_ratio
            vals.append((label, v.get(3000, np.nan), v.min(), v.max()))
        for y, (label, mid, lo, hi) in enumerate(vals):
            good = mid >= 1
            ax.plot([lo, hi], [y, y], color=BLUE if good else ORANGE, lw=1.4, zorder=2)
            ax.scatter(mid, y, s=28, color=BLUE if good else ORANGE, zorder=3,
                       edgecolor="white", linewidth=0.7)
            label_end(ax, hi, y, f"{mid:.2g}x" if mid < 10 else f"{mid:.0f}x", dx=5)
        ax.axvline(1, color=MUTED, lw=1)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 600)
        ax.set_yticks(range(len(arms)), [a[1] for a in arms])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"ResNet8, {fmt}", loc="left", color=INK)
        ax.set_xlabel("variance reduction vs uniform")
        _clean(ax)
    fig.tight_layout()
    return save(fig, "fig16_sampling")


# ----------------------------------------------------------------- multi-bit

def fig_multibit():
    d = pd.read_csv(RESULTS / "e21_multibit_summary.csv")
    d["net"] = d.campaign.str.extract(r"_(s\d+)_")[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    rows = [(f, s) for s in ("element", "scale") for f in ("e3m2", "e4m3", "e5m2")]
    for y, (fmt, site) in enumerate(rows):
        g = d[(d.fmt == fmt) & (d.site == site)]
        ax.scatter(g.ratio_2_to_1, [y - 0.12] * len(g), s=24, color=BLUE, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(g.ratio_adj_to_1, [y + 0.12] * len(g), s=24, color=ORANGE, zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [f"{f} {s}" for f, s in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.7, 1.75)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("damage relative to a single flip")
    ax.set_title("(a) a second flip in the same word", loc="left", color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="random pair"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="adjacent pair")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    _clean(ax)

    ax = axes[1]
    for y, (fmt, site) in enumerate(rows):
        g = d[(d.fmt == fmt) & (d.site == site)]
        ax.scatter(g.p_at_1pct, [y - 0.12] * len(g), s=24, color=BLUE, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(g.p_at_10pct, [y + 0.12] * len(g), s=24, color=ORANGE, zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.axvspan(1e-8, 1e-6, color=GRID, zorder=0)
    ax.annotate("one fault per\ninference holds", (1.3e-8, 2.5), fontsize=7,
                color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 0.1)
    ax.set_yticks(range(len(rows)), [f"{f} {s}" for f, s in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("per-bit flip probability p")
    ax.set_title("(b) p at which doubles reach a share", loc="left", color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="1% of damage"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="10% of damage")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    _clean(ax)
    fig.tight_layout()
    return save(fig, "fig17_multibit")


# ------------------------------------------------------------------ NaN guard

def fig_nan_guard():
    d = pd.read_csv(RESULTS / "e23_nan_guard_summary.csv")
    pat = re.compile(r"^(resnet8|repvgg_a0|vit_small)(_w16)?(_c100)?(_s\d+)?_(e\dm\d)")
    parts = d.run.str.extract(pat)
    d["model"], d["c100"], d["fmt"] = parts[0], parts[2].notna(), parts[4]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, c100, title in ((axes[0], False, "CIFAR-10"), (axes[1], True, "CIFAR-100")):
        rows = [(m, f) for m in ("resnet8", "repvgg_a0", "vit_small") for f in ("e4m3", "e5m2")]
        for y, (m, f) in enumerate(rows):
            g = d[(d.model == m) & (d.fmt == f) & (d.c100 == c100)]
            for _, r in g.iterrows():
                ax.plot([r.guarded_upper95, r.element_rate_unguarded], [y, y], color=GRID,
                        lw=1.6, zorder=1)
            ax.scatter(g.element_rate_unguarded, [y] * len(g), s=22, color=ORANGE, zorder=3,
                       edgecolor="white", linewidth=0.6)
            ax.scatter(g.guarded_upper95, [y] * len(g), s=22, color=BLUE, zorder=3,
                       edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(rows)), [f"{MODEL_NAME[m]} {f}" for m, f in rows])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", color=INK)
        _clean(ax)
    fig.supxlabel("inferences corrupted per element fault (each dot one trained network)",
                  fontsize=8, color=INK, x=0.42)
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label="unguarded"),
                            Line2D([], [], marker="o", ls="", color=BLUE,
                                   label="guarded (95% upper bound)")],
                   loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return save(fig, "fig18_nan_guard")


# ---------------------------------------------------------------- activations

def fig_activations():
    d = pd.read_csv(RESULTS / "e04_corrected_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, site in zip(axes, ("element", "scale")):
        g = d[d.site == site]
        for _, r in g.iterrows():
            c100 = r.model.endswith("_c100")
            ax.errorbar(r.weight, r.act, yerr=[[r.act - r.act_lo], [r.act_hi - r.act]],
                        fmt="none", ecolor=FMT[r.fmt], elinewidth=0.9, alpha=0.6, zorder=2)
            ax.scatter(r.weight, r.act, s=30, color=FMT[r.fmt], zorder=3,
                       marker="s" if c100 else "o", edgecolor="white", linewidth=0.6)
        lims = np.array([min(g.weight.min(), g.act_lo.min()) * 0.7,
                         max(g.weight.max(), g.act_hi.max()) * 1.4])
        ax.plot(lims, lims, color=MUTED, lw=1)
        ax.fill_between(lims, lims / 2, lims * 2, color=GRID, alpha=0.6, zorder=0, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lims); ax.set_ylim(lims)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("weight fault, per inference")
        ax.set_title(f"{site} faults", loc="left", color=INK)
        _clean(ax)
    axes[0].set_ylabel("activation fault, per inference")
    axes[0].annotate("band: within 2x", (0.05, 0.9), xycoords="axes fraction",
                     fontsize=7, color=INK2)
    hs = [Line2D([], [], marker="o", ls="", color=FMT[f], label=f) for f in ("e4m3", "e5m2")]
    hs += [Line2D([], [], marker="o", ls="", color=INK2, label="CIFAR-10"),
           Line2D([], [], marker="s", ls="", color=INK2, label="CIFAR-100")]
    axes[1].legend(handles=hs, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return save(fig, "fig19_activations")


# -------------------------------------------------------------- CIFAR-100

def fig_cifar100():
    d = pd.read_csv(RESULTS / "e24_cifar100_vs_cifar10.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
    for ax, m in zip(axes, ("resnet8_w16", "repvgg_a0", "vit_small")):
        g = d[d.pair == m]
        ends = []
        for fmt in [f for f in FMT_ORDER if f in set(g.fmt)]:
            s = g[g.fmt == fmt].set_index("dataset").element
            y = [max(s["CIFAR-10"], 1e-6), max(s["CIFAR-100"], 1e-6)]
            ax.plot([0, 1], y, color=FMT[fmt], lw=1.6, marker="o", ms=4)
            ends.append([np.log10(y[1]), fmt, y[1]])
        # spread end labels that would collide (min 0.3 decades apart)
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.3)
        for ly, fmt, y1 in ends:
            ax.annotate(fmt, (1, y1), xytext=(1.08, 10 ** ly), textcoords="data",
                        fontsize=7, color=INK2, va="center")
        ax.set_xticks([0, 1], ["CIFAR-10", "CIFAR-100"])
        ax.set_xlim(-0.25, 1.55)
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
        ax.set_title(MODEL_NAME[m], loc="left", color=INK)
        _clean(ax)
    axes[0].set_ylabel("inferences corrupted\nper element fault")
    fig.suptitle("A harder task raises the NaN-free formats; the NaN-coded ones barely move",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    return save(fig, "fig20_cifar100")


# ---------------------------------------------------------------- mechanism

def fig_sensitivity():
    d = pd.read_csv(RESULTS / "e11_vit_small_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    for _, r in d.iterrows():
        for delta, mk in (("0.5", "o"), ("1.5", "s"), ("3.0", "D")):
            ax.scatter(r[f"pred_{delta}"], r[f"meas_{delta}"], s=24, marker=mk,
                       color=FMT[r.fmt], zorder=3, edgecolor="white", linewidth=0.6)
    lim = [0, 0.16]
    ax.plot(lim, lim, color=MUTED, lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted from sensitivity s")
    ax.set_ylabel("measured flip rate")
    ax.set_title("(a) probe: one weight moved by\n0.5 / 1.5 / 3 rms (r = +0.998)",
                 loc="left", color=INK)
    hs = [Line2D([], [], marker="o", ls="", color=FMT[f], label=f) for f in FMT_ORDER]
    ax.legend(handles=hs, loc="upper left")
    _clean(ax)

    ax = axes[1]
    pert = pd.read_csv(RESULTS / "e19_per_inference_rates.csv")
    pert = pert[(pert.model == "vit_small") & (pert.site == "element_pert")
                & (pert.exp == "e01")].groupby("fmt").per_inference.mean()
    for _, r in d.iterrows():
        ax.scatter(r.median_s, max(pert[r.fmt], 1e-6), s=34, color=FMT[r.fmt], zorder=3,
                   edgecolor="white", linewidth=0.7)
        if r.fmt == "e5m2":          # sits on top of e3m2: one label for both
            continue
        name = "e3m2, e5m2" if r.fmt == "e3m2" else r.fmt
        label_end(ax, r.median_s, max(pert[r.fmt], 1e-6),
                  f"{name} ({ELEMENT_FORMATS[r.fmt].man_bits} mantissa bits)", dx=6)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(NullFormatter())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.008, 0.6)
    ax.set_xlabel("median sensitivity s of the quantised network")
    ax.set_ylabel("perturbation damage\nper element fault")
    ax.set_title("(b) sensitivity, not mantissa width,\ntracks perturbation damage (ViT)",
                 loc="left", color=INK)
    _clean(ax)
    fig.tight_layout()
    return save(fig, "fig21_sensitivity")


def fig_nonfinite_width():
    d = pd.read_csv(RESULTS / "width_sweep_multiseed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, col, title, ylab in (
            (axes[0], "nonfinite_rate", "(a) NaN/Inf rate is a format property",
             "element faults giving NaN/Inf"),
            (axes[1], "nonfinite_share", "(b) ...so its share of failures grows",
             "NaN/Inf share of failures")):
        for fmt in ("e4m3", "e5m2"):
            g = d[d.fmt == fmt]
            for _, s in g.groupby("seed"):
                s = s.sort_values("params")
                ax.plot(s.params, s[col] * 100, color=FMT[fmt], lw=0.6, alpha=0.25)
            m = g.groupby("params")[col].mean() * 100
            ax.plot(m.index, m.values, color=FMT[fmt], lw=2)
            label_end(ax, m.index[-1], m.values[-1], fmt, dx=6)
        ax.set_xscale("log")
        ax.set_xlim(right=ax.get_xlim()[1] * 2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("parameters (ResNet8, widths 8-48)")
        ax.set_ylabel(ylab + " (%)")
        ax.set_title(title, loc="left", color=INK)
        _clean(ax)
    fig.tight_layout()
    return save(fig, "fig22_nonfinite_width")


def main() -> None:
    (FIGS / "png").mkdir(parents=True, exist_ok=True)
    made = [f() for f in (fig_alphabets, fig_training, fig_accuracy, fig_acc_vs_vuln,
                          fig_format_order, fig_scale_vs_element, fig_block_size,
                          fig_block_rule, fig_eval_sets, fig_near_ties, fig_zero_inflation,
                          fig_sampling, fig_multibit, fig_nan_guard, fig_activations,
                          fig_cifar100, fig_sensitivity, fig_nonfinite_width)]
    print("wrote " + ", ".join(made))


if __name__ == "__main__":
    main()
