"""Figures for the report, generated from the raw campaign results.

Four figures, each chosen for the job its data does:

1. per-bit vulnerability from the exhaustive campaign -- magnitude across an
   ordered category, so bars, one series per panel;
2. failure rate against model capacity -- change along a continuous variable,
   so lines, one per format, with every seed drawn faintly behind the mean;
3. per-layer vulnerability against layer size -- a relationship between two
   quantities, so a scatter with the extremes labelled;
4. how well each severity measure ranks failures -- ROC curves.

Colours come from the validated categorical palette (slots 1-3, plus violet for
the fourth ROC series); the aqua slot sits below 3:1 against the surface, so
every series carries a direct label as relief. Text never wears a series colour.

Run::

    PYTHONPATH=. python -m experiments.make_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "figures"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
FMT_COLOR = {"e3m2": BLUE, "e4m3": ORANGE, "e5m2": AQUA}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
})


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def fig_per_bit(d: pd.DataFrame) -> None:
    """Exhaustive per-bit rates: one panel per fault site, one series each."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, site, label in zip(axes, ("element", "scale"),
                               ("element code", "shared E8M0 scale")):
        g = d[d.site == site].groupby("bit").sdc.mean()
        bars = ax.bar(g.index, g.values, color=BLUE, width=0.68)
        for b in bars:
            b.set_linewidth(0)
        ax.set_title(label, loc="left", color=INK)
        ax.set_xlabel("bit position (0 = least significant)")
        ax.set_ylim(0, 1.22)
        ax.set_xticks(range(8))
        _clean(ax)
        # label only the positions worth naming; a value alone where the bar is narrow
        note = {"element": [(7, "sign bit")], "scale": [(3, ""), (7, "")]}
        for bit, text in note[site]:
            if bit in g.index:
                txt = f"{text}\n{g[bit]:.3f}" if text else f"{g[bit]:.3f}"
                ax.annotate(txt, (bit, g[bit]), textcoords="offset points",
                            xytext=(0, 4), ha="center", va="bottom",
                            fontsize=7, color=INK2)
    axes[0].set_ylabel("SDC rate (exhaustive)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_per_bit.pdf", bbox_inches="tight")
    plt.close(fig)


def capacity_frame() -> pd.DataFrame | None:
    """Per-inference element rate against capacity, per seed, from the E19 rescore.

    The width-sweep summary only kept the any-image rate, which F52 showed is
    governed by the evaluation set; the rescored table carries the per-inference
    rate for every seed of every width, so the figure is built from that instead.
    """
    resc, sweep = RESULTS / "e19_per_inference_rates.csv", RESULTS / "width_sweep_multiseed.csv"
    if not (resc.exists() and sweep.exists()):
        return None
    d = pd.read_csv(resc)
    d = d[(d.exp == "e01") & (d.site == "element") & (d.K == 32)
          & (d.scale_mode == "ocp") & d.model.str.match(r"resnet8_w\d+$")]
    d = d.assign(width=d.model.str.extract(r"_w(\d+)$").astype(int))
    params = (pd.read_csv(sweep).groupby("width").params.first())
    d = d.join(params, on="width").dropna(subset=["params"])
    return d.rename(columns={"per_inference": "rate"})


def fig_capacity(w: pd.DataFrame) -> None:
    """Per-inference element rate against parameter count, every seed per width."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for fmt in ("e3m2", "e4m3", "e5m2"):
        g = w[w.fmt == fmt]
        for _, s in g.groupby("seed"):                     # every seed, faint
            s = s.sort_values("params")
            ax.plot(s.params, s.rate, color=FMT_COLOR[fmt],
                    lw=0.6, alpha=0.25, zorder=1)
        m = g.groupby("params").rate.mean().sort_index()
        ax.plot(m.index, m.values, color=FMT_COLOR[fmt], lw=2, zorder=2, label=fmt)
        ax.annotate(fmt, (m.index[-1], m.values[-1]), textcoords="offset points",
                    xytext=(5, -1), fontsize=7.5, color=INK2, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("inferences corrupted per element fault")
    ax.set_title("Capacity helps least where\nNaN/Inf codes exist",
                 loc="left", color=INK)
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
    ax.legend(loc="lower left")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_capacity.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_layers(d: pd.DataFrame) -> None:
    """Per-layer vulnerability against how much storage the layer occupies."""
    g = d.groupby("tensor").agg(sdc=("sdc", "mean"), bits=("sdc", "size"))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.scatter(g.bits, g.sdc, s=34, color=BLUE, zorder=3, linewidth=0)
    for name, ha, dx in ((g.sdc.idxmax(), "left", 6), (g.sdc.idxmin(), "right", -6)):
        r = g.loc[name]
        ax.annotate(name, (r.bits, r.sdc), textcoords="offset points",
                    xytext=(dx, 6), ha=ha, fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("bits of storage in the layer")
    ax.set_ylabel("SDC rate (exhaustive)")
    ax.set_title("The most fragile layer is the smallest", loc="left", color=INK)
    ax.set_ylim(0, 1.0)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_layers.pdf", bbox_inches="tight")
    plt.close(fig)


def _roc(score: np.ndarray, label: np.ndarray):
    order = np.argsort(-score)
    y = label[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return np.r_[0, fpr], np.r_[0, tpr], auc


def fig_severity(s: pd.DataFrame) -> None:
    """How well each severity measure ranks the faults that actually failed."""
    series = [("log_severity", VIOLET, "the plan's measure"),
              ("rel_error", AQUA, None), ("delta_rms", ORANGE, None),
              ("margin_shift", BLUE, None)]
    y = s.sdc.to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("chance", (0.62, 0.58), fontsize=7, color=MUTED, rotation=34)
    for name, color, note in series:
        x = s[name].to_numpy()
        ok = np.isfinite(x)
        fpr, tpr, auc = _roc(x[ok], y[ok])
        ax.plot(fpr, tpr, color=color, lw=2, zorder=2,
                label=f"{name}  AUC {auc:.3f}")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("Does the severity measure find the failures?", loc="left", color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_severity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    ex = RESULTS / "e06.csv.gz"
    if ex.exists():
        d = pd.read_csv(ex)
        fig_per_bit(d); made.append("fig1_per_bit")
        fig_layers(d); made.append("fig3_layers")

    cap = capacity_frame()
    if cap is not None and len(cap):
        fig_capacity(cap); made.append("fig2_capacity")

    sv = RESULTS / "e12_vit_small_severity_metrics.csv"
    if sv.exists():
        fig_severity(pd.read_csv(sv)); made.append("fig4_severity")

    print("wrote " + ", ".join(f"docs/figures/{m}.pdf" for m in made))


if __name__ == "__main__":
    main()
