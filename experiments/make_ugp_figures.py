"""Figures for the UGP report (docs/ugp_report).

Run::

    PYTHONPATH=. python -m experiments.make_ugp_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mxfi.cases import label

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "ugp_report" / "fig"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def short(case: str) -> str:
    return label(case.replace("scale:", ""))


def case_damage() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, fmt in zip(axes, ["e3m2", "e4m3"]):
        d = pd.read_csv(RESULTS / f"e25_{fmt}_labelled.csv.gz",
                        usecols=["case", "damage"], low_memory=False)
        d["label"] = d.case.map(short)
        g = d.groupby("label").damage.agg(["sum", "size"])
        g["damage_share"] = g["sum"] / g["sum"].sum()
        g["fault_share"] = g["size"] / g["size"].sum()
        g = g.sort_values("damage_share", ascending=False).head(8)[::-1]
        y = np.arange(len(g))
        ax.barh(y, g.damage_share * 100, color=BLUE, height=0.62)
        for yi, (ds, fs) in enumerate(zip(g.damage_share, g.fault_share)):
            ax.text(ds * 100 + 1.5, yi, f"{ds:.0%} of damage / {fs:.1%} of faults",
                    va="center", fontsize=7, color=INK)
        ax.set_yticks(y, g.index)
        ax.set_xlim(0, 105)
        ax.set_xlabel("share of total damage (%)")
        ax.set_title(f"{fmt}, ResNet8 (exhaustive)", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "case_damage.pdf")
    plt.close(fig)


def e29_guards() -> None:
    p = RESULTS / "e29_summary.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    names = {"resnet8_w16": "ResNet8\nCIFAR-10", "resnet8_c100": "ResNet8\nCIFAR-100",
             "repvgg_a0": "RepVGG-A0\nCIFAR-10"}
    t["x"] = t.model.map(names) + "\n" + t.fmt
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(t))
    w = 0.26
    series = [("rate", "no protection", BLUE), ("rate_guard", "guards G1+G2", ORANGE),
              ("rate_relu", "NaN-zeroing ReLU", AQUA)]
    for k, (col, name, c) in enumerate(series):
        v = t[col].to_numpy()
        ax.bar(x + (k - 1) * w, v, w - 0.03, color=c, label=name)
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi * 1.12, f"{vi:.2g}", ha="center", va="bottom",
                    fontsize=5.5, rotation=90, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(t[["rate", "rate_guard", "rate_relu"]].min().min() / 3,
                t.rate.max() * 8)
    ax.set_xticks(x, t.x, fontsize=7)
    ax.set_ylabel("per-inference failure rate")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "e29_guards.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    case_damage()
    e29_guards()
    print("written", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
