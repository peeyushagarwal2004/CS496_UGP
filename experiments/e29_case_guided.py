"""E29 -- the flip-case algorithm used as a sampling plan and a protection plan.

E25-E28 built and measured the case algorithm (docs/flip_algorithm.md) on one
network, ResNet8 width 16, from its exhaustive campaigns. This asks whether it
carries over, and whether it is useful in practice, on networks that have no
exhaustive campaign.

For one (network, format):

1. Stage 0/1, exact and free. Every stored bit of every MX weight tensor is put
   into a stratum ``(case, bit_was)`` with `mxfi.cases`. Element cases depend
   only on (code, bit), so they are counted from a histogram of codes; scale
   cases are counted per block. No injection is needed for this.
2. Masked cases (E0 sign of zero, S0 zero block) are damage 0 by construction and
   are not injected. Every other stratum gets `n` uniform injections (all of it
   when it is smaller), and the per-inference rate is the stratified mean.
3. Each injected fault is scored under three read/datapath settings:
     base   the PyTorch datapath used everywhere else (NaN passes ReLU)
     guard  stage-2 guards G1 + G2: a code that became NaN/Inf decodes as 0,
            and a scale above the tensor's clean band zeroes its block
     relu   stage-3 NaN-zeroing ReLU (``x > 0 ? x : 0``)
   A guard only changes E1 and above-band scale faults, and the ReLU only
   changes faults whose output was non-finite, so the others are not re-run.

The ResNet8 runs use the checkpoints its exhaustive campaigns were recorded on,
so the stratified estimate can be checked against the full ground truth.

Run::

    PYTHONPATH=. python -m experiments.e29_case_guided --run resnet8_e3m2
    PYTHONPATH=. python -m experiments.e29_case_guided --summary
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mxfi.campaign import Evaluator
from mxfi.cases import classify_scales, element_cases, label
from mxfi.data import campaign_images
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IMAGES = 200
CATASTROPHIC = {"E1", "S1", "S2"}
MASKED = {"E0", "S0"}

# run name -> (model, checkpoint, format, injections per stratum, exhaustive truth)
RUNS = {
    "resnet8_e3m2": ("resnet8_w16", "cluster/checkpoints/resnet8_w16_cifar10.pt", "e3m2", 60,
                     "results/e25_e3m2_labelled.csv.gz"),
    "resnet8_e4m3": ("resnet8_w16", "cluster/checkpoints/resnet8_w16_s9_cifar10.pt", "e4m3", 60,
                     "results/e25_e4m3_labelled.csv.gz"),
    "resnet8_c100_e3m2": ("resnet8_c100", "cluster/checkpoints/resnet8_cifar100.pt", "e3m2", 60, ""),
    "resnet8_c100_e4m3": ("resnet8_c100", "cluster/checkpoints/resnet8_cifar100.pt", "e4m3", 60, ""),
    "resnet8_c100_e5m2": ("resnet8_c100", "cluster/checkpoints/resnet8_cifar100.pt", "e5m2", 60, ""),
    "repvgg_e3m2": ("repvgg_a0", "checkpoints/repvgg_a0_cifar10.pt", "e3m2", 25, ""),
    "repvgg_e4m3": ("repvgg_a0", "checkpoints/repvgg_a0_cifar10.pt", "e4m3", 25, ""),
}


@contextmanager
def nan_zeroing_relu():
    """ReLU as ``x > 0 ? x : 0``: NaN and -Inf become 0, +Inf survives."""
    original = F.relu
    F.relu = lambda x, inplace=False: torch.where(x > 0, x, torch.zeros_like(x))
    try:
        yield
    finally:
        F.relu = original


# ---------------------------------------------------------------- stage 0/1
class ElementSpace:
    """Every valid element lane of one tensor, sorted by code, for fast sampling."""

    def __init__(self, t):
        valid = t.valid_mask()
        self.shape = t.codes.shape
        self.flat = np.flatnonzero(valid.ravel())
        codes = t.codes.ravel()[self.flat].astype(np.int64)
        self.order = np.argsort(codes, kind="stable")
        self.counts = np.bincount(codes, minlength=t.fmt.n_codes)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])

    def positions(self, code: int, k: int, rng, every: bool = False) -> list[tuple]:
        """`k` uniform lanes holding `code` (or every one of them)."""
        n = int(self.counts[code])
        pick = np.arange(n) if every else rng.choice(n, size=k, replace=k > n)
        idx = self.flat[self.order[self.starts[code] + pick]]
        return [tuple(int(v) for v in np.unravel_index(i, self.shape)) for i in idx]


def fault_space(mx: MXModel):
    """Exact stratum sizes, plus what is needed to sample each stratum."""
    el_rows, sc_parts, spaces = [], [], {}
    for name, lay in mx.layers.items():
        t = lay.mx
        fmt = t.fmt
        tab = element_cases(fmt)
        sp = ElementSpace(t)
        spaces[name] = sp
        for code in np.flatnonzero(sp.counts):
            for bit in range(fmt.width):
                el_rows.append((name, int(code), bit, label(tab["case"][code, bit]),
                                (int(code) >> bit) & 1, int(sp.counts[code])))
        rows, blks = np.indices(t.scales.shape).reshape(2, -1)
        for bit in range(8):
            b = np.full(len(rows), bit)
            c = classify_scales(t, (rows, blks), b)
            sc_parts.append(pd.DataFrame({
                "tensor": name, "row": rows, "blk": blks, "bit": bit,
                "label": [label(x) for x in c["case"]], "bit_was": c["bit_was"],
                "above_band": c["above_band"]}))
    el = pd.DataFrame(el_rows, columns=["tensor", "code", "bit", "label", "bit_was", "count"])
    sc = pd.concat(sc_parts, ignore_index=True)
    return el, sc, spaces


def strata(el: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    a = el.groupby(["label", "bit_was"])["count"].sum().rename("N").reset_index()
    a["site"] = "element"
    b = sc.groupby(["label", "bit_was"]).size().rename("N").reset_index()
    b["site"] = "scale"
    s = pd.concat([a, b], ignore_index=True)
    s["W"] = s.N / s.N.sum()
    return s


def draw(stratum, el, sc, spaces, n: int, rng) -> list[dict]:
    """`n` uniform faults of one stratum (all of it when it is smaller)."""
    if stratum.site == "scale":
        g = sc[(sc.label == stratum.label) & (sc.bit_was == stratum.bit_was)]
        g = g if len(g) <= n else g.sample(n, random_state=int(rng.integers(1 << 31)))
        return [{"site": "scale", "tensor": r.tensor, "index": (int(r.row), int(r.blk)),
                 "bit": int(r.bit), "above_band": bool(r.above_band)}
                for r in g.itertuples(index=False)]
    g = el[(el.label == stratum.label) & (el.bit_was == stratum.bit_was)]
    total = int(g["count"].sum())
    out = []
    if total <= n:                                   # the whole stratum
        for r in g.itertuples(index=False):
            out += [{"site": "element", "tensor": r.tensor, "index": p, "bit": int(r.bit),
                     "above_band": False}
                    for p in spaces[r.tensor].positions(int(r.code), 0, rng, every=True)]
        return out
    pick = rng.choice(len(g), size=n, p=(g["count"] / total).to_numpy())
    for i, k in zip(*np.unique(pick, return_counts=True)):
        r = g.iloc[i]
        out += [{"site": "element", "tensor": r.tensor, "index": p, "bit": int(r.bit),
                 "above_band": False}
                for p in spaces[r.tensor].positions(int(r.code), int(k), rng)]
    return out


# ------------------------------------------------------------- injection
def run(name: str, limit_strata: int = 0) -> None:
    model, ckpt, fmt, n, _ = RUNS[name]
    torch.set_num_threads(4)
    net, _ = load_trained(model, path=ROOT / ckpt)
    mx = MXModel(to_deploy(net), MXConfig(fmt=fmt, block_size=32)).quantize_weights()
    ev = Evaluator(campaign_images(model, IMAGES))
    golden = ev.golden(mx.module)
    el, sc, spaces = fault_space(mx)
    st = strata(el, sc)
    st.to_csv(RESULTS / f"e29_{name}_strata.csv", index=False)
    print(f"{name}: golden {golden.accuracy:.4f}, {int(st.N.sum()):,} faults, "
          f"{len(st)} strata", flush=True)

    def score(preds) -> float:
        return int((preds != golden.predictions).sum()) / IMAGES

    rng = np.random.default_rng(29)
    rows = []
    todo = st[~st.label.isin(MASKED)]
    if limit_strata:
        todo = todo.head(limit_strata)
    for s in todo.itertuples(index=False):
        faults = draw(s, el, sc, spaces, n, rng)
        for f in faults:
            lay = mx.layers[f["tensor"]]
            with mx.word_fault(f["tensor"], f["site"], f["index"], 1 << f["bit"]):
                p, fin = ev.predict(mx.module)
            base, nf = score(p), not fin
            guard, nf_g = base, nf
            if s.label == "E1" or f["above_band"]:
                K = lay.mx.block_size
                if f["site"] == "element":
                    row, blk, lane = f["index"]
                    patch = (row, blk * K + lane, np.zeros(1, np.float32))
                else:
                    row, blk = f["index"]
                    red = int(np.prod(lay.torch_shape[1:]))
                    patch = (row, blk * K, np.zeros(min(K, red - blk * K), np.float32))
                with mx._patched(lay, patch):
                    p, fin = ev.predict(mx.module)
                guard, nf_g = score(p), not fin
            relu, nf_r = base, nf
            if nf:
                with nan_zeroing_relu(), mx.word_fault(f["tensor"], f["site"], f["index"],
                                                       1 << f["bit"]):
                    p, fin = ev.predict(mx.module)
                relu, nf_r = score(p), not fin
            rows.append({"label": s.label, "bit_was": s.bit_was, "site": f["site"],
                         "tensor": f["tensor"], "index": str(f["index"]), "bit": f["bit"],
                         "above_band": f["above_band"], "base": base, "nonfinite": nf,
                         "guard": guard, "nonfinite_guard": nf_g,
                         "relu": relu, "nonfinite_relu": nf_r})
        print(f"  {s.label:>3} bit_was={s.bit_was} N={s.N:>9,}  n={len(faults):>3}  "
              f"mean {np.mean([r['base'] for r in rows[-len(faults):]]):.4f}", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / f"e29_{name}_faults.csv", index=False)
    pd.DataFrame(rows).assign(golden=golden.accuracy).to_csv(
        RESULTS / f"e29_{name}_faults.csv", index=False)
    print("done", flush=True)


# --------------------------------------------------------------- analysis
def estimate(st: pd.DataFrame, d: pd.DataFrame, col: str) -> tuple[float, float]:
    """Stratified mean of `col` and its standard error (masked strata exactly 0)."""
    m = d.groupby(["label", "bit_was"])[col].agg(["mean", "var", "size"]).reset_index()
    s = st.merge(m, on=["label", "bit_was"], how="left")
    s[["mean", "var"]] = s[["mean", "var"]].fillna(0.0)
    s["size"] = s["size"].fillna(0)
    rate = float((s.W * s["mean"]).sum())
    fpc = np.where(s["size"] > 0, 1 - s["size"] / s.N, 0)
    var = np.where(s["size"] > 1, s.W ** 2 * s["var"] / s["size"].clip(lower=1) * fpc, 0)
    return rate, float(np.sqrt(var.sum()))


def uniform_se(st: pd.DataFrame, d: pd.DataFrame, n_total: int) -> float:
    """SE a uniform campaign of the same size would have had."""
    m = d.groupby(["label", "bit_was"])["base"].agg(["mean", "var"]).reset_index()
    s = st.merge(m, on=["label", "bit_was"], how="left").fillna(0.0)
    mu = (s.W * s["mean"]).sum()
    sigma2 = (s.W * (s["var"] + (s["mean"] - mu) ** 2)).sum()
    return float(np.sqrt(sigma2 / n_total))


def summarise(name: str) -> tuple[dict, pd.DataFrame]:
    model, _, fmt, _, truth = RUNS[name]
    st = pd.read_csv(RESULTS / f"e29_{name}_strata.csv")
    d = pd.read_csv(RESULTS / f"e29_{name}_faults.csv")
    d["nf_damage"] = d.base * d.nonfinite
    rate, se = estimate(st, d, "base")
    g, g_se = estimate(st, d, "guard")
    r, r_se = estimate(st, d, "relu")
    nf, _ = estimate(st, d, "nf_damage")

    per = d.groupby(["label", "bit_was"])["base"].mean().rename("mean").reset_index()
    cs = st.merge(per, on=["label", "bit_was"], how="left").fillna({"mean": 0.0})
    cs["contrib"] = cs.W * cs["mean"]
    up = cs[cs.bit_was == 0].contrib.sum() / cs.contrib.sum()
    cat = cs[cs.label.isin(CATASTROPHIC)]
    by_case = (cs.groupby("label").agg(fault_share=("W", "sum"), damage=("contrib", "sum"))
               .assign(damage_share=lambda x: x.damage / x.damage.sum(),
                       mean_damage=lambda x: x.damage / x.fault_share))
    out = {
        "run": name, "model": model, "fmt": fmt, "golden": float(d.golden.iloc[0]),
        "faults": int(st.N.sum()), "injected": len(d),
        "forward_passes": len(d) + int((d.label.eq("E1") | d.above_band).sum())
        + int(d.nonfinite.sum()),
        "rate": rate, "se": se, "uniform_se_same_n": uniform_se(st, d, len(d)),
        "zero_one_share": up,
        "catastrophic_fault_share": cat.W.sum(),
        "catastrophic_damage_share": cat.contrib.sum() / cs.contrib.sum(),
        "catastrophic_all_0to1": bool((cat[cat.N > 0].bit_was == 0).all()),
        "nonfinite_share": nf / rate,
        "rate_guard": g, "guard_x": rate / g, "rate_relu": r, "relu_x": rate / r,
    }
    out["variance_reduction"] = (out["uniform_se_same_n"] / se) ** 2
    if truth:
        t = pd.read_csv(ROOT / truth, usecols=["damage"], low_memory=False)
        out["exhaustive_rate"] = float(t.damage.mean())
        out["z_vs_exhaustive"] = (rate - out["exhaustive_rate"]) / se
    return out, by_case.sort_values("damage", ascending=False)


def summary() -> None:
    rows, text = [], []
    for name in RUNS:
        if not (RESULTS / f"e29_{name}_faults.csv").exists():
            continue
        s, by_case = summarise(name)
        rows.append(s)
        text += [f"## {name}  ({s['model']}, {s['fmt']}, golden {s['golden']:.4f})",
                 by_case.to_string(float_format=lambda v: f"{v:.5f}"), ""]
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "e29_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        body = t.set_index("run").T.to_string(float_format=lambda v: f"{v:.5g}")
    out = body + "\n\n" + "\n".join(text)
    (RESULTS / "e29_case_guided.txt").write_text(out)
    print(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="*", default=[])
    p.add_argument("--limit-strata", type=int, default=0, help="smoke test")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    for name in a.run:
        run(name, a.limit_strata)
    if a.summary:
        summary()


if __name__ == "__main__":
    main()
