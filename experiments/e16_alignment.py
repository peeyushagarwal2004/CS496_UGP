"""E16 -- the last open question: does alignment make the sensitivity tail?

F44 factorised the sensitivity of a linear layer,

    dm/dW[o,i] = sum_p a_i^(p) g_o^(p),

measured all three marginals -- stored weights, input activations, output
gradients -- and found them identical to within 1.03x across element formats,
while the sensitivity tail itself differs by 14x. By elimination the difference
must live in the *alignment*: whether the per-position terms
$x_p = a_i^{(p)} g_o^{(p)}$ reinforce each other or cancel.

Elimination is an argument, not a measurement, so this measures it. Alignment can
be destroyed without touching either marginal: permute which position's
activation vector meets which position's gradient vector,

    Gshuf = sum_p a^(p) g^(pi(p))^T,

and every $a_i$ and every $g_o$ still contributes exactly the same multiset of
values. Any difference in tail between $G$ and $G_\\mathrm{shuf}$ is alignment
and nothing else. Two numbers follow:

* the **alignment gain**, tail of $G$ over tail of $G_\\mathrm{shuf}$ -- how much
  of the observed tail is alignment rather than the marginals;
* the **coherence** $C=|\\sum_p x_p|/\\sum_p|x_p|$ of the most sensitive weights,
  which is $1$ when every position pushes the same way and about $P^{-1/2}$ when
  the terms are independent in sign.

If alignment is the mechanism, the gain must both exceed one and *track* the
per-format vulnerability of F45. If the gain is the same for every format, the
tail has a fourth source and F44's conclusion by elimination was wrong.

Run on the ViT, whose every weight matrix is a Linear, so the factorisation is
exact rather than an unfolding of convolution patches::

    PYTHONPATH=. python3 -m experiments.e16_alignment --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mxfi.data import campaign_subset
from mxfi.models import to_deploy
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained, setup_device

RESULTS = Path(__file__).resolve().parent.parent / "results"
FORMATS = ["e5m2", "e3m2", "e4m3", "e2m3", "e2m1"]
# perturbation-driven SDC per format, from the F45 table
F45_SDC = {"e5m2": 0.0040, "e3m2": 0.0041, "e4m3": 0.0329, "e2m3": 0.0236, "e2m1": 0.0500}


def tail(g: torch.Tensor) -> float:
    """Outlier ratio of a matrix: the 99.9th percentile of |g| over its rms."""
    a = g.abs().flatten().float()
    rms = float(torch.sqrt((a ** 2).mean()))
    if a.numel() > 4_000_000:                       # torch.quantile has a size limit
        a = a[torch.randint(0, a.numel(), (1_000_000,), device=a.device)]
    return float(torch.quantile(a, 0.999)) / max(rms, 1e-30)


def coherence(A: torch.Tensor, Gg: torch.Tensor,
              o: torch.Tensor, i: torch.Tensor) -> float:
    """Mean |sum_p x_p| / sum_p |x_p| over the given (out, in) weight positions."""
    x = Gg[:, o] * A[:, i]                          # (positions, k)
    num = x.sum(dim=0).abs()
    den = x.abs().sum(dim=0).clamp_min(1e-30)
    return float((num / den).mean())


def measure(mx: MXModel, xs: torch.Tensor, device: str, perms: int, topk: int,
            align_images: int, rng: np.random.Generator) -> dict:
    """Alignment gain and coherence per layer, aggregated over the images."""
    linear = {n: l.module for n, l in mx.layers.items()
              if isinstance(l.module, nn.Linear)}
    cache: dict[str, list] = {n: [None, None] for n in linear}
    handles = []

    def make(name):
        def hook(_m, inputs, output):
            cache[name][0] = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if output.requires_grad:
                output.register_hook(
                    lambda g, n=name: cache[n].__setitem__(
                        1, g.detach().reshape(-1, g.shape[-1])))
        return hook

    for n, m in linear.items():
        m.weight.requires_grad_(True)
        handles.append(m.register_forward_hook(make(n)))

    acc = {n: {"real": 0.0, "shuf": 0.0, "c_top": [], "c_rand": [], "P": 0,
               "grms": [], "med": []} for n in linear}
    # every ratio above divides a matrix by its own rms, so it can only see the
    # *shape* of the distribution. The scale is a separate question, and needs
    # the absolute gradient magnitude and the margin that normalises it.
    rms_w = {n: float(torch.sqrt((m.weight.detach() ** 2).mean()))
             for n, m in linear.items()}
    margins = []
    # E11 defines s as the *maximum over images* of |dm/dw| rms / m, so one image
    # with a small margin raises s for every weight at once. Keeping the running
    # maximum with and without that division separates the two.
    best_s = {n: torch.zeros_like(m.weight) for n, m in linear.items()}
    best_g = {n: torch.zeros_like(m.weight) for n, m in linear.items()}

    for img in range(xs.shape[0]):
        mx.module.zero_grad(set_to_none=True)
        z = mx.module(xs[img : img + 1].to(device))[0]
        t2 = torch.topk(z, 2)
        margin = t2.values[0] - t2.values[1]
        margins.append(max(float(margin), 1e-6))
        margin.backward()

        with torch.no_grad():
            for n, m in linear.items():
                if m.weight.grad is not None:
                    g = m.weight.grad.abs() * rms_w[n]
                    torch.maximum(best_g[n], g, out=best_g[n])
                    torch.maximum(best_s[n], g / margins[-1], out=best_s[n])

            if img >= align_images:
                continue
            for n in linear:
                A, Gg = cache[n]
                if A is None or Gg is None:
                    continue
                P = A.shape[0]
                G = Gg.T @ A                        # the exact dm/dW for this image
                # the same marginals with the alignment destroyed
                shuf = np.mean([tail(Gg[torch.randperm(P, device=A.device)].T @ A)
                                for _ in range(perms)])
                acc[n]["real"] = max(acc[n]["real"], tail(G))
                acc[n]["shuf"] = max(acc[n]["shuf"], float(shuf))
                acc[n]["P"] = P
                # the sensitivity itself, in the units of F38: |dm/dw| rms / m
                scaled = G.abs() * (rms_w[n] / margins[-1])
                acc[n]["grms"].append(float(torch.sqrt((scaled ** 2).mean())))
                acc[n]["med"].append(float(scaled.flatten().median()))

                flat = G.abs().flatten()
                k = min(topk, flat.numel())
                top = torch.topk(flat, k).indices
                o_t, i_t = top // G.shape[1], top % G.shape[1]
                o_r = torch.as_tensor(rng.integers(0, G.shape[0], k), device=A.device)
                i_r = torch.as_tensor(rng.integers(0, G.shape[1], k), device=A.device)
                acc[n]["c_top"].append(coherence(A, Gg, o_t, i_t))
                acc[n]["c_rand"].append(coherence(A, Gg, o_r, i_r))

    for h in handles:
        h.remove()
    for m in linear.values():
        m.weight.requires_grad_(False)

    # the global distribution over every quantised weight, as E11 defines it
    flat_s = torch.cat([b.flatten() for b in best_s.values()]).cpu().numpy()
    flat_g = torch.cat([b.flatten() for b in best_g.values()]).cpu().numpy()
    sizes = np.array([linear[n].weight.numel() for n in linear], float)
    w = sizes / sizes.sum()
    real = np.array([acc[n]["real"] for n in linear])
    shuf = np.array([acc[n]["shuf"] for n in linear])
    return {
        "layers": len(linear),
        "positions": int(np.median([acc[n]["P"] for n in linear])),
        "tail_real": float((real * w).sum()),
        "tail_shuffled": float((shuf * w).sum()),
        "alignment_gain": float((real * w).sum() / max((shuf * w).sum(), 1e-30)),
        "gain_max_layer": float((real / np.maximum(shuf, 1e-30)).max()),
        "coherence_top": float(sum(np.mean(acc[n]["c_top"]) * wi
                                   for n, wi in zip(linear, w))),
        "coherence_random": float(sum(np.mean(acc[n]["c_rand"]) * wi
                                      for n, wi in zip(linear, w))),
        "margin": float(np.mean(margins)),
        "grad_rms": float(sum(np.mean(acc[n]["grms"]) * wi / rms_w[n]
                              * np.mean(margins) for n, wi in zip(linear, w))),
        "s_rms": float(sum(np.mean(acc[n]["grms"]) * wi for n, wi in zip(linear, w))),
        "s_median": float(sum(np.mean(acc[n]["med"]) * wi for n, wi in zip(linear, w))),
        "margin_min": float(np.min(margins)),
        "margin_p5": float(np.quantile(margins, 0.05)),
        "s_max_median": float(np.median(flat_s)),
        "g_max_median": float(np.median(flat_g)),
        "s_max_p99": float(np.quantile(flat_s, 0.99)),
        "g_max_p99": float(np.quantile(flat_g, 0.99)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--images", type=int, default=5)
    p.add_argument("--perms", type=int, default=4)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--align-images", type=int, default=50,
                   help="images used for the (costlier) permutation test")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    dev = setup_device(a.device)
    xs = campaign_subset(n_per_class=a.images, download=False).tensors[0]
    rng = np.random.default_rng(0)

    rows = []
    for fmt in FORMATS:
        net, _ = load_trained(a.model, seed=a.seed)
        mx = MXModel(to_deploy(net).to(dev),
                     MXConfig(fmt=fmt, block_size=a.block_size)).quantize_weights()
        r = measure(mx, xs, dev, a.perms, a.topk, a.align_images, rng)
        rows.append({"fmt": fmt, **r, "sdc_f45": F45_SDC[fmt]})
        mx.restore_weights()
        print(f"  {fmt}: gain {r['alignment_gain']:.3f}, coherence top "
              f"{r['coherence_top']:.3f} against random {r['coherence_random']:.3f}")

    df = pd.DataFrame(rows)
    out = RESULTS / f"e16_{a.model}_alignment.csv"
    df.to_csv(out, index=False)

    n = int(df.positions.iloc[0])
    print(f"\n{a.model}, {int(df.layers.iloc[0])} Linear layers, {n} positions per "
          f"image, {xs.shape[0]} images, {a.perms} permutations each")
    print(f"coherence for terms independent in sign would be about {n ** -0.5:.3f}\n")
    print(f"{'fmt':>6} {'tail real':>10} {'tail shuffled':>14} {'gain':>7} "
          f"{'coh top':>9} {'coh rand':>9} {'SDC (F45)':>10}")
    for _, r in df.iterrows():
        print(f"{r.fmt:>6} {r.tail_real:>10.2f} {r.tail_shuffled:>14.2f} "
              f"{r.alignment_gain:>6.2f}x {r.coherence_top:>9.3f} "
              f"{r.coherence_random:>9.3f} {r.sdc_f45:>10.4f}")

    print(f"\nshape against scale -- every ratio above is a matrix over its own rms,\n"
          f"so it can only see shape. These are the absolute quantities:\n")
    print(f"{'fmt':>6} {'|dm/dW| rms':>12} {'margin m':>9} {'rms of s':>9} "
          f"{'median s':>9} {'SDC (F45)':>10}")
    for _, r in df.iterrows():
        print(f"{r.fmt:>6} {r.grad_rms:>12.3e} {r.margin:>9.3f} {r.s_rms:>9.4f} "
              f"{r.s_median:>9.4f} {r.sdc_f45:>10.4f}")
    lo, hi = df.s_rms.min(), df.s_rms.max()
    g_lo, g_hi = df.grad_rms.min(), df.grad_rms.max()
    m_lo, m_hi = df.margin.min(), df.margin.max()
    print(f"\n  spread across formats: sensitivity {hi / max(lo, 1e-30):.2f}x, "
          f"from gradient magnitude {g_hi / max(g_lo, 1e-30):.2f}x "
          f"and margin {m_hi / max(m_lo, 1e-30):.2f}x")
    print(f"  spread in tail shape {df.tail_real.max() / df.tail_real.min():.2f}x, "
          f"in alignment gain {df.alignment_gain.max() / df.alignment_gain.min():.2f}x")

    print("\nthe sensitivity E11 actually measures is a maximum over images, and the\n"
          "margin sits in its denominator, so separate the two:\n")
    print(f"{'fmt':>6} {'median s':>9} {'99th s':>9} {'median |g|rms':>14} "
          f"{'99th |g|rms':>12} {'min margin':>11} {'SDC (F45)':>10}")
    for _, r in df.iterrows():
        print(f"{r.fmt:>6} {r.s_max_median:>9.4f} {r.s_max_p99:>9.3f} "
              f"{r.g_max_median:>14.4f} {r.g_max_p99:>12.3f} "
              f"{r.margin_min:>11.4f} {r.sdc_f45:>10.4f}")
    sp = lambda c: float(df[c].max() / max(df[c].min(), 1e-30))
    print(f"\n  spread across formats: median s {sp('s_max_median'):.2f}x, "
          f"99th pct s {sp('s_max_p99'):.2f}x")
    print(f"  with the margin divided out: median |g|rms {sp('g_max_median'):.2f}x, "
          f"99th pct {sp('g_max_p99'):.2f}x")
    print(f"  smallest margin over the images: {sp('margin_min'):.2f}x spread")

    from scipy import stats as st
    for col, label in (("alignment_gain", "alignment gain"),
                       ("coherence_top", "coherence of the top weights"),
                       ("tail_real", "tail shape of dm/dW"),
                       ("s_rms", "rms sensitivity (a scale, not a shape)"),
                       ("grad_rms", "absolute gradient magnitude"),
                       ("s_max_median", "median s, maximised over images"),
                       ("g_max_median", "the same with the margin divided out"),
                       ("margin_min", "the smallest margin over the images")):
        pr = st.pearsonr(df[col], df.sdc_f45)[0]
        sp = st.spearmanr(df[col], df.sdc_f45)[0]
        print(f"\n  {label} against per-format SDC: "
              f"pearson {pr:+.3f}, spearman {sp:+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
