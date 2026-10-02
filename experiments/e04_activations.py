"""E04 -- weight vs activation fault sensitivity (objective O4).

Weight and activation faults are **not** comparable on the metric E01 used,
and conflating them would overstate activation robustness by ~2 orders of
magnitude.  The two differ physically:

* a **weight** fault is *persistent* -- it sits in weight memory and corrupts
  every inference that reads it
* an **activation** fault is *transient* -- it corrupts one buffer during one
  inference and is gone

So "SDC = any of 200 images changed" is the right question for a weight fault
and the wrong one for an activation fault, which can only ever affect the
single inference it lands in.  The comparable quantity for both is

    P(a given inference is corrupted | one fault)

which for weights is the mean fraction of images whose prediction changed
(``change_rate`` in E01) and for activations is whether *that* image's
prediction changed.  This experiment measures the activation side under a
per-inference protocol: batch of one, fault injected into that inference's
own activation buffer.

Because activation blocks run down the channel axis and the batch axis is
separate, each image's blocks are independent -- so batch-of-one is not an
approximation, it is the exact per-inference fault space.

Run::

    .venv/Scripts/python.exe -m experiments.e04_activations --n 2000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mxfi.data import campaign_images
from mxfi.faults import fault_space, inject
from mxfi.models import to_deploy
from mxfi.sampling import FaultSite, bit_population, sample_uniform
from mxfi.stats import binomial_rate
from mxfi.torch_mx import MXConfig, MXModel
from mxfi.train import load_trained

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="resnet8",
                   choices=sorted(__import__("mxfi.models", fromlist=["MODELS"]).MODELS))
    p.add_argument("--fmt", default="e4m3")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--n", type=int, default=2000, help="total injections")
    p.add_argument("--images", type=int, default=25, help="distinct images used")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = MXConfig(fmt=a.fmt, block_size=a.block_size, quantise_activations=True)

    from mxfi.train import setup_device
    dev = setup_device(a.device)
    net, _ = load_trained(a.model)
    mx = MXModel(to_deploy(net).to(dev), cfg).quantize_weights()
    mx.enable_activation_quantisation()

    ds = campaign_images(a.model, 100)
    xs, ys = ds.tensors
    rng = np.random.default_rng(a.seed)
    picks = rng.choice(len(xs), size=a.images, replace=False)

    per_image = a.n // a.images
    print(f"{cfg.label()} | {a.images} images x {per_image} injections "
          f"= {a.images * per_image}\n")

    rows = []
    t0 = time.time()
    for j, idx in enumerate(picks):
        x = xs[idx : idx + 1].to(dev)
        with torch.no_grad():
            golden_pred = int(mx.module(x).argmax(1))

        # this inference's own activation fault space
        acts = mx.activation_space(x)
        if j == 0:
            tot = sum(fault_space(q)["total_bits"] for q in acts.values())
            wpop = sum(bit_population(mx.weight_tensors()).values())
            print(f"activation fault space (per inference): {tot:,} bits")
            print(f"weight fault space (whole model):       {wpop:,} bits\n")

        faults = sample_uniform(acts, per_image, rng)
        for f in faults:
            with mx.activation_fault(f):
                with torch.no_grad():
                    out = mx.module(x)
            finite = bool(torch.isfinite(out).all())
            pred = int(out.argmax(1)) if finite else -1
            rows.append({
                "image": int(idx), "tensor": f.tensor, "site": f.site,
                "bit": f.bit, "corrupted": pred != golden_pred,
                "nonfinite": not finite,
                "block_size": a.block_size, "fmt": a.fmt,
            })
        if (j + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  {j+1}/{a.images} images, {len(rows)} injections, "
                  f"{el:.0f}s ({el/len(rows)*1000:.0f} ms each)")

    df = pd.DataFrame(rows)
    out = RESULTS / f"e04_{a.model}_activations_{cfg.label()}.csv"
    df.to_csv(out, index=False)

    print(f"\n=== ACTIVATION faults: P(inference corrupted | fault) ===")
    overall = binomial_rate(df["corrupted"].to_numpy())
    print(f"overall: {overall}")
    for site, g in df.groupby("site"):
        est = binomial_rate(g["corrupted"].to_numpy())
        print(f"  {site:8} n={est.n:5d}  {est.rate:.4f} "
              f"[{est.lo:.4f}, {est.hi:.4f}]  nonfinite {g.nonfinite.mean():.2%}")

    print("\n--- by bit (element site) ---")
    e = df[df.site == "element"]
    print(e.groupby("bit")["corrupted"].agg(["size", "mean"]).to_string())

    print("\n--- by layer ---")
    print(df.groupby("tensor")["corrupted"].agg(["size", "mean"])
            .sort_values("mean", ascending=False).to_string())

    # ---- the comparable weight number, from E01
    w_csv = RESULTS / f"e01_{a.model}_{a.fmt}-K{a.block_size}-ocp-w-n3000.csv"
    if w_csv.exists():
        w = pd.read_csv(w_csv)
        print("\n=== WEIGHT vs ACTIVATION, on the same per-inference metric ===")
        print(f"{'site':10} {'weight':>12} {'activation':>12}   ratio")
        for site in ("element", "scale"):
            wv = w[w.site == site]["change_rate"].mean()
            av = df[df.site == site]["corrupted"].mean()
            print(f"{site:10} {wv:12.4f} {av:12.4f}   {wv/max(av,1e-9):6.1f}x")
        wv = w["change_rate"].mean(); av = df["corrupted"].mean()
        print(f"{'overall':10} {wv:12.4f} {av:12.4f}   {wv/max(av,1e-9):6.1f}x")
        print("\n(weight = mean fraction of the 200-image subset whose prediction "
              "changed;\n activation = fraction of injections that changed their "
              "own inference)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
