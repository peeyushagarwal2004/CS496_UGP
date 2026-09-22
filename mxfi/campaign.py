"""Fault-injection campaign driver: golden-vs-faulty comparison and metrics.

One injection costs one forward pass over a fixed evaluation batch, so the
loop here is deliberately thin -- the expensive part is the model, not the
bookkeeping.  Every row of the result carries three things:

* **what was injected** -- tensor, site, bit, index
* **what happened** -- top-1 changes, accuracy drop, non-finite output
* **what the representation predicted would happen** -- the a-priori severity
  from the exact impact tables

Carrying the predicted severity alongside the observed outcome is what makes
the value-aware analysis possible: it is the correlation between the two that
says whether severity is a usable proxy for failure, which is the open
question RQ2 of the MX-TreeFI draft.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .faults import element_impact_table, scale_impact_table
from .sampling import FaultSite
from .stats import Estimate, StratumCount, binomial_rate, stratified_rate
from .torch_mx import MXModel

__all__ = ["Evaluator", "Golden", "run_campaign", "summarise"]


@dataclass
class Golden:
    """Fault-free reference for the fixed evaluation batch."""

    predictions: np.ndarray
    labels: np.ndarray
    accuracy: float

    @property
    def n(self) -> int:
        return int(self.labels.size)


class Evaluator:
    """Runs a model over one fixed dataset, returning predictions.

    The dataset is materialised into batched tensors once, so every injection
    replays byte-identical inputs and differences are attributable to the
    fault alone.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 256,
                 device: str = "cpu"):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.device = device
        # moved once, so every injection replays identical device tensors
        self.batches = [(x.to(device), y) for x, y in loader]
        self.labels = torch.cat([y for _, y in self.batches]).numpy()

    @torch.no_grad()
    def predict(self, model: torch.nn.Module) -> tuple[np.ndarray, bool]:
        """Top-1 predictions, plus whether any logit was non-finite."""
        model.eval()
        preds, finite = [], True
        for x, _ in self.batches:
            out = model(x)
            finite &= bool(torch.isfinite(out).all())
            # a NaN row has no meaningful argmax; mark it as class -1
            bad = ~torch.isfinite(out).all(dim=1)
            p = out.argmax(1)
            p[bad] = -1
            preds.append(p)
        return torch.cat(preds).cpu().numpy(), finite

    def golden(self, model: torch.nn.Module) -> Golden:
        preds, finite = self.predict(model)
        if not finite:
            raise RuntimeError("fault-free model produced non-finite logits")
        return Golden(preds, self.labels, float((preds == self.labels).mean()))


# ------------------------------------------------------------ severity lookup

def _predicted_severity(mx: MXModel, site: FaultSite) -> dict:
    """A-priori impact of this fault, from the exact enumeration tables."""
    tensor = mx.weight_tensors()[site.tensor]
    f = site.fault
    if f.site == "element":
        code = int(tensor.codes[f.index])
        t = element_impact_table(tensor.fmt)
        return {
            "code": code,
            "value_before": float(t["before"][code, f.bit]),
            "value_after": float(t["after"][code, f.bit]),
            "log_severity": float(t["log_severity"][code, f.bit]),
            "rel_error": float(t["rel_error"][code, f.bit]),
            "sign_flip": bool(t["sign_flip"][code, f.bit]),
            "to_nonfinite": bool(t["to_nonfinite"][code, f.bit]),
            "blast_radius": 1,
        }
    byte = int(tensor.scales[f.index])
    t = scale_impact_table()
    return {
        "code": byte,
        "value_before": float(byte),
        "value_after": float(t["after"][byte, f.bit]),
        "log_severity": float(t["log_severity"][byte, f.bit]),
        "rel_error": float("inf"),
        "sign_flip": False,
        "to_nonfinite": bool(t["to_nan"][byte, f.bit]),
        "blast_radius": int(tensor.block_size),
    }


# ------------------------------------------------------------------- the loop

def run_campaign(mx: MXModel, faults: Sequence[FaultSite], ev: Evaluator,
                 golden: Golden, weights: np.ndarray | None = None,
                 out_csv: Path | str | None = None,
                 flush_every: int = 200, progress: bool = True) -> pd.DataFrame:
    """Inject each fault, score it against `golden`, and tabulate.

    Results are flushed to `out_csv` periodically so a long campaign survives
    an interrupted run.
    """
    rows: list[dict] = []
    out_csv = Path(out_csv) if out_csv else None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    it: Iterable = enumerate(faults)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(list(it), total=len(faults), desc="injections", unit="fi")
        except ImportError:
            pass

    for i, site in it:
        with mx.fault(site):
            preds, finite = ev.predict(mx.module)

        changed = int((preds != golden.predictions).sum())
        acc = float((preds == golden.labels).mean())
        row = {
            "i": i,
            "tensor": site.tensor,
            "site": site.site,
            "bit": site.bit,
            "index": str(site.fault.index),
            "changed": changed,
            "change_rate": changed / golden.n,
            "sdc": changed > 0,
            "accuracy": acc,
            "acc_drop": golden.accuracy - acc,
            "nonfinite": not finite,
            "weight": float(weights[i]) if weights is not None else np.nan,
        }
        row.update(_predicted_severity(mx, site))
        rows.append(row)

        if out_csv and (i + 1) % flush_every == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


# ------------------------------------------------------------------ summaries

def summarise(df: pd.DataFrame, by: Sequence[str] = ("site",),
              conf: float = 0.95) -> pd.DataFrame:
    """SDC rate with confidence intervals, grouped by the given columns."""
    out = []
    for key, g in df.groupby(list(by), dropna=False):
        est = binomial_rate(g["sdc"].to_numpy(), conf)
        key = key if isinstance(key, tuple) else (key,)
        out.append({
            **dict(zip(by, key)),
            "n": est.n,
            "sdc_rate": est.rate,
            "ci_lo": est.lo,
            "ci_hi": est.hi,
            "mean_change_rate": float(g["change_rate"].mean()),
            "mean_acc_drop": float(g["acc_drop"].mean()),
            "nonfinite_rate": float(g["nonfinite"].mean()),
        })
    return pd.DataFrame(out).sort_values("sdc_rate", ascending=False)


def weighted_rate(df: pd.DataFrame, conf: float = 0.95) -> Estimate:
    """Model-wide SDC rate from a stratified campaign, using inclusion weights.

    Falls back to the plain binomial estimate when the campaign was uniform.
    """
    if df["weight"].isna().all():
        return binomial_rate(df["sdc"].to_numpy(), conf)

    strata = {}
    for key, g in df.groupby(["tensor", "site"], dropna=False):
        # every draw in a stratum shares one inclusion weight
        strata[key] = StratumCount(int(g["sdc"].sum()), len(g),
                                   float(g["weight"].iloc[0] * len(g)))
    return stratified_rate(strata, conf)
