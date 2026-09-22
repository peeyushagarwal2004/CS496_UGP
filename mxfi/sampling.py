"""Fault-space sampling over a whole MX-quantised model.

A campaign draws faults from the union of every tensor's two-site bit space.
Two samplers are provided, and the distinction matters for interpretation:

:func:`sample_uniform`
    Every stored bit is equally likely.  This is the physically faithful
    occurrence model for a uniform per-bit upset rate -- and it means scale
    bits are drawn only ``8 / (8 + K*width)`` of the time (about 3% at K=32,
    8-bit elements).  Estimates need no reweighting.

:func:`sample_stratified`
    Budget is allocated across strata (tensor x site, and optionally bit
    position) by an explicit rule, and each draw carries an inclusion weight
    so the aggregate estimate stays unbiased.  This is what makes the rare
    but high-blast-radius scale stratum affordable to measure.

The weights returned here are the ``W`` term of the drafts' allocation
``n ∝ π·W·√(r(1−r))·Φ``; :func:`neyman_allocation` implements the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .codec import MXTensor
from .faults import Fault

__all__ = ["FaultSite", "Stratum", "bit_population", "sample_uniform",
           "sample_stratified", "neyman_allocation", "proportional_allocation"]


@dataclass(frozen=True)
class FaultSite:
    """A fault located in a named tensor of a model."""

    tensor: str
    fault: Fault

    @property
    def site(self) -> str:
        return self.fault.site

    @property
    def bit(self) -> int:
        return self.fault.bit


@dataclass(frozen=True)
class Stratum:
    """One (tensor, site) population, optionally narrowed to a bit position."""

    tensor: str
    site: str
    bit: int | None = None

    def key(self) -> tuple:
        return (self.tensor, self.site, self.bit)


def _site_width(mx: MXTensor, site: str) -> int:
    return mx.fmt.width if site == "element" else 8


def _valid_indices(mx: MXTensor, site: str) -> np.ndarray:
    """Flat indices of injectable slots, excluding padding lanes."""
    if site == "scale":
        return np.arange(mx.scales.size)
    return np.flatnonzero(mx.valid_mask().ravel())


def bit_population(tensors: Mapping[str, MXTensor],
                   bitwise: bool = False) -> dict[tuple, int]:
    """Number of addressable bits per stratum.

    With ``bitwise`` the population is split by bit position as well, which is
    what a per-bit sensitivity map needs.
    """
    pop: dict[tuple, int] = {}
    for name, mx in tensors.items():
        for site in ("element", "scale"):
            n_slots = len(_valid_indices(mx, site))
            w = _site_width(mx, site)
            if bitwise:
                for b in range(w):
                    pop[Stratum(name, site, b).key()] = n_slots
            else:
                pop[Stratum(name, site).key()] = n_slots * w
    return pop


# ------------------------------------------------------------------ samplers

def sample_uniform(tensors: Mapping[str, MXTensor], n: int,
                   rng: np.random.Generator) -> list[FaultSite]:
    """Draw `n` faults uniformly over every stored bit of every tensor.

    Sampling is with replacement, which is the right model for independent
    upsets and keeps the estimator a plain binomial.
    """
    pop = bit_population(tensors)
    keys = list(pop)
    counts = np.array([pop[k] for k in keys], dtype=np.float64)
    if counts.sum() == 0:
        raise ValueError("empty fault space")

    picks = rng.multinomial(n, counts / counts.sum())
    out: list[FaultSite] = []
    for (name, site, _), k in zip(keys, picks):
        out.extend(_draw_within(tensors[name], name, site, None, int(k), rng))
    rng.shuffle(out)
    return out


def sample_stratified(tensors: Mapping[str, MXTensor],
                      allocation: Mapping[tuple, int],
                      rng: np.random.Generator,
                      ) -> tuple[list[FaultSite], np.ndarray]:
    """Draw the requested number of faults from each stratum.

    Returns the faults and their **inclusion weights** -- the stratum's share
    of the total bit population divided by the number of draws taken from it.
    A weighted mean of per-fault outcomes is then an unbiased estimate of the
    model-wide rate, regardless of how skewed the allocation was.
    """
    pop = bit_population(tensors, bitwise=any(k[2] is not None for k in allocation))
    total = sum(pop.values())

    faults: list[FaultSite] = []
    weights: list[float] = []
    for key, n_draw in allocation.items():
        if n_draw <= 0:
            continue
        if key not in pop:
            raise KeyError(f"stratum {key} is not in the fault space")
        name, site, bit = key
        drawn = _draw_within(tensors[name], name, site, bit, int(n_draw), rng)
        faults.extend(drawn)
        weights.extend([pop[key] / total / n_draw] * len(drawn))

    return faults, np.asarray(weights, dtype=np.float64)


def _draw_within(mx: MXTensor, name: str, site: str, bit: int | None,
                 n: int, rng: np.random.Generator) -> list[FaultSite]:
    """Sample `n` faults inside one (tensor, site[, bit]) stratum."""
    if n <= 0:
        return []
    slots = _valid_indices(mx, site)
    shape = mx.scales.shape if site == "scale" else mx.codes.shape
    width = _site_width(mx, site)

    flat = slots[rng.integers(0, len(slots), size=n)]
    idx = np.unravel_index(flat, shape)
    bits = (np.full(n, bit, dtype=np.int64) if bit is not None
            else rng.integers(0, width, size=n))

    return [FaultSite(name, Fault(site, tuple(int(a) for a in pos), int(b)))
            for pos, b in zip(zip(*idx), bits)]


# --------------------------------------------------------------- allocations

def proportional_allocation(tensors: Mapping[str, MXTensor], n: int,
                            bitwise: bool = False) -> dict[tuple, int]:
    """Split `n` across strata in proportion to bit population.

    Equivalent in expectation to :func:`sample_uniform`; useful as the
    baseline an unequal allocation is compared against.
    """
    pop = bit_population(tensors, bitwise=bitwise)
    total = sum(pop.values())
    return _largest_remainder({k: v / total for k, v in pop.items()}, n)


def neyman_allocation(tensors: Mapping[str, MXTensor], n: int,
                      rates: Mapping[tuple, float],
                      blast_weight: Mapping[tuple, float] | None = None,
                      bitwise: bool = False,
                      floor: int = 1) -> dict[tuple, int]:
    """Allocate `n` injections by ``W · sqrt(r(1-r)) · Φ``.

    Parameters
    ----------
    rates : pilot estimate of each stratum's failure rate ``r``.  Strata with
        rates near 0 or 1 need few samples; those near 0.5 need many.
    blast_weight : the drafts' ``Φ``.  Passing ``K`` for scale strata and 1
        for element strata up-weights the stratum whose faults hit K values.
    floor : minimum draws per stratum, so no stratum is estimated from zero
        samples.
    """
    pop = bit_population(tensors, bitwise=bitwise)
    share = {}
    for k, w in pop.items():
        r = float(np.clip(rates.get(k, 0.5), 0.0, 1.0))
        phi = float(blast_weight.get(k, 1.0)) if blast_weight else 1.0
        share[k] = w * np.sqrt(max(r * (1 - r), 1e-12)) * phi

    total = sum(share.values())
    if total <= 0:
        return proportional_allocation(tensors, n, bitwise)

    n_floor = floor * len(share)
    if n_floor >= n:
        return {k: floor for k in share}
    alloc = _largest_remainder({k: v / total for k, v in share.items()}, n - n_floor)
    return {k: alloc[k] + floor for k in share}


def _largest_remainder(fracs: Mapping[tuple, float], n: int) -> dict[tuple, int]:
    """Apportion `n` by fraction, distributing the rounding remainder fairly."""
    raw = {k: v * n for k, v in fracs.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    left = n - sum(out.values())
    if left:
        order = sorted(raw, key=lambda k: raw[k] - np.floor(raw[k]), reverse=True)
        for k in order[:left]:
            out[k] += 1
    return out


def default_blast_weight(tensors: Mapping[str, MXTensor],
                         bitwise: bool = False) -> dict[tuple, float]:
    """Φ = block size for scale strata, 1 for element strata."""
    out = {}
    for name, mx in tensors.items():
        for site in ("element", "scale"):
            phi = float(mx.block_size) if site == "scale" else 1.0
            widths = range(_site_width(mx, site)) if bitwise else [None]
            for b in widths:
                out[Stratum(name, site, b).key()] = phi
    return out
