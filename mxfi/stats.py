"""Estimators and confidence intervals for fault-injection campaigns.

A campaign measures a failure rate (SDC or top-1 change) from a finite number
of injections, so every reported number needs an interval.  Two regimes:

* **uniform sampling** -- outcomes are i.i.d. Bernoulli, so a Wilson interval
  on the raw counts is exact enough and behaves properly when the observed
  rate is 0 or 1 (where the normal approximation collapses).
* **stratified sampling** -- each stratum contributes ``W_h`` of the
  population and is measured with ``n_h`` draws, so the estimate is the
  weighted sum and the variance is ``Σ W_h² p_h(1-p_h)/n_h``.

:func:`injection_reduction` turns the variance difference between the two into
the headline "fewer injections at the same confidence" number.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import numpy as np
from scipy.stats import norm

__all__ = ["Estimate", "wilson_interval", "binomial_rate", "StratumCount",
           "stratified_rate", "required_samples", "injection_reduction"]


@dataclass(frozen=True)
class Estimate:
    """A rate with a confidence interval and the effort that produced it."""

    rate: float
    lo: float
    hi: float
    n: int
    conf: float = 0.95

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2

    @property
    def stderr(self) -> float:
        return self.half_width / norm.ppf(0.5 + self.conf / 2)

    def __str__(self) -> str:
        return (f"{self.rate:.4%} [{self.lo:.4%}, {self.hi:.4%}] "
                f"({self.conf:.0%} CI, n={self.n})")


def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for `k` successes in `n` Bernoulli trials.

    Preferred over the normal approximation because it stays inside [0, 1] and
    remains informative when ``k == 0`` -- the common case for a robust layer.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = norm.ppf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def binomial_rate(outcomes, conf: float = 0.95) -> Estimate:
    """Failure rate from a flat array of 0/1 outcomes (uniform sampling)."""
    y = np.asarray(outcomes, dtype=bool)
    n, k = int(y.size), int(y.sum())
    lo, hi = wilson_interval(k, n, conf)
    return Estimate(k / n if n else 0.0, lo, hi, n, conf)


@dataclass(frozen=True)
class StratumCount:
    """Per-stratum tally: `k` failures in `n` draws, covering share `W` of bits."""

    k: int
    n: int
    W: float

    @property
    def rate(self) -> float:
        return self.k / self.n if self.n else 0.0

    @property
    def var_contrib(self) -> float:
        """``W² p(1-p)/n`` -- this stratum's share of the estimator variance."""
        if self.n <= 1:
            return self.W ** 2 * 0.25          # worst case, unmeasurable
        p = self.rate
        return self.W ** 2 * p * (1 - p) / self.n


def stratified_rate(strata: Mapping[tuple, StratumCount],
                    conf: float = 0.95) -> Estimate:
    """Combine per-stratum tallies into one model-wide rate.

    Shares are renormalised to sum to 1, so a partially covered fault space
    still yields a rate for the region actually sampled.
    """
    total_w = sum(s.W for s in strata.values())
    if total_w <= 0:
        return Estimate(0.0, 0.0, 1.0, 0, conf)

    norm_ = {k: StratumCount(s.k, s.n, s.W / total_w) for k, s in strata.items()}
    rate = sum(s.W * s.rate for s in norm_.values())
    var = sum(s.var_contrib for s in norm_.values())
    z = norm.ppf(0.5 + conf / 2)
    half = z * sqrt(max(var, 0.0))
    n = sum(s.n for s in norm_.values())
    return Estimate(rate, max(0.0, rate - half), min(1.0, rate + half), n, conf)


def required_samples(rate: float, target_half_width: float,
                     conf: float = 0.95) -> int:
    """Injections needed to pin `rate` to +-`target_half_width`."""
    if target_half_width <= 0:
        raise ValueError("target_half_width must be positive")
    z = norm.ppf(0.5 + conf / 2)
    r = float(np.clip(rate, 0.0, 1.0))
    return int(np.ceil(z * z * max(r * (1 - r), 1e-12) / target_half_width ** 2))


def injection_reduction(uniform: Estimate, stratified: Estimate) -> float:
    """How many times fewer injections stratification buys at equal precision.

    Both estimators reach a given half-width at ``n ∝ variance``, so the ratio
    of the sample sizes that would give matching precision is::

        (n_uniform_needed / n_uniform_actual) vs the same for stratified

    Reported as ``n_uniform_equivalent / n_stratified``: a value of 10 means
    the uniform campaign would need ten times the injections to match.
    """
    if stratified.stderr <= 0 or uniform.stderr <= 0:
        return float("nan")
    # n needed scales as 1/stderr**2 at fixed rate
    n_uniform_equiv = uniform.n * (uniform.stderr / stratified.stderr) ** 2
    return n_uniform_equiv / stratified.n
