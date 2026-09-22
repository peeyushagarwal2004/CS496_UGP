"""Tests for fault-space sampling and the campaign estimators."""

import numpy as np
import pytest

from mxfi.codec import quantize
from mxfi.sampling import (Stratum, bit_population, default_blast_weight,
                           neyman_allocation, proportional_allocation,
                           sample_stratified, sample_uniform)
from mxfi.stats import (Estimate, StratumCount, binomial_rate,
                        injection_reduction, required_samples, stratified_rate,
                        wilson_interval)


@pytest.fixture
def tensors():
    rng = np.random.default_rng(0)
    return {
        "conv1": quantize(rng.standard_normal((16, 128)).astype(np.float32), "e4m3", 32),
        "fc":    quantize(rng.standard_normal((10, 64)).astype(np.float32), "e2m1", 16),
    }


# ------------------------------------------------------------- bit population

def test_bit_population_matches_storage(tensors):
    pop = bit_population(tensors)
    assert pop[Stratum("conv1", "element").key()] == 16 * 128 * 8
    assert pop[Stratum("conv1", "scale").key()] == 16 * 4 * 8
    assert pop[Stratum("fc", "element").key()] == 10 * 64 * 4
    assert pop[Stratum("fc", "scale").key()] == 10 * 4 * 8


def test_bitwise_population_splits_by_position(tensors):
    pop = bit_population(tensors, bitwise=True)
    # one entry per bit position, each covering all slots
    assert pop[Stratum("conv1", "element", 0).key()] == 16 * 128
    assert sum(v for k, v in pop.items() if k[0] == "conv1" and k[1] == "element") \
        == 16 * 128 * 8


def test_population_excludes_padding():
    q = quantize(np.ones((1, 40), np.float32), "e4m3", 32)
    pop = bit_population({"t": q})
    assert pop[Stratum("t", "element").key()] == 40 * 8      # not 64 * 8


# -------------------------------------------------------------------- uniform

def test_uniform_sampling_hits_sites_in_proportion_to_bits(tensors):
    """Scale bits are ~3% of storage, so they must be ~3% of uniform draws."""
    rng = np.random.default_rng(1)
    n = 40000
    faults = sample_uniform(tensors, n, rng)
    assert len(faults) == n

    pop = bit_population(tensors)
    want = sum(v for k, v in pop.items() if k[1] == "scale") / sum(pop.values())
    got = sum(f.site == "scale" for f in faults) / n
    assert got == pytest.approx(want, abs=4 * np.sqrt(want * (1 - want) / n))
    assert want < 0.05, "scale bits should be a small minority of storage"


def test_uniform_draws_are_in_range(tensors):
    rng = np.random.default_rng(2)
    for f in sample_uniform(tensors, 2000, rng):
        mx = tensors[f.tensor]
        if f.site == "element":
            assert 0 <= f.bit < mx.fmt.width
            assert mx.valid_mask()[f.fault.index]
        else:
            assert 0 <= f.bit < 8
            assert len(f.fault.index) == mx.scales.ndim


# ----------------------------------------------------------------- allocation

def test_proportional_allocation_sums_to_n(tensors):
    for n in (7, 100, 1001):
        alloc = proportional_allocation(tensors, n)
        assert sum(alloc.values()) == n


def test_neyman_gives_more_budget_to_uncertain_strata(tensors):
    """A stratum near r=0.5 needs more samples than one near r=0."""
    keys = list(bit_population(tensors))
    rates = {k: 0.001 for k in keys}
    rates[Stratum("conv1", "scale").key()] = 0.5
    alloc = neyman_allocation(tensors, 10000, rates)
    assert sum(alloc.values()) == 10000

    # despite holding only ~3% of the bits, the uncertain stratum wins budget
    scale_share = alloc[Stratum("conv1", "scale").key()] / 10000
    pop = bit_population(tensors)
    bit_share = pop[Stratum("conv1", "scale").key()] / sum(pop.values())
    assert scale_share > bit_share


def test_blast_weight_upweights_the_scale_stratum(tensors):
    """Phi = K makes the K-value blast radius visible to the allocator."""
    keys = list(bit_population(tensors))
    rates = {k: 0.1 for k in keys}
    plain = neyman_allocation(tensors, 10000, rates)
    withphi = neyman_allocation(tensors, 10000, rates,
                                blast_weight=default_blast_weight(tensors))
    k = Stratum("conv1", "scale").key()
    assert withphi[k] > plain[k]


def test_allocation_floor_is_respected(tensors):
    alloc = neyman_allocation(tensors, 8, {k: 0.0 for k in bit_population(tensors)},
                              floor=2)
    assert all(v >= 2 for v in alloc.values())


# ----------------------------------------------------------------- stratified

def test_stratified_weights_sum_to_one(tensors):
    rng = np.random.default_rng(3)
    alloc = proportional_allocation(tensors, 1000)
    faults, w = sample_stratified(tensors, alloc, rng)
    assert len(faults) == len(w) == 1000
    assert w.sum() == pytest.approx(1.0)


def test_stratified_respects_the_requested_allocation(tensors):
    rng = np.random.default_rng(4)
    alloc = {Stratum("conv1", "scale").key(): 300,
             Stratum("fc", "element").key(): 700}
    faults, w = sample_stratified(tensors, alloc, rng)
    got = {}
    for f in faults:
        got[(f.tensor, f.site)] = got.get((f.tensor, f.site), 0) + 1
    assert got[("conv1", "scale")] == 300
    assert got[("fc", "element")] == 700


def test_unknown_stratum_is_rejected(tensors):
    rng = np.random.default_rng(5)
    with pytest.raises(KeyError):
        sample_stratified(tensors, {Stratum("nope", "element").key(): 10}, rng)


def test_stratified_estimate_is_unbiased_against_a_known_truth(tensors):
    """The headline property: a skewed allocation still recovers the true rate.

    Ground truth is a synthetic oracle whose failure rate differs 100x between
    sites.  A deliberately lopsided allocation -- most budget on the rare scale
    stratum -- must still estimate the *population* rate correctly.
    """
    rng = np.random.default_rng(6)
    true_rate = {"element": 0.01, "scale": 0.90}

    pop = bit_population(tensors)
    total = sum(pop.values())
    truth = sum(pop[k] / total * true_rate[k[1]] for k in pop)

    # 80% of the budget on scale strata, which hold ~3% of the bits
    alloc = {}
    scale_keys = [k for k in pop if k[1] == "scale"]
    elem_keys = [k for k in pop if k[1] == "element"]
    for k in scale_keys:
        alloc[k] = 8000 // len(scale_keys)
    for k in elem_keys:
        alloc[k] = 2000 // len(elem_keys)

    faults, w = sample_stratified(tensors, alloc, rng)
    outcomes = np.array([rng.random() < true_rate[f.site] for f in faults])

    est = float(np.sum(w * outcomes))
    assert est == pytest.approx(truth, abs=0.01), f"est {est} vs truth {truth}"

    # the same draws, aggregated through the stratum estimator
    counts = {}
    for k in alloc:
        m = np.array([(f.tensor, f.site) == (k[0], k[1]) for f in faults])
        counts[k] = StratumCount(int(outcomes[m].sum()), int(m.sum()),
                                 pop[k] / total)
    combined = stratified_rate(counts)
    assert combined.rate == pytest.approx(truth, abs=0.01)
    assert combined.lo <= truth <= combined.hi


# ---------------------------------------------------------------- estimators

def test_wilson_handles_the_degenerate_cases():
    lo, hi = wilson_interval(0, 100)
    assert lo == pytest.approx(0.0, abs=1e-12)   # exact zero up to fp roundoff
    assert 0 < hi < 0.05                         # informative with zero failures
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0, abs=1e-12) and 0.95 < lo < 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_brackets_the_point_estimate():
    for k, n in [(1, 10), (5, 10), (9, 10), (37, 1000)]:
        lo, hi = wilson_interval(k, n)
        assert lo <= k / n <= hi


def test_wilson_interval_narrows_with_n():
    widths = [np.diff(wilson_interval(int(0.1 * n), n))[0] for n in (100, 1000, 10000)]
    assert widths[0] > widths[1] > widths[2]


def test_binomial_rate_matches_counts():
    y = np.zeros(1000, bool); y[:37] = True
    e = binomial_rate(y)
    assert e.rate == pytest.approx(0.037) and e.n == 1000
    assert e.lo < 0.037 < e.hi


def test_required_samples_scales_inversely_with_squared_error():
    a = required_samples(0.5, 0.01)
    b = required_samples(0.5, 0.005)
    assert b == pytest.approx(4 * a, rel=0.01)
    assert required_samples(0.5, 0.01) > required_samples(0.01, 0.01)


def test_injection_reduction_rewards_lower_variance():
    wide = Estimate(0.1, 0.05, 0.15, 1000)
    tight = Estimate(0.1, 0.09, 0.11, 1000)
    assert injection_reduction(wide, tight) > 1.0
    assert injection_reduction(wide, wide) == pytest.approx(1.0)
