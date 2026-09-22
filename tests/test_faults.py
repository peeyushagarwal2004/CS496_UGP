"""Tests for two-site bit-fault injection.

The impact tables are enumerated over the whole (code, bit) space, so these
checks are exhaustive rather than sampled.
"""

import numpy as np
import pytest

from mxfi.codec import quantize
from mxfi.faults import (Fault, apply_scale_fault_to_block, block_of_element,
                         decoded_block_after_scale_fault, element_impact_table,
                         fault_space, inject, scale_impact_table)
from mxfi.formats import ELEMENT_FORMATS, E8M0_NAN, decode_e8m0, get_format

FORMATS = list(ELEMENT_FORMATS)


@pytest.fixture
def tensor():
    rng = np.random.default_rng(0)
    return quantize(rng.standard_normal((4, 128)).astype(np.float32), "e4m3", 32)


# ------------------------------------------------------------------ mechanics

def test_injection_does_not_mutate_the_original(tensor):
    before_codes = tensor.codes.copy()
    before_scales = tensor.scales.copy()
    inject(tensor, Fault("element", (0, 0, 0), 3))
    inject(tensor, Fault("scale", (0, 0), 3))
    assert np.array_equal(tensor.codes, before_codes)
    assert np.array_equal(tensor.scales, before_scales)


def test_injection_is_an_involution(tensor):
    for f in (Fault("element", (1, 2, 3), 5), Fault("scale", (1, 2), 6)):
        assert np.array_equal(inject(inject(tensor, f), f).decode(), tensor.decode())


def test_bad_site_and_bit_are_rejected(tensor):
    with pytest.raises(ValueError):
        Fault("mantissa", (0, 0, 0), 1)
    with pytest.raises(ValueError):
        inject(tensor, Fault("element", (0, 0, 0), 8))    # e4m3 is 8 bits wide
    with pytest.raises(ValueError):
        inject(tensor, Fault("scale", (0, 0), 8))


# ---------------------------------------------------------------- blast radius

@pytest.mark.parametrize("K", [4, 8, 16, 32, 64])
def test_blast_radius_is_one_for_elements_and_K_for_scales(K):
    """The structural asymmetry the whole study rests on."""
    rng = np.random.default_rng(1)
    x = (np.abs(rng.standard_normal((2, 256))) + 0.5).astype(np.float32)
    q = quantize(x, "e4m3", K)
    base = q.decode()

    changed = (inject(q, Fault("element", (0, 0, 0), 6)).decode() != base).sum()
    assert changed == 1

    # a scale flip perturbs every element of its block (all non-zero here)
    changed = (inject(q, Fault("scale", (0, 0), 1)).decode() != base).sum()
    assert changed == K


def test_block_of_element_maps_to_its_scale(tensor):
    idx = (2, 3, 17)
    assert block_of_element(idx) == (2, 3)
    assert tensor.scales[block_of_element(idx)].shape == ()


def test_scale_fault_rescales_the_block_exactly(tensor):
    """Decoded block after a scale flip is the clean block times 2**delta."""
    blk, bit = (1, 2), 3
    _, mult = apply_scale_fault_to_block(int(tensor.scales[blk]), bit)
    clean = tensor.fmt.table()[tensor.codes[blk]] * decode_e8m0(tensor.scales[blk])
    assert np.allclose(decoded_block_after_scale_fault(tensor, blk, bit),
                       clean * mult, rtol=0, atol=0)


# ---------------------------------------------------------------- fault space

def test_fault_space_accounting(tensor):
    fs = fault_space(tensor)
    assert fs["elements"] == 4 * 128
    assert fs["blocks"] == 4 * 4
    assert fs["element_bits"] == 4 * 128 * 8
    assert fs["scale_bits"] == 16 * 8
    assert fs["total_bits"] == fs["element_bits"] + fs["scale_bits"]
    # element:scale bit ratio is exactly K * width / 8
    assert fs["element_bits"] / fs["scale_bits"] == 32 * 8 / 8


def test_fault_space_excludes_padding():
    q = quantize(np.ones((1, 40), np.float32), "e4m3", 32)
    assert fault_space(q, valid_only=True)["elements"] == 40
    assert fault_space(q, valid_only=False)["elements"] == 64


# ------------------------------------------------- element impact enumeration

@pytest.mark.parametrize("name", FORMATS)
def test_element_table_matches_the_alphabet(name):
    """`after` is exactly the table entry of the flipped code, for all codes."""
    f = get_format(name)
    t = element_impact_table(f)
    tab = f.table()
    for b in range(f.width):
        want = tab[np.arange(f.n_codes) ^ (1 << b)]
        got = t["after"][:, b]
        assert np.array_equal(np.isnan(got), np.isnan(want))
        m = ~np.isnan(want)
        assert np.array_equal(got[m], want[m])


@pytest.mark.parametrize("name", FORMATS)
def test_sign_bit_is_invisible_to_the_log_severity_metric(name):
    """Regression guard on a real flaw in the drafts' severity definition.

    ``d = |log v' - log v|`` scores a sign flip as *zero* damage even though
    the relative error is 2.  Any value-aware risk score built on the log
    metric alone would deprioritise the sign bit to nothing.
    """
    f = get_format(name)
    t = element_impact_table(f)
    sb = f.sign_bit
    finite = ~t["from_nonfinite"][:, sb] & ~t["to_nonfinite"][:, sb]
    nonzero = finite & (t["before"][:, sb] != 0)

    assert np.all(t["log_severity"][nonzero, sb] == 0.0)   # the blind spot
    assert np.allclose(t["rel_error"][nonzero, sb], 2.0)   # the real damage


@pytest.mark.parametrize("name", FORMATS)
def test_exponent_bit_severity_doubles_per_position(name):
    """Flipping exponent bit b scales the value by ~2**(2**b)."""
    f = get_format(name)
    t = element_impact_table(f)
    for b in range(f.man_bits, f.man_bits + f.exp_bits):
        m = (~t["from_nonfinite"][:, b] & ~t["to_nonfinite"][:, b]
             & (t["before"][:, b] != 0) & (t["after"][:, b] != 0))
        sev = t["log_severity"][m, b]
        # subnormal <-> normal crossings blur the exact value, so take the mode
        assert np.median(sev) == pytest.approx(2.0 ** (b - f.man_bits), abs=0.5)


@pytest.mark.parametrize("name", FORMATS)
def test_mantissa_bits_are_the_mildest(name):
    """Severity is monotone in bit significance -- the basis for per-bit maps."""
    f = get_format(name)
    t = element_impact_table(f)
    ok = ~t["from_nonfinite"] & ~t["to_nonfinite"] & np.isfinite(t["log_severity"])
    mean = [t["log_severity"][ok[:, b], b].mean() for b in range(f.width)]
    mag_bits = mean[: f.man_bits + f.exp_bits]        # exclude the sign bit
    assert np.all(np.diff(mag_bits) > 0)


def test_only_fp8_formats_can_flip_into_nonfinite():
    """FP6/FP4 have no special codes, so no element flip can produce NaN/Inf."""
    for name in ("e3m2", "e2m3", "e2m1"):
        assert not element_impact_table(get_format(name))["to_nonfinite"].any()
    for name in ("e4m3", "e5m2"):
        assert element_impact_table(get_format(name))["to_nonfinite"].any()


# --------------------------------------------------- scale impact enumeration

def test_scale_delta_is_plus_or_minus_two_to_the_bit():
    t = scale_impact_table()
    for b in range(8):
        s = np.arange(256)
        want = np.where((s >> b) & 1, -(1 << b), (1 << b))
        assert np.array_equal(t["delta_exp"][:, b], want)


def test_scale_multiplier_matches_delta():
    t = scale_impact_table()
    ok = ~t["to_nan"] & ~t["from_nan"] & (np.abs(t["delta_exp"]) < 1000)
    assert np.allclose(t["multiplier"][ok],
                       np.exp2(t["delta_exp"][ok].astype(float)))


def test_scale_msb_is_the_single_most_damaging_bit():
    """Bit 7 shifts the exponent by 128 -- past the float32 range entirely."""
    t = scale_impact_table()
    assert np.all(np.abs(t["delta_exp"][:, 7]) == 128)
    assert t["overflows_f32"][:, 7].all()
    assert not t["overflows_f32"][:, :7].any()


def test_exactly_eight_bytes_flip_into_the_nan_scale():
    """255 has eight single-bit neighbours; each NaNs an entire block."""
    t = scale_impact_table()
    assert t["to_nan"].sum() == 8
    assert np.array_equal(np.flatnonzero(t["to_nan"].any(axis=1)),
                          np.sort(E8M0_NAN ^ (1 << np.arange(8))))


def test_scale_severity_dominates_element_severity_per_flip():
    """Per flip, the worst scale bit outweighs the worst element bit by K.

    This is the ``K*H_R`` vs ``32*H_r`` parity argument of the drafts, but
    stated as severity per individual fault rather than per block.
    """
    f = get_format("e4m3")
    K = 32
    worst_elem = np.nanmax(np.where(
        np.isfinite(element_impact_table(f)["log_severity"]),
        element_impact_table(f)["log_severity"], np.nan))
    st = scale_impact_table()
    worst_scale_per_elem = st["log_severity"][st["log_severity"] < np.inf].max()
    assert worst_scale_per_elem * K > worst_elem
