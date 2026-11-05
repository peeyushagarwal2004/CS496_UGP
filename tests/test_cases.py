"""Tests for the single-bit flip case taxonomy.

The element tables cover the whole (code, bit) space and the scale checks cover
all 256 x 8 byte flips, so these are exhaustive rather than sampled.
"""

import numpy as np
import pytest

from mxfi.cases import (bit_field, classify_elements, classify_scales, label,
                        element_cases, scale_cases, value_class)
from mxfi.codec import quantize
from mxfi.faults import Fault, inject
from mxfi.formats import ELEMENT_FORMATS, get_format

FORMATS = list(ELEMENT_FORMATS)


@pytest.mark.parametrize("name", FORMATS)
def test_every_element_flip_has_one_case(name):
    t = element_cases(get_format(name))
    assert (t["transition"] != "").all()
    assert set(t["field"].ravel()) == {"sign", "exponent", "mantissa"}


@pytest.mark.parametrize("name", FORMATS)
def test_fields_partition_the_bits(name):
    f = get_format(name)
    fields = [bit_field(f, b) for b in range(f.width)]
    assert fields.count("sign") == 1
    assert fields.count("exponent") == f.exp_bits
    assert fields.count("mantissa") == f.man_bits


@pytest.mark.parametrize("name", ["e3m2", "e2m3", "e2m1"])
def test_formats_without_specials_never_reach_one(name):
    t = element_cases(get_format(name))
    assert not (t["transition"] == "to_special").any()
    assert np.isfinite(t["rel_error"]).all()


def test_e4m3_nan_neighbours():
    f = get_format("e4m3")
    t = element_cases(f)
    near = np.flatnonzero(t["near_special"][:, 0] & (value_class(f) != "special"))
    mags = sorted({float(abs(f.table()[c])) for c in near})
    assert mags == [1.875, 30.0, 120.0, 240.0, 352.0, 416.0, 448.0]


@pytest.mark.parametrize("name", FORMATS)
def test_sign_flip_of_zero_is_masked(name):
    f = get_format(name)
    t = element_cases(f)
    assert t["transition"][0, f.sign_bit] == "masked"
    assert t["rel_error"][0, f.sign_bit] == 0


def test_scale_delta_matches_xor():
    x = np.repeat(np.arange(255), 8)
    bit = np.tile(np.arange(8), 255)
    c = scale_cases(x, bit, np.ones_like(x, float), np.zeros_like(x, bool))
    assert np.array_equal(c["x_new"], x + c["delta"])
    assert np.array_equal(np.abs(c["delta"]), 1 << bit)


def test_scale_to_nan_only_from_seven_ones():
    x = np.repeat(np.arange(255), 8)
    bit = np.tile(np.arange(8), 255)
    c = scale_cases(x, bit, np.ones_like(x, float), np.zeros_like(x, bool))
    seven = np.array([bin(int(v)).count("1") == 7 for v in x])
    assert np.array_equal(c["case"] == "to_nan", seven & (c["x_new"] == 255))


def test_scale_cases_follow_delta():
    x = np.array([0b01110101] * 8)              # 117, a typical weight scale
    c = scale_cases(x, np.arange(8), np.ones(8), np.zeros(8, bool))
    assert list(c["delta"]) == [-1, 2, -4, 8, -16, -32, -64, 128]
    assert list(c["case"]) == ["down_small", "up_small", "erase", "up_large",
                               "erase", "erase", "erase", "up_overflow"]


def test_zero_block_is_masked_unless_nan():
    c = scale_cases(np.array([127, 127, 0]), np.array([7, 0, 7]),
                    np.zeros(3), np.ones(3, bool))
    assert list(c["case"]) == ["to_nan", "zero_block", "zero_block"]


def test_element_cases_agree_with_injection():
    rng = np.random.default_rng(0)
    for name in FORMATS:
        mx = quantize(rng.standard_normal((2, 64)).astype(np.float32), name, 32)
        f = mx.fmt
        base = mx.decode()
        for _ in range(50):
            idx = (int(rng.integers(2)), int(rng.integers(2)), int(rng.integers(32)))
            b = int(rng.integers(f.width))
            c = classify_elements(mx, tuple(np.array([i]) for i in idx), np.array([b]))
            diff = np.abs(inject(mx, Fault("element", idx, b)).decode() - base)
            scale = 2.0 ** (int(mx.scales[idx[:2]]) - 127)
            want = c["rel_error"][0] * 2.0 ** f.emax * scale
            if np.isfinite(want):
                assert np.isclose(diff.max(), want, rtol=1e-6, atol=0)
            else:
                assert not np.isfinite(diff).all()


@pytest.mark.parametrize("name", FORMATS)
def test_every_flip_into_a_special_code_is_zero_to_one(name):
    t = element_cases(get_format(name))
    codes = np.arange(get_format(name).n_codes)[:, None]
    bits = np.arange(get_format(name).width)[None, :]
    was = np.broadcast_to((codes >> bits) & 1, t["transition"].shape)
    assert (was[t["transition"] == "to_special"] == 0).all()


def test_demotion_flags_only_the_lone_top_binade_element():
    f = get_format("e3m2")
    # block 0: one top-binade value (16) among small ones; block 1: two of them
    x = np.zeros((1, 64), np.float32)
    x[0, :32] = 0.5
    x[0, 0] = 16.0
    x[0, 32:] = 0.5
    x[0, 32], x[0, 33] = 16.0, 20.0
    mx = quantize(x, f, 32)
    exp_bit = f.man_bits                     # lowest exponent bit
    idx = (np.array([0, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]))
    c = classify_elements(mx, idx, np.array([exp_bit] * 3))
    assert list(c["demotes_max"]) == [True, False, False]


def test_scale_band_flags():
    mx = quantize(np.random.default_rng(2).standard_normal((4, 64)).astype(np.float32),
                  "e4m3", 32)
    lo, hi = int(mx.scales.min()), int(mx.scales.max())
    idx = (np.zeros(8, int), np.zeros(8, int))
    c = classify_scales(mx, idx, np.arange(8))
    assert (c["out_of_band"] == ((c["x_new"] < lo) | (c["x_new"] > hi))).all()
    assert c["above_band"][7]                # +128 is always far above any clean band


def test_scale_cases_agree_with_injection():
    rng = np.random.default_rng(1)
    mx = quantize(rng.standard_normal((2, 64)).astype(np.float32), "e4m3", 32)
    for b in range(8):
        idx = (np.array([0]), np.array([1]))
        c = classify_scales(mx, idx, np.array([b]))
        out = inject(mx, Fault("scale", (0, 1), b)).decode()[0, 32:64]
        ref = mx.decode()[0, 32:64]
        nz = ref != 0
        with np.errstate(over="ignore", invalid="ignore"):
            ratio = out[nz] / ref[nz]
        if c["case"][0] in ("up_small", "up_large", "down_small", "erase"):
            assert np.allclose(ratio, 2.0 ** c["delta"][0])


@pytest.mark.parametrize("name", FORMATS)
def test_every_element_case_has_a_label(name):
    t = element_cases(get_format(name))
    labels = {label(c) for c in t["case"].ravel()}
    assert labels <= {"E0", "E1", "E2", "E3", "E3a", "E3b", "E4", "E4a", "E4b",
                      "E5", "E5a", "E6", "E6a", "E7"}


def test_every_scale_flip_has_a_label():
    x = np.repeat(np.arange(256), 16)
    bit = np.tile(np.repeat(np.arange(8), 2), 256)
    zero = np.tile([False, True], 256 * 8)
    c = scale_cases(x, bit, np.ones_like(x, float), zero)
    labels = [label(s) for s in c["case"]]
    assert set(labels) == {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    assert (c["case"][x == 255] == "from_nan").all()
