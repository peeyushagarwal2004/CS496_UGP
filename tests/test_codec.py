"""Correctness tests for the MX codec.

The element alphabets have at most 256 codes, so most properties here are
checked *exhaustively* over the whole code space rather than sampled.
"""

import numpy as np
import pytest

from mxfi.codec import (SCALE_MODES, _round_to_alphabet, headroom, quantize,
                        quantize_dequantize)
from mxfi.formats import (ELEMENT_FORMATS, E8M0_BIAS, decode_e8m0,
                          encode_e8m0, get_format)

FORMATS = list(ELEMENT_FORMATS)


# ------------------------------------------------------------------ formats

@pytest.mark.parametrize("name", FORMATS)
def test_alphabet_geometry(name):
    """Field widths, bias and sign symmetry match the OCP spec."""
    f = get_format(name)
    t = f.table()
    assert t.shape == (f.n_codes,)
    assert f.width == 1 + f.exp_bits + f.man_bits
    half = f.n_codes // 2
    pos, neg = t[:half], t[half:]
    fin = np.isfinite(pos) & np.isfinite(neg)
    assert np.array_equal(pos[fin], -neg[fin])


@pytest.mark.parametrize("name", FORMATS)
def test_positive_half_is_monotone(name):
    """Magnitude increases with code -- the property _pos_alphabet relies on."""
    f = get_format(name)
    vals = f.table()[: f.n_codes // 2]
    vals = vals[np.isfinite(vals)]
    assert np.all(np.diff(vals) > 0)


def test_known_spec_values():
    """Spot-check the published constants of each MX element format."""
    assert get_format("e4m3").max_normal == 448.0
    assert get_format("e5m2").max_normal == 57344.0
    assert get_format("e3m2").max_normal == 28.0
    assert get_format("e2m3").max_normal == 7.5
    assert get_format("e2m1").max_normal == 6.0
    v = sorted({abs(x) for x in get_format("e2m1").table().tolist()})
    assert v == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for n in ("e3m2", "e2m3", "e2m1"):
        assert np.all(np.isfinite(get_format(n).table()))
    assert np.isnan(get_format("e4m3").table()).sum() == 2      # +-NaN, no Inf
    assert np.isinf(get_format("e5m2").table()).sum() == 2


def test_emax_matches_top_binade():
    """emax is the exponent the scale rule aligns the block maximum onto."""
    assert [get_format(n).emax for n in
            ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1")] == [8, 15, 4, 2, 2]


def test_e8m0_roundtrip():
    """E8M0 covers 2**-127 .. 2**127 exactly, with 255 reserved for NaN."""
    exps = np.arange(-127, 128)
    assert np.array_equal(encode_e8m0(exps).astype(int), exps + E8M0_BIAS)
    assert np.array_equal(decode_e8m0(encode_e8m0(exps)),
                          np.exp2(exps.astype(np.float32)))
    assert np.isnan(decode_e8m0(np.uint8(255)))
    assert encode_e8m0(np.array([-200, 200])).tolist() == [0, 254]


# -------------------------------------------------------------------- codec

@pytest.mark.parametrize("name", FORMATS)
def test_representable_values_are_exact(name):
    """Anything already in the alphabet re-encodes to itself, bit for bit.

    Checked under ``scale_mode="fit"``; the OCP rule deliberately clips the
    block maximum, which `test_ocp_scale_rule_clips_the_block_maximum` pins.
    """
    f = get_format(name)
    t = f.table()
    vals = t[np.isfinite(t)].astype(np.float32)
    q = quantize(vals[:, None], f, block_size=1, scale_mode="fit")
    assert np.array_equal(q.decode().ravel(), vals)


@pytest.mark.parametrize("name", FORMATS)
def test_rounding_is_nearest_even(name):
    """Rounding matches brute-force nearest with tie-to-even-mantissa.

    Exercised on `_round_to_alphabet` directly: the scale rule would otherwise
    renormalise every probe into the top binade and mask the tie behaviour.
    """
    f = get_format(name)
    t = f.table()
    pos = np.unique(np.abs(t[np.isfinite(t)]))
    mids = (pos[:-1] + pos[1:]) / 2
    probe = np.concatenate([pos, mids, mids * 0.999, mids * 1.001,
                            [pos[-1] * 4]])          # saturating probe
    probe = np.abs(probe).astype(np.float32)

    got_codes = _round_to_alphabet(probe, f)
    got = np.abs(t[got_codes])
    man_mask = (1 << f.man_bits) - 1
    for a, g, c in zip(probe, got, got_codes):
        if a >= pos[-1]:
            assert g == pos[-1], f"{f.name}: {a} must saturate"
            continue
        d = np.abs(pos - a)
        best = np.flatnonzero(d == d.min())
        if len(best) == 1:
            assert g == pos[best[0]], f"{f.name}: {a} -> {g}"
        else:
            cand = pos[best]
            man = [int(np.flatnonzero(t == v)[0]) & man_mask for v in cand]
            want = cand[int(np.argmin([m % 2 for m in man]))]
            assert g == want, f"{f.name}: tie at {a} -> {g}, want {want}"
            assert int(c) % 2 == 0, "tie must resolve to the even code"


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("K", [1, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("mode", SCALE_MODES)
def test_quantisation_is_idempotent(name, K, mode):
    """Q(Q(x)) == Q(x): the strongest single check on scale/element agreement."""
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((6, 128)) * rng.lognormal(0, 2, (6, 1))).astype(np.float32)
    once = quantize_dequantize(x, name, K, scale_mode=mode)
    twice = quantize_dequantize(once, name, K, scale_mode=mode)
    assert np.array_equal(once, twice)


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("mode", SCALE_MODES)
def test_scaled_block_lands_in_range(name, mode):
    """No element ever overflows the alphabet, and the max lands where promised."""
    f = get_format(name)
    rng = np.random.default_rng(3)
    for _ in range(200):
        blk = (rng.standard_normal(32) * 10.0 ** rng.uniform(-20, 20)).astype(np.float32)
        q = quantize(blk, f, block_size=32, scale_mode=mode)
        if not 0 < int(q.scales[0]) < 254:      # scale itself railed; skip
            continue
        assert np.abs(f.table()[q.codes]).max() <= f.max_normal
        scaled = np.abs(blk) / decode_e8m0(q.scales)[0]
        if mode == "ocp":                       # top binade, clipping allowed
            assert 2 ** f.emax <= scaled.max() < 2 ** (f.emax + 1)
        else:                                   # provably no clipping
            assert scaled.max() <= f.max_normal


@pytest.mark.parametrize("name", FORMATS)
def test_scale_rule_matches_ocp_formula(name):
    """scales == clamp(floor(log2(amax)) - emax, -127, 127) + 127."""
    f = get_format(name)
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((5, 96)) * 10.0 ** rng.uniform(-6, 6, (5, 96))).astype(np.float32)
    q = quantize(x, f, block_size=32)
    amax = np.abs(x.reshape(5, 3, 32)).max(-1)
    want = np.clip(np.floor(np.log2(amax.astype(np.float64))) - f.emax, -127, 127) + 127
    assert np.array_equal(q.scales.astype(int), want.astype(int))


# ------------------------------------------------- the clipping confound

def test_headroom_values():
    """Fraction of the top binade each format can represent."""
    assert headroom(get_format("e4m3")) == 0.875     # 448 / 512
    assert headroom(get_format("e5m2")) == 0.875     # 57344 / 65536
    assert headroom(get_format("e3m2")) == 0.875     # 28 / 32
    assert headroom(get_format("e2m3")) == 0.9375    # 7.5 / 8
    assert headroom(get_format("e2m1")) == 0.75      # 6 / 8  <- MXFP4 is worst


@pytest.mark.parametrize("name", FORMATS)
def test_ocp_scale_rule_clips_the_block_maximum(name):
    """Under OCP the block max is clipped exactly when it exceeds the headroom.

    This is spec-faithful, not a defect -- but it is a confound a format
    comparison must control, hence ``scale_mode="fit"``.
    """
    f = get_format(name)
    lo, hi = f.max_normal, 2.0 ** (f.emax + 1)
    v = np.array([lo + 0.6 * (hi - lo)], np.float32)   # well inside the dead zone

    got = quantize(v, f, 1, scale_mode="ocp").decode()[0]
    assert got == np.float32(lo), f"{f.name}: OCP must saturate the block max"

    got_fit = quantize(v, f, 1, scale_mode="fit").decode()[0]
    assert got_fit != got, f"{f.name}: fit must escape the dead zone"
    assert abs(got_fit - v[0]) < abs(got - v[0]), "fit must be strictly closer"


def test_fit_mode_never_clips_the_block_maximum():
    """Across formats and blocks, 'fit' keeps the block max representable."""
    rng = np.random.default_rng(23)
    for name in FORMATS:
        f = get_format(name)
        x = (rng.standard_normal((40, 32)) * 10.0 ** rng.uniform(-8, 8, (40, 1))).astype(np.float32)
        q = quantize(x, f, 32, scale_mode="fit")
        scaled = np.abs(x) / decode_e8m0(q.scales)   # (40, 1) broadcasts over (40, 32)
        ok = np.broadcast_to((q.scales > 0) & (q.scales < 254), scaled.shape)
        assert np.all(scaled[ok] <= f.max_normal * (1 + 1e-6))


def test_all_zero_block():
    """An all-zero block decodes to zeros and does not produce a NaN scale."""
    q = quantize(np.zeros((2, 32), np.float32), "e4m3", 32)
    assert np.all(q.scales != 255)
    assert np.array_equal(q.decode(), np.zeros((2, 32), np.float32))


def test_nonfinite_input_is_not_sanitised():
    """Non-finite values propagate rather than being laundered into finite ones."""
    x = np.array([[np.nan, np.inf, -np.inf, 1.0] * 8], np.float32)
    assert np.isnan(quantize_dequantize(x, "e4m3", 32)).all()


# --------------------------------------------------------- shape / blocking

@pytest.mark.parametrize("shape,axis,K", [
    ((64,), 0, 32), ((3, 64), -1, 32), ((3, 64), 1, 16),
    ((2, 3, 40), -1, 8), ((7, 5), 0, 4), ((2, 3, 4, 10), 2, 32),
])
def test_shape_and_axis_roundtrip(shape, axis, K):
    """Decoding restores the original shape for any axis, padded or not."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape).astype(np.float32)
    q = quantize(x, "e4m3", K, axis=axis)
    assert q.decode().shape == shape
    n = shape[axis]
    assert q.pad == (-n) % K
    assert q.codes.shape[-2:] == ((n + q.pad) // K, K)


def test_axis_choice_changes_blocking_not_shape():
    """Blocking down rows vs across columns is a real, different grouping."""
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((32, 32)) * 10.0 ** rng.uniform(-4, 4, (32, 1))).astype(np.float32)
    a = quantize_dequantize(x, "e2m1", 32, axis=-1)
    b = quantize_dequantize(x, "e2m1", 32, axis=0)
    assert a.shape == b.shape == x.shape
    assert not np.array_equal(a, b)


def test_padding_is_marked_invalid_and_inert():
    """Pad lanes are flagged, and they do not perturb the real values."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 40)).astype(np.float32)
    q = quantize(x, "e4m3", 32)
    assert q.pad == 24
    m = q.valid_mask()
    assert m.sum() == 2 * 40 and (~m).sum() == 2 * 24
    assert np.all(q.codes[~m] == 0)
    ref = quantize(np.pad(x, ((0, 0), (0, 24))), "e4m3", 32).decode()[:, :40]
    assert np.array_equal(q.decode(), ref)


def test_bit_budget_accounting():
    """Reported storage matches the two-site geometry of the fault model."""
    q = quantize(np.zeros((4, 128), np.float32), "e2m1", 32)
    assert q.n_blocks == 4 * 4 and q.n_elements == 4 * 128
    assert q.scale_bits == 8 * 16
    assert q.element_bits == 4 * 512
    assert q.element_bits / q.scale_bits == pytest.approx(16.0)


def test_copy_is_deep():
    q = quantize(np.ones((1, 32), np.float32), "e4m3", 32)
    c = q.copy()
    c.codes[0, 0, 0] ^= 0xFF
    c.scales[0, 0] ^= 0xFF
    assert q.codes[0, 0, 0] != c.codes[0, 0, 0]
    assert q.scales[0, 0] != c.scales[0, 0]


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2", "e3m2", "e2m1"])
def test_nan_input_makes_its_block_nan_and_only_its_block(fmt):
    """A NaN input must survive re-quantisation, as the OCP conversion implies.

    Mapping it to zero instead silently erased every NaN that reached an
    activation-quantised layer, which hid the NaN pathway from activation faults.
    """
    x = np.linspace(-1, 1, 64, dtype=np.float32).reshape(1, 64)
    x[0, 5] = np.nan
    out = quantize(x, fmt, block_size=32).decode()
    assert np.isnan(out[0, :32]).all()
    assert np.isfinite(out[0, 32:]).all()
    clean = quantize(np.nan_to_num(x), fmt, block_size=32).decode()
    assert np.array_equal(out[0, 32:], clean[0, 32:])
