"""Bit-addressable MX (microscaling) codec.

Encoding a tensor produces two *separately addressable* bit arrays -- the
per-block E8M0 scales and the per-element codes -- which is what makes the two
fault sites of this study independently injectable::

    scales : uint8[..., n_blocks]              <- site X, 8 bits each
    codes  : uint8[..., n_blocks, block_size]  <- site r, fmt.width bits each

Two scale-selection rules are offered, because the choice is a confound the
reliability study has to control rather than inherit:

``scale_mode="ocp"`` (default) is the OCP MX v1.0 shared-exponent rule::

    X = clamp(floor(log2(max|v| in block)) - fmt.emax, -127, 127)

It aligns the block maximum with the format's top binade.  Because
``max_normal`` lies *below* the top of that binade, the largest element in a
block can saturate -- by up to 12.5% for e4m3/e5m2/e3m2, 6.25% for e2m3 and
25% for e2m1.  See `headroom`.

``scale_mode="fit"`` instead picks the smallest scale that cannot clip::

    X = clamp(ceil(log2(max|v| in block / fmt.max_normal)), -127, 127)

Use it to check that a format comparison is measuring fault behaviour and not
the differing clipping headroom of MXFP4 versus MXFP8.

Elements are round-to-nearest-even onto the format alphabet and saturate at
max_normal (quantisation never emits Inf/NaN).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .formats import E8M0_NAN, ElementFormat, decode_e8m0, encode_e8m0, get_format

__all__ = ["MXTensor", "quantize", "quantize_dequantize", "headroom",
           "SCALE_MODES"]

SCALE_MODES = ("ocp", "fit")


def headroom(fmt: ElementFormat) -> float:
    """Fraction of the top binade the format can actually represent.

    ``max_normal / 2**(emax+1)``.  The shortfall is the worst-case relative
    clipping the OCP scale rule can inflict on a block maximum.
    """
    return fmt.max_normal / 2.0 ** (fmt.emax + 1)


@lru_cache(maxsize=None)
def _pos_alphabet(fmt: ElementFormat) -> tuple[np.ndarray, np.ndarray]:
    """Finite non-negative magnitudes of `fmt`, ascending, with their codes.

    Sign-magnitude float codes are monotone in magnitude, so the positive half
    of the code space is already sorted; we only drop the special codes.
    """
    half = fmt.n_codes // 2
    vals = fmt.table()[:half]
    codes = np.arange(half, dtype=np.uint8)
    keep = np.isfinite(vals)
    return vals[keep].astype(np.float32), codes[keep]


def _round_to_alphabet(mag: np.ndarray, fmt: ElementFormat) -> np.ndarray:
    """Round non-negative magnitudes to `fmt` codes, nearest-even, saturating.

    Ties land exactly halfway between adjacent representable values, so
    breaking to the even *code* is identical to breaking to the even mantissa.
    """
    vals, codes = _pos_alphabet(fmt)
    hi = np.searchsorted(vals, mag, side="left").clip(1, len(vals) - 1)
    lo = hi - 1
    d_lo, d_hi = mag - vals[lo], vals[hi] - mag
    # tie -> whichever neighbour has an even code
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (codes[hi] % 2 == 0))
    idx = np.where(pick_hi, hi, lo)
    idx = np.where(mag >= vals[-1], len(vals) - 1, idx)  # saturate
    return codes[idx]


@dataclass
class MXTensor:
    """A tensor held in MX form, with both fault sites exposed as uint8 arrays."""

    scales: np.ndarray          # uint8[..., n_blocks]
    codes: np.ndarray           # uint8[..., n_blocks, block_size]
    fmt: ElementFormat
    block_size: int
    orig_shape: tuple[int, ...]
    axis: int
    pad: int                    # lanes appended to fill the final block
    scale_mode: str = "ocp"

    # ---- geometry -------------------------------------------------------
    @property
    def n_blocks(self) -> int:
        return int(self.scales.size)

    @property
    def n_elements(self) -> int:
        return int(self.codes.size)

    @property
    def scale_bits(self) -> int:
        return 8 * self.n_blocks

    @property
    def element_bits(self) -> int:
        return self.fmt.width * self.n_elements

    def valid_mask(self) -> np.ndarray:
        """Bool, shape of `codes`: False on lanes that are only padding.

        Padding lanes are real storage but carry no model value, so campaigns
        must not spend injections on them.
        """
        m = np.ones(self.codes.shape, dtype=bool)
        if self.pad:
            m[..., -1, self.block_size - self.pad:] = False
        return m

    # ---- decode ---------------------------------------------------------
    def decode(self) -> np.ndarray:
        """Reconstruct the float32 tensor in its original shape."""
        vals = self.fmt.table()[self.codes]                 # (..., nblk, K)
        # A faulted scale can reach 2**127, so the product legitimately
        # overflows to +-inf.  That is the physical outcome of a scale-MSB
        # flip, not an error, so the warning is suppressed rather than fixed.
        with np.errstate(over="ignore", invalid="ignore"):
            vals = vals * decode_e8m0(self.scales)[..., None]
        flat = vals.reshape(*vals.shape[:-2], -1)
        if self.pad:
            flat = flat[..., : flat.shape[-1] - self.pad]
        out = flat.reshape(_moved_shape(self.orig_shape, self.axis))
        return np.moveaxis(out, -1, self.axis).astype(np.float32)

    def copy(self) -> "MXTensor":
        return MXTensor(self.scales.copy(), self.codes.copy(), self.fmt,
                        self.block_size, self.orig_shape, self.axis, self.pad,
                        self.scale_mode)


def _moved_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    """`shape` with `axis` moved to the end."""
    ax = axis % len(shape)
    return tuple(s for i, s in enumerate(shape) if i != ax) + (shape[ax],)


def quantize(x: np.ndarray, fmt: str | ElementFormat = "e4m3",
             block_size: int = 32, axis: int = -1,
             scale_mode: str = "ocp") -> MXTensor:
    """Encode `x` into MX form with `block_size` elements sharing each scale."""
    if not isinstance(fmt, ElementFormat):
        fmt = get_format(fmt)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}")

    x = np.asarray(x, dtype=np.float32)
    orig_shape = x.shape
    xm = np.moveaxis(x, axis, -1)

    n = xm.shape[-1]
    pad = (-n) % block_size
    if pad:
        xm = np.pad(xm, [(0, 0)] * (xm.ndim - 1) + [(0, pad)])
    blocks = xm.reshape(*xm.shape[:-1], (n + pad) // block_size, block_size)

    # -- shared scale
    amax = np.max(np.abs(blocks), axis=-1).astype(np.float64)
    with np.errstate(divide="ignore"):
        if scale_mode == "ocp":
            # align the block maximum with the format's top binade (may clip it)
            exp = np.floor(np.log2(amax)) - fmt.emax
        else:
            # smallest scale that provably cannot clip the block maximum
            exp = np.ceil(np.log2(amax / fmt.max_normal))
    exp = np.where(amax > 0, exp, 0.0)                      # all-zero block
    scales = encode_e8m0(np.clip(np.nan_to_num(exp), -127, 127).astype(np.int32))
    # a NaN anywhere in a block makes its maximum NaN, so the OCP conversion
    # (Algorithm 1 of the MX paper: shared scale from max|V|, NaN not clamped)
    # yields a NaN scale and the whole block decodes to NaN. Mapping NaN to zero
    # instead would silently stop a NaN from propagating through any layer whose
    # input is re-quantised -- which is what activation quantisation does.
    scales = np.where(np.isnan(blocks).any(axis=-1), np.uint8(E8M0_NAN), scales)

    # -- elements: divide out the (clamped) scale, then round onto the alphabet
    scaled = blocks / decode_e8m0(scales)[..., None]
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=fmt.max_normal,
                           neginf=-fmt.max_normal)
    codes = _round_to_alphabet(np.abs(scaled), fmt)
    codes |= (np.signbit(scaled).astype(np.uint8) << fmt.sign_bit)

    return MXTensor(scales, codes.astype(np.uint8), fmt, block_size,
                    orig_shape, axis % x.ndim if x.ndim else 0, pad, scale_mode)


def quantize_dequantize(x: np.ndarray, fmt: str | ElementFormat = "e4m3",
                        block_size: int = 32, axis: int = -1,
                        scale_mode: str = "ocp") -> np.ndarray:
    """Fake-quantisation round trip: encode to MX and decode straight back."""
    return quantize(x, fmt, block_size, axis, scale_mode).decode()
