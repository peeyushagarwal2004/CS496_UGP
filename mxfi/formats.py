"""Microscaling (MX) element and scale format definitions.

Follows the OCP Microscaling Formats (MX) specification v1.0.

An MX block stores K elements sharing one 8-bit E8M0 scale::

    v[i] = decode_element(code[i]) * 2 ** (X - 127)

The element alphabets here are tiny (16 codes for FP4 up to 256 for FP8), so
every format is materialised as an *exact* lookup table over all 2**w codes.
That table is both the codec's decode path and the "exact per-code
enumeration" the study needs to reason about fault impact analytically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ElementFormat", "ELEMENT_FORMATS", "get_format", "E8M0_BIAS",
           "E8M0_NAN", "decode_e8m0", "encode_e8m0"]

# ---------------------------------------------------------------- E8M0 scale

E8M0_BIAS = 127
E8M0_NAN = 255  # the one reserved code; 0..254 are powers of two


def decode_e8m0(byte: np.ndarray) -> np.ndarray:
    """Decode E8M0 scale bytes to float32 powers of two. Code 255 -> NaN."""
    byte = np.asarray(byte, dtype=np.uint8)
    nan = byte == E8M0_NAN
    e = np.where(nan, 0, byte.astype(np.int32) - E8M0_BIAS)
    out = np.exp2(e.astype(np.float64))          # float64 so 2**127 cannot overflow
    return np.where(nan, np.nan, out).astype(np.float32)


def encode_e8m0(exp: np.ndarray) -> np.ndarray:
    """Encode integer exponents to E8M0 bytes, saturating into [0, 254]."""
    return np.clip(np.asarray(exp) + E8M0_BIAS, 0, 254).astype(np.uint8)


# ------------------------------------------------------------ element format

@dataclass(frozen=True)
class ElementFormat:
    """A sign-magnitude floating-point element format of at most 8 bits.

    Attributes
    ----------
    name : canonical name, e.g. "e4m3".
    exp_bits, man_bits : field widths; total width is 1 + exp_bits + man_bits.
    has_inf, has_nan : whether the all-ones exponent encodes specials.  The MX
        element formats below FP8 have no specials, so every code is a number.
    """

    name: str
    exp_bits: int
    man_bits: int
    has_inf: bool
    has_nan: bool

    # -- derived geometry -------------------------------------------------
    @property
    def width(self) -> int:
        return 1 + self.exp_bits + self.man_bits

    @property
    def n_codes(self) -> int:
        return 1 << self.width

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def sign_bit(self) -> int:
        """Index of the sign bit (bit 0 is the LSB of the mantissa)."""
        return self.width - 1

    # -- the exact alphabet ----------------------------------------------
    def table(self) -> np.ndarray:
        """float32[n_codes]: decoded value of every bit pattern.

        Index by the raw code; specials decode to +/-inf or NaN.
        """
        codes = np.arange(self.n_codes, dtype=np.int64)
        sign = 1.0 - 2.0 * ((codes >> (self.exp_bits + self.man_bits)) & 1)
        exp = (codes >> self.man_bits) & ((1 << self.exp_bits) - 1)
        man = codes & ((1 << self.man_bits) - 1)
        scale = float(1 << self.man_bits)

        # subnormals (exp == 0) have no implicit leading 1 and a fixed exponent
        sub = (man / scale) * 2.0 ** (1 - self.bias)
        nrm = (1.0 + man / scale) * 2.0 ** (exp.astype(np.float64) - self.bias)
        val = np.where(exp == 0, sub, nrm) * sign

        if self.has_inf or self.has_nan:
            top = exp == (1 << self.exp_bits) - 1
            if self.has_inf:
                # IEEE-style: mantissa 0 is Inf, anything else is NaN
                val = np.where(top & (man == 0), sign * np.inf, val)
                val = np.where(top & (man != 0), np.nan, val)
            else:
                # OCP E4M3: only the all-ones mantissa is NaN, the rest are finite
                val = np.where(top & (man == (1 << self.man_bits) - 1), np.nan, val)
        return val.astype(np.float32)

    def finite_mask(self) -> np.ndarray:
        return np.isfinite(self.table())

    @property
    def max_normal(self) -> float:
        """Largest finite magnitude representable."""
        t = self.table()
        return float(np.max(np.abs(t[np.isfinite(t)])))

    @property
    def emax(self) -> int:
        """floor(log2(max_normal)) -- the shared-scale alignment target.

        The OCP scale rule lines the block maximum up with this exponent, so
        the largest element in a block lands in the format's top binade.
        """
        return int(np.floor(np.log2(self.max_normal)))

    @property
    def min_subnormal(self) -> float:
        t = np.abs(self.table())
        nz = t[np.isfinite(t) & (t > 0)]
        return float(np.min(nz))


ELEMENT_FORMATS: dict[str, ElementFormat] = {
    # MXFP8
    "e4m3": ElementFormat("e4m3", 4, 3, has_inf=False, has_nan=True),
    "e5m2": ElementFormat("e5m2", 5, 2, has_inf=True, has_nan=True),
    # MXFP6 -- no specials, every code is a finite number
    "e3m2": ElementFormat("e3m2", 3, 2, has_inf=False, has_nan=False),
    "e2m3": ElementFormat("e2m3", 2, 3, has_inf=False, has_nan=False),
    # MXFP4
    "e2m1": ElementFormat("e2m1", 2, 1, has_inf=False, has_nan=False),
}

# friendly aliases used in the paper drafts
_ALIASES = {
    "mxfp8": "e4m3", "mxfp8_e4m3": "e4m3", "fp8_e4m3": "e4m3",
    "mxfp8_e5m2": "e5m2", "fp8_e5m2": "e5m2",
    "mxfp6": "e3m2", "mxfp6_e3m2": "e3m2", "mxfp6_e2m3": "e2m3",
    "mxfp4": "e2m1", "mxfp4_e2m1": "e2m1",
}


def get_format(name: str) -> ElementFormat:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in ELEMENT_FORMATS:
        raise KeyError(
            f"unknown element format {name!r}; "
            f"known: {sorted(ELEMENT_FORMATS)} plus aliases {sorted(_ALIASES)}"
        )
    return ELEMENT_FORMATS[key]
