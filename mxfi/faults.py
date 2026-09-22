"""Two-site bit-fault injection for MX tensors.

An MX tensor exposes two structurally different fault sites:

``element``
    One code of ``fmt.width`` bits.  A flip perturbs exactly one value.

``scale``
    One E8M0 byte shared by ``block_size`` elements.  A flip rescales the
    whole block by ``2 ** (+-2**bit)``, so its blast radius is K values.

Both are addressed uniformly by :class:`Fault` and applied by :func:`inject`.

Because the alphabets are tiny, the impact of *every* fault is enumerable in
closed form rather than sampled -- :func:`element_impact_table` and
:func:`scale_impact_table` build the complete (code, bit) severity maps that
the campaign uses for value-aware stratification.

Severity note
-------------
The drafts define element severity as ``d = |log v' - log v|``.  That measure
is blind to the sign bit: negating a value leaves ``|v|`` unchanged, so a
sign flip scores zero severity while actually inducing a relative error of 2.
The tables below therefore report ``log_severity`` (the drafts' metric),
``rel_error``, and an explicit ``sign_flip`` mask, so the campaign can use a
severity that does not silently discard the sign bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .codec import MXTensor
from .formats import E8M0_NAN, ElementFormat, decode_e8m0

__all__ = ["Fault", "SITES", "inject", "fault_space", "element_impact_table",
           "scale_impact_table", "block_of_element"]

SITES = ("element", "scale")


@dataclass(frozen=True)
class Fault:
    """A single-bit fault at one of the two MX sites.

    Parameters
    ----------
    site : "element" or "scale".
    index : index into ``MXTensor.codes`` (element) or ``.scales`` (scale).
    bit : bit position, 0 = LSB.  Bounded by ``fmt.width`` / 8 respectively.
    """

    site: str
    index: tuple[int, ...]
    bit: int

    def __post_init__(self) -> None:
        if self.site not in SITES:
            raise ValueError(f"site must be one of {SITES}, got {self.site!r}")
        if self.bit < 0:
            raise ValueError("bit must be non-negative")


def inject(mx: MXTensor, fault: Fault) -> MXTensor:
    """Return a copy of `mx` with `fault` applied. The input is left untouched."""
    width = mx.fmt.width if fault.site == "element" else 8
    if fault.bit >= width:
        raise ValueError(
            f"bit {fault.bit} out of range for {fault.site} site of width {width}")

    out = mx.copy()
    target = out.codes if fault.site == "element" else out.scales
    target[fault.index] ^= np.uint8(1 << fault.bit)
    return out


def block_of_element(index: tuple[int, ...]) -> tuple[int, ...]:
    """Index of the scale governing the element at `index`.

    ``codes`` is ``(..., n_blocks, block_size)`` and ``scales`` is
    ``(..., n_blocks)``, so dropping the final axis maps element to scale.
    """
    return tuple(index[:-1])


def fault_space(mx: MXTensor, valid_only: bool = True) -> dict[str, int]:
    """Addressable bit counts per site -- the population a campaign samples from.

    With ``valid_only`` the padding lanes are excluded, since they hold no
    model value and must not consume injection budget.
    """
    n_elem = int(mx.valid_mask().sum()) if valid_only else mx.n_elements
    elem_bits = n_elem * mx.fmt.width
    scale_bits = mx.n_blocks * 8
    return {
        "elements": n_elem,
        "blocks": mx.n_blocks,
        "element_bits": elem_bits,
        "scale_bits": scale_bits,
        "total_bits": elem_bits + scale_bits,
        "element_bit_fraction": elem_bits / (elem_bits + scale_bits),
    }


# --------------------------------------------------------- exact enumeration

@lru_cache(maxsize=None)
def element_impact_table(fmt: ElementFormat) -> dict[str, np.ndarray]:
    """Exact impact of every (code, bit) element flip. Arrays are (n_codes, width).

    Returns
    -------
    dict with keys:
      ``before`` / ``after``  decoded values at unit scale
      ``log_severity``        ``|log2|after| - log2|before||``; inf if either is
                              zero/non-finite (a change of infinite log-ratio)
      ``rel_error``           ``|after - before| / max(|before|, min_subnormal)``
      ``sign_flip``           the flipped bit was the sign bit
      ``to_nonfinite``        the flip produced NaN or Inf
      ``from_nonfinite``      the original code was NaN or Inf
    """
    t = fmt.table()
    codes = np.arange(fmt.n_codes, dtype=np.int64)[:, None]
    bits = np.arange(fmt.width, dtype=np.int64)[None, :]

    before = np.broadcast_to(t[:, None], (fmt.n_codes, fmt.width))
    after = t[codes ^ (1 << bits)]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_sev = np.abs(np.log2(np.abs(after)) - np.log2(np.abs(before)))
    # zero <-> non-zero and any non-finite endpoint is an infinite log ratio
    log_sev = np.where(np.isfinite(log_sev), log_sev, np.inf)

    denom = np.maximum(np.abs(before), fmt.min_subnormal)
    with np.errstate(invalid="ignore"):
        rel = np.abs(after - before) / denom
    rel = np.where(np.isnan(after) | np.isnan(before), np.inf, rel)

    return {
        "before": before.astype(np.float32),
        "after": after.astype(np.float32),
        "log_severity": log_sev.astype(np.float64),
        "rel_error": rel.astype(np.float64),
        "sign_flip": np.broadcast_to(bits == fmt.sign_bit,
                                     (fmt.n_codes, fmt.width)).copy(),
        "to_nonfinite": ~np.isfinite(after),
        "from_nonfinite": ~np.isfinite(before),
    }


@lru_cache(maxsize=None)
def scale_impact_table() -> dict[str, np.ndarray]:
    """Exact impact of every (scale byte, bit) flip. Arrays are (256, 8).

    A scale flip multiplies every element of the block by ``2 ** delta_exp``
    where ``delta_exp = +-2**bit``, so ``log_severity`` is the *per-element*
    severity; multiply by the block size for the block severity ``D_s`` of the
    drafts.  Flips that land on code 255 turn the entire block into NaN.
    """
    s = np.arange(256, dtype=np.int64)[:, None]
    bits = np.arange(8, dtype=np.int64)[None, :]
    after = s ^ (1 << bits)

    # +2**bit when the bit was clear, -2**bit when it was set
    delta = np.where((s >> bits) & 1, -(1 << bits), (1 << bits))

    was_nan = s == E8M0_NAN
    now_nan = after == E8M0_NAN
    multiplier = np.where(now_nan | was_nan, np.nan,
                          np.exp2(np.clip(delta, -1074, 1023).astype(np.float64)))

    log_sev = np.where(now_nan | was_nan, np.inf, np.abs(delta).astype(np.float64))

    return {
        "before": np.broadcast_to(s, (256, 8)).copy(),
        "after": after.astype(np.int64),
        "delta_exp": delta.astype(np.int64),
        "multiplier": multiplier,
        "log_severity": log_sev,          # |delta_exp|, in log2 units
        "to_nan": now_nan,
        "from_nan": np.broadcast_to(was_nan, (256, 8)).copy(),
        "overflows_f32": np.abs(delta) >= 128,
    }


def apply_scale_fault_to_block(scale_byte: int, bit: int) -> tuple[int, float]:
    """Convenience: ``(new_byte, block multiplier)`` for one scale flip."""
    tab = scale_impact_table()
    return int(tab["after"][scale_byte, bit]), float(tab["multiplier"][scale_byte, bit])


def decoded_block_after_scale_fault(mx: MXTensor, block_index: tuple[int, ...],
                                    bit: int) -> np.ndarray:
    """Decoded values of one block after flipping `bit` of its shared scale."""
    codes = mx.codes[block_index]
    new_byte, _ = apply_scale_fault_to_block(int(mx.scales[block_index]), bit)
    return mx.fmt.table()[codes] * decode_e8m0(np.uint8(new_byte))
