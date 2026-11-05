"""Case taxonomy for single-bit flips in MX tensors.

Every single-bit fault in an MX tensor falls into exactly one case of a small
decision tree. The tree has two roots, one per fault site.

Scale flip (one E8M0 byte X shared by K elements, bit j)
    A flip of bit j moves the exponent by exactly ``delta = +2**j`` if the bit
    was 0 and ``-2**j`` if it was 1, so the whole block is multiplied by
    ``2**delta``. How far it moves is fixed by j alone; the value of X only
    decides the direction and whether a boundary is crossed::

        zero_block   every element is +-0 and X' is finite: masked exactly
        to_nan       X' = 255: the whole block becomes NaN (zero blocks too)
        up_overflow  the block leaves the float32 range, in decode or (always
                     for delta = +128) in the accumulation that reads it
        up_large     delta >= +8: the block grows by 256x or more
        up_small     delta in {+1, +2, +4}
        down_small   delta in {-1, -2}
        erase        delta <= -4: the block shrinks 16x or more, as if deleted
        from_nan     X = 255 already (only after an earlier fault): the block
                     becomes finite again

    The checks run in the order to_nan, from_nan, zero_block, up_overflow,
    then the delta bands, and the first that matches is the case.

Element flip (one code of a sign/exponent/mantissa format, bit b)
    The case is the field the bit belongs to, the direction of the flip and the
    transition it causes between value classes::

        sign      zero -> masked (+0 <-> -0); otherwise v -> -v, error 2|v|
        exponent  up:   grow by 2**(2**k), from_zero, sub_to_normal, to_special
                  down: shrink by 2**(2**k), normal_to_sub, to_zero
        mantissa  up:   grow, from_zero, to_special (e4m3 top binade only)
                  down: shrink, to_zero
        any       from_special: the code was already NaN/Inf (only after an
                  earlier fault, or e5m2 Inf -> NaN)

    The value class of the code is the "where in the range" axis: zero,
    subnormal, normal, top binade (where the OCP rule puts every block maximum)
    and max (the saturation code). ``near_special`` marks codes one flip away
    from NaN/Inf.

Everything here is exact enumeration over the code tables, not sampling.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .codec import MXTensor
from .formats import E8M0_BIAS, E8M0_NAN, ElementFormat

__all__ = ["bit_field", "value_class", "element_cases", "scale_cases",
           "classify_elements", "classify_scales", "label", "FP32_MAX"]

FP32_MAX = float(np.finfo(np.float32).max)

# Short names used in docs/flip_algorithm.md. Every case the two trees can
# produce has exactly one entry here.
_SCALE_LABELS = {
    "zero_block": "S0", "to_nan": "S1", "up_overflow": "S2", "up_large": "S3",
    "up_small": "S4", "down_small": "S5", "erase": "S6", "from_nan": "S7",
}
_ELEMENT_LABELS = {
    "sign:sign:masked": "E0",
    "sign:sign:negate": "E2",
    "exponent:up:grow": "E3", "exponent:up:from_zero": "E3a",
    "exponent:up:sub_to_normal": "E3b",
    "exponent:down:shrink": "E4", "exponent:down:to_zero": "E4a",
    "exponent:down:normal_to_sub": "E4b",
    "mantissa:up:grow": "E5", "mantissa:up:from_zero": "E5a",
    "mantissa:down:shrink": "E6", "mantissa:down:to_zero": "E6a",
}


def label(case: str) -> str:
    """Short name (S0-S7, E0-E7) of a case string from either tree."""
    if case in _SCALE_LABELS:
        return _SCALE_LABELS[case]
    if case.endswith(":to_special"):
        return "E1"
    if case.endswith(":from_special"):
        return "E7"
    return _ELEMENT_LABELS[case]


def bit_field(fmt: ElementFormat, bit: int) -> str:
    """'sign', 'exponent' or 'mantissa' for element bit `bit` (0 = LSB)."""
    if bit == fmt.sign_bit:
        return "sign"
    return "exponent" if bit >= fmt.man_bits else "mantissa"


def _fields(fmt: ElementFormat, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exp = (codes >> fmt.man_bits) & ((1 << fmt.exp_bits) - 1)
    man = codes & ((1 << fmt.man_bits) - 1)
    return exp, man


@lru_cache(maxsize=None)
def _top_exponent(fmt: ElementFormat) -> int:
    """Largest exponent field that still holds a finite value (the top binade)."""
    t = fmt.table()
    exp, _ = _fields(fmt, np.arange(fmt.n_codes))
    return int(exp[np.isfinite(t)].max())


@lru_cache(maxsize=None)
def value_class(fmt: ElementFormat) -> np.ndarray:
    """Value class of every code: special, zero, subnormal, normal, top, max."""
    t = fmt.table()
    exp, man = _fields(fmt, np.arange(fmt.n_codes))
    out = np.full(fmt.n_codes, "normal", dtype=object)
    out[exp == _top_exponent(fmt)] = "top"
    out[np.abs(t) == fmt.max_normal] = "max"
    out[(exp == 0) & (man != 0)] = "subnormal"
    out[(exp == 0) & (man == 0)] = "zero"
    out[~np.isfinite(t)] = "special"
    return out


@lru_cache(maxsize=None)
def element_cases(fmt: ElementFormat) -> dict[str, np.ndarray]:
    """Case of every (code, bit) element flip. Arrays are (n_codes, width).

    ``rel_error`` is ``|after - before| / 2**emax``: the error in units of the
    top binade, which is where the OCP rule puts the block maximum. So 1.0
    means "an error as large as the biggest weight in the block".
    """
    t = fmt.table().astype(np.float64)
    codes = np.arange(fmt.n_codes)[:, None]
    bits = np.arange(fmt.width)[None, :]
    flipped = codes ^ (1 << bits)
    before = np.broadcast_to(t[:, None], flipped.shape)
    after = t[flipped]

    field = np.broadcast_to(
        np.array([bit_field(fmt, b) for b in range(fmt.width)], dtype=object),
        flipped.shape)
    direction = np.where(field == "sign", "sign",
                         np.where((codes >> bits) & 1, "down", "up"))

    cls = value_class(fmt)
    c_before = np.broadcast_to(cls[:, None], flipped.shape)
    c_after = cls[flipped]

    trans = np.full(flipped.shape, "", dtype=object)
    grow = np.abs(after) > np.abs(before)
    trans[:] = np.where(grow, "grow", "shrink")
    trans[(c_before == "subnormal") & ~np.isin(c_after, ["zero", "subnormal"])] = "sub_to_normal"
    trans[~np.isin(c_before, ["zero", "subnormal"]) & (c_after == "subnormal")] = "normal_to_sub"
    trans[(c_before != "zero") & (c_after == "zero")] = "to_zero"
    trans[(c_before == "zero") & (c_after != "zero")] = "from_zero"
    trans[field == "sign"] = "negate"
    trans[(field == "sign") & (c_before == "zero")] = "masked"
    trans[c_after == "special"] = "to_special"
    trans[c_before == "special"] = "from_special"

    with np.errstate(invalid="ignore"):
        rel = np.abs(after - before) / 2.0 ** fmt.emax
    rel = np.where(np.isfinite(after) & np.isfinite(before), rel, np.inf)

    near_special = (c_after == "special").any(axis=1) & (cls != "special")
    return {
        "field": field.copy(),
        "direction": direction,
        "from_class": c_before.copy(),
        "to_class": c_after,
        "transition": trans,
        "case": (field + ":" + direction + ":" + trans).astype(object),
        "rel_error": rel,
        "before": before.copy(),
        "after": after,
        "near_special": np.broadcast_to(near_special[:, None], flipped.shape).copy(),
    }


def scale_cases(x: np.ndarray, bit: np.ndarray, block_max: np.ndarray,
                zero_block: np.ndarray) -> dict[str, np.ndarray]:
    """Case of each scale flip.

    Parameters
    ----------
    x : original E8M0 byte of each fault.
    bit : flipped bit, 0..7.
    block_max : largest |element value| in the block at unit scale.
    zero_block : every element of the block is +-0.
    """
    x = np.asarray(x, dtype=np.int64)
    bit = np.asarray(bit, dtype=np.int64)
    was_set = (x >> bit) & 1
    delta = np.where(was_set, -(1 << bit), 1 << bit)
    x_new = x ^ (1 << bit)

    with np.errstate(over="ignore"):
        new_max = np.asarray(block_max, np.float64) * np.exp2(
            (x_new - E8M0_BIAS).astype(np.float64))

    case = np.select([delta >= 8, delta > 0, delta >= -2],
                     ["up_large", "up_small", "down_small"], "erase").astype(object)
    case[(new_max > FP32_MAX) | (delta == 128)] = "up_overflow"
    case[np.asarray(zero_block, bool)] = "zero_block"
    case[x == E8M0_NAN] = "from_nan"
    case[x_new == E8M0_NAN] = "to_nan"
    return {"delta": delta, "x_new": x_new, "case": case}


def classify_elements(mx: MXTensor, index: tuple[np.ndarray, ...],
                      bit: np.ndarray) -> dict[str, np.ndarray]:
    """Element cases for faults at `index` (arrays into ``mx.codes``) and `bit`.

    Besides the case, each fault gets the context its damage depends on:

    ``bit_was``      0 for a 0->1 flip, 1 for a 1->0 flip. Every flip into a
                     special code is 0->1, so the memory's flip direction matters.
    ``abs_error``    the change of the stored weight itself, |v' - v| * 2**(X-127);
                     ``rel_error`` times the block's magnitude.
    ``demotes_max``  the flip moves the block's only top-binade element out of the
                     top binade. The OCP rule guarantees every non-zero block has
                     one, so this breaks an invariant a reader can check for free.
    """
    f = mx.fmt
    codes = mx.codes[index].astype(np.int64)
    tab = element_cases(f)
    out = {k: v[codes, bit] for k, v in tab.items()}
    t = f.table()
    blk = np.abs(t[mx.codes]).max(axis=-1)           # block max at unit scale
    out["block_max"] = blk[index[:-1]]
    out["is_block_max"] = np.abs(t[codes]) == out["block_max"]
    out["code"] = codes
    out["bit_was"] = (codes >> bit) & 1
    scale = np.exp2(mx.scales[index[:-1]].astype(np.float64) - E8M0_BIAS)
    out["abs_error"] = out["rel_error"] * 2.0 ** f.emax * scale

    top = _fields(f, mx.codes.astype(np.int64))[0] == _top_exponent(f)
    top &= np.isfinite(t[mx.codes]) & mx.valid_mask()
    out["demotes_max"] = ((top.sum(axis=-1)[index[:-1]] == 1) & top[index]
                          & (out["field"] == "exponent"))
    return out


def classify_scales(mx: MXTensor, index: tuple[np.ndarray, ...], bit: np.ndarray,
                    band: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """Scale cases for faults at `index` (arrays into ``mx.scales``) and `bit`.

    ``out_of_band`` marks a faulty byte outside `band`, by default the range of
    this tensor's own clean scales. For weights that range is known offline; for
    activations it is calibrated on a few inputs. ``above_band`` is the half of
    it that grows the block, the half a guard should correct.
    """
    t = np.abs(mx.fmt.table()[mx.codes])
    block_max = t.max(axis=-1)[index]
    zero_block = (t == 0).all(axis=-1)[index]
    out = scale_cases(mx.scales[index], bit, block_max, zero_block)
    out["x"] = mx.scales[index].astype(np.int64)
    out["bit_was"] = (out["x"] >> np.asarray(bit, np.int64)) & 1
    lo, hi = band if band is not None else (int(mx.scales.min()), int(mx.scales.max()))
    out["above_band"] = out["x_new"] > hi
    out["out_of_band"] = out["above_band"] | (out["x_new"] < lo)
    return out
