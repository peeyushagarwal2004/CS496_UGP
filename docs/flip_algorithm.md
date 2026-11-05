# Single-bit flip algorithm for MX tensors (version 2)

Version 1 sorted a flip by what it does to the stored number: a scale branch (S0–S6) and an
element branch (E0–E6). That is only one link in a longer chain. A flipped bit becomes a wrong
answer only if all five of these happen:

```
 fault model      encoding         checks           datapath          decision
 the bit really → the stored     → nothing at     → the computation → the top-1
 changes          number changes   read notices     carries it         prediction moves
 (stage 0)        (stage 1)        (stage 2)        (stage 3)          (stage 4)
```

Version 2 adds the four missing stages and measures each one. Every number comes from one of
these:

- the exhaustive ResNet8 campaigns, in which every stored weight bit was flipped once (483,904
  faults for e3m2, 638,624 for e4m3), joined with their cases (E25, E27);
- new injections of the catastrophic cases under different datapaths (E26);
- exact counting and re-encoding of real activations, with no injection (E28).

Code: `mxfi/cases.py`. Experiments: `experiments/e25`–`e28`. Raw output: `results/e25`–`e28*.txt`.

## What version 1 was missing

| gap | what it changes | measured |
|---|---|---|
| Flip direction (0→1 against 1→0) and stuck-at faults | Every catastrophic case needs a 0→1 flip, so the memory's flip direction sets the risk | 0→1 flips cause 87–89% of all damage, though only 40–43% of stored bits are 0 |
| Silent against detectable outcomes | A NaN output can be caught with one check; a wrong finite answer cannot | 56% (e3m2) and 68% (e4m3) of damage gives NaN/Inf outputs |
| Free checks at read time | A clean MX tensor obeys invariants a reader can check without extra memory | Checks flag 85% of damage; two guards cut the rate 4.3× and 5.2× |
| Datapath semantics | Whether NaN survives ReLU, and how overflow behaves, decide whether a case is catastrophic | NaN-zeroing ReLU: element→NaN drops from 1.0 to 0.08; the +128 scale flip is unchanged |
| Error size, block magnitude, input activity | Finite damage follows the size of the error reaching the next layer, not the case | Variance explained on ordinary element flips: 5% → 67% |
| Re-quantisation of activations | An error is re-encoded at the next layer and can wipe its block neighbours | FP4: an error that doubles one value flushes 15% of its 31 neighbours |
| Activations against weights | Activations are often exactly 0 and have per-input scales | 40–59% of activation values are 0 |
| Where the format puts NaN | A format with NaN next to zero (FNUZ) would be hit through its most common value | FNUZ-style placement: 5–7% of activation flips → NaN, against 0.7–1.2% for OCP e4m3 |

## The algorithm

Each `if` below is checked in order and the first one that matches decides. Every
(code, bit) flip of every format and every scale byte (0–255 × 8 bits, zero block or not) lands
in exactly one case. `tests/test_cases.py` checks this exhaustively through `mxfi.cases.label`.

```
ALGORITHM ASSESS(fault f, fault model M, datapath P)

── Stage 0: does the bit actually change? ──────────────────────── fault model
 1  b ← current value of the bit
 2  if f is stuck-at-v and b = v:           return F0 MASKED (stuck value already stored)
 3  if f hits a padding lane of the last block:
 4      if P drops padding at decode:       return F0 MASKED
 5      else: continue; the lane is multiplied by 0, so only E1 (NaN/Inf) still matters
 6  weight(f) ← exposure(f) × p(b → not b)  ▷ p(0→1) and p(1→0) depend on the technology;
 7                                          ▷ weights are always resident, activations
 8                                          ▷ only while they sit in a buffer

── Stage 1: what does the stored number become? ─────────────────── encoding
    SCALE byte X → X' = X xor mask, Δ = X' − X, block multiplied by 2^Δ
 9  if X' = 255:                            S1 TO NaN       (whole block NaN, zero blocks too)
10  elif X = 255:                           S7 FROM NaN     (only after an earlier fault)
11  elif every element is ±0:               S0 ZERO BLOCK   (masked)
12  elif |block|·2^(X'−127) > FP32 max
         or Δ = +128:                       S2 OVERFLOW     (see stage 3: usually ±Inf)
13  elif Δ ≥ 8:                             S3 GROW ×256 or more
14  elif Δ > 0:                             S4 GROW ×2/×4  (×16 for one bit; Δ < 8 in general)
15  elif Δ ≥ −2:                            S5 SHRINK ÷2/÷4
16  else:                                   S6 ERASE        (÷8 or more; ÷16 for one bit)

    ELEMENT code c → c' = c xor mask, value v → v'
17  if c is NaN/Inf:                        E7 FROM SPECIAL (earlier fault, or e5m2 Inf → NaN)
18  elif c' is NaN/Inf:                     E1 TO SPECIAL   (E1n NaN, E1i ±Inf in e5m2)
19  elif sign bit:  v = ±0 → E0 MASKED;     else E2 NEGATE  (|δ| = 2|v|)
20  elif exponent bit set 0→1:
        v = 0 → E3a FROM ZERO; v subnormal → E3b SUB→NORMAL; else E3 GROW ×2^(2^k)
21  elif exponent bit cleared 1→0:
        v' = 0 → E4a TO ZERO; v' subnormal → E4b NORMAL→SUB; else E4 SHRINK ÷2^(2^k)
22  elif mantissa bit set:    v = 0 → E5a FROM ZERO;  else E5 GROW
23  else (mantissa cleared):  v' = 0 → E6a TO ZERO;   else E6 SHRINK

    Several bits of one word (F5): lines 9–16 hold as written with Δ = X' − X.
    For an element, lines 17–18 still hold; when the bits span fields or flip both ways,
    "field" and "direction" no longer mean anything, so use the value-class change and |δ|.

24  context ← bit_was (0→1 or 1→0), |δ| = |v' − v|·2^(X−127),
              value class of c (zero / subnormal / normal / top binade / max),
              one flip from NaN?, block maximum?, demotes the block's only top-binade element?,
              X' inside the tensor's clean band?, zero-block scale encoding (127 or 0),
              block partial sum P_B, layer, input activity a
25  special-code placement is a parameter (OCP e4m3: S.1111.111; e5m2: exponent 11111;
    FNUZ: 1.0000.000, the −0 pattern, so a sign flip of +0 lands in E1)

── Stage 2: does anything notice at read time? ──────────────────── free checks
26  D1  element code is NaN/Inf (a clean tensor never stores one; none exist in e3m2/e2m3/e2m1)
27        → G1 decode it as 0             ▷ E1 becomes "one weight zeroed"
28  D2  X' outside the tensor's clean band (weights: known offline; activations: calibrated)
          D2a for weights: bit 7 set (no clean weight scale reaches 128)
29        → G2 if above the band, treat the block as 0   ▷ S1/S2/S3 become S6 erase
30  D3  a non-zero block no longer has a top-binade element (OCP invariant) → reload weights

── Stage 3: what does the datapath do with it? ──────────────────── propagation
31  NaN:   x·0 = NaN, so zero inputs (and zero padding lanes) do not mask it
32         NaN-zeroing ReLU (x > 0 ? x : 0) follows → only that output channel is erased
33         else (last layer, NaN-passing ReLU, max/avg pool) → NaN reaches the output
34  ±Inf:  Inf·0 = NaN, and +Inf + −Inf = NaN when both meet in a later sum
35         +Inf passes any ReLU; −Inf is zeroed by any ReLU (a BN with γ < 0 flips the sign first)
36  S2:    the block's partial sum becomes P_B·2^Δ, which overflows when |P_B|·2^Δ > FP32 max
37         (for Δ = +128 that is |P_B| > 1); a saturating accumulator makes it a silent error
38         instead. Measured: 98.7% of +128 flips gave non-finite outputs, not 100%
39  finite: output error ≈ δ·a for an element, (2^Δ − 1)·P_B for a scale
40         zero inputs mask it completely; ReLU masks part of a negative error
41  re-quantisation (activations, all-MX pipelines):
42         an error that makes a value r× its block maximum raises the block's scale, and
43         neighbours far below the new maximum are flushed to 0 (up to 31 more values);
44         how far depends on the format's dynamic range (P7)
45         a NaN reaching re-quantisation → NaN scale → 32 NaNs (P8);
46         an Inf there is implementation-defined (NaN scale, or saturation to the max code)

── Stage 4: does the answer change? ─────────────────────────────── decision
47  outcome ← O1 NO CHANGE | O2 SILENT CORRUPTION | O3 NON-FINITE OUTPUT (caught by D4)
48  damage ← 0 for masked cases;
49           1 for an image whose logit row is not finite (counted as class −1, a mismatch);
50           otherwise G(|g · output error| / margin), rising with the size of the error

── Stage 5: aggregate and protect ───────────────────────────────── use
51  rate        = Σ_f weight(f) · damage(f)
52  silent rate = the part of the rate that no check (D1–D4) catches
53  protection order = bit positions sorted by damage per stored bit
```

## Every case covered

| stage | cases |
|---|---|
| 0 fault model | F1 transient 0→1 · F2 transient 1→0 · F3 stuck-at-1 · F4 stuck-at-0 · F0 masked (stuck value already stored, or a padding lane dropped at decode) · F5 several bits of one word · persistent (weights) against transient (activations) |
| 1 scale | S0 zero block · S1 to NaN (X' = 255) · S2 overflow · S3 grow ×256 or more · S4 grow ×2/×4/×16 · S5 shrink ÷2/÷4 · S6 erase (÷16 or more) · S7 from NaN (X = 255) |
| 1 element | E0 masked (±0 sign) · E1 to NaN/Inf · E2 negate · E3 exponent grow (E3a from zero, E3b subnormal→normal) · E4 exponent shrink (E4a to zero, E4b normal→subnormal) · E5 mantissa grow (E5a from zero) · E6 mantissa shrink (E6a to zero) · E7 from NaN/Inf |
| 1 context | 0→1 or 1→0 · value class (zero / subnormal / normal / top binade / max) · one flip from NaN · block maximum · demotes the maximum · scale in band · zero-block scale encoding (127 or 0) · special-code placement (OCP, e5m2 Inf, FNUZ) · layer · error size · input activity |
| 2 checks | D1 special code → G1 zero it · D2 scale out of band (D2a bit 7 set) → G2 erase the block · D3 lost block maximum |
| 3 datapath | P1 NaN propagates · P2 NaN zeroed by ReLU → channel erased · P3 overflow to ±Inf (+Inf passes ReLU, −Inf is zeroed, Inf·0 and Inf − Inf give NaN) · P4 saturating overflow (silent) · P5 zero input masks finite errors only · P6 ReLU sign asymmetry · P7 re-quantisation spreads the error · P8 NaN block → NaN scale → 32 NaNs |
| 4 decision | O1 no change · O2 silent corruption · O3 non-finite output (detectable) |

Which cases occur depends on the format. e3m2, e2m3 and e2m1 have no special codes, so E1, E7
and D1 never occur. e4m3 reaches NaN from the top binade through both exponent and mantissa
bits. e5m2 reaches ±Inf and NaN only through exponent bits, and a mantissa flip of an Inf gives
NaN (E7). In a clean tensor S7 and E7 never occur, because nothing stores NaN. They occur only
after an earlier fault, which matters for persistent weight faults that pile up between reloads.

### Corrections to the previous draft

- S7 and E7 (flips of a value that is already NaN/Inf) were missing. The code labelled S7 as
  S5/S6, which reads a repair as a shrink.
- The check order was not stated. It matters: X' = 255 wins over a zero block, and a code that
  becomes NaN wins over every field rule.
- "Correct for any multi-bit mask" was too strong. The scale bands hold with Δ = X' − X (so
  Δ = −3 is S6, ÷8), but an element's field and direction stop meaning anything once the bits
  span fields.
- S2 does not always overflow. Δ = +128 multiplies the block by 2^128, which overflows only when
  |P_B| > 1. 1.3% of these flips stayed finite in e3m2 and e4m3.
- +Inf and −Inf behave differently: any ReLU zeroes −Inf. Inf·0 = NaN, so for e5m2 Inf,
  zero inputs do not mask it either.
- Padding lanes in the last block were not covered. They are masked when decode drops them, but
  hardware that multiplies the whole block by a padded 0 input still lets a NaN through.
- A non-finite logit row counts as a mismatch (class −1). So "≈ 1" is exactly 1 for each image
  whose output is not finite.

## Stage 0: the fault model

Every flip into NaN/Inf (E1), to scale 255 (S1) or to +128 (S2) sets a bit that was 0: NaN codes
and large exponents are runs of 1s. All 2,468 (e3m2) and 10,321 (e4m3) catastrophic faults are
0→1.

| | e3m2 | e4m3 |
|---|---|---|
| stored bits that are 0 | 43% | 40% |
| damage from 0→1 flips (= stuck-at-1) | 87.5% | 88.7% |
| rate, 0→1 flips only | 0.00768 | 0.02089 |
| rate, 1→0 flips only | 0.00110 | 0.00267 |
| ratio | 7.0× | 7.8× |

So the per-inference rate is not a property of the format alone. A memory whose upsets are
mostly 1→0 (for example charge loss in DRAM true cells) would see about 8× fewer failures than
the symmetric model assumes. A stuck-at-0 bit is almost harmless, and a stuck-at-1 bit is the
dangerous kind.

## Stage 1: encoding

The version 1 tree is unchanged: S0–S6 and E0–E6 with the measured damage in `e25_flip_cases.txt`.
What is new is how far the case alone goes (held-out R², two-fold):

| model | all faults, e3m2 / e4m3 | ordinary element flips, e3m2 / e4m3 |
|---|---|---|
| site only | 0.22 / 0.09 | 0 / – |
| + case (version 1 tree) | 0.91 / 0.97 | 0.02 / 0.05 |
| + bit position | 0.92 / 0.98 | 0.02 / 0.06 |
| + layer | 0.94 / 0.98 | 0.19 / 0.21 |
| + size of the weight error | 0.94 / 0.98 | 0.54 / 0.54 |
| size × input activity instead | 0.94 / 0.98 | 0.67 / 0.64 |
| + sign of the error | 0.94 / 0.98 | 0.70 / 0.66 |

- **The case decides the catastrophic part almost completely.** E1 is 1.0 for every fault, and S2
  has a median of 1.0.
- **Within ordinary element flips, the case explains almost nothing (2–5%).** What matters is the
  size of the error reaching the next layer: |v' − v| × 2^(X−127) × the activity of the input the
  weight multiplies. The block's scale X is a factor here that version 1 ignored.
- **Graded scale cases are wide too.** For S4 the median is 0.04–0.06, but the 90th percentile is
  0.33–0.43 and the maximum 0.93. That depends on the block's partial sum P_B.

**Activations behave differently** (counting over ResNet8 and RepVGG-A0 inputs):

| | ResNet8 | RepVGG-A0 |
|---|---|---|
| activation values that are exactly 0 | 40% | 55% |
| element flips masked (sign of 0) | 5–11% | 7–15% |
| element flips from zero to a tiny value | 33–35% | 44–48% |
| element flips → NaN/Inf, OCP e4m3 / e5m2 | 1.2% / 5.2% | 0.7% / 3.5% |
| element flips → NaN with FNUZ-style placement | 5.0% | 6.8% |
| scale flips → NaN (scale 127 occurs in FP6/FP4) | 0.2% | 1.0% |

OCP e4m3 puts NaN next to its largest values. An FNUZ format puts it at the −0 pattern, so the
sign flip of +0 gives NaN. Zero is the most common activation value, so that placement would
raise NaN flips 4–10×. Where a format puts its special codes is a design parameter, and the
algorithm takes it as one.

## Stage 2: free checks and guards

A clean OCP MX tensor obeys three invariants. Checking them needs no extra memory:

| check | catches | e3m2 damage flagged | e4m3 damage flagged |
|---|---|---|---|
| D1 element code is NaN/Inf | E1 | – | 52.2% |
| D2a scale bit 7 is set (no clean scale reaches 128) | S2 | 56.5% | 16.0% |
| D2b scale outside the tensor's clean band | S2, S3, part of S4–S6 | 85.7% | 32.1% |
| D3 block lost its top-binade element | some E4 | 0.1% | 0.2% |
| D1 + D2b + D3 at read (3.5% / 4.1% of faults) | | 85.8% | 84.5% |
| D4 NaN/Inf output | | 56.2% | 68.1% |

After every check, the silent rate left is 0.00124 for e3m2 (from 0.00878) and 0.00364 for e4m3
(from 0.0236).

Correcting what D1 and D2 detect:

| | e3m2 rate | e4m3 rate |
|---|---|---|
| no guard | 0.00878 | 0.0236 |
| G2: scale above band → block treated as 0 | 0.00206 (4.3× lower) | 0.0168 (1.4×) |
| G1: special code → 0 (measured) | – | 0.0113 (2.1×) |
| G1 + G2 | 0.00206 (4.3×) | 0.00454 (5.2×) |

- G2's effect is exact. A block treated as 0 has the measured damage of the same block's ÷2^64
  flip.
- G1 was measured by injecting 2,000 E1 faults with the code decoded as 0. Mean damage fell from
  1.0 to 0.0064.
- Activation scales can use the same check with a band calibrated on a few inputs. On 10 unseen
  images, 0.01% of clean blocks fell outside it, and it caught 100% of flips of bits 3–7, 80–94% of
  bit 2, 36–59% of bit 1 and under 1% of bit 0.

## Stage 3: the datapath

Injected again under a ReLU that zeroes NaN (`x > 0 ? x : 0`, IEEE 754-2008 maxNum), against the
PyTorch ReLU used everywhere else, which passes NaN on:

| case | NaN-passing ReLU | NaN-zeroing ReLU | read guard G1 |
|---|---|---|---|
| E1 element → NaN, conv layers | 1.00 | 0.05–0.29 (that output channel is erased) | 0.002–0.11 |
| E1 element → NaN, final layer | 1.00 | 1.00 (no ReLU follows) | 0.002–0.02 |
| E1 overall | 1.00 | 0.085 | 0.0064 |
| S2 scale +128, e3m2 / e4m3 | 0.98 / 0.98 | 0.97 / 0.97 | (G2: erase) |

So the same flip is catastrophic or mild depending on hardware the study never modelled.
Zeroing NaN in the ReLU turns a NaN weight into "this output channel is gone". The +128 scale flip
usually produces +Inf, which passes any ReLU, so only a scale check (D2/G2) helps there. A
negative partial sum gives −Inf instead, which a ReLU zeroes, and a partial sum below 1 does not
overflow at all. Either can leave the output finite, as happened for 1.3% of these flips; the two
were not separated here.

**Zero inputs mask finite errors only.** A weight error that only multiplies zeros changes
nothing, but NaN·0 = NaN, so a NaN weight still corrupts the output. ResNet8 has no input channel
that is zero on all 200 images, so this rule follows from the arithmetic but is untested here.

**ReLU asymmetry is real but small.** In hidden layers, where inputs are ≥ 0, an error that raises
the pre-activation is 1.07–1.12× worse than one that lowers it. At the largest errors that rises
to 1.2× (e3m2) and 1.4× (e4m3), because a large positive error can switch a unit on, while a
negative one can at most switch it off. The stem and final layer show no such effect.

**Re-quantisation spreads errors in narrow formats.** In an all-MX pipeline every layer's input is
re-encoded, and an error that makes one value r× its block maximum raises the block's scale. The
table shows what happens to the other 31 values of real ResNet8 activation blocks (share of
non-zero neighbours flushed to 0):

| format | r = 2 | r = 4 | r = 16 | r = 256 | r = 65536 |
|---|---|---|---|---|---|
| e5m2 | 0 | 0 | 0 | 0 | 0 |
| e4m3 | 0 | 0 | 0.01% | 0.1% | 29% |
| e3m2 | 0.5% | 0.9% | 3.9% | 55% | 100% |
| e2m3 | 3.7% | 7.4% | 28% | 100% | 100% |
| e2m1 | 15% | 29% | 91% | 100% | 100% |

This reverses the NaN story. The no-NaN formats are the ones whose errors spread at
re-quantisation. In FP4, an error that merely doubles one value removes a neighbour from 63% of
blocks. A NaN reaching a re-quantisation point becomes a NaN scale and 32 NaNs (P8). Weight-only
campaigns, including the exhaustive ones, cannot see either effect.

## Stage 4: the decision

The per-inference rate splits into a part that is silent and a part that comes with NaN/Inf
outputs, which one isfinite check on the logits catches:

| | e3m2 | e4m3 |
|---|---|---|
| silent corruption | 0.00384 (44%) | 0.00750 (32%) |
| non-finite output | 0.00494 (56%) | 0.01606 (68%) |

For a weight fault, one caught NaN output is enough: the fault is persistent, so the system can
reload the weights and the fault becomes a detected error instead of a silent one. Most of e4m3's
extra damage over e3m2 is this detectable kind.

## Stage 5: protection order

Bit positions sorted by damage per stored bit:

- **e3m2:** scale bit 7 alone is 0.5% of storage and 57% of damage. Scale bits 7, 3, 2 and 1 are 2%
  of storage and 83% of damage. Protecting by bit position works well.
- **e4m3:** 27% of storage is needed for half the damage, because NaN damage is spread over every
  element bit. Protecting by bit position works badly here. The read guard G1 plus the scale check
  G2 do far better, at no storage cost.

## What it still does not cover

- Faults outside stored MX values: accumulators, partial sums, biases, normalisation parameters,
  control logic, and addressing (a block paired with the wrong scale). The element branch is
  format-generic, so FP32/BF16 values can use it with their own special codes, but this was not
  measured.
- Multi-fault accumulation in weight memory over time. Single flips within a word were measured
  earlier to be sub-additive (E21), but faults piling up across words between reloads were not.
- Damage under a saturating-overflow datapath, an FP16 accumulator, or a full MX activation
  pipeline end to end. These are rules here (P4, P3 thresholds, P7), not injected measurements.
- One network, ResNet8 width 16, measured exhaustively. Activation results cover ResNet8 and
  RepVGG-A0.

## Reproducing

```bash
python -m pytest tests/test_cases.py -q
python -m experiments.e25_flip_cases
python -m experiments.e26_datapath          # about 20 min on CPU
python -m experiments.e27_fault_model       # after e25 and e26
python -m experiments.e28_activation_cases
```
