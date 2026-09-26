# Findings — ResNet8 / CIFAR-10, weight faults, K=32

Setup: ResNet8 trained to **87.01%** top-1 (fp32, matches MLPerf Tiny reference),
BN-folded, weights quantised to MX. Campaign = 3,000 uniformly sampled bit flips over
the whole weight bit space, each scored on a fixed 200-image class-balanced test subset.
SDC = any top-1 prediction change on any image of the subset.

Raw results: `results/e00_quantized_accuracy.csv`, `results/e01_*-n3000.csv`.

---

## F1. The core hypothesis holds — same accuracy, very different reliability

| format | bits | accuracy | SDC rate (95% CI) |
|---|---|---|---|
| `e5m2` | 8 | 0.8480 | **28.90%** [27.31, 30.55] |
| `e3m2` | 6 | 0.8485 | **51.27%** [49.48, 53.05] |
| `e4m3` | 8 | 0.8508 | 32.30% [30.65, 33.99] |

`e5m2` and `e3m2` differ in accuracy by **0.05 pp** — statistically indistinguishable —
but differ in fault vulnerability by **1.77×**, with completely disjoint confidence
intervals. This is exactly the claim the drafts make, now measured on a trained network
under a controlled comparison.

Normalising for storage does not rescue `e3m2`. At a fixed per-bit upset rate the
model-level exposure is `SDC_rate × bits_per_value`:

| format | bits | SDC × bits |
|---|---|---|
| `e5m2` | 8 | 2.31 |
| `e4m3` | 8 | 2.58 |
| `e3m2` | 6 | **3.08** |

**Design guidance:** MXFP6-`e3m2` buys 25% less storage than MXFP8-`e5m2` at the same
accuracy, but costs ~33% more fault-induced failure. Accuracy alone is not a reliability
proxy — which is the drafts' headline, confirmed.

## F2. The failure *mode* differs, not just the rate

| | `e5m2` (8b) | `e3m2` (6b) |
|---|---|---|
| non-finite outputs | **7.70%** | **0.00%** |
| mean images corrupted / element fault | 15.6 / 200 | 0.7 / 200 |
| worst bit | exp bit 3: 65.8 images, 32.7% non-finite | sign bit: 0.86 images |
| per-bit SDC spread | 0.02 – 0.51 | 0.38 – 0.56 |

`e5m2` fails **rarely but catastrophically** — it has Inf/NaN codes, and its 5 exponent
bits let a single flip produce astronomically large values. `e3m2` fails **often but
mildly** — it has no special codes at all, so no element flip can ever produce a
non-finite value, and its 3 exponent bits cap the perturbation.

This distinction matters for mitigation: `e5m2` wants range clamping or NaN detection;
`e3m2` wants broad ECC, because every one of its bits is roughly equally risky.

## F3. The drafts' severity metric does not predict failure — and inverts on the sign bit

The drafts define element severity as `d = |log v' − log v|`. Measured against observed
SDC across all three formats (n = 8,374 element faults):

```
Spearman(log_severity, sdc) = 0.026     <- no relationship
Spearman(rel_error,    sdc) = 0.231     <- weak, but ~9x better
```

Bucketed, the metric is non-monotone and its *zero* bucket is the **worst**:

| predicted log_severity | n | observed SDC |
|---|---|---|
| **= 0 (sign flips)** | 1236 | **49.76%** ← highest |
| (0, 0.5] | 2179 | 18.91% |
| (0.5, 1.5] | 1674 | 30.11% |
| (1.5, 3] | 1043 | 43.43% |
| (3, 9] | 1860 | 39.25% |
| > 9 | 382 | 24.35% |

The sign bit is the **single most damaging bit position** in both `e4m3` (52.8% SDC, above
every exponent bit) and `e3m2` (56.1%) — and the metric scores it as *zero damage*.

**Consequence:** a TreeFI-style value-aware allocation built on `log_severity` would
deprioritise the most dangerous bit in the word to nothing, and otherwise allocate budget
almost at random. **This must be fixed in §3 of both drafts before the method is claimed
to work.** `rel_error` is a usable starting point; a severity that handles sign, exponent
and saturation separately would be better.

## F4. Shared-scale faults are 2–3× more likely to corrupt, and hit 40× more values

| format | element SDC | scale SDC | ratio |
|---|---|---|---|
| `e4m3` | 30.3% | **91.8%** | 3.0× |
| `e5m2` | 27.1% | **82.5%** | 3.0× |
| `e3m2` | 49.5% | **92.8%** | 1.9× |

Scale faults corrupt **37–41 of 200 images** on average versus 0.7–15.6 for element
faults, and **9–13% produce non-finite output**. The structural asymmetry the project is
built on is real and large.

## F5. Uniform sampling starves the stratum that matters

Scale bits are **3.09%** of storage, so a 3,000-injection uniform campaign drew only
**97–125 scale faults**. Their CI is correspondingly wide ([0.85, 0.96] for `e4m3`) while
the element CI is tight. The most consequential stratum is the least measured — this is
the empirical case for stratified allocation, and it is worth reporting as a motivating
result rather than an implementation detail.

## F6. The OCP scale rule reverses the format ranking

At K=32, fp32 baseline 0.8701:

| format | OCP | fit (no clipping) |
|---|---|---|
| `e4m3` (8b) | 0.8508 | **0.8610** |
| `e2m3` (6b) | **0.8572** | 0.8578 |

Under the OCP rule the 6-bit format beats the 8-bit one; under a non-clipping scale rule
the ordering flips back. `e2m3` only wins because it clips less (93.75% headroom vs
87.5%). A format comparison run solely under OCP would report an artifact of the scale
rule as a property of the format.

## F7. Block size is nearly free in accuracy — so it is a clean reliability axis

Across K = 8/16/32/64 every format moves < ~1 pp (`e4m3`: 0.8482 → 0.8514). Any
reliability difference found across block sizes is therefore not confounded by an
accuracy difference. Objective O3 can be answered cleanly.

---

## Caveats

- SDC is defined as *any* top-1 change on a 200-image subset, so absolute rates are high
  by construction; they are comparable across configurations, not to other papers.
- Mean accuracy drop is occasionally slightly negative (a fault flipping a wrong
  prediction to a right one). SDC is the stable metric at this subset size.
- The scale stratum has only ~100 samples per campaign; its rates are directionally
  solid but need stratified sampling to tighten.
- Weights only in F1-F7. Activations are covered in F12-F13.

---

# Stratified allocation (E02) — ResNet8, e4m3, K=32

Three allocations at an identical 3,000-injection budget, pilot rates taken from the E01
uniform campaign. Raw: `results/e02_*.csv`.

## F8. The estimator is unbiased — all three arms agree

| arm | scale samples | model-wide SDC | scale-stratum SDC |
|---|---|---|---|
| `uniform` | 96 | 0.3252 ± 0.0158 | 0.9062 ± 0.0593 |
| `neyman` | 61 | 0.3279 ± 0.0163 | 0.8852 ± 0.0809 |
| `neyman_phi` | 1083 | 0.3274 ± 0.0195 | 0.8947 ± 0.0183 |

Model-wide estimates agree to within 0.003 despite allocations differing by 18x on the
scale stratum. The inclusion-weighted estimator works on real data.

## F9. The blast-radius weight Φ buys 10.5x on the scale stratum

`neyman_phi` moves 36.1% of the budget to scale bits (which hold 3.09% of storage) and
cuts the scale-stratum interval from ±0.0593 to ±0.0183 — a **10.5x variance reduction**,
i.e. a uniform campaign would need ~31,500 injections to measure that stratum as
precisely as 3,000 stratified ones do.

This is the defensible version of the drafts' Φ claim: it makes the rare,
high-blast-radius stratum affordable to characterise.

## F10. But Φ costs model-wide precision — it is a trade, not a free win

The same arm *widens* the model-wide interval from ±0.0158 to ±0.0195 (0.65x, i.e. worse).
Over-sampling a stratum that holds 3% of the population necessarily under-samples the 97%
that dominates the model-wide rate.

**The drafts present Φ as an improvement to a single allocation formula. It is not — it
selects which estimate is precise.** Report Φ as tuning the estimator toward per-site
characterisation, and state which quantity the "fewer injections at equal confidence"
claim refers to. For the model-wide rate, Φ makes the claim false.

## F11. Layer x site stratification alone buys nothing — the negative result

Plain Neyman (no Φ) scored **0.94x** against uniform on the model-wide rate: marginally
*worse*, not better. The reason is visible in the pilot: per-layer SDC rates cluster
tightly around 0.3, so `sqrt(r(1-r))` is nearly constant and Neyman degenerates to
proportional allocation — which is uniform sampling, plus rounding and floor overhead.

**Implication for the supporting sampling tool.** TreeFI's reported reduction (up to 72x) comes
from stratifying by *value interval* within a layer, exploiting the spread of bit-flip
impact across the value distribution. Stratifying by layer and site does not reproduce it,
because that spread is not where the variance lives. A credible MX-TreeFI result therefore
would require value-interval strata coupled across the (scale, element) pair, which is
**not yet implemented**.

Practical consequence for this project: for model-wide rates, **plain uniform sampling is
the honest default** — layer/site stratification adds machinery without precision. Use
stratification only where it demonstrably pays, which so far is the scale stratum via Phi
(F9). Whether value-interval strata pay is worth one experiment, but it is a
cost-control question, not a contribution.


---

# Tensor type and block size (E03/E03b/E04)

## F12. Weights vs activations must be compared per *inference*, not per campaign

A weight fault is **persistent** -- it sits in weight memory and corrupts every inference
that reads it. An activation fault is **transient** -- it corrupts one buffer during one
inference and is gone. So "SDC = any of 200 images changed" is the right question for a
weight fault and the wrong one for an activation fault, which can only ever affect the
single inference it lands in. Scoring activations that way understates them by ~2 orders
of magnitude.

The comparable quantity is `P(a given inference is corrupted | one fault)`:

| site | weight | activation | ratio |
|---|---|---|---|
| element | 0.0170 | 0.0032 | 5.3x |
| scale | 0.1885 | 0.1532 | **1.2x** |
| overall | 0.0225 | 0.0125 | 1.8x |

Activation blocks run down the channel axis and the batch axis is separate, so each
image's blocks are independent -- batch-of-one is the *exact* per-inference fault space,
not an approximation.

**Objective O4 as written does not specify this normalisation. It must, or the
weight-vs-activation comparison is not meaningful.**

## F13. The shared scale dominates in *both* tensor types

Element faults are 5.3x worse in weights than activations, but scale faults are nearly
**equally** dangerous in both (1.2x). Within activations the site asymmetry is even more
extreme than in weights: **48x** (15.3% vs 0.32%) against ~11x for weights.

Activation element faults are remarkably benign -- 0.32%, with **zero** non-finite outputs
in 1,876 injections. A corrupted activation feeds few output positions and is clamped by
the next ReLU.

Per layer, the stem `conv1` dominates at 11.4%, ~10x the next layer, since its errors
propagate through the whole network; `layer3.conv2` and `layer3.shortcut.0` showed zero
corruption. Bit 7 (sign) is again the worst element bit, consistent with F3.

**Design consequence:** protecting shared scales pays regardless of which tensor is being
hardened. That is the single most actionable result the project has produced.

## F14. Block size: the O3 trade, measured (scale side)

| | K=8 | K=16 | K=32 | K=64 |
|---|---|---|---|---|
| scale-fault SDC (`e4m3`) | 0.731 | 0.890 | 0.924 | 0.941 |
| images corrupted / scale fault | 40.1 | 45.3 | 48.7 | 50.1 |
| scale bits as share of storage | 11.1% | 5.9% | 3.1% | 1.7% |

Monotone in all three, in both `e4m3` and `e2m1`. Doubling K halves the scale-bit
footprint while making each scale fault strictly more damaging. With accuracy flat across
K (F7), this is a clean efficiency-vs-reliability curve with no accuracy confound.

## F15. The OCP scale rule *manufactures* a spurious block-size trend

Element-fault SDC with **matched faults** -- one fixed set of (layer, weight position, bit)
triples injected into every K, so only the block grouping varies:

| K | OCP | fit (no clipping) |
|---|---|---|
| 8 | 0.1973 | 0.2220 |
| 16 | 0.3113 | 0.2273 |
| 32 | 0.3200 | 0.2307 |
| 64 | 0.1447 | 0.2313 |
| **spread** | **2.21x, non-monotone** | **1.04x, CIs overlap -- flat** |

Under a non-clipping scale rule the element side is **flat in K**, exactly as the
perturbation physics predicts (for a fixed weight and bit, mean `|dw|` moves only ~9%
across K=8..64). The entire 2.2x non-monotone swing is an artifact of OCP clipping.

The mechanism: small blocks mean *more* block maxima, and the OCP rule clips every one of
them by up to 12.5%. At K=8 there are 8x as many clipped maxima as at K=64. The signature
is visible directly in quantisation error on `layer3.conv2`, which *decreases* with K
(0.005677 -> 0.005174) -- backwards from the usual expectation. `fit` also gives uniformly
better and more stable accuracy (0.8650 at every K, vs 0.8350-0.8400 under OCP).

**This is a stronger version of F6.** Clipping does not merely reverse format rankings --
it fabricates an apparent "optimal block size for reliability" around K=16-32 that does
not exist. Anyone running the block-size sweep under the spec-default OCP rule, which is
the obvious thing to do, would report that artifact as a result.

## F16. Severity falls with K; frequency does not

Under `fit`, separating the two components of impact:

| K | element SDC (frequency) | mean images changed (severity) | non-finite |
|---|---|---|---|
| 8 | 0.2220 | 3.86 | 1.73% |
| 16 | 0.2273 | 3.08 | 1.33% |
| 32 | 0.2307 | 2.82 | 1.20% |
| 64 | 0.2313 | 2.70 | 1.13% |

Frequency is flat -- a bit flip is a bit flip. Severity falls monotonically, because a
larger block has a larger maximum, so typical values sit further below the shared scale,
land in lower binades, and a flipped bit moves them less in absolute terms.

Combined with F14, the complete O3 answer: **larger K makes element faults milder and
scale faults worse, while shrinking the scale-bit target.** The two sites move in opposite
directions, which is why block size cannot be tuned on a single aggregate number.


---

# Cross-architecture replication: ResNet8 vs RepVGG-A0

RepVGG-A0 (7.0M params fused, 30 epochs, **91.44%** top-1) re-parameterised to single 3x3
convolutions, versus ResNet8 (78K params, 60 epochs, 87.01%) BN-folded. Same campaigns,
same protocol. Raw: `results/*_repvgg_a0_*`, summary
`results/cross_architecture_summary.csv`.

## F17. The F1 phenomenon replicates -- its *direction* does not

| model | accuracy gap (`e5m2` vs `e3m2`) | `e5m2` SDC | `e3m2` SDC | ratio |
|---|---|---|---|---|
| ResNet8 | 0.05 pp | 0.2890 | 0.5127 | `e3m2` **1.77x worse** |
| RepVGG-A0 | 0.07 pp | 0.0597 | 0.0083 | `e3m2` **7.2x safer** |

Both models: matched accuracy to <0.1 pp, vulnerability differing with **disjoint**
confidence intervals. So the drafts' core claim -- *accuracy is not a reliability proxy* --
is confirmed twice, on a residual net and a fused plain net.

But which format is safer **flips**. Any paper reporting "MXFP6-`e3m2` is less reliable
than MXFP8-`e5m2`" from one model would be stating an architecture-specific artifact as a
property of the format. **Report the phenomenon, not the ranking**, unless the ranking is
shown on several architectures.

## F18. Why it flips: the dominant failure mode shifts with model capacity

Share of element-fault SDCs that were **non-finite** (NaN/Inf) rather than ordinary
numeric perturbation:

| model | `e5m2` element SDC | of which non-finite | `e3m2` element SDC | of which non-finite |
|---|---|---|---|---|
| ResNet8 | 0.2711 | **28.2%** | 0.4946 | 0.0% |
| RepVGG-A0 | 0.0541 | **100.0%** | 0.0007 | 0.0% |

On RepVGG **every single** element failure in `e5m2` was a non-finite event -- 158 of 158.
Not one ordinary perturbation changed a prediction.

The mechanism is then forced:

* A 7M-parameter network **absorbs** mild perturbations that a 78K one cannot. ResNet8's
  `e3m2` produced 1422 SDCs purely by numeric perturbation; RepVGG's produced **2**.
* What capacity cannot absorb is a NaN or Inf, which propagates unconditionally.
* Only formats carrying Inf/NaN codes can produce one. `e5m2` has them; `e3m2`, `e2m3` and
  `e2m1` have no special codes at all, so no element flip can ever produce a non-finite
  value (F2, confirmed here at n=2885).

**So as capacity grows, the failure rate stops being driven by how much a bit flip
perturbs a value and becomes driven by whether the format can represent NaN/Inf at all.**
For large models the design rule is simply: *prefer element formats without special
codes*. This also predicts that on a large Transformer, `e4m3` (which has NaN) should look
worse than its bit width suggests -- a cheap, falsifiable test.

## F19. F4 replicates and *strengthens* with model size

| model | `e4m3` element | `e4m3` scale | ratio |
|---|---|---|---|
| ResNet8 | 0.3031 | 0.9175 | 3.0x |
| RepVGG-A0 | 0.0117 | 0.2073 | **17.8x** |

Element SDC collapses by ~26x between the models while scale SDC falls only ~4.4x, so the
asymmetry widens. The reason follows F18: capacity absorbs a single corrupted weight, but
a scale fault corrupts 32 at once and still lands.

`e3m2` on RepVGG reaches a 288x ratio, but that rests on **2 element SDCs in 2885
injections** -- directionally right, not a number to quote. Quote `e4m3`'s 17.8x.

**The project's most robust result: shared-scale faults dominate on both architectures,
both tensor types (F13), and the gap grows with model size.**

## F20. F15 replicates, more strongly; F6's ranking-reversal does not

Element SDC across K, matched faults:

| model | rule | K8 | K16 | K32 | K64 | spread |
|---|---|---|---|---|---|---|
| ResNet8 | ocp | 0.1973 | 0.3113 | 0.3200 | 0.1447 | 2.21x |
| ResNet8 | fit | 0.2220 | 0.2273 | 0.2307 | 0.2313 | **1.04x flat** |
| RepVGG-A0 | ocp | 0.0193 | 0.0100 | 0.0027 | 0.0040 | **7.25x** |
| RepVGG-A0 | fit | 0.0227 | 0.0200 | 0.0153 | 0.0180 | 1.48x, CIs overlap |

Under OCP the apparent block-size trend is **7.25x** on RepVGG -- larger than ResNet8's
2.21x, as predicted from its wider fused-kernel dynamic range. Under a non-clipping rule
it flattens on both. **F15 is a two-architecture result: the OCP scale rule fabricates a
block-size reliability trend.** (RepVGG's `fit` row is flat only by overlapping CIs, not
as cleanly as ResNet8's -- worth more samples before final write-up.)

The **accuracy** half of F6 needs narrowing. Clipping cost, `fit` minus `ocp`, in pp at
K=32:

| format | ResNet8 | RepVGG-A0 |
|---|---|---|
| `e2m1` | +0.55 | **+7.88** |
| `e4m3` | +1.02 | +0.27 |
| `e3m2` | **-0.90** | +0.32 |
| `e5m2` | **-0.80** | +0.30 |

On RepVGG clipping costs accuracy for *every* format, and MXFP4 loses a punishing 7.9 pp.
On ResNet8 two formats came out *better* under clipping, which is what produced F6's
"ranking reversal". That reversal does **not** replicate. Retain: *OCP clipping costs
accuracy, severely for MXFP4.* Drop: *it reverses format rankings.*

## Caveats on the comparison

- The models are **not matched** on training budget (30 vs 60 epochs) or accuracy (91.4%
  vs 87.0%). A difference between them could stem from either. F18's capacity mechanism is
  supported by the non-finite shares directly, not by the accuracy gap alone, but a
  width-swept study would pin it properly.
- One seed per model.
- `e2m1` at K=64 dips on both models (0.7588 here, 0.7335 on ResNet8 under `fit`) while
  every other format is flat in K. Appearing twice suggests a real MXFP4 interaction;
  unexplained.


---

# Width sweep: capacity isolated (E05)

ResNet8 at five base widths, **identical topology, data and 30-epoch budget**, 35x
parameter range. This removes the confounds in F18's original evidence, which compared two
models differing in depth, width, topology, training budget and accuracy at once. Formats
chosen as a gradient in special-code availability: `e5m2` (Inf+NaN, 8 codes), `e4m3` (NaN
only, 2), `e3m2` (none). Raw: `results/width_sweep.csv`.

| width | params | accuracy |
|---|---|---|
| 8 | 19,954 | 0.7997 |
| 16 | 78,042 | 0.8508 |
| 24 | 174,274 | 0.8773 |
| 32 | 308,650 | 0.8840 |
| 48 | 691,834 | 0.8956 |

## F21. The dose-response predicted by F18 is observed, in the exact predicted order

> **Revised by F23 (2 seeds).** The full three-format ordering held for seed 0 only. What
> survives is narrower: `e3m2` gains more from capacity than `e5m2`. The
> `e4m3`-vs-`e5m2` order is unresolved. The mechanism claims below (non-finite rate flat,
> non-finite share rising) replicate (F24). Seed-0 numbers are kept here as originally
> reported.

Element SDC, width 8 -> 48 (seed 0):

| format | special codes | SDC at w=8 | SDC at w=48 | ratio |
|---|---|---|---|---|
| `e3m2` | **0** | 0.4156 | 0.0268 | **0.06x** |
| `e4m3` | 2 | 0.3657 | 0.0570 | **0.16x** |
| `e5m2` | 8 | 0.4469 | 0.1209 | **0.27x** |

F18 predicted the benefit of capacity should rank `e3m2` < `e4m3` < `e5m2`, tracking
special-code count. That is exactly what happened, monotonically. All three formats start
within 0.37-0.45 at the smallest width and fan out by 4.5x at the largest.

The two mechanisms separate as the model predicts:

* **Perturbation-driven SDC falls with capacity for every format** -- correlation against
  log(params): `e5m2` r=-0.885, `e4m3` r=-0.817, `e3m2` r=-0.520.
* **Non-finite *rate* is flat in width** (`e5m2` 0.073 -> 0.080 across 35x capacity). This
  is the key control: the chance a bit flip lands on a NaN/Inf code is a property of the
  *format*, not the model, and no amount of capacity absorbs a NaN.
* **Non-finite *share* of failures therefore rises** -- `e5m2` 0.162 -> 0.662 (r=+0.803),
  `e4m3` 0.042 -> 0.223 (r=+0.523), `e3m2` structurally 0.000 at every width.

So capacity absorbs perturbations and cannot absorb non-finite values; formats that can
produce the latter keep a floor of failures that scaling never removes. **F18 is now a
mechanism demonstrated on a controlled axis, not an explanation fitted to two models.**

## F22. But the per-width scatter is large -- treat the endpoints, not the curve

Perturbation-driven SDC by width:

| format | 8 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|
| `e5m2` | 0.3744 | 0.2677 | 0.1558 | 0.2514 | 0.0409 |
| `e4m3` | 0.3505 | 0.3989 | 0.0393 | **0.0024** | 0.0443 |
| `e3m2` | 0.4156 | 0.1551 | **0.5366** | 0.2357 | 0.0268 |

The trend is real at the endpoints but badly non-monotone in between -- `e4m3` at w=32
(0.0024) and `e3m2` at w=24 (0.5366) are stark outliers. Each width is **one trained model
with one seed**, and between-model variation in loss-landscape sensitivity evidently
swamps the capacity trend at any single width. Injection count is not the limit here
(n=2918 element faults per cell); *model* count is.

Consequences for the write-up:

* Quote the **endpoint ratios and the ordering** (F21), which are robust and were predicted
  in advance. Do not present the five-point curves as smooth trends.
* The r values rest on 5 points, 1 seed each. They support the direction; they are not
  strong evidence on their own.
* Accuracy still rises with width (0.7997 -> 0.8956), so capacity and accuracy remain
  correlated by construction. What rules out "more accurate models are simply more robust"
  is that the benefit differs *by format* in the predicted order -- a uniform accuracy
  effect would shift all three equally.
* The proper fix is 3-5 seeds per width. At ~0.5-6 h training per width that is a
  multi-day run, and it is the single highest-value use of that time.

> **Update (F23).** The first extra seed confirms the diagnosis and shows the bullet on
> "endpoint ratios and the ordering" was too optimistic. Endpoint ratios move a lot
> between seeds as well.


---

# Width sweep, multiple seeds (E05b)

Every width retrained under additional seeds (`run_width_sweep_seeds.sh`); same
topology, data, 30-epoch budget and 3,000-fault campaigns per cell. **This section uses
seeds 0 and 1, the complete seeds as of 2026-09-19.** Seed 2 is partway through and is
left out so every width has the same number of models. Raw:
`results/width_sweep_multiseed.csv`; analysis output:
`results/width_sweep_multiseed_after_seed1.txt`
(`python -m experiments.width_sweep_multiseed`).

| width | params | acc seed 0 | acc seed 1 |
|---|---|---|---|
| 8 | 19,954 | 0.7997 | 0.7866 |
| 16 | 78,042 | 0.8508 | 0.8505 |
| 24 | 174,274 | 0.8773 | 0.8695 |
| 32 | 308,650 | 0.8840 | 0.8855 |
| 48 | 691,834 | 0.8956 | 0.9008 |

Accuracy is reproducible to within 1.3 pp at every width. Fault sensitivity is not (below).

## F23. With two seeds, only part of F21's ordering holds

> **Confounded by the SDC metric (F26).** Every number below uses "any of 200 images
> changes", which mostly tracks how many near-tie images each golden model has. Under a
> tie-robust metric, every format falls steadily with width, and at w=48 there are too
> few events left to rank the formats. Treat this finding as unresolved, not as evidence
> for or against a format ordering.

Endpoint ratio, element SDC(w=48) / SDC(w=8):

| format | special codes | seed 0 | seed 1 | mean ± std |
|---|---|---|---|---|
| `e3m2` | 0 | 0.064x | **0.008x** | 0.036 ± 0.040x |
| `e4m3` | 2 | 0.156x | 0.199x | 0.177 ± 0.030x |
| `e5m2` | 8 | 0.271x | 0.103x | 0.187 ± 0.118x |

Seed 1 gives `e3m2` < `e5m2` < `e4m3`, so the predicted order held in **1 of 2 seeds**.
A random order would match at least once in two seeds with p = 0.31, so this is no
evidence for the full ordering.

Capacity slope, d log(element SDC) / d log(params), fitted per seed. The differences are
paired within a trained model, so model-to-model variance cancels:

| comparison | difference ± se | t | negative in |
|---|---|---|---|
| `e3m2` − `e4m3` | −0.233 ± 0.454 | −0.51 | 1/2 seeds |
| `e4m3` − `e5m2` | −0.230 ± 0.244 | −0.94 | 1/2 seeds |
| `e3m2` − `e5m2` | **−0.463 ± 0.210** | **−2.20** | **2/2 seeds** |

What this supports:

* **Supported:** the format with no special codes gains the most from capacity, compared
  with the format with the most. `e3m2` is also the safest format at w=48 in both seeds
  (0.027 and 0.006, against 0.057–0.121 for the others).
* **Not supported:** a *graded* dose-response in special-code count. Whether `e4m3`
  sits between the other two is unresolved. With 2 seeds and t ≈ −0.5 to −0.9, the test
  lacks the power to say.
* The t = −2.2 rests on 2 seeds (1 degree of freedom for the between-seed se). It is a
  consistent direction, not significance.

## F24. The mechanism replicates, and it is clearer in seed 1

F18 predicts two mechanisms: capacity absorbs perturbations, and a NaN/Inf code yields a
format-fixed floor that capacity cannot absorb. Both show up in the new seed:

* **Non-finite rate stays flat in width**, now across 10 models: `e4m3` 0.0122–0.0134
  (max/min 1.10x), `e5m2` 0.0710–0.0831 (1.17x). `e3m2` is 0 in every model, as it must
  be.
* **Non-finite share of failures rises with width.** For `e5m2` the mean goes from 0.13 at
  w=8 to 0.81 at w=48. At seed 1, w=48, **97% of `e5m2` failures are non-finite**:
  perturbation-driven SDC is 0.0024, while the non-finite rate alone is 0.072. On that
  model, `e5m2` fails almost only through its special codes, which is F18's mechanism in
  its purest form.

So the *mechanism* (F18) is on firmer ground than the *ordering* (F21). The mechanism
rests on quantities that barely vary between models (non-finite rate). The ordering rests
on perturbation-driven SDC, which varies a lot.

## F25. The scatter comes from the models, not from sampling

> **Reinterpreted by F26.** The scatter is real, but most of it is not a difference in how
> sensitive the models are to faults. It is which images in the 200-image subset happen
> to sit near a class tie in each quantised model. The `e4m3` w=32 "exception" below is
> explained there.

Between-seed std divided by injection-sampling std, per cell:

| format | 8 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|
| `e3m2` | 18.4 | 0.8 | 43.2 | 22.7 | 6.4 |
| `e4m3` | 3.2 | 26.3 | 10.1 | 0.8 | 3.6 |
| `e5m2` | 20.9 | 13.2 | 12.8 | 21.2 | 6.0 |

The median is **12.8x**. Two trained models of the same width, identical except for the
seed, differ in fault sensitivity by roughly an order of magnitude more than the
injection CI. F22's outliers were properties of the models:

* `e3m2` at w=24: 0.537 in seed 0 but **0.024** in seed 1, a 22x swing between two
  models of equal accuracy (0.877 vs 0.870).
* `e4m3` at w=16: 0.411 in seed 0 but 0.109 in seed 1.
* **Exception:** `e4m3` at w=32 is anomalously low in *both* seeds (element SDC 0.0175
  and 0.0148; perturbation-driven 0.0024 and 0.0034). F22 called this an outlier, but it
  replicates. It is unexplained and should be checked against seeds 2–4 before being
  read as anything.

Consequences:

* A single-model fault-injection result is a sample of size 1 from a wide distribution,
  **even when its CI is tight**. Per-model CIs understate the uncertainty in any claim
  about a *format* or *architecture*. That also applies to F1/F17's cross-architecture
  comparison, which used one model per architecture.
* More injections per model will not fix this. More models will. Seeds 2–4 are running.
  Re-run `experiments.width_sweep_multiseed` after each seed and revisit F23.


---

# The SDC metric counts near-ties (E05c)

## F26. "Any image changes" SDC is dominated by near-tie images in the golden model

Every campaign so far scores a fault as SDC if **any** of the 200 subset images changes
its top-1 prediction. Investigating the `e4m3` w=32 anomaly (F25) shows that this rate
depends mostly on how many images the *fault-free quantised* model places almost exactly
between two classes. Such an image flips under a nudge of almost any size, so one
knife-edge image can dominate a whole campaign.

**The anomaly, reproduced directly.** On the seed-0 w=32 model, the same 60 random
element sign flips (hidden layers) were injected under each format:

| format | sign-flip SDC | median max \|Δlogit\| | max \|Δlogit\| |
|---|---|---|---|
| `e4m3` | **1/60** | 0.125 | 2.09 |
| `e3m2` | **26/60** | 0.118 | 2.23 |
| `e5m2` | 27/60 | 0.118 | 2.23 |

The faults disturb the logits by the same amount, yet the SDC counts differ 26x. So the
difference sits in the model receiving the fault, not in the fault. The quantised
baselines are healthy (`e4m3` subset accuracy 0.895 and 0.840, level with the other
formats). Element magnitudes and code usage match the other widths, and the campaign
code checks out.

**The cause: near-tie images.** Golden top1 − top2 logit margin on the 200-image subset,
w=32:

| seed | format | smallest margin | 2nd | 3rd | images < 0.2 | perturbation SDC |
|---|---|---|---|---|---|---|
| 0 | `e4m3` | **0.340** | 0.396 | 0.402 | **0** | 0.0024 |
| 0 | `e3m2` | 0.001 | 0.043 | 0.078 | 4 | 0.2357 |
| 0 | `e5m2` | 0.001 | 0.039 | 0.076 | 4 | 0.2749 |
| 1 | `e4m3` | **0.216** | 0.224 | 0.321 | **0** | 0.0035 |
| 1 | `e3m2` | 0.049 | 0.097 | 0.167 | 5 | 0.0319 |
| 1 | `e5m2` | 0.053 | 0.105 | 0.167 | 5 | 0.0280 |

The two `e4m3` w=32 models are the only models of all 39 (widths 8–48, seeds 0–2, three
formats) with no image margin below 0.2. Across all 39, perturbation-driven SDC tracks
the golden near-tie count:

* Spearman(SDC, number of images with margin < 0.2) = **+0.89** (p = 4e-14)
* Spearman(SDC, smallest margin) = −0.76 (p = 1e-8)

F22's other outlier is the same effect: seed 0, w=24, `e3m2` has an exact tie
(margin 0.0000) and shows 0.537 SDC, while the seed-1 model of that width shows 0.024.

**Why formats differ on the same model.** `e3m2` and `e5m2` both keep 2 mantissa bits,
so over the range the weights occupy they round alike. Their golden logits, near-ties
and SDC rates nearly coincide model by model (table above). `e4m3` rounds to 3 mantissa
bits, which moves its logits by a few hundredths. That is enough to create or remove a
tie. Near-tie count is therefore effectively a random draw per (model, format), and it
is exactly the variable the metric is most sensitive to.

**Under tie-robust metrics** (seeds 0–1, perturbation-driven element faults), seed
max/min ratios shrink:

| metric | w=24 `e3m2` | w=32 `e5m2` | w=48 `e5m2` |
|---|---|---|---|
| any image changes (current) | 22.3x | 9.8x | 17.2x |
| ≥ 3 of 200 images change | 6.7x | 1.8x | 3.7x |

With "≥ 3 images change", every format falls steadily with width. They start at
0.16–0.21 at w=8 and reach 0.0002–0.0003 at w=48, which is about 1 event in 3,000
injections. That floor has too few events to rank the formats. The per-width curves
are also far smoother than in F22, so much of F22's "non-monotone scatter" was
near-tie noise as well.

What this changes:

* **F25** is reinterpreted: most of the between-seed scatter comes from near-tie images,
  not from how sensitive the models are to faults.
* **F23** is confounded: the format ordering it tests is not measurable with the current
  metric and data.
* **F21**'s capacity effect survives (every format improves with width under every
  metric tried). Its format-specific slopes do not.
* **F24 is unaffected.** A non-finite output makes every prediction invalid, so it is
  SDC however large the margins are. The non-finite rate never depended on the margin.
* **Earlier "any image changes" comparisons need rechecking**, notably F1 (`e5m2` vs
  `e3m2`, 1.77x) and F17–F20 (cross-architecture). A format or architecture gap of a
  few x is within what near-tie differences alone produce. These have not been
  re-examined yet. They may hold, but they are no longer established.

**Fix.** The campaign CSVs store only the *count* of changed images, not which ones, so
the existing data cannot separate knife-edge flips from real damage. Future campaigns
should log the per-image change set (or golden margins) so SDC can be scored while
excluding images with golden margin below δ, or as margin-normalised damage. They
should also use a larger image subset, to reduce how much one image can matter.
Changing `mxfi/campaign.py` is deferred until the running multi-seed sweep finishes,
so seeds 2–4 are scored like seeds 0–1. Re-running the campaigns afterwards reuses the
trained checkpoints.

Reproduce with `experiments/e05c_near_ties.py`:

    .venv/Scripts/python.exe -m experiments.e05c_near_ties signflip --width 32 --train-seed 0
    .venv/Scripts/python.exe -m experiments.e05c_near_ties margins
    .venv/Scripts/python.exe -m experiments.e05c_near_ties robust --seeds 0 1

`margins` re-quantises each checkpoint and runs the 200-image `campaign_subset`
(→ `results/e05c_golden_margins.csv`). It covers every *finished* campaign, so its cell
count grows as the sweep runs: the 39 cells above are seeds 0–1 plus seed 2 at
w=8–24. `robust` rescores the `changed` / `acc_drop` / `change_rate` columns of the
existing `e01_*` CSVs (→ `results/e05c_tie_robust_metrics.csv`).

---

# Multi-seed width sweep (E05b) -- 3 of 5 seeds

Seeds 0-2 complete: ResNet8 at widths 8/16/24/32/48, 30 epochs each, three formats,
3000 injections per cell (135,000 injections). Raw: `results/width_sweep_multiseed.csv`.
Seeds 3-4 still running. Design note: every seed of a width is probed at the *same* fault
sites, and all three formats run on the *same* trained model, so format comparisons are
paired within a seed.

## F23. The F18 mechanism is confirmed; the three-way ordering is not yet

**Confirmed -- the non-finite rate is a property of the format, not the model.** Across a
35x capacity range it moves by only 9%:

| format | non-finite rate, w8 -> w48 | spread |
|---|---|---|
| `e5m2` | 0.0735+-0.0046 -> 0.0764+-0.0042 | 1.09x |
| `e4m3` | 0.0128+-0.0022 -> 0.0128+-0.0012 | 1.09x |
| `e3m2` | **0.0000 at every width and seed** | -- |

This is the control the whole mechanism rests on: capacity absorbs perturbations but can
never absorb a NaN, and only formats with special codes can produce one. The share of
failures that are non-finite therefore climbs with capacity (`e5m2` 0.124 -> 0.849).

**Confirmed -- capacity helps least where special codes exist.** Capacity slope,
d log(element SDC) / d log(params), paired within seed:

| format | special codes | slope |
|---|---|---|
| `e3m2` | 0 | **-0.981 +- 0.221** |
| `e4m3` | 2 | -0.701 +- 0.047 |
| `e5m2` | 8 | **-0.506 +- 0.100** |

The extremes separate cleanly: `slope[e3m2] - slope[e5m2] = -0.475 +- 0.122`, **negative in
3/3 seeds, t = -3.90** (one-tailed p ~ 0.03 at df=2, direction predicted in advance).

**Not confirmed -- the strict e3m2 < e4m3 < e5m2 ordering.** It held in seed 0 and seed 2
but not seed 1, where `e4m3` (0.199x) and `e5m2` (0.103x) swapped. 2 of 3 seeds gives
P >= 2 by chance = 0.074, which is not enough. The adjacent-pair slope differences are also
indecisive: `e3m2`-`e4m3` t = -1.05, `e4m3`-`e5m2` t = -1.35, each negative in 2/3 seeds.

**So state F21 as a two-point claim, not a three-point one**: formats with no special codes
benefit substantially more from capacity than formats with Inf *and* NaN. Whether `e4m3`
(NaN only) sits strictly between them is unresolved at 3 seeds. Seeds 3-4 may settle it; if
they do not, the honest report is a monotone trend in the means with a resolved gap only
between the extremes.

## F24. F22's scatter was model-to-model, quantified

Between-seed standard deviation of element SDC, divided by the standard deviation expected
from the 3000 injections alone:

| format | w8 | w16 | w24 | w32 | w48 |
|---|---|---|---|---|---|
| `e3m2` | 14.0 | 4.0 | 37.8 | 18.1 | 5.1 |
| `e4m3` | 7.8 | 23.3 | 7.3 | 16.5 | 6.6 |
| `e5m2` | 16.7 | 9.6 | 9.9 | 16.6 | 4.6 |

Median **9.9x**. Differences between independently trained models are about ten times
larger than the uncertainty from finite injection counts. **More injections per cell would
not have fixed F22; only more seeds can.** Any future MX reliability result quoted from a
single trained model should carry this caveat -- it is a general lesson about fault-injection
methodology, not a quirk of this sweep.

---

# Exhaustive ground truth (E06) -- ResNet8-w16, MXFP8, K=32

**Every one of the 638,624 bits** of the model's MX weight storage was flipped and scored.
This is not an estimate and carries no confidence interval. Run on an A100 at ~170
injections/s (~1.1 h). Raw: `results/e06.csv.gz`.

## F53. The sampling methodology is validated

*(Renumbered: this and F54 were written as F25 and F26, numbers the multi-seed width sweep had already used. The F23 and F24 in that block and in the 3-of-5-seeds block are deliberate revisions of the same findings, not collisions.)*

| | value |
|---|---|
| **exhaustive truth** | **0.417114** |
| n=3000 sampled estimate | 0.427000 |
| 95% CI of that estimate | [0.409404, 0.444783] |
| error | +0.0099 (**2.37% relative**) |
| **CI covers the truth** | **YES** |

Every sampled failure rate in this project rests on the assumption that 3000 uniform
injections estimate the true rate with the stated interval. That assumption is now
*verified* rather than asserted, on the one model where checking it is affordable. This is
the ground-truth validation the drafts ask for.

## F54. The site asymmetry, measured exactly

| site | n | SDC rate | images corrupted | non-finite |
|---|---|---|---|---|
| element | 618,880 | 0.400297 | 3.27 | 0.0127 |
| **scale** | 19,744 | **0.944236** | **49.89** | **0.1233** |

Exact ratio **2.36x** in failure *rate* and **15.3x** in blast radius. Note what the exact
numbers add over the sampled ones: **94.4% of all possible shared-scale bit flips corrupt
the model**, and two scale bit positions are **catastrophic without exception**:

| scale bit | SDC rate |
|---|---|
| 3 | **1.000000** |
| 7 | **1.000000** |
| others | 0.883 - 0.957 |

Bit 7 shifts the exponent by 128 (past the float32 range, hence `inf`); bit 3 shifts it by
8, i.e. scaling the whole block by 256 or 1/256. Not one of the 2,468 flips at either
position left the predictions intact. A design that protects only a couple of bits per
scale byte should protect these two first.

## F27. The sign bit is confirmed worst, exactly

Element SDC by bit position, over the complete space (77,360 flips per bit):

| bit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **7 (sign)** |
|---|---|---|---|---|---|---|---|---|
| SDC | 0.130 | 0.212 | 0.314 | 0.421 | 0.532 | 0.528 | 0.474 | **0.593** |

The sign bit is the single most damaging element bit in the entire fault space -- higher
than every exponent bit -- and the drafts' severity metric scores it as **zero damage**
(F3). This is now exact rather than sampled, which closes that argument.

## F28. Per-layer vulnerability spans 3x, and it is not the largest layers

| layer | bits | SDC |
|---|---|---|
| `conv1` (stem) | 3,584 | **0.853** |
| `layer2.shortcut.0` | 4,352 | 0.726 |
| `layer1.conv2` | 19,072 | 0.724 |
| `layer2.conv1` | 38,144 | 0.667 |
| `layer1.conv1` | 19,072 | 0.650 |
| `layer2.conv2` | 76,032 | 0.560 |
| `fc` | 5,280 | 0.550 |
| `layer3.conv1` | 152,064 | 0.448 |
| `layer3.shortcut.0` | 16,896 | 0.411 |
| `layer3.conv2` | 304,128 | **0.289** |

Vulnerability runs **opposite to size**: the 3,584-bit stem is the most fragile layer at
0.853, the 304,128-bit final convolution the most robust at 0.289. Since a uniform campaign
samples in proportion to size, it spends 48% of its budget on the most robust layer and
0.6% on the most fragile -- an argument for layer-stratified sampling that F11 could not
make from layer/site rates alone, because those were measured too coarsely to show this
3x spread.

---

# Ten-seed width sweep, all-GPU (E05c)

Fifty models: ResNet8 at five widths, ten independently trained seeds each, every model
trained *and* measured on one A100 so the dataset is internally consistent. 150 campaigns,
450,000 injections. The four complete CPU seeds remain as an independent replication on
different hardware. Raw: `results/width_sweep_multiseed.csv`.

## F29. The dose-response is established: capacity helps in inverse proportion to special codes

Capacity slope $\mathrm{d}\log(\mathrm{SDC})/\mathrm{d}\log(\mathrm{params})$, ten seeds:

| format | NaN/Inf codes | slope |
|---|---|---|
| `e3m2` | 0 | **-1.095 +- 0.098** |
| `e4m3` | 2 | -0.705 +- 0.038 |
| `e5m2` | 8 | **-0.495 +- 0.032** |

Paired within seed (same trained model, so model-to-model variance cancels):

| comparison | difference | t | seeds negative |
|---|---|---|---|
| `e3m2` - `e4m3` | -0.390 +- 0.097 | **-4.00** | 9/10 |
| `e4m3` - `e5m2` | -0.210 +- 0.052 | **-4.01** | 8/10 |
| `e3m2` - `e5m2` | -0.600 +- 0.075 | **-8.01** | 10/10 |

At three seeds only the extremes separated (F23). With ten, **both adjacent pairs separate
as well** (t = 4.00 and 4.01, df = 9, p ~ 0.003 each), and the slopes are monotone in
special-code count. The middle format sits where F18 predicts. The ordering is no longer an
open question.

## F30. Use the slope test, not the endpoint ratio

The simpler statistic -- whether SDC(w48)/SDC(w8) ranks the three formats correctly --
held in only **5 of 10** seeds (P >= 5 by chance = 0.015). It is not that the effect is
absent; it is that this statistic throws away information. It uses two of the five widths
and divides one noisy number by another, whereas the slope regresses across all five and
is paired across formats within a seed. Both statistics were specified before the data
were seen, and they disagree: the well-powered one resolves the question and the crude one
does not. The write-up should report the slope test and say plainly that the endpoint
ratio is too noisy at this sample size.

## F31. The control is now very clean

The non-finite rate remains a property of the format across a 35x change in capacity:
`e4m3` varies over 0.0107-0.0128 (1.20x, r = -0.07 against log-parameters), `e5m2` over
0.0714-0.0774 (1.08x, r = -0.15), and `e3m2` is exactly zero at every width and every
seed, as having no special codes requires. With ten seeds the correlations are
indistinguishable from zero, which is what the mechanism predicts: capacity absorbs
perturbations, and nothing absorbs a NaN.

Model-to-model spread remains about **9.8x** the uncertainty contributed by 3000
injections (F24 confirmed at ten seeds).

---

# Vision transformer on CIFAR-10 (E07)

A 2.69M-parameter ViT (DeiT-Tiny width and head count, depth 6, 4x4 patches), three seeds,
100 epochs, 80.4-81.9% top-1. Every weight matrix is a `Linear`, so MX blocks run along the
model dimension; LayerNorm is left unquantised, as in deployment. Five formats, 3000
injections each, 45,000 injections total. Raw: `results/e01_vit_small*`.

## F32. The cleanest matched-accuracy comparison in the project, and F18 holds hardest here

Quantised accuracy at K=32 sits within **0.2 points of fp32 for every format except MXFP4**:
e4m3 0.8193, e2m3 0.8190, e5m2 0.8174, e3m2 0.8173, e2m1 0.8108, against fp32 0.8194. So
`e5m2` and `e3m2` differ in accuracy by **0.01 points** -- tighter than the 0.05 achieved on
ResNet8 -- and yet:

| format | bits | NaN/Inf codes | element SDC | element non-finite |
|---|---|---|---|---|
| `e5m2` | 8 | 8 | **0.0730 +- 0.0089** | 0.0689 +- 0.0042 |
| `e4m3` | 8 | 2 | 0.0453 +- 0.0296 | 0.0124 +- 0.0023 |
| `e3m2` | 6 | 0 | **0.0041 +- 0.0047** | **0.0000** |

`e5m2` is **17.8x** more vulnerable than `e3m2` at 0.01 points of accuracy difference, and
the ordering follows special-code count exactly. The mechanism is visible directly:
**94% of `e5m2`'s element failures were non-finite events** (0.0689 of 0.0730), against 27%
for `e4m3` and, necessarily, 0% for `e3m2`. F18 was derived on CNNs and predicted in advance
that a transformer would behave this way; it does, and more sharply, because the ViT is
accurate enough that ordinary perturbations almost never flip a prediction while a NaN
always does.

Neither of the structural worries about transformers materialised: LayerNorm did not mask
the damage, and attention's softmax did not absorb it.

## F33. Shared-scale dominance holds on a third architecture

| format | scale SDC | element SDC | ratio | images corrupted per scale fault |
|---|---|---|---|---|
| `e5m2` | 0.2525 | 0.0730 | 3.4x | 29.7 / 200 |
| `e4m3` | 0.4512 | 0.0453 | 12.1x | 20.1 / 200 |
| `e3m2` | 0.2955 | 0.0041 | **156x** | 17.5 / 200 |
| `e2m3` | 0.4015 | 0.0236 | 21.0x | 15.8 / 200 |
| `e2m1` | 0.4127 | 0.0500 | 8.6x | 18.2 / 200 |

Scale faults also produce non-finite outputs at 7-10% **for every format, including those
with no special element codes**, which is the expected consequence of the E8M0 scale having
its own NaN code and an MSB worth 128 exponent steps. Protecting the shared scale is now
demonstrated to pay on convolutional, re-parameterised and attention architectures alike.

## F34. Special codes are not the whole story -- an unexplained gap

`e3m2` and `e2m3` are both 6-bit, both free of special codes, and within 0.2 points of each
other in accuracy, yet their element SDC differs by **5.8x** (0.0041 vs 0.0236). The naive
explanations run the wrong way: `e3m2` has *coarser* mantissa steps (2 bits vs 3) and a
*wider* exponent reach (up to 16x per flip vs 4x), so on severity grounds it should be the
more fragile of the two, not the less. Whatever drives this is a property of how the two
exponent/mantissa splits distribute real weights across their code space, and it is not
captured by any measure used so far. It should be resolved before the write-up claims that
special codes alone explain format vulnerability.

---

# Resolving F34: why two 6-bit formats differ (E08-E10)

F34 reported that `e3m2` and `e2m3` -- both 6-bit, both without NaN/Inf codes, matched on
accuracy -- differ 5.8x in element SDC. The effect is solid: pooled over three ViT seeds,
35/8604 against 203/8604, Fisher exact p = 2.5e-30, spread uniformly across bit positions
and layers rather than concentrated anywhere.

## F35. Three hypotheses about the fault, all refuted

**It is not perturbation size.** Enumerating every (element, bit) pair exactly, `e2m3`
perturbs weights *less* than `e3m2`: mean |dw| of 0.656 against 0.717 layer-RMS units,
P(|dw| > 1 rms) of 0.233 against 0.252. Severity arguments predict the same ordering --
`e3m2` has coarser mantissa steps and a 16x exponent reach against 4x -- so on per-fault
severity `e3m2` should be the worse format, not the better one.

**It is not that the two deliver different perturbations in practice.** Recomputing the
exact |dw| for the faults actually injected and binning by size, `e2m3` fails **7.0x** more
often *within matched perturbation bins* -- 0.1045 against 0.0178 in the largest bin.
Matched size, very different damage.

**It is not which weights get hit.** In matched perturbation bins the two formats strike
weights of similar magnitude, and renormalising the perturbation per output neuron rather
than per layer changes the gap not at all (6.4x against 6.3x).

**Nor is it margin compression.** The golden margin distributions are indistinguishable:
median normalised margin 2.853 for `e3m2` against 2.863 for `e2m3`, with the same fraction
of low-margin images. Accuracy and margins match; vulnerability does not.

## F36. The fragility is a property of the quantised network, not of the encoding

Perturbing one random weight by a fixed multiple of the layer RMS -- a probe with no bit
patterns in it at all, identical for every format -- separates the networks completely:

| quantised network | mantissa bits | delta = 0.5 rms | 1.5 rms | 3.0 rms |
|---|---|---|---|---|
| `e5m2` | 2 | 0.0000 | 0.0000 | 0.0013 |
| `e3m2` | 2 | 0.0000 | 0.0000 | 0.0013 |
| `e4m3` | 3 | 0.0080 | 0.0493 | 0.1380 |
| `e2m3` | 3 | 0.0013 | 0.0187 | 0.0513 |
| `e2m1` | 1 | 0.0160 | 0.0740 | 0.1407 |

At 1.5 RMS the 2-mantissa networks never flip a prediction in 1500 trials while the
3-mantissa networks flip 2-5% of the time. Since the probe never touches a code, the
difference cannot come from how faults are encoded. **Two networks of equal accuracy and
equal margins can differ by more than an order of magnitude in how much a weight
perturbation of given size disturbs them.**

## F37. Mantissa width, not exponent width, is what tracks vulnerability

Separating perturbation-driven failures from non-finite ones across the four 6- and 8-bit
formats on the ViT:

| format | exponent bits | mantissa bits | perturbation-driven element SDC |
|---|---|---|---|
| `e5m2` | 5 | 2 | 0.0040 |
| `e3m2` | 3 | 2 | 0.0041 |
| `e4m3` | 4 | 3 | 0.0329 |
| `e2m3` | 2 | 3 | 0.0236 |

Exponent width does essentially nothing: `e5m2` and `e3m2` differ by 5 versus 3 exponent
bits and agree to within 0.0001, which also makes sense of their being near-identical as
networks (max weight difference 0.001, against 0.06 for every other pair -- the shared
scale supplies the range, so the exponent field is largely redundant). Mantissa width does
almost everything: going from 2 to 3 mantissa bits costs a factor of 6-8.

**The counterintuitive part is that the *finer* format is the fragile one.** So format
choice has two separate levers, and they are easy to confuse: NaN/Inf codes decide whether
catastrophic failures are possible at all (F18, F32), while mantissa width decides how
badly ordinary perturbations hurt. F1's headline -- accuracy does not predict reliability
-- gets a second, independent mechanism here.

**What remains open.** Why a coarser mantissa yields a network less sensitive to weight
perturbation is unexplained. Margins are ruled out, so the next measurement is the
sensitivity of the logits to weight changes, ||dz/dw||, which the margin metric cannot
see. Until that is done, F36 and F37 are a well-supported empirical result with no
mechanism, and should be written up that way.

---

# Closing the open questions (E11-E13)

## F38. The mechanism behind F36/F37: sensitivity, and specifically its tail

For image $x$ with margin $m=z_{(1)}-z_{(2)}$, moving weight $w_i$ by $\delta$ shifts the
margin by about $\delta\,\partial m/\partial w_i$, so the dimensionless sensitivity of a
weight is $s_i = |\partial m/\partial w_i|\cdot\mathrm{rms}/m$, maximised over the
evaluation images because a weight fault is exposed to every inference at once. A
perturbation of `delta * rms` should flip a prediction when $s_i\,\delta > 1$.

Computed by backpropagation for every weight, this predicts the format-agnostic probe of
F36 almost exactly -- **correlation +0.998 across 15 (format, perturbation-size) cells**:

| network | mantissa | median s | 99th pct | predicted vs measured at 1.5 rms |
|---|---|---|---|---|
| `e5m2` | 2 | 0.0122 | 0.162 | 0.0002 vs 0.0000 |
| `e3m2` | 2 | 0.0121 | 0.161 | 0.0002 vs 0.0000 |
| `e4m3` | 3 | 0.1249 | 2.309 | 0.0471 vs 0.0493 |
| `e2m3` | 3 | 0.0492 | 1.264 | 0.0154 vs 0.0187 |
| `e2m1` | 1 | 0.1187 | 3.585 | 0.0677 vs 0.0740 |

Decomposing $s$ shows where the difference is **not**: mean margin (3.59 vs 3.63), logit
spread (1.441 vs 1.444) and mean $|\partial m/\partial w|\cdot\mathrm{rms}$ (0.000697 vs
0.000685) are all equal across formats to three digits. **Only the tail differs** -- 99th
percentile 0.161 against 2.309, maximum 2.02 against 27.9. Vulnerability is set by a small
population of weights on which a large gradient coincides with a small margin, and coarse
quantisation suppresses that population. A plausible reading, not yet tested, is that with
only four mantissa values per binade the extreme individual weights are rounded away, so no
single weight stays disproportionately influential.

## F39. A severity measure that works, replacing the one in the plan

Four candidates over 39,337 element faults that stayed finite, on the ViT:

| measure | Spearman | AUC | what it knows |
|---|---|---|---|
| `log_severity` = $|\log_2|v'|-\log_2|v||$ | **-0.035** | **0.435** | number system only (the plan's) |
| `rel_error` = $|v'-v|/|v|$ | 0.107 | 0.699 | number system only |
| `delta_rms` = $|v'-v|/\mathrm{rms}$ | 0.184 | 0.845 | number system + layer scale |
| **`margin_shift`** = $s_i\,|v'-v|/\mathrm{rms}$ | **0.261** | **0.990** | number system + network |

The plan's measure has an AUC of 0.435, i.e. it ranks failures slightly *worse* than
chance. Each step towards an absolute, network-aware quantity improves matters, and
`margin_shift` -- the fraction of the decision margin a fault consumes -- reaches
AUC 0.990. Its natural threshold needs no tuning: **flagging faults with
`margin_shift > 1` catches 98.5% of all failures while flagging 4.4% of the fault space**
(precision 0.544, recall 0.985).

That is directly usable for the value-aware sampling the plan wanted: concentrating budget
on the flagged 4.4% would capture almost every failure. It also carries a caveat the plan's
formulation hides -- severity is **not** a property of the number system. The same bit
flip in the same format has very different consequences in two networks of equal accuracy
(F36), so any severity table computed from the format alone is incomplete in principle.

## F40. The MXFP4 dip at K=64 is not a systematic effect

A dip at K=64 appeared for `e2m1` on both CNNs and was left open. It does not reproduce on
the ViT: accuracy is flat across block size (0.8063, 0.8094, 0.8108, 0.8090 under OCP) and
quantisation error rises smoothly with K (0.110 to 0.118), with negligible padding. The two
CNN dips also occurred under *different* scale rules, one in OCP and one in fit. Given the
~10x model-to-model spread of F24, single-model variance is the most economical
explanation. It should not be reported as a property of MXFP4.

---

# Firming up the claims (E13-E15)

## F41. The mantissa effect, now exact rather than sampled

The exhaustive campaign was repeated for `e3m2` on the same ResNet8, giving two complete
enumerations of the same network under two formats -- 638,624 and 483,904 injections, no
sampling error in either:

| format | mantissa | special codes | model SDC | element SDC | scale SDC | element non-finite |
|---|---|---|---|---|---|---|
| `e4m3` | 3 | 2 | 0.4171 | 0.4003 | 0.9442 | 0.0127 |
| `e3m2` | 2 | 0 | **0.1079** | **0.0785** | 0.7978 | **0.0000** |

Element faults are **5.1x** less damaging in `e3m2`, exactly. The zero in the last column is
now a statement about the whole fault space rather than a sample: across all 483,904 bits,
**no** `e3m2` fault produced a non-finite value, as a format with no special codes requires.
Scale faults remain dominant under both formats (10.2x over element faults in `e3m2`), and
the sampled campaign again covered the truth (estimate 0.1070, interval
[0.0964, 0.1186], truth 0.1079) -- a second independent validation of the sampling.

## F42. RepVGG with three seeds: the ordering holds on a third architecture

RepVGG-A0 previously rested on a single trained model, which F24's variance result made
uncomfortable. Retrained three times on GPU (91.45% top-1, matching the CPU model to 0.01
points):

| format | mantissa | special codes | element SDC | element non-finite | scale/element |
|---|---|---|---|---|---|
| `e5m2` | 2 | 8 | 0.0583 +- 0.0082 | 0.0538 | 5.1x |
| `e4m3` | 3 | 2 | 0.0107 +- 0.0052 | 0.0069 | 28.7x |
| `e3m2` | 2 | 0 | **0.0012 +- 0.0005** | **0.0000** | **232x** |

Both mechanisms are visible at once. `e5m2` is worst and almost all of its failures are
non-finite (0.0538 of 0.0583, i.e. 92%), the F18 pathway. Among the rest, `e3m2` beats
`e4m3` by 8.9x with zero non-finite events, the F37 mantissa effect. The shared-scale ratio
reaches 232x in `e3m2`, simply because its element rate is so low -- the scale is then
almost the only way to break the network.

## F43. Activations barely matter on the larger models

Activation campaigns, previously run only on ResNet8, extended to both larger models
(2000 per-inference injections each):

| model | element | scale | weight/activation, element | weight/activation, scale |
|---|---|---|---|---|
| ResNet8 | 0.0032 | 0.1532 | 5.3x | 1.2x |
| RepVGG-A0 | 0.0005 | 0.0484 | 9.3x | 2.4x |
| ViT | **0.0000** | **0.0000** | unbounded | unbounded |

On the transformer **not one of 2000 activation faults changed its own inference** (95%
intervals [0, 0.0020] for element and [0, 0.0688] for scale), and RepVGG is close behind. A
transient fault in one activation value of one inference is therefore a minor concern on
these models compared with a persistent weight fault, which is exposed to every inference
and, at the scale site, corrupts $K$ values at once. The ratio column is reported as
unbounded rather than as a number, since the denominator is zero.

The likely reason is dilution plus renormalisation: one corrupted value among an
inference's activations passes through many mixing layers, and in the transformer every
block re-normalises its input. Note this does **not** contradict F32, where LayerNorm failed
to mask *weight* faults -- a weight fault perturbs every position that weight touches, on
every inference, whereas an activation fault perturbs a single value once.

---

# Correcting F37 (E14)

## F44. The sensitivity tail is not inherited from any marginal tail

For a linear layer the gradient factorises as
$\partial m/\partial W[o,i]=\sum_p a_i^{(p)} g_o^{(p)}$, so a heavy tail in the sensitivity
$s$ should trace back to a heavy tail in the activations, in the output gradients, or in the
stored weights. Measuring each separately as an outlier ratio (max / rms) over the
evaluation set:

| format | mantissa | stored weights | activations | output gradients |
|---|---|---|---|---|
| `e5m2` | 2 | 2.90 | 13.20 | 158.8 |
| `e3m2` | 2 | 2.90 | 13.20 | 158.8 |
| `e4m3` | 3 | 2.90 | 13.29 | 164.6 |
| `e2m3` | 3 | 3.10 | 13.25 | 162.4 |
| `e2m1` | 1 | 2.53 | 13.57 | 155.6 |

All three agree to within **1.01-1.03x** across formats, while the sensitivity tail itself
differs by **14x** at the 99th percentile (F38). None of the marginal distributions explains
it. What differs must be the *alignment* -- which particular (input, output) pairs line up
constructively for particular weights -- and that is a property of the specific quantised
function, not of any summary statistic of its parts. The working hypothesis in F38, that
coarse mantissas round away outlier weights, is refuted: the weight outlier ratio is
identical across formats.

## F45. It is sensitivity that predicts vulnerability, not mantissa width

F37 claimed mantissa width tracks vulnerability. Extending the comparison to all five
formats shows that claim was an artefact of the subset it was drawn from:

| format | mantissa | accuracy | median $s$ | perturbation-driven SDC |
|---|---|---|---|---|
| `e5m2` | 2 | 0.8174 | 0.0122 | 0.0040 |
| `e3m2` | 2 | 0.8173 | 0.0121 | 0.0041 |
| `e4m3` | 3 | 0.8193 | 0.1249 | 0.0329 |
| `e2m3` | 3 | 0.8190 | 0.0492 | 0.0236 |
| **`e2m1`** | **1** | 0.8108 | **0.1187** | **0.0500** |

| predictor | correlation with perturbation-driven SDC |
|---|---|
| median sensitivity $s$ | **+0.925** |
| mantissa bits | **-0.251** |

`e2m1` has the fewest mantissa bits and the highest failure rate, which breaks the
monotone reading. Within the four 6- and 8-bit formats the 2-mantissa pair really is
6-8x safer than the 3-mantissa pair -- that part stands, and is confirmed exhaustively on
ResNet8 (element rates 0.0785 against 0.4003) -- but mantissa width is a **correlate that
holds inside that group and fails outside it**, not the cause.

**So the design guidance changes.** "Prefer coarser mantissas" is not supportable; MXFP4 is
the coarsest format tested and among the most vulnerable. What is supportable is:

1. avoid element formats carrying NaN/Inf codes, which is a property of the format and
   holds everywhere tested;
2. for everything else, **measure $s$ on the quantised network** -- it takes one
   backpropagation pass per image and predicts the failure rate at $r=+0.93$ across formats
   and $+0.998$ against a direct perturbation probe -- rather than inferring robustness from
   the format's field widths.

This also sharpens F39's caveat. Severity is not a property of the number system, and
neither is robustness: two formats of identical width and accuracy differ by an order of
magnitude, and the only reliable way to know which is which is to measure the network.

---

# The value-aware campaign (E15)

E12 showed `margin_shift` ranks faults well. Ranking is not the goal; spending fewer
injections for the same answer is, which is what the project plan wanted from value-aware
sampling. Because the exhaustive campaigns recorded the outcome of *every* fault, any
sampling strategy can be replayed against known ground truth hundreds of times without a
single further inference. Strategies are compared on RMSE against the true rate at equal
budget; the variance ratio converts directly into "how many uniform injections would buy
the same precision".

The predictor is nearly free, which is the economic argument: the sensitivities need one
backpropagation pass per evaluation image, and each fault then costs a table lookup.
Measuring a fault costs a forward pass over the whole evaluation set.

## F46. Seven times fewer injections, where failures are rare

ResNet8, e3m2, 483,904 exhaustively measured faults, true rate 0.1079. Scoring the entire
fault space and splitting it at `margin_shift` of 0.1 and 1.0:

| stratum | share of fault space | true failure rate |
|---|---|---|
| low, $<0.1$ | 38.6% | **0.0000** |
| middle, $0.1$--$1$ | 47.2% | 0.0044 |
| high, $>1$ | 14.2% | **0.7439** |

The low stratum contains **no failures at all** among 186,000 faults, and the high stratum
holds almost all of them in 14% of the space. Neyman allocation over these three strata,
with 20% of the budget spent on a pilot to estimate them:

| budget | RMSE uniform | RMSE value-aware | variance ratio | equivalent uniform budget |
|---|---|---|---|---|
| 500 | 0.01354 | 0.00508 | **7.1x** | 3,552 |
| 1000 | 0.01005 | 0.00389 | **6.7x** | 6,691 |
| 3000 | 0.00533 | 0.00201 | **7.0x** | 21,007 |

Both estimators are unbiased (bias below $2\times10^{-4}$ at the largest budget), so the
gain is real variance reduction and not a shifted estimate. The predictor also transfers:
it was developed on the ViT and reaches AUC 0.9889 here, on a different architecture and
format, over the complete fault space.

## F47. It pays most where failures are rare, but it always pays

The same procedure on the same model under e4m3, where the true rate is 0.4171:

| stratum | share | true failure rate |
|---|---|---|
| low | 10.0% | **0.0000** |
| middle | 28.2% | 0.0083 |
| high | 61.9% | 0.6706 |

AUC is 0.8819 and the variance ratio 2.1--2.3x: a real but much smaller gain than
e3m2's 7x. The reason is visible in the shares. At a failure rate of 0.42 the high
stratum swallows 62% of the fault space instead of 14%, so there is far less to
concentrate on, even though the predictor still isolates a stratum of 10% that
contains no failures at all.

The trend is the useful part: the gain grows as the failure rate falls, which is
the same direction as the need, since a campaign measuring a low rate is the one
that needs many injections for a given relative precision. A campaign measuring
40% already gets a tight interval from a few thousand uniform injections.

**Correction.** The first version of this finding reported no gain at all on e4m3
(AUC 0.705, ratio 1.0-1.1x) and explained it as a structural saturation of the
first-order severity measure. That was an artefact: the e4m3 ground truth was
recorded from a different trained copy of ResNet8 than the one scoring the faults
(see F50). Matched, the null result disappears. The structural explanation was
wrong and is withdrawn -- what remains is a smaller gain, for the ordinary reason
that a common failure leaves less room for stratification.

---

# Closing the last open question (E16-E17)

F44 concluded *by elimination* that the sensitivity difference between formats must
be alignment, having ruled out the weight, activation and gradient marginals.
Elimination is only as good as the list, and the list turned out to be wrong.

## F48. It is not alignment. It is the margin of the hardest image

Alignment can be destroyed without touching either marginal: permute which
position's activation vector meets which position's gradient vector,
$\tilde G=\sum_p a^{(p)}(g^{(\pi(p))})^{\!\top}$, and every $a_i$ and every $g_o$ still
contributes exactly the same multiset of values. On the ViT's 25 Linear layers, 50
images, 4 permutations each:

| format | tail of $G$ | tail of $\tilde G$ | alignment gain |
|---|---|---|---|
| `e5m2` | 11.40 | 11.14 | 1.02x |
| `e3m2` | 11.40 | 11.17 | 1.02x |
| `e4m3` | 11.28 | 11.32 | 1.00x |
| `e2m3` | 11.34 | 11.33 | 1.00x |
| `e2m1` | 11.39 | 11.38 | 1.00x |

Destroying the alignment moves the tail by at most $2\%$, and the gain is the same
for every format (spread $1.02$x). Alignment is refuted.

The reason F44 went wrong is visible in hindsight: *every* quantity it measured was
a shape statistic, a matrix divided by its own rms, and shape is format-invariant
here (spread $1.00$x). What varies is a scale. Splitting
$s=|\partial m/\partial w|\,\mathrm{rms}/m$ into its two factors, over 200 images, the
same definition and images as E11:

| format | median $s$ | 99th pct $s$ | median $\lvert g\rvert\,\mathrm{rms}$ | 99th pct | smallest margin | SDC |
|---|---|---|---|---|---|---|
| `e5m2` | 0.0121 | 0.141 | 0.0077 | 0.073 | 0.2330 | 0.0040 |
| `e3m2` | 0.0121 | 0.140 | 0.0077 | 0.073 | 0.2271 | 0.0041 |
| `e4m3` | 0.1241 | 2.012 | 0.0076 | 0.073 | 0.0174 | 0.0329 |
| `e2m3` | 0.0489 | 1.107 | 0.0075 | 0.071 | 0.0197 | 0.0236 |
| `e2m1` | 0.1178 | 3.121 | 0.0081 | 0.077 | 0.0070 | 0.0500 |

The median $s$ column reproduces F45 to three decimals, so this is the same
quantity F45 drew its conclusion from. Its spread across formats is $10.29$x, and
$22.24$x at the 99th percentile. **With the margin divided out the spread is
$1.08$x at both.** The gradient field is the same in every format; the margin of
the closest-to-the-boundary image among the 200 varies by $33$x, and since E11
defines $s$ as a maximum over images, that one image raises $s$ for every weight
in the network at once.

A second mechanism does hold, for a different question. Among the per-position
terms $x_p=a_i^{(p)}g_o^{(p)}$ that build one weight's gradient, the most sensitive
weights have coherence $|\sum_p x_p|/\sum_p|x_p| = 0.843$, against $0.466$ for
randomly chosen weights and $0.124$ for terms independent in sign. So *within* a
network, a weight is sensitive because its positions agree rather than because any
one of them is large -- but this too is identical across formats (0.843-0.845).

## F49. That margin belongs to the evaluation set, not to the network

If near-ties are the mechanism, a statistic of the margin alone should predict
vulnerability -- no gradients, no injections, one forward pass. Tested against
fifteen networks (3 seeds x 5 formats) whose failure rates came from campaigns
already run:

| margins measured on | $1/\text{min margin}$ | mean $1/m$ | $m$ at the 0.1st pct |
|---|---|---|---|
| the campaigns' own 200 images | **+0.983** | +0.971 | -0.858 |
| 2000 fresh images | +0.209 | +0.295 | -0.331 |

(Spearman against the measured failure rate.) The predictor is nearly perfect on
the images the campaign scored and worthless on fresh ones. Separating the two
sources of variation explains why:

| varying | on fresh images | on the campaign's images |
|---|---|---|
| seeds, within a format | **+1.000** in all 5 formats | +1.000 in 4 of 5 |
| formats, within a seed | -0.100, +0.600, -0.500 | +1.000, +1.000, +0.600 |

The seed-to-seed component of vulnerability is a real property of the trained
network and generalises to data it has never seen. The format-to-format component,
on this model at 200 evaluation images, is a property of the **(network,
evaluation set) pair**. Quantising to a different element format nudges the
logits slightly; whether that nudge parks one of the 200 images on a decision
boundary is luck of the sample, and when it does the campaign's rate rises --
truthfully for those images, but not durably.

This does not touch F46-F47: value-aware sampling estimates *a given campaign's*
rate against that campaign's own ground truth, and a $7$x variance reduction is a
$7$x variance reduction whatever the rate means. What it qualifies is the
interpretation -- F45's "sensitivity predicts vulnerability" holds within an
evaluation set and should not be read as a property of the format, and any
cross-format ranking measured on one small image set inherits the same doubt.

---

# What the campaigns were actually measured on (E18)

## F50. Two exhaustive campaigns, two different networks

Training is not bit-reproducible across machines, and the same model name refers to
different weights on each. Of the 24 checkpoints that exist both on this laptop and
on the cluster, **none** has identical weights -- accuracies agree to a few tenths
of a percent, so nothing looks wrong.

Whether a recorded campaign belongs to the checkpoint in hand is testable rather
than assumable: replay a sample of its faults and compare outcomes. Over 300
recorded faults each:

| exhaustive campaign | agrees with checkpoint A | agrees with checkpoint B |
|---|---|---|
| e3m2, 483,904 faults | 253/300 | **300/300** |
| e4m3, 638,624 faults | **300/300** | 202/300 |

The two campaigns were recorded from **two different trained networks**. So the
comparison F45 quoted as the exhaustive confirmation of the mantissa effect --
element rates $0.0785$ for e3m2 against $0.4003$ for e4m3, a factor of five -- puts
one network's e3m2 next to another network's e4m3. Given that model-to-model spread
was already measured at about ten times the injection uncertainty (F31), a
five-fold gap between two different networks carries no information about the
format.

Two guards now exist: `tools/verify_ground_truth.py` performs the replay test, and
E15 runs it automatically and refuses to score a campaign that its checkpoint did
not produce. The first version of F47 was wrong for exactly this reason.

## F51. The mantissa effect is real -- the metric used to confirm it was not

Redone properly: one fault list of 1500 element faults per format, replayed against nine
class-balanced image sets of 200 images (the set every earlier campaign used, plus eight
disjoint ones), on each of the two networks separately. The design is paired within a
format, so fold-to-fold variation carries no fault-sampling noise. On network A's original
200 images the e4m3 rate comes out 0.4087 against its exhaustive 0.4003, confirming the
setup reproduces the reference it should.

Under the SDC definition used throughout this study -- a fault counts as failing if **any**
of the 200 inferences changes -- the two formats are indistinguishable:

| network | e3m2 | e4m3 | ratio | per-fold ratio |
|---|---|---|---|---|
| A | 0.2152 | 0.2214 | **1.03x** | 0.34x -- 2.58x |
| B | 0.1623 | 0.1504 | **0.93x** | 0.47x -- 2.14x |

The same faults on the same folds, scored by the **per-inference** rate -- what fraction of
the 200 inferences each fault actually corrupts -- say the opposite:

| network | e3m2 | e4m3 | ratio | e4m3 spread across folds |
|---|---|---|---|---|
| A | 0.00202 | 0.01124 | **5.57x** | 1.27x |
| B | 0.00129 | 0.00987 | **7.64x** | 1.15x |

Two independently trained networks agree on a factor of $5.6$--$7.6$, and the per-inference
rate barely moves across image sets where the any-image rate moved sixfold. So F45's
mantissa effect is real and roughly the size it claimed; what was wrong was the metric used
to confirm it, and the fact that the confirmation compared two different networks (F50).

## F52. The any-image SDC metric is the confound

`sdc` asks whether a fault corrupts *at least one* of $n$ inferences. That quantity grows
with $n$ and saturates towards 1, so it is not a property of the fault at all beyond a
point -- it is mostly a question of whether the image set contains something near a decision
boundary. The consequences, on the folds above:

| network | format | any-image spread | mean CI width | spread / CI | per-inference spread |
|---|---|---|---|---|---|
| A | e3m2 | 0.1233 -- 0.3593 | 0.0405 | **5.8x** | 2.38x |
| A | e4m3 | 0.1227 -- 0.4087 | 0.0406 | **7.0x** | **1.27x** |
| B | e3m2 | 0.0467 -- 0.4207 | 0.0340 | **11.0x** | 7.52x |
| B | e4m3 | 0.0713 -- 0.2080 | 0.0355 | **3.8x** | **1.15x** |

And the fold's own closest-to-the-boundary image predicts its any-image rate: Spearman
$+0.80$ and $+0.83$ on network A, $+0.92$ and $+0.93$ on network B -- F49's mechanism seen
at the level of whole image sets rather than networks. The Wilson interval is not wrong; it
answers a narrower question than it appears to, covering the uncertainty from sampling
*faults* while the choice of evaluation images is a second and larger source that no
injection count reduces.

This also explains F49 rather than contradicting it. The sensitivity $s$ carries $1/\min m$,
which is a near-tie detector; the any-image metric is a near-tie amplifier; so $s$ predicts
that metric well and predicts nothing about fresh data. Under the per-inference rate the
whole chain is better behaved.

This is the same confound F26 found in the width sweep, where near-tie count explained the
e4m3 w=32 anomaly at Spearman $0.89$ over 39 cells. F26 diagnosed it for one anomaly; the
folds above show it governs cross-format comparison in general, and the per-inference rate
measures the effect F26's tie-robust metrics were reaching for.

**What to report.** The per-inference rate, as the primary number. The any-image rate is
meaningful only for a stated input distribution and a stated $n$, and two campaigns should
never be compared under it unless both used the same images. Every rate quoted elsewhere in
these findings is an any-image rate at $n=200$, and the comparisons among them inherit this
caveat -- the ones that survive it are those resting on the non-finite pathway, which does
not depend on the evaluation set.

**Scope.** F46-F47's variance reductions estimate the any-image rate, since that is what the
exhaustive campaigns recorded. The machinery carries over unchanged to the per-inference
rate, but the numbers would have to be re-derived.

---

# Every campaign rescored per inference (E19)

F52 established that the any-image SDC rate saturates in the number of evaluation
images and mostly reports whether that set holds a near-tie. Applying the lesson
needed no new injections: `run_campaign` has always recorded `changed` and
`change_rate` per fault, so all 119 campaigns -- 1,441,528 recorded faults across
8 models -- already carried their own per-inference rate, and only the summaries
drew on the any-image column. Intervals differ by necessity: Wilson for the
any-image proportion, and the standard error of a mean of per-fault proportions for
the per-inference rate.

## F55. The format ordering holds on every model per inference, and on three of eight otherwise

Element faults, $K=32$, OCP rule, averaged over the available seeds:

| model | seeds | e3m2 | e4m3 | e5m2 | any-image order | per-inference order |
|---|---|---|---|---|---|---|
| RepVGG-A0 | 3 | 0.00001 | 0.00688 | 0.05381 | holds | holds |
| ResNet8 | 1 | 0.00332 | 0.01699 | 0.07801 | **reversed** | holds |
| ResNet8-w8 | 5 | 0.01003 | 0.02005 | 0.08143 | **reversed** | holds |
| ResNet8-w16 | 5 | 0.00180 | 0.01374 | 0.07831 | holds | holds |
| ResNet8-w24 | 4 | 0.00121 | 0.01210 | 0.07585 | **reversed** | holds |
| ResNet8-w32 | 4 | 0.00060 | 0.01354 | 0.07934 | **reversed** | holds |
| ResNet8-w48 | 4 | 0.00035 | 0.01302 | 0.07812 | **reversed** | holds |
| ViT | 3 | 0.00003 | 0.01258 | 0.06897 | holds | holds |

$\text{e3m2} < \text{e4m3} < \text{e5m2}$ holds in **8 of 8** models per inference and
**3 of 8** under the any-image rate. The 95% intervals are disjoint between all three
formats on every model; the narrowest separation is ResNet8-w8, where e3m2 is
$0.01003\,[0.00925,0.01081]$ against e4m3 $0.02005\,[0.01584,0.02425]$.

F17 -- "the format ranking's direction reverses across architectures, never quote it
from one model" -- was therefore a metric artefact. The ranking is stable across
every architecture, width and seed measured here. The warning it issued was sound
advice for the wrong reason: what varied was not the architecture but which
evaluation images sat near a boundary.

## F56. Shared-scale dominance survives, and grows where the element rate is low

| model | format | scale/element, any-image | scale/element, per inference |
|---|---|---|---|
| RepVGG-A0 | e3m2 | 200.7x | **8424x** |
| ViT | e3m2 | 72.6x | **3212x** |
| ResNet8-w48 | e3m2 | 7.2x | 434x |
| ResNet8-w16 | e4m3 | 4.6x | 13.3x |
| ResNet8-w16 | e5m2 | 3.3x | 2.6x |

The direction never reverses, on any model or format, under either metric. The ratio
grows enormously for e3m2 because its element rate per inference is near zero -- the
RepVGG figure divides by $0.00001\,[0.00000,0.00003]$ and should be read as a lower
bound of order $10^3$, not a point estimate. For e5m2 the ratio shrinks below the
any-image figure, because that format's element faults already corrupt everything
through the NaN pathway. "Protect the shared scale first" is the most robust design
conclusion in the study.

## F57. The block-size story is weaker than F15 claimed

F15 held that OCP clipping *fabricates* a block-size trend absent under a
non-clipping rule. Element faults, e4m3:

| model | rule | K8 | K16 | K32 | K64 | spread |
|---|---|---|---|---|---|---|
| RepVGG-A0 | ocp | 0.01535 | 0.01000 | 0.00267 | 0.00334 | 5.76x |
| RepVGG-A0 | fit | 0.01272 | 0.00872 | 0.00273 | 0.00341 | 4.66x |
| ResNet8 | ocp | 0.02408 | 0.02082 | 0.01592 | 0.01360 | 1.77x |
| ResNet8 | fit | 0.01931 | 0.01541 | 0.01411 | 0.01348 | 1.43x |

Under the any-image rate the two rules differed by $4.9\times$ in spread on RepVGG
($7.25$ against $1.48$); per inference they differ by $1.24\times$ ($5.76$ against
$4.66$), and on ResNet8 the non-clipping rule shows the *larger* spread. So the
per-inference data carry a genuine, monotone block-size effect -- larger blocks
corrupt fewer inferences -- present under **both** scale rules, and the sharp
OCP-versus-fit contrast that F15 rested on is specific to the any-image metric.

What survives of F15: the OCP rule does distort the any-image trend, and a
non-clipping control is still worth running. What does not: the conclusion that the
apparent block-size effect is entirely an artefact. Its mechanism is not established
here and should not be asserted.

## F58. The capacity dose-response sharpens into a clean match with special-code count

Ratio of the widest model to the narrowest, ResNet8 w8 through w48 (a $35\times$
parameter range); below $1$ means capacity helps:

| format | special codes | any-image | per inference |
|---|---|---|---|
| e3m2 | 0 | 0.11x | **0.04x** (25x better) |
| e4m3 | 2 | 0.13x | **0.65x** |
| e5m2 | 8 | 0.27x | **0.96x** (no benefit) |

Per inference the mechanism is almost exact: the format with no special codes gains
$25\times$ from a $35\times$ increase in capacity, the format with eight gains
nothing measurable. F29's dose-response is confirmed and considerably sharper than
the any-image version. Consistently, $96.9\%$ of e5m2's per-inference element
damage is non-finite -- capacity absorbs perturbations and cannot absorb a NaN.

## F59. F45's sensitivity result is metric-invariant

The one headline that needed no revision. Perturbation-driven element faults on the
ViT (non-finite events excluded), three seeds:

| predictor | vs any-image rate | vs per-inference rate |
|---|---|---|
| median sensitivity $s$ | **+0.928** | **+0.928** |
| mantissa bits | -0.248 | -0.225 |

Both correlations are unchanged to three decimals, and e2m1 keeps the highest
perturbation-driven rate despite having the fewest mantissa bits. Separating the two
mechanisms is what makes this robust: once the NaN pathway is excluded, what is left
is perturbation damage, and that is governed by sensitivity under either metric.
