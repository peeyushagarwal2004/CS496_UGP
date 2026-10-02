# mxfi — Reliability-Aware Fault Injection for Microscaling-Based DNNs

Fault-injection study of DNN reliability when weights and activations are stored in
**microscaling (MX / MXFP) block floating-point**, where a block of *K* elements shares
one 8-bit E8M0 scale:

```
v[i] = decode(code[i]) * 2**(X - 127)
```

This creates two structurally different fault sites — an **element** flip perturbs one
value, a **shared-scale** flip rescales all *K*. Characterising that asymmetry is the
point of the project. The plan it follows is
[`papers/Reliability-Aware-Microscaling-FI.pdf`](papers/Reliability-Aware-Microscaling-FI.pdf).

**Results:** the report is [`docs/report.pdf`](docs/report.pdf) (5 pages). Every measured
finding, with its numbers, caveats and later corrections, is in
[`docs/findings.md`](docs/findings.md).

## Status

Complete for CIFAR-10. All five objectives of the plan (O1–O5) are answered, and both
of its open design choices are settled empirically (E21, E22). The one part of the plan
not carried out is DeiT on ImageNet: ImageNet is not available on any machine this
project can reach.

| covered | detail |
|---|---|
| models | ResNet8 (78k, 87.0%), RepVGG-A0 (7.0M, 91.4%, 3 seeds), a DeiT-Tiny-width ViT on CIFAR-10 (2.7M, 81.9%, 3 seeds), ResNet8 at 5 widths × 10 seeds |
| formats | e4m3, e5m2 (MXFP8), e3m2, e2m3 (MXFP6), e2m1 (MXFP4); E8M0 scale |
| block sizes | 8, 16, 32, 64; OCP scale rule and a non-clipping control |
| tensors | weights and activations |
| fault sites | element vs scale, every bit position, every layer; single- and double-bit; a NaN read guard |
| ground truth | two exhaustive campaigns on ResNet8-w16 (483,904 and 638,624 faults) |
| scale | 119 sampled campaigns, 1.44M recorded faults, all rescored per inference |

209 tests, all passing.

## Headline results

Quote **per-inference** rates — the fraction of evaluated inferences a fault corrupts.
The conventional "any image changed" SDC rate measures whether the evaluation set
happens to contain a near-tie image, and it hid the format effect entirely (F52).

1. **Accuracy does not predict reliability.** Formats within 0.01 pp of each other in
   accuracy differ by orders of magnitude in the inferences their faults corrupt. The
   ordering e3m2 < e4m3 < e5m2 holds on 8 of 8 models (F55).
2. **NaN/Inf codes are the dominant mechanism.** Only formats that own special codes
   produce non-finite outputs, and model capacity absorbs perturbations but never a NaN:
   over a 35× capacity range the format with no special codes improves 25×, the one with
   eight not at all (F58). Prefer element formats without NaN/Inf codes — or guard them:
   decoding special element codes as zero on read cuts element damage 4.4× (e4m3) and
   36× (e5m2) on ResNet8 (F68).
3. **Protect the shared scale first.** Scale faults outweigh element faults on every
   model and format, per inference by 13× to ~1000× (F56).
4. **Weights before activations**: 5–9× more damaging per inference (F43).
5. **Sensitivity, not mantissa width**, predicts perturbation damage (ρ = +0.93, F59);
   its cross-format spread is the margin of the hardest evaluation image (F48–F49).
6. **The plan's severity metric fails** (AUC 0.435, scores the sign bit — the worst bit
   for perturbation damage — as harmless; F39, F67);
   `margin_shift` reaches AUC 0.990 (F39). Quantile-stratifying it cuts the injections
   needed for a given precision **57–120×** (F61).
7. **Single-bit injection is sufficient** (F62–F64): a double flip in one word does only
   1.04–1.55× the damage of a single flip, and within-word doubles cannot matter at any
   bit-error rate where single-fault injection is itself valid.
8. **Enumerate, don't learn** (F65–F66): exact per-code enumeration of the decoded change
   gives 48–65× variance reduction with no injections; TreeFI-style intervals learned
   from a pilot do no better than uniform sampling.

## Layout

| path | contents |
|---|---|
| `mxfi/` | the library: formats, codec, fault model, sampling, statistics, PyTorch wrapper, models, training, campaign runner |
| `experiments/` | one script per experiment, `e00`–`e23`; each docstring states the question and the run command |
| `tools/` | GPU patches, fast-injection verifier, and `verify_ground_truth` (checks a recorded campaign belongs to a checkpoint) |
| `results/` | every campaign CSV and summary |
| `docs/` | report, findings, figures |
| `*.sh` | the long-running sweep drivers (laptop and cluster) |

The core (`formats`, `codec`, `faults`, `sampling`, `stats`) is **pure numpy**; only
`torch_mx`, `models`, `vit`, `data`, `train` and `campaign` need torch.

## Setup

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

The venv is Python 3.13 with torch 2.14.0+cpu. GPU runs used an A100 cluster
(Python 3.8, torch 2.1.2); apply `tools/device_support.py` and `tools/fast_inject.py`
to a fresh checkout before running there.

**Checkpoint provenance.** Training is not bit-reproducible across machines, so the same
checkpoint name can hold different weights on two machines while agreeing on accuracy.
Before using a recorded campaign as ground truth, check it against the checkpoint:

```bash
PYTHONPATH=. python -m tools.verify_ground_truth results/<campaign>.csv --fmt e3m2 --train-seed 0
```

## Why BatchNorm is folded

Campaigns run on the BN-folded model. An unfolded BatchNorm re-normalises whatever the
corrupted convolution produced, partially masking the fault and understating damage.
Deployed MX accelerators fold BN, so an unfolded campaign would describe a network nobody
ships. `fold_batchnorm` is exact to float32 roundoff (1.9e-8 on ResNet8).

## Supported formats

| format | bits | exp/man | max normal | emax | special codes |
|---|---|---|---|---|---|
| `e4m3` (MXFP8) | 8 | 4/3 | 448 | 8 | NaN (2) |
| `e5m2` (MXFP8) | 8 | 5/2 | 57344 | 15 | Inf, NaN (8) |
| `e3m2` (MXFP6) | 6 | 3/2 | 28 | 4 | none |
| `e2m3` (MXFP6) | 6 | 2/3 | 7.5 | 2 | none |
| `e2m1` (MXFP4) | 4 | 2/1 | 6 | 2 | none |

Shared scale is always E8M0 (bias 127, code 255 reserved for NaN).

## Quick start

```python
import numpy as np
from mxfi import quantize, Fault, inject, fault_space

x = np.random.randn(4, 128).astype(np.float32)
q = quantize(x, "e4m3", block_size=32)        # -> scales uint8[4,4], codes uint8[4,4,32]

inject(q, Fault("element", (0, 0, 0), 7)).decode()   # 1 value changes
inject(q, Fault("scale",   (0, 0),    7)).decode()   # 32 values change
```

On a network, `MXModel.fault(site)` applies one single-bit fault for the duration of a
`with` block, and `MXModel.word_fault(tensor, site, index, mask)` applies any multi-bit
XOR mask to one stored word.

## Scope

`papers/MX-TreeFI.pdf` is **reference material only** — it frames the problem and records
how TreeFI (ICCAD 2026) approaches FP32 statistical FI. Value-aware stratified sampling
appears here strictly as a *supporting tool* to keep the campaign affordable; it is not
the contribution.
