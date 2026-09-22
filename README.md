# mxfi — Reliability-Aware Fault Injection for Microscaling-Based DNNs

Fault-injection study of DNN reliability when weights and activations are stored in
**microscaling (MX / MXFP) block floating-point**, where a block of *K* elements shares
one 8-bit E8M0 scale:

```
v[i] = decode(code[i]) * 2**(X - 127)
```

This creates two structurally different fault sites — a **element** flip perturbs one
value, a **shared-scale** flip rescales all *K*. Characterising that asymmetry is the
point of the project. Working drafts are in [`papers/`](papers/).

## Status

| component | state |
|---|---|
| `mxfi/formats.py` — MX element + E8M0 alphabets, exact per-code tables | done |
| `mxfi/codec.py` — bit-addressable encode/decode, two scale rules | done |
| `mxfi/faults.py` — two-site injection, exhaustive impact enumeration | done |
| `mxfi/sampling.py` — uniform + stratified fault sampling, Neyman allocation | done |
| `mxfi/stats.py` — Wilson / stratified estimators, injection-reduction metric | done |
| `mxfi/torch_mx.py` — MX-quantised `Linear`/`Conv2d`, injectable fault sites | done |
| `mxfi/models.py` — ResNet8 (78,042 params) + BN folding | done |
| `mxfi/data.py` — CIFAR-10 loaders + fixed campaign subset | done |
| `mxfi/train.py` — resumable CPU training | done |
| `mxfi/campaign.py` — golden-vs-faulty runner, severity-annotated results | done |
| trained ResNet8 checkpoint — **87.01%** top-1 | done |
| RepVGG-A0 | next |
| E00 quantised accuracy sweep, E01 uniform FI campaign | done |
| exhaustive ResNet8 ground truth | feasible — see below |

194 tests, all passing. The core is **pure numpy** — only `mxfi.torch_mx`,
`mxfi.models`, `mxfi.data`, `mxfi.train` and `mxfi.campaign` need torch.

## Measured throughput (this machine, 4 CPU threads)

| operation | cost |
|---|---|
| ResNet8 training | 82 s/epoch → ~90 min for 60 epochs |
| campaign injection @100 images | 47 ms |
| campaign injection @200 images | 89 ms |

**Exhaustive ground truth is affordable.** The full MXFP8 weight bit space of ResNet8 is
618,880 element bits + 19,744 scale bits = **638,624 faults** — about **16 h at 200
images**, or 8 h at 100. MXFP4 halves it to 329,184 faults (~8 h). Both are overnight
runs, so the statistical estimates can be validated against true exhaustive numbers
rather than against a larger sample.

## Why BatchNorm is folded

Campaigns run on the BN-folded model. An unfolded BatchNorm re-normalises whatever the
corrupted convolution produced, partially masking the fault and understating damage.
Deployed MX accelerators fold BN, so an unfolded campaign would describe a network nobody
ships. `fold_batchnorm` is exact to float32 roundoff (1.9e-8 on ResNet8).

## Setup

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

The venv is Python 3.13 with torch 2.14.0+cpu (the default `python` on this machine is
3.14, which has no torch). **No CUDA GPU here** — CIFAR-scale models run locally;
DeiT/ImageNet cells will need Colab, Kaggle, or a cluster.

## Supported formats

| format | bits | exp/man | max normal | emax | headroom |
|---|---|---|---|---|---|
| `e4m3` (MXFP8) | 8 | 4/3 | 448 | 8 | 87.5% |
| `e5m2` (MXFP8) | 8 | 5/2 | 57344 | 15 | 87.5% |
| `e3m2` (MXFP6) | 6 | 3/2 | 28 | 4 | 87.5% |
| `e2m3` (MXFP6) | 6 | 2/3 | 7.5 | 2 | 93.75% |
| `e2m1` (MXFP4) | 4 | 2/1 | 6 | 2 | 75% |

Shared scale is always E8M0 (bias 127, code 255 reserved for NaN).

## Findings

**Measured results are in [`docs/findings.md`](docs/findings.md)** — the core hypothesis
confirmed (`e5m2` vs `e3m2`: 0.05 pp accuracy apart, 1.77x apart in fault vulnerability),
the severity metric shown not to predict failure, and shared-scale faults measured at
2-3x the element SDC rate with a 40x blast radius.

### Representation-level findings

**1. The drafts' severity metric is blind to the sign bit.** `d = |log v' − log v|` scores
a sign flip as *zero* damage while the relative error is 2.0. Any value-aware risk score
built on it would rank the sign bit as the safest in the word. `element_impact_table`
reports `log_severity`, `rel_error` and an explicit `sign_flip` mask; the discrepancy is
pinned by `test_sign_bit_is_invisible_to_the_log_severity_metric`. **The severity
definition in §3 of both drafts needs fixing before any campaign runs.**

**2. Matched-accuracy format pairs exist, and they differ structurally.** `e5m2` (8-bit)
and `e3m2` (6-bit) give near-identical SQNR across Gaussian, heavy-tailed and outlier
data — both have 2 mantissa bits, and block scaling makes the extra exponent range nearly
useless. Same accuracy, different bit width and exponent/mantissa split, so necessarily
different per-bit vulnerability. That is the cleanest test of the core hypothesis.

**3. The OCP scale rule clips block maxima, unequally across formats.** The rule aligns
the block max into `[2**emax, 2**(emax+1))`, but `max_normal` sits below the top of that
binade, so the largest element saturates — up to 12.5% for e4m3/e5m2/e3m2, 6.25% for
e2m3, **25% for e2m1**. A naive MXFP4-vs-MXFP8 comparison would partly measure clipping,
not fault behaviour. Use `scale_mode="fit"` (provably non-clipping) as the control.

**4. Catastrophic bits are identified exactly.** Scale **bit 7** shifts the exponent by
128 — past the entire float32 range (`inf` logits on a real network). **8 of the 256**
scale byte values flip into the reserved NaN code, NaN-ing a whole block.

## Quick start

```python
import numpy as np
from mxfi import quantize, Fault, inject, fault_space

x = np.random.randn(4, 128).astype(np.float32)
q = quantize(x, "e4m3", block_size=32)        # -> scales uint8[4,4], codes uint8[4,4,32]

inject(q, Fault("element", (0, 0, 0), 7)).decode()   # 1 value changes
inject(q, Fault("scale",   (0, 0),    7)).decode()   # 32 values change
```

## Scope (settled)

**The project is `papers/Reliability-Aware-Microscaling-FI.pdf`** — a reliability-analysis
methodology characterising how element and shared-scale faults propagate through
MX-quantised DNNs, and which representation characteristics drive vulnerability.

`papers/MX-TreeFI.pdf` is **reference material only** — it frames the problem and records
how TreeFI (ICCAD 2026) approaches FP32 statistical FI. TreeFI is prior work, not the
method to reproduce. Value-aware stratified sampling appears here strictly as a
*supporting tool* to keep the multi-axis campaign affordable; it is not a contribution.

Remaining open questions (both empirical, both scoped as supporting-tool choices):

- Learned value intervals vs exact per-code enumeration — the exact tables are already
  built and cheap, which weakens the case for learned intervals.
- Single-bit vs multi-bit (H-model) fault occurrence.
