# Bit-flip reliability of microscaling (MX) neural networks

CS496 undergraduate project, IIT Kanpur. Peeyush Agarwal.

**Report:** [docs/report.pdf](docs/report.pdf) (5 pages)
**Full experiment log:** [docs/findings.md](docs/findings.md)

## The question

Microscaling formats (MXFP8, MXFP6, MXFP4) store a tensor in blocks of 32 values. Each
value gets a small 4 to 8 bit element code, and the whole block shares one 8-bit exponent
called the scale:

```
v[i] = decode(code[i]) * 2^(X - 127)
```

So there are two very different places a bit flip can land. A flip in an element code
changes one number. A flip in the shared scale changes all 32 numbers in the block at once.
The project measures how much this matters in practice: which formats, bits, layers and
tensors are most vulnerable when DNN weights and activations are stored in MX, and why.

## What I built

`mxfi` is a small fault-injection library. It encodes weights and activations into real MX
bit patterns, flips bits in that encoding (in an element or in a scale), decodes, and checks
how many predictions change against the fault-free model.

Models: ResNet8, RepVGG-A0 and a small vision transformer (DeiT-Tiny width), trained on
CIFAR-10 and CIFAR-100. Formats: e4m3, e5m2, e3m2, e2m3, e2m1. Block sizes 8 to 64. Most
campaigns sample 3000 faults. Two exhaustive campaigns flip every single stored bit of a
ResNet8 (about 1.1 million faults), which gives exact failure rates to check the sampling
against.

## Main results

All rates are the fraction of inferences a single fault corrupts. The common "did any test
image change" SDC rate turned out to mostly measure whether the test set happens to contain
a borderline image, so I don't use it for comparisons (the report explains this).

- Formats with the same accuracy can differ in fault vulnerability by orders of magnitude.
  The order e3m2 < e4m3 < e5m2 held on every model I tested, on both datasets.
- Most of the damage comes from bit flips that turn a value into NaN or Inf. The 6-bit and
  4-bit formats have no such codes and avoid the problem. For formats that do have them,
  decoding those codes as zero when reading cuts the damage by at least 29x on RepVGG and the
  transformer.
- The shared scale is the weakest point: a scale fault does 13x to about 1000x the damage of
  an element fault.
- Per fault, activations are about as vulnerable as weights.
- The severity score proposed in the original project plan does not predict failures (it
  gives sign flips a score of zero). A gradient-based score does, and sampling with it needs
  57 to 120x fewer injections for the same precision.
- Single-bit fault injection is enough. Two flips in the same word do only slightly more
  damage than one. Enumerating the code table exactly works much better than learning value
  intervals from a pilot run, the way TreeFI does for FP32.
- On CIFAR-100 everything above still holds, but the gap between formats shrinks, because a
  harder task makes every format more sensitive to small perturbations.

ImageNet was not available on any machine I had access to, so CIFAR-100 stands in for the
DeiT-on-ImageNet experiment in the original plan.

## Repository layout

| path | contents |
|---|---|
| `mxfi/` | the library: formats, encode/decode, fault model, sampling, statistics, PyTorch wrapper, models, training, campaign runner |
| `experiments/` | one script per experiment (`e00` to `e24`); each file's docstring says what it tests and how to run it |
| `tests/` | unit tests (`pytest`) |
| `tools/` | GPU patches and two checks that a recorded campaign belongs to a given checkpoint |
| `results/` | raw campaign outputs from my laptop |
| `checkpoints/` | trained networks from my laptop, with per-epoch training logs |
| `cluster/` | everything from the GPU server: networks, results, logs, run scripts (see `cluster/README.md`) |
| `data/` | CIFAR-10 and CIFAR-100 |
| `docs/` | report, findings, figures |
| `papers/` | the original project proposals |

`cluster/` is kept separate on purpose. Training isn't bit-reproducible across machines, so
some networks on the laptop and on the server have the same file name but different weights.
Mixing them would pair a campaign with the wrong network.

## Running it

Clone with [Git LFS](https://git-lfs.com) installed, otherwise the three largest dataset files
come down as small pointer files.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pytest tests -q
```

(On Linux or macOS use `.venv/bin/python`.) Then, for example:

```bash
python -m mxfi.train --model resnet8 --epochs 60
python -m experiments.e01_uniform_baseline --model resnet8 --fmt e4m3
```

Set `PYTHONPATH=.` when running from the repository root. A model name ending in `_c100`
(e.g. `vit_small_c100`) trains and evaluates on CIFAR-100. The heavier runs used an A100
with `--device cuda`.

A minimal example of the library on its own:

```python
import numpy as np
from mxfi import quantize, Fault, inject

x = np.random.randn(4, 128).astype(np.float32)
q = quantize(x, "e4m3", block_size=32)

inject(q, Fault("element", (0, 0, 0), 7)).decode()   # changes 1 value
inject(q, Fault("scale",   (0, 0),    7)).decode()   # changes 32 values
```

## Formats

| format | bits | exponent/mantissa | largest value | NaN/Inf codes |
|---|---|---|---|---|
| e4m3 (MXFP8) | 8 | 4/3 | 448 | NaN |
| e5m2 (MXFP8) | 8 | 5/2 | 57344 | NaN, Inf |
| e3m2 (MXFP6) | 6 | 3/2 | 28 | none |
| e2m3 (MXFP6) | 6 | 2/3 | 7.5 | none |
| e2m1 (MXFP4) | 4 | 2/1 | 6 | none |

The shared scale is always E8M0 (an 8-bit power of two, with code 255 reserved for NaN).
