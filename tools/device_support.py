"""Add GPU support to the pipeline, idempotently.

The MX codec itself stays on the CPU in NumPy -- that is deliberate, since it
is exact and cheap (one small host->device copy per injection).  What moves to
the GPU is the expensive part: the forward passes.  Only four places bound the
pipeline to the CPU:

1. ``Evaluator`` kept its fixed batches on the host and never moved them.
2. ``Evaluator.predict`` called ``.numpy()`` on device tensors.
3. The activation hook rebuilt a CPU tensor from NumPy and handed it to a
   model that may live on the GPU.
4. ``train`` never moved the model or batches.

Numerical fidelity matters more here than speed: results measured on GPU must
be comparable with the CPU results already collected, so TF32 is disabled and
cuDNN is put in deterministic mode whenever a CUDA device is selected.

Run once per checkout::

    python tools/device_support.py            # reports what it changed
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(rel: str, pairs: list[tuple[str, str]]) -> str:
    path = ROOT / rel
    src = path.read_text(encoding="utf-8")
    done, skipped = 0, 0
    for old, new in pairs:
        if new in src:                      # already applied
            skipped += 1
            continue
        if src.count(old) != 1:
            raise SystemExit(f"{rel}: expected exactly one of {old[:60]!r}, "
                             f"found {src.count(old)}")
        src = src.replace(old, new)
        done += 1
    path.write_text(src, encoding="utf-8")
    return f"{rel}: {done} applied, {skipped} already present"


def main() -> None:
    out = []

    # ---- 1/2. Evaluator holds its batches on the target device
    out.append(patch("mxfi/campaign.py", [
        ('    def __init__(self, dataset: Dataset, batch_size: int = 256):\n'
         '        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)\n'
         '        self.batches = [(x, y) for x, y in loader]',
         '    def __init__(self, dataset: Dataset, batch_size: int = 256,\n'
         '                 device: str = "cpu"):\n'
         '        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)\n'
         '        self.device = device\n'
         '        # moved once, so every injection replays identical device tensors\n'
         '        self.batches = [(x.to(device), y) for x, y in loader]'),
        ('        return torch.cat(preds).numpy(), finite',
         '        return torch.cat(preds).cpu().numpy(), finite'),
    ]))

    # ---- 3. activation hook must return a tensor on the model's device
    out.append(patch("mxfi/torch_mx.py", [
        ('            out = torch.from_numpy(mx.decode()).to(x.dtype)',
         '            out = torch.from_numpy(mx.decode()).to(device=x.device,\n'
         '                                                   dtype=x.dtype)'),
    ]))

    # ---- 4. training on the selected device, with a determinism helper
    out.append(patch("mxfi/train.py", [
        ('CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"\n',
         'CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"\n\n\n'
         'def setup_device(device: str = "cpu") -> str:\n'
         '    """Select a device and make CUDA as numerically faithful as the CPU.\n\n'
         '    TF32 would silently drop matmul/conv precision to ~10 mantissa bits,\n'
         '    which can flip borderline predictions and make GPU-measured failure\n'
         '    rates incomparable with the CPU ones already collected.\n'
         '    """\n'
         '    if device.startswith("cuda"):\n'
         '        if not torch.cuda.is_available():\n'
         '            raise RuntimeError("CUDA requested but not available")\n'
         '        torch.backends.cuda.matmul.allow_tf32 = False\n'
         '        torch.backends.cudnn.allow_tf32 = False\n'
         '        torch.backends.cudnn.deterministic = True\n'
         '        torch.backends.cudnn.benchmark = False\n'
         '    return device\n'),
        ('def train(model: str = "resnet8", epochs: int = 60, batch_size: int = 128,\n'
         '          lr: float = 0.1, weight_decay: float = 1e-4, momentum: float = 0.9,\n'
         '          seed: int = 0, workers: int = 0, out: Path | None = None,\n'
         '          resume: bool = True) -> dict:',
         'def train(model: str = "resnet8", epochs: int = 60, batch_size: int = 128,\n'
         '          lr: float = 0.1, weight_decay: float = 1e-4, momentum: float = 0.9,\n'
         '          seed: int = 0, workers: int = 0, out: Path | None = None,\n'
         '          resume: bool = True, device: str = "cpu") -> dict:'),
        ('    model = build_model(arch)', '    device = setup_device(device)\n'
                                          '    model = build_model(arch).to(device)'),
        ('            loss = crit(model(x), y)',
         '            x, y = x.to(device), y.to(device)\n'
         '            loss = crit(model(x), y)'),
        ('        acc = evaluate_accuracy(model, test_loader)',
         '        acc = evaluate_accuracy(model, test_loader, device)'),
        ('    p.add_argument("--no-resume", action="store_true")',
         '    p.add_argument("--no-resume", action="store_true")\n'
         '    p.add_argument("--device", default="cpu")'),
        ('    train(model=a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,\n'
         '          seed=a.seed, workers=a.workers, resume=not a.no_resume)',
         '    train(model=a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,\n'
         '          seed=a.seed, workers=a.workers, resume=not a.no_resume,\n'
         '          device=a.device)'),
    ]))

    # ---- campaign entry point
    out.append(patch("experiments/e01_uniform_baseline.py", [
        ('    p.add_argument("--seed", type=int, default=0)\n',
         '    p.add_argument("--seed", type=int, default=0)\n'
         '    p.add_argument("--device", default="cpu")\n'),
        ('    folded = to_deploy(net)\n    mx = MXModel(folded, cfg).quantize_weights()',
         '    from mxfi.train import setup_device\n'
         '    dev = setup_device(a.device)\n'
         '    folded = to_deploy(net).to(dev)\n'
         '    mx = MXModel(folded, cfg).quantize_weights()'),
        ('    ev = Evaluator(ds)', '    ev = Evaluator(ds, device=dev)'),
    ]))

    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
