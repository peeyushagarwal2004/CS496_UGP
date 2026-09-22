"""Train a CIFAR-10 reference model to convergence on CPU.

Kept deliberately plain -- the study is about what faults do to a trained
network, not about squeezing the last point of accuracy out of it.  What does
matter is that the checkpoint is *reproducible* and that the folded,
MX-quantised baseline accuracy is recorded alongside it, since every reported
failure rate is relative to that baseline.

Run::

    .venv/Scripts/python.exe -m mxfi.train --epochs 60
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import cifar10_loaders
from .models import MODELS, build_model, count_parameters

__all__ = ["train", "evaluate_accuracy", "load_trained", "checkpoint_stem",
           "CHECKPOINT_DIR"]

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def setup_device(device: str = "cpu") -> str:
    """Select a device and make CUDA as numerically faithful as the CPU.

    TF32 would silently drop matmul/conv precision to ~10 mantissa bits,
    which can flip borderline predictions and make GPU-measured failure
    rates incomparable with the CPU ones already collected.
    """
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return device


def checkpoint_stem(model: str, seed: int = 0) -> str:
    """Checkpoint file stem for a (model, training seed) pair.

    Seed 0 keeps the original unsuffixed name, so every checkpoint trained
    before seeds were tracked stays loadable unchanged.
    """
    return f"{model}_cifar10" if seed == 0 else f"{model}_s{seed}_cifar10"


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader,
                      device: str = "cpu") -> float:
    """Top-1 accuracy over `loader`."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1)
        correct += int((pred == y.to(device)).sum())
        total += y.numel()
    return correct / max(total, 1)


def train(model: str = "resnet8", epochs: int = 60, batch_size: int = 128,
          lr: float = 0.1, weight_decay: float = 1e-4, momentum: float = 0.9,
          seed: int = 0, workers: int = 0, out: Path | None = None,
          resume: bool = True, device: str = "cpu", optimizer: str = "sgd",
          label_smoothing: float = 0.0) -> dict:
    """Train a CIFAR-10 model, checkpointing after every epoch."""
    torch.manual_seed(seed)
    out = Path(out) if out else CHECKPOINT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"{checkpoint_stem(model, seed)}.pt"
    log_path = out / f"{checkpoint_stem(model, seed)}.log.json"
    arch = model

    train_loader, test_loader = cifar10_loaders(batch_size, workers)
    device = setup_device(device)
    model = build_model(arch).to(device)
    # transformers do not train under the SGD recipe that suits these CNNs
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    start_epoch, best, history = 0, 0.0, []
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best_acc"]
        history = state.get("history", [])
        print(f"resumed from epoch {start_epoch} (best {best:.4f})")

    print(f"{arch}: {count_parameters(model):,} params | "
          f"{len(train_loader)} batches/epoch | {torch.get_num_threads()} threads")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * y.numel()
            seen += y.numel()
        sched.step()

        acc = evaluate_accuracy(model, test_loader, device)
        best = max(best, acc)
        rec = {"epoch": epoch, "loss": running / seen, "test_acc": acc,
               "lr": sched.get_last_lr()[0], "secs": time.time() - t0}
        history.append(rec)
        print(f"epoch {epoch:3d}  loss {rec['loss']:.4f}  "
              f"acc {acc:.4f}  best {best:.4f}  {rec['secs']:.1f}s")

        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "best_acc": best, "acc": acc, "history": history,
                    "config": {"arch": arch, "epochs": epochs,
                               "batch_size": batch_size, "lr": lr,
                               "seed": seed, "optimizer": optimizer,
                               "label_smoothing": label_smoothing}}, ckpt_path)
        log_path.write_text(json.dumps(history, indent=2))

        # graceful stop for long sweeps: the checkpoint for this epoch is
        # already on disk, so halting here loses nothing and resumes cleanly
        stop = os.environ.get("STOP_FILE")
        if stop and Path(stop).exists():
            print(f"stop requested via {stop}; halting after epoch {epoch}")
            break

    print(f"\ndone: final {history[-1]['test_acc']:.4f}, best {best:.4f}")
    print(f"checkpoint: {ckpt_path}")
    return {"best_acc": best, "final_acc": history[-1]["test_acc"],
            "path": str(ckpt_path), "history": history}


def load_trained(model: str = "resnet8", path: Path | None = None,
                 seed: int = 0) -> tuple[nn.Module, dict]:
    """Load a trained model and its checkpoint metadata."""
    path = (Path(path) if path
            else CHECKPOINT_DIR / f"{checkpoint_stem(model, seed)}.pt")
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}; run "
            f"`python -m mxfi.train --model {model}` first")
    state = torch.load(path, map_location="cpu", weights_only=False)
    net = build_model(state.get("config", {}).get("arch", model))
    net.load_state_dict(state["model"])
    return net.eval(), state


def main() -> None:
    p = argparse.ArgumentParser(description="Train a CIFAR-10 model")
    p.add_argument("--model", default="resnet8", choices=sorted(MODELS))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--optimizer", default="sgd", choices=("sgd", "adamw"))
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    a = p.parse_args()
    train(model=a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
          seed=a.seed, workers=a.workers, resume=not a.no_resume,
          device=a.device, optimizer=a.optimizer, weight_decay=a.weight_decay,
          label_smoothing=a.label_smoothing)


if __name__ == "__main__":
    main()
