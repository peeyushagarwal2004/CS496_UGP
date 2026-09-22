"""CIFAR-10 loading for training and for fault-injection campaigns.

Campaigns need something training does not: a *fixed, small, deterministic*
evaluation batch.  A single fault costs one forward pass, so a campaign of
10k injections over the full 10k-image test set is 100M image-inferences --
out of reach on CPU.  :func:`campaign_subset` therefore draws a fixed,
class-balanced subset once, and every injection is scored on exactly that
subset, so run-to-run differences come from the fault and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

__all__ = ["CIFAR10_MEAN", "CIFAR10_STD", "cifar10_loaders", "campaign_subset",
           "DATA_ROOT"]

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _transforms(train: bool):
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])


def cifar10_loaders(batch_size: int = 128, workers: int = 0,
                    root: Path | str = DATA_ROOT,
                    download: bool = True) -> tuple[DataLoader, DataLoader]:
    """Standard CIFAR-10 train/test loaders with the usual augmentation.

    ``workers=0`` by default: on a 4-core CPU the worker processes cost more
    than they save for a model this small.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(root, train=True, download=download,
                             transform=_transforms(True))
    test = datasets.CIFAR10(root, train=False, download=download,
                            transform=_transforms(False))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=workers, drop_last=False),
        DataLoader(test, batch_size=256, shuffle=False, num_workers=workers),
    )


def campaign_subset(n_per_class: int = 20, root: Path | str = DATA_ROOT,
                    seed: int = 0, download: bool = True) -> TensorDataset:
    """A fixed, class-balanced slice of the test set, materialised in memory.

    Returns a ``TensorDataset`` of pre-normalised images so that every
    injection replays byte-identical inputs with no dataloader randomness.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    test = datasets.CIFAR10(root, train=False, download=download,
                            transform=_transforms(False))

    targets = np.asarray(test.targets)
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for c in np.unique(targets):
        idx = np.flatnonzero(targets == c)
        picks.extend(rng.choice(idx, size=n_per_class, replace=False).tolist())
    picks.sort()

    sub = Subset(test, picks)
    xs = torch.stack([sub[i][0] for i in range(len(sub))])
    ys = torch.tensor([sub[i][1] for i in range(len(sub))], dtype=torch.long)
    return TensorDataset(xs, ys)
