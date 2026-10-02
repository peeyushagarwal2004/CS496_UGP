"""CIFAR-10 / CIFAR-100 loading for training and for fault-injection campaigns.

Campaigns need something training does not: a *fixed, small, deterministic*
evaluation batch.  A single fault costs one forward pass, so a campaign of
10k injections over the full 10k-image test set is 100M image-inferences --
out of reach on CPU.  :func:`campaign_subset` therefore draws a fixed,
class-balanced subset once, and every injection is scored on exactly that
subset, so run-to-run differences come from the fault and nothing else.

CIFAR-100 stands in for ImageNet, which no machine this project can reach
holds. Which dataset a model uses is part of its registry name: a ``_c100``
suffix (``vit_small_c100``) means CIFAR-100, anything else CIFAR-10, so result
and checkpoint names from the two can never collide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

__all__ = ["CIFAR10_MEAN", "CIFAR10_STD", "CIFAR100_MEAN", "CIFAR100_STD",
           "DATASETS", "dataset_of", "loaders", "cifar10_loaders",
           "campaign_subset", "campaign_images", "DATA_ROOT"]

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# name -> (torchvision class, mean, std, number of classes)
DATASETS = {
    "cifar10": (datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD, 10),
    "cifar100": (datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD, 100),
}
C100_SUFFIX = "_c100"


def dataset_of(model: str) -> str:
    """The dataset a registry model name was trained on."""
    return "cifar100" if model.endswith(C100_SUFFIX) else "cifar10"


def _transforms(train: bool, dataset: str = "cifar10"):
    _, mean, std, _ = DATASETS[dataset]
    norm = transforms.Normalize(mean, std)
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])


def loaders(dataset: str = "cifar10", batch_size: int = 128, workers: int = 0,
            root: Path | str = DATA_ROOT,
            download: bool = True) -> tuple[DataLoader, DataLoader]:
    """Standard train/test loaders with the usual augmentation.

    ``workers=0`` by default: on a 4-core CPU the worker processes cost more
    than they save for a model this small.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cls = DATASETS[dataset][0]
    train = cls(root, train=True, download=download, transform=_transforms(True, dataset))
    test = cls(root, train=False, download=download, transform=_transforms(False, dataset))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=workers, drop_last=False),
        DataLoader(test, batch_size=256, shuffle=False, num_workers=workers),
    )


def cifar10_loaders(batch_size: int = 128, workers: int = 0,
                    root: Path | str = DATA_ROOT,
                    download: bool = True) -> tuple[DataLoader, DataLoader]:
    """CIFAR-10 loaders; kept for the callers written before CIFAR-100."""
    return loaders("cifar10", batch_size, workers, root, download)


def campaign_subset(n_per_class: int = 20, root: Path | str = DATA_ROOT,
                    seed: int = 0, download: bool = True,
                    dataset: str = "cifar10") -> TensorDataset:
    """A fixed, class-balanced slice of the test set, materialised in memory.

    Returns a ``TensorDataset`` of pre-normalised images so that every
    injection replays byte-identical inputs with no dataloader randomness.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    test = DATASETS[dataset][0](root, train=False, download=download,
                                transform=_transforms(False, dataset))

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


def campaign_images(model: str, images: int = 200, download: bool = False,
                    seed: int = 0) -> TensorDataset:
    """The campaign subset for `model`'s dataset, `images` in total.

    Every campaign scores the same number of images whatever the dataset, so a
    per-inference rate has the same resolution and an injection the same cost:
    200 images is 20 per class on CIFAR-10 and 2 per class on CIFAR-100.
    """
    dataset = dataset_of(model)
    classes = DATASETS[dataset][3]
    if images % classes:
        raise ValueError(f"{images} images do not split evenly over {classes} classes")
    return campaign_subset(images // classes, seed=seed, download=download,
                           dataset=dataset)
