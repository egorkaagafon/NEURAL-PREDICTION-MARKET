"""
Data loading utilities for CIFAR‑10, CIFAR‑100, SVHN.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Standard CIFAR‑10 train/test loaders with optional augmentation."""

    normalize = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    if augmentation:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([T.ToTensor(), normalize])

    test_transform = T.Compose([T.ToTensor(), normalize])

    train_ds = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=train_transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=test_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return train_loader, test_loader


def get_cifar10_loaders_with_val(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: bool = True,
    val_fraction: float = 0.1,
    seed: int = 42,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 train/val/test.  Val uses test transforms (no augmentation)."""

    normalize = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    if augmentation:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([T.ToTensor(), normalize])

    test_transform = T.Compose([T.ToTensor(), normalize])

    # Full training set — will split into train + val
    full_train_ds = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=train_transform,
    )
    # Same data but with test transform for val evaluation
    val_base_ds = torchvision.datasets.CIFAR10(
        root, train=True, download=False, transform=test_transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=test_transform,
    )

    # Stratified split — equal class proportions in train and val
    n = len(full_train_ds)
    n_val = int(n * val_fraction)
    targets_arr = np.array(full_train_ds.targets)
    classes = np.unique(targets_arr)
    rng = np.random.RandomState(seed)

    val_indices = []
    for c in classes:
        c_idx = np.where(targets_arr == c)[0]
        rng.shuffle(c_idx)
        n_c_val = int(len(c_idx) * val_fraction)
        val_indices.append(c_idx[:n_c_val])
    val_idx = np.concatenate(val_indices)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(val_base_ds, val_idx)

    dl_kwargs = dict(
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    return train_loader, val_loader, test_loader


# ── Dataset statistics ──────────────────────────────────────────────
DATASET_STATS = {
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std":  [0.2470, 0.2435, 0.2616],
        "num_classes": 10,
    },
    "cifar100": {
        "mean": [0.5071, 0.4867, 0.4408],
        "std":  [0.2675, 0.2565, 0.2761],
        "num_classes": 100,
    },
    "svhn": {
        "mean": [0.4377, 0.4438, 0.4728],
        "std":  [0.1980, 0.2010, 0.1970],
        "num_classes": 10,
    },
}


def get_cifar100_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Standard CIFAR‑100 train/test loaders with optional augmentation."""

    stats = DATASET_STATS["cifar100"]
    normalize = T.Normalize(mean=stats["mean"], std=stats["std"])

    if augmentation:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([T.ToTensor(), normalize])

    test_transform = T.Compose([T.ToTensor(), normalize])

    train_ds = torchvision.datasets.CIFAR100(
        root, train=True, download=True, transform=train_transform,
    )
    test_ds = torchvision.datasets.CIFAR100(
        root, train=False, download=True, transform=test_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return train_loader, test_loader


def get_loaders(
    dataset: str = "cifar10",
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Universal loader dispatcher — returns (train_loader, test_loader)."""
    kw = dict(root=root, batch_size=batch_size, num_workers=num_workers,
              augmentation=augmentation, persistent_workers=persistent_workers,
              prefetch_factor=prefetch_factor)
    if dataset == "cifar10":
        return get_cifar10_loaders(**kw)
    elif dataset == "cifar100":
        return get_cifar100_loaders(**kw)
    else:
        raise ValueError(f"Unknown training dataset: {dataset}. "
                         f"Supported: cifar10, cifar100")


def num_classes_for(dataset: str) -> int:
    """Return the number of classes for a dataset name."""
    if dataset in DATASET_STATS:
        return DATASET_STATS[dataset]["num_classes"]
    raise ValueError(f"Unknown dataset: {dataset}")


def get_ood_loader(
    dataset_name: str = "cifar100",
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """Load an OOD test set for uncertainty evaluation."""

    # Use CIFAR‑10 normalization so distribution shift is real
    normalize = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
    transform = T.Compose([T.ToTensor(), normalize])

    if dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root, train=False, download=True, transform=transform,
        )
    elif dataset_name == "svhn":
        ds = torchvision.datasets.SVHN(
            root, split="test", download=True, transform=transform,
        )
    else:
        raise ValueError(f"Unknown OOD dataset: {dataset_name}")

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def download_all(root: str = "./data"):
    """Pre‑download every dataset used in the project.

    Run this once on a new machine:
        python download_data.py          # or
        python -c "from data_utils import download_all; download_all()"
    """
    print("Downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(root, train=True, download=True)
    torchvision.datasets.CIFAR10(root, train=False, download=True)

    print("Downloading CIFAR-100...")
    torchvision.datasets.CIFAR100(root, train=True, download=True)
    torchvision.datasets.CIFAR100(root, train=False, download=True)

    print("Downloading SVHN...")
    torchvision.datasets.SVHN(root, split="train", download=True)
    torchvision.datasets.SVHN(root, split="test", download=True)

    print("\n✓ All datasets downloaded to", root)
