"""
Data loading utilities for CIFAR‑10, CIFAR‑100, SVHN.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: bool = True,
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
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


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
