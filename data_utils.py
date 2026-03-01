"""
Data loading utilities for CIFAR-10, CIFAR-100, SVHN, Tiny ImageNet,
and OpenOOD v1.5 benchmark datasets.

OpenOOD protocol for Tiny ImageNet (ID = 200 classes):
  Near-OOD:  SSB-hard, ImageNet-O
  Far-OOD:   iNaturalist, SUN, Places, Textures (DTD)
"""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
import glob
from typing import Tuple, Optional, List, Dict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, Dataset
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
    "tiny_imagenet": {
        "mean": [0.4802, 0.4481, 0.3975],
        "std":  [0.2770, 0.2691, 0.2821],
        "num_classes": 200,
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


# ══════════════════════════════════════════════════════════════════════
#  Tiny ImageNet (200 classes, 64×64 native)
# ══════════════════════════════════════════════════════════════════════

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download_and_extract_tiny_imagenet(root: str = "./data") -> str:
    """Download Tiny ImageNet and return path to extracted directory."""
    tiny_dir = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(tiny_dir) and os.path.isdir(os.path.join(tiny_dir, "train")):
        return tiny_dir

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")

    if not os.path.isfile(zip_path):
        import urllib.request
        print("Downloading Tiny ImageNet (~237 MB)...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("  Download complete.")

    print("Extracting Tiny ImageNet...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    os.remove(zip_path)
    print(f"  Extracted to {tiny_dir}")
    return tiny_dir


def _reorganize_tiny_imagenet_val(tiny_dir: str) -> None:
    """Reorganize TinyImageNet val/ into class sub-directories for ImageFolder.

    Original structure:  val/images/val_0.JPEG, val_annotations.txt
    Target structure:    val/n01443537/val_0.JPEG  (class sub-folders)
    """
    val_dir = os.path.join(tiny_dir, "val")
    val_img_dir = os.path.join(val_dir, "images")

    if not os.path.isdir(val_img_dir):
        return  # already reorganized

    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    with open(annotations_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, class_id = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(val_img_dir, fname)
            dst = os.path.join(class_dir, fname)
            if os.path.exists(src):
                shutil.move(src, dst)

    # Clean up empty images dir
    if os.path.isdir(val_img_dir):
        try:
            os.rmdir(val_img_dir)
        except OSError:
            pass


def get_tiny_imagenet_loaders_224(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Tiny ImageNet resized to 224×224 with ImageNet normalization.

    Train set = official 100 000 images, Test set = official val 10 000.
    """
    tiny_dir = _download_and_extract_tiny_imagenet(root)
    _reorganize_tiny_imagenet_val(tiny_dir)

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomCrop(image_size, padding=image_size // 8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])

    train_ds = torchvision.datasets.ImageFolder(
        os.path.join(tiny_dir, "train"), transform=train_transform,
    )
    test_ds = torchvision.datasets.ImageFolder(
        os.path.join(tiny_dir, "val"), transform=test_transform,
    )

    dl_kwargs = dict(
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    return train_loader, test_loader


def get_tiny_imagenet_loaders_224_with_val(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tiny ImageNet 224px with train/val/test split.

    Train + Val come from the official training set (100 000 images, stratified).
    Test = official validation set (10 000 images).
    """
    tiny_dir = _download_and_extract_tiny_imagenet(root)
    _reorganize_tiny_imagenet_val(tiny_dir)

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomCrop(image_size, padding=image_size // 8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(tiny_dir, "train")
    full_train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_base_ds = torchvision.datasets.ImageFolder(train_dir, transform=test_transform)
    test_ds = torchvision.datasets.ImageFolder(
        os.path.join(tiny_dir, "val"), transform=test_transform,
    )

    # Stratified split
    n = len(full_train_ds)
    targets_arr = np.array([s[1] for s in full_train_ds.samples])
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
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    return train_loader, val_loader, test_loader


def num_classes_for(dataset: str) -> int:
    """Return the number of classes for a dataset name."""
    if dataset in DATASET_STATS:
        return DATASET_STATS[dataset]["num_classes"]
    raise ValueError(f"Unknown dataset: {dataset}")


def get_ood_loader(
    dataset_name: str = "textures",
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
    max_samples: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Load an OOD test set for uncertainty evaluation.

    Supports the full OpenOOD v1.5 protocol for Tiny ImageNet:

    Near-OOD (semantic shift):
      ssb_hard     -- Semantic Shift Benchmark (ImageNet-21K subsets
                      NOT in Tiny-ImageNet 200 classes)
      imagenet_o   -- Adversarially-filtered natural images

    Far-OOD:
      inaturalist  -- Plants / animals (iNaturalist 2021 subset)
      sun          -- Scene Understanding (SUN397)
      places       -- Places365-Standard (small version)
      textures     -- Describable Textures Dataset (DTD)

    Legacy (still supported):
      svhn, cifar10, cifar100

    Parameters
    ----------
    dataset_name : str
        One of the names above.
    root : str
        Data root directory.
    batch_size : int
    num_workers : int
    image_size : int
        Target image size (224 for pretrained backbones).
    max_samples : int
        If > 0, randomly subsample to this many images (useful for
        very large datasets like SUN / Places).
    seed : int
        Random seed for subsampling.
    """

    # ImageNet normalization for 224px models
    if image_size > 32:
        normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        transform = T.Compose([
            T.Resize(image_size + 32),     # slight over-resize
            T.CenterCrop(image_size),
            T.ToTensor(),
            normalize,
        ])
    else:
        normalize = T.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )
        transform = T.Compose([T.ToTensor(), normalize])

    ds = _load_ood_dataset(dataset_name, root, transform, image_size)

    # Optional subsampling for large datasets
    if max_samples > 0 and len(ds) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), size=max_samples, replace=False)
        ds = Subset(ds, indices)
        print(f"  OOD '{dataset_name}': subsampled {max_samples} / {len(ds)+max_samples}")

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  OpenOOD v1.5 dataset registry and loaders
# ══════════════════════════════════════════════════════════════════════

# Metadata for each OOD dataset
OOD_REGISTRY: Dict[str, dict] = {
    # ── Near-OOD ──
    "ssb_hard": {
        "category": "near-ood",
        "description": "SSB-hard: ImageNet-21K classes semantically close to "
                       "Tiny-ImageNet but non-overlapping (Semantic Shift Benchmark)",
        "auto_download": True,
    },
    "imagenet_o": {
        "category": "near-ood",
        "description": "ImageNet-O: adversarially filtered natural images "
                       "(Hendrycks et al., 2021)",
        "auto_download": True,
    },
    # ── Far-OOD ──
    "inaturalist": {
        "category": "far-ood",
        "description": "iNaturalist: 10K plant/animal images outside ImageNet "
                       "(OpenOOD curated subset)",
        "auto_download": True,
    },
    "sun": {
        "category": "far-ood",
        "description": "SUN397: scene understanding dataset",
        "auto_download": True,
    },
    "places": {
        "category": "far-ood",
        "description": "Places365: location/scene classification (small version)",
        "auto_download": True,
    },
    "textures": {
        "category": "far-ood",
        "description": "DTD: Describable Textures Dataset",
        "auto_download": True,
    },
    # ── Legacy (backward compat) ──
    "svhn":    {"category": "legacy", "auto_download": True},
    "cifar10": {"category": "legacy", "auto_download": True},
    "cifar100": {"category": "legacy", "auto_download": True},
}

# Download URLs / Google Drive IDs for OpenOOD-formatted datasets.
# Official IDs taken from https://github.com/Jingkang50/OpenOOD
# (scripts/download/download.py  —  v1.5 branch).
_OPENOOD_URLS = {
    # Direct HTTP links (preferred when available)
    "imagenet_o":  "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar",
}

# Google-Drive file-IDs for datasets that have no stable HTTP mirror.
_GDRIVE_IDS: dict[str, str] = {
    "ssb_hard":    "1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE",
    "inaturalist": "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj",
    "sun":         "1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp",
    "places":      "1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3",
    "textures":    "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
}


def _download_file(url: str, dest: str) -> None:
    """Download a file with progress indication."""
    import urllib.request
    import sys

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        sys.stdout.write(f"\r    downloading {pct:.0f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


def _download_gdrive(file_id: str, dest: str) -> None:
    """Download from Google Drive via *gdown*."""
    try:
        import gdown  # type: ignore
    except ImportError:
        raise RuntimeError(
            "gdown is required for Google-Drive downloads.\n"
            "  pip install gdown"
        )
    gdown.download(id=file_id, output=dest, quiet=False)


def _extract_archive(archive_path: str, dest_dir: str) -> None:
    """Extract .zip or .tar/.tar.gz archive."""
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.endswith(".tar") or archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def _download_openood_dataset(name: str, root: str) -> str:
    """Download and extract an OpenOOD benchmark dataset.

    Returns the path to the dataset directory (ImageFolder-compatible).
    """
    ood_root = os.path.join(root, "ood_datasets")
    os.makedirs(ood_root, exist_ok=True)

    # Check if already extracted
    dataset_dir = os.path.join(ood_root, name)
    if os.path.isdir(dataset_dir) and _has_images(dataset_dir):
        return dataset_dir

    url = _OPENOOD_URLS.get(name)
    gdrive_id = _GDRIVE_IDS.get(name)
    if url is None and gdrive_id is None:
        raise ValueError(
            f"No download URL for OOD dataset '{name}'. "
            f"Please download manually to {dataset_dir}/"
        )

    print(f"  Downloading OOD dataset: {name}")
    if url is not None:
        ext = ".tar" if url.endswith(".tar") else ".zip"
        archive_path = os.path.join(ood_root, f"{name}{ext}")
        _download_file(url, archive_path)
    else:
        archive_path = os.path.join(ood_root, f"{name}.zip")
        _download_gdrive(gdrive_id, archive_path)  # type: ignore[arg-type]

    print(f"  Extracting {name}...")
    _extract_archive(archive_path, ood_root)
    os.remove(archive_path)

    # Handle ImageNet-O special case: extracts as 'imagenet-o/' not 'imagenet_o/'
    alt_dir = os.path.join(ood_root, name.replace("_", "-"))
    if not os.path.isdir(dataset_dir) and os.path.isdir(alt_dir):
        os.rename(alt_dir, dataset_dir)

    # Handle DTD special case: extracts as 'dtd/' not 'textures/'
    if name == "textures":
        dtd_dir = os.path.join(ood_root, "dtd")
        if not os.path.isdir(dataset_dir) and os.path.isdir(dtd_dir):
            # DTD has images/ subdir with class folders
            dtd_images = os.path.join(dtd_dir, "images")
            if os.path.isdir(dtd_images):
                os.rename(dtd_images, dataset_dir)
                shutil.rmtree(dtd_dir, ignore_errors=True)
            else:
                os.rename(dtd_dir, dataset_dir)

    # Ensure the directory is ImageFolder-compatible
    _ensure_imagefolder_structure(dataset_dir)

    if not _has_images(dataset_dir):
        print(f"  WARNING: {dataset_dir} has no images. "
              f"Check the extraction or download manually.")

    return dataset_dir


def _has_images(dir_path: str) -> bool:
    """Quick check if directory has image files (recursively)."""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".JPEG"}
    for root_d, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1] in IMG_EXTS:
                return True
    return False


def _ensure_imagefolder_structure(dataset_dir: str) -> None:
    """If images are in the root (no class sub-folders), create a dummy class.

    ImageFolder requires at least one sub-directory level.
    """
    if not os.path.isdir(dataset_dir):
        return

    # Check if there are already sub-directories with images
    subdirs = [d for d in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, d))]

    has_class_dirs = any(_has_images(os.path.join(dataset_dir, d)) for d in subdirs)
    if has_class_dirs:
        return  # Already in ImageFolder format

    # Check if images are directly in root
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".JPEG"}
    root_images = [f for f in os.listdir(dataset_dir)
                   if os.path.splitext(f)[1] in IMG_EXTS]

    if root_images:
        # Move images into a dummy class folder
        dummy_dir = os.path.join(dataset_dir, "0")
        os.makedirs(dummy_dir, exist_ok=True)
        for img in root_images:
            src = os.path.join(dataset_dir, img)
            dst = os.path.join(dummy_dir, img)
            shutil.move(src, dst)
        print(f"  Reorganized {len(root_images)} images into ImageFolder format")


def _load_ood_dataset(
    name: str,
    root: str,
    transform: T.Compose,
    image_size: int,
) -> Dataset:
    """Dispatch OOD dataset loading by name."""

    # ── Legacy torchvision datasets ──
    if name == "svhn":
        return torchvision.datasets.SVHN(
            root, split="test", download=True, transform=transform,
        )
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root, train=False, download=True, transform=transform,
        )
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(
            root, train=False, download=True, transform=transform,
        )

    # ── Textures (DTD) via torchvision ──
    if name == "textures":
        try:
            # Try torchvision built-in first (available since torchvision 0.13)
            ds = torchvision.datasets.DTD(
                root=root, split="test", download=True, transform=transform,
            )
            return ds
        except Exception:
            pass
        # Fallback: download from OpenOOD
        dataset_dir = _download_openood_dataset(name, root)
        return torchvision.datasets.ImageFolder(dataset_dir, transform=transform)

    # ── OpenOOD benchmark datasets (ImageFolder format) ──
    if name in ("ssb_hard", "imagenet_o", "inaturalist", "sun", "places"):
        dataset_dir = _download_openood_dataset(name, root)
        return torchvision.datasets.ImageFolder(dataset_dir, transform=transform)

    raise ValueError(
        f"Unknown OOD dataset: '{name}'.\n"
        f"Supported datasets:\n"
        f"  Near-OOD: ssb_hard, imagenet_o\n"
        f"  Far-OOD:  inaturalist, sun, places, textures\n"
        f"  Legacy:   svhn, cifar10, cifar100"
    )


def get_all_ood_loaders(
    ood_names: List[str],
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
    max_samples: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Load multiple OOD datasets at once. Returns {name: DataLoader}."""
    loaders = {}
    for name in ood_names:
        try:
            loaders[name] = get_ood_loader(
                name, root=root, batch_size=batch_size,
                num_workers=num_workers, image_size=image_size,
                max_samples=max_samples, seed=seed,
            )
            print(f"  OOD '{name}': {len(loaders[name].dataset)} images")
        except Exception as e:
            print(f"  WARNING: Failed to load OOD dataset '{name}': {e}")
            print(f"           Skipping '{name}' in OOD evaluation.")
    return loaders


def list_ood_datasets() -> None:
    """Print a summary table of all supported OOD datasets."""
    print(f"\n{'name':<15s} {'category':<10s} description")
    print("-" * 80)
    for name, info in OOD_REGISTRY.items():
        cat = info.get("category", "")
        desc = info.get("description", "")
        print(f"{name:<15s} {cat:<10s} {desc}")


def download_all(root: str = "./data"):
    """Pre-download every dataset used in the project.

    Run this once on a new machine:
        python download_data.py          # or
        python -c "from data_utils import download_all; download_all()"
    """
    print("=" * 60)
    print("Downloading ID datasets")
    print("=" * 60)

    print("Downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(root, train=True, download=True)
    torchvision.datasets.CIFAR10(root, train=False, download=True)

    print("Downloading CIFAR-100...")
    torchvision.datasets.CIFAR100(root, train=True, download=True)
    torchvision.datasets.CIFAR100(root, train=False, download=True)

    print("Downloading SVHN...")
    torchvision.datasets.SVHN(root, split="train", download=True)
    torchvision.datasets.SVHN(root, split="test", download=True)

    print("Downloading Tiny ImageNet...")
    _download_and_extract_tiny_imagenet(root)
    _reorganize_tiny_imagenet_val(os.path.join(root, "tiny-imagenet-200"))

    print("\n" + "=" * 60)
    print("Downloading OpenOOD v1.5 benchmark datasets")
    print("=" * 60)

    ood_names = ["textures", "ssb_hard", "imagenet_o",
                 "inaturalist", "sun", "places"]
    for name in ood_names:
        try:
            _download_openood_dataset(name, root)
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    print(f"\n{'='*60}")
    print(f"All datasets downloaded to {root}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════
#  224px loaders for pretrained ViT/DeiT backbones
# ══════════════════════════════════════════════════════════════════════

# ImageNet normalization (used by all timm pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_cifar10_loaders_224(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-10 resized to 224×224 with ImageNet normalization.

    For pretrained ViT/DeiT backbones trained on ImageNet.
    Augmentation: RandomCrop(224 from 256) + flip (standard ImageNet pipeline).
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomCrop(image_size, padding=image_size // 8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        normalize,
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=train_transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=test_transform,
    )

    dl_kwargs = dict(
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    return train_loader, test_loader


def get_cifar10_loaders_224_with_val(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 224px with train/val/test split (stratified)."""
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomCrop(image_size, padding=image_size // 8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        normalize,
    ])

    full_train_ds = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=train_transform,
    )
    val_base_ds = torchvision.datasets.CIFAR10(
        root, train=True, download=False, transform=test_transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=test_transform,
    )

    # Stratified split
    n = len(full_train_ds)
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
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    return train_loader, val_loader, test_loader
