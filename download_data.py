#!/usr/bin/env python3
"""
Pre-download all datasets used in the project.

Run this once on a new machine before training:

    python download_data.py [--root ./data]

Downloads:
  ID datasets:
    CIFAR-10        (~170 MB)
    CIFAR-100       (~170 MB)
    SVHN            (~200 MB)
    Tiny ImageNet   (~237 MB)

  OpenOOD v1.5 benchmark (OOD evaluation):
    Textures / DTD  (~600 MB)
    SSB-hard        (Semantic Shift Benchmark)
    ImageNet-O      (~200 MB)
    iNaturalist     (~1 GB curated subset)
    SUN             (SUN397 subset)
    Places          (Places365 subset)
"""

import argparse
from data_utils import download_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all NPM datasets")
    parser.add_argument("--root", default="./data", help="Where to store datasets")
    args = parser.parse_args()
    download_all(args.root)
