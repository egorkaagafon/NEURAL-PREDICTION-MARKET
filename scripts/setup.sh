#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  setup.sh — First-time setup on a new machine
# ═══════════════════════════════════════════════════════════════════
set -e

echo "══════════════════════════════════════════"
echo "  NPM — Environment Setup"
echo "══════════════════════════════════════════"

# 1. Install Python dependencies
echo "[1/2] Installing Python packages..."
pip install -r requirements.txt

# 2. Download all datasets (CIFAR-10, CIFAR-100, SVHN)
echo "[2/2] Downloading datasets..."
python download_data.py

echo ""
echo "✓ Setup complete. Next: bash scripts/run_all.sh"
