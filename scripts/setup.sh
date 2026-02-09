#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  setup.sh — First-time setup on a new machine (uv)
# ═══════════════════════════════════════════════════════════════════
set -e

echo "══════════════════════════════════════════"
echo "  NPM — Environment Setup (uv)"
echo "══════════════════════════════════════════"

# 0. Check uv is installed
if ! command -v uv &>/dev/null; then
    echo "[0/2] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv $(uv --version)"

# 1. Sync project (creates .venv, installs all deps)
echo "[1/2] Syncing project dependencies..."
uv sync

# 2. Download all datasets (CIFAR-10, CIFAR-100, SVHN)
echo "[2/2] Downloading datasets..."
uv run python download_data.py

echo ""
echo "✓ Setup complete. Next: bash scripts/run_all.sh"
