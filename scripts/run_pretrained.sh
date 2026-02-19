#!/usr/bin/env bash
# Train NPM + baselines using pretrained frozen ViT backbone (T4-optimised)
set -euo pipefail

EPOCHS="${1:-50}"
echo "=== Pretrained backbone experiment (${EPOCHS} epochs) ==="

cd "$(dirname "$0")/.."
python -m experiments.run_pretrained \
    --config configs/pretrained.yaml \
    --epochs "${EPOCHS}"
