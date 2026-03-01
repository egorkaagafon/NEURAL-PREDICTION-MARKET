#!/usr/bin/env bash
# Train NPM + baselines using pretrained frozen backbone (auto-adapts to GPU)
# Supports: deit_tiny_patch16_224, deit_small_patch16_224, resnet18, resnet50
set -euo pipefail

EPOCHS="${1:-50}"
BACKBONE="${2:-}"  # optional: pass backbone name as 2nd arg
echo "=== Pretrained backbone experiment (${EPOCHS} epochs) ==="

cd "$(dirname "$0")/.."

BACKBONE_ARG=""
if [[ -n "${BACKBONE}" ]]; then
    BACKBONE_ARG="--backbone ${BACKBONE}"
    echo "Backbone override: ${BACKBONE}"
fi

python -m experiments.run_pretrained \
    --config configs/pretrained.yaml \
    --epochs "${EPOCHS}" \
    ${BACKBONE_ARG}
