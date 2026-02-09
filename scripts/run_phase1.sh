#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  run_phase1.sh — Phase 1 only: NPM vs baselines
# ═══════════════════════════════════════════════════════════════════
set -e
EPOCHS=${1:-100}
DEVICE=${2:-cuda}
echo "Phase 1: epochs=${EPOCHS}, device=${DEVICE}"
uv run python experiments/run_phase1.py --config configs/default.yaml --epochs "${EPOCHS}" --device "${DEVICE}"
echo "→ results/phase1_results.json"
