#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  run_phase3.sh — Phase 3 only: Ablation study
# ═══════════════════════════════════════════════════════════════════
set -e
EPOCHS=${1:-100}
DEVICE=${2:-cuda}
echo "Phase 3: epochs=${EPOCHS}, device=${DEVICE}"
python experiments/run_phase3.py --config configs/default.yaml --epochs "${EPOCHS}" --device "${DEVICE}"
echo "→ results/phase3_ablation.json"
