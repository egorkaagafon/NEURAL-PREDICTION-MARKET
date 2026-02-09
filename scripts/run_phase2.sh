#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  run_phase2.sh — Phase 2 only: OOD detection
#  Requires a trained checkpoint.
#
#  Usage:
#    bash scripts/run_phase2.sh runs/<timestamp>/ckpt_epoch200.pt
# ═══════════════════════════════════════════════════════════════════
set -e
CKPT=${1:?Usage: bash scripts/run_phase2.sh <checkpoint_path>}
DEVICE=${2:-cuda}
echo "Phase 2: checkpoint=${CKPT}, device=${DEVICE}"
python experiments/run_phase2.py --checkpoint "${CKPT}" --device "${DEVICE}"
echo "→ results/phase2_results.json"
