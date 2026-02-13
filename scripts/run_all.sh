#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  run_all.sh — Full research pipeline (all 3 phases)
#
#  Usage:
#    bash scripts/run_all.sh                          # full run (200 epochs)
#    bash scripts/run_all.sh --quick                  # quick test (20 epochs)
#    bash scripts/run_all.sh --skip=ensemble          # skip deep ensemble
#    bash scripts/run_all.sh --quick --skip="ensemble mc_dropout"
#
#  Expected total time (on 1× A100):
#    --quick  : ~30 min
#    full     : ~6-8 hours
# ═══════════════════════════════════════════════════════════════════
set -e

EPOCHS=200
DEVICE="cuda"
SKIP=""

for arg in "$@"; do
    case "$arg" in
        --quick)  EPOCHS=20; echo "⚡ Quick mode: ${EPOCHS} epochs" ;;
        --skip=*) SKIP="${arg#--skip=}"; echo "⏭  Skip: ${SKIP}" ;;
    esac
done

echo "Device: ${DEVICE}, Epochs: ${EPOCHS}"
echo ""

# ──────────────────────────────────────────────────────────────────
#  PHASE 1: NPM vs Baselines on CIFAR-10
# ──────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════"
echo "  PHASE 1 — Sanity Check (NPM vs Baselines)"
echo "══════════════════════════════════════════"
uv run python experiments/run_phase1.py \
    --config configs/default.yaml \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}" \
    ${SKIP:+--skip $SKIP}

echo ""
echo "→ Results: results/phase1_results.json"
echo ""

# ──────────────────────────────────────────────────────────────────
#  Train main NPM model (for Phase 2 checkpoint)
# ──────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════"
echo "  Training main NPM (for Phase 2)"
echo "══════════════════════════════════════════"
uv run python train.py \
    --config configs/default.yaml \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}"

# Find the latest checkpoint
LATEST_CKPT=$(ls -t runs/*/ckpt_epoch*.pt 2>/dev/null | head -1)
if [[ -z "$LATEST_CKPT" ]]; then
    echo "⚠ No checkpoint found, skipping Phase 2"
else
    echo ""
    echo "→ Checkpoint: ${LATEST_CKPT}"
    echo ""

    # ──────────────────────────────────────────────────────────────
    #  PHASE 2: OOD Detection
    # ──────────────────────────────────────────────────────────────
    echo "══════════════════════════════════════════"
    echo "  PHASE 2 — OOD Detection (CIFAR-100, SVHN)"
    echo "══════════════════════════════════════════"
    uv run python experiments/run_phase2.py \
        --checkpoint "${LATEST_CKPT}" \
        --device "${DEVICE}"

    echo ""
    echo "→ Results: results/phase2_results.json"
    echo ""
fi

# ──────────────────────────────────────────────────────────────────
#  PHASE 3: Ablation Study
# ──────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════"
echo "  PHASE 3 — Ablation Study"
echo "══════════════════════════════════════════"
uv run python experiments/run_phase3.py \
    --config configs/default.yaml \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}"

echo ""
echo "→ Results: results/phase3_ablation.json"
echo ""

# ──────────────────────────────────────────────────────────────────
#  Summary
# ──────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════"
echo "  ✓ ALL PHASES COMPLETE"
echo "══════════════════════════════════════════"
echo ""
echo "Results:"
echo "  results/phase1_results.json    — NPM vs Ensemble vs MC-Dropout vs MoE"
echo "  results/phase2_results.json    — OOD detection AUROC/AUPR"
echo "  results/phase3_ablation.json   — Ablation: what matters in NPM"
echo ""
echo "TensorBoard logs:"
echo "  uv run tensorboard --logdir runs/"
