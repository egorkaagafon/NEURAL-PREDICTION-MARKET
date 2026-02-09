#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  train_cifar100.sh — Train NPM on CIFAR-100
# ═══════════════════════════════════════════════════════════════════
set -e
EPOCHS=${1:-200}
DEVICE=${2:-cuda}
echo "Training NPM on CIFAR-100: epochs=${EPOCHS}, device=${DEVICE}"
python train.py --config configs/cifar100.yaml --epochs "${EPOCHS}" --device "${DEVICE}"
