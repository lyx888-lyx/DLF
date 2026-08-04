#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/cfcompat_cmixup_full_only_v2/mosi/seed${SEED}"

python3 smoke_test_cfcompat_cmixup_full_only.py

python3 train_cfcompat_cmixup_full_only.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu-ids "${GPU}" \
  --num-workers 1 \
  --result-root "${RESULT_ROOT}" \
  --model-save-dir "${MODEL_SAVE_DIR}"

python3 audit_cfcompat_cmixup_full_only.py \
  --result-dir "${OUTPUT_DIR}"

echo "CFCompat full-view C-Mixup v2 run complete: ${OUTPUT_DIR}"
