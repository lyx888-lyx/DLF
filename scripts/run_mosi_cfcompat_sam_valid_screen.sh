#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/cfcompat_sam_v1/mosi/valid_screen"

python3 smoke_test_cfcompat_sam.py

python3 train_cfcompat_sam_valid_screen.py \
  --dataset mosi \
  --seeds 1111 1114 \
  --gpu-ids "${GPU}" \
  --num-workers 1 \
  --result-root "${RESULT_ROOT}" \
  --model-save-dir "${MODEL_SAVE_DIR}"

python3 audit_cfcompat_sam_valid_screen.py \
  --result-dir "${OUTPUT_DIR}"

echo "CFCompatKD + SAM Valid screen complete: ${OUTPUT_DIR}"
