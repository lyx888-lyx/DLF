#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-1114}"
GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/cfcompat_video_aware_sampling_v1/mosi/seed${SEED}"

if [[ "${SEED}" != "1111" && "${SEED}" != "1114" ]]; then
  echo "SEED must be 1111 or 1114; got ${SEED}" >&2
  exit 2
fi

python3 smoke_test_cfcompat_video_aware_sampling.py

python3 audit_video_aware_sampling_domains.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu-ids "${GPU}" \
  --result-root "${RESULT_ROOT}" \
  --samples-per-video 4

python3 train_cfcompat_video_aware_sampling.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu-ids "${GPU}" \
  --num-workers 1 \
  --result-root "${RESULT_ROOT}" \
  --model-save-dir "${MODEL_SAVE_DIR}" \
  --samples-per-video 4

python3 audit_cfcompat_video_aware_sampling.py \
  --result-dir "${OUTPUT_DIR}"

echo "CFCompat video-aware sampling run complete: ${OUTPUT_DIR}"
