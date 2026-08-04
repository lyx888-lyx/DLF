#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/cfcompat_teacher_consensus_v1/mosi/valid_screen"

python3 -m py_compile \
  trains/singleTask/cfcompat_teacher_consensus_utils.py \
  smoke_test_cfcompat_teacher_consensus.py \
  train_cfcompat_teacher_consensus_valid_screen.py \
  audit_cfcompat_teacher_consensus_valid_screen.py

required_assets=(
  "${MODEL_SAVE_DIR}/DLF_mosi_seed1111_best.pth"
  "${MODEL_SAVE_DIR}/DLF_mosi_seed1112_best.pth"
  "${MODEL_SAVE_DIR}/DLF_mosi_seed1113_best.pth"
  "${MODEL_SAVE_DIR}/DLF_mosi_seed1114_best.pth"
  "${MODEL_SAVE_DIR}/DLF_mosi_seed1115_best.pth"
  "${RESULT_ROOT}/missing_baseline/cf_compat_kd_v1/benchmark_train/mosi_per_seed.csv"
  "${RESULT_ROOT}/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1114/mosi_per_seed.csv"
  "${RESULT_ROOT}/missing_baseline/moddrop/train/mosi_per_seed.csv"
  "${RESULT_ROOT}/missing_baseline/moddrop_benchmark_multiseed_v1/seed1114/mosi_per_seed.csv"
  "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1/mosi/train_counterfactual_compatibility.csv"
  "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1/mosi/cf_compat_config.json"
  "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/train_counterfactual_compatibility.csv"
  "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/cf_compat_config.json"
)

for asset in "${required_assets[@]}"; do
  if [[ ! -f "${asset}" ]]; then
    echo "Missing required teacher-consensus asset: ${asset}" >&2
    exit 1
  fi
done

python3 smoke_test_cfcompat_teacher_consensus.py

python3 train_cfcompat_teacher_consensus_valid_screen.py \
  --dataset mosi \
  --seeds 1111 1114 \
  --gpu-ids "${GPU}" \
  --num-workers 1 \
  --result-root "${RESULT_ROOT}" \
  --model-save-dir "${MODEL_SAVE_DIR}"

python3 audit_cfcompat_teacher_consensus_valid_screen.py \
  --result-dir "${OUTPUT_DIR}"

echo "CFCompatKD teacher-consensus Valid screen complete: ${OUTPUT_DIR}"
