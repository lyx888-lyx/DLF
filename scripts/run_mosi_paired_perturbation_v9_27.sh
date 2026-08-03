#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ROOT="result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}"
COMPAT="scripts/v925_scipy_compat_runner.py"
EXTRA_ARGS=()
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-resume)
fi

python3 "${COMPAT}" scripts/smoke_test_paired_perturbation_v9_27.py

python3 "${COMPAT}" \
  analyze_paired_perturbation_relative_stability_v9_27.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --v919-root "${ROOT}" \
  --outer-folds 5 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --noise-scale 0.01 \
  --temporal-mask-fraction 0.05 \
  --shrinkage-lambda 0.01 \
  --bootstrap-repetitions 2000 \
  --bootstrap-seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

python3 "${COMPAT}" scripts/audit_paired_perturbation_v9_27.py \
  --v919-root "${ROOT}" \
  --outer-folds 5
