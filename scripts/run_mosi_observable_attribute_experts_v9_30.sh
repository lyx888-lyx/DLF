#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
ROOT="${V919_ROOT:-result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}}"
OUTPUT_DIR="${V930_OUTPUT_DIR:-${ROOT}/v930_observable_attribute_experts}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-18}"
EARLY_STOP="${EARLY_STOP:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
APP_FLOOR="${APP_FLOOR:-0.10}"
APP_POWER="${APP_POWER:-2.0}"
SHRINKAGE_LAMBDA="${SHRINKAGE_LAMBDA:-0.01}"
COMPAT="scripts/v925_scipy_compat_runner.py"

python3 "${COMPAT}" scripts/smoke_test_observable_attribute_experts_v9_30.py

python3 "${COMPAT}" analyze_observable_attribute_experts_v9_30.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --v919-root "${ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --outer-folds "${OUTER_FOLDS}" \
  --batch-size "${BATCH_SIZE}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-epochs "${MAX_EPOCHS}" \
  --early-stop "${EARLY_STOP}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --applicability-floor "${APP_FLOOR}" \
  --applicability-power "${APP_POWER}" \
  --shrinkage-lambda "${SHRINKAGE_LAMBDA}"

python3 "${COMPAT}" scripts/audit_observable_attribute_experts_v9_30.py \
  --result-dir "${OUTPUT_DIR}"
