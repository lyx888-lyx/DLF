#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
OUTER_FOLD="${OUTER_FOLD:?Set OUTER_FOLD to 0,1,2,3,or 4}"
NUM_WORKERS="${NUM_WORKERS:-1}"

python3 train_same_stack_nested_crossfit_v9_19.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --outer-fold "${OUTER_FOLD}" \
  --outer-folds 5 \
  --inner-folds 3 \
  --num-workers "${NUM_WORKERS}"

python3 scripts/audit_same_stack_nested_crossfit_v9_19.py \
  --root "result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}" \
  --outer-fold "${OUTER_FOLD}"
