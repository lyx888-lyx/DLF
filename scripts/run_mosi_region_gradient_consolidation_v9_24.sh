#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
ROOT="result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}"

python3 scripts/smoke_test_region_gradient_consolidation_v9_24.py

for FOLD in 0 1 2 3 4; do
  python3 train_region_gradient_consolidation_v9_24.py \
    --dataset mosi \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --v919-root "${ROOT}" \
    --outer-fold "${FOLD}" \
    --outer-folds 5 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --max-epochs 40 \
    --early-stop 8 \
    --gradient-clip-norm 1.0 \
    --minimum-mgda-direction-norm 0.02 \
    --resume
done

python3 aggregate_region_gradient_consolidation_v9_24.py \
  --v919-root "${ROOT}" \
  --outer-folds 5 \
  --bootstrap-repetitions 2000 \
  --bootstrap-seed "${SEED}"

python3 scripts/audit_region_gradient_consolidation_v9_24.py \
  --v919-root "${ROOT}" \
  --outer-folds 5
