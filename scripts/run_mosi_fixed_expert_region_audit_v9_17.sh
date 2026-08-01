#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-500}"

python3 train_fixed_expert_region_audit_v9_17.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --num-workers "${NUM_WORKERS}" \
  --bootstrap-repeats "${BOOTSTRAP_REPEATS}"

python3 scripts/audit_fixed_expert_region_audit_v9_17.py \
  --root "result/fixed_expert_region_audit_v917/mosi/seed_${SEED}"
