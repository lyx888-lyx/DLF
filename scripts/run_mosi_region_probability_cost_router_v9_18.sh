#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"

python3 train_region_probability_cost_router_v9_18.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --v917-root "result/fixed_expert_region_audit_v917/mosi/seed_${SEED}" \
  --strict-pool "result/strict_nested_frontier_coach_v98/mosi/seed_${SEED}/strict_nested_frontier_pool/strict_nested_v93_frontier_pool_v98.pth"

python3 scripts/audit_region_probability_cost_router_v9_18.py \
  --root "result/region_probability_cost_router_v918/mosi/seed_${SEED}"
