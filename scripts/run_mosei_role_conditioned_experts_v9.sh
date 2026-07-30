#!/usr/bin/env bash
set -euo pipefail

extra=()
if [[ -n "${TEACHER_GLOB:-}" ]]; then
  extra+=(--teacher-glob "${TEACHER_GLOB}")
fi
if [[ -n "${TEACHER_CACHE:-}" ]]; then
  extra+=(--teacher-cache "${TEACHER_CACHE}")
fi

python3 train_role_conditioned_experts_v9.py \
  --dataset mosei \
  --seed "${SEED:-1111}" \
  --gpu "${GPU:-0}" \
  --save-root "${SAVE_ROOT:-./result/role_conditioned_experts_v9}" \
  "${extra[@]}" \
  "$@"

python3 scripts/audit_role_conditioned_experts_v9.py \
  --root "${SAVE_ROOT:-./result/role_conditioned_experts_v9}/mosei/seed_${SEED:-1111}"
