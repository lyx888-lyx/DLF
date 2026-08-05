#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/cfcompat_safe_projection_v1/mosi/valid_screen"
MODEL_DIR="${MODEL_SAVE_DIR}/missing_baseline/cfcompat_safe_projection_v1/mosi/valid_screen"
export OUTPUT_DIR

python3 -m py_compile \
  trains/singleTask/cfcompat_safe_projection_utils.py \
  train_cfcompat_safe_projection_valid_screen.py \
  audit_cfcompat_safe_projection_valid_screen.py \
  smoke_test_cfcompat_safe_projection.py

for seed in 1112 1113 1115; do
  required_assets=(
    "${MODEL_SAVE_DIR}/DLF_mosi_seed${seed}_best.pth"
    "${RESULT_ROOT}/missing_baseline/moddrop_benchmark_multiseed_v1/seed${seed}/mosi_per_seed.csv"
    "${RESULT_ROOT}/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed${seed}/mosi_per_seed.csv"
    "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed${seed}/train_counterfactual_compatibility.csv"
    "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed${seed}/cf_compat_config.json"
  )
  for asset in "${required_assets[@]}"; do
    if [[ ! -f "${asset}" ]]; then
      echo "Missing required Safe-CFCompat asset: ${asset}" >&2
      exit 1
    fi
  done
done

python3 smoke_test_cfcompat_safe_projection.py

rm -rf "${OUTPUT_DIR}" "${MODEL_DIR}"

python3 train_cfcompat_safe_projection_valid_screen.py \
  --dataset mosi \
  --seeds 1112 1113 1115 \
  --gpu-ids "${GPU}" \
  --num-workers 1 \
  --result-root "${RESULT_ROOT}" \
  --model-save-dir "${MODEL_SAVE_DIR}"

python3 audit_cfcompat_safe_projection_valid_screen.py \
  --result-dir "${OUTPUT_DIR}"

python3 - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["OUTPUT_DIR"])
summary = json.loads(
    (root / "safe_projection_valid_screen_summary.json").read_text(
        encoding="utf-8"
    )
)
grid = pd.read_csv(root / "safe_projection_valid_grid_summary.csv")
groups = pd.read_csv(root / "safe_projection_group_metrics.csv")

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)

print("\n================ Safe-CFCompat grid ================")
columns = [
    "Seed",
    "Run",
    "BestValidEpoch",
    "J_valid",
    "valid_LAV_MAE",
    "valid_LA_MAE",
    "valid_LV_MAE",
    "valid_L_MAE",
    "projection_wrong_direction_fraction",
    "projection_overshoot_fraction",
    "projection_unchanged_fraction",
]
print(
    grid[columns]
    .sort_values(["Seed", "Run"])
    .to_string(index=False, float_format=lambda x: f"{x:+.6f}")
)

print("\n================ Candidate gates ================")
for run in ("safe_uniform", "safe_cfcompat"):
    gate = summary["candidate_gates"][run]
    print(f"\n{run}")
    print("passed:", gate["passed"])
    print(
        "mean J gain vs CFCompatKD:",
        f"{gate['mean_gain_valid_J_vs_CFCompatKD']:+.6f}",
    )
    for key, value in gate["checks"].items():
        print(f"  {key}: {value}")
    for seed, evidence in gate["per_seed"].items():
        print(
            f"  Seed {seed}: "
            f"J gain={evidence['gain_valid_J_vs_CFCompatKD']:+.6f}, "
            f"Q1={evidence['Q1_candidate_gain_vs_DLF']:+.6f}, "
            f"Q4={evidence['Q4_candidate_gain_vs_DLF']:+.6f}, "
            f"harm={evidence['candidate_harmful_imitation_rate']:.6f}"
        )

print("\n================ Failure-repair groups ================")
selected = groups.loc[
    groups.Seed.astype(str).ne("POOLED")
    & (
        (
            groups.GroupType.eq("baseline_error_quartile")
            & groups.GroupValue.isin(["Q1_easy", "Q4_hard"])
        )
        | (
            groups.GroupType.eq("teacher_condition")
            & groups.GroupValue.eq("better_and_correct")
        )
        | (
            groups.GroupType.eq("all")
            & groups.GroupValue.eq("ALL")
        )
    )
]
print(
    selected[
        [
            "Seed",
            "Run",
            "GroupType",
            "GroupValue",
            "N",
            "gain_vs_DLF",
            "harmful_imitation_rate",
            "helpful_imitation_rate",
        ]
    ]
    .sort_values(["Seed", "GroupType", "GroupValue", "Run"])
    .to_string(index=False, float_format=lambda x: f"{x:+.6f}")
)

print("\nverdict:", summary["verdict"])
print("official Test was not constructed")
print("report:", root / "safe_projection_valid_screen_report.md")
PY

echo "Safe-CFCompatKD Valid screen complete: ${OUTPUT_DIR}"
