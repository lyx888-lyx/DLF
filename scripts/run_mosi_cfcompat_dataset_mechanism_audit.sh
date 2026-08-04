#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/mosi_cfcompat_dataset_mechanism_audit_v1/mosi/train_valid_audit"

python3 -m py_compile \
  trains/singleTask/mosi_cfcompat_audit_utils.py \
  trains/singleTask/mosi_cfcompat_audit_v2_utils.py \
  analyze_mosi_cfcompat_dataset_mechanism.py \
  analyze_mosi_cfcompat_dataset_mechanism_v2.py \
  audit_mosi_cfcompat_dataset_mechanism.py \
  audit_mosi_cfcompat_dataset_mechanism_v2.py \
  audit_mosi_cfcompat_dataset_mechanism_v3.py \
  smoke_test_mosi_cfcompat_dataset_mechanism.py

required_assets=(
  "${MODEL_SAVE_DIR}/DLF_mosi_seed1111_best.pth"
  "${MODEL_SAVE_DIR}/DLF_mosi_seed1114_best.pth"
  "${RESULT_ROOT}/missing_baseline/moddrop/train/mosi_per_seed.csv"
  "${RESULT_ROOT}/missing_baseline/moddrop_benchmark_multiseed_v1/seed1114/mosi_per_seed.csv"
  "${RESULT_ROOT}/missing_baseline/cf_compat_kd_v1/benchmark_train/mosi_per_seed.csv"
  "${RESULT_ROOT}/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1114/mosi_per_seed.csv"
  "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1/mosi/train_counterfactual_compatibility.csv"
  "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1/mosi/cf_compat_config.json"
  "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/train_counterfactual_compatibility.csv"
  "${RESULT_ROOT}/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/cf_compat_config.json"
)

for asset in "${required_assets[@]}"; do
  if [[ ! -f "${asset}" ]]; then
    echo "Missing required MOSI CFCompat audit asset: ${asset}" >&2
    exit 1
  fi
done

python3 smoke_test_mosi_cfcompat_dataset_mechanism.py

python3 analyze_mosi_cfcompat_dataset_mechanism_v2.py \
  --dataset mosi \
  --seeds 1111 1114 \
  --gpu-ids "${GPU}" \
  --num-workers 1 \
  --result-root "${RESULT_ROOT}" \
  --model-save-dir "${MODEL_SAVE_DIR}" \
  --bootstrap-replicates 2000

python3 audit_mosi_cfcompat_dataset_mechanism_v3.py \
  --result-dir "${OUTPUT_DIR}"

python3 - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ.get(
    "OUTPUT_DIR",
    os.path.join(
        os.environ.get("RESULT_ROOT", "result"),
        "missing_baseline/mosi_cfcompat_dataset_mechanism_audit_v1/mosi/train_valid_audit",
    ),
))
summary = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
print("\n================ MOSI dataset limitations ================")
for key, value in summary["dataset_limitations"].items():
    print(f"{key}: {value}")
print("\n================ CFCompat mechanism ================")
mechanism = summary["mechanism_assessment"]
print("verdict:", mechanism["verdict"])
print("mean two-seed J gain:", f"{mechanism['mean_two_seed_J_gain']:+.6f}")
print("bootstrap:", mechanism["video_bootstrap_J_gain"])
for key, value in mechanism["checks"].items():
    print(f"{key}: {value}")
print("\n================ Overall prediction summary ================")
print(pd.read_csv(root / "overall_prediction_summary.csv").to_string(index=False))
print("\n================ Top descriptive opportunities ================")
opportunities = pd.read_csv(root / "opportunity_ranking.csv")
print(opportunities.head(25).to_string(index=False))
print("\nreport:", root / "audit_report.md")
PY

echo "MOSI CFCompatKD dataset-mechanism audit complete: ${OUTPUT_DIR}"
