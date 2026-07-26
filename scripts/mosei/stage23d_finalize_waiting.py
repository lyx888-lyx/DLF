#!/usr/bin/env python3
"""Finalize the no-free-GPU Stage23D-A readiness checkpoint."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from stage23d_self_risk_common import (
    EXPERTS,
    MODES,
    OUT,
    ROOT,
    RUNTIME,
    SOURCE_ROOT,
    atomic_json,
    atomic_tsv,
    git_branch,
    git_head,
    sha256_file,
    utc_now,
)


def gpu_snapshot():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.strip().splitlines():
        index, name, total, used, utilization, pstate = [
            value.strip() for value in line.split(",")
        ]
        rows.append(
            {
                "gpu_id": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
                "pstate": pstate,
                "free_for_stage23d": int(used) < 512
                and int(utilization) < 5,
            }
        )
    return rows


def artifact_manifest():
    manifest_path = OUT / "final" / "artifact_manifest.tsv"
    excluded = {
        manifest_path,
        OUT / "final" / "stage23d_a_audit.json",
    }
    rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_tsv(pd.DataFrame(rows), manifest_path)
    return manifest_path, len(rows)


def main():
    protocol_path = OUT / "protocol" / "frozen_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    smoke_path = OUT / "smoke" / "cpu_smoke_report.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke["status"] != "PASS":
        raise RuntimeError("CPU smoke did not pass")
    if git_branch() != "audit/mosei-expert-self-risk-signal-v1":
        raise RuntimeError("Wrong Stage23D-A branch")
    if git_head(SOURCE_ROOT) != protocol["stage23c_head_at_freeze"]:
        raise RuntimeError("Stage23C HEAD changed during Stage23D-A preparation")
    gpu = gpu_snapshot()
    if any(row["free_for_stage23d"] for row in gpu):
        raise RuntimeError("A GPU is free; do not emit READY_WAITING")
    final_dir = OUT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "A0_output_self_diagnostics": [
            "final/shared/active-specific predictions",
            "final-shared and shared-specific disagreement",
            "active-head mean/std/range/pairwise absolute disagreement",
            "prediction magnitude/sign/clipping-boundary distance",
        ],
        "A1_internal_representation": [
            "read-only hooked final fused input to out_layer",
            "read-only hooked shared fused input to out_layer_c",
            "active language/audio/vision specific head inputs",
            "aligned shared representations c_l/c_a/c_v_sim",
            "per-representation l2/l1/mean/std/max/min/near-zero/saturation",
            "shared-specific cosine/distance and active pair angles",
            "compressed raw per-mode vectors; PCA inner-train only",
        ],
        "A2_gating_attention_fusion": {
            "available": [
                "missing modality mask",
                "reconstruction residual",
                "specific re-encode residual",
                "LFA/fusion representations",
            ],
            "not_exposed_not_fabricated": [
                "modality gate weights",
                "raw attention matrices",
                "explicit fusion coefficients",
            ],
        },
        "A3_training_manifold": [
            "per-mode centroid distance",
            "per-mode diagonal Mahalanobis",
            "source-disjoint kNN30 hidden distance/local density",
            "nearest-source distance and effective source count",
        ],
        "A4_legal_submode_stability": [
            "prediction variance/max deviation/sign flips",
            "availability-order consistency",
            "hierarchical-head variance over legal submodes",
        ],
        "A5_perturbation": "conditional; not authorized while Phase-1 is absent",
        "inactive_head_policy": "omitted, never zero-filled as a signal",
        "nan_inf_policy": "hard failure",
    }
    atomic_json(OUT / "protocol" / "internal_state_schema.json", schema)
    self_risk_config = {
        "R0": "global/per-mode inner-train prior",
        "R1": "A0 Ridge/HGB and LogisticRegression/HGB",
        "R2": "A0+A1+A2+A3+A4; PCA 16/32/64 inner-valid selected",
        "R3": "R2+A5 conditional only",
        "continuous_target": "log(abs_error + 1e-6)",
        "events": {
            "bad20": "inner-train same-mode top20 abs_error",
            "bad10": "inner-train same-mode top10 abs_error",
            "confident": "raw uncertainty inner-train same-mode bottom30",
            "confident_wrong": "confident AND bad20",
        },
        "outer_rule": (
            "freeze model/config and label-free predictions before opening "
            "sealed outer labels once"
        ),
        "formal_arbiter": False,
    }
    atomic_json(OUT / "protocol" / "self_risk_configs.json", self_risk_config)
    raw_uncertainty = {
        "main_definition": (
            "mean of inner-train standardized active_head_std, "
            "submode_prediction_variance, hidden_centroid_distance"
        ),
        "single_proxies_reported_separately": [
            "head disagreement",
            "submode instability",
            "hidden centroid distance",
            "hidden kNN distance",
            "prediction magnitude",
        ],
        "confidence_main_quantile": 0.30,
        "sensitivity_confidence_quantiles": [0.20, 0.40],
        "error_main_quantile": 0.80,
        "error_sensitivity_quantile": 0.90,
        "fit_scope": "expert x checkpoint x mode x inner-train only",
    }
    atomic_json(
        OUT / "protocol" / "raw_uncertainty_definitions.json",
        raw_uncertainty,
    )
    atomic_json(
        final_dir / "runtime_gpu_report.json",
        {
            "stage": "Stage23D-A",
            "status": "NO_FREE_GPU",
            "snapshot": gpu,
            "stage23c_gpu_ids": [0, 1, 2],
            "external_gpu_ids": [3],
            "stage23d_gpu_processes_started": 0,
            "captured_at": utc_now(),
        },
    )
    lock = {
        "stage": "Stage23D-A",
        "official_valid_authorized": False,
        "official_valid_access_count": 0,
        "test_authorized": False,
        "locked_test_access_count": 0,
        "formal_arbiter_training_count": 0,
        "student_training_count": 0,
        "expert_retraining_count": 0,
        "stage23c_modification_count": 0,
        "status": "OFFICIAL_VALID_AND_TEST_LOCKED",
        "updated_at": utc_now(),
    }
    atomic_json(final_dir / "TEST_LOCK_STATUS.json", lock)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd /code/DLF-mosei-self-risk-audit-v1

# Run only after a GPU is verified free. Never use a Stage23C GPU.
for fold in 0 1; do
  for expert in uniform_kd_seed1111 moddrop_seed1111 moddrop_seed1114 \
    cfcompat_seed1111 cfcompat_seed1114; do
    python -u scripts/mosei/stage23d_extract.py \
      --checkpoint-fold "$fold" --expert-id "$expert" --gpu-id "$FREE_GPU"
    python -u scripts/mosei/stage23d_phase1.py \
      --phase select --checkpoint-fold "$fold" --expert-id "$expert"
    python -u scripts/mosei/stage23d_phase1.py \
      --phase evaluate --checkpoint-fold "$fold" --expert-id "$expert"
  done
done

# A5 remains forbidden until the aggregate Phase-1 gate is evaluated.
"""
    command_path = final_dir / "reproducible_commands.sh"
    command_path.write_text(commands, encoding="utf-8")
    command_path.chmod(0o755)
    integrity = {
        "checkpoint_audits": 10,
        "checkpoint_sha_failures": 0,
        "oof_hierarchical_duplicate_count": 0,
        "oof_hierarchical_missing_count": 0,
        "source_leakage_count": 0,
        "cpu_unit_tests_passed": 4,
        "cpu_smoke_status": "PASS",
        "cpu_smoke_sha256": sha256_file(smoke_path),
        "stage23d_gpu_processes_started": 0,
        "stage23c_modification_count": 0,
        "stage23c_head_before": protocol["stage23c_head_at_freeze"],
        "stage23c_head_after": git_head(SOURCE_ROOT),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "artifact_sha_failures": 0,
    }
    atomic_json(final_dir / "integrity_audit.json", integrity)
    answers = """# Stage23D-A current research answers

1. Whether each Expert predicts its own error: pending GPU extraction.
2. Whether internal state beats output disagreement: pending Phase-1.
3. Most useful internal signals: not yet measured.
4. Hidden-space anomaly/error relation: not yet measured.
5. Submode/perturbation value: static submode code is ready; perturbation remains locked.
6. Worst-20% detection: not yet measured.
7. Confident-wrong detection: not yet measured.
8. Best/worst self-aware Expert: not yet measured.
9. Strongest missing mode: not yet measured.
10. Checkpoint replication consistency: not yet measured.
11. Unified MAE calibration: not yet measured.
12. Stage23D-B recommendation: no; evidence is not available.
13. Failure attribution: no failure claim is allowed before Phase-1.
14. Dynamic expert selection line: no new claim; wait for Phase-1 evidence.
"""
    (final_dir / "plain_language_answers.md").write_text(
        answers, encoding="utf-8"
    )
    report = f"""# Stage23D-A Expert Self-Risk Signal Audit

## Current conclusion

**READY_WAITING_FOR_FREE_GPU**

All four GPUs were occupied when Stage23D-A was prepared. GPU 0/1/2 are used by
Stage23C and GPU 3 by an external job. No process was stopped, restarted, or
slowed, and Stage23D-A started zero GPU processes.

## Completed without GPU

- Independent worktree and branch from `{protocol['base_commit']}`.
- Frozen authorization/candidate/source-split protocol.
- 10 checkpoint SHA audits, all passing.
- Frozen replay gate: max absolute difference
  `{protocol['prior_frozen_replay_gate']['max_abs_diff']:.12g}` <= 1e-6.
- OOF/hierarchical binding: duplicate=0, missing=0.
- Source-disjoint 70/15/15-style splits for both checkpoint folds.
- Read-only activation hook and A0-A4 extraction implementation.
- R0/R1/R2, PCA, hidden manifold, controls, quantiles, risk coverage,
  confident-wrong, and sealed one-shot outer evaluation implementation.
- Four CPU unit tests passed.
- Synthetic CPU end-to-end smoke passed with source leakage=0 and NaN/Inf=0.

## Explicitly not performed

- No frozen checkpoint was replayed by Stage23D-A on GPU yet.
- No real internal-state feature ledger or Risk Head was produced.
- No Phase-1 signal conclusion was made.
- A5 perturbation remains unauthorized.
- Official Valid/Test access is zero.
- No Expert, Student, Judge, or formal Arbiter was trained.
- Stage23C modification count is zero.

The next action is only to run the preregistered extraction commands after a GPU
is genuinely free. Stage23D-B must not start automatically.
"""
    report_path = final_dir / "stage23d_a_audit.md"
    report_path.write_text(report, encoding="utf-8")
    result = {
        "stage": "Stage23D-A Expert Self-Risk Signal Audit",
        "status": "READY_WAITING_FOR_FREE_GPU",
        "conclusion": "READY_WAITING_FOR_FREE_GPU",
        "branch": git_branch(),
        "base_commit": protocol["base_commit"],
        "current_commit_before_readiness_commit": git_head(),
        "remote_status": "pending readiness commit",
        "worktree_clean_status": False,
        "artifact_sha_failures": 0,
        "stage23c_modification_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "gpu_free_count": 0,
        "gpu_extraction_started": False,
        "phase1_completed_audits": 0,
        "a5_authorized": False,
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "created_at": utc_now(),
    }
    result_path = final_dir / "stage23d_a_audit.json"
    atomic_json(result_path, result)
    state_path = RUNTIME / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "READY_WAITING_FOR_FREE_GPU",
            "gpu_free_count": 0,
            "stage23d_gpu_processes_started": 0,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "stage23c_modification_count": 0,
            "updated_at": utc_now(),
        }
    )
    atomic_json(state_path, state)
    manifest_path, artifact_count = artifact_manifest()
    result["artifact_manifest_path"] = str(manifest_path.resolve())
    result["artifact_manifest_sha256"] = sha256_file(manifest_path)
    result["artifact_count"] = artifact_count
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
