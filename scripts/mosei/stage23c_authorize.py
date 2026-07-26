#!/usr/bin/env python
"""Freeze the Stage23C train-OOF-only distillation protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "result" / "arbiter_audit_v1" / "mosei"
V2 = ROOT / "result" / "arbiter_audit_v2" / "mosei"
V2B = ROOT / "result" / "arbiter_audit_v2b" / "mosei"
OUT = ROOT / "result" / "oracle_distillation_v1" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23c"
EXPERTS = (
    "uniform_kd_seed1111",
    "moddrop_seed1111",
    "moddrop_seed1114",
    "cfcompat_seed1111",
    "cfcompat_seed1114",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=str(ROOT), text=True
    ).strip()


def main():
    expected_branch = "experiment/mosei-soft-oracle-hierarchical-distillation-v1"
    if git("branch", "--show-current") != expected_branch:
        raise RuntimeError("Stage23C must run on its isolated branch.")
    status_lines = [
        line
        for line in git("status", "--porcelain").splitlines()
        if line.strip()
    ]
    allowed_bootstrap = {"?? scripts/mosei/stage23c_authorize.py"}
    if set(status_lines) - allowed_bootstrap:
        raise RuntimeError("Authorization requires a clean worktree.")

    v2a_path = V2 / "analysis_v2a" / "stage23a_v2a_signal_audit.json"
    v2b_path = V2B / "final" / "stage23a_v2b_audit.json"
    v2b_lock_path = V2B / "final" / "TEST_LOCK_STATUS.json"
    v2a = json.loads(v2a_path.read_text(encoding="utf-8"))
    v2b = json.loads(v2b_path.read_text(encoding="utf-8"))
    lock = json.loads(v2b_lock_path.read_text(encoding="utf-8"))
    if v2a.get("status") != "SIGNAL_AUDIT_FAIL":
        raise RuntimeError("Frozen v2a conclusion changed.")
    if v2b.get("status") != "LOCAL_COMPETENCE_WEAK":
        raise RuntimeError("Frozen v2b conclusion changed.")
    if lock["official_valid_access_count"] or lock["locked_test_access_count"]:
        raise RuntimeError("A protected data lock is open.")

    candidates = {
        "student_seed": [1111],
        "soft_oracle_tau": [0.02, 0.05, 0.10, 0.20],
        "fixed_teacher_l2": [0.0, 0.001, 0.01],
        "lambda_final": 1.0,
        "lambda_hier_ratio": [0.25, 0.50, 1.00],
        "shuffle_seeds": [23611, 23612, 23613],
        "methods": [
            "S0_supervised_moddrop",
            "S1_uniform_single_teacher_kd",
            "S2_equal_ensemble_final_kd",
            "S3_fixed_stacking_final_kd",
            "S4_soft_oracle_final_kd",
            "S5_soft_oracle_hierarchical_kd",
            "S6_shuffled_soft_oracle_hierarchical_kd",
        ],
    }
    candidate_path = OUT / "protocol" / "candidate_table.json"
    write_json(candidate_path, candidates)
    protocol = {
        "stage": "Stage23C OOF Soft-Oracle Hierarchical Distillation Audit",
        "status": "FROZEN_AUTHORIZED_TRAIN_OOF_STUDENT_SCREEN_ONLY",
        "created_at": utc_now(),
        "branch": expected_branch,
        "parent_commit": git("rev-parse", "HEAD"),
        "frozen_parent_conclusions": {
            "Stage23A-v2a": "SIGNAL_AUDIT_FAIL",
            "Stage23A-v2b": "LOCAL_COMPETENCE_WEAK",
        },
        "parent_artifacts": {
            str(v2a_path.relative_to(ROOT)): sha256(v2a_path),
            str(v2b_path.relative_to(ROOT)): sha256(v2b_path),
            str(v2b_lock_path.relative_to(ROOT)): sha256(v2b_lock_path),
        },
        "expert_ids": list(EXPERTS),
        "oracle_target_audit_authorized": True,
        "student_screen_training_authorized": True,
        "student_second_seed_authorized": False,
        "full_data_training_authorized": False,
        "official_valid_authorized": False,
        "test_authorized": False,
        "judge_training_authorized": False,
        "joint_specialized_moe_authorized": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "outer_evaluation_access_count": 0,
        "student_seed": 1111,
        "student_protocol": {
            "architecture": "DLF wrapped by frozen Stage1 MissingModalityWrapper",
            "initialization": (
                "one direction-specific clean DLF checkpoint shared by every "
                "method; clean screen uses inner-train/inner-valid and final "
                "clean retrain uses the frozen clean epoch on development side"
            ),
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "batch_size": 16,
            "max_epochs": 30,
            "early_stopping_patience": 6,
            "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5)",
            "gradient_clip_value": 0.6,
            "gradient_accumulation_batches": 10,
            "supervised_objective": (
                "original complete-view DLF objective plus one uniformly sampled "
                "LA/LV/L ModDrop task objective"
            ),
            "scalar_kd": "SmoothL1, lambda_final=1.0",
            "mode_sampling": "Stage23A sample_missing_masks; LA/LV/L only",
            "checkpoint_selection": "inner-valid J only",
            "outer_rule": (
                "freeze all method configurations/epochs, retrain on full "
                "development side, write all label-free predictions, then open "
                "outer labels once for metrics"
            ),
        },
        "direction_protocol": {
            "A": {
                "development_expert_fold": 0,
                "inner_train_samples": 7438,
                "inner_valid_samples": 1094,
                "outer_expert_fold": 1,
                "outer_samples": 7794,
            },
            "B": {
                "development_expert_fold": 1,
                "inner_train_samples": 6933,
                "inner_valid_samples": 861,
                "outer_expert_fold": 0,
                "outer_samples": 8532,
            },
        },
        "active_hierarchical_heads": {
            "LAV": [
                "logits_c",
                "logits_l_hetero",
                "logits_a_hetero",
                "logits_v_hetero",
            ],
            "LA": ["logits_c", "logits_l_hetero", "logits_a_hetero"],
            "LV": ["logits_c", "logits_l_hetero", "logits_v_hetero"],
            "L": ["logits_c", "logits_l_hetero"],
        },
        "teacher_rules": {
            "soft_responsibility": "softmax(-regret/tau), same sample and mode",
            "convex_hull_required": True,
            "outer_oracle_target_forbidden": True,
            "raw_hidden_state_distillation": False,
            "inactive_specific_head_loss": False,
            "hierarchical_loss_normalized_by_active_head_count": True,
        },
        "selection_order": [
            "select tau by inner-valid soft-oracle final target J",
            "select fixed stacking L2 by inner-valid fixed target J",
            "select clean and S0-S4 epochs by inner-valid J",
            "select S5 lambda_hier ratio and epoch by inner-valid J",
            "freeze lambda_hier then select each preregistered S6 epoch",
            "retrain every method on full development side for frozen epoch",
            "predict each outer method exactly once without labels",
            "open outer labels only after every prediction is frozen",
        ],
        "outer_search_expansion_forbidden": True,
        "candidate_table_path": str(candidate_path.resolve()),
        "candidate_table_sha256": sha256(candidate_path),
        "possible_conclusions": [
            "ORACLE_DISTILLATION_PASS",
            "ORACLE_DISTILLATION_WEAK",
            "ORACLE_DISTILLATION_FAIL",
        ],
    }
    protocol_path = OUT / "protocol" / "frozen_protocol_manifest.json"
    write_json(protocol_path, protocol)
    state = {
        "stage": "Stage23C",
        "status": protocol["status"],
        "protocol_manifest_path": str(protocol_path.resolve()),
        "protocol_manifest_sha256": sha256(protocol_path),
        "student_seed": 1111,
        "student_second_seed_count": 0,
        "outer_evaluation_access_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "judge_trained": False,
        "joint_specialized_moe_trained": False,
        "full_data_training_started": False,
        "updated_at": utc_now(),
    }
    write_json(RUNTIME / "state.json", state)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
