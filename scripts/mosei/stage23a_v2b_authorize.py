#!/usr/bin/env python
"""Freeze Stage23A-v2b Source-Aware Local Competence Audit protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V2A = ROOT / "result" / "arbiter_audit_v2" / "mosei"
V2B = ROOT / "result" / "arbiter_audit_v2b" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23a_v2b"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parent_report = V2A / "analysis_v2a" / "stage23a_v2a_signal_audit.json"
    parent_lock = V2A / "final_v2a" / "TEST_LOCK_STATUS.json"
    parent = json.loads(parent_report.read_text(encoding="utf-8"))
    lock = json.loads(parent_lock.read_text(encoding="utf-8"))
    if parent["status"] != "SIGNAL_AUDIT_FAIL":
        raise RuntimeError("v2a conclusion must remain SIGNAL_AUDIT_FAIL.")
    if lock["official_valid_access_count"] or lock["locked_test_access_count"]:
        raise RuntimeError("A parent data lock is open.")

    candidates = {
        "spaces": ["C", "D", "H"],
        "distance_families": ["standardized_euclidean", "cosine"],
        "K": [15, 30, 60],
        "hybrid_beta": [0.25, 0.50, 0.75],
        "distance_weighting": [
            "reciprocal_epsilon_1e-6",
            "exponential_query_neighbor_median_bandwidth",
        ],
        "local_risk_definition": "distance-weighted neighbor absolute error",
        "DRS": ["Local-DS", "Local-DW", "Local-DWS"],
        "DW_tau": [0.02, 0.05, 0.10, 0.20],
        "DWS_risk_margin": [0.02, 0.05, 0.10],
        "safe_coverages": [0.10, 0.20, 0.30, 0.50, 1.00],
        "control_seeds": {
            "random_same_mode_source": [23401, 23402, 23403],
            "shuffled_content": [23411, 23412, 23413],
            "decision_randomization": [23421, 23422, 23423],
        },
        "common_configuration": {
            "space": "C",
            "distance": "standardized_euclidean",
            "K": 30,
            "weighting": "reciprocal_epsilon_1e-6",
            "DRS": "Local-DW",
            "tau": 0.05,
            "safe_coverage": 0.20,
        },
        "selection_order": [
            "select Space/distance/K/weighting by inner-valid local-risk correlation; ranking accuracy is tie-breaker",
            "select DS/DW/DWS and small tau/margin candidates by inner-valid J",
            "select safe coverage/absolute estimated-gain threshold by inner-valid J; no-improvement falls back to zero dynamic coverage",
            "freeze all choices, then transform/predict outer once",
        ],
        "outer_search_expansion_forbidden": True,
    }
    protocol = {
        "stage": "Stage23A-v2b Source-Aware Local Competence Audit",
        "status": "FROZEN_AUTHORIZED_TRAIN_OOF_LOCAL_AUDIT_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_v2a_commit": "4e6178bdcaaf5cf7730e81f7b20e305466e7cf83",
        "parent_v2a_branch": "audit/mosei-feature-rich-meta-judge-v2a-signal-audit",
        "parent_v2a_conclusion": "SIGNAL_AUDIT_FAIL",
        "parent_v2a_report_path": str(parent_report.resolve()),
        "parent_v2a_report_sha256": sha(parent_report),
        "parent_v2a_lock_path": str(parent_lock.resolve()),
        "parent_v2a_lock_sha256": sha(parent_lock),
        "local_competence_audit_authorized": True,
        "local_drs_authorized": True,
        "safe_fallback_audit_authorized": True,
        "feature_rich_judge_training_authorized": False,
        "official_valid_authorized": False,
        "test_authorized": False,
        "student_training_authorized": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "outer_evaluation_access_count": 0,
        "student_trained": False,
        "expert_retrained_or_modified": False,
        "raw_hierarchical_consistency_in_main_method": False,
        "optional_consistency_ablation": "pre-check fold identity only; stop if AUROC remains clearly above random",
        "reference_bank": "Direction-specific inner-train Train-OOF rows; same mode; exclude query source; maximum one nearest clip per source",
        "content_space": "Direction-specific block-wise content PCA plus effective availability; compare only jointly visible/effective blocks",
        "decision_space": "five predictions, five within-row prediction ranks, committee std/range/polarity; standardized with inner-train statistics only",
        "hybrid_space": "beta*normalized_content_distance+(1-beta)*normalized_decision_distance",
        "neighbor_ground_truth_rule": "inner-train labels may define historical competence; query label never enters retrieval or prediction",
        "candidates": candidates,
        "promotion_gate": {
            "mean_delta_J_vs_strong_static_max": -0.003,
            "worst_direction_delta_J_max": 0.001,
            "mean_delta_J_vs_strongest_random_or_shuffled_max": -0.002,
            "missing_modes_improved_minimum": 2,
            "corr_no_clear_drop": True,
            "classification_no_systematic_drop": True,
            "high_confidence_trigger_region_must_gain": True,
            "directions_signal_and_gain_consistent": True,
            "not_equivalent_to_global_or_mode_only": True,
            "outer_predictions_from_one_frozen_evaluation": True,
        },
        "possible_conclusions": [
            "LOCAL_COMPETENCE_PASS",
            "LOCAL_COMPETENCE_WEAK",
            "LOCAL_COMPETENCE_FAIL",
        ],
        "failure_action": "permanently close all post-hoc Judge work on this frozen Expert pool; no Official Valid/Test or Student; move to jointly trained structurally specialized Experts",
    }
    protocol_path = V2B / "protocol" / "frozen_protocol_manifest.json"
    candidate_path = V2B / "protocol" / "candidate_table.json"
    write_json(candidate_path, candidates)
    protocol["candidate_table_path"] = str(candidate_path.resolve())
    protocol["candidate_table_sha256"] = sha(candidate_path)
    write_json(protocol_path, protocol)
    state = {
        "stage": "Stage23A-v2b",
        "status": protocol["status"],
        "protocol_manifest_path": str(protocol_path.resolve()),
        "protocol_manifest_sha256": sha(protocol_path),
        "outer_evaluation_access_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "feature_rich_judge_training_authorized": False,
        "student_training_authorized": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(RUNTIME / "state.json", state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
