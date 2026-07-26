#!/usr/bin/env python
"""Freeze the Stage23A-v2a feature/signal-audit authorization.

This creates an immutable child manifest.  The Stage23A-v2 protocol manifest is
kept byte-for-byte unchanged and is referenced by path and SHA256.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from stage23a_v2_common import RUNTIME_ROOT, V2_ROOT, atomic_json, git_head, sha256_file


def main():
    protocol_dir = V2_ROOT / "protocol"
    parent_path = protocol_dir / "frozen_protocol_manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_sha = sha256_file(parent_path)
    if parent_sha != "ba4226714ba0f19e7ec75715a68891f6baaf61ef7d1829ba422d2d81d2bcfa3f":
        raise RuntimeError("Frozen parent manifest SHA changed: {}".format(parent_sha))
    if parent.get("locked_test_access_count") != 0:
        raise RuntimeError("Test lock count is not zero.")

    strong_static = {
        "name": "strong_static",
        "candidate_methods": ["equal_average", "per_mode_constrained_fixed_stacking"],
        "stacking_objective": "per-mode inner-train MAE under nonnegative simplex weights summing to one",
        "selection_scope": "for each Direction, choose one candidate using inner-valid J only",
        "application_scope": "freeze selected candidate and weights before one-shot outer evaluation",
        "outer_evaluation_used_for_selection": False,
        "gain_definition": "oracle_expert_selection_J - strong_static_J; negative is better",
        "experts": [
            "uniform_kd_seed1111",
            "moddrop_seed1111",
            "moddrop_seed1114",
            "cfcompat_seed1111",
            "cfcompat_seed1114",
        ],
    }
    effective_availability = {
        "all_zero_vision_clip_count_expected": 482,
        "vision_effective_available_rule": "vision has at least one finite nonzero timestep",
        "all_zero_vision_policy": {
            "vision_effective_available": 0,
            "raw_vision_summary": "all zeros",
            "vision_scaler_pca_fit": "exclude ineffective clips",
            "vision_pca_after_transform": "force all zeros",
            "effective_availability_mask": "vision bit is mode availability AND vision_effective_available",
            "mode_one_hot": "unchanged",
        },
        "text_audio_policy": "same effective-availability rule; expected no all-zero Train clips",
    }
    authorization = {
        "stage": "Stage23A-v2a Feature Extraction and Signal Audit",
        "status": "AUTHORIZED_FOR_FEATURE_EXTRACTION_AND_SIGNAL_AUDIT_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_at_authorization": git_head(),
        "parent_frozen_manifest_path": str(parent_path.resolve()),
        "parent_frozen_manifest_sha256": parent_sha,
        "implementation_authorized": True,
        "feature_extraction_authorized": True,
        "signal_audit_authorized": True,
        "judge_training_authorized": False,
        "official_valid_authorized": False,
        "test_authorized": False,
        "student_training_authorized": False,
        "judge_training_started": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "student_trained": False,
        "expert_checkpoint_modified": False,
        "strong_static_protocol": strong_static,
        "effective_availability_protocol": effective_availability,
        "permitted_actions": [
            "frozen Expert hierarchical-head extraction",
            "Train-only fixed input content summaries",
            "Direction-specific Train-only normalization and PCA",
            "frozen 57D meta-feature construction",
            "leakage-free signal audits and simple probes",
        ],
        "forbidden_actions": [
            "formal dual-head Judge training",
            "Official Valid access",
            "Test access",
            "Student training",
            "Expert checkpoint modification or retraining",
            "feature or hyperparameter selection using outer evaluation",
        ],
        "next_action": "extract frozen hierarchical outputs and Train-only content summaries",
    }
    auth_path = protocol_dir / "v2a_authorization_manifest.json"
    static_path = protocol_dir / "strong_static_protocol.json"
    availability_path = protocol_dir / "effective_availability_protocol.json"
    atomic_json(static_path, strong_static)
    atomic_json(availability_path, effective_availability)
    authorization["strong_static_protocol_path"] = str(static_path.resolve())
    authorization["strong_static_protocol_sha256"] = sha256_file(static_path)
    authorization["effective_availability_protocol_path"] = str(availability_path.resolve())
    authorization["effective_availability_protocol_sha256"] = sha256_file(availability_path)
    atomic_json(auth_path, authorization)

    state = {
        "stage": "Stage23A-v2a",
        "status": authorization["status"],
        "authorization_manifest_path": str(auth_path.resolve()),
        "authorization_manifest_sha256": sha256_file(auth_path),
        "implementation_authorized": True,
        "feature_extraction_authorized": True,
        "signal_audit_authorized": True,
        "judge_training_authorized": False,
        "official_valid_authorized": False,
        "test_authorized": False,
        "student_training_authorized": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RUNTIME_ROOT / "state.json", state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
