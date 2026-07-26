#!/usr/bin/env python
"""Independent artifact/lock verification for the completed v2a audit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stage23a_v2_common import EXPERTS, RUNTIME_ROOT, V2_ROOT, atomic_json, sha256_file


def main():
    assertions = []

    def check(name, condition, detail):
        assertions.append({"assertion": name, "pass": bool(condition), "detail": detail})
        if not condition:
            raise RuntimeError("{}: {}".format(name, detail))

    parent = V2_ROOT / "protocol" / "frozen_protocol_manifest.json"
    check(
        "parent_manifest_sha_unchanged",
        sha256_file(parent)
        == "ba4226714ba0f19e7ec75715a68891f6baaf61ef7d1829ba422d2d81d2bcfa3f",
        sha256_file(parent),
    )
    authorization = json.loads(
        (V2_ROOT / "protocol" / "v2a_authorization_manifest.json").read_text()
    )
    check("feature_authorized", authorization["feature_extraction_authorized"], "true")
    for key in (
        "judge_training_authorized",
        "official_valid_authorized",
        "test_authorized",
        "student_training_authorized",
    ):
        check("{}_locked".format(key), not authorization[key], str(authorization[key]))

    hierarchical_rows = 0
    hierarchical_keys = []
    fold_sample_sets = []
    for fold, expected in ((0, 170640), (1, 155880)):
        manifest_path = (
            V2_ROOT
            / "features"
            / "hierarchical"
            / "hierarchical_fold{}_manifest.json".format(fold)
        )
        manifest = json.loads(manifest_path.read_text())
        hierarchical_rows += int(manifest["rows"])
        check("fold{}_hierarchical_status".format(fold), manifest["status"] == "PASS", manifest["status"])
        check(
            "fold{}_preferred_replay".format(fold),
            manifest["max_abs_final_replay_diff"] <= 1e-6,
            str(manifest["max_abs_final_replay_diff"]),
        )
        check("fold{}_rows".format(fold), manifest["rows"] == expected, str(manifest["rows"]))
        check("fold{}_duplicates".format(fold), manifest["duplicates"] == 0, str(manifest["duplicates"]))
        check("fold{}_missing".format(fold), manifest["missing_values"] == 0, str(manifest["missing_values"]))
        check(
            "fold{}_artifact_sha".format(fold),
            sha256_file(manifest["output_path"]) == manifest["output_sha256"],
            manifest["output_sha256"],
        )
        key_frame = pd.read_csv(
            manifest["output_path"],
            usecols=["sample_id", "mode", "expert_id"],
            dtype={"sample_id": str},
        )
        hierarchical_keys.append(key_frame)
        fold_sample_sets.append(set(key_frame["sample_id"]))
    check("hierarchical_total_rows", hierarchical_rows == 326520, str(hierarchical_rows))
    combined_keys = pd.concat(hierarchical_keys, ignore_index=True)
    check(
        "hierarchical_unique_keys",
        not combined_keys.duplicated(["sample_id", "mode", "expert_id"]).any(),
        str(int(combined_keys.duplicated(["sample_id", "mode", "expert_id"]).sum())),
    )
    check(
        "hierarchical_all_samples",
        combined_keys["sample_id"].nunique() == 16326,
        str(combined_keys["sample_id"].nunique()),
    )
    check(
        "hierarchical_fold_sample_overlap_zero",
        len(fold_sample_sets[0] & fold_sample_sets[1]) == 0,
        str(len(fold_sample_sets[0] & fold_sample_sets[1])),
    )

    content = json.loads(
        (
            V2_ROOT
            / "features"
            / "content_raw"
            / "content_summary_manifest.json"
        ).read_text()
    )
    check("content_train_only", content["accessed_partitions"] == ["train"], str(content["accessed_partitions"]))
    check(
        "vision_ineffective_482",
        content["effective_availability"]["vision_ineffective"] == 482,
        str(content["effective_availability"]["vision_ineffective"]),
    )

    for direction, counts in (
        ("A", {"inner_train": 29752, "inner_valid": 4376, "outer_evaluation": 31176}),
        ("B", {"inner_train": 27732, "inner_valid": 3444, "outer_evaluation": 34128}),
    ):
        root = V2_ROOT / "features" / "meta57" / "direction_{}".format(direction)
        manifest = json.loads((root / "feature_manifest.json").read_text())
        check(
            "direction_{}_labels_not_inputs".format(direction),
            manifest["labels_sources_folds_absent_from_X"],
            "true",
        )
        check(
            "direction_{}_vision_zero".format(direction),
            manifest["ineffective_vision_feature_block_forced_zero"],
            "true",
        )
        for role, expected in counts.items():
            values = np.load(root / "{}_features_57d.npz".format(role))
            check(
                "{}_{}_shape".format(direction, role),
                values["X"].shape == (expected, 57),
                str(values["X"].shape),
            )
            check(
                "{}_{}_finite".format(direction, role),
                bool(np.isfinite(values["X"]).all()),
                "finite",
            )

    report = json.loads(
        (V2_ROOT / "analysis_v2a" / "stage23a_v2a_signal_audit.json").read_text()
    )
    check(
        "signal_status_valid",
        report["status"]
        in ("SIGNAL_AUDIT_PASS", "SIGNAL_AUDIT_WEAK", "SIGNAL_AUDIT_FAIL"),
        report["status"],
    )
    lock = json.loads(
        (V2_ROOT / "final_v2a" / "TEST_LOCK_STATUS.json").read_text()
    )
    check("locked_test_zero", lock["locked_test_access_count"] == 0, str(lock["locked_test_access_count"]))
    check("official_valid_zero", lock["official_valid_access_count"] == 0, str(lock["official_valid_access_count"]))
    check("student_not_trained", not lock["student_trained"], str(lock["student_trained"]))
    check("judge_not_started", not lock["judge_training_started"], str(lock["judge_training_started"]))

    verification = {
        "stage": "Stage23A-v2a independent artifact verification",
        "status": "PASS",
        "assertion_count": len(assertions),
        "assertions": assertions,
        "locked_test_access_count": 0,
        "official_valid_access_count": 0,
    }
    path = V2_ROOT / "analysis_v2a" / "verification_report.json"
    atomic_json(path, verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
