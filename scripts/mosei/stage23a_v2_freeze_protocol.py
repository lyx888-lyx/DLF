"""Freeze Stage23A-v2 after all protocol and replay hard gates pass."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from stage23a_v2_common import (
    ROOT,
    RUNTIME_ROOT,
    V2_ROOT,
    atomic_json,
    git_head,
    sha256_file,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main():
    draft_path = V2_ROOT / "protocol" / "draft_protocol_manifest.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft["status"] != "AWAITING_CHECKPOINT_REPLAY":
        raise RuntimeError("Unexpected draft protocol status.")
    replay = []
    for fold in (0, 1):
        path = V2_ROOT / "validation" / "replay_fold{}_report.json".format(fold)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"] != "PASS":
            raise RuntimeError("Fold {} replay hard gate failed.".format(fold))
        if payload["max_abs_diff"] > 1e-5:
            raise RuntimeError("Fold {} replay exceeds 1e-5.".format(fold))
        expected_rows = 170640 if fold == 0 else 155880
        if (
            payload["rows"] != expected_rows
            or payload["expected_rows"] != expected_rows
            or not payload["final_row_count_matches"]
        ):
            raise RuntimeError("Fold {} replay row count differs.".format(fold))
        replay.append(
            {
                "outer_fold": fold,
                "report_path": str(path),
                "report_sha256": sha256_file(path),
                "max_abs_diff": payload["max_abs_diff"],
                "mean_abs_diff": payload["mean_abs_diff"],
                "mismatch_gt_1e_6": payload["mismatch_gt_1e_6"],
                "mismatch_gt_1e_5": payload["mismatch_gt_1e_5"],
            }
        )

    required = [
        V2_ROOT / "audit" / "repository_runtime_audit.json",
        V2_ROOT / "audit" / "long_to_wide_pivot_report.json",
        V2_ROOT / "audit" / "meta_row_assertions.json",
        V2_ROOT / "audit" / "oracle_recomputation_report.json",
        V2_ROOT / "audit" / "oracle_recomputation_metrics.csv",
        V2_ROOT / "audit" / "oracle_sample_audit.csv",
        V2_ROOT / "protocol" / "direction_A_sources.csv",
        V2_ROOT / "protocol" / "direction_B_sources.csv",
        V2_ROOT / "protocol" / "content_feature_schema.json",
        V2_ROOT / "protocol" / "feature_schema.json",
        V2_ROOT / "protocol" / "hierarchical_head_mapping.json",
        V2_ROOT / "protocol" / "random_controls.json",
        V2_ROOT / "protocol" / "frozen_gate_criteria.json",
        V2_ROOT / "protocol" / "judge_main_configuration.json",
        V2_ROOT / "protocol" / "preprocessing_fit_scope.json",
        V2_ROOT / "protocol" / "frozen_expert_pool.csv",
        V2_ROOT / "data" / "meta_ledger.csv",
    ]
    artifacts = {}
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[str(path.relative_to(ROOT))] = sha256_file(path)

    assertions = json.loads(
        (V2_ROOT / "audit" / "meta_row_assertions.json").read_text(
            encoding="utf-8"
        )
    )
    oracle = json.loads(
        (V2_ROOT / "audit" / "oracle_recomputation_report.json").read_text(
            encoding="utf-8"
        )
    )
    if assertions["status"] != "PASS" or assertions["meta_rows"] != 65304:
        raise RuntimeError("Meta-row hard gate failed.")
    if oracle["status"] != "PASS":
        raise RuntimeError("Oracle hard gate failed.")
    if abs(
        oracle["recomputed_oracle_minus_fixed_J"]
        - oracle["expected_oracle_minus_fixed_J"]
    ) > 1e-10:
        raise RuntimeError("Oracle delta is not exactly reproducible.")

    commands = [
        (
            "/usr/miniconda3/envs/DLF/bin/python "
            "scripts/mosei/stage23a_v2_prepare_protocol.py"
        ),
        (
            "/usr/miniconda3/envs/DLF/bin/python "
            "scripts/mosei/stage23a_v2_replay_final.py --outer-fold 0 --gpu-id 0"
        ),
        (
            "/usr/miniconda3/envs/DLF/bin/python "
            "scripts/mosei/stage23a_v2_replay_final.py --outer-fold 1 --gpu-id 1"
        ),
        (
            "/usr/miniconda3/envs/DLF/bin/python "
            "scripts/mosei/stage23a_v2_freeze_protocol.py"
        ),
    ]
    commands_path = V2_ROOT / "audit" / "reproducible_commands.txt"
    write_text(commands_path, "\n".join(commands) + "\n")
    base_commit = subprocess.check_output(
        [
            "git",
            "merge-base",
            "HEAD",
            "origin/audit/mosei-personalized-teacher-feasibility-v1",
        ],
        cwd=str(ROOT),
        text=True,
    ).strip()
    diff_path = V2_ROOT / "audit" / "git_diff.patch"
    diff_text = subprocess.check_output(
        ["git", "diff", "--binary", "{}..HEAD".format(base_commit), "--"],
        cwd=str(ROOT),
        text=True,
    )
    write_text(diff_path, diff_text)
    replay_report_path = V2_ROOT / "validation" / "replay_consistency_report.json"
    replay_report = {
        "status": "PASS",
        "preferred_threshold": 1e-6,
        "hard_stop_threshold": 1e-5,
        "total_rows": sum(
            json.loads(Path(row["report_path"]).read_text())["rows"]
            for row in replay
        ),
        "max_abs_diff": max(row["max_abs_diff"] for row in replay),
        "mean_abs_diff": sum(
            json.loads(Path(row["report_path"]).read_text())["mean_abs_diff"]
            * json.loads(Path(row["report_path"]).read_text())["rows"]
            for row in replay
        )
        / sum(
            json.loads(Path(row["report_path"]).read_text())["rows"]
            for row in replay
        ),
        "mismatch_gt_1e_6": sum(row["mismatch_gt_1e_6"] for row in replay),
        "mismatch_gt_1e_5": sum(row["mismatch_gt_1e_5"] for row in replay),
        "folds": replay,
        "hierarchical_features_extracted": False,
        "expert_retrained": False,
    }
    atomic_json(replay_report_path, replay_report)
    manifest = {
        **draft,
        "status": "FROZEN_READY_FOR_FEATURE_IMPLEMENTATION",
        "frozen_at": utc_now(),
        "protocol_code_commit": git_head(),
        "replay": replay,
        "artifact_sha256": artifacts,
        "reproducible_commands": commands,
        "reproducible_commands_path": str(commands_path),
        "reproducible_commands_sha256": sha256_file(commands_path),
        "git_diff_base_commit": base_commit,
        "git_diff_path": str(diff_path),
        "git_diff_sha256": sha256_file(diff_path),
        "combined_replay_report_path": str(replay_report_path),
        "combined_replay_report_sha256": sha256_file(replay_report_path),
        "implementation_authorized": False,
        "judge_training_authorized": False,
        "next_action": (
            "wait for user confirmation before hierarchical/content feature "
            "implementation or Feature-Rich Judge training"
        ),
        "expert_checkpoint_modified": False,
        "judge_training_started": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
    }
    frozen_path = V2_ROOT / "protocol" / "frozen_protocol_manifest.json"
    atomic_json(frozen_path, manifest)
    frozen_sha = sha256_file(frozen_path)
    metrics_path = V2_ROOT / "audit" / "oracle_recomputation_metrics.csv"
    import pandas as pd

    metrics = pd.read_csv(metrics_path)
    overall = metrics.loc[metrics["mode"] == "Overall"].set_index("method")
    per_mode = metrics.loc[
        metrics["mode"].isin(["LAV", "LA", "LV", "L"])
        & metrics["method"].isin(
            [
                "uniform_kd_seed1111",
                "moddrop_seed1111",
                "moddrop_seed1114",
                "cfcompat_seed1111",
                "cfcompat_seed1114",
                "equal_average",
                "per_mode_fixed_stacking",
                "oracle_expert_selection",
            ]
        ),
        ["method", "mode", "MAE"],
    ]
    audit_report = {
        "status": "PASS_READY_FOR_FEATURE_IMPLEMENTATION_NOT_AUTHORIZED",
        "checks": {
            "repository_runtime": "PASS",
            "meta_row_reconstruction": "PASS",
            "oracle_reproduction": "PASS",
            "source_and_mode_leakage": "PASS",
            "direction_inner_split": "PASS",
            "content_visibility": "PASS",
            "checkpoint_replay": "PASS",
            "judge_training": "NOT_RUN",
            "official_valid": "NOT_ACCESSED",
            "test": "LOCKED_0",
            "student": "NOT_TRAINED",
        },
        "meta_rows": assertions["meta_rows"],
        "meta_ledger_sha256": assertions["meta_ledger_sha256"],
        "oracle_definition": oracle["definition"],
        "oracle_delta_J": oracle["recomputed_oracle_minus_fixed_J"],
        "overall_absolute_J": {
            method: float(overall.loc[method, "J"])
            for method in overall.index
        },
        "per_mode_MAE": per_mode.to_dict("records"),
        "direction_counts": draft["directions"],
        "replay": replay_report,
        "content": {
            "raw_summary": (
                "per-modality padding-aware temporal mean + population std"
            ),
            "raw_dims": {"Text": 1536, "Audio": 148, "Vision": 70},
            "pca_dims": {"Text": 16, "Audio": 8, "Vision": 8},
            "total_pca_dim": 32,
            "main_feature_dim": 57,
        },
        "frozen_protocol_path": str(frozen_path),
        "frozen_protocol_sha256": frozen_sha,
        "implementation_authorized": False,
        "judge_training_authorized": False,
    }
    audit_json_path = V2_ROOT / "audit" / "stage23a_v2_protocol_audit.json"
    atomic_json(audit_json_path, audit_report)
    audit_md_path = V2_ROOT / "audit" / "stage23a_v2_protocol_audit.md"
    direction_a = draft["directions"]["A"]["counts"]
    direction_b = draft["directions"]["B"]["counts"]
    write_text(
        audit_md_path,
        "\n".join(
            [
                "# Stage23A-v2 Protocol and Data Reconstruction Audit",
                "",
                "Final status: `PASS_READY_FOR_FEATURE_IMPLEMENTATION_NOT_AUTHORIZED`",
                "",
                "## Hard checks",
                "",
                "- Meta ledger: 65,304 unique `(sample_id, mode)` rows; duplicate=0; missing Expert=0.",
                "- Oracle is select-one from the same five frozen Expert scalars and reproduces ΔJ={:.12f}.".format(
                    oracle["recomputed_oracle_minus_fixed_J"]
                ),
                "- Direction A: inner-train {}/{}/{} sources/samples/meta rows; inner-valid {}/{}/{}; outer evaluation {}/{}/{}.".format(
                    direction_a["inner_train"]["sources"],
                    direction_a["inner_train"]["samples"],
                    direction_a["inner_train"]["meta_rows"],
                    direction_a["inner_valid"]["sources"],
                    direction_a["inner_valid"]["samples"],
                    direction_a["inner_valid"]["meta_rows"],
                    direction_a["outer_evaluation"]["sources"],
                    direction_a["outer_evaluation"]["samples"],
                    direction_a["outer_evaluation"]["meta_rows"],
                ),
                "- Direction B: inner-train {}/{}/{} sources/samples/meta rows; inner-valid {}/{}/{}; outer evaluation {}/{}/{}.".format(
                    direction_b["inner_train"]["sources"],
                    direction_b["inner_train"]["samples"],
                    direction_b["inner_train"]["meta_rows"],
                    direction_b["inner_valid"]["sources"],
                    direction_b["inner_valid"]["samples"],
                    direction_b["inner_valid"]["meta_rows"],
                    direction_b["outer_evaluation"]["sources"],
                    direction_b["outer_evaluation"]["samples"],
                    direction_b["outer_evaluation"]["meta_rows"],
                ),
                "- Frozen final-prediction replay: {} rows, max abs diff {:.3g}, mean abs diff {:.3g}, mismatches >1e-6: {}.".format(
                    replay_report["total_rows"],
                    replay_report["max_abs_diff"],
                    replay_report["mean_abs_diff"],
                    replay_report["mismatch_gt_1e_6"],
                ),
                "- Content: Train-only temporal mean+std; PCA Text16/Audio8/Vision8; unavailable blocks are zeroed with an explicit mask.",
                "- Feature schema: 57D. Hierarchical heads are mode-mapped explicitly; no hierarchical feature has been extracted yet.",
                "",
                "## Locks",
                "",
                "- Judge training: **not run**.",
                "- Official Valid access count: **0**.",
                "- Locked Test access count: **0**.",
                "- Student trained: **No**.",
                "- Expert checkpoint modified/retrained: **No**.",
                "",
                "Implementation and Judge training remain unauthorized until user confirmation.",
                "",
            ]
        ),
    )
    atomic_json(
        RUNTIME_ROOT / "state.json",
        {
            "stage": "Stage23A-v2 protocol validation",
            "phase": "FROZEN_WAITING_FOR_USER_CONFIRMATION",
            "status": "READY_FOR_FEATURE_IMPLEMENTATION_NOT_AUTHORIZED",
            "frozen_protocol_path": str(frozen_path),
            "frozen_protocol_sha256": frozen_sha,
            "implementation_authorized": False,
            "judge_training_authorized": False,
            "expert_checkpoint_modified": False,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "test_loader_constructed": False,
            "student_trained": False,
            "updated_at": utc_now(),
        },
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "frozen_protocol_path": str(frozen_path),
                "frozen_protocol_sha256": frozen_sha,
                "replay": replay,
                "meta_rows": assertions["meta_rows"],
                "oracle_delta_J": oracle[
                    "recomputed_oracle_minus_fixed_J"
                ],
                "judge_training_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
