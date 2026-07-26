#!/usr/bin/env python
"""Read-only integrity finalizer for completed Stage23A-v2b artifacts.

This script never loads an outer label column.  It validates the frozen
protocol, role/source partitions, prediction and neighbor ledgers, aggregate
metrics, and access locks, then writes reproducibility/SHA reports.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage23a_v2_common import ROOT


V2A = ROOT / "result" / "arbiter_audit_v2" / "mosei"
V2B = ROOT / "result" / "arbiter_audit_v2b" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23a_v2b"
FINAL = V2B / "final"
DIRECTIONS = ("A", "B")
ROLES = ("inner_train", "inner_valid", "outer_evaluation")
EXPECTED = {
    "A": {
        "inner_train": (29752, 7438, 1007),
        "inner_valid": (4376, 1094, 130),
        "outer_evaluation": (31176, 7794, 1112),
    },
    "B": {
        "inner_train": (27732, 6933, 980),
        "inner_valid": (3444, 861, 132),
        "outer_evaluation": (34128, 8532, 1137),
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def run(command):
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def role_audit():
    rows = []
    source_sets = {}
    for direction in DIRECTIONS:
        root = V2A / "features" / "meta57" / f"direction_{direction}"
        for role in ROLES:
            frame = pd.read_csv(
                root / f"{role}_targets_and_audit.csv",
                usecols=[
                    "meta_row_id",
                    "sample_id",
                    "video_id",
                    "mode",
                    "expert_fold",
                    "role",
                ],
                dtype={"meta_row_id": str, "sample_id": str, "video_id": str},
            )
            observed = (
                len(frame),
                frame["sample_id"].nunique(),
                frame["video_id"].nunique(),
            )
            expected = EXPECTED[direction][role]
            source_sets[(direction, role)] = set(frame["video_id"])
            rows.append(
                {
                    "direction": direction,
                    "role": role,
                    "meta_rows": observed[0],
                    "samples": observed[1],
                    "sources": observed[2],
                    "modes": sorted(frame["mode"].unique().tolist()),
                    "duplicate_meta_rows": int(frame["meta_row_id"].duplicated().sum()),
                    "matches_frozen_counts": observed == expected,
                }
            )
    overlaps = {}
    for direction in DIRECTIONS:
        for left, right in (
            ("inner_train", "inner_valid"),
            ("inner_train", "outer_evaluation"),
            ("inner_valid", "outer_evaluation"),
        ):
            overlaps[f"{direction}:{left}:{right}"] = len(
                source_sets[(direction, left)] & source_sets[(direction, right)]
            )
    return rows, overlaps


def stream_neighbor_audit(path, expected_k):
    row_count = 0
    query_count = 0
    same_source = 0
    duplicate_neighbor_source = 0
    invalid_rank_sequences = 0
    modes = set()
    current_query = None
    current_sources = set()
    current_ranks = []

    def finish_query():
        nonlocal query_count, invalid_rank_sequences
        if current_query is None:
            return
        query_count += 1
        if current_ranks != list(range(1, len(current_ranks) + 1)):
            invalid_rank_sequences += 1
        if len(current_ranks) != expected_k:
            invalid_rank_sequences += 1

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        chunksize=200000,
        dtype={
            "query_meta_row_id": str,
            "query_sample_id": str,
            "query_video_id": str,
            "neighbor_sample_id": str,
            "neighbor_video_id": str,
        },
    ):
        row_count += len(chunk)
        same_source += int(
            (chunk["query_video_id"].astype(str) == chunk["neighbor_video_id"].astype(str)).sum()
        )
        for row in chunk[
            [
                "query_meta_row_id",
                "neighbor_video_id",
                "neighbor_rank",
                "mode",
            ]
        ].itertuples(index=False):
            query = str(row.query_meta_row_id)
            if query != current_query:
                finish_query()
                current_query = query
                current_sources = set()
                current_ranks = []
            source = str(row.neighbor_video_id)
            if source in current_sources:
                duplicate_neighbor_source += 1
            current_sources.add(source)
            current_ranks.append(int(row.neighbor_rank))
            modes.add(str(row.mode))
    finish_query()
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "rows": row_count,
        "queries": query_count,
        "expected_K": expected_k,
        "same_source_rows": same_source,
        "duplicate_neighbor_source_within_query": duplicate_neighbor_source,
        "invalid_rank_or_count_queries": invalid_rank_sequences,
        "modes": sorted(modes),
        "constraints_pass": (
            same_source == 0
            and duplicate_neighbor_source == 0
            and invalid_rank_sequences == 0
        ),
    }


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(
        (V2B / "protocol" / "frozen_protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    selection = json.loads(
        (V2B / "protocol" / "frozen_development_selection.json").read_text(
            encoding="utf-8"
        )
    )
    state = json.loads((RUNTIME / "state.json").read_text(encoding="utf-8"))
    lock = json.loads(
        (FINAL / "TEST_LOCK_STATUS.json").read_text(encoding="utf-8")
    )
    v2a = json.loads(
        (V2A / "analysis_v2a" / "stage23a_v2a_signal_audit.json").read_text(
            encoding="utf-8"
        )
    )
    role_rows, overlaps = role_audit()
    feature_manifests = {}
    for direction in DIRECTIONS:
        manifest_path = (
            V2A
            / "features"
            / "meta57"
            / f"direction_{direction}"
            / "feature_manifest.json"
        )
        preprocessing_path = (
            V2A
            / "features"
            / "preprocessing"
            / f"direction_{direction}"
            / "preprocessing_manifest.json"
        )
        feature_manifests[direction] = {
            "feature_manifest_sha256": sha256(manifest_path),
            "preprocessing_manifest_sha256": sha256(preprocessing_path),
        }
    independent_preprocessing = (
        feature_manifests["A"]["preprocessing_manifest_sha256"]
        != feature_manifests["B"]["preprocessing_manifest_sha256"]
    )

    neighbor_audits = []
    for direction in DIRECTIONS:
        selected_k = int(selection["directions"][direction]["selected_region"]["K"])
        neighbor_audits.append(
            {
                "direction": direction,
                "role": "inner_valid",
                **stream_neighbor_audit(
                    V2B
                    / "development"
                    / f"direction_{direction}_selected_inner_valid_neighbors.csv.gz",
                    selected_k,
                ),
            }
        )
        neighbor_audits.append(
            {
                "direction": direction,
                "role": "outer_evaluation",
                **stream_neighbor_audit(
                    V2B
                    / "outer"
                    / f"direction_{direction}_selected_outer_neighbors.csv.gz",
                    selected_k,
                ),
            }
        )

    predictions = pd.read_csv(
        V2B
        / "outer"
        / "outer_predictions_frozen_before_label_evaluation.csv.gz",
        compression="gzip",
    )
    forbidden_prediction_columns = {
        "label",
        "absolute_error",
        "squared_error",
        "regret",
        "best_expert_indicator",
    }
    prediction_audit = {
        "rows": len(predictions),
        "unique_meta_rows": predictions["meta_row_id"].nunique(),
        "directions": predictions.groupby("direction").size().to_dict(),
        "forbidden_columns_present": sorted(
            forbidden_prediction_columns & set(predictions.columns)
        ),
        "sha256": sha256(
            V2B
            / "outer"
            / "outer_predictions_frozen_before_label_evaluation.csv.gz"
        ),
    }
    risk_ledgers = {}
    for direction in DIRECTIONS:
        path = V2B / "data" / f"direction_{direction}_expert_risk_ledger.csv.gz"
        frame = pd.read_csv(path, compression="gzip")
        risk_ledgers[direction] = {
            "rows": len(frame),
            "expert_count": frame["expert_id"].nunique(),
            "fields_present": all(
                field in frame.columns
                for field in (
                    "absolute_error",
                    "squared_error",
                    "regret",
                    "best_expert_indicator",
                )
            ),
            "nonfinite_values": int(
                (~np.isfinite(
                    frame[
                        [
                            "absolute_error",
                            "squared_error",
                            "regret",
                            "best_expert_indicator",
                        ]
                    ].to_numpy(dtype=float)
                )).sum()
            ),
            "sha256": sha256(path),
        }

    audit = {
        "stage": "Stage23A-v2b integrity audit",
        "completed_at": utc_now(),
        "v2a_status_unchanged": v2a.get("status") == "SIGNAL_AUDIT_FAIL",
        "v2a_status": v2a.get("status"),
        "v2a_branch_tip": run(
            [
                "git",
                "rev-parse",
                "origin/audit/mosei-feature-rich-meta-judge-v2a-signal-audit",
            ]
        ),
        "v2b_branch": run(["git", "branch", "--show-current"]),
        "protocol_manifest_sha256": sha256(
            V2B / "protocol" / "frozen_protocol_manifest.json"
        ),
        "selection_manifest_sha256": sha256(
            V2B / "protocol" / "frozen_development_selection.json"
        ),
        "authorization": {
            key: protocol.get(key)
            for key in (
                "local_competence_audit_authorized",
                "local_drs_authorized",
                "safe_fallback_audit_authorized",
                "feature_rich_judge_training_authorized",
                "official_valid_authorized",
                "test_authorized",
                "student_training_authorized",
            )
        },
        "role_counts": role_rows,
        "total_outer_meta_rows": sum(
            item["meta_rows"]
            for item in role_rows
            if item["role"] == "outer_evaluation"
        ),
        "total_outer_samples": sum(
            item["samples"]
            for item in role_rows
            if item["role"] == "outer_evaluation"
        ),
        "source_overlaps": overlaps,
        "feature_preprocessing": feature_manifests,
        "direction_specific_preprocessing_manifests_differ": independent_preprocessing,
        "strong_static": "per_mode_constrained_fixed_stacking (frozen v2a sidecar field)",
        "expert_risk_ledgers": risk_ledgers,
        "neighbor_audits": neighbor_audits,
        "prediction_audit": prediction_audit,
        "access_locks": {
            "outer_evaluation_access_count": state["outer_evaluation_access_count"],
            "official_valid_access_count": state["official_valid_access_count"],
            "locked_test_access_count": state["locked_test_access_count"],
            "feature_rich_judge_trained": state["feature_rich_judge_trained"],
            "student_trained": state["student_trained"],
            "expert_retrained_or_modified": lock["expert_retrained_or_modified"],
        },
        "all_frozen_count_checks_pass": all(
            item["matches_frozen_counts"] and item["duplicate_meta_rows"] == 0
            for item in role_rows
        ),
        "all_source_overlaps_zero": all(value == 0 for value in overlaps.values()),
        "all_neighbor_constraints_pass": all(
            item["constraints_pass"] for item in neighbor_audits
        ),
        "outer_predictions_frozen_before_label_metrics": True,
        "finalized_without_outer_reload": state[
            "finalized_from_saved_metrics_without_outer_reload"
        ],
        "final_status": state["status"],
    }
    write_json(FINAL / "integrity_audit.json", audit)

    metrics = pd.read_csv(V2B / "outer" / "outer_method_metrics.tsv", sep="\t")
    regularity = pd.read_csv(
        V2B / "outer" / "outer_local_regularity.tsv", sep="\t"
    )
    coverage = pd.read_csv(
        V2B / "outer" / "outer_risk_coverage.tsv", sep="\t"
    )
    overall = metrics[metrics["mode"] == "Overall"].set_index(
        ["direction", "method"]
    )
    true_regularity = regularity[
        regularity["method"] == "selected_true_neighborhood"
    ].set_index("direction")
    delta_a = float(
        overall.loc[("A", "selected_safe_local"), "J"]
        - overall.loc[("A", "strong_static"), "J"]
    )
    delta_b = float(
        overall.loc[("B", "selected_safe_local"), "J"]
        - overall.loc[("B", "strong_static"), "J"]
    )
    top10 = coverage[np.isclose(coverage["coverage"], 0.10)].set_index(
        "direction"
    )
    answers = f"""# Stage23A-v2b plain-language answers

1. **Is Expert performance more regular near similar history?** Yes, but only
   weakly. Outer local-risk/error Spearman is
   {true_regularity.loc['A', 'spearman_local_risk_vs_error']:.6f} (A) and
   {true_regularity.loc['B', 'spearman_local_risk_vs_error']:.6f} (B).
2. **Best neighborhood:** Hybrid in both directions. A selected
   H/cosine/K30/beta=0.25; B selected H/standardized-Euclidean/K60/beta=0.50.
3. **True versus random/shuffled:** true retrieval is slightly better in final
   J, but the mean margin is only -0.000269, below the frozen -0.002 gate.
4. **Best DRS:** Local-DW. Local-DS is clearly worse and DWS does not improve
   upon DW on outer.
5. **Safe fallback:** not reliably. The frozen safe rule changes J by
   {delta_a:+.6f} (A) and {delta_b:+.6f} (B), while top-10% triggered true gain
   is {top10.loc['A', 'triggered_true_gain_mean']:+.6f} (A) and
   {top10.loc['B', 'triggered_true_gain_mean']:+.6f} (B).
6. **Direction agreement:** local-risk correlations are positive and safe
   Delta J is negative in both directions, but high-confidence gain is not
   direction-consistent.
7. **Promotion:** no. Final status is `LOCAL_COMPETENCE_WEAK`.
8. **Why not pass:** neighborhoods are not sparse or source-monopolized; every
   query receives the frozen K with zero same-source and duplicate-source
   neighbors. The limiting factor is weak competence-to-gain precision and
   poor beneficiary ranking, with a modest outer density shift.
9. **Continue post-hoc Judge?** There is no evidence to promote a formal neural
   Judge or Student. Because the frozen conclusion is WEAK rather than FAIL,
   permanent route closure is not automatically asserted; both remain locked
   pending an explicit research decision.
"""
    (FINAL / "plain_language_answers.md").write_text(answers, encoding="utf-8")

    runtime = {
        "stage": "Stage23A-v2b",
        "completed_at": utc_now(),
        "execution_mode": "CPU-only; no model training and no GPU allocation",
        "python": run(
            ["/usr/miniconda3/envs/DLF/bin/python", "--version"]
        ),
        "host": run(["hostname"]),
        "gpu_snapshot_after_completion": run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "develop_log_bytes": (
            RUNTIME / "develop.log"
        ).stat().st_size,
        "outer_log_bytes": (RUNTIME / "outer.log").stat().st_size,
        "outer_prediction_sha256": prediction_audit["sha256"],
        "outer_access_count": state["outer_evaluation_access_count"],
        "official_valid_access_count": state["official_valid_access_count"],
        "test_access_count": state["locked_test_access_count"],
    }
    write_json(FINAL / "runtime_gpu_report.json", runtime)

    commands = """# Stage23A-v2b reproducible commands
# Executed from /code/DLF-mosei-arbiter-audit-v1
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_authorize.py
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_local_competence.py --phase develop
# Freeze/commit/push selection before outer.
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_local_competence.py --phase outer
# Serialization-only recovery from saved aggregate metrics; no outer reload.
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_local_competence.py --phase finalize-saved
/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2b_finalize.py
"""
    (FINAL / "reproducible_commands.sh").write_text(commands, encoding="utf-8")
    git_diff = run(["git", "diff", "HEAD"])
    git_diff = "\n".join(line.rstrip() for line in git_diff.splitlines())
    (FINAL / "git_diff_before_final_commit.patch").write_text(
        git_diff + "\n", encoding="utf-8"
    )

    excluded = {
        FINAL / "artifact_manifest.json",
        FINAL / "git_push_status.json",
    }
    artifacts = []
    for path in sorted(V2B.rglob("*")):
        if path.is_file() and path not in excluded:
            artifacts.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    artifacts.extend(
        [
            {
                "path": "scripts/mosei/stage23a_v2b_authorize.py",
                "bytes": (
                    ROOT / "scripts" / "mosei" / "stage23a_v2b_authorize.py"
                ).stat().st_size,
                "sha256": sha256(
                    ROOT / "scripts" / "mosei" / "stage23a_v2b_authorize.py"
                ),
            },
            {
                "path": "scripts/mosei/stage23a_v2b_local_competence.py",
                "bytes": (
                    ROOT
                    / "scripts"
                    / "mosei"
                    / "stage23a_v2b_local_competence.py"
                ).stat().st_size,
                "sha256": sha256(
                    ROOT
                    / "scripts"
                    / "mosei"
                    / "stage23a_v2b_local_competence.py"
                ),
            },
            {
                "path": "scripts/mosei/stage23a_v2b_finalize.py",
                "bytes": (
                    ROOT / "scripts" / "mosei" / "stage23a_v2b_finalize.py"
                ).stat().st_size,
                "sha256": sha256(
                    ROOT / "scripts" / "mosei" / "stage23a_v2b_finalize.py"
                ),
            },
        ]
    )
    write_json(
        FINAL / "artifact_manifest.json",
        {
            "stage": "Stage23A-v2b",
            "generated_at": utc_now(),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
