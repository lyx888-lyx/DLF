#!/usr/bin/env python3
"""Freeze Stage23D-A assets, source splits, candidates, and authorizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage23d_self_risk_common import (
    ACTIVE_HEADS,
    EXPERTS,
    FOLDS,
    HEADS,
    LEGAL_SUBMODES,
    MODES,
    OUT,
    PCA_DIMS,
    ROOT,
    RUNTIME,
    SHUFFLE_SEEDS,
    SOURCE_ROOT,
    SPLIT_SEED,
    V1_SOURCE,
    V2B_LOCAL,
    V2_LOCAL,
    atomic_csv,
    atomic_json,
    atomic_tsv,
    expert_directory,
    git_branch,
    git_head,
    hierarchical_path,
    oof_path,
    sha256_file,
    sha256_json,
    source_stratified_split,
    utc_now,
)


BASE_COMMIT = "99d18cc49c52b799c83a65af2ed308264d8c649c"
BRANCH = "audit/mosei-expert-self-risk-signal-v1"


def audit_assets():
    rows = []
    replay_rows = []
    binding_rows = []
    for fold in FOLDS:
        oof = oof_path(fold)
        fold_manifest = oof.parent / "fold_manifest.json"
        fold_data = json.loads(fold_manifest.read_text(encoding="utf-8"))
        if fold_data["prediction_sha256"] != sha256_file(oof):
            raise RuntimeError(f"OOF SHA mismatch fold {fold}")
        frame = pd.read_csv(oof)
        expected_samples = 8532 if fold == 0 else 7794
        if (
            len(frame) != expected_samples * len(MODES) * len(EXPERTS)
            or frame.duplicated(["sample_id", "mode", "expert_id"]).any()
        ):
            raise RuntimeError(f"OOF binding failure fold {fold}")
        hierarchical = hierarchical_path(fold)
        hframe = pd.read_csv(
            hierarchical,
            usecols=["sample_id", "mode", "expert_id", "output_logit"],
        )
        joined = frame.merge(
            hframe,
            on=["sample_id", "mode", "expert_id"],
            validate="one_to_one",
        )
        difference = np.abs(
            joined["prediction"].to_numpy(dtype=float)
            - joined["output_logit"].to_numpy(dtype=float)
        )
        binding_rows.append(
            {
                "checkpoint_fold": fold,
                "oof_rows": len(frame),
                "hierarchical_rows": len(hframe),
                "joined_rows": len(joined),
                "duplicates": int(
                    frame.duplicated(["sample_id", "mode", "expert_id"]).sum()
                ),
                "missing": int(joined["output_logit"].isna().sum()),
                "max_abs_oof_hierarchical_diff": float(difference.max()),
                "binding_pass": bool(difference.max() <= 1e-6),
            }
        )
        replay_report_path = (
            V2_LOCAL / "validation" / f"replay_fold{fold}_report.json"
        )
        replay = json.loads(replay_report_path.read_text(encoding="utf-8"))
        replay_rows.append(
            {
                "checkpoint_fold": fold,
                "status": replay["status"],
                "max_abs_diff": replay["max_abs_diff"],
                "mismatch_gt_1e_6": replay["mismatch_gt_1e_6"],
                "rows": replay["rows"],
                "report_path": str(replay_report_path.resolve()),
                "report_sha256": sha256_file(replay_report_path),
            }
        )
        if replay["status"] != "PASS" or replay["max_abs_diff"] > 1e-6:
            raise RuntimeError(f"Frozen replay gate failed fold {fold}")
        for expert in EXPERTS:
            directory = expert_directory(fold, expert)
            manifest_path = directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = Path(manifest["checkpoint"])
            checkpoint_sha = sha256_file(checkpoint)
            if checkpoint_sha != manifest["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint SHA mismatch {fold}/{expert}")
            rows.append(
                {
                    "expert_id": expert,
                    "checkpoint_fold": fold,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "run_manifest_path": str(manifest_path.resolve()),
                    "run_manifest_sha256": sha256_file(manifest_path),
                    "best_epoch": manifest["best_epoch"],
                    "best_inner_train_only_J": manifest[
                        "best_inner_train_only_J"
                    ],
                    "training_side": f"all Train sources except outer_fold{fold}",
                    "oof_evaluation_side": f"outer_fold{fold}",
                    "official_valid_access_count": manifest[
                        "official_valid_access_count"
                    ],
                    "locked_test_access_count": manifest[
                        "locked_test_access_count"
                    ],
                    "oof_path": str(oof.resolve()),
                    "oof_sha256": sha256_file(oof),
                    "hierarchical_path": str(hierarchical.resolve()),
                    "hierarchical_sha256": sha256_file(hierarchical),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(replay_rows), pd.DataFrame(binding_rows)


def build_splits():
    split_source = pd.read_csv(
        V1_SOURCE / "protocol" / "source_splits.csv",
        dtype={"sample_id": str, "video_id": str},
    )
    meta = pd.read_csv(
        V2_LOCAL / "data" / "meta_ledger.csv",
        usecols=["sample_id", "video_id", "label", "expert_fold"],
    ).drop_duplicates("sample_id")
    records = []
    source_records = []
    summaries = []
    for fold in FOLDS:
        local = meta.loc[meta["expert_fold"].astype(int) == fold].copy()
        roles = source_stratified_split(local, SPLIT_SEED + fold)
        roles.insert(0, "checkpoint_fold", fold)
        source_records.append(roles)
        samples = split_source.loc[
            split_source["outer_fold"].astype(int) == fold
        ].merge(
            local[["sample_id", "label"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        samples = samples.merge(
            roles[["video_id", "self_risk_role"]],
            on="video_id",
            how="left",
            validate="many_to_one",
        )
        samples.insert(0, "checkpoint_fold", fold)
        records.append(samples)
        sets = {
            role: set(
                roles.loc[roles["self_risk_role"] == role, "video_id"]
            )
            for role in ("inner_train", "inner_valid", "outer")
        }
        if sets["inner_train"] & sets["inner_valid"] or (
            sets["inner_train"] | sets["inner_valid"]
        ) & sets["outer"]:
            raise RuntimeError(f"Source leakage fold {fold}")
        for role in ("inner_train", "inner_valid", "outer"):
            selected = samples.loc[samples["self_risk_role"] == role]
            summaries.append(
                {
                    "checkpoint_fold": fold,
                    "role": role,
                    "sources": selected["video_id"].nunique(),
                    "samples": selected["sample_id"].nunique(),
                    "label_mean": selected["label"].mean(),
                    "label_std": selected["label"].std(),
                }
            )
    return (
        pd.concat(source_records, ignore_index=True),
        pd.concat(records, ignore_index=True),
        pd.DataFrame(summaries),
    )


def main():
    if git_head() != BASE_COMMIT or git_branch() != BRANCH:
        raise RuntimeError("Stage23D-A worktree branch/base mismatch")
    if not SOURCE_ROOT.exists():
        raise RuntimeError("Read-only Stage23C source worktree is missing")
    stage23c_head = git_head(SOURCE_ROOT)
    assets, replay, bindings = audit_assets()
    sources, samples, split_summary = build_splits()
    if not bindings["binding_pass"].all():
        raise RuntimeError("OOF/hierarchical binding failed")
    protocol_dir = OUT / "protocol"
    audit_dir = OUT / "audit"
    atomic_tsv(assets, audit_dir / "checkpoint_audit.tsv")
    atomic_tsv(replay, audit_dir / "prediction_replay_audit.tsv")
    atomic_tsv(bindings, audit_dir / "oof_hierarchical_binding_audit.tsv")
    atomic_csv(sources, protocol_dir / "source_split_manifest.csv")
    atomic_csv(samples, protocol_dir / "sample_split_manifest.csv")
    atomic_tsv(split_summary, protocol_dir / "source_split_summary.tsv")
    split_sha = sha256_file(protocol_dir / "source_split_manifest.csv")
    sample_split_sha = sha256_file(protocol_dir / "sample_split_manifest.csv")
    candidate_table = {
        "stage": "Stage23D-A Phase-1 preregistered candidates",
        "split_seed": SPLIT_SEED,
        "source_split_ratio": [0.70, 0.15, 0.15],
        "pca_dimensions": list(PCA_DIMS),
        "ridge_alpha": [0.1, 1.0, 10.0],
        "logistic_C": [0.1, 1.0, 10.0],
        "nonlinear_probe": {
            "algorithm": "HistGradientBoosting",
            "max_depth": 3,
            "max_iter": 100,
            "learning_rate": 0.05,
        },
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "fixed_bad_error_threshold": 1.0,
        "bad20": "inner-train expert/checkpoint/mode abs_error top 20%",
        "bad10": "inner-train expert/checkpoint/mode abs_error top 10%",
        "confident": "fixed raw uncertainty inner-train bottom 30%",
        "confident_wrong": "confident AND bad20",
        "raw_uncertainty": (
            "mean of inner-train standardized active_head_std, "
            "submode_prediction_variance, hidden_centroid_distance"
        ),
        "risk_models": {
            "R0": "global and per-mode inner-train priors",
            "R1": "A0 output-only Ridge/LogisticRegression",
            "R2": "A0+A1+A2+A3+A4 Ridge/LogisticRegression and one fixed HGB",
            "R3": "R2+A5; unauthorized until Phase-1 WEAK/PASS",
        },
        "phase1_models": ["R0", "R1", "R2"],
        "negative_controls": [
            "N0 shuffled internal state, three seeds",
            "N1 matched Gaussian",
            "N2 length/mask only",
            "N3 output-only",
            "N4 checkpoint identity scalar audit",
            "N5 prediction magnitude/sign/mode prior",
        ],
        "outer_selection_forbidden": True,
    }
    candidate_path = protocol_dir / "candidate_table.json"
    atomic_json(candidate_path, candidate_table)
    hook_schema = {
        "mechanism": "read-only forward_pre_hook; no model source modification",
        "modules": {
            "out_layer": "final fused representation used by final scalar head",
            "out_layer_c": "shared fused representation",
            "out_layer_l_high": "language-specific representation",
            "out_layer_a_high": "audio-specific representation when active",
            "out_layer_v_high": "vision-specific representation when active",
        },
        "existing_forward_outputs": [
            "fusion_input",
            "specific_hidden_l/a/v",
            "c_l/c_a/c_v",
            "c_l_sim/c_a_sim/c_v_sim",
            "lfa_l/lfa_a/lfa_v",
            "lfa_cross_a/lfa_cross_v",
            "reconstruction tensors",
        ],
        "not_exposed_not_fabricated": [
            "modality gate weights",
            "raw attention matrices",
            "explicit fusion coefficients",
        ],
        "inactive_specific_heads_are_omitted": True,
    }
    atomic_json(protocol_dir / "hook_schema.json", hook_schema)
    protocol = {
        "stage": "Stage23D-A Expert Self-Risk Signal Audit",
        "status": "FROZEN_READY_FOR_CPU_IMPLEMENTATION_GPU_EXTRACTION_LOCKED",
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "created_at": utc_now(),
        "frozen_parent_conclusions": {
            "Stage23A-v2a": "SIGNAL_AUDIT_FAIL",
            "Stage23A-v2b": "LOCAL_COMPETENCE_WEAK",
        },
        "stage23c_conclusion": "NOT_FROZEN_NOT_USED",
        "stage23c_main_worktree": str(SOURCE_ROOT),
        "stage23c_head_at_freeze": stage23c_head,
        "stage23c_modification_authorized": False,
        "stage23c_modification_count": 0,
        "expert_ids": list(EXPERTS),
        "checkpoint_audit_count": len(assets),
        "checkpoint_replications": 2,
        "prior_frozen_replay_gate": {
            "status": "PASS",
            "max_abs_diff": float(replay["max_abs_diff"].max()),
            "threshold": 1e-6,
        },
        "source_split_sha256": split_sha,
        "sample_split_sha256": sample_split_sha,
        "source_split_seed": SPLIT_SEED,
        "active_heads": {key: list(value) for key, value in ACTIVE_HEADS.items()},
        "legal_submodes": {key: list(value) for key, value in LEGAL_SUBMODES.items()},
        "internal_state_extraction_authorized": True,
        "self_risk_signal_audit_authorized": True,
        "static_feature_probe_authorized": True,
        "limited_perturbation_probe_authorized": "conditional",
        "formal_arbiter_training_authorized": False,
        "expert_selection_authorized": False,
        "student_training_authorized": False,
        "expert_retraining_authorized": False,
        "official_valid_authorized": False,
        "test_authorized": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "outer_access_count_per_risk_head": 0,
        "phase1_gate": {
            "mean_error_spearman": 0.25,
            "worst_checkpoint_fold_mean_spearman": 0.15,
            "expert_configs_positive_both_replications": "at least 4/5",
            "bad20_mean_auroc": 0.63,
            "confident_wrong_mean_auroc": 0.63,
            "R2_beats_R1": True,
            "R2_beats_strongest_shuffle": True,
            "missing_modes_positive": "at least 2/3",
            "risk_coverage_monotonic": True,
            "not_mode_length_prediction_only": True,
        },
        "possible_conclusions": [
            "SELF_RISK_SIGNAL_PASS",
            "SELF_RISK_SIGNAL_WEAK",
            "SELF_RISK_SIGNAL_FAIL",
            "READY_WAITING_FOR_FREE_GPU",
        ],
        "candidate_table_path": str(candidate_path.resolve()),
        "candidate_table_sha256": sha256_file(candidate_path),
        "hook_schema_path": str(
            (protocol_dir / "hook_schema.json").resolve()
        ),
        "hook_schema_sha256": sha256_file(protocol_dir / "hook_schema.json"),
    }
    protocol_path = protocol_dir / "frozen_protocol_manifest.json"
    atomic_json(protocol_path, protocol)
    state = {
        "stage": "Stage23D-A",
        "status": "PROTOCOL_FROZEN_GPU_AVAILABILITY_NOT_YET_CHECKED",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "stage23c_modification_count": 0,
        "updated_at": utc_now(),
    }
    atomic_json(RUNTIME / "state.json", state)
    print(
        json.dumps(
            {
                "protocol_path": str(protocol_path),
                "protocol_sha256": sha256_file(protocol_path),
                "checkpoint_audits": len(assets),
                "split_sources": len(sources),
                "split_samples": len(samples),
                "max_replay_diff": replay["max_abs_diff"].max(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
