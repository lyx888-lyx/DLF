"""Prepare and audit Stage23A-v2 data without training a Meta-Judge.

Only the MOSEI Train partition and existing frozen Expert OOF products are
used. Official Valid and Test are not indexed or loaded.
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage23a_v2_common import (
    COMPONENTS,
    EXPERTS,
    MODE_ACTIVE_HEADS,
    MODE_AVAILABILITY,
    MODES,
    ROOT,
    RUNTIME_ROOT,
    V1_ROOT,
    V2_ROOT,
    atomic_csv,
    atomic_json,
    git_head,
    metric_rows,
    optimize_simplex,
    overall_j,
    sha256_file,
    sha256_json,
    source_video,
    stable_bucket,
)


DATASET = Path("/data4t/lyx/datasets/MOSEI/Processed/aligned_50.pkl")
EXPECTED_ORACLE_DELTA = -0.15361230650316943
INNER_SPLIT_SEED = 2302
INNER_VALID_MODULO = 8
SHUFFLE_SEEDS = (23021, 23022, 23023)
NOISE_SEEDS = (23121, 23122, 23123)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def repository_runtime_audit():
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=str(ROOT), text=True
    ).splitlines()
    process_text = subprocess.check_output(
        ["ps", "-eo", "pid,ppid,etime,%cpu,%mem,cmd"], text=True
    )
    matched = [
        line
        for line in process_text.splitlines()
        if any(token in line.lower() for token in ("stage23", "mosei", "arbiter"))
        and "stage23a_v2_prepare_protocol" not in line
    ]
    gpu_rows = []
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in output.splitlines():
            values = [value.strip() for value in line.split(",")]
            gpu_rows.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "memory_used_mib": int(values[2]),
                    "memory_total_mib": int(values[3]),
                    "utilization_percent": int(values[4]),
                }
            )
    except (OSError, subprocess.CalledProcessError):
        gpu_rows = []
    pid_files = {}
    for path in sorted((ROOT / "runtime" / "stage23a").glob("*.pid")):
        pid = int(path.read_text().strip())
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        pid_files[str(path.relative_to(ROOT))] = {"pid": pid, "alive": alive}
    return {
        "stage": "Stage23A-v2 protocol audit",
        "created_at": utc_now(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
        ).strip(),
        "head": git_head(),
        "git_status_before_generated_outputs": status,
        "matched_processes": matched,
        "gpu_snapshot": gpu_rows,
        "stage23a_pid_files": pid_files,
        "judge_training_started": False,
        "expert_checkpoint_modified": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
    }


def load_and_validate_long():
    frames = []
    input_rows = []
    protocol_path = V1_ROOT / "protocol" / "preregistered_protocol.json"
    split_path = V1_ROOT / "protocol" / "source_splits.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if tuple(protocol["expert_ids"]) != EXPERTS:
        raise RuntimeError("Frozen Expert pool differs from v1 protocol.")
    if any(
        (
            protocol["official_valid_access_count"],
            protocol["locked_test_access_count"],
            protocol["test_loader_constructed"],
            protocol["student_trained"],
        )
    ):
        raise RuntimeError("v1 protocol data locks are not closed.")
    for fold in (0, 1):
        directory = V1_ROOT / "expert_oof" / "outer_fold{}".format(fold)
        prediction_path = directory / "oof_predictions.csv"
        manifest_path = directory / "fold_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(prediction_path) != manifest["prediction_sha256"]:
            raise RuntimeError("Fold {} prediction SHA mismatch.".format(fold))
        if tuple(manifest["expert_ids"]) != EXPERTS:
            raise RuntimeError("Fold {} Expert pool mismatch.".format(fold))
        if not manifest["source_disjoint"]:
            raise RuntimeError("Fold {} is not source disjoint.".format(fold))
        if any(
            (
                manifest["official_valid_access_count"],
                manifest["locked_test_access_count"],
                manifest["test_loader_constructed"],
                manifest["student_trained"],
            )
        ):
            raise RuntimeError("Fold {} data locks are not closed.".format(fold))
        local = pd.read_csv(
            prediction_path, dtype={"sample_id": str, "video_id": str}
        )
        frames.append(local)
        input_rows.append(
            {
                "outer_fold": fold,
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
                "fold_manifest_path": str(manifest_path),
                "fold_manifest_sha256": sha256_file(manifest_path),
                "rows": int(len(local)),
                "samples": int(local["sample_id"].nunique()),
                "sources": int(local["video_id"].nunique()),
            }
        )
    long = pd.concat(frames, ignore_index=True)
    return long, protocol, protocol_path, split_path, input_rows


def pivot_meta_ledger(long, split_path):
    required = {
        "sample_id",
        "video_id",
        "train_index",
        "outer_fold",
        "mode",
        "expert_id",
        "prediction",
        "label",
    }
    if set(long.columns) != required:
        raise RuntimeError("Unexpected long ledger schema: {}".format(long.columns))
    key = ["sample_id", "mode", "expert_id"]
    if int(long.duplicated(key).sum()) != 0:
        raise RuntimeError("Long ledger duplicates sample/mode/Expert.")
    group_count = long.groupby(["sample_id", "mode"]).expert_id.nunique()
    if not (group_count == len(EXPERTS)).all():
        raise RuntimeError("A sample/mode does not bind exactly five Experts.")
    identity = ["sample_id", "video_id", "train_index", "outer_fold", "mode", "label"]
    wide = long.pivot(index=identity, columns="expert_id", values="prediction").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={expert: "prediction__{}".format(expert) for expert in EXPERTS})
    prediction_columns = ["prediction__{}".format(expert) for expert in EXPERTS]
    if wide[prediction_columns].isna().any().any():
        raise RuntimeError("Pivoted meta ledger has a missing Expert prediction.")
    wide["expert_fold"] = wide.pop("outer_fold").astype(int)
    split = pd.read_csv(split_path, dtype={"sample_id": str, "video_id": str})
    split = split[["sample_id", "judge_fold"]]
    wide = wide.merge(split, on="sample_id", how="left", validate="many_to_one")
    if wide["judge_fold"].isna().any():
        raise RuntimeError("A meta row has no frozen judge-fold binding.")
    wide["judge_fold"] = wide["judge_fold"].astype(int)
    wide["meta_row_id"] = [
        sha256_json([sample_id, mode])
        for sample_id, mode in zip(wide["sample_id"], wide["mode"])
    ]
    wide["content_binding_key"] = wide["sample_id"]
    wide["hierarchical_binding_key"] = wide["meta_row_id"]
    wide = wide.sort_values(["train_index", "mode"], kind="mergesort").reset_index(
        drop=True
    )

    assertions = {
        "long_rows": int(len(long)),
        "expected_long_rows": 326520,
        "meta_rows": int(len(wide)),
        "expected_meta_rows": 65304,
        "unique_sample_mode": int(
            wide[["sample_id", "mode"]].drop_duplicates().shape[0]
        ),
        "duplicate_meta_rows": int(
            wide.duplicated(["sample_id", "mode"]).sum()
        ),
        "missing_expert_predictions": int(wide[prediction_columns].isna().sum().sum()),
        "samples": int(wide["sample_id"].nunique()),
        "sources": int(wide["video_id"].nunique()),
        "modes": sorted(wide["mode"].unique().tolist()),
        "expert_fold0_meta_rows": int((wide["expert_fold"] == 0).sum()),
        "expert_fold1_meta_rows": int((wide["expert_fold"] == 1).sum()),
        "expected_fold0_meta_rows": 34128,
        "expected_fold1_meta_rows": 31176,
        "five_experts_per_sample_mode": bool((group_count == 5).all()),
        "four_modes_per_sample": bool(
            (wide.groupby("sample_id")["mode"].nunique() == 4).all()
        ),
        "one_expert_fold_per_sample": bool(
            (wide.groupby("sample_id")["expert_fold"].nunique() == 1).all()
        ),
        "one_expert_fold_per_source": bool(
            (wide.groupby("video_id")["expert_fold"].nunique() == 1).all()
        ),
        "video_mapping_matches": bool(
            (wide["video_id"] == wide["sample_id"].map(source_video)).all()
        ),
    }
    expected = {
        "long_rows": 326520,
        "meta_rows": 65304,
        "unique_sample_mode": 65304,
        "duplicate_meta_rows": 0,
        "missing_expert_predictions": 0,
        "samples": 16326,
        "expert_fold0_meta_rows": 34128,
        "expert_fold1_meta_rows": 31176,
    }
    for name, value in expected.items():
        if assertions[name] != value:
            raise RuntimeError("Meta assertion failed: {} != {}".format(name, value))
    for name in (
        "five_experts_per_sample_mode",
        "four_modes_per_sample",
        "one_expert_fold_per_sample",
        "one_expert_fold_per_source",
        "video_mapping_matches",
    ):
        if not assertions[name]:
            raise RuntimeError("Meta assertion failed: {}".format(name))
    return wide, prediction_columns, assertions


def crossfit_per_mode_stacking(wide, prediction_columns):
    result = np.zeros(len(wide), dtype=np.float64)
    weight_rows = []
    matrix = wide[prediction_columns].to_numpy(dtype=np.float64)
    for valid_fold in (0, 1):
        fit_fold = wide["judge_fold"].to_numpy() != valid_fold
        for mode in MODES:
            fit = fit_fold & (wide["mode"].to_numpy() == mode)
            valid = (~fit_fold) & (wide["mode"].to_numpy() == mode)
            weights = optimize_simplex(
                matrix[fit], wide.loc[fit, "label"].to_numpy(dtype=np.float64)
            )
            result[valid] = matrix[valid].dot(weights)
            for expert, weight in zip(EXPERTS, weights):
                weight_rows.append(
                    {
                        "validation_judge_fold": valid_fold,
                        "fit_judge_fold": 1 - valid_fold,
                        "mode": mode,
                        "expert_id": expert,
                        "weight": float(weight),
                    }
                )
    return result, pd.DataFrame(weight_rows)


def oracle_audit(wide, prediction_columns):
    # Preserve the v1 pivot/evaluator row order exactly. SLSQP minimizes a
    # non-smooth MAE objective, so a different summation order can move the
    # final simplex solution at roughly 1e-6 even when all rows are identical.
    work = wide.sort_values(
        ["sample_id", "video_id", "mode", "label", "judge_fold"],
        kind="mergesort",
    ).reset_index(drop=True)
    metrics = []
    for expert, column in zip(EXPERTS, prediction_columns):
        metrics.extend(metric_rows(work, expert, column))
    work["equal_average"] = work[prediction_columns].mean(axis=1)
    work["per_mode_fixed_stacking"], weights = crossfit_per_mode_stacking(
        work, prediction_columns
    )
    matrix = work[prediction_columns].to_numpy(dtype=np.float64)
    errors = np.abs(matrix - work["label"].to_numpy(dtype=np.float64)[:, None])
    oracle_index = errors.argmin(axis=1)
    work["oracle_selected_expert"] = [EXPERTS[index] for index in oracle_index]
    work["oracle_expert_selection"] = matrix[np.arange(len(work)), oracle_index]
    for method in (
        "equal_average",
        "per_mode_fixed_stacking",
        "oracle_expert_selection",
    ):
        metrics.extend(metric_rows(work, method, method))
    metrics = pd.DataFrame(metrics)

    overall = metrics.loc[metrics["mode"] == "Overall"].set_index("method")
    best_single = overall.loc[list(EXPERTS), "J"].idxmin()
    deltas = []
    for mode in list(MODES) + ["Overall"]:
        local = metrics.loc[metrics["mode"] == mode].set_index("method")
        for reference in (
            "per_mode_fixed_stacking",
            best_single,
            "equal_average",
        ):
            metric = "J" if mode == "Overall" else "MAE"
            deltas.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "reference": reference,
                    "oracle": float(local.loc["oracle_expert_selection", metric]),
                    "reference_value": float(local.loc[reference, metric]),
                    "oracle_minus_reference": float(
                        local.loc["oracle_expert_selection", metric]
                        - local.loc[reference, metric]
                    ),
                }
            )
    deltas = pd.DataFrame(deltas)
    reproduced = float(
        deltas.loc[
            (deltas["mode"] == "Overall")
            & (deltas["reference"] == "per_mode_fixed_stacking"),
            "oracle_minus_reference",
        ].iloc[0]
    )
    if abs(reproduced - EXPECTED_ORACLE_DELTA) > 1e-10:
        raise RuntimeError(
            "Oracle delta was not reproduced: {} vs {}".format(
                reproduced, EXPECTED_ORACLE_DELTA
            )
        )

    proportions = (
        work["oracle_selected_expert"]
        .value_counts(normalize=False)
        .rename_axis("expert_id")
        .reset_index(name="selected_rows")
    )
    proportions["selection_fraction"] = proportions["selected_rows"] / len(work)
    proportions = proportions.set_index("expert_id").reindex(EXPERTS).reset_index()

    audit_rows = []
    for (fold, mode), group in work.groupby(["expert_fold", "mode"], sort=True):
        selected = group.assign(
            audit_hash=[
                sha256_json(["stage23a_v2_oracle_sample_audit", value, mode])
                for value in group["sample_id"]
            ]
        ).sort_values("audit_hash", kind="mergesort").head(3)
        for _, row in selected.iterrows():
            record = {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "mode": row["mode"],
                "expert_fold": int(row["expert_fold"]),
                "label": float(row["label"]),
            }
            for expert, column in zip(EXPERTS, prediction_columns):
                record["prediction__{}".format(expert)] = float(row[column])
                record["absolute_error__{}".format(expert)] = float(
                    abs(row[column] - row["label"])
                )
            record.update(
                {
                    "oracle_selected_expert": row["oracle_selected_expert"],
                    "fixed_prediction": float(row["per_mode_fixed_stacking"]),
                    "oracle_prediction": float(row["oracle_expert_selection"]),
                }
            )
            audit_rows.append(record)
    sample_audit = pd.DataFrame(audit_rows)
    return work, metrics, weights, deltas, proportions, sample_audit, best_single


def direction_manifests(wide):
    rows = []
    summaries = {}
    for name, development_fold, evaluation_fold in (
        ("A", 0, 1),
        ("B", 1, 0),
    ):
        local_rows = []
        source_counts = (
            wide[["video_id", "expert_fold", "sample_id"]]
            .drop_duplicates()
            .groupby(["video_id", "expert_fold"])
            .size()
            .rename("sample_count")
            .reset_index()
        )
        for _, row in source_counts.iterrows():
            source = row["video_id"]
            fold = int(row["expert_fold"])
            if fold == evaluation_fold:
                role = "outer_evaluation"
            elif stable_bucket(
                source,
                "stage23a_v2_direction_{}_seed{}".format(name, INNER_SPLIT_SEED),
                INNER_VALID_MODULO,
            ) == 0:
                role = "inner_valid"
            else:
                role = "inner_train"
            if fold not in (development_fold, evaluation_fold):
                raise RuntimeError("Unexpected Expert fold.")
            if fold == evaluation_fold and role != "outer_evaluation":
                raise RuntimeError("Outer evaluation source entered development.")
            local_rows.append(
                {
                    "direction": name,
                    "video_id": source,
                    "expert_fold": fold,
                    "role": role,
                    "sample_count": int(row["sample_count"]),
                    "meta_row_count": int(row["sample_count"]) * len(MODES),
                    "inner_split_seed": INNER_SPLIT_SEED,
                    "inner_split_modulo": INNER_VALID_MODULO,
                }
            )
        local = pd.DataFrame(local_rows).sort_values(
            ["role", "video_id"], kind="mergesort"
        )
        by_role = local.groupby("role").agg(
            sources=("video_id", "nunique"),
            samples=("sample_count", "sum"),
            meta_rows=("meta_row_count", "sum"),
        )
        for role in ("inner_train", "inner_valid", "outer_evaluation"):
            if role not in by_role.index or int(by_role.loc[role, "sources"]) == 0:
                raise RuntimeError("Direction {} has empty {}.".format(name, role))
        expected_dev = local.loc[
            local["role"].isin(["inner_train", "inner_valid"]), "expert_fold"
        ].unique()
        expected_eval = local.loc[
            local["role"] == "outer_evaluation", "expert_fold"
        ].unique()
        if expected_dev.tolist() != [development_fold]:
            raise RuntimeError("Direction {} development fold leaked.".format(name))
        if expected_eval.tolist() != [evaluation_fold]:
            raise RuntimeError("Direction {} evaluation fold leaked.".format(name))
        path = V2_ROOT / "protocol" / "direction_{}_sources.csv".format(name)
        atomic_csv(local, path)
        summaries[name] = {
            "development_expert_fold": development_fold,
            "outer_evaluation_expert_fold": evaluation_fold,
            "inner_split_seed": INNER_SPLIT_SEED,
            "inner_valid_rule": (
                "sha256(stage23a_v2_direction_{name}_seed{seed}|video_id) "
                "mod {modulo} == 0"
            ).format(name=name, seed=INNER_SPLIT_SEED, modulo=INNER_VALID_MODULO),
            "source_manifest_path": str(path),
            "source_manifest_sha256": sha256_file(path),
            "counts": {
                role: {
                    key: int(value)
                    for key, value in by_role.loc[role].to_dict().items()
                }
                for role in by_role.index
            },
            "outer_evaluation_used_for_selection": False,
        }
        rows.append(local)
    return summaries


def validate_content_inputs(protocol):
    dataset_sha = sha256_file(DATASET)
    # The monolithic pickle is deserialized by the original project loader, but
    # only its Train key is selected. Valid/Test keys are never indexed.
    with DATASET.open("rb") as handle:
        train = pickle.load(handle)["train"]
    ids = [str(value) for value in train["id"]]
    if sha256_json(ids) != protocol["train_ordered_sample_id_sha256"]:
        raise RuntimeError("Content Train IDs differ from the frozen OOF protocol.")
    expected = {
        "text": (16326, 50, 768),
        "audio": (16326, 50, 74),
        "vision": (16326, 50, 35),
    }
    details = {}
    for modality, shape in expected.items():
        value = np.asarray(train[modality])
        if tuple(value.shape) != shape:
            raise RuntimeError(
                "{} shape differs: {} vs {}".format(modality, value.shape, shape)
            )
        if not np.isfinite(value).all():
            raise RuntimeError("{} contains non-finite values.".format(modality))
        active = np.any(value != 0, axis=2)
        active_count = active.sum(axis=1)
        details[modality] = {
            "train_shape": list(value.shape),
            "dtype": str(value.dtype),
            "temporal_length": int(value.shape[1]),
            "feature_dim": int(value.shape[2]),
            "summary": "padding-aware temporal mean concatenated with population std",
            "summary_dim": int(2 * value.shape[2]),
            "padding_rule": "timestep is active iff any feature value is non-zero",
            "zero_active_timestep_clips": int((active_count == 0).sum()),
            "zero_active_timestep_policy": (
                "emit an all-zero temporal summary; do not borrow another modality"
            ),
            "active_timestep_min": int(active_count.min()),
            "active_timestep_max": int(active_count.max()),
            "active_timestep_mean": float(active_count.mean()),
            "finite": True,
        }
    del train
    gc.collect()
    schema = {
        "dataset_path": str(DATASET),
        "dataset_sha256": dataset_sha,
        "accessed_partition": "train only",
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "fit_unit": "one unique clip; four modes are never repeated during PCA fitting",
        "raw_inputs": details,
        "normalization": (
            "per-modality StandardScaler fit on unique inner-train clips only"
        ),
        "pca": {
            "Text": {
                "source_key": "text",
                "input_dim": details["text"]["summary_dim"],
                "output_dim": 16,
                "solver": "randomized",
                "random_state": INNER_SPLIT_SEED,
            },
            "Audio": {
                "source_key": "audio",
                "input_dim": details["audio"]["summary_dim"],
                "output_dim": 8,
                "solver": "randomized",
                "random_state": INNER_SPLIT_SEED,
            },
            "Vision": {
                "source_key": "vision",
                "input_dim": details["vision"]["summary_dim"],
                "output_dim": 8,
                "solver": "randomized",
                "random_state": INNER_SPLIT_SEED,
            },
        },
        "pca_total_dim": 32,
        "visibility": {
            mode: {
                "availability_mask": list(MODE_AVAILABILITY[mode]),
                "content_blocks": {
                    "Text16": "visible",
                    "Audio8": "visible"
                    if MODE_AVAILABILITY[mode][1]
                    else "zero",
                    "Vision8": "visible"
                    if MODE_AVAILABILITY[mode][2]
                    else "zero",
                },
            }
            for mode in MODES
        },
        "forbidden": [
            "joint PCA across modalities",
            "PCA fit on all Train",
            "PCA fit with four duplicated mode rows per clip",
            "reading unavailable Audio/Vision blocks for LA/LV/L",
            "raw DLF hidden representations from different Expert folds",
        ],
    }
    return schema


def frozen_design(content_schema):
    feature_schema = {
        "name": "Stage23A-v2 main 57D feature schema",
        "status": "FROZEN_BEFORE_JUDGE_TRAINING",
        "blocks": [
            {"name": "mode_one_hot", "dim": 4},
            {"name": "expert_predictions", "dim": 5, "experts": list(EXPERTS)},
            {
                "name": "committee_disagreement",
                "dim": 3,
                "features": ["prediction_std", "prediction_max_min", "polarity_split"],
            },
            {
                "name": "robust_outlier",
                "dim": 5,
                "definition": (
                    "abs(prediction-median_prediction)/"
                    "max(sample_prediction_MAD, mode_inner_train_scale_floor)"
                ),
                "scale_floor": (
                    "max(inner-train mode-specific Q10 of sample prediction MAD, 1e-3)"
                ),
                "clip": [0.0, 10.0],
            },
            {
                "name": "hierarchical_consistency",
                "dim": 5,
                "per_expert": "MAD(final, shared, active mode-specific heads)",
                "mode_active_heads": {
                    mode: list(heads) for mode, heads in MODE_ACTIVE_HEADS.items()
                },
                "ghost_head_policy": "missing-modality specific heads are ignored by explicit mode mapping",
            },
            {
                "name": "content_pca",
                "dim": 32,
                "allocation": {"Text": 16, "Audio": 8, "Vision": 8},
                "schema_path": str(
                    V2_ROOT / "protocol" / "content_feature_schema.json"
                ),
            },
            {
                "name": "availability_mask",
                "dim": 3,
                "mapping": {
                    mode: list(value) for mode, value in MODE_AVAILABILITY.items()
                },
            },
        ],
        "total_dim": 57,
        "optional_not_in_main": {
            "name": "final_to_active_head_median",
            "dim": 5,
            "status": "pre-registered ablation only; cannot replace main configuration",
        },
    }
    if sum(block["dim"] for block in feature_schema["blocks"]) != 57:
        raise RuntimeError("Feature schema is not 57D.")

    controls = {
        "shuffled_content": {
            "seeds": list(SHUFFLE_SEEDS),
            "train_rule": (
                "within each mode, deterministic cross-source derangement of content "
                "on meta inner-train; predictions, labels, and source split unchanged"
            ),
            "inner_valid_outer_rule": (
                "evaluate the Judge learned from shuffled inner-train content on "
                "normally paired transformed content; no evaluation label is used"
            ),
            "report": "mean, variance, and strongest (lowest-J) negative control",
        },
        "matched_random_noise": {
            "seeds": list(NOISE_SEEDS),
            "rule": (
                "57D schema keeps non-content blocks; each content PCA coordinate is "
                "replaced by independent hash-keyed noise matched to inner-train "
                "coordinate mean/std; evaluation noise never uses labels"
            ),
            "report": "mean, variance, and strongest (lowest-J) negative control",
        },
    }
    judge = {
        "status": "FROZEN_NOT_TRAINED",
        "architecture": {
            "input_dim": 57,
            "hidden_dims": [32, 16],
            "activation": "ReLU",
            "dropout": 0.1,
            "dynamic_head": "5-way softmax",
            "actionability_head": "sigmoid",
            "actionability_output_bias": math.log(0.02 / 0.98),
        },
        "seed": INNER_SPLIT_SEED,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "max_epochs": 100,
            "early_stop_patience": 10,
            "selection_metric": "inner-valid J",
        },
        "targets": {
            "regret": "abs(y-p_e)-min_k(abs(y-p_k))",
            "preference": "softmax(-regret/tau)",
            "tau": (
                "clip(median positive regret on inner-train, 0.02, 0.50)"
            ),
            "fixed_advantage": "fixed_error-oracle_select_one_error",
            "actionability": "sigmoid((fixed_advantage-0.005)/tau_action)",
            "tau_action": (
                "clip(IQR(fixed_advantage)/1.349 on inner-train, 0.01, 0.25)"
            ),
        },
        "loss": {
            "prediction": "SmoothL1(beta=1.0)",
            "lambda_preference_kl": 0.1,
            "lambda_actionability_bce": 0.1,
            "lambda_fallback_alpha_squared": 0.05,
        },
        "fit_scope": (
            "all preprocessing, stacking, targets, temperatures, scale floors, "
            "model fitting and early stopping use only the Direction development "
            "side's fixed inner-train/inner-valid; outer evaluation is one-shot"
        ),
    }
    gates = {
        "primary_reference": "per-mode fixed stacking fit on inner-train only",
        "mean_delta_J_max": -0.003,
        "worst_direction_delta_J_max": 0.001,
        "delta_J_vs_strongest_shuffled_content_max": -0.002,
        "missing_modes_with_mean_MAE_improvement_min": 2,
        "corr": {
            "mean_delta_min": -0.002,
            "worst_direction_delta_min": -0.005,
        },
        "classification": {
            "metrics": ["Acc7", "Acc5", "Acc2", "F1"],
            "systematic_degradation_definition": (
                "for any metric, more than half of direction×mode deltas are < -1e-8"
            ),
            "systematic_degradation_allowed": False,
        },
        "directions": "both Direction A and B delta J must be <= 0",
        "dynamic_noncollapse_each_direction": {
            "alpha_std_min": 0.01,
            "fraction_alpha_ge_0p1_min": 0.05,
            "mean_weight_l1_from_fallback_min": 0.01,
        },
        "trigger_actionability_each_direction": (
            "triggered mean actual gain > 0 and > untriggered mean actual gain"
        ),
        "failure_action": (
            "permanently close Judge/Arbiter; do not access Official Valid, "
            "change depth, tune thresholds, train Student, or access Test"
        ),
    }
    return feature_schema, controls, judge, gates


def main():
    if "feature-rich-meta-judge-v2-protocol" not in subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
    ).strip():
        raise RuntimeError("Run on the dedicated Stage23A-v2 protocol branch.")
    audit = repository_runtime_audit()
    atomic_json(V2_ROOT / "audit" / "repository_runtime_audit.json", audit)

    long, protocol, protocol_path, split_path, input_rows = load_and_validate_long()
    wide, prediction_columns, assertions = pivot_meta_ledger(long, split_path)
    meta_path = V2_ROOT / "data" / "meta_ledger.csv"
    atomic_csv(wide, meta_path)
    assertions["meta_ledger_path"] = str(meta_path)
    assertions["meta_ledger_sha256"] = sha256_file(meta_path)
    assertions["schema"] = [
        {"column": column, "dtype": str(wide[column].dtype)}
        for column in wide.columns
    ]
    assertions["status"] = "PASS"
    atomic_json(V2_ROOT / "audit" / "meta_row_assertions.json", assertions)
    atomic_json(
        V2_ROOT / "audit" / "long_to_wide_pivot_report.json",
        {
            "status": "PASS",
            "identity_columns": [
                "sample_id",
                "video_id",
                "train_index",
                "mode",
                "label",
                "expert_fold",
            ],
            "prediction_columns": prediction_columns,
            "binding_columns": [
                "meta_row_id",
                "content_binding_key",
                "hierarchical_binding_key",
            ],
            "assertions_path": str(
                V2_ROOT / "audit" / "meta_row_assertions.json"
            ),
            "meta_ledger_sha256": sha256_file(meta_path),
        },
    )

    (
        oracle_work,
        oracle_metrics,
        fixed_weights,
        oracle_deltas,
        proportions,
        sample_audit,
        best_single,
    ) = oracle_audit(wide, prediction_columns)
    analysis = V2_ROOT / "audit"
    atomic_csv(oracle_metrics, analysis / "oracle_recomputation_metrics.csv")
    atomic_csv(fixed_weights, analysis / "per_mode_fixed_stacking_weights.csv")
    atomic_csv(oracle_deltas, analysis / "oracle_deltas.csv")
    atomic_csv(proportions, analysis / "oracle_selection_proportions.csv")
    atomic_csv(sample_audit, analysis / "oracle_sample_audit.csv")
    overall_delta = float(
        oracle_deltas.loc[
            (oracle_deltas["mode"] == "Overall")
            & (oracle_deltas["reference"] == "per_mode_fixed_stacking"),
            "oracle_minus_reference",
        ].iloc[0]
    )
    oracle_report = {
        "status": "PASS",
        "definition": (
            "For each unique (sample_id, mode), select exactly one scalar prediction "
            "with minimum absolute error from the five frozen Experts in that same mode."
        ),
        "forbidden_operations_absent": {
            "free_convex_combination": True,
            "ground_truth_interpolation": True,
            "cross_mode_selection": True,
            "clean_component_inclusion": True,
            "official_valid_or_test_labels": True,
        },
        "same_evaluator_for_all_methods": True,
        "J_definition": (
            "0.5*MAE(LAV)+0.5*mean(MAE(LA),MAE(LV),MAE(L))"
        ),
        "best_single": best_single,
        "expected_oracle_minus_fixed_J": EXPECTED_ORACLE_DELTA,
        "recomputed_oracle_minus_fixed_J": overall_delta,
        "absolute_difference_from_expected": abs(
            overall_delta - EXPECTED_ORACLE_DELTA
        ),
        "sample_audit_rows": int(len(sample_audit)),
        "metrics_path": str(analysis / "oracle_recomputation_metrics.csv"),
        "weights_path": str(analysis / "per_mode_fixed_stacking_weights.csv"),
        "deltas_path": str(analysis / "oracle_deltas.csv"),
        "selection_path": str(analysis / "oracle_selection_proportions.csv"),
        "sample_audit_path": str(analysis / "oracle_sample_audit.csv"),
    }
    atomic_json(analysis / "oracle_recomputation_report.json", oracle_report)

    directions = direction_manifests(wide)
    content_schema = validate_content_inputs(protocol)
    atomic_json(V2_ROOT / "protocol" / "content_feature_schema.json", content_schema)
    preprocessing_scope = {
        "status": "FROZEN_BEFORE_JUDGE_TRAINING",
        "fit_unit": "one unique clip/source-bound meta row; content PCA sees each clip once",
        "directions": directions,
        "inner_train_fit_only": [
            "per-modality temporal-summary normalization",
            "per-modality PCA",
            "per-mode fixed simplex stacking",
            "robust outlier scale floor",
            "soft-regret temperature",
            "actionability temperature",
            "feature selection (none in main configuration)",
            "Judge model parameters",
        ],
        "inner_valid_use_only": [
            "early stopping",
            "single pre-registered epoch selection",
        ],
        "outer_evaluation_use_only": [
            "one-shot transform with frozen inner-train preprocessors",
            "one-shot prediction",
            "metrics after predictions are frozen",
        ],
        "outer_evaluation_forbidden": [
            "normalization or PCA fit",
            "fixed stacking fit",
            "temperature/lambda/dropout/weight-decay selection",
            "scale-floor or threshold selection",
            "feature selection",
            "seed or epoch selection",
        ],
        "source_grouping": (
            "all clips and all four modes from one video_id remain in one role"
        ),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
    }
    atomic_json(
        V2_ROOT / "protocol" / "preprocessing_fit_scope.json",
        preprocessing_scope,
    )
    feature_schema, controls, judge, gates = frozen_design(content_schema)
    atomic_json(V2_ROOT / "protocol" / "feature_schema.json", feature_schema)
    atomic_json(V2_ROOT / "protocol" / "random_controls.json", controls)
    atomic_json(V2_ROOT / "protocol" / "frozen_gate_criteria.json", gates)
    atomic_json(V2_ROOT / "protocol" / "judge_main_configuration.json", judge)
    atomic_json(
        V2_ROOT / "protocol" / "hierarchical_head_mapping.json",
        {
            "status": "FROZEN_BEFORE_REPLAY",
            "per_expert_output_dim": 1,
            "definition": "MAD(final, shared, active mode-specific heads)",
            "mode_active_heads": {
                mode: list(heads) for mode, heads in MODE_ACTIVE_HEADS.items()
            },
            "missing_specific_head_policy": (
                "explicitly excluded by mode even if a non-zero ghost logit exists"
            ),
            "replay_first_step": (
                "only final prediction is replayed for checkpoint consistency; "
                "hierarchical features may be extracted only after the <=1e-5 gate"
            ),
        },
    )

    pool_rows = []
    for fold in (0, 1):
        directory = V1_ROOT / "expert_oof" / "outer_fold{}".format(fold)
        for component in COMPONENTS:
            group = "components" if component.startswith("clean") else "experts"
            component_dir = directory / group / component
            run_manifest_path = component_dir / "run_manifest.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            checkpoint = Path(run_manifest["checkpoint"])
            if sha256_file(checkpoint) != run_manifest["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint SHA mismatch: {}".format(component))
            pool_rows.append(
                {
                    "expert_fold": fold,
                    "component": component,
                    "role": "base_component"
                    if component.startswith("clean")
                    else "committee_expert",
                    "committee_member": component in EXPERTS,
                    "run_manifest_path": str(run_manifest_path),
                    "run_manifest_sha256": sha256_file(run_manifest_path),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "best_epoch": int(run_manifest["best_epoch"]),
                    "best_inner_train_only_J": float(
                        run_manifest["best_inner_train_only_J"]
                    ),
                }
            )
    pool = pd.DataFrame(pool_rows)
    pool_path = V2_ROOT / "protocol" / "frozen_expert_pool.csv"
    atomic_csv(pool, pool_path)

    draft = {
        "stage": "Stage23A-v2 Feature-Rich OOF Meta-Judge protocol validation",
        "status": "AWAITING_CHECKPOINT_REPLAY",
        "created_at": utc_now(),
        "code_commit_before_protocol_outputs": git_head(),
        "inputs": {
            "dataset_path": str(DATASET),
            "dataset_sha256": content_schema["dataset_sha256"],
            "accessed_dataset_partition": "train only",
            "v1_protocol_path": str(protocol_path),
            "v1_protocol_sha256": sha256_file(protocol_path),
            "v1_source_split_path": str(split_path),
            "v1_source_split_sha256": sha256_file(split_path),
            "outer_folds": input_rows,
            "frozen_expert_pool_path": str(pool_path),
            "frozen_expert_pool_sha256": sha256_file(pool_path),
        },
        "meta_ledger": {
            "path": str(meta_path),
            "sha256": sha256_file(meta_path),
            "rows": int(len(wide)),
            "assertion_status": "PASS",
        },
        "oracle": oracle_report,
        "directions": directions,
        "content_feature_schema_path": str(
            V2_ROOT / "protocol" / "content_feature_schema.json"
        ),
        "feature_schema_path": str(V2_ROOT / "protocol" / "feature_schema.json"),
        "hierarchical_mapping_path": str(
            V2_ROOT / "protocol" / "hierarchical_head_mapping.json"
        ),
        "random_controls_path": str(
            V2_ROOT / "protocol" / "random_controls.json"
        ),
        "gate_criteria_path": str(
            V2_ROOT / "protocol" / "frozen_gate_criteria.json"
        ),
        "judge_configuration_path": str(
            V2_ROOT / "protocol" / "judge_main_configuration.json"
        ),
        "preprocessing_fit_scope_path": str(
            V2_ROOT / "protocol" / "preprocessing_fit_scope.json"
        ),
        "replay_gate": {
            "preferred_max_abs_diff": 1e-6,
            "hard_stop_max_abs_diff": 1e-5,
            "hierarchical_extraction_allowed_before_pass": False,
        },
        "judge_training_started": False,
        "expert_checkpoint_modified": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
    }
    atomic_json(V2_ROOT / "protocol" / "draft_protocol_manifest.json", draft)
    atomic_json(
        RUNTIME_ROOT / "state.json",
        {
            "stage": "Stage23A-v2 protocol validation",
            "phase": "AWAITING_CHECKPOINT_REPLAY",
            "status": "PROTOCOL_AND_DATA_RECONSTRUCTION_PASS",
            "judge_training_started": False,
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
                "status": draft["status"],
                "meta_rows": len(wide),
                "oracle_delta_J": overall_delta,
                "direction_counts": directions,
                "dataset_sha256": content_schema["dataset_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
