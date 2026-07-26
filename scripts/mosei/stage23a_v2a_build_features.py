#!/usr/bin/env python
"""Fit Direction-specific preprocessors and construct the frozen 57D ledger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from stage23a_v2_common import (
    EXPERTS,
    MODE_AVAILABILITY,
    MODES,
    V2_ROOT,
    atomic_csv,
    atomic_json,
    optimize_simplex,
    overall_j,
    sha256_file,
    sha256_json,
)


PREDICTION_COLUMNS = ["prediction__{}".format(expert) for expert in EXPERTS]
PCA_DIMENSIONS = {"text": 16, "audio": 8, "vision": 8}
RAW_DIMENSIONS = {"text": 1536, "audio": 148, "vision": 70}
ROLE_NAMES = ("inner_train", "inner_valid", "outer_evaluation")


def load_content():
    root = V2_ROOT / "features" / "content_raw"
    metadata = pd.read_csv(
        root / "content_summary_ledger.csv",
        dtype={"sample_id": str, "video_id": str},
    ).sort_values("train_index")
    matrices = {}
    for modality in ("text", "audio", "vision"):
        mean = np.load(
            root / "{}_temporal_mean_float32.npy".format(modality),
            allow_pickle=False,
        )
        std = np.load(
            root / "{}_temporal_std_float32.npy".format(modality),
            allow_pickle=False,
        )
        matrices[modality] = np.concatenate([mean, std], axis=1)
        if matrices[modality].shape != (16326, RAW_DIMENSIONS[modality]):
            raise RuntimeError("Raw summary shape mismatch {}.".format(modality))
    return metadata, matrices


def load_consistency():
    paths = [
        V2_ROOT
        / "features"
        / "hierarchical"
        / "hierarchical_logits_fold{}.csv.gz".format(fold)
        for fold in (0, 1)
    ]
    pieces = [
        pd.read_csv(
            path,
            usecols=[
                "sample_id",
                "mode",
                "expert_id",
                "hierarchical_consistency_mad",
            ],
            dtype={"sample_id": str},
        )
        for path in paths
    ]
    long = pd.concat(pieces, ignore_index=True)
    if len(long) != 326520:
        raise RuntimeError("Hierarchical row count mismatch.")
    wide = long.pivot(
        index=["sample_id", "mode"],
        columns="expert_id",
        values="hierarchical_consistency_mad",
    )
    wide = wide.reindex(columns=list(EXPERTS))
    wide.columns = ["consistency__{}".format(value) for value in wide.columns]
    return wide.reset_index()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def fit_preprocessors(direction, role_by_video, content_meta, content):
    train_videos = {
        video
        for video, role in role_by_video.items()
        if role == "inner_train"
    }
    fit_mask = content_meta["video_id"].isin(train_videos).to_numpy()
    if not fit_mask.any():
        raise RuntimeError("No inner-train clips for Direction {}.".format(direction))
    output_dir = V2_ROOT / "features" / "preprocessing" / "direction_{}".format(
        direction
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed = {}
    details = {}
    for modality in ("text", "audio", "vision"):
        effective = (
            content_meta["{}_effective_available".format(modality)]
            .astype(int)
            .to_numpy()
            == 1
        )
        modality_fit = fit_mask & effective
        fit_values = np.asarray(content[modality][modality_fit], dtype=np.float64)
        scaler = StandardScaler().fit(fit_values)
        pca = PCA(
            n_components=PCA_DIMENSIONS[modality],
            svd_solver="randomized",
            random_state=2302,
        ).fit(scaler.transform(fit_values))
        all_values = pca.transform(
            scaler.transform(np.asarray(content[modality], dtype=np.float64))
        )
        all_values[~effective] = 0.0
        transformed[modality] = all_values.astype(np.float32)
        scaler_path = output_dir / "{}_scaler.joblib".format(modality)
        pca_path = output_dir / "{}_pca.joblib".format(modality)
        joblib.dump(scaler, scaler_path, compress=3)
        joblib.dump(pca, pca_path, compress=3)
        details[modality] = {
            "fit_clips": int(modality_fit.sum()),
            "fit_sources": int(
                content_meta.loc[modality_fit, "video_id"].nunique()
            ),
            "ineffective_clips_excluded_from_fit": int(
                (fit_mask & ~effective).sum()
            ),
            "input_dim": RAW_DIMENSIONS[modality],
            "output_dim": PCA_DIMENSIONS[modality],
            "input_feature_ordering": "temporal_mean raw coordinates 0..D-1, then population_std raw coordinates 0..D-1",
            "normalization": "StandardScaler fitted on effective inner-train unique clips",
            "pca_solver": "randomized",
            "pca_random_state": 2302,
            "explained_variance_ratio": [
                float(value) for value in pca.explained_variance_ratio_
            ],
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "scaler_path": str(scaler_path.resolve()),
            "scaler_sha256": sha256_file(scaler_path),
            "pca_path": str(pca_path.resolve()),
            "pca_sha256": sha256_file(pca_path),
            "finite_transformed": bool(np.isfinite(all_values).all()),
            "ineffective_transformed_forced_zero": bool(
                np.all(all_values[~effective] == 0)
            ),
        }
    fit_rows = content_meta.loc[
        fit_mask, ["sample_id", "video_id", "train_index"]
    ].sort_values("train_index")
    fit_path = output_dir / "ordered_inner_train_fit_clips.csv"
    atomic_csv(fit_rows, fit_path)
    report = {
        "direction": direction,
        "fit_scope": "unique inner-train clips only",
        "fit_samples": int(fit_mask.sum()),
        "fit_sources": int(len(train_videos)),
        "ordered_fit_sample_id_sha256": sha256_json(
            fit_rows["sample_id"].tolist()
        ),
        "fit_ledger_path": str(fit_path.resolve()),
        "fit_ledger_sha256": sha256_file(fit_path),
        "modalities": details,
        "outer_evaluation_used_for_fit_or_selection": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
    }
    report_path = output_dir / "preprocessing_manifest.json"
    atomic_json(report_path, report)
    return transformed, report


def fit_strong_static(direction, frame):
    weights = {}
    for mode in MODES:
        local = frame.loc[
            (frame["role"] == "inner_train") & (frame["mode"] == mode)
        ]
        weights[mode] = optimize_simplex(
            local[PREDICTION_COLUMNS].to_numpy(dtype=np.float64),
            local["label"].to_numpy(dtype=np.float64),
        )
    output = frame.copy()
    output["equal_average"] = output[PREDICTION_COLUMNS].mean(axis=1)
    output["per_mode_constrained_fixed_stacking"] = np.nan
    for mode in MODES:
        mask = output["mode"] == mode
        output.loc[mask, "per_mode_constrained_fixed_stacking"] = (
            output.loc[mask, PREDICTION_COLUMNS].to_numpy(dtype=np.float64)
            @ weights[mode]
        )
    valid = output.loc[output["role"] == "inner_valid"]
    valid_j = {
        candidate: overall_j(valid, candidate)
        for candidate in (
            "equal_average",
            "per_mode_constrained_fixed_stacking",
        )
    }
    selected = min(valid_j, key=lambda value: (valid_j[value], value))
    output["strong_static"] = output[selected]
    weight_rows = []
    for mode in MODES:
        for expert, value in zip(EXPERTS, weights[mode]):
            weight_rows.append(
                {
                    "direction": direction,
                    "mode": mode,
                    "expert_id": expert,
                    "weight": float(value),
                }
            )
    return output, weights, valid_j, selected, pd.DataFrame(weight_rows)


def construct(direction, meta, consistency, content_meta, content_pca):
    source_manifest = pd.read_csv(
        V2_ROOT / "protocol" / "direction_{}_sources.csv".format(direction),
        dtype={"video_id": str},
    )
    role_by_video = source_manifest.set_index("video_id")["role"].to_dict()
    frame = meta.copy()
    frame["role"] = frame["video_id"].map(role_by_video)
    if frame["role"].isna().any():
        raise RuntimeError("Direction role binding missing.")
    frame = frame.merge(
        consistency, on=["sample_id", "mode"], how="left", validate="one_to_one"
    )
    if frame.filter(like="consistency__").isna().any().any():
        raise RuntimeError("Hierarchical consistency binding missing.")
    frame, weights, valid_j, selected, weight_frame = fit_strong_static(
        direction, frame
    )

    predictions = frame[PREDICTION_COLUMNS].to_numpy(dtype=np.float64)
    prediction_median = np.median(predictions, axis=1)
    prediction_mad = np.median(
        np.abs(predictions - prediction_median[:, None]), axis=1
    )
    floors = {}
    for mode in MODES:
        mask = (frame["role"] == "inner_train") & (frame["mode"] == mode)
        floors[mode] = max(float(np.quantile(prediction_mad[mask], 0.10)), 1e-3)

    content_index = content_meta.set_index("sample_id")
    position_by_id = pd.Series(
        np.arange(len(content_meta)), index=content_meta["sample_id"]
    ).to_dict()
    positions = np.asarray(
        [position_by_id[value] for value in frame["sample_id"]], dtype=np.int64
    )
    feature_names = (
        ["mode__{}".format(mode) for mode in MODES]
        + ["prediction__{}".format(expert) for expert in EXPERTS]
        + ["disagreement__std", "disagreement__range", "disagreement__polarity_split"]
        + ["outlier__{}".format(expert) for expert in EXPERTS]
        + ["consistency__{}".format(expert) for expert in EXPERTS]
        + ["content__text_pc{:02d}".format(value + 1) for value in range(16)]
        + ["content__audio_pc{:02d}".format(value + 1) for value in range(8)]
        + ["content__vision_pc{:02d}".format(value + 1) for value in range(8)]
        + ["effective_available__text", "effective_available__audio", "effective_available__vision"]
    )
    if len(feature_names) != 57:
        raise RuntimeError("Frozen feature-name count is not 57.")
    X = np.zeros((len(frame), 57), dtype=np.float32)
    cursor = 0
    for mode in MODES:
        X[:, cursor] = (frame["mode"].to_numpy() == mode).astype(np.float32)
        cursor += 1
    X[:, cursor : cursor + 5] = predictions.astype(np.float32)
    cursor += 5
    X[:, cursor] = predictions.std(axis=1, ddof=0)
    X[:, cursor + 1] = predictions.max(axis=1) - predictions.min(axis=1)
    positive = (predictions > 0).sum(axis=1)
    nonpositive = 5 - positive
    X[:, cursor + 2] = np.minimum(positive, nonpositive) / 5.0
    cursor += 3
    row_floors = np.asarray([floors[mode] for mode in frame["mode"]])
    denominators = np.maximum(prediction_mad, row_floors)
    X[:, cursor : cursor + 5] = np.clip(
        np.abs(predictions - prediction_median[:, None]) / denominators[:, None],
        0.0,
        10.0,
    ).astype(np.float32)
    cursor += 5
    consistency_columns = ["consistency__{}".format(expert) for expert in EXPERTS]
    X[:, cursor : cursor + 5] = frame[consistency_columns].to_numpy(
        dtype=np.float32
    )
    cursor += 5

    base_availability = np.asarray(
        [MODE_AVAILABILITY[mode] for mode in frame["mode"]], dtype=np.int8
    )
    clip_effective = content_index.loc[
        frame["sample_id"],
        [
            "text_effective_available",
            "audio_effective_available",
            "vision_effective_available",
        ],
    ].to_numpy(dtype=np.int8)
    effective = base_availability * clip_effective
    for modality, width, effective_col in (
        ("text", 16, 0),
        ("audio", 8, 1),
        ("vision", 8, 2),
    ):
        block = content_pca[modality][positions].copy()
        block[effective[:, effective_col] == 0] = 0.0
        X[:, cursor : cursor + width] = block
        cursor += width
    X[:, cursor : cursor + 3] = effective.astype(np.float32)
    cursor += 3
    if cursor != 57 or not np.isfinite(X).all():
        raise RuntimeError("57D feature construction failed.")
    forbidden_tokens = ("label", "source", "video", "fold", "train_index")
    if any(
        any(token in name.lower() for token in forbidden_tokens)
        for name in feature_names
    ):
        raise RuntimeError("Forbidden field leaked into feature names.")

    absolute_error = np.abs(predictions - frame["label"].to_numpy()[:, None])
    oracle_index = np.argmin(absolute_error, axis=1)
    frame["oracle_expert_index"] = oracle_index
    frame["oracle_expert_id"] = [EXPERTS[value] for value in oracle_index]
    frame["oracle_prediction"] = predictions[np.arange(len(frame)), oracle_index]
    frame["strong_static_absolute_error"] = np.abs(
        frame["strong_static"] - frame["label"]
    )
    frame["oracle_absolute_error"] = absolute_error.min(axis=1)
    frame["oracle_gain"] = (
        frame["strong_static_absolute_error"] - frame["oracle_absolute_error"]
    )

    output_dir = V2_ROOT / "features" / "meta57" / "direction_{}".format(direction)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    for role in ROLE_NAMES:
        mask = frame["role"].to_numpy() == role
        path = output_dir / "{}_features_57d.npz".format(role)
        save_npz(
            path,
            X=X[mask],
            feature_names=np.asarray(feature_names, dtype="U64"),
            meta_row_id=frame.loc[mask, "meta_row_id"].to_numpy(dtype="U64"),
        )
        sidecar = frame.loc[
            mask,
            [
                "meta_row_id",
                "sample_id",
                "video_id",
                "train_index",
                "mode",
                "label",
                "expert_fold",
                "role",
                "equal_average",
                "per_mode_constrained_fixed_stacking",
                "strong_static",
                "oracle_expert_index",
                "oracle_expert_id",
                "oracle_prediction",
                "strong_static_absolute_error",
                "oracle_absolute_error",
                "oracle_gain",
            ],
        ]
        sidecar_path = output_dir / "{}_targets_and_audit.csv".format(role)
        atomic_csv(sidecar, sidecar_path)
        feature_rows.append(
            {
                "direction": direction,
                "role": role,
                "rows": int(mask.sum()),
                "feature_dim": 57,
                "feature_path": str(path.resolve()),
                "feature_sha256": sha256_file(path),
                "sidecar_path": str(sidecar_path.resolve()),
                "sidecar_sha256": sha256_file(sidecar_path),
                "finite": bool(np.isfinite(X[mask]).all()),
            }
        )
    weight_path = output_dir / "strong_static_weights.csv"
    atomic_csv(weight_frame, weight_path)
    schema_path = output_dir / "feature_names.json"
    atomic_json(
        schema_path,
        {
            "total_dim": 57,
            "feature_names": feature_names,
            "polarity_split_definition": "min(number predictions >0, number predictions <=0)/5",
            "labels_sources_and_folds_are_inputs": False,
        },
    )
    report = {
        "direction": direction,
        "status": "PASS",
        "rows": int(len(frame)),
        "feature_dim": 57,
        "feature_files": feature_rows,
        "feature_schema_path": str(schema_path.resolve()),
        "feature_schema_sha256": sha256_file(schema_path),
        "outlier_mode_q10_floor": floors,
        "outlier_clip": [0.0, 10.0],
        "strong_static": {
            "candidate_valid_J": valid_j,
            "selected_by_inner_valid_only": selected,
            "weights_path": str(weight_path.resolve()),
            "weights_sha256": sha256_file(weight_path),
            "outer_evaluation_used_for_selection": False,
        },
        "all_zero_vision_rows": int(
            (
                content_index.loc[
                    frame["sample_id"], "vision_effective_available"
                ].to_numpy(dtype=int)
                == 0
            ).sum()
        ),
        "ineffective_vision_feature_block_forced_zero": bool(
            np.all(
                X[
                    content_index.loc[
                        frame["sample_id"], "vision_effective_available"
                    ].to_numpy(dtype=int)
                    == 0,
                    46:54,
                ]
                == 0
            )
        ),
        "labels_sources_folds_absent_from_X": True,
        "outer_evaluation_used_for_fit_or_selection": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "judge_training_started": False,
        "student_trained": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output_dir / "feature_manifest.json", report)
    return report


def main():
    authorization = json.loads(
        (V2_ROOT / "protocol" / "v2a_authorization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not authorization["feature_extraction_authorized"]:
        raise RuntimeError("Feature extraction not authorized.")
    meta = pd.read_csv(
        V2_ROOT / "data" / "meta_ledger.csv",
        dtype={"sample_id": str, "video_id": str},
    )
    if len(meta) != 65304:
        raise RuntimeError("Meta ledger row count mismatch.")
    consistency = load_consistency()
    content_meta, raw_content = load_content()
    reports = {}
    preprocessing_reports = {}
    for direction in ("A", "B"):
        source_manifest = pd.read_csv(
            V2_ROOT / "protocol" / "direction_{}_sources.csv".format(direction),
            dtype={"video_id": str},
        )
        role_by_video = source_manifest.set_index("video_id")["role"].to_dict()
        content_pca, prep_report = fit_preprocessors(
            direction, role_by_video, content_meta, raw_content
        )
        preprocessing_reports[direction] = prep_report
        reports[direction] = construct(
            direction, meta, consistency, content_meta, content_pca
        )
        print(
            "Direction {} complete; strong_static={}".format(
                direction,
                reports[direction]["strong_static"][
                    "selected_by_inner_valid_only"
                ],
            ),
            flush=True,
        )
    combined = {
        "stage": "Stage23A-v2a frozen 57D meta-feature construction",
        "status": "PASS",
        "directions": reports,
        "preprocessing": preprocessing_reports,
        "feature_dim": 57,
        "labels_sources_and_folds_excluded_from_input": True,
        "outer_evaluation_used_for_fit_or_selection": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "judge_training_started": False,
        "student_trained": False,
    }
    atomic_json(
        V2_ROOT / "features" / "meta57" / "feature_build_manifest.json",
        combined,
    )
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
