#!/usr/bin/env python3
"""CPU Phase-1 risk probes for one frozen Expert checkpoint replication."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from stage23d_self_risk_common import (
    EXPERTS,
    MODE_MASKS,
    MODES,
    OUT,
    PCA_DIMS,
    SHUFFLE_SEEDS,
    atomic_json,
    atomic_tsv,
    risk_labels,
    sha256_file,
    utc_now,
)


R1_PREFIXES = (
    "prediction",
    "shared_prediction",
    "final_shared",
    "active_head",
    "shared_specific_abs",
    "head_active",
)
N2_FEATURES = (
    "text_length",
    "audio_length",
    "vision_length",
    "mask_l",
    "mask_a",
    "mask_v",
)
N5_FEATURES = (
    "prediction",
    "prediction_absolute_magnitude",
    "prediction_sign",
    "prediction_distance_to_clip_boundary",
    "mask_l",
    "mask_a",
    "mask_v",
)
EXCLUDED_NUMERIC = {
    "train_index",
    "checkpoint_fold",
    "label",
}
# PCA uses the core A1 head-driving representations. The much larger replayable
# A2 LFA/cross-attention tensors remain preserved in the raw artifact, while
# their train-safe scalar summaries enter R2 directly. This prevents padded
# sequence coordinates from turning the probe into a length detector.
PCA_CORE_DIMS = {
    "LAV": 750,
    "LA": 650,
    "LV": 650,
    "L": 550,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("select", "evaluate"))
    parser.add_argument("--checkpoint-fold", required=True, type=int, choices=(0, 1))
    parser.add_argument("--expert-id", required=True, choices=EXPERTS)
    return parser.parse_args()


def atomic_gzip_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as compressed:
            frame.to_csv(compressed, index=False, float_format="%.10g")
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def safe_spearman(left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = spearmanr(left, right).correlation
    return float(value if np.isfinite(value) else 0.0)


def safe_pearson(left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = pearsonr(left, right).statistic
    return float(value if np.isfinite(value) else 0.0)


def safe_auc(labels, scores):
    return (
        float(roc_auc_score(labels, scores))
        if len(np.unique(labels)) == 2
        else 0.5
    )


def safe_auprc(labels, scores):
    return (
        float(average_precision_score(labels, scores))
        if len(np.unique(labels)) == 2
        else float(np.mean(labels))
    )


def ece(labels, probability, bins=10):
    labels = np.asarray(labels, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (
            probability <= right if right == 1 else probability < right
        )
        if mask.any():
            total += mask.mean() * abs(
                probability[mask].mean() - labels[mask].mean()
            )
    return float(total)


def fixed_precision_recall(labels, probability, precision_target=0.8):
    order = np.argsort(-np.asarray(probability))
    labels = np.asarray(labels, dtype=int)[order]
    true_positive = np.cumsum(labels)
    predicted = np.arange(1, len(labels) + 1)
    precision = true_positive / predicted
    recall = true_positive / max(labels.sum(), 1)
    valid = precision >= precision_target
    return float(recall[valid].max()) if valid.any() else 0.0


def feature_paths(fold, expert, mode):
    directory = OUT / "features" / f"checkpoint_fold{fold}" / expert
    return (
        directory / f"static_internal_features_{mode}.csv.gz",
        directory / f"raw_internal_vectors_{mode}.npz",
    )


def core_pca_features(raw_features, mode):
    raw_features = np.asarray(raw_features, dtype=np.float32)
    active_modalities = int(sum(MODE_MASKS[mode]))
    specific_tail_dimensions = 100 * active_modalities
    core = np.concatenate(
        [
            raw_features[:, :450],
            raw_features[:, -specific_tail_dimensions:],
        ],
        axis=1,
    )
    expected = PCA_CORE_DIMS[mode]
    if core.shape[1] != expected:
        raise RuntimeError(
            f"Recovered core width {core.shape[1]} differs from expected "
            f"{expected} for {mode}"
        )
    return core


def load_mode(fold, expert, mode):
    scalar_path, raw_path = feature_paths(fold, expert, mode)
    if not scalar_path.exists() or not raw_path.exists():
        raise RuntimeError(f"Missing extraction artifacts {fold}/{expert}/{mode}")
    frame = pd.read_csv(scalar_path, dtype={"sample_id": str, "video_id": str})
    raw = np.load(raw_path)
    raw_ids = raw["sample_id"].astype(str)
    if not np.array_equal(frame["sample_id"].astype(str).to_numpy(), raw_ids):
        raise RuntimeError("Raw/scalar sample binding mismatch")
    if not np.array_equal(
        frame["row_binding_sha256"].astype(str).to_numpy(),
        raw["row_binding_sha256"].astype(str),
    ):
        raise RuntimeError("Raw/scalar row SHA mismatch")
    raw_features = raw["features"].astype(np.float32, copy=False)
    return frame, core_pca_features(raw_features, mode)


def label_paths(fold, expert, mode):
    directory = OUT / "features" / f"checkpoint_fold{fold}" / expert
    return (
        directory / f"development_risk_labels_{mode}.csv.gz",
        directory / "sealed_outer_labels" / f"outer_labels_{mode}.csv.gz",
    )


@lru_cache(maxsize=None)
def global_train_abs_error_prior(fold, expert):
    errors = []
    for mode in MODES:
        scalar_path, _ = feature_paths(fold, expert, mode)
        development_path, _ = label_paths(fold, expert, mode)
        scalar = pd.read_csv(
            scalar_path,
            usecols=["sample_id", "self_risk_role", "prediction"],
            dtype={"sample_id": str},
        )
        labels = pd.read_csv(
            development_path,
            usecols=["sample_id", "label"],
            dtype={"sample_id": str},
        )
        train = scalar.loc[scalar["self_risk_role"] == "inner_train"].merge(
            labels, on="sample_id", validate="one_to_one"
        )
        errors.extend(np.abs(train["label"] - train["prediction"]).tolist())
    if not errors:
        raise RuntimeError("Global R0 prior has no inner-train errors")
    return float(np.mean(errors))


def geometry_features(raw, sources, train_mask, pca_dimension):
    scaler = StandardScaler().fit(raw[train_mask])
    standardized = scaler.transform(raw)
    dimension = min(
        int(pca_dimension),
        standardized[train_mask].shape[0] - 1,
        standardized.shape[1],
    )
    pca = PCA(n_components=dimension, random_state=23170)
    pca.fit(standardized[train_mask])
    values = pca.transform(standardized)
    train = values[train_mask]
    centroid = train.mean(axis=0)
    variance = np.maximum(train.var(axis=0), 1e-8)
    centroid_distance = np.linalg.norm(values - centroid, axis=1)
    mahalanobis = np.sqrt(
        np.sum(np.square(values - centroid) / variance, axis=1)
    )
    train_indices = np.flatnonzero(train_mask)
    train_sources = np.asarray(sources)[train_mask]
    neighbor_count = min(len(train_indices), 200)
    search = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    search.fit(train)
    distances, positions = search.kneighbors(values)
    knn_mean = np.empty(len(values), dtype=float)
    nearest_source = np.empty(len(values), dtype=float)
    effective_sources = np.empty(len(values), dtype=int)
    for row in range(len(values)):
        query_source = str(sources[row])
        by_source = {}
        for distance, position in zip(distances[row], positions[row]):
            source = str(train_sources[position])
            if source == query_source:
                continue
            by_source[source] = min(float(distance), by_source.get(source, np.inf))
        ordered = sorted(by_source.values())
        selected = ordered[:30]
        if not selected:
            raise RuntimeError("No source-disjoint hidden neighbor")
        knn_mean[row] = float(np.mean(selected))
        nearest_source[row] = float(selected[0])
        effective_sources[row] = len(selected)
    geometry = pd.DataFrame(
        {
            "hidden_centroid_distance": centroid_distance,
            "hidden_diagonal_mahalanobis": mahalanobis,
            "hidden_knn30_distance": knn_mean,
            "hidden_local_density": 1.0 / (knn_mean + 1e-8),
            "hidden_nearest_source_distance": nearest_source,
            "hidden_effective_neighbor_sources": effective_sources,
        }
    )
    pca_frame = pd.DataFrame(
        values, columns=[f"pca_{index:02d}" for index in range(values.shape[1])]
    )
    return pd.concat([geometry, pca_frame], axis=1), scaler, pca


def scalar_numeric(frame):
    columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in EXCLUDED_NUMERIC
    ]
    return frame[columns].astype(float)


def columns_with_prefix(columns, prefixes):
    return [
        column
        for column in columns
        if any(column == prefix or column.startswith(prefix) for prefix in prefixes)
    ]


def standardize_uncertainty(frame, train_mask):
    components = (
        "active_head_std",
        "submode_prediction_variance",
        "hidden_centroid_distance",
    )
    values = []
    parameters = {}
    for column in components:
        mean = float(frame.loc[train_mask, column].mean())
        std = float(frame.loc[train_mask, column].std())
        std = std if std > 1e-12 else 1.0
        values.append((frame[column].to_numpy(dtype=float) - mean) / std)
        parameters[column] = {"mean": mean, "std": std}
    return np.mean(values, axis=0), parameters


def fit_ridge(train_x, train_y, alpha):
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=float(alpha)).fit(scaler.transform(train_x), train_y)
    return ("ridge", scaler, model)


def fit_hgb_regression(train_x, train_y):
    model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=100,
        learning_rate=0.05,
        random_state=23170,
    ).fit(train_x, train_y)
    return ("hgb", None, model)


def predict_model(bundle, values):
    _, scaler, model = bundle
    transformed = scaler.transform(values) if scaler is not None else values
    return model.predict(transformed)


def fit_classifier(train_x, train_y, algorithm, parameter):
    if len(np.unique(train_y)) < 2:
        return ("prior", None, float(np.mean(train_y)))
    if algorithm == "logistic":
        scaler = StandardScaler().fit(train_x)
        model = LogisticRegression(
            C=float(parameter),
            class_weight="balanced",
            max_iter=2000,
            random_state=23170,
        ).fit(scaler.transform(train_x), train_y)
        return ("logistic", scaler, model)
    model = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=100,
        learning_rate=0.05,
        random_state=23170,
    ).fit(train_x, train_y)
    return ("hgb", None, model)


def predict_probability(bundle, values):
    algorithm, scaler, model = bundle
    if algorithm == "prior":
        return np.full(len(values), model, dtype=float)
    transformed = scaler.transform(values) if scaler is not None else values
    return model.predict_proba(transformed)[:, 1]


def estimator_parameter_count(bundle_or_model):
    model = (
        bundle_or_model[2]
        if isinstance(bundle_or_model, tuple)
        else bundle_or_model
    )
    if isinstance(model, (float, np.floating)):
        return 1
    if hasattr(model, "coef_"):
        return int(np.asarray(model.coef_).size + np.asarray(model.intercept_).size)
    if hasattr(model, "estimators_"):
        return int(
            sum(
                estimator.tree_.node_count
                for estimator in np.asarray(model.estimators_).reshape(-1)
            )
        )
    if hasattr(model, "_predictors"):
        return int(
            sum(
                predictor.nodes.size
                for iteration in model._predictors
                for predictor in iteration
            )
        )
    return 0


def cross_source_shuffle(values, sources, seed):
    rng = np.random.RandomState(int(seed))
    unique = np.unique(sources)
    permuted = unique.copy()
    rng.shuffle(permuted)
    if len(permuted) > 1 and np.any(permuted == unique):
        permuted = np.roll(permuted, 1)
    mapping = dict(zip(unique, permuted))
    indices_by_source = {
        source: np.flatnonzero(np.asarray(sources) == source) for source in unique
    }
    result = np.empty_like(values)
    for target_source in unique:
        donor_source = mapping[target_source]
        target_indices = indices_by_source[target_source]
        donor_indices = indices_by_source[donor_source]
        chosen = rng.choice(donor_indices, size=len(target_indices), replace=True)
        result[target_indices] = values[chosen]
    return result


def choose_regression(
    features,
    labels,
    train_mask,
    valid_mask,
    algorithms,
):
    rows = []
    bundles = {}
    for name, parameter in algorithms:
        bundle = (
            fit_ridge(features[train_mask], labels[train_mask], parameter)
            if name == "ridge"
            else fit_hgb_regression(features[train_mask], labels[train_mask])
        )
        prediction = predict_model(bundle, features[valid_mask])
        risk = np.maximum(np.exp(prediction) - 1e-6, 0.0)
        actual = np.maximum(np.exp(labels[valid_mask]) - 1e-6, 0.0)
        identifier = f"{name}_{parameter}"
        rows.append(
            {
                "algorithm": name,
                "parameter": parameter,
                "inner_valid_spearman": safe_spearman(risk, actual),
                "inner_valid_MAE": mean_absolute_error(actual, risk),
            }
        )
        bundles[identifier] = bundle
    table = pd.DataFrame(rows).sort_values(
        ["inner_valid_spearman", "inner_valid_MAE", "algorithm"],
        ascending=[False, True, True],
    )
    selected = table.iloc[0]
    return (
        bundles[f"{selected['algorithm']}_{selected['parameter']}"],
        table,
        selected.to_dict(),
    )


def choose_classifier(features, labels, train_mask, valid_mask):
    rows = []
    bundles = {}
    candidates = [
        ("logistic", 0.1),
        ("logistic", 1.0),
        ("logistic", 10.0),
        ("hgb", "fixed"),
    ]
    for algorithm, parameter in candidates:
        bundle = fit_classifier(
            features[train_mask], labels[train_mask], algorithm, parameter
        )
        probability = predict_probability(bundle, features[valid_mask])
        identifier = f"{algorithm}_{parameter}"
        rows.append(
            {
                "algorithm": algorithm,
                "parameter": parameter,
                "inner_valid_AUROC": safe_auc(labels[valid_mask], probability),
                "inner_valid_AUPRC": safe_auprc(labels[valid_mask], probability),
            }
        )
        bundles[identifier] = bundle
    table = pd.DataFrame(rows).sort_values(
        ["inner_valid_AUROC", "inner_valid_AUPRC", "algorithm"],
        ascending=[False, False, True],
    )
    selected = table.iloc[0]
    return (
        bundles[f"{selected['algorithm']}_{selected['parameter']}"],
        table,
        selected.to_dict(),
    )


def build_mode_selection(fold, expert, mode, output_dir):
    frame, raw = load_mode(fold, expert, mode)
    development_label_path, _ = label_paths(fold, expert, mode)
    development_labels = pd.read_csv(
        development_label_path, dtype={"sample_id": str}
    )[["sample_id", "label", "row_binding_sha256"]]
    frame = frame.merge(
        development_labels,
        on=["sample_id", "row_binding_sha256"],
        how="left",
        validate="one_to_one",
    )
    train_mask = frame["self_risk_role"].to_numpy() == "inner_train"
    valid_mask = frame["self_risk_role"].to_numpy() == "inner_valid"
    outer_mask = frame["self_risk_role"].to_numpy() == "outer"
    if not train_mask.any() or not valid_mask.any() or not outer_mask.any():
        raise RuntimeError("Empty self-risk role")
    scalar = scalar_numeric(frame)
    candidate_rows = []
    candidates = {}
    # Fit one train-only 64-D PCA and use nested prefixes for the preregistered
    # 16/32/64 candidates. This preserves a common coordinate basis and avoids
    # repeating the expensive source-disjoint neighbor search three times.
    geometry_max, raw_scaler, pca = geometry_features(
        raw,
        frame["video_id"].astype(str).to_numpy(),
        train_mask,
        max(PCA_DIMS),
    )
    geometry_columns = [
        column for column in geometry_max.columns if not column.startswith("pca_")
    ]
    for dimension in PCA_DIMS:
        nested_pca_columns = [
            column
            for column in geometry_max.columns
            if column.startswith("pca_")
        ][: int(dimension)]
        geometry = geometry_max[geometry_columns + nested_pca_columns]
        all_features = pd.concat(
            [scalar.reset_index(drop=True), geometry.reset_index(drop=True)],
            axis=1,
        )
        uncertainty, uncertainty_parameters = standardize_uncertainty(
            all_features, train_mask
        )
        labelled = all_features.copy()
        labelled.insert(0, "raw_uncertainty", uncertainty)
        target_frame = frame[["mode", "self_risk_role", "prediction"]].copy()
        target_frame["label"] = frame["label"].fillna(0.0)
        target_frame["raw_uncertainty"] = uncertainty
        target_frame, thresholds = risk_labels(target_frame)
        algorithms = [
            ("ridge", 0.1),
            ("ridge", 1.0),
            ("ridge", 10.0),
            ("hgb", "fixed"),
        ]
        bundle, table, selected = choose_regression(
            labelled.to_numpy(dtype=float),
            target_frame["log_abs_error"].to_numpy(dtype=float),
            train_mask,
            valid_mask,
            algorithms,
        )
        for row in table.to_dict(orient="records"):
            candidate_rows.append(
                {
                    "mode": mode,
                    "pca_dimension": dimension,
                    **row,
                }
            )
        candidates[dimension] = {
            "features": labelled,
            "targets": target_frame,
            "thresholds": thresholds,
            "uncertainty_parameters": uncertainty_parameters,
            "raw_scaler": raw_scaler,
            "pca": pca,
            "regression": bundle,
            "selected_regression": selected,
        }
    candidate_table = pd.DataFrame(candidate_rows).sort_values(
        [
            "inner_valid_spearman",
            "inner_valid_MAE",
            "pca_dimension",
            "algorithm",
        ],
        ascending=[False, True, True, True],
    )
    chosen_dimension = int(candidate_table.iloc[0]["pca_dimension"])
    chosen = candidates[chosen_dimension]
    features = chosen["features"].to_numpy(dtype=float)
    targets = chosen["targets"]
    r2_regression = chosen["regression"]
    r2_bad, bad_table, bad_selected = choose_classifier(
        features,
        targets["bad20"].to_numpy(dtype=int),
        train_mask,
        valid_mask,
    )
    r2_cw, cw_table, cw_selected = choose_classifier(
        features,
        targets["confident_wrong"].to_numpy(dtype=int),
        train_mask,
        valid_mask,
    )
    r1_columns = columns_with_prefix(scalar.columns, R1_PREFIXES)
    r1 = scalar[r1_columns].to_numpy(dtype=float)
    r1_regression, r1_table, r1_selected = choose_regression(
        r1,
        targets["log_abs_error"].to_numpy(dtype=float),
        train_mask,
        valid_mask,
        [("ridge", 0.1), ("ridge", 1.0), ("ridge", 10.0), ("hgb", "fixed")],
    )
    r1_bad, _, _ = choose_classifier(
        r1, targets["bad20"].to_numpy(dtype=int), train_mask, valid_mask
    )
    r1_cw, _, _ = choose_classifier(
        r1,
        targets["confident_wrong"].to_numpy(dtype=int),
        train_mask,
        valid_mask,
    )
    controls = {}
    control_features = {
        "N2_length_mask_only": scalar[
            [column for column in N2_FEATURES if column in scalar]
        ].to_numpy(dtype=float),
        "N5_label_bin_prior": scalar[
            [column for column in N5_FEATURES if column in scalar]
        ].to_numpy(dtype=float),
    }
    for name, values in control_features.items():
        controls[name] = {
            "regression": fit_ridge(
                values[train_mask],
                targets.loc[train_mask, "log_abs_error"].to_numpy(),
                1.0,
            ),
            "bad20": fit_classifier(
                values[train_mask],
                targets.loc[train_mask, "bad20"].to_numpy(),
                "logistic",
                1.0,
            ),
            "confident_wrong": fit_classifier(
                values[train_mask],
                targets.loc[train_mask, "confident_wrong"].to_numpy(),
                "logistic",
                1.0,
            ),
            "values": values,
        }
    rng = np.random.RandomState(23091 + fold)
    gaussian = rng.normal(
        loc=features[train_mask].mean(axis=0),
        scale=np.maximum(features[train_mask].std(axis=0), 1e-8),
        size=features.shape,
    )
    controls["N1_matched_gaussian"] = {
        "regression": fit_ridge(
            gaussian[train_mask],
            targets.loc[train_mask, "log_abs_error"].to_numpy(),
            1.0,
        ),
        "bad20": fit_classifier(
            gaussian[train_mask],
            targets.loc[train_mask, "bad20"].to_numpy(),
            "logistic",
            1.0,
        ),
        "confident_wrong": fit_classifier(
            gaussian[train_mask],
            targets.loc[train_mask, "confident_wrong"].to_numpy(),
            "logistic",
            1.0,
        ),
        "values": gaussian,
    }
    for seed in SHUFFLE_SEEDS:
        shuffled_train = cross_source_shuffle(
            features[train_mask],
            frame.loc[train_mask, "video_id"].astype(str).to_numpy(),
            seed,
        )
        controls[f"N0_shuffle_seed{seed}"] = {
            "regression": fit_ridge(
                shuffled_train,
                targets.loc[train_mask, "log_abs_error"].to_numpy(),
                1.0,
            ),
            "bad20": fit_classifier(
                shuffled_train,
                targets.loc[train_mask, "bad20"].to_numpy(),
                "logistic",
                1.0,
            ),
            "confident_wrong": fit_classifier(
                shuffled_train,
                targets.loc[train_mask, "confident_wrong"].to_numpy(),
                "logistic",
                1.0,
            ),
            "values": features,
        }
    prediction = frame[
        [
            "sample_id",
            "video_id",
            "mode",
            "self_risk_role",
            "row_binding_sha256",
        ]
    ].copy()
    prediction["raw_uncertainty"] = targets["raw_uncertainty"].to_numpy()
    for proxy in (
        "active_head_std",
        "submode_prediction_variance",
        "hidden_centroid_distance",
        "hidden_diagonal_mahalanobis",
        "hidden_knn30_distance",
        "hidden_nearest_source_distance",
    ):
        if proxy in chosen["features"]:
            prediction[f"proxy__{proxy}"] = chosen["features"][proxy].to_numpy()
    prediction["R0_mode_expected_abs_error"] = float(
        targets.loc[train_mask, "abs_error"].mean()
    )
    prediction["R0_global_expected_abs_error"] = global_train_abs_error_prior(
        fold, expert
    )
    for name, values, regression, bad_model, cw_model in (
        ("R1", r1, r1_regression, r1_bad, r1_cw),
        ("R2", features, r2_regression, r2_bad, r2_cw),
    ):
        predicted_log = predict_model(regression, values)
        prediction[f"{name}_expected_abs_error"] = np.maximum(
            np.exp(predicted_log) - 1e-6, 0.0
        )
        prediction[f"{name}_P_bad20"] = predict_probability(bad_model, values)
        prediction[f"{name}_P_confident_wrong"] = predict_probability(
            cw_model, values
        )
    for suffix in (
        "expected_abs_error",
        "P_bad20",
        "P_confident_wrong",
    ):
        prediction[f"N3_output_only_{suffix}"] = prediction[f"R1_{suffix}"]
    for name, control in controls.items():
        values = control["values"]
        predicted_log = predict_model(control["regression"], values)
        prediction[f"{name}_expected_abs_error"] = np.maximum(
            np.exp(predicted_log) - 1e-6, 0.0
        )
        prediction[f"{name}_P_bad20"] = predict_probability(
            control["bad20"], values
        )
        prediction[f"{name}_P_confident_wrong"] = predict_probability(
            control["confident_wrong"], values
        )
    quantiles = {}
    for quantile in (0.5, 0.8, 0.9):
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            random_state=23170,
        ).fit(features[train_mask], targets.loc[train_mask, "abs_error"])
        prediction[f"R2_q{int(quantile * 100)}_abs_error"] = np.maximum(
            model.predict(features), 0.0
        )
        quantiles[str(quantile)] = model
    outer_prediction = prediction.loc[outer_mask].drop(
        columns=["self_risk_role"]
    )
    if any(
        column in outer_prediction.columns
        for column in (
            "label",
            "abs_error",
            "bad20",
            "confident_wrong",
        )
    ):
        raise RuntimeError("Outer label leaked into frozen risk prediction")
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    outer_path = mode_dir / "outer_risk_predictions_label_free.csv.gz"
    atomic_gzip_csv(outer_prediction, outer_path)
    frozen_feature_frame = pd.concat(
        [
            frame[
                [
                    "sample_id",
                    "video_id",
                    "mode",
                    "self_risk_role",
                    "row_binding_sha256",
                ]
            ].reset_index(drop=True),
            chosen["features"].reset_index(drop=True),
        ],
        axis=1,
    )
    frozen_feature_path = mode_dir / "frozen_static_probe_features_label_free.csv.gz"
    atomic_gzip_csv(frozen_feature_frame, frozen_feature_path)
    atomic_tsv(candidate_table, mode_dir / "regression_candidates.tsv")
    atomic_tsv(bad_table, mode_dir / "bad20_candidates.tsv")
    atomic_tsv(cw_table, mode_dir / "confident_wrong_candidates.tsv")
    atomic_tsv(chosen["thresholds"], mode_dir / "risk_label_thresholds.tsv")
    bundle_path = mode_dir / "frozen_probe_bundle.joblib"
    joblib.dump(
        {
            "pca_dimension": chosen_dimension,
            "raw_scaler": chosen["raw_scaler"],
            "pca": chosen["pca"],
            "feature_columns": list(chosen["features"].columns),
            "R1_columns": r1_columns,
            "R1_regression": r1_regression,
            "R2_regression": r2_regression,
            "R2_bad20": r2_bad,
            "R2_confident_wrong": r2_cw,
            "quantiles": quantiles,
            "controls": {
                name: {
                    key: value
                    for key, value in control.items()
                    if key != "values"
                }
                for name, control in controls.items()
            },
        },
        bundle_path,
    )
    r2_parameter_count = sum(
        estimator_parameter_count(model)
        for model in (
            r2_regression,
            r2_bad,
            r2_cw,
            *quantiles.values(),
        )
    )
    r1_parameter_count = sum(
        estimator_parameter_count(model)
        for model in (r1_regression, r1_bad, r1_cw)
    )
    return {
        "mode": mode,
        "selected_pca_dimension": chosen_dimension,
        "selected_R2_regression": chosen["selected_regression"],
        "selected_R2_bad20": bad_selected,
        "selected_R2_confident_wrong": cw_selected,
        "selected_R1_regression": r1_selected,
        "uncertainty_parameters": chosen["uncertainty_parameters"],
        "outer_prediction_path": str(outer_path.resolve()),
        "outer_prediction_sha256": sha256_file(outer_path),
        "outer_prediction_rows": len(outer_prediction),
        "outer_labels_in_prediction": False,
        "frozen_static_feature_path": str(frozen_feature_path.resolve()),
        "frozen_static_feature_sha256": sha256_file(frozen_feature_path),
        "frozen_static_feature_contains_label": False,
        "bundle_path": str(bundle_path.resolve()),
        "bundle_sha256": sha256_file(bundle_path),
        "R1_primary_probe_parameter_count": r1_parameter_count,
        "R2_primary_probe_parameter_count": r2_parameter_count,
    }


def select(cli):
    protocol = json.loads(
        (OUT / "protocol" / "frozen_protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not protocol["static_feature_probe_authorized"]:
        raise RuntimeError("Static probes not authorized")
    output_dir = (
        OUT
        / "phase1"
        / f"checkpoint_fold{cli.checkpoint_fold}"
        / cli.expert_id
    )
    selection_path = output_dir / "frozen_inner_valid_selection.json"
    if selection_path.exists():
        print(selection_path.read_text(encoding="utf-8"))
        return
    modes = [
        build_mode_selection(
            cli.checkpoint_fold, cli.expert_id, mode, output_dir
        )
        for mode in MODES
    ]
    selection = {
        "stage": "Stage23D-A inner-valid frozen self-risk probe selection",
        "status": "FROZEN_OUTER_LABELS_NOT_OPENED",
        "checkpoint_fold": cli.checkpoint_fold,
        "expert_id": cli.expert_id,
        "selection_scope": "self-risk inner-train/inner-valid only",
        "modes": {row["mode"]: row for row in modes},
        "outer_prediction_count": len(modes),
        "outer_evaluation_access_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "frozen_at": utc_now(),
    }
    atomic_json(selection_path, selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


def risk_coverage(actual, predicted, model, fold, expert, mode):
    rows = []
    order = np.argsort(predicted)
    baseline = float(np.mean(actual))
    for coverage in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        retained_count = max(1, int(np.ceil(coverage * len(order))))
        retained = order[:retained_count]
        rejected = order[retained_count:]
        rows.append(
            {
                "checkpoint_fold": fold,
                "expert_id": expert,
                "mode": mode,
                "model": model,
                "coverage": coverage,
                "retained_samples": len(retained),
                "retained_actual_MAE": float(np.mean(actual[retained])),
                "rejected_actual_MAE": (
                    float(np.mean(actual[rejected])) if len(rejected) else 0.0
                ),
                "random_expected_MAE": baseline,
                "improvement_vs_random": baseline
                - float(np.mean(actual[retained])),
            }
        )
    return rows


def evaluate(cli):
    output_dir = (
        OUT
        / "phase1"
        / f"checkpoint_fold{cli.checkpoint_fold}"
        / cli.expert_id
    )
    selection_path = output_dir / "frozen_inner_valid_selection.json"
    result_path = output_dir / "outer_evaluation_manifest.json"
    access_lock_path = output_dir / "outer_evaluation_access_lock.json"
    if result_path.exists():
        print(result_path.read_text(encoding="utf-8"))
        return
    if access_lock_path.exists():
        raise RuntimeError(
            "Outer evaluation access was already claimed; refusing a second access"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["outer_evaluation_access_count"] != 0:
        raise RuntimeError("Outer labels already opened")
    atomic_json(
        access_lock_path,
        {
            "stage": "Stage23D-A one-shot outer label access lock",
            "checkpoint_fold": cli.checkpoint_fold,
            "expert_id": cli.expert_id,
            "outer_evaluation_access_count": 1,
            "status": "CLAIMED_BEFORE_LABEL_FILE_OPEN",
            "claimed_at": utc_now(),
        },
    )
    metric_rows = []
    coverage_rows = []
    calibration_rows = []
    proxy_rows = []
    sensitivity_rows = []
    diagnostic_frames = []
    for mode in MODES:
        frame, _ = load_mode(cli.checkpoint_fold, cli.expert_id, mode)
        _, sealed_label_path = label_paths(
            cli.checkpoint_fold, cli.expert_id, mode
        )
        outer_labels_sealed = pd.read_csv(
            sealed_label_path, dtype={"sample_id": str}
        )
        outer = frame.loc[frame["self_risk_role"] == "outer"].copy()
        prediction_path = Path(
            selection["modes"][mode]["outer_prediction_path"]
        )
        if sha256_file(prediction_path) != selection["modes"][mode][
            "outer_prediction_sha256"
        ]:
            raise RuntimeError("Frozen outer risk prediction SHA mismatch")
        prediction = pd.read_csv(prediction_path, dtype={"sample_id": str})
        joined = prediction.merge(
            outer[["sample_id", "prediction"]].merge(
                outer_labels_sealed[["sample_id", "label"]],
                on="sample_id",
                validate="one_to_one",
            ),
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        actual = np.abs(joined["label"] - joined["prediction"]).to_numpy()
        thresholds = pd.read_csv(
            output_dir / mode / "risk_label_thresholds.tsv", sep="\t"
        ).iloc[0]
        bad20 = (actual >= float(thresholds["error_top20_threshold"])).astype(int)
        bad10 = (actual >= float(thresholds["error_top10_threshold"])).astype(int)
        fixed_bad = (
            actual > float(thresholds["fixed_error_threshold"])
        ).astype(int)
        confident = (
            joined["raw_uncertainty"].to_numpy()
            <= float(thresholds["uncertainty_bottom30_threshold"])
        )
        confident_wrong = confident & (bad20 == 1)
        diagnostic = joined[
            [
                "sample_id",
                "video_id",
                "mode",
                "R2_expected_abs_error",
                "R2_q90_abs_error",
                "R2_P_bad20",
                "R2_P_confident_wrong",
            ]
        ].copy()
        diagnostic.insert(0, "expert_id", cli.expert_id)
        diagnostic.insert(0, "checkpoint_fold", cli.checkpoint_fold)
        diagnostic["actual_abs_error"] = actual
        diagnostic["bad20"] = bad20
        diagnostic["confident_wrong"] = confident_wrong.astype(int)
        diagnostic_frames.append(diagnostic)
        for model in (
            "R0_global",
            "R0_mode",
            "R1",
            "R2",
            "N3_output_only",
            "N2_length_mask_only",
            "N5_label_bin_prior",
            "N1_matched_gaussian",
            *[f"N0_shuffle_seed{seed}" for seed in SHUFFLE_SEEDS],
        ):
            risk = joined[f"{model}_expected_abs_error"].to_numpy()
            probability = (
                joined[f"{model}_P_bad20"].to_numpy()
                if f"{model}_P_bad20" in joined
                else np.full(len(joined), 0.20)
            )
            cw_probability = (
                joined[f"{model}_P_confident_wrong"].to_numpy()
                if f"{model}_P_confident_wrong" in joined
                else None
            )
            metric_rows.append(
                {
                    "checkpoint_fold": cli.checkpoint_fold,
                    "expert_id": cli.expert_id,
                    "mode": mode,
                    "model": model,
                    "Error_Spearman": safe_spearman(risk, actual),
                    "Error_Pearson": safe_pearson(risk, actual),
                    "Risk_MAE": mean_absolute_error(actual, risk),
                    "Risk_RMSE": mean_squared_error(
                        actual, risk, squared=False
                    ),
                    "bad20_AUROC": safe_auc(
                        bad20, probability
                    ),
                    "bad20_AUPRC": safe_auprc(
                        bad20, probability
                    ),
                    "bad20_Brier": brier_score_loss(
                        bad20, probability
                    ),
                    "bad20_ECE": ece(bad20, probability),
                    "bad10_AUROC": safe_auc(bad10, probability),
                    "fixed_bad_AUROC": safe_auc(fixed_bad, probability),
                    "fixed_bad_AUPRC": safe_auprc(fixed_bad, probability),
                    "confident_wrong_AUROC": (
                        safe_auc(confident_wrong, cw_probability)
                        if cw_probability is not None
                        else "NA"
                    ),
                    "confident_wrong_AUPRC": (
                        safe_auprc(confident_wrong, cw_probability)
                        if cw_probability is not None
                        else "NA"
                    ),
                    "confident_wrong_recall_at_precision_0p8": (
                        fixed_precision_recall(
                            confident_wrong, cw_probability
                        )
                        if cw_probability is not None
                        else "NA"
                    ),
                }
            )
            coverage_rows.extend(
                risk_coverage(
                    actual,
                    risk,
                    model,
                    cli.checkpoint_fold,
                    cli.expert_id,
                    mode,
                )
            )
        for column in [
            column
            for column in joined.columns
            if column.startswith("proxy__")
        ]:
            proxy = joined[column].to_numpy(dtype=float)
            proxy_rows.append(
                {
                    "checkpoint_fold": cli.checkpoint_fold,
                    "expert_id": cli.expert_id,
                    "mode": mode,
                    "proxy": column.removeprefix("proxy__"),
                    "Error_Spearman": safe_spearman(proxy, actual),
                    "Error_Pearson": safe_pearson(proxy, actual),
                    "bad20_AUROC": safe_auc(bad20, proxy),
                    "bad10_AUROC": safe_auc(bad10, proxy),
                    "confident_wrong_AUROC": safe_auc(
                        confident_wrong, proxy
                    ),
                }
            )
        for confidence_percent in (20, 30, 40):
            confidence_threshold = float(
                thresholds[f"uncertainty_bottom{confidence_percent}_threshold"]
            )
            for error_percent, error_threshold in (
                (10, float(thresholds["error_top10_threshold"])),
                (20, float(thresholds["error_top20_threshold"])),
            ):
                event = (
                    joined["raw_uncertainty"].to_numpy()
                    <= confidence_threshold
                ) & (actual >= error_threshold)
                probability = joined["R2_P_confident_wrong"].to_numpy()
                sensitivity_rows.append(
                    {
                        "checkpoint_fold": cli.checkpoint_fold,
                        "expert_id": cli.expert_id,
                        "mode": mode,
                        "confidence_bottom_percent": confidence_percent,
                        "error_top_percent": error_percent,
                        "event_count": int(event.sum()),
                        "event_prevalence": float(event.mean()),
                        "R2_AUROC": safe_auc(event, probability),
                        "R2_AUPRC": safe_auprc(event, probability),
                        "R2_recall_at_precision_0p8": fixed_precision_recall(
                            event, probability
                        ),
                    }
                )
        for quantile in (50, 80, 90):
            predicted = joined[f"R2_q{quantile}_abs_error"].to_numpy()
            interval_width = np.maximum(
                joined["R2_q90_abs_error"].to_numpy()
                - joined["R2_q50_abs_error"].to_numpy(),
                0.0,
            )
            alpha = quantile / 100.0
            residual = actual - predicted
            pinball = np.mean(
                np.maximum(alpha * residual, (alpha - 1.0) * residual)
            )
            calibration_rows.append(
                {
                    "checkpoint_fold": cli.checkpoint_fold,
                    "expert_id": cli.expert_id,
                    "mode": mode,
                    "quantile": alpha,
                    "empirical_coverage": float(np.mean(actual <= predicted)),
                    "pinball_loss": float(pinball),
                    "mean_predicted_quantile": float(np.mean(predicted)),
                    "mean_q90_minus_q50_interval_width": float(
                        np.mean(interval_width)
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    coverage = pd.DataFrame(coverage_rows)
    calibration = pd.DataFrame(calibration_rows)
    proxies = pd.DataFrame(proxy_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    atomic_tsv(metrics, output_dir / "outer_metrics.tsv")
    atomic_tsv(coverage, output_dir / "risk_coverage.tsv")
    atomic_tsv(calibration, output_dir / "quantile_calibration.tsv")
    atomic_tsv(proxies, output_dir / "individual_proxy_metrics.tsv")
    atomic_tsv(
        sensitivity, output_dir / "confident_wrong_sensitivity.tsv"
    )
    diagnostic_path = output_dir / "outer_diagnostic_ledger.csv.gz"
    atomic_gzip_csv(diagnostics, diagnostic_path)
    result = {
        "stage": "Stage23D-A one-shot Risk Head outer evaluation",
        "status": "COMPLETED",
        "checkpoint_fold": cli.checkpoint_fold,
        "expert_id": cli.expert_id,
        "selection_sha256": sha256_file(selection_path),
        "outer_access_lock_sha256": sha256_file(access_lock_path),
        "metrics_path": str((output_dir / "outer_metrics.tsv").resolve()),
        "metrics_sha256": sha256_file(output_dir / "outer_metrics.tsv"),
        "coverage_path": str((output_dir / "risk_coverage.tsv").resolve()),
        "coverage_sha256": sha256_file(output_dir / "risk_coverage.tsv"),
        "calibration_path": str(
            (output_dir / "quantile_calibration.tsv").resolve()
        ),
        "calibration_sha256": sha256_file(
            output_dir / "quantile_calibration.tsv"
        ),
        "individual_proxy_metrics_sha256": sha256_file(
            output_dir / "individual_proxy_metrics.tsv"
        ),
        "confident_wrong_sensitivity_sha256": sha256_file(
            output_dir / "confident_wrong_sensitivity.tsv"
        ),
        "outer_diagnostic_ledger_sha256": sha256_file(diagnostic_path),
        "outer_evaluation_access_count": 1,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "arbiter_trained": False,
        "student_trained": False,
        "expert_retrained": False,
        "completed_at": utc_now(),
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    cli = parse_args()
    if cli.phase == "select":
        select(cli)
    else:
        evaluate(cli)


if __name__ == "__main__":
    main()
