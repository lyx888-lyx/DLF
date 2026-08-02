"""Strict expert self-risk estimation utilities for V9.25.

V9.25 does not route samples or alter any expert prediction. It estimates,
from saved strict OOF V9.19 pools, whether each frozen specialist knows:
(1) whether a sample belongs to its region, (2) how large its error may be,
(3) whether it will beat V9.21, and (4) whether it may fail severely.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

from .model.SemanticCostCoachV99 import ACTION_NAMES, SPECIALIST_NAMES
from .no_train_decomposition_v920 import region_index

AUDIT_VERSION = "expert_self_risk_audit_v925_v1"
EXPERT_NAMES = tuple(SPECIALIST_NAMES)
EXPERT_ACTION_INDEX = {name: ACTION_NAMES.index(name) for name in EXPERT_NAMES}
EXPERT_REGION_INDEX = {
    "strong_negative": 0,
    "boundary": 2,
    "positive": 3,
    "strong_positive": 4,
}
BINARY_TARGET_NAMES = (
    "membership",
    "large_error_030",
    "large_error_050",
    "large_error_100",
    "win_vs_baseline",
    "large_gain_010",
    "large_harm_030",
)
CONTINUOUS_TARGET_NAMES = ("absolute_error", "gain_vs_baseline")


@dataclass(frozen=True)
class ExpertSelfRiskConfigV925:
    ridge_lambda: float = 1.0
    logistic_l2: float = 1.0
    calibration_l2: float = 0.01
    optimizer_max_iterations: int = 1000
    calibration_bins: int = 10
    error_threshold_030: float = 0.30
    error_threshold_050: float = 0.50
    error_threshold_100: float = 1.00
    large_gain_threshold: float = 0.10
    large_harm_threshold: float = 0.30
    safe_rank_fraction: float = 0.20
    harm_probability_penalty: float = 0.30
    absolute_tail_penalty: float = 0.10
    native_high_confidence_threshold: float = 0.80
    required_positive_outer_folds: int = 4
    required_win_auc: float = 0.60
    required_large_harm_auc: float = 0.60
    required_error_spearman: float = 0.20
    required_safe_gain_ci_low: float = 0.0
    required_gain_over_native_rank: float = 0.002

    def validate(self) -> None:
        if min(self.ridge_lambda, self.logistic_l2, self.calibration_l2) < 0:
            raise ValueError("regularization must be non-negative")
        if self.optimizer_max_iterations < 10:
            raise ValueError("optimizer_max_iterations is too small")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least two")
        if not 0.0 < self.safe_rank_fraction < 1.0:
            raise ValueError("safe_rank_fraction must be in (0,1)")
        thresholds = (
            self.error_threshold_030,
            self.error_threshold_050,
            self.error_threshold_100,
            self.large_gain_threshold,
            self.large_harm_threshold,
        )
        if min(thresholds) <= 0.0:
            raise ValueError("risk thresholds must be positive")


def _np(value, dtype=np.float64) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _vector(value) -> np.ndarray:
    return _np(value).reshape(-1)


def _matrix(value) -> np.ndarray:
    array = _np(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected matrix, got {array.shape}")
    return array


def validate_pool(payload: Mapping[str, object]) -> int:
    required = (
        "labels",
        "anchor",
        "function_space",
        "expert_predictions",
        "expert_confidences",
        "expert_corrections",
        "sample_ids",
        "group_ids",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise KeyError(f"V9.19 pool missing fields: {missing}")
    n = len(_vector(payload["labels"]))
    if _matrix(payload["function_space"]).shape != (n, 4):
        raise ValueError("function_space must have shape [N,4]")
    for key in (
        "expert_predictions",
        "expert_confidences",
        "expert_corrections",
    ):
        if _matrix(payload[key]).shape != (n, len(EXPERT_NAMES)):
            raise ValueError(f"{key} must have shape [N,4]")
    if len(payload["sample_ids"]) != n or len(payload["group_ids"]) != n:
        raise ValueError("pool identifiers do not align")
    return n


def _region_geometry(prediction: np.ndarray, expert: str):
    value = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if expert == "strong_negative":
        inside = value < -1.5
        outside = np.maximum(value + 1.5, 0.0)
        center = value + 2.25
    elif expert == "boundary":
        inside = (value >= -0.5) & (value <= 0.5)
        outside = np.maximum(np.abs(value) - 0.5, 0.0)
        center = value
    elif expert == "positive":
        inside = (value > 0.5) & (value <= 1.5)
        outside = np.maximum(
            np.maximum(0.5 - value, value - 1.5), 0.0
        )
        center = value - 1.0
    elif expert == "strong_positive":
        inside = value > 1.5
        outside = np.maximum(1.5 - value, 0.0)
        center = value - 2.25
    else:
        raise ValueError(f"unknown expert: {expert}")
    return inside.astype(np.float64), outside, center


def build_expert_features(payload, baseline_prediction, expert: str):
    """Build the fixed deployment-visible feature vector for one specialist."""
    n = validate_pool(payload)
    if expert not in EXPERT_NAMES:
        raise ValueError(f"unknown expert: {expert}")
    expert_index = EXPERT_NAMES.index(expert)
    baseline = _vector(baseline_prediction)
    if len(baseline) != n:
        raise ValueError("baseline prediction length mismatch")
    anchor = _vector(payload["anchor"])
    function_space = _matrix(payload["function_space"])
    predictions = _matrix(payload["expert_predictions"])
    confidences = _matrix(payload["expert_confidences"])
    corrections = _matrix(payload["expert_corrections"])
    actions = np.column_stack([anchor, predictions])
    own = predictions[:, expert_index]
    native = np.clip(confidences[:, expert_index], 0.0, 1.0)
    correction = corrections[:, expert_index]
    inside, outside, center = _region_geometry(own, expert)
    own_action = EXPERT_ACTION_INDEX[expert]
    own_rank = (
        np.sum(
            actions < actions[:, own_action : own_action + 1], axis=1
        )
        / 4.0
    )
    stats = {
        "action_mean": actions.mean(axis=1),
        "action_std": actions.std(axis=1),
        "action_range": actions.max(axis=1) - actions.min(axis=1),
        "action_median": np.median(actions, axis=1),
    }
    columns, names = [], []

    def add(name, value):
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(array) != n:
            raise ValueError(f"feature {name} has wrong length")
        names.append(name)
        columns.append(array)

    for index in range(4):
        add(f"function_space_{index}", function_space[:, index])
    add("anchor", anchor)
    add("baseline", baseline)
    for index, name in enumerate(ACTION_NAMES):
        add(f"action_{name}", actions[:, index])
    add("own_prediction", own)
    add("own_confidence", native)
    add("own_correction", correction)
    add("own_abs_correction", np.abs(correction))
    add("own_minus_anchor", own - anchor)
    add("own_abs_minus_anchor", np.abs(own - anchor))
    add("own_minus_baseline", own - baseline)
    add("own_abs_minus_baseline", np.abs(own - baseline))
    for name, value in stats.items():
        add(name, value)
    add("own_action_rank", own_rank)
    add("prediction_inside_own_region", inside)
    add("prediction_distance_outside_region", outside)
    add("prediction_signed_center_distance", center)
    add("prediction_abs_center_distance", np.abs(center))
    add(
        "confidence_x_abs_baseline_gap",
        native * np.abs(own - baseline),
    )
    add("confidence_x_abs_correction", native * np.abs(correction))
    add("confidence_x_outside_distance", native * outside)
    add("own_prediction_squared", own**2)
    add("anchor_squared", anchor**2)
    add("baseline_squared", baseline**2)
    add("action_std_squared", stats["action_std"] ** 2)
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("non-finite V9.25 feature")
    return {
        "matrix": matrix,
        "feature_names": names,
        "native_confidence": native,
        "expert_prediction": own,
        "anchor_prediction": anchor,
        "baseline_prediction": baseline,
    }


def build_expert_targets(payload, baseline_prediction, expert, config):
    n = validate_pool(payload)
    labels = _vector(payload["labels"])
    baseline = _vector(baseline_prediction)
    prediction = _matrix(payload["expert_predictions"])[
        :, EXPERT_NAMES.index(expert)
    ]
    if len(baseline) != n:
        raise ValueError("baseline prediction length mismatch")
    regions = _np(
        region_index(torch.tensor(labels, dtype=torch.float32)),
        np.int64,
    ).reshape(-1)
    error = np.abs(prediction - labels)
    gain = np.abs(baseline - labels) - error
    return {
        "membership": (
            regions == EXPERT_REGION_INDEX[expert]
        ).astype(np.float64),
        "absolute_error": error,
        "large_error_030": (
            error > config.error_threshold_030
        ).astype(np.float64),
        "large_error_050": (
            error > config.error_threshold_050
        ).astype(np.float64),
        "large_error_100": (
            error > config.error_threshold_100
        ).astype(np.float64),
        "gain_vs_baseline": gain,
        "win_vs_baseline": (gain > 0.0).astype(np.float64),
        "large_gain_010": (
            gain > config.large_gain_threshold
        ).astype(np.float64),
        "large_harm_030": (
            gain < -config.large_harm_threshold
        ).astype(np.float64),
    }


def _standardizer(features):
    value = np.asarray(features, dtype=np.float64)
    mean = value.mean(axis=0)
    scale = value.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return {"mean": mean, "scale": scale}


def _transform(features, scaler):
    return (
        np.asarray(features, dtype=np.float64) - scaler["mean"]
    ) / scaler["scale"]


def _fit_ridge(features, target, regularization):
    scaler = _standardizer(features)
    value = _transform(features, scaler)
    design = np.column_stack([np.ones(len(value)), value])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficient = np.linalg.pinv(
        design.T @ design + float(regularization) * penalty
    ) @ (design.T @ np.asarray(target))
    return {
        "kind": "ridge",
        "scaler": scaler,
        "coefficient": coefficient,
    }


def _predict_ridge(model, features):
    value = _transform(features, model["scaler"])
    return np.column_stack([np.ones(len(value)), value]) @ np.asarray(
        model["coefficient"]
    )


def _sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _logit(probability):
    value = np.clip(
        np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6
    )
    return np.log(value) - np.log1p(-value)


def _fit_logistic(features, target, regularization, max_iterations):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prevalence = float(target.mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        return {
            "kind": "constant_logistic",
            "probability": float(
                np.clip(prevalence, 1e-6, 1.0 - 1e-6)
            ),
        }
    scaler = _standardizer(features)
    value = _transform(features, scaler)
    design = np.column_stack([np.ones(len(value)), value])
    initial = np.zeros(design.shape[1])
    initial[0] = np.log(prevalence / (1.0 - prevalence))

    def objective(coefficient):
        logits = design @ coefficient
        return float(
            (
                np.logaddexp(0.0, logits) - target * logits
            ).mean()
            + 0.5
            * float(regularization)
            * np.square(coefficient[1:]).sum()
            / len(target)
        )

    def gradient(coefficient):
        result = (
            design.T @ (_sigmoid(design @ coefficient) - target)
        ) / len(target)
        result[1:] += (
            float(regularization) * coefficient[1:] / len(target)
        )
        return result

    optimized = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-12,
        },
    )
    if not optimized.success:
        raise RuntimeError(f"logistic fit failed: {optimized.message}")
    return {
        "kind": "logistic",
        "scaler": scaler,
        "coefficient": optimized.x,
    }


def _predict_logistic(model, features):
    if model["kind"] == "constant_logistic":
        return np.full(
            len(features), model["probability"], dtype=np.float64
        )
    value = _transform(features, model["scaler"])
    design = np.column_stack([np.ones(len(value)), value])
    return _sigmoid(design @ np.asarray(model["coefficient"]))


def _fit_platt(probability, target, regularization, max_iterations):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prevalence = float(target.mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        return {
            "kind": "constant_platt",
            "probability": float(
                np.clip(prevalence, 1e-6, 1.0 - 1e-6)
            ),
        }
    design = np.column_stack(
        [np.ones(len(target)), _logit(probability)]
    )
    initial = np.array(
        [np.log(prevalence / (1.0 - prevalence)), 1.0]
    )

    def objective(coefficient):
        logits = design @ coefficient
        return float(
            (
                np.logaddexp(0.0, logits) - target * logits
            ).mean()
            + 0.5
            * float(regularization)
            * coefficient[1] ** 2
            / len(target)
        )

    def gradient(coefficient):
        result = (
            design.T @ (_sigmoid(design @ coefficient) - target)
        ) / len(target)
        result[1] += (
            float(regularization) * coefficient[1] / len(target)
        )
        return result

    optimized = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-12,
        },
    )
    if not optimized.success:
        raise RuntimeError(
            f"Platt calibration failed: {optimized.message}"
        )
    return {"kind": "platt", "coefficient": optimized.x}


def _apply_platt(model, probability):
    if model["kind"] == "constant_platt":
        return np.full(len(probability), model["probability"])
    coefficient = np.asarray(model["coefficient"])
    return _sigmoid(
        coefficient[0] + coefficient[1] * _logit(probability)
    )


def _fit_affine(prediction, target, regularization):
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    design = np.column_stack([np.ones(len(prediction)), prediction])
    penalty = np.diag([0.0, float(regularization)])
    coefficient = np.linalg.pinv(
        design.T @ design + penalty
    ) @ (design.T @ target)
    return {"coefficient": coefficient}


def _apply_affine(model, prediction):
    coefficient = np.asarray(model["coefficient"])
    prediction = np.asarray(prediction).reshape(-1)
    return coefficient[0] + coefficient[1] * prediction


def _finite_quantile(values, coverage):
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    rank = int(np.ceil((len(values) + 1) * float(coverage)))
    return float(values[min(max(rank, 1), len(values)) - 1])


def fit_risk_bundle(
    fit_features,
    fit_targets,
    calibration_features,
    calibration_targets,
    config,
):
    config.validate()
    binary_models = {}
    for name in BINARY_TARGET_NAMES:
        base = _fit_logistic(
            fit_features,
            fit_targets[name],
            config.logistic_l2,
            config.optimizer_max_iterations,
        )
        calibrator = _fit_platt(
            _predict_logistic(base, calibration_features),
            calibration_targets[name],
            config.calibration_l2,
            config.optimizer_max_iterations,
        )
        binary_models[name] = {
            "base": base,
            "calibrator": calibrator,
        }
    continuous_models = {}
    for name in CONTINUOUS_TARGET_NAMES:
        base = _fit_ridge(
            fit_features, fit_targets[name], config.ridge_lambda
        )
        calibrator = _fit_affine(
            _predict_ridge(base, calibration_features),
            calibration_targets[name],
            config.calibration_l2,
        )
        continuous_models[name] = {
            "base": base,
            "calibrator": calibrator,
        }
    error_calibration = np.maximum(
        _apply_affine(
            continuous_models["absolute_error"]["calibrator"],
            _predict_ridge(
                continuous_models["absolute_error"]["base"],
                calibration_features,
            ),
        ),
        0.0,
    )
    residual = (
        np.asarray(calibration_targets["absolute_error"])
        - error_calibration
    )
    offsets = {
        str(quantile): _finite_quantile(
            residual, float(quantile) / 100.0
        )
        for quantile in (50, 80, 95)
    }
    return {
        "version": AUDIT_VERSION,
        "config": asdict(config),
        "binary_models": binary_models,
        "continuous_models": continuous_models,
        "absolute_error_upper_offsets": offsets,
    }


def predict_risk_bundle(bundle, features, config):
    result = {}
    for name, model in bundle["binary_models"].items():
        result[f"pred_{name}_prob"] = np.clip(
            _apply_platt(
                model["calibrator"],
                _predict_logistic(model["base"], features),
            ),
            0.0,
            1.0,
        )
    for name, model in bundle["continuous_models"].items():
        value = _apply_affine(
            model["calibrator"],
            _predict_ridge(model["base"], features),
        )
        result[f"pred_{name}"] = (
            np.maximum(value, 0.0)
            if name == "absolute_error"
            else value
        )
    for quantile, offset in bundle[
        "absolute_error_upper_offsets"
    ].items():
        result[f"pred_error_upper_{quantile}"] = np.maximum(
            result["pred_absolute_error"] + float(offset), 0.0
        )
    result["safe_gain_score"] = (
        result["pred_gain_vs_baseline"]
        - config.harm_probability_penalty
        * result["pred_large_harm_030_prob"]
        - config.absolute_tail_penalty
        * result["pred_large_error_050_prob"]
    )
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("non-finite risk prediction")
    return result


def _subset(targets, indices):
    return {
        name: np.asarray(value)[indices]
        for name, value in targets.items()
    }


def crossfit_risk_predictions(features, targets, fold_index, config):
    features = np.asarray(features, dtype=np.float64)
    folds = np.asarray(fold_index, dtype=np.int64).reshape(-1)
    unique = sorted(int(value) for value in np.unique(folds))
    if len(unique) < 3:
        raise ValueError("at least three group folds are required")
    outputs = None
    rows = []
    for position, holdout_fold in enumerate(unique):
        calibration_fold = unique[(position + 1) % len(unique)]
        holdout = np.flatnonzero(folds == holdout_fold)
        calibration = np.flatnonzero(folds == calibration_fold)
        fit = np.flatnonzero(
            (folds != holdout_fold) & (folds != calibration_fold)
        )
        if min(len(fit), len(calibration), len(holdout)) == 0:
            raise RuntimeError("empty V9.25 tri-split")
        bundle = fit_risk_bundle(
            features[fit],
            _subset(targets, fit),
            features[calibration],
            _subset(targets, calibration),
            config,
        )
        local = predict_risk_bundle(
            bundle, features[holdout], config
        )
        if outputs is None:
            outputs = {
                name: np.full(len(features), np.nan)
                for name in local
            }
        for name, value in local.items():
            outputs[name][holdout] = value
        rows.append(
            {
                "holdout_fold": holdout_fold,
                "calibration_fold": calibration_fold,
                "fit_sample_count": len(fit),
                "calibration_sample_count": len(calibration),
                "holdout_sample_count": len(holdout),
            }
        )
    if outputs is None or any(
        not np.isfinite(value).all() for value in outputs.values()
    ):
        raise FloatingPointError("incomplete cross-fit predictions")
    return {"predictions": outputs, "split_rows": rows}


def fit_outer_risk_bundle(
    inner_features, inner_targets, fold_index, config
):
    folds = np.asarray(fold_index, dtype=np.int64).reshape(-1)
    unique = sorted(int(value) for value in np.unique(folds))
    if len(unique) < 2:
        raise ValueError("outer risk fit needs at least two folds")
    calibration_fold = unique[0]
    calibration = np.flatnonzero(folds == calibration_fold)
    fit = np.flatnonzero(folds != calibration_fold)
    return {
        "bundle": fit_risk_bundle(
            np.asarray(inner_features)[fit],
            _subset(inner_targets, fit),
            np.asarray(inner_features)[calibration],
            _subset(inner_targets, calibration),
            config,
        ),
        "calibration_fold": calibration_fold,
        "fit_sample_count": len(fit),
        "calibration_sample_count": len(calibration),
    }


def binary_auc(target, probability):
    target = np.asarray(target).reshape(-1) > 0.5
    probability = np.asarray(probability).reshape(-1)
    positive = int(target.sum())
    negative = int((~target).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(probability, method="average")
    return float(
        (
            ranks[target].sum()
            - positive * (positive + 1) / 2
        )
        / (positive * negative)
    )


def expected_calibration_error(target, probability, bins):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        0.0,
        1.0,
    )
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = 0.0
    for index in range(int(bins)):
        mask = (probability >= edges[index]) & (
            (probability <= edges[index + 1])
            if index == int(bins) - 1
            else (probability < edges[index + 1])
        )
        if mask.any():
            total += float(mask.mean()) * abs(
                float(probability[mask].mean())
                - float(target[mask].mean())
            )
    return float(total)


def binary_metric_row(target, probability, bins):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        1e-8,
        1.0 - 1e-8,
    )
    return {
        "sample_count": len(target),
        "positive_rate": float(target.mean()),
        "auc": binary_auc(target, probability),
        "brier": float(np.square(probability - target).mean()),
        "log_loss": float(
            -(
                target * np.log(probability)
                + (1.0 - target) * np.log1p(-probability)
            ).mean()
        ),
        "ece": expected_calibration_error(
            target, probability, bins
        ),
    }


def continuous_metric_row(target, prediction):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    pearson = (
        float(np.corrcoef(target, prediction)[0, 1])
        if np.std(target) > 1e-12
        and np.std(prediction) > 1e-12
        else float("nan")
    )
    rank = spearmanr(target, prediction).statistic
    return {
        "sample_count": len(target),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
        "mae": float(np.abs(prediction - target).mean()),
        "rmse": float(
            np.sqrt(np.square(prediction - target).mean())
        ),
        "pearson": pearson,
        "spearman": (
            float(rank) if np.isfinite(rank) else float("nan")
        ),
    }


def upper_bound_metric_row(error, upper):
    error = np.asarray(error, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    missed = error > upper
    return {
        "sample_count": len(error),
        "coverage": float((~missed).mean()),
        "mean_upper_bound": float(upper.mean()),
        "mean_excess_when_missed": (
            float((error[missed] - upper[missed]).mean())
            if missed.any()
            else 0.0
        ),
    }


def add_rank_selection_flags(
    frame,
    fraction,
    group_columns=("outer_fold", "expert"),
):
    result = frame.copy()
    result["selected_native_top"] = False
    result["selected_risk_top"] = False
    for _, local in result.groupby(list(group_columns), sort=False):
        count = max(1, int(np.ceil(len(local) * float(fraction))))
        result.loc[
            local["native_confidence"].nlargest(count).index,
            "selected_native_top",
        ] = True
        result.loc[
            local["safe_gain_score"].nlargest(count).index,
            "selected_risk_top",
        ] = True
    return result


def selection_metric_row(frame, selected_column, config):
    selected = frame[selected_column].astype(bool).to_numpy()
    gain = frame["gain_vs_baseline"].to_numpy(float)
    error = frame["absolute_error"].to_numpy(float)
    membership = frame["membership"].to_numpy(float)
    if not selected.any():
        return {
            "sample_count": len(frame),
            "selected_count": 0,
            "coverage": 0.0,
            "mean_gain_vs_baseline": float("nan"),
            "win_rate": float("nan"),
            "large_harm_rate_030": float("nan"),
            "mean_absolute_error": float("nan"),
            "membership_rate": float("nan"),
        }
    return {
        "sample_count": len(frame),
        "selected_count": int(selected.sum()),
        "coverage": float(selected.mean()),
        "mean_gain_vs_baseline": float(gain[selected].mean()),
        "win_rate": float((gain[selected] > 0.0).mean()),
        "large_harm_rate_030": float(
            (
                gain[selected] < -config.large_harm_threshold
            ).mean()
        ),
        "mean_absolute_error": float(error[selected].mean()),
        "membership_rate": float(membership[selected].mean()),
    }


def confidently_wrong_metrics(frame, config):
    confidence = frame["native_confidence"].to_numpy(float)
    error = frame["absolute_error"].to_numpy(float)
    predicted = frame["pred_large_error_050_prob"].to_numpy(float)
    mask = (
        confidence >= config.native_high_confidence_threshold
    ) & (error > config.error_threshold_050)
    count = max(1, int(np.ceil(len(frame) * 0.20)))
    top = np.zeros(len(frame), dtype=bool)
    top[
        np.argsort(-predicted, kind="stable")[:count]
    ] = True
    return {
        "sample_count": len(frame),
        "confidently_wrong_count": int(mask.sum()),
        "confidently_wrong_rate": float(mask.mean()),
        "mean_predicted_large_error_probability": (
            float(predicted[mask].mean()) if mask.any() else 0.0
        ),
        "top20_large_error_risk_recall": (
            float((top & mask).sum() / mask.sum())
            if mask.any()
            else float("nan")
        ),
    }


def group_bootstrap_selected_gain(
    frame, selected_column, repetitions, seed
):
    selected = frame[frame[selected_column].astype(bool)].copy()
    if selected.empty:
        return {
            "mean_gain": float("nan"),
            "gain_ci_low": float("nan"),
            "gain_ci_high": float("nan"),
            "positive_probability": float("nan"),
        }
    groups = (
        selected["outer_fold"].astype(str)
        + "|"
        + selected["group_id"].astype(str)
    ).to_numpy(object)
    gain = selected["gain_vs_baseline"].to_numpy(float)
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    indices = {
        group: np.flatnonzero(groups == group) for group in unique
    }
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(repetitions))
    for repetition in range(int(repetitions)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        total = 0.0
        count = 0
        for group in sampled:
            rows = indices[group]
            total += float(gain[rows].sum())
            count += len(rows)
        values[repetition] = total / max(1, count)
    return {
        "mean_gain": float(gain.mean()),
        "gain_ci_low": float(np.quantile(values, 0.025)),
        "gain_ci_high": float(np.quantile(values, 0.975)),
        "positive_probability": float((values > 0.0).mean()),
    }


__all__ = [
    "AUDIT_VERSION",
    "EXPERT_NAMES",
    "EXPERT_ACTION_INDEX",
    "EXPERT_REGION_INDEX",
    "BINARY_TARGET_NAMES",
    "CONTINUOUS_TARGET_NAMES",
    "ExpertSelfRiskConfigV925",
    "validate_pool",
    "build_expert_features",
    "build_expert_targets",
    "fit_risk_bundle",
    "predict_risk_bundle",
    "crossfit_risk_predictions",
    "fit_outer_risk_bundle",
    "binary_auc",
    "expected_calibration_error",
    "binary_metric_row",
    "continuous_metric_row",
    "upper_bound_metric_row",
    "add_rank_selection_flags",
    "selection_metric_row",
    "confidently_wrong_metrics",
    "group_bootstrap_selected_gain",
]
