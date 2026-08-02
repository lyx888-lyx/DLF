"""Support-and-stability expert self-knowledge utilities for V9.26.

V9.26 keeps every V9.19 expert and the V9.21 baseline frozen. It augments the
V9.25 deployment-visible signature with two new evidence classes:

1. historical support: group-excluded nearest neighbours from strict inner OOF
   pools, including their observed expert error and gain versus V9.21;
2. cross-view stability: disagreement among the saved language, audio, vision,
   fusion, anchor, baseline, and specialist predictions.

No sample is routed and no expert prediction is replaced or mixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .expert_self_risk_v925 import (
    EXPERT_NAMES,
    ExpertSelfRiskConfigV925,
    build_expert_features,
    build_expert_targets,
    fit_risk_bundle,
    predict_risk_bundle,
    validate_pool,
)

AUDIT_VERSION = "support_stability_self_knowledge_v926_v1"
PRIMARY_METHOD = "support_stability"
BASE_METHOD = "v925_base"


@dataclass(frozen=True)
class SupportStabilityConfigV926:
    """Pre-registered V9.26 configuration."""

    risk: ExpertSelfRiskConfigV925 = field(
        default_factory=ExpertSelfRiskConfigV925
    )
    neighbor_counts: tuple[int, ...] = (8, 32)
    distance_epsilon: float = 1e-8
    minimum_reference_samples: int = 40
    exclude_same_conversation: bool = True
    required_win_auc: float = 0.60
    required_gain_spearman: float = 0.20
    required_large_harm_auc: float = 0.60
    required_positive_outer_folds: int = 4
    required_safe_gain_ci_low: float = 0.0
    required_gain_over_native_rank: float = 0.002
    required_win_auc_gain_over_v925: float = 0.03
    required_gain_spearman_gain_over_v925: float = 0.05

    def validate(self) -> None:
        self.risk.validate()
        if not self.neighbor_counts:
            raise ValueError("neighbor_counts cannot be empty")
        if any(int(value) <= 0 for value in self.neighbor_counts):
            raise ValueError("neighbor counts must be positive")
        if tuple(sorted(set(self.neighbor_counts))) != self.neighbor_counts:
            raise ValueError("neighbor_counts must be unique and sorted")
        if self.distance_epsilon <= 0.0:
            raise ValueError("distance_epsilon must be positive")
        if self.minimum_reference_samples < max(self.neighbor_counts) + 1:
            raise ValueError("minimum_reference_samples is too small")


def _vector(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _matrix(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 2:
        raise ValueError(f"expected matrix, got {result.shape}")
    return result


def _column_stack(columns: Sequence[np.ndarray]) -> np.ndarray:
    result = np.column_stack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in columns]
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite V9.26 feature")
    return result


def build_cross_view_stability_features(
    payload: Mapping[str, object],
    baseline_prediction,
    expert: str,
) -> dict[str, object]:
    """Construct deterministic cross-view and cross-model stability summaries."""

    n = validate_pool(payload)
    if expert not in EXPERT_NAMES:
        raise ValueError(f"unknown expert: {expert}")
    expert_index = EXPERT_NAMES.index(expert)
    anchor = _vector(payload["anchor"])
    baseline = _vector(baseline_prediction)
    function_space = _matrix(payload["function_space"])
    specialists = _matrix(payload["expert_predictions"])
    confidence = _matrix(payload["expert_confidences"])[
        :, expert_index
    ]
    own = specialists[:, expert_index]
    actions = np.column_stack([anchor, specialists])
    if len(baseline) != n:
        raise ValueError("baseline prediction length mismatch")

    fs_median = np.median(function_space, axis=1)
    fs_std = function_space.std(axis=1)
    fs_range = function_space.max(axis=1) - function_space.min(axis=1)
    fs_mad = np.median(
        np.abs(function_space - fs_median[:, None]), axis=1
    )
    fs_positive = (function_space > 0.0).mean(axis=1)
    fs_negative = (function_space < 0.0).mean(axis=1)
    fs_sign_agreement = np.maximum(fs_positive, fs_negative)

    action_mean = actions.mean(axis=1)
    action_median = np.median(actions, axis=1)
    action_std = actions.std(axis=1)
    action_range = actions.max(axis=1) - actions.min(axis=1)
    action_mad = np.median(
        np.abs(actions - action_median[:, None]), axis=1
    )
    action_positive = (actions > 0.0).mean(axis=1)
    action_negative = (actions < 0.0).mean(axis=1)
    action_sign_agreement = np.maximum(action_positive, action_negative)

    own_direction = np.sign(own - baseline)
    view_direction = np.sign(function_space - baseline[:, None])
    direction_agreement = (
        view_direction == own_direction[:, None]
    ).mean(axis=1)
    view_near_own = (
        np.abs(function_space - own[:, None]) <= 0.25
    ).mean(axis=1)
    action_near_own = (
        np.abs(actions - own[:, None]) <= 0.25
    ).mean(axis=1)

    other = np.delete(specialists, expert_index, axis=1)
    other_gap = np.abs(other - own[:, None])
    sorted_actions = np.sort(actions, axis=1)
    action_gaps = np.diff(sorted_actions, axis=1)

    names = [
        "stability_fs_std",
        "stability_fs_range",
        "stability_fs_mad",
        "stability_fs_sign_agreement",
        "stability_fs_anchor_mean_abs_gap",
        "stability_fs_baseline_mean_abs_gap",
        "stability_fs_own_mean_abs_gap",
        "stability_fs_own_max_abs_gap",
        "stability_fs_own_near_fraction_025",
        "stability_view_direction_agreement",
        "stability_action_std",
        "stability_action_range",
        "stability_action_mad",
        "stability_action_sign_agreement",
        "stability_action_own_near_fraction_025",
        "stability_own_to_action_median",
        "stability_own_to_action_mean",
        "stability_own_to_nearest_other",
        "stability_own_to_mean_other",
        "stability_min_action_gap",
        "stability_mean_action_gap",
        "stability_confidence_over_fs_instability",
        "stability_confidence_over_action_instability",
    ]
    columns = [
        fs_std,
        fs_range,
        fs_mad,
        fs_sign_agreement,
        np.abs(function_space - anchor[:, None]).mean(axis=1),
        np.abs(function_space - baseline[:, None]).mean(axis=1),
        np.abs(function_space - own[:, None]).mean(axis=1),
        np.abs(function_space - own[:, None]).max(axis=1),
        view_near_own,
        direction_agreement,
        action_std,
        action_range,
        action_mad,
        action_sign_agreement,
        action_near_own,
        np.abs(own - action_median),
        np.abs(own - action_mean),
        other_gap.min(axis=1),
        other_gap.mean(axis=1),
        action_gaps.min(axis=1),
        action_gaps.mean(axis=1),
        confidence / (1.0 + fs_std),
        confidence / (1.0 + action_std),
    ]
    return {"matrix": _column_stack(columns), "feature_names": names}


def build_support_representation(
    payload: Mapping[str, object],
    baseline_prediction,
    expert: str,
) -> dict[str, object]:
    """Build a compact deployment-visible representation for neighbour search."""

    n = validate_pool(payload)
    if expert not in EXPERT_NAMES:
        raise ValueError(f"unknown expert: {expert}")
    expert_index = EXPERT_NAMES.index(expert)
    anchor = _vector(payload["anchor"])
    baseline = _vector(baseline_prediction)
    function_space = _matrix(payload["function_space"])
    specialists = _matrix(payload["expert_predictions"])
    confidence = _matrix(payload["expert_confidences"])[
        :, expert_index
    ]
    correction = _matrix(payload["expert_corrections"])[
        :, expert_index
    ]
    own = specialists[:, expert_index]
    actions = np.column_stack([anchor, specialists])
    if len(baseline) != n:
        raise ValueError("baseline prediction length mismatch")

    names = [
        *(f"support_function_space_{index}" for index in range(4)),
        "support_anchor",
        "support_baseline",
        "support_own_prediction",
        "support_native_confidence",
        "support_own_correction",
        "support_own_abs_correction",
        "support_own_minus_anchor",
        "support_own_minus_baseline",
        "support_action_mean",
        "support_action_std",
        "support_action_range",
    ]
    matrix = _column_stack(
        [
            *(function_space[:, index] for index in range(4)),
            anchor,
            baseline,
            own,
            confidence,
            correction,
            np.abs(correction),
            own - anchor,
            own - baseline,
            actions.mean(axis=1),
            actions.std(axis=1),
            actions.max(axis=1) - actions.min(axis=1),
        ]
    )
    return {
        "matrix": matrix,
        "feature_names": names,
        "native_confidence": confidence,
    }


def _fit_support_library(
    representation,
    targets: Mapping[str, np.ndarray],
    native_confidence,
    group_ids,
    config: SupportStabilityConfigV926,
) -> dict[str, object]:
    value = np.asarray(representation, dtype=np.float64)
    if len(value) < config.minimum_reference_samples:
        raise ValueError("support reference set is too small")
    mean = value.mean(axis=0)
    scale = value.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    target_copy = {
        name: np.asarray(values, dtype=np.float64).reshape(-1)
        for name, values in targets.items()
    }
    if any(len(values) != len(value) for values in target_copy.values()):
        raise ValueError("support target length mismatch")
    return {
        "version": AUDIT_VERSION,
        "mean": mean,
        "scale": scale,
        "reference": value,
        "reference_standardized": (value - mean) / scale,
        "targets": target_copy,
        "native_confidence": np.asarray(
            native_confidence, dtype=np.float64
        ).reshape(-1),
        "group_ids": np.asarray(
            [str(value) for value in group_ids], dtype=object
        ),
        "neighbor_counts": tuple(config.neighbor_counts),
    }


def _nearest_indices(
    query,
    query_groups,
    library: Mapping[str, object],
    config: SupportStabilityConfigV926,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = (
        np.asarray(query, dtype=np.float64) - library["mean"]
    ) / library["scale"]
    reference = np.asarray(
        library["reference_standardized"], dtype=np.float64
    )
    distance = np.sqrt(
        np.maximum(
            np.square(value[:, None, :] - reference[None, :, :]).mean(
                axis=2
            ),
            0.0,
        )
    )
    excluded = np.zeros_like(distance, dtype=bool)
    if config.exclude_same_conversation and query_groups is not None:
        query_group_array = np.asarray(
            [str(value) for value in query_groups], dtype=object
        )
        excluded = (
            query_group_array[:, None]
            == np.asarray(library["group_ids"], dtype=object)[None, :]
        )
        distance[excluded] = np.inf
    maximum_k = min(max(config.neighbor_counts), distance.shape[1])
    finite_count = np.isfinite(distance).sum(axis=1)
    if np.any(finite_count < maximum_k):
        raise RuntimeError(
            "not enough group-excluded neighbours for V9.26"
        )
    partition = np.argpartition(
        distance, kth=maximum_k - 1, axis=1
    )[:, :maximum_k]
    local_distance = np.take_along_axis(distance, partition, axis=1)
    order = np.argsort(local_distance, axis=1, kind="stable")
    indices = np.take_along_axis(partition, order, axis=1)
    distances = np.take_along_axis(distance, indices, axis=1)
    return indices, distances, excluded


def build_historical_support_features(
    query_representation,
    query_group_ids,
    library: Mapping[str, object],
    config: SupportStabilityConfigV926,
) -> dict[str, object]:
    """Summarize group-excluded historical neighbours and their outcomes."""

    indices, distances, excluded = _nearest_indices(
        query_representation, query_group_ids, library, config
    )
    targets = library["targets"]
    native = np.asarray(library["native_confidence"], dtype=np.float64)
    columns: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, value) -> None:
        names.append(name)
        columns.append(np.asarray(value, dtype=np.float64).reshape(-1))

    for requested_k in config.neighbor_counts:
        k = min(int(requested_k), indices.shape[1])
        local_index = indices[:, :k]
        local_distance = distances[:, :k]
        weights = 1.0 / (local_distance + config.distance_epsilon)
        weights /= weights.sum(axis=1, keepdims=True)

        membership = np.asarray(targets["membership"])[local_index]
        error = np.asarray(targets["absolute_error"])[local_index]
        gain = np.asarray(targets["gain_vs_baseline"])[local_index]
        win = np.asarray(targets["win_vs_baseline"])[local_index]
        harm = np.asarray(targets["large_harm_030"])[local_index]
        large_gain = np.asarray(targets["large_gain_010"])[local_index]
        local_native = native[local_index]

        prefix = f"support_k{k}"
        add(f"{prefix}_distance_min", local_distance.min(axis=1))
        add(f"{prefix}_distance_mean", local_distance.mean(axis=1))
        add(f"{prefix}_distance_max", local_distance.max(axis=1))
        add(f"{prefix}_membership_rate", membership.mean(axis=1))
        add(f"{prefix}_native_confidence_mean", local_native.mean(axis=1))
        add(f"{prefix}_absolute_error_mean", error.mean(axis=1))
        add(f"{prefix}_absolute_error_median", np.median(error, axis=1))
        add(f"{prefix}_absolute_error_q80", np.quantile(error, 0.8, axis=1))
        add(f"{prefix}_gain_mean", gain.mean(axis=1))
        add(f"{prefix}_gain_median", np.median(gain, axis=1))
        add(f"{prefix}_win_rate", win.mean(axis=1))
        add(f"{prefix}_large_gain_rate", large_gain.mean(axis=1))
        add(f"{prefix}_large_harm_rate", harm.mean(axis=1))
        add(
            f"{prefix}_weighted_absolute_error",
            (weights * error).sum(axis=1),
        )
        add(f"{prefix}_weighted_gain", (weights * gain).sum(axis=1))
        add(f"{prefix}_weighted_win", (weights * win).sum(axis=1))
        add(f"{prefix}_weighted_harm", (weights * harm).sum(axis=1))

    return {
        "matrix": _column_stack(columns),
        "feature_names": names,
        "minimum_neighbour_distance": distances[:, 0],
        "same_group_exclusion_count": excluded.sum(axis=1),
    }


def _subset(targets: Mapping[str, np.ndarray], indices) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value)[indices]
        for name, value in targets.items()
    }


def _augment_features(
    base_features,
    stability_features,
    support_representation,
    group_ids,
    library,
    config,
) -> dict[str, object]:
    support = build_historical_support_features(
        support_representation,
        group_ids,
        library,
        config,
    )
    matrix = np.column_stack(
        [base_features, stability_features, support["matrix"]]
    )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("non-finite augmented V9.26 matrix")
    return {"matrix": matrix, "support": support}


def prepare_expert_inputs(
    payload: Mapping[str, object],
    baseline_prediction,
    expert: str,
    config: SupportStabilityConfigV926,
) -> dict[str, object]:
    config.validate()
    base = build_expert_features(payload, baseline_prediction, expert)
    stability = build_cross_view_stability_features(
        payload, baseline_prediction, expert
    )
    representation = build_support_representation(
        payload, baseline_prediction, expert
    )
    targets = build_expert_targets(
        payload, baseline_prediction, expert, config.risk
    )
    return {
        "base": base,
        "stability": stability,
        "representation": representation,
        "targets": targets,
        "group_ids": np.asarray(
            [str(value) for value in payload["group_ids"]], dtype=object
        ),
        "feature_names": [
            *base["feature_names"],
            *stability["feature_names"],
        ],
    }


def crossfit_support_stability_predictions(
    payload: Mapping[str, object],
    baseline_prediction,
    expert: str,
    fold_index,
    config: SupportStabilityConfigV926,
) -> dict[str, object]:
    """Strict tri-split risk cross-fit with split-local support libraries."""

    inputs = prepare_expert_inputs(
        payload, baseline_prediction, expert, config
    )
    folds = np.asarray(fold_index, dtype=np.int64).reshape(-1)
    unique = sorted(int(value) for value in np.unique(folds))
    if len(unique) < 3:
        raise ValueError("at least three risk folds are required")
    outputs = None
    rows = []
    final_feature_names = None
    for position, holdout_fold in enumerate(unique):
        calibration_fold = unique[(position + 1) % len(unique)]
        holdout = np.flatnonzero(folds == holdout_fold)
        calibration = np.flatnonzero(folds == calibration_fold)
        fit = np.flatnonzero(
            (folds != holdout_fold) & (folds != calibration_fold)
        )
        if min(len(fit), len(calibration), len(holdout)) == 0:
            raise RuntimeError("empty V9.26 tri-split")

        library = _fit_support_library(
            inputs["representation"]["matrix"][fit],
            _subset(inputs["targets"], fit),
            inputs["representation"]["native_confidence"][fit],
            inputs["group_ids"][fit],
            config,
        )
        fit_aug = _augment_features(
            inputs["base"]["matrix"][fit],
            inputs["stability"]["matrix"][fit],
            inputs["representation"]["matrix"][fit],
            inputs["group_ids"][fit],
            library,
            config,
        )
        calibration_aug = _augment_features(
            inputs["base"]["matrix"][calibration],
            inputs["stability"]["matrix"][calibration],
            inputs["representation"]["matrix"][calibration],
            inputs["group_ids"][calibration],
            library,
            config,
        )
        holdout_aug = _augment_features(
            inputs["base"]["matrix"][holdout],
            inputs["stability"]["matrix"][holdout],
            inputs["representation"]["matrix"][holdout],
            inputs["group_ids"][holdout],
            library,
            config,
        )
        bundle = fit_risk_bundle(
            fit_aug["matrix"],
            _subset(inputs["targets"], fit),
            calibration_aug["matrix"],
            _subset(inputs["targets"], calibration),
            config.risk,
        )
        local = predict_risk_bundle(
            bundle, holdout_aug["matrix"], config.risk
        )
        if outputs is None:
            outputs = {
                name: np.full(len(folds), np.nan, dtype=np.float64)
                for name in local
            }
        for name, value in local.items():
            outputs[name][holdout] = value
        support_names = holdout_aug["support"]["feature_names"]
        final_feature_names = [*inputs["feature_names"], *support_names]
        rows.append(
            {
                "holdout_fold": holdout_fold,
                "calibration_fold": calibration_fold,
                "fit_sample_count": len(fit),
                "calibration_sample_count": len(calibration),
                "holdout_sample_count": len(holdout),
                "support_reference_count": len(fit),
                "feature_count": len(final_feature_names),
                "holdout_min_support_distance_mean": float(
                    holdout_aug["support"][
                        "minimum_neighbour_distance"
                    ].mean()
                ),
            }
        )
    if outputs is None or any(
        not np.isfinite(value).all() for value in outputs.values()
    ):
        raise FloatingPointError("incomplete V9.26 cross-fit predictions")
    return {
        "predictions": outputs,
        "split_rows": rows,
        "feature_names": final_feature_names,
        "inputs": inputs,
    }


def fit_outer_support_stability_bundle(
    inner_payload: Mapping[str, object],
    inner_baseline_prediction,
    expert: str,
    fold_index,
    config: SupportStabilityConfigV926,
) -> dict[str, object]:
    """Fit one outer-deployment V9.26 bundle using inner OOF only."""

    inputs = prepare_expert_inputs(
        inner_payload, inner_baseline_prediction, expert, config
    )
    folds = np.asarray(fold_index, dtype=np.int64).reshape(-1)
    unique = sorted(int(value) for value in np.unique(folds))
    if len(unique) < 2:
        raise ValueError("outer risk fit needs at least two folds")
    calibration_fold = unique[0]
    calibration = np.flatnonzero(folds == calibration_fold)
    fit = np.flatnonzero(folds != calibration_fold)
    library = _fit_support_library(
        inputs["representation"]["matrix"][fit],
        _subset(inputs["targets"], fit),
        inputs["representation"]["native_confidence"][fit],
        inputs["group_ids"][fit],
        config,
    )
    fit_aug = _augment_features(
        inputs["base"]["matrix"][fit],
        inputs["stability"]["matrix"][fit],
        inputs["representation"]["matrix"][fit],
        inputs["group_ids"][fit],
        library,
        config,
    )
    calibration_aug = _augment_features(
        inputs["base"]["matrix"][calibration],
        inputs["stability"]["matrix"][calibration],
        inputs["representation"]["matrix"][calibration],
        inputs["group_ids"][calibration],
        library,
        config,
    )
    bundle = fit_risk_bundle(
        fit_aug["matrix"],
        _subset(inputs["targets"], fit),
        calibration_aug["matrix"],
        _subset(inputs["targets"], calibration),
        config.risk,
    )
    feature_names = [
        *inputs["feature_names"],
        *fit_aug["support"]["feature_names"],
    ]
    return {
        "version": AUDIT_VERSION,
        "bundle": bundle,
        "support_library": library,
        "feature_names": feature_names,
        "calibration_fold": calibration_fold,
        "fit_sample_count": len(fit),
        "calibration_sample_count": len(calibration),
        "support_reference_count": len(fit),
        "config": asdict(config),
    }


def predict_outer_support_stability(
    fitted: Mapping[str, object],
    outer_payload: Mapping[str, object],
    outer_baseline_prediction,
    expert: str,
    config: SupportStabilityConfigV926,
) -> dict[str, object]:
    """Predict V9.26 risks on an unseen outer conversation holdout."""

    inputs = prepare_expert_inputs(
        outer_payload, outer_baseline_prediction, expert, config
    )
    augmented = _augment_features(
        inputs["base"]["matrix"],
        inputs["stability"]["matrix"],
        inputs["representation"]["matrix"],
        inputs["group_ids"],
        fitted["support_library"],
        config,
    )
    expected_feature_count = len(fitted["feature_names"])
    if augmented["matrix"].shape[1] != expected_feature_count:
        raise RuntimeError("V9.26 feature schema changed at outer inference")
    return {
        "predictions": predict_risk_bundle(
            fitted["bundle"], augmented["matrix"], config.risk
        ),
        "inputs": inputs,
        "support": augmented["support"],
    }


def average_precision(target, probability) -> float:
    """Average precision without a scikit-learn dependency."""

    y = np.asarray(target, dtype=np.float64).reshape(-1) > 0.5
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ranked = y[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


__all__ = [
    "AUDIT_VERSION",
    "PRIMARY_METHOD",
    "BASE_METHOD",
    "SupportStabilityConfigV926",
    "build_cross_view_stability_features",
    "build_support_representation",
    "build_historical_support_features",
    "prepare_expert_inputs",
    "crossfit_support_stability_predictions",
    "fit_outer_support_stability_bundle",
    "predict_outer_support_stability",
    "average_precision",
]
