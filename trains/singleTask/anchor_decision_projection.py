"""Label-free decision-preserving projection for Stage 9B ADPEP.

The decision mapping mirrors the frozen Stage 9A regression evaluator in
``missing_utils.regression_metrics``:

* Acc7: ``np.round(np.clip(prediction, -3, 3))``
* Acc5: ``np.round(np.clip(prediction, -2, 2))``
* Acc2/F1 prediction class: ``prediction > 0``

All operations use float32 because Stage 9A converts saved predictions to
float32 torch tensors before invoking the evaluator.
"""
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


SUPPORTED_DATASETS = ("mosi", "mosei")
SUPPORTED_VARIANTS = ("adpep57", "adpep_all")


@dataclass(frozen=True)
class DecisionInterval:
    lower: np.float32
    upper: np.float32
    lower_closed: bool
    upper_closed: bool

    def contains(self, value) -> bool:
        value = np.float32(value)
        lower_ok = value > self.lower or (
            self.lower_closed and value == self.lower
        )
        upper_ok = value < self.upper or (
            self.upper_closed and value == self.upper
        )
        return bool(lower_ok and upper_ok)


@dataclass(frozen=True)
class ProjectionResult:
    value: np.float32
    lower: np.float32
    upper: np.float32
    lower_closed: bool
    upper_closed: bool
    pe5_already_feasible: bool
    boundary_adjusted: bool
    fallback_to_anchor: bool
    fallback_reason: str


def _as_float32(values):
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError("Predictions contain NaN/Inf.")
    return array


def evaluator_decisions(values, dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact Stage 9A Acc7/Acc5/Acc2 prediction decisions."""
    if str(dataset).lower() not in SUPPORTED_DATASETS:
        raise ValueError("Unsupported dataset: {}.".format(dataset))
    prediction = _as_float32(values)
    acc7 = np.round(np.clip(prediction, np.float32(-3), np.float32(3)))
    acc5 = np.round(np.clip(prediction, np.float32(-2), np.float32(2)))
    acc2 = prediction > np.float32(0)
    return acc7, acc5, acc2


def decision_signature(value, dataset, variant):
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError("Unsupported ADPEP variant: {}.".format(variant))
    acc7, acc5, acc2 = evaluator_decisions([value], dataset)
    signature = (int(acc7[0]), int(acc5[0]))
    if variant == "adpep_all":
        signature += (bool(acc2[0]),)
    return signature


def _round_class_interval(decision, limit):
    decision = int(decision)
    if decision < -limit or decision > limit:
        raise ValueError("Rounded decision is outside the clipped class range.")
    lower = np.float32(-np.inf) if decision == -limit else np.float32(decision - 0.5)
    upper = np.float32(np.inf) if decision == limit else np.float32(decision + 0.5)
    # np.round uses round-half-to-even. A half-integer belongs to the even class.
    closed = decision % 2 == 0
    return DecisionInterval(
        lower=lower,
        upper=upper,
        lower_closed=False if np.isneginf(lower) else closed,
        upper_closed=False if np.isposinf(upper) else closed,
    )


def _binary_interval(positive):
    if positive:
        return DecisionInterval(
            lower=np.float32(0),
            upper=np.float32(np.inf),
            lower_closed=False,
            upper_closed=False,
        )
    return DecisionInterval(
        lower=np.float32(-np.inf),
        upper=np.float32(0),
        lower_closed=False,
        upper_closed=True,
    )


def _intersect(left, right):
    if left.lower > right.lower:
        lower, lower_closed = left.lower, left.lower_closed
    elif right.lower > left.lower:
        lower, lower_closed = right.lower, right.lower_closed
    else:
        lower = left.lower
        lower_closed = left.lower_closed and right.lower_closed

    if left.upper < right.upper:
        upper, upper_closed = left.upper, left.upper_closed
    elif right.upper < left.upper:
        upper, upper_closed = right.upper, right.upper_closed
    else:
        upper = left.upper
        upper_closed = left.upper_closed and right.upper_closed

    if lower > upper or (
        lower == upper and not (lower_closed and upper_closed)
    ):
        raise RuntimeError("Decision constraints have an empty intersection.")
    return DecisionInterval(lower, upper, lower_closed, upper_closed)


def safe_interval(anchor_prediction, dataset, variant):
    """Return the maximal connected interval containing the anchor decision."""
    signature = decision_signature(anchor_prediction, dataset, variant)
    interval = _intersect(
        _round_class_interval(signature[0], 3),
        _round_class_interval(signature[1], 2),
    )
    if variant == "adpep_all":
        interval = _intersect(interval, _binary_interval(signature[2]))
    if not interval.contains(np.float32(anchor_prediction)):
        raise RuntimeError("Constructed safe interval does not contain anchor.")
    return interval


def _interior_boundary(boundary, toward_positive):
    target = np.float32(np.inf if toward_positive else -np.inf)
    return np.nextafter(np.float32(boundary), target, dtype=np.float32)


def project_prediction(anchor_prediction, ensemble_prediction, dataset, variant):
    """Project one PE5 prediction without accepting or reading a label."""
    anchor = _as_float32([anchor_prediction])[0]
    ensemble = _as_float32([ensemble_prediction])[0]
    interval = safe_interval(anchor, dataset, variant)
    feasible = interval.contains(ensemble)
    boundary_adjusted = False

    if feasible:
        candidate = ensemble
    elif ensemble < interval.lower or (
        ensemble == interval.lower and not interval.lower_closed
    ):
        candidate = interval.lower
        if not interval.lower_closed:
            candidate = _interior_boundary(candidate, toward_positive=True)
            boundary_adjusted = True
    else:
        candidate = interval.upper
        if not interval.upper_closed:
            candidate = _interior_boundary(candidate, toward_positive=False)
            boundary_adjusted = True

    anchor_signature = decision_signature(anchor, dataset, variant)
    candidate_signature = decision_signature(candidate, dataset, variant)
    fallback = False
    reason = ""
    if candidate_signature != anchor_signature:
        adjusted = np.nextafter(candidate, anchor, dtype=np.float32)
        candidate = adjusted
        boundary_adjusted = True
        candidate_signature = decision_signature(candidate, dataset, variant)
    if candidate_signature != anchor_signature:
        candidate = anchor
        fallback = True
        reason = "post_projection_decision_mismatch"
    if decision_signature(candidate, dataset, variant) != anchor_signature:
        raise RuntimeError("Fallback anchor does not preserve its own decisions.")

    return ProjectionResult(
        value=np.float32(candidate),
        lower=interval.lower,
        upper=interval.upper,
        lower_closed=interval.lower_closed,
        upper_closed=interval.upper_closed,
        pe5_already_feasible=feasible,
        boundary_adjusted=boundary_adjusted,
        fallback_to_anchor=fallback,
        fallback_reason=reason,
    )


def project_array(anchor_predictions, ensemble_predictions, dataset, variant):
    """Vector convenience wrapper; still intentionally has no label argument."""
    anchors = _as_float32(anchor_predictions).reshape(-1)
    ensembles = _as_float32(ensemble_predictions).reshape(-1)
    if anchors.shape != ensembles.shape:
        raise ValueError("Anchor and ensemble prediction shapes differ.")
    results = [
        project_prediction(anchor, ensemble, dataset, variant)
        for anchor, ensemble in zip(anchors, ensembles)
    ]
    return np.asarray([result.value for result in results], dtype=np.float32), results


def select_anchor_seed(validation_rows, seeds):
    """Select solely by validation J, breaking ties by lower numeric seed."""
    fixed = tuple(int(seed) for seed in seeds)
    if len(fixed) != 5 or len(set(fixed)) != 5:
        raise ValueError("ADPEP requires exactly five unique fixed seeds.")
    candidates = {}
    for row in validation_rows:
        seed = int(row["Seed"])
        if seed not in fixed:
            continue
        value = float(row["J"])
        if not np.isfinite(value):
            raise FloatingPointError("Validation J contains NaN/Inf.")
        if seed in candidates:
            raise ValueError("Duplicate validation J for seed {}.".format(seed))
        candidates[seed] = value
    if set(candidates) != set(fixed):
        raise ValueError("Validation J is incomplete for the fixed seed set.")
    return min(fixed, key=lambda seed: (candidates[seed], seed))


def retention_ratio(anchor_value, pe5_value, method_value, lower_is_better=True):
    if lower_is_better:
        pe5_gain = float(anchor_value) - float(pe5_value)
        method_gain = float(anchor_value) - float(method_value)
    else:
        pe5_gain = float(pe5_value) - float(anchor_value)
        method_gain = float(method_value) - float(anchor_value)
    if pe5_gain <= 0:
        return pe5_gain, method_gain, None
    return pe5_gain, method_gain, method_gain / pe5_gain
