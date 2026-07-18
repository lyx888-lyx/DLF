"""Frozen Huber robust center for Stage 14 DCRC."""

from dataclasses import dataclass

import numpy as np


KAPPA = 1.345
MAD_SCALE = 1.4826
SCALE_FLOOR = 1e-6
MAX_ITERATIONS = 20
CONVERGENCE_TOLERANCE = 1e-12
WEIGHT_EPSILON = 1e-12


@dataclass(frozen=True)
class HuberResult:
    center: float
    median: float
    mad: float
    scale: float
    weights: np.ndarray
    iterations: int
    converged: bool
    degenerate_mad: bool
    fallback_reason: str


def huber_center(values):
    """Compute the exact preregistered Huber IRLS center."""
    original = np.asarray(values, dtype=np.float64).reshape(-1)
    if original.size not in (4, 5):
        raise ValueError("DCRC Huber center requires four or five members.")
    if not np.isfinite(original).all():
        raise FloatingPointError("Consensus members contain NaN/Inf.")
    order = np.argsort(original, kind="mergesort")
    sorted_values = original[order]
    median = float(np.median(sorted_values))
    mad = float(np.median(np.abs(sorted_values - median)))
    scale = MAD_SCALE * mad
    weights_sorted = np.ones_like(sorted_values)
    if scale < SCALE_FLOOR:
        center = float(np.mean(sorted_values))
        weights = np.empty_like(weights_sorted)
        weights[order] = weights_sorted
        return HuberResult(
            center=float(np.clip(center, -3.0, 3.0)),
            median=median,
            mad=mad,
            scale=scale,
            weights=weights,
            iterations=0,
            converged=True,
            degenerate_mad=True,
            fallback_reason="degenerate_mad_arithmetic_mean",
        )
    theta = median
    converged = False
    fallback_reason = ""
    iterations = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        standardized = np.abs(sorted_values - theta) / max(
            scale, SCALE_FLOOR
        )
        weights_sorted = np.ones_like(standardized)
        outlier = standardized > KAPPA
        weights_sorted[outlier] = KAPPA / standardized[outlier]
        denominator = float(weights_sorted.sum())
        if denominator <= WEIGHT_EPSILON:
            theta = median
            fallback_reason = "weight_sum_epsilon_median"
            iterations = iteration
            break
        next_theta = float(
            np.dot(weights_sorted, sorted_values) / denominator
        )
        iterations = iteration
        if abs(next_theta - theta) <= CONVERGENCE_TOLERANCE:
            theta = next_theta
            converged = True
            break
        theta = next_theta
    weights = np.empty_like(weights_sorted)
    weights[order] = weights_sorted
    return HuberResult(
        center=float(np.clip(theta, -3.0, 3.0)),
        median=median,
        mad=mad,
        scale=scale,
        weights=weights,
        iterations=iterations,
        converged=converged,
        degenerate_mad=False,
        fallback_reason=fallback_reason,
    )


def huber_centers(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Consensus matrix must be sample-by-member.")
    results = [huber_center(row) for row in matrix]
    return np.asarray([row.center for row in results]), results


def trimmed_middle_three(matrix):
    matrix = np.sort(np.asarray(matrix, dtype=np.float64), axis=1)
    if matrix.shape[1] != 5:
        raise ValueError("Middle-3 diagnostic requires five members.")
    return matrix[:, 1:4].mean(axis=1)
