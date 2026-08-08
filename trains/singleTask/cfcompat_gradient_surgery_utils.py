"""Asymmetric residual-gradient surgery utilities for CFCompatKD v12.

v11 found that the all-sample supervised missing-label gradient strongly
conflicts with the OOF Teacher-beneficial loss gradient, while the selective
DISTILL/PRESERVE gradient has the opposite alignment.  v12 therefore preserves
both objectives but removes only the component of the accumulated supervised
missing gradient that is anti-aligned with the accumulated selective gradient.

The operation is asymmetric by construction::

    if <g_sup, g_sel> < 0:
        g_sup_safe = g_sup - <g_sup, g_sel> / ||g_sel||^2 * g_sel
    else:
        g_sup_safe = g_sup
    g_update = g_sup_safe + g_sel

No label-based inference routing is introduced.  The surgery is a Train-only
optimizer operation on the residual-head parameters.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from .cfcompat_conservative_crossfit_residual_utils import (
    BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
    CONSENSUS_MIN_AGREE,
    DEV_SEED,
    J_MAX_DEGRADATION_VS_V8,
    NEAR_OPTIMAL_REL_TOL,
    NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
    N_FOLDS,
    OVERALL_NTR_MAX_DEGRADATION_VS_V4,
    RESIDUAL_INIT_SEED,
)


VERSION = "cfcompat_gradient_surgery_valid_screen_v12"
METHOD = "DLF-Frozen-S0-Asymmetric-Gradient-Surgery-CFCompatKD-v12"
OUTPUT_TAG = "cfcompat_gradient_surgery_v12"
RUN = "frozen_s0_asymmetric_gradient_surgery"
SURGERY_EPS = 1e-12


def jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def zero_gradient_tuple(parameters: Sequence[torch.nn.Parameter]) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.zeros_like(parameter) for parameter in parameters)


def detached_gradient_tuple(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    retain_graph: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Return a dense detached gradient tuple, replacing unused entries by zero."""
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    result = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            result.append(torch.zeros_like(parameter))
        else:
            result.append(gradient.detach().clone())
    return tuple(result)


def add_gradient_tuples(
    first: Sequence[torch.Tensor], second: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    if len(first) != len(second):
        raise ValueError("Gradient tuples have different lengths.")
    return tuple(a + b for a, b in zip(first, second))


def gradient_dot(first: Sequence[torch.Tensor], second: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(first) != len(second):
        raise ValueError("Gradient tuples have different lengths.")
    if not first:
        raise ValueError("Gradient tuple is empty.")
    result = torch.zeros((), device=first[0].device, dtype=first[0].dtype)
    for a, b in zip(first, second):
        result = result + torch.sum(a * b)
    return result


def gradient_norm_sq(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return gradient_dot(gradients, gradients)


def asymmetric_project_supervised(
    supervised: Sequence[torch.Tensor],
    selective: Sequence[torch.Tensor],
    eps: float = SURGERY_EPS,
):
    """Project only the supervised gradient when it conflicts with selective.

    Returns ``(projected_supervised, update_gradient, diagnostics)``.  A positive
    dot means the two objectives locally agree under gradient descent; a
    negative dot triggers surgery.  The post-projection dot is numerically zero
    or positive, while the selective gradient itself is never changed.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if len(supervised) != len(selective) or not supervised:
        raise ValueError("Gradient tuples must be non-empty and aligned.")

    dot_before_t = gradient_dot(supervised, selective)
    sup_sq_t = gradient_norm_sq(supervised)
    sel_sq_t = gradient_norm_sq(selective)
    dot_before = float(dot_before_t.detach().cpu())
    sup_sq = float(sup_sq_t.detach().cpu())
    sel_sq = float(sel_sq_t.detach().cpu())
    conflict = bool(dot_before < 0.0 and sel_sq > eps)

    if conflict:
        coefficient_t = dot_before_t / sel_sq_t
        projected = tuple(
            g_sup - coefficient_t * g_sel
            for g_sup, g_sel in zip(supervised, selective)
        )
        coefficient = float(coefficient_t.detach().cpu())
    else:
        projected = tuple(g.clone() for g in supervised)
        coefficient = 0.0

    update = add_gradient_tuples(projected, selective)
    post_dot = float(gradient_dot(projected, selective).detach().cpu())
    projected_sq = float(gradient_norm_sq(projected).detach().cpu())
    update_sq = float(gradient_norm_sq(update).detach().cpu())
    denominator = (sup_sq * sel_sq) ** 0.5
    cosine = float(dot_before / denominator) if denominator > 0.0 else 0.0
    removed_fraction = (
        float(max(0.0, 1.0 - (projected_sq / sup_sq) ** 0.5))
        if sup_sq > 0.0
        else 0.0
    )

    if conflict and post_dot < -1e-6:
        raise RuntimeError(
            "Gradient surgery failed to remove conflict: post_dot={}".format(post_dot)
        )
    diagnostics = {
        "conflict": conflict,
        "gradient_dot_before": dot_before,
        "gradient_cosine_before": cosine,
        "supervised_gradient_l2": float(sup_sq ** 0.5),
        "selective_gradient_l2": float(sel_sq ** 0.5),
        "projected_supervised_gradient_l2": float(projected_sq ** 0.5),
        "update_gradient_l2": float(update_sq ** 0.5),
        "projection_coefficient": coefficient,
        "post_projection_dot": post_dot,
        "supervised_l2_removed_fraction": removed_fraction,
    }
    return projected, update, diagnostics


def assign_gradient_tuple(
    parameters: Sequence[torch.nn.Parameter], gradients: Sequence[torch.Tensor]
) -> None:
    if len(parameters) != len(gradients):
        raise ValueError("Parameters and gradients differ in length.")
    for parameter, gradient in zip(parameters, gradients):
        if gradient.shape != parameter.shape:
            raise ValueError("Gradient shape mismatch.")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite gradient produced by v12 surgery.")
        parameter.grad = gradient.detach().clone()


def summarize_surgery_windows(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("No gradient-surgery window rows supplied.")
    conflicts = np.asarray([bool(row["conflict"]) for row in rows], dtype=bool)
    cosine = np.asarray([float(row["gradient_cosine_before"]) for row in rows], dtype=float)
    removed = np.asarray(
        [float(row["supervised_l2_removed_fraction"]) for row in rows], dtype=float
    )
    post_dot = np.asarray([float(row["post_projection_dot"]) for row in rows], dtype=float)
    if not np.isfinite(cosine).all() or not np.isfinite(removed).all() or not np.isfinite(post_dot).all():
        raise FloatingPointError("Non-finite surgery diagnostics.")
    return {
        "window_count": int(len(rows)),
        "conflict_window_count": int(conflicts.sum()),
        "conflict_window_fraction": float(conflicts.mean()),
        "mean_pre_surgery_cosine": float(cosine.mean()),
        "mean_conflict_cosine": float(cosine[conflicts].mean()) if conflicts.any() else 0.0,
        "mean_supervised_l2_removed_fraction": float(removed.mean()),
        "mean_removed_fraction_on_conflict": float(removed[conflicts].mean()) if conflicts.any() else 0.0,
        "min_post_projection_dot": float(post_dot.min()),
    }


def development_signal_gate(
    candidate_j: float,
    v8_j: float,
    candidate_transfer: Mapping,
    v8_transfer: Mapping,
    v4_transfer: Mapping,
) -> dict:
    """Keep the frozen v9/v10 promotion criteria unchanged for v12."""
    beneficial_degradation = (
        candidate_transfer["teacher_beneficial"]["negative_transfer_rate"]
        - v8_transfer["teacher_beneficial"]["negative_transfer_rate"]
    )
    nonbeneficial_reduction = (
        v8_transfer["teacher_nonbeneficial"]["negative_transfer_rate"]
        - candidate_transfer["teacher_nonbeneficial"]["negative_transfer_rate"]
    )
    overall_degradation_vs_v4 = (
        candidate_transfer["all_missing"]["negative_transfer_rate"]
        - v4_transfer["all_missing"]["negative_transfer_rate"]
    )
    checks = {
        "valid_J_noninferior_to_v8": candidate_j - v8_j <= J_MAX_DEGRADATION_VS_V8,
        "beneficial_teacher_NTR_retained": beneficial_degradation
        <= BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
        "nonbeneficial_teacher_NTR_reduced": nonbeneficial_reduction
        >= NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
        "overall_NTR_not_materially_worse_than_v4": overall_degradation_vs_v4
        <= OVERALL_NTR_MAX_DEGRADATION_VS_V4,
    }
    return {
        "candidate_J": float(candidate_j),
        "v8_J": float(v8_j),
        "delta_J_candidate_minus_v8": float(candidate_j - v8_j),
        "beneficial_teacher_NTR_degradation_vs_v8": float(beneficial_degradation),
        "nonbeneficial_teacher_NTR_reduction_vs_v8": float(nonbeneficial_reduction),
        "overall_NTR_degradation_vs_v4": float(overall_degradation_vs_v4),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "thresholds": {
            "J_max_degradation_vs_v8": J_MAX_DEGRADATION_VS_V8,
            "beneficial_NTR_max_degradation_vs_v8": BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
            "nonbeneficial_NTR_reduction_required_vs_v8": NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
            "overall_NTR_max_degradation_vs_v4": OVERALL_NTR_MAX_DEGRADATION_VS_V4,
        },
    }


__all__ = [
    "BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8",
    "CONSENSUS_MIN_AGREE",
    "DEV_SEED",
    "J_MAX_DEGRADATION_VS_V8",
    "METHOD",
    "NEAR_OPTIMAL_REL_TOL",
    "NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8",
    "N_FOLDS",
    "OUTPUT_TAG",
    "OVERALL_NTR_MAX_DEGRADATION_VS_V4",
    "RESIDUAL_INIT_SEED",
    "RUN",
    "SURGERY_EPS",
    "VERSION",
    "add_gradient_tuples",
    "asymmetric_project_supervised",
    "assign_gradient_tuple",
    "detached_gradient_tuple",
    "development_signal_gate",
    "gradient_dot",
    "gradient_norm_sq",
    "jsonable",
    "summarize_surgery_windows",
    "zero_gradient_tuple",
]
