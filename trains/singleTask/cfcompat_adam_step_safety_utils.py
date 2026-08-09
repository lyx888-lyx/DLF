"""Train-only actual-Adam-step functional safety utilities for CFCompatKD v13.

v13 keeps the v12 residual model/objective and raw gradient surgery unchanged,
but protects the *parameter displacement that Adam actually proposes*.  Safety
anchors are fixed Train-only sentinel groups selected before fold training:
Teacher-beneficial events and Frozen-S0-beneficial events, each using the
historical 0.02 margins.  No Valid/Test information enters sentinel selection.

For each missing mode, the proposed Adam displacement is projected onto the
intersection of up to two first-order safety halfspaces::

    <delta_theta, grad L_teacher_beneficial> <= 0
    <delta_theta, grad L_s0_beneficial>      <= 0

The closest feasible displacement in Euclidean parameter space is used.  With
two constraints this projection is solved by a deterministic active-set check;
zero displacement is retained only as a numerical fallback and is always
feasible.  Adam's internal first/second-moment state is left untouched, so each
future proposed step still reflects the original optimizer history and is
checked again before it reaches the model parameters.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from .cfcompat_adapter_isolation_utils import S0_PROTECT_MARGIN
from .cfcompat_conservative_crossfit_residual_utils import (
    BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
    CONSENSUS_MIN_AGREE,
    DEV_SEED,
    J_MAX_DEGRADATION_VS_V8,
    NEAR_OPTIMAL_REL_TOL,
    NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
    N_FOLDS,
    OVERALL_NTR_MAX_DEGRADATION_VS_V4,
)
from .cfcompat_regret_preserve_utils import DISTILL_MARGIN


VERSION = "cfcompat_adam_step_functional_safety_valid_screen_v13"
METHOD = "DLF-Frozen-S0-Adam-Step-Functional-Safety-CFCompatKD-v13"
OUTPUT_TAG = "cfcompat_adam_step_safety_v13"
RUN = "frozen_s0_adam_step_functional_safety"

SAFETY_GROUPS = ("TEACHER_BENEFICIAL", "S0_BENEFICIAL")
SAFETY_EPS = 1e-24
# Reuses the long-standing noninferiority convention.  This is frozen before
# seeing v13 Valid results and is used only by the direct-v12 improvement gate.
NONBENEFICIAL_NTR_MAX_DEGRADATION_VS_V12 = 0.03


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


def clone_tensor_tuple(values: Sequence[torch.Tensor]):
    return tuple(value.detach().clone() for value in values)


def flatten_tensor_tuple(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot flatten an empty tensor tuple.")
    return torch.cat([value.detach().reshape(-1).to(torch.float64) for value in values])


def unflatten_like(vector: torch.Tensor, references: Sequence[torch.Tensor]):
    if vector.ndim != 1:
        raise ValueError("Expected a flat vector.")
    result = []
    offset = 0
    for reference in references:
        count = int(reference.numel())
        piece = vector[offset : offset + count]
        if piece.numel() != count:
            raise ValueError("Flat vector is shorter than reference tensors.")
        result.append(piece.reshape(reference.shape).to(device=reference.device, dtype=reference.dtype))
        offset += count
    if offset != int(vector.numel()):
        raise ValueError("Flat vector has unused entries.")
    return tuple(result)


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left.to(torch.float64), right.to(torch.float64)).detach().cpu())


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.to(torch.float64)).detach().cpu())


def vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    ln = _norm(left)
    rn = _norm(right)
    return float(_dot(left, right) / (ln * rn)) if ln > 0.0 and rn > 0.0 else 0.0


def _feasible(vector: torch.Tensor, gradients: Sequence[torch.Tensor]) -> bool:
    # Only a scale-aware floating-point tolerance is used here.  The scientific
    # trigger remains sign(dot)>0; this tolerance is for validating candidates.
    eps = float(torch.finfo(torch.float64).eps) * 128.0
    vn = _norm(vector)
    for gradient in gradients:
        gn = _norm(gradient)
        if gn <= SAFETY_EPS:
            continue
        tolerance = eps * max(vn * gn, 1.0)
        if _dot(vector, gradient) > tolerance:
            return False
    return True


def project_two_halfspaces(
    proposed_delta: torch.Tensor,
    first_gradient: torch.Tensor,
    second_gradient: torch.Tensor,
):
    """Closest displacement satisfying two homogeneous safety halfspaces.

    Constraints are ``<delta, g_i> <= 0``.  The function operates in fp64
    vector space and returns the projected vector plus deterministic diagnostics.
    """
    delta = proposed_delta.to(torch.float64)
    gradients = [first_gradient.to(torch.float64), second_gradient.to(torch.float64)]
    if any(g.shape != delta.shape for g in gradients):
        raise ValueError("Safety gradients must match proposed displacement shape.")
    if not torch.isfinite(delta).all() or any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("Non-finite actual-step safety vector.")

    dots_before = [_dot(delta, g) for g in gradients]
    norms_sq = [_dot(g, g) for g in gradients]
    active_before = [bool(dot > 0.0 and norm_sq > SAFETY_EPS) for dot, norm_sq in zip(dots_before, norms_sq)]
    if not any(active_before):
        return delta.clone(), {
            "projected": False,
            "active_constraint_count_before": 0,
            "first_dot_before": dots_before[0],
            "second_dot_before": dots_before[1],
            "first_dot_after": dots_before[0],
            "second_dot_after": dots_before[1],
            "proposed_delta_l2": _norm(delta),
            "final_delta_l2": _norm(delta),
            "removed_delta_l2_fraction": 0.0,
            "solution": "UNCHANGED_FEASIBLE",
        }

    candidates = []

    def add_candidate(name, value, multipliers):
        if _feasible(value, gradients):
            distance = _dot(delta - value, delta - value)
            candidates.append((distance, name, value, multipliers))

    # Single-active-constraint candidates.
    for index, (gradient, dot_before, norm_sq) in enumerate(zip(gradients, dots_before, norms_sq)):
        if dot_before > 0.0 and norm_sq > SAFETY_EPS:
            lam = dot_before / norm_sq
            value = delta - lam * gradient
            multipliers = [0.0, 0.0]
            multipliers[index] = float(lam)
            add_candidate("FIRST_ONLY" if index == 0 else "SECOND_ONLY", value, multipliers)

    # Both-active candidate from the 2x2 Gram system.
    if norms_sq[0] > SAFETY_EPS and norms_sq[1] > SAFETY_EPS:
        cross = _dot(gradients[0], gradients[1])
        gram = torch.tensor(
            [[norms_sq[0], cross], [cross, norms_sq[1]]],
            dtype=torch.float64,
            device=delta.device,
        )
        rhs = torch.tensor(dots_before, dtype=torch.float64, device=delta.device)
        determinant = float(torch.det(gram).detach().cpu())
        scale = max(norms_sq[0] * norms_sq[1], 1.0)
        if abs(determinant) > 256.0 * torch.finfo(torch.float64).eps * scale:
            lam = torch.linalg.solve(gram, rhs)
            lam_values = [float(lam[0].detach().cpu()), float(lam[1].detach().cpu())]
            # KKT multipliers for active inequality constraints must be nonnegative.
            if min(lam_values) >= -1e-12:
                value = delta - lam[0] * gradients[0] - lam[1] * gradients[1]
                add_candidate("BOTH_ACTIVE", value, lam_values)

    # Zero is always feasible; it prevents numerical singularities from silently
    # producing an unsafe step.  It wins only if no closer feasible candidate exists.
    add_candidate("ZERO_FALLBACK", torch.zeros_like(delta), [0.0, 0.0])
    if not candidates:
        raise RuntimeError("No feasible actual-step safety projection candidate.")
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, solution, final, multipliers = candidates[0]

    first_after = _dot(final, gradients[0])
    second_after = _dot(final, gradients[1])
    if not _feasible(final, gradients):
        raise RuntimeError("Actual-step safety projection returned an infeasible displacement.")
    proposed_norm = _norm(delta)
    final_norm = _norm(final)
    removed_norm = _norm(delta - final)
    return final, {
        "projected": True,
        "active_constraint_count_before": int(sum(active_before)),
        "first_dot_before": dots_before[0],
        "second_dot_before": dots_before[1],
        "first_dot_after": first_after,
        "second_dot_after": second_after,
        "first_multiplier": float(multipliers[0]),
        "second_multiplier": float(multipliers[1]),
        "proposed_delta_l2": proposed_norm,
        "final_delta_l2": final_norm,
        "removed_delta_l2_fraction": float(removed_norm / proposed_norm) if proposed_norm > 0.0 else 0.0,
        "solution": solution,
    }


def direct_v12_metric_improvement_gate(
    candidate_j: float,
    v12_j: float,
    candidate_transfer: Mapping,
    v12_transfer: Mapping,
):
    """Pre-run metric gate encoding the user's stated end goal: real improvement.

    J, beneficial NTR, and overall NTR must strictly improve over v12.  The
    nonbeneficial repair that v12 achieved may degrade by at most the existing
    3pp noninferiority convention.  This gate is frozen before v13 Valid is run.
    """
    candidate_b = float(candidate_transfer["teacher_beneficial"]["negative_transfer_rate"])
    v12_b = float(v12_transfer["teacher_beneficial"]["negative_transfer_rate"])
    candidate_nb = float(candidate_transfer["teacher_nonbeneficial"]["negative_transfer_rate"])
    v12_nb = float(v12_transfer["teacher_nonbeneficial"]["negative_transfer_rate"])
    candidate_all = float(candidate_transfer["all_missing"]["negative_transfer_rate"])
    v12_all = float(v12_transfer["all_missing"]["negative_transfer_rate"])
    checks = {
        "J_strictly_improves_v12": bool(float(candidate_j) < float(v12_j)),
        "beneficial_NTR_strictly_improves_v12": bool(candidate_b < v12_b),
        "overall_NTR_strictly_improves_v12": bool(candidate_all < v12_all),
        "nonbeneficial_repair_retained_vs_v12": bool(
            candidate_nb - v12_nb <= NONBENEFICIAL_NTR_MAX_DEGRADATION_VS_V12
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_J": float(candidate_j),
        "v12_J": float(v12_j),
        "delta_J_candidate_minus_v12": float(candidate_j) - float(v12_j),
        "beneficial_NTR_reduction_vs_v12": v12_b - candidate_b,
        "overall_NTR_reduction_vs_v12": v12_all - candidate_all,
        "nonbeneficial_NTR_degradation_vs_v12": candidate_nb - v12_nb,
        "nonbeneficial_NTR_max_degradation_vs_v12": NONBENEFICIAL_NTR_MAX_DEGRADATION_VS_V12,
    }


__all__ = [
    "BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8",
    "CONSENSUS_MIN_AGREE",
    "DEV_SEED",
    "DISTILL_MARGIN",
    "J_MAX_DEGRADATION_VS_V8",
    "METHOD",
    "NEAR_OPTIMAL_REL_TOL",
    "NONBENEFICIAL_NTR_MAX_DEGRADATION_VS_V12",
    "NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8",
    "N_FOLDS",
    "OUTPUT_TAG",
    "OVERALL_NTR_MAX_DEGRADATION_VS_V4",
    "RUN",
    "SAFETY_GROUPS",
    "S0_PROTECT_MARGIN",
    "VERSION",
    "clone_tensor_tuple",
    "direct_v12_metric_improvement_gate",
    "flatten_tensor_tuple",
    "jsonable",
    "project_two_halfspaces",
    "unflatten_like",
    "vector_cosine",
]
