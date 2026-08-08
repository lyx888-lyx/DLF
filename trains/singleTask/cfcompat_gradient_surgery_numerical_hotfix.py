"""Numerically stable hotfix for CFCompatKD v12 gradient surgery.

This module preserves the exact asymmetric projection used by v12 while
performing projection geometry in float64 and validating the post-projection
conflict with a scale-free cosine tolerance.  It exists only to avoid false
failures from float32 roundoff such as an absolute post-dot around -1e-6.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch

from .cfcompat_gradient_surgery_utils import (
    SURGERY_EPS,
    add_gradient_tuples,
)


def _dot_fp64(
    first: Sequence[torch.Tensor], second: Sequence[torch.Tensor]
) -> torch.Tensor:
    if len(first) != len(second) or not first:
        raise ValueError("Gradient tuples must be non-empty and aligned.")
    result = torch.zeros((), device=first[0].device, dtype=torch.float64)
    for a, b in zip(first, second):
        result = result + torch.sum(a.to(torch.float64) * b.to(torch.float64))
    return result


def _norm_sq_fp64(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return _dot_fp64(gradients, gradients)


def _project_once_fp64(
    supervised: Sequence[torch.Tensor],
    selective: Sequence[torch.Tensor],
    coefficient: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        (
            g_sup.to(torch.float64)
            - coefficient * g_sel.to(torch.float64)
        ).to(dtype=g_sup.dtype)
        for g_sup, g_sel in zip(supervised, selective)
    )


def asymmetric_project_supervised_stable(
    supervised: Sequence[torch.Tensor],
    selective: Sequence[torch.Tensor],
    eps: float = SURGERY_EPS,
):
    """Exact v12 projection with numerically stable diagnostics/checks.

    The scientific rule is unchanged: when the accumulated supervised and
    selective gradients conflict, only the supervised component parallel to
    the negative selective direction is removed.  The selective gradient is
    never modified.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if len(supervised) != len(selective) or not supervised:
        raise ValueError("Gradient tuples must be non-empty and aligned.")
    if any(not torch.is_floating_point(value) for value in supervised + selective):
        raise TypeError("Gradient surgery requires floating tensors.")

    dot_before_t = _dot_fp64(supervised, selective)
    sup_sq_t = _norm_sq_fp64(supervised)
    sel_sq_t = _norm_sq_fp64(selective)
    dot_before = float(dot_before_t.detach().cpu())
    sup_sq = float(sup_sq_t.detach().cpu())
    sel_sq = float(sel_sq_t.detach().cpu())
    conflict = bool(dot_before < 0.0 and sel_sq > eps)

    coefficient = 0.0
    cleanup_coefficient = 0.0
    cleanup_passes = 0
    if conflict:
        coefficient_t = dot_before_t / sel_sq_t
        projected = _project_once_fp64(supervised, selective, coefficient_t)
        coefficient = float(coefficient_t.detach().cpu())

        # Casting the analytically orthogonal fp64 result back to float32 can
        # leave a tiny negative dot.  One fp64 cleanup projection removes that
        # representational remainder without changing the intended geometry.
        post_dot_t = _dot_fp64(projected, selective)
        if float(post_dot_t.detach().cpu()) < 0.0:
            cleanup_t = post_dot_t / sel_sq_t
            projected = _project_once_fp64(projected, selective, cleanup_t)
            cleanup_coefficient = float(cleanup_t.detach().cpu())
            cleanup_passes = 1
    else:
        projected = tuple(g.clone() for g in supervised)

    update = add_gradient_tuples(projected, selective)
    post_dot_t = _dot_fp64(projected, selective)
    projected_sq_t = _norm_sq_fp64(projected)
    update_sq_t = _norm_sq_fp64(update)
    post_dot = float(post_dot_t.detach().cpu())
    projected_sq = float(projected_sq_t.detach().cpu())
    update_sq = float(update_sq_t.detach().cpu())

    denominator = (sup_sq * sel_sq) ** 0.5
    cosine = float(dot_before / denominator) if denominator > 0.0 else 0.0
    post_denominator = (projected_sq * sel_sq) ** 0.5
    post_cosine = float(post_dot / post_denominator) if post_denominator > 0.0 else 0.0

    # The old fixed absolute -1e-6 dot threshold was scale-dependent.  A
    # cosine tolerance tied to the storage dtype is the correct numerical
    # criterion.  64 ulps is deliberately conservative for reductions across
    # many residual-head tensors.
    dtype_eps = float(torch.finfo(supervised[0].dtype).eps)
    post_cosine_tolerance = 64.0 * dtype_eps
    if conflict and post_cosine < -post_cosine_tolerance:
        raise RuntimeError(
            "Gradient surgery left a material conflict: post_dot={} "
            "post_cosine={} tolerance={}".format(
                post_dot, post_cosine, post_cosine_tolerance
            )
        )

    removed_fraction = (
        float(max(0.0, 1.0 - (projected_sq / sup_sq) ** 0.5))
        if sup_sq > 0.0
        else 0.0
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
        "numerical_cleanup_coefficient": cleanup_coefficient,
        "numerical_cleanup_passes": int(cleanup_passes),
        "post_projection_dot": post_dot,
        "post_projection_cosine": post_cosine,
        "post_projection_cosine_tolerance": post_cosine_tolerance,
        "supervised_l2_removed_fraction": removed_fraction,
    }
    return projected, update, diagnostics


__all__ = ["asymmetric_project_supervised_stable"]
