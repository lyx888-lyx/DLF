"""Post-surgery Train-OOF gradient audit utilities for CFCompatKD v12.1.

This module is diagnostic only.  It replays the *analytic* asymmetric v12
projection in flattened float64 gradient space at a frozen v12 checkpoint:

    if <g_sup, g_sel> < 0:
        g_sup_projected = g_sup - <g_sup,g_sel>/||g_sel||^2 * g_sel
    else:
        g_sup_projected = g_sup

and defines g_update = g_sup_projected + g_sel.

The purpose is not to reconstruct the historical Adam trajectory.  It asks a
local first-order question at each frozen conservative v12 fold checkpoint:
whether the projected supervised direction and the resulting surgery update
still harm held-out Train (OOF) beneficial groups even after direct
supervised-vs-selective conflict has been removed.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import torch


VERSION = "cfcompat_post_surgery_train_oof_audit_v12p1"
METHOD = "DLF-Frozen-v12-Post-Surgery-Train-OOF-Audit-v12.1"
OUTPUT_TAG = "cfcompat_post_surgery_audit_v12p1"
DEV_SEED = 1113
N_FOLDS = 5

POST_SURGERY_COMPONENTS = (
    "SUPERVISED_ALL_RAW",
    "SELECTIVE_ONLY",
    "TOTAL_RESIDUAL_OBJECTIVE_RAW",
    "SUPERVISED_PROJECTED",
    "SUPERVISED_REMOVED_CONFLICT",
    "SURGERY_UPDATE",
)


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


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector))


def replay_asymmetric_projection(
    supervised: torch.Tensor,
    selective: torch.Tensor,
    eps: float = 1e-24,
) -> dict:
    """Replay the v12 asymmetric surgery exactly in CPU float64 vector space."""
    if supervised.ndim != 1 or selective.ndim != 1:
        raise ValueError("Expected flattened one-dimensional gradient vectors.")
    if supervised.shape != selective.shape or supervised.numel() == 0:
        raise ValueError("Supervised/selective vectors must be non-empty and aligned.")
    if supervised.dtype != torch.float64 or selective.dtype != torch.float64:
        supervised = supervised.to(torch.float64)
        selective = selective.to(torch.float64)
    if not torch.isfinite(supervised).all() or not torch.isfinite(selective).all():
        raise FloatingPointError("Non-finite gradient supplied to post-surgery audit.")

    dot_before = float(torch.dot(supervised, selective))
    sup_sq = float(torch.dot(supervised, supervised))
    sel_sq = float(torch.dot(selective, selective))
    conflict = bool(dot_before < 0.0 and sel_sq > eps)
    coefficient = float(dot_before / sel_sq) if conflict else 0.0
    if conflict:
        projected = supervised - coefficient * selective
    else:
        projected = supervised.clone()
    removed = supervised - projected
    update = projected + selective

    post_dot = float(torch.dot(projected, selective))
    sup_norm = sup_sq ** 0.5
    sel_norm = sel_sq ** 0.5
    projected_norm = _norm(projected)
    removed_norm = _norm(removed)
    update_norm = _norm(update)
    cosine_before = (
        float(dot_before / (sup_norm * sel_norm))
        if sup_norm > 0.0 and sel_norm > 0.0
        else 0.0
    )
    post_cosine = (
        float(post_dot / (projected_norm * sel_norm))
        if projected_norm > 0.0 and sel_norm > 0.0
        else 0.0
    )
    if conflict and post_cosine < -1e-12:
        raise RuntimeError(
            "Float64 replay left material supervised/selective conflict: {}".format(
                post_cosine
            )
        )

    return {
        "SUPERVISED_ALL_RAW": supervised,
        "SELECTIVE_ONLY": selective,
        "TOTAL_RESIDUAL_OBJECTIVE_RAW": supervised + selective,
        "SUPERVISED_PROJECTED": projected,
        "SUPERVISED_REMOVED_CONFLICT": removed,
        "SURGERY_UPDATE": update,
        "diagnostics": {
            "conflict": conflict,
            "gradient_dot_before": dot_before,
            "gradient_cosine_before": cosine_before,
            "projection_coefficient": coefficient,
            "post_projection_dot": post_dot,
            "post_projection_cosine": post_cosine,
            "supervised_gradient_l2": sup_norm,
            "selective_gradient_l2": sel_norm,
            "projected_supervised_gradient_l2": projected_norm,
            "removed_gradient_l2": removed_norm,
            "surgery_update_gradient_l2": update_norm,
            "supervised_l2_removed_fraction": (
                float(removed_norm / sup_norm) if sup_norm > 0.0 else 0.0
            ),
        },
    }


def headline_lookup(
    aggregate: pd.DataFrame,
    component: str,
    group: str,
    mode: str = "ALL",
) -> dict:
    local = aggregate.loc[
        aggregate.Mode.astype(str).eq(str(mode))
        & aggregate.TrainComponent.astype(str).eq(str(component))
        & aggregate.OOFGroup.astype(str).eq(str(group))
    ]
    if len(local) != 1:
        raise RuntimeError(
            "Expected one aggregate row for {} / {} / {}, got {}".format(
                mode, component, group, len(local)
            )
        )
    row = local.iloc[0]
    return {
        "mean_gradient_dot": float(row.mean_gradient_dot),
        "mean_gradient_cosine": float(row.mean_gradient_cosine),
        "median_gradient_cosine": float(row.median_gradient_cosine),
        "harm_fold_count": int(row.harm_fold_count),
        "improve_fold_count": int(row.improve_fold_count),
        "fold_count": int(row.FoldCount),
    }


def headline_findings(aggregate: pd.DataFrame, projection_frame: pd.DataFrame) -> dict:
    """Return fixed descriptive findings; no promotion gate is defined in v12.1."""
    groups = (
        "OOF_TEACHER_BENEFICIAL",
        "OOF_TEACHER_NONBENEFICIAL",
        "OOF_S0_BENEFICIAL",
        "OOF_S0_NONBENEFICIAL",
    )
    components = (
        "SUPERVISED_ALL_RAW",
        "SELECTIVE_ONLY",
        "SUPERVISED_PROJECTED",
        "SURGERY_UPDATE",
    )
    matrix = {
        component: {group: headline_lookup(aggregate, component, group) for group in groups}
        for component in components
    }
    all_projection = projection_frame.loc[projection_frame.Mode.astype(str).eq("ALL")]
    if len(all_projection) != N_FOLDS:
        raise RuntimeError("Expected one ALL projection diagnostic per fold.")
    return {
        "gradient_matrix": matrix,
        "all_mode_projection_conflict_fold_count": int(
            all_projection.conflict.astype(bool).sum()
        ),
        "all_mode_mean_raw_sup_selective_cosine": float(
            all_projection.gradient_cosine_before.mean()
        ),
        "all_mode_mean_post_projection_cosine": float(
            all_projection.post_projection_cosine.mean()
        ),
        "all_mode_mean_supervised_l2_removed_fraction": float(
            all_projection.supervised_l2_removed_fraction.mean()
        ),
        "projected_supervised_harm_teacher_beneficial_folds": int(
            matrix["SUPERVISED_PROJECTED"]["OOF_TEACHER_BENEFICIAL"]["harm_fold_count"]
        ),
        "surgery_update_harm_teacher_beneficial_folds": int(
            matrix["SURGERY_UPDATE"]["OOF_TEACHER_BENEFICIAL"]["harm_fold_count"]
        ),
        "projected_supervised_harm_s0_beneficial_folds": int(
            matrix["SUPERVISED_PROJECTED"]["OOF_S0_BENEFICIAL"]["harm_fold_count"]
        ),
        "surgery_update_harm_s0_beneficial_folds": int(
            matrix["SURGERY_UPDATE"]["OOF_S0_BENEFICIAL"]["harm_fold_count"]
        ),
    }


__all__ = [
    "DEV_SEED",
    "METHOD",
    "N_FOLDS",
    "OUTPUT_TAG",
    "POST_SURGERY_COMPONENTS",
    "VERSION",
    "headline_findings",
    "headline_lookup",
    "jsonable",
    "replay_asymmetric_projection",
]
