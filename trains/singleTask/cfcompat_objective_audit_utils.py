"""Train-OOF objective-audit utilities for CFCompatKD v11.

v11 is diagnostic only. It does not train or select a new model. The audit
loads the frozen v10 conservative cross-fit residual checkpoints and asks two
questions using only MOSI Train folds:

1. What targets do SUPERVISED_MISSING / DISTILL / PRESERVE present to each
   held-out Train event?
2. At the frozen checkpoint, would an infinitesimal gradient-descent step from
   each training-objective component improve or harm held-out Train groups?

The second question is measured with first-order gradient alignment. If
``dot(g_train_component, g_oof_loss) > 0``, a gradient-descent step ``-g_train``
locally decreases the OOF loss; if the dot product is negative it locally
increases the OOF loss.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cfcompat_regret_preserve_utils import DISTILL_MARGIN
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_objective_train_oof_audit_v11"
METHOD = "DLF-Frozen-S0-CFCompatKD-Objective-Train-OOF-Audit-v11"
OUTPUT_TAG = "cfcompat_objective_audit_v11"
DEV_SEED = 1113
N_FOLDS = 5
AUDIT_MODES = ("ALL",) + tuple(MISSING_MODES)

BASE_TRAIN_COMPONENTS = (
    "SUPERVISED_BRANCH_DISTILL",
    "SUPERVISED_BRANCH_PRESERVE",
    "SUPERVISED_BRANCH_ABSTAIN",
    "SUPERVISED_TEACHER_BENEFICIAL",
    "SUPERVISED_TEACHER_NONBENEFICIAL",
    "DISTILL_KD",
    "PRESERVE_SCALED",
)
DERIVED_TRAIN_COMPONENTS = (
    "SUPERVISED_ALL",
    "SELECTIVE_ONLY",
    "TOTAL_RESIDUAL_OBJECTIVE",
)
TRAIN_COMPONENTS = BASE_TRAIN_COMPONENTS + DERIVED_TRAIN_COMPONENTS
OOF_GROUPS = (
    "OOF_ALL",
    "OOF_TEACHER_BENEFICIAL",
    "OOF_TEACHER_NONBENEFICIAL",
    "OOF_S0_BENEFICIAL",
    "OOF_S0_NONBENEFICIAL",
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


def branch_name(distill: bool, preserve: bool, abstain: bool) -> str:
    flags = [bool(distill), bool(preserve), bool(abstain)]
    if sum(flags) != 1:
        raise ValueError("DISTILL/PRESERVE/ABSTAIN must be exclusive.")
    return "DISTILL" if distill else "PRESERVE" if preserve else "ABSTAIN"


def sign_relation(delta: float, desired: float, eps: float = 1e-12) -> str:
    if abs(delta) <= eps:
        return "zero"
    if abs(desired) <= eps:
        return "moves_from_exact_label"
    return "toward_label" if delta * desired > 0 else "away_from_label"


def enrich_oof_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add frozen-S0 and target-geometry diagnostics to raw OOF events."""
    required = {
        "fold", "sample_index", "sample_id", "mode", "label",
        "baseline_prediction", "s0_prediction", "student_prediction",
        "residual_delta", "teacher_prediction", "teacher_safe_target",
        "preserve_safe_target", "distill", "preserve", "decision_abstain",
        "teacher_beneficial", "current_regressed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("OOF audit frame lacks columns: {}".format(sorted(missing)))
    result = frame.copy()
    numeric_cols = [
        "label", "baseline_prediction", "s0_prediction", "student_prediction",
        "residual_delta", "teacher_prediction", "teacher_safe_target",
        "preserve_safe_target",
    ]
    if not np.isfinite(result[numeric_cols].to_numpy(np.float64)).all():
        raise FloatingPointError("OOF audit frame contains NaN/Inf.")

    result["branch"] = [
        branch_name(d, p, a)
        for d, p, a in zip(result.distill, result.preserve, result.decision_abstain)
    ]
    result["baseline_error"] = (result.baseline_prediction - result.label).abs()
    result["s0_error"] = (result.s0_prediction - result.label).abs()
    result["current_error"] = (result.student_prediction - result.label).abs()
    result["teacher_error"] = (result.teacher_prediction - result.label).abs()
    result["s0_gain_vs_baseline"] = result.baseline_error - result.s0_error
    result["current_gain_vs_baseline"] = result.baseline_error - result.current_error
    result["current_gain_vs_s0"] = result.s0_error - result.current_error
    result["s0_beneficial"] = result.s0_gain_vs_baseline >= DISTILL_MARGIN
    result["current_regressed_vs_s0_margin"] = (
        result.current_error - result.s0_error >= DISTILL_MARGIN
    )
    result["residual_abs_delta"] = result.residual_delta.abs()
    result["residual_relation_to_s0_label"] = [
        sign_relation(float(delta), float(label - s0))
        for delta, label, s0 in zip(
            result.residual_delta, result.label, result.s0_prediction
        )
    ]
    result["residual_crossed_label_from_s0"] = (
        (result.s0_prediction - result.label)
        * (result.student_prediction - result.label)
        < 0
    )
    result["residual_crossed_baseline_from_s0"] = (
        (result.s0_prediction - result.baseline_prediction)
        * (result.student_prediction - result.baseline_prediction)
        < 0
    )

    branch_target = result.student_prediction.to_numpy(np.float64).copy()
    active = np.zeros(len(result), dtype=bool)
    distill = result.distill.astype(bool).to_numpy()
    preserve = result.preserve.astype(bool).to_numpy()
    branch_target[distill] = result.loc[distill, "teacher_safe_target"].to_numpy(np.float64)
    branch_target[preserve] = result.loc[preserve, "preserve_safe_target"].to_numpy(np.float64)
    active[distill | preserve] = True
    result["branch_target_active"] = active
    result["branch_target_prediction"] = branch_target
    result["branch_target_shift"] = branch_target - result.student_prediction.to_numpy(np.float64)
    result["branch_target_abs_shift"] = np.abs(result.branch_target_shift)
    result["branch_target_error"] = np.abs(branch_target - result.label.to_numpy(np.float64))
    result["branch_target_locally_safe"] = (
        (~active) | (result.branch_target_error <= result.current_error + 1e-8)
    )
    result["supervised_missing_active"] = True
    result["supervised_target_prediction"] = result.label
    result["supervised_target_abs_shift"] = (
        result.label - result.student_prediction
    ).abs()
    return result


def event_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize OOF event geometry without using Official Valid/Test."""
    rows = []
    groupings = [
        ("ALL", []),
        ("BRANCH", ["branch"]),
        ("TEACHER_BENEFICIAL", ["teacher_beneficial"]),
        ("S0_BENEFICIAL", ["s0_beneficial"]),
        ("BRANCH_X_TEACHER", ["branch", "teacher_beneficial"]),
        ("BRANCH_X_S0", ["branch", "s0_beneficial"]),
        ("MODE_X_BRANCH", ["mode", "branch"]),
    ]
    for family, columns in groupings:
        groups = [((), frame)] if not columns else frame.groupby(columns, dropna=False, sort=True)
        for key, local in groups:
            if not isinstance(key, tuple):
                key = (key,)
            row = {
                "group_family": family,
                "N": int(len(local)),
                "mean_s0_gain_vs_baseline": float(local.s0_gain_vs_baseline.mean()),
                "mean_current_gain_vs_baseline": float(local.current_gain_vs_baseline.mean()),
                "mean_current_gain_vs_s0": float(local.current_gain_vs_s0.mean()),
                "mean_abs_residual": float(local.residual_abs_delta.mean()),
                "current_regressed_vs_s0_rate": float(
                    local.current_regressed_vs_s0_margin.astype(bool).mean()
                ),
                "residual_crossed_label_rate": float(
                    local.residual_crossed_label_from_s0.astype(bool).mean()
                ),
                "residual_crossed_baseline_rate": float(
                    local.residual_crossed_baseline_from_s0.astype(bool).mean()
                ),
                "branch_target_active_rate": float(local.branch_target_active.astype(bool).mean()),
                "branch_target_locally_safe_rate": float(local.branch_target_locally_safe.astype(bool).mean()),
                "mean_branch_target_abs_shift": float(local.branch_target_abs_shift.mean()),
                "mean_supervised_target_abs_shift": float(local.supervised_target_abs_shift.mean()),
            }
            for name, value in zip(columns, key):
                row[name] = value
            rows.append(row)
    return pd.DataFrame(rows)


def zero_vector_like_parameters(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    count = int(sum(parameter.numel() for parameter in parameters))
    return torch.zeros(count, dtype=torch.float64)


def gradient_vector(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    retain_graph: bool = True,
) -> torch.Tensor:
    """Flatten gradients to CPU float64; unused parameters contribute zeros."""
    if not parameters:
        raise ValueError("Gradient audit requires trainable residual parameters.")
    if not loss.requires_grad:
        return zero_vector_like_parameters(parameters)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    parts = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            parts.append(torch.zeros(parameter.numel(), dtype=torch.float64))
        else:
            parts.append(gradient.detach().reshape(-1).cpu().to(torch.float64))
    vector = torch.cat(parts)
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Gradient vector contains NaN/Inf.")
    return vector


def vector_add(*vectors: torch.Tensor) -> torch.Tensor:
    if not vectors:
        raise ValueError("vector_add requires at least one vector.")
    result = vectors[0].clone()
    for vector in vectors[1:]:
        result.add_(vector)
    return result


def derive_train_component_vectors(base: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    missing = set(BASE_TRAIN_COMPONENTS).difference(base)
    if missing:
        raise ValueError("Missing base train gradients: {}".format(sorted(missing)))
    result = {key: base[key] for key in BASE_TRAIN_COMPONENTS}
    supervised_all = vector_add(
        base["SUPERVISED_BRANCH_DISTILL"],
        base["SUPERVISED_BRANCH_PRESERVE"],
        base["SUPERVISED_BRANCH_ABSTAIN"],
    )
    result["SUPERVISED_ALL"] = supervised_all
    result["SELECTIVE_ONLY"] = vector_add(base["DISTILL_KD"], base["PRESERVE_SCALED"])
    result["TOTAL_RESIDUAL_OBJECTIVE"] = vector_add(
        supervised_all, result["SELECTIVE_ONLY"]
    )
    return result


def gradient_influence_row(
    fold: int,
    mode: str,
    component: str,
    oof_group: str,
    train_gradient: torch.Tensor,
    oof_gradient: torch.Tensor,
    oof_count: int,
) -> dict:
    train_norm = float(torch.linalg.vector_norm(train_gradient))
    oof_norm = float(torch.linalg.vector_norm(oof_gradient))
    dot = float(torch.dot(train_gradient, oof_gradient))
    if train_norm > 0.0 and oof_norm > 0.0:
        cosine = float(dot / (train_norm * oof_norm))
    else:
        cosine = 0.0
    first_order_change = -dot
    if dot > 1e-15:
        effect = "IMPROVE_OOF"
    elif dot < -1e-15:
        effect = "HARM_OOF"
    else:
        effect = "NEUTRAL"
    return {
        "Fold": int(fold),
        "Mode": str(mode),
        "TrainComponent": str(component),
        "OOFGroup": str(oof_group),
        "OOFN": int(oof_count),
        "train_gradient_l2": train_norm,
        "oof_loss_gradient_l2": oof_norm,
        "gradient_dot": dot,
        "gradient_cosine": cosine,
        "predicted_first_order_oof_loss_change_per_unit_step": first_order_change,
        "predicted_effect_of_gradient_descent": effect,
    }


def aggregate_influence(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Fold", "Mode", "TrainComponent", "OOFGroup", "gradient_dot",
        "gradient_cosine", "predicted_effect_of_gradient_descent",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Influence frame lacks columns: {}".format(sorted(missing)))
    rows = []
    for (mode, component, group), local in frame.groupby(
        ["Mode", "TrainComponent", "OOFGroup"], sort=True
    ):
        rows.append(
            {
                "Mode": mode,
                "TrainComponent": component,
                "OOFGroup": group,
                "FoldCount": int(local.Fold.nunique()),
                "mean_gradient_dot": float(local.gradient_dot.mean()),
                "median_gradient_dot": float(local.gradient_dot.median()),
                "mean_gradient_cosine": float(local.gradient_cosine.mean()),
                "median_gradient_cosine": float(local.gradient_cosine.median()),
                "harm_fold_count": int(
                    local.predicted_effect_of_gradient_descent.eq("HARM_OOF").sum()
                ),
                "improve_fold_count": int(
                    local.predicted_effect_of_gradient_descent.eq("IMPROVE_OOF").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def headline_findings(events: pd.DataFrame, aggregate: pd.DataFrame) -> dict:
    active = events.loc[events.branch_target_active.astype(bool)]
    branch_safe = float(active.branch_target_locally_safe.mean()) if len(active) else 1.0
    abstain = events.loc[events.branch.astype(str).eq("ABSTAIN")]

    def lookup(component: str, group: str) -> dict:
        local = aggregate.loc[
            aggregate.Mode.astype(str).eq("ALL")
            & aggregate.TrainComponent.astype(str).eq(component)
            & aggregate.OOFGroup.astype(str).eq(group)
        ]
        if len(local) != 1:
            return {}
        row = local.iloc[0]
        return {
            "mean_gradient_cosine": float(row.mean_gradient_cosine),
            "mean_gradient_dot": float(row.mean_gradient_dot),
            "harm_fold_count": int(row.harm_fold_count),
            "improve_fold_count": int(row.improve_fold_count),
        }

    return {
        "oof_event_count": int(len(events)),
        "active_branch_target_locally_safe_fraction": branch_safe,
        "abstain_event_fraction": float(len(abstain) / len(events)),
        "abstain_still_has_supervised_missing_loss": True,
        "supervised_all_to_oof_teacher_beneficial": lookup(
            "SUPERVISED_ALL", "OOF_TEACHER_BENEFICIAL"
        ),
        "selective_only_to_oof_teacher_beneficial": lookup(
            "SELECTIVE_ONLY", "OOF_TEACHER_BENEFICIAL"
        ),
        "total_objective_to_oof_teacher_beneficial": lookup(
            "TOTAL_RESIDUAL_OBJECTIVE", "OOF_TEACHER_BENEFICIAL"
        ),
        "supervised_abstain_to_oof_teacher_beneficial": lookup(
            "SUPERVISED_BRANCH_ABSTAIN", "OOF_TEACHER_BENEFICIAL"
        ),
        "supervised_teacher_nonbeneficial_to_oof_teacher_beneficial": lookup(
            "SUPERVISED_TEACHER_NONBENEFICIAL", "OOF_TEACHER_BENEFICIAL"
        ),
    }


__all__ = [
    "AUDIT_MODES",
    "BASE_TRAIN_COMPONENTS",
    "DEV_SEED",
    "METHOD",
    "N_FOLDS",
    "OOF_GROUPS",
    "OUTPUT_TAG",
    "TRAIN_COMPONENTS",
    "VERSION",
    "aggregate_influence",
    "derive_train_component_vectors",
    "enrich_oof_event_frame",
    "event_summary",
    "gradient_influence_row",
    "gradient_vector",
    "headline_findings",
    "jsonable",
    "zero_vector_like_parameters",
]
