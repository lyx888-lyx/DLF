"""Objectives and diagnostics for bidirectional tail residual experts V9.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F


TAIL_ROLE_NAMES = ("shared_tail", "strong_negative", "strong_positive")
MECHANISM_NAMES = (
    "sign_repair",
    "magnitude_increase",
    "magnitude_decrease",
    "near_correct",
)


@dataclass(frozen=True)
class TailLossWeights:
    global_mae: float = 0.20
    tail_mae: float = 0.80
    residual_huber: float = 1.00
    gain_margin: float = 0.25
    outside_retention: float = 0.40
    applicability_bce: float = 0.25
    mechanism_ce: float = 0.12
    teacher_distill: float = 0.12
    correction_shrink: float = 0.02


def validate_tail_role(role: str) -> str:
    role = str(role)
    if role not in TAIL_ROLE_NAMES:
        raise ValueError(
            "tail role must be one of %s, got %r" % (TAIL_ROLE_NAMES, role)
        )
    return role


def exact_tail_mask(labels: torch.Tensor, role: str) -> torch.Tensor:
    role = validate_tail_role(role)
    values = labels.view(-1)
    if role == "strong_negative":
        return values < -1.5
    if role == "strong_positive":
        return values > 1.5
    return torch.abs(values) > 1.5


def soft_tail_membership(
    labels: torch.Tensor,
    role: str,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Monotone tail curriculum with no non-tail membership floor."""

    role = validate_tail_role(role)
    values = labels.view(-1)
    temperature = max(float(temperature), 1e-4)
    if role == "strong_negative":
        score = (-1.5 - values) / temperature
    elif role == "strong_positive":
        score = (values - 1.5) / temperature
    else:
        score = (torch.abs(values) - 1.5) / temperature
    return torch.sigmoid(score)


def normalized_weighted_mean(
    values: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    values = values.view(-1)
    weights = weights.view(-1).to(values)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def residual_mechanism_index(
    anchor_prediction: torch.Tensor,
    labels: torch.Tensor,
    magnitude_tolerance: float = 0.10,
) -> torch.Tensor:
    """Classify the correction mechanism relative to a fixed anchor.

    0: sign repair; 1: increase absolute magnitude; 2: decrease absolute
    magnitude; 3: already close in magnitude.  The definition is symmetric for
    positive and negative sentiment and therefore does not force corrections
    toward either tail.
    """

    anchor = anchor_prediction.view(-1)
    target = labels.view(-1)
    tolerance = float(magnitude_tolerance)
    result = torch.full_like(target, 3, dtype=torch.long)

    sign_error = (anchor * target <= 0.0) & (torch.abs(target) > tolerance)
    same_sign = ~sign_error
    under = same_sign & (torch.abs(anchor) + tolerance < torch.abs(target))
    over = same_sign & (torch.abs(anchor) > torch.abs(target) + tolerance)
    result[sign_error] = 0
    result[under] = 1
    result[over] = 2
    return result


def _balanced_binary_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    logits = logits.view(-1)
    targets = targets.view(-1).to(logits)
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives.item() <= 0 or negatives.item() <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets)
    positive_weight = (negatives / positives).detach().clamp(0.25, 8.0)
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight
    )


def tail_residual_loss(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    fixed_anchor_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    role: str,
    membership_temperature: float,
    gain_margin: float,
    gain_fraction: float,
    teacher_scale: float,
    weights: TailLossWeights,
) -> Dict[str, torch.Tensor]:
    """Bidirectional residual objective relative to an immutable anchor."""

    role = validate_tail_role(role)
    prediction = outputs["prediction"]
    correction = outputs["correction"]
    raw_correction = outputs["raw_correction"]
    labels = labels.view_as(prediction)
    anchor = fixed_anchor_prediction.view_as(prediction).detach()
    teacher = teacher_prediction.view_as(prediction).detach()

    membership = soft_tail_membership(
        labels, role, temperature=membership_temperature
    )
    outside = (1.0 - membership).clamp_min(0.0)
    exact_target = exact_tail_mask(labels, role).to(prediction)

    absolute_error = torch.abs(prediction - labels)
    anchor_error = torch.abs(anchor - labels)
    residual_target = (labels - anchor).detach()

    global_mae = absolute_error.mean()
    tail_mae = normalized_weighted_mean(absolute_error, membership)
    residual_values = F.smooth_l1_loss(
        correction, residual_target, reduction="none", beta=0.25
    )
    residual_huber = normalized_weighted_mean(residual_values, membership)

    required_gain = torch.minimum(
        anchor_error * float(gain_fraction),
        anchor_error.new_full(anchor_error.shape, float(gain_margin)),
    )
    gain_margin_loss = normalized_weighted_mean(
        F.relu(absolute_error - anchor_error + required_gain), membership
    )
    outside_retention = normalized_weighted_mean(
        F.smooth_l1_loss(
            prediction, anchor, reduction="none", beta=0.10
        ),
        outside + 1e-3,
    )

    applicability_bce = _balanced_binary_cross_entropy(
        outputs["applicability_logit"], exact_target
    )
    mechanism_target = residual_mechanism_index(anchor, labels)
    mechanism_values = F.cross_entropy(
        outputs["mechanism_logits"], mechanism_target, reduction="none"
    )
    mechanism_ce = 0.25 * mechanism_values.mean() + 0.75 * normalized_weighted_mean(
        mechanism_values, membership
    )

    teacher_values = F.smooth_l1_loss(
        prediction, teacher, reduction="none", beta=0.25
    )
    teacher_distill = float(teacher_scale) * normalized_weighted_mean(
        teacher_values, membership
    )
    correction_shrink = (
        raw_correction.square().mean()
        + normalized_weighted_mean(correction.square(), outside + 1e-3)
    )

    components = {
        "global_mae": global_mae,
        "tail_mae": tail_mae,
        "residual_huber": residual_huber,
        "gain_margin": gain_margin_loss,
        "outside_retention": outside_retention,
        "applicability_bce": applicability_bce,
        "mechanism_ce": mechanism_ce,
        "teacher_distill": teacher_distill,
        "correction_shrink": correction_shrink,
    }
    total = sum(
        getattr(weights, name) * value for name, value in components.items()
    )
    return {"total": total, **components}


def safe_corr(prediction: torch.Tensor, labels: torch.Tensor) -> float:
    x = prediction.view(-1).double()
    y = labels.view(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(
        x.square().sum() * y.square().sum()
    ).clamp_min(1e-12)
    return float((x * y).sum().div(denominator).item())


def tail_capability_metrics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    result: Dict[str, float] = {
        "global_mae": float(torch.abs(prediction - labels).mean().item()),
        "global_corr": safe_corr(prediction, labels),
    }
    for role in ("strong_negative", "strong_positive", "shared_tail"):
        mask = exact_tail_mask(labels, role)
        result[role + "_count"] = int(mask.sum().item())
        result[role + "_mae"] = (
            float(torch.abs(prediction[mask] - labels[mask]).mean().item())
            if mask.any()
            else math.nan
        )
    return result


def residual_diagnostic_rows(
    anchor_prediction: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: Sequence[object],
    split_name: str,
):
    """Summarize why the fixed anchor fails in each tail region."""

    if len(sample_ids) != labels.size(0):
        raise ValueError("diagnostic sample ids and labels have different lengths.")
    rows = []
    mechanisms = residual_mechanism_index(anchor_prediction, labels)
    residual = labels.view(-1) - anchor_prediction.view(-1)
    absolute_error = torch.abs(residual)
    for role in ("strong_negative", "strong_positive", "shared_tail"):
        role_mask = exact_tail_mask(labels, role)
        if not role_mask.any():
            continue
        role_residual = residual[role_mask]
        role_error = absolute_error[role_mask]
        rows.append(
            {
                "split": split_name,
                "role": role,
                "mechanism": "all",
                "count": int(role_mask.sum().item()),
                "anchor_mae": float(role_error.mean().item()),
                "mean_residual": float(role_residual.mean().item()),
                "median_residual": float(role_residual.median().item()),
                "positive_residual_rate": float((role_residual > 0).float().mean().item()),
                "negative_residual_rate": float((role_residual < 0).float().mean().item()),
                "mae_contribution": float(role_error.sum().item()),
            }
        )
        for index, mechanism_name in enumerate(MECHANISM_NAMES):
            mask = role_mask & (mechanisms == index)
            if not mask.any():
                continue
            mechanism_residual = residual[mask]
            mechanism_error = absolute_error[mask]
            rows.append(
                {
                    "split": split_name,
                    "role": role,
                    "mechanism": mechanism_name,
                    "count": int(mask.sum().item()),
                    "anchor_mae": float(mechanism_error.mean().item()),
                    "mean_residual": float(mechanism_residual.mean().item()),
                    "median_residual": float(mechanism_residual.median().item()),
                    "positive_residual_rate": float((mechanism_residual > 0).float().mean().item()),
                    "negative_residual_rate": float((mechanism_residual < 0).float().mean().item()),
                    "mae_contribution": float(mechanism_error.sum().item()),
                }
            )
    return rows
