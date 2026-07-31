"""Objectives and diagnostics for OOF-supervised tail experts V9.2."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
class OOFTailLossWeights:
    global_mae: float = 0.30
    tail_mae: float = 0.75
    residual_huber: float = 1.20
    gain_margin: float = 0.20
    outside_retention: float = 0.45
    applicability_bce: float = 0.18
    mechanism_ce: float = 0.10
    correction_shrink: float = 0.025


def validate_tail_role(role: str) -> str:
    role = str(role)
    if role not in TAIL_ROLE_NAMES:
        raise ValueError(f"unsupported tail role: {role}")
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


def normalized_weighted_mean(values, weights):
    values = values.view(-1)
    weights = weights.view(-1).to(values)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def residual_mechanism_index(
    anchor: torch.Tensor,
    labels: torch.Tensor,
    tolerance: float = 0.10,
) -> torch.Tensor:
    anchor = anchor.view(-1)
    labels = labels.view(-1)
    result = torch.full_like(labels, 3, dtype=torch.long)
    sign_error = (anchor * labels <= 0.0) & (torch.abs(labels) > tolerance)
    same_sign = ~sign_error
    under = same_sign & (torch.abs(anchor) + tolerance < torch.abs(labels))
    over = same_sign & (torch.abs(anchor) > torch.abs(labels) + tolerance)
    result[sign_error] = 0
    result[under] = 1
    result[over] = 2
    return result


def _balanced_bce(logits, targets):
    logits = logits.view(-1)
    targets = targets.view(-1).to(logits)
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives.item() <= 0 or negatives.item() <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets)
    pos_weight = (negatives / positives).detach().clamp(0.25, 8.0)
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )


def oof_tail_residual_loss(
    outputs,
    labels: torch.Tensor,
    role: str,
    membership_temperature: float,
    gain_margin: float,
    gain_fraction: float,
    weights: OOFTailLossWeights,
):
    role = validate_tail_role(role)
    prediction = outputs["prediction"]
    anchor = outputs["anchor"].detach()
    labels = labels.view_as(prediction)
    membership = soft_tail_membership(
        labels, role, temperature=membership_temperature
    )
    outside = (1.0 - membership).clamp_min(0.0)
    absolute_error = torch.abs(prediction - labels)
    anchor_error = torch.abs(anchor - labels)
    residual_target = (labels - anchor).detach()

    residual_values = F.smooth_l1_loss(
        outputs["correction"], residual_target, reduction="none", beta=0.25
    )
    required_gain = torch.minimum(
        anchor_error * float(gain_fraction),
        anchor_error.new_full(anchor_error.shape, float(gain_margin)),
    )
    gain_values = F.relu(absolute_error - anchor_error + required_gain)
    outside_values = F.smooth_l1_loss(
        prediction, anchor, reduction="none", beta=0.10
    )

    gate_target = exact_tail_mask(labels, role).to(prediction)
    mechanism_target = residual_mechanism_index(anchor, labels)
    mechanism_values = F.cross_entropy(
        outputs["mechanism_logits"], mechanism_target, reduction="none"
    )

    components = {
        "global_mae": absolute_error.mean(),
        "tail_mae": normalized_weighted_mean(absolute_error, membership),
        "residual_huber": normalized_weighted_mean(
            residual_values, membership
        ),
        "gain_margin": normalized_weighted_mean(gain_values, membership),
        "outside_retention": normalized_weighted_mean(
            outside_values, outside + 1e-3
        ),
        "applicability_bce": _balanced_bce(
            outputs["applicability_logit"], gate_target
        ),
        "mechanism_ce": 0.25 * mechanism_values.mean()
        + 0.75 * normalized_weighted_mean(mechanism_values, membership),
        "correction_shrink": outputs["raw_correction"].square().mean()
        + normalized_weighted_mean(
            outputs["correction"].square(), outside + 1e-3
        ),
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
    denominator = torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(1e-12)
    return float((x * y).sum().div(denominator).item())


def tail_capability_metrics(prediction: torch.Tensor, labels: torch.Tensor):
    prediction = prediction.view(-1, 1)
    labels = labels.view(-1, 1)
    result = {
        "global_mae": float(torch.abs(prediction - labels).mean().item()),
        "global_corr": safe_corr(prediction, labels),
    }
    for role in TAIL_ROLE_NAMES:
        mask = exact_tail_mask(labels, role)
        result[f"{role}_count"] = int(mask.sum().item())
        result[f"{role}_mae"] = (
            float(torch.abs(prediction[mask] - labels[mask]).mean().item())
            if mask.any()
            else math.nan
        )
    return result


def residual_diagnostic_rows(
    anchor: torch.Tensor,
    labels: torch.Tensor,
    split: str,
    fold_index: torch.Tensor | None = None,
):
    anchor = anchor.view(-1, 1).detach().cpu()
    labels = labels.view(-1, 1).detach().cpu()
    mechanisms = residual_mechanism_index(anchor, labels)
    residual = labels - anchor
    rows = []
    for role in TAIL_ROLE_NAMES:
        role_mask = exact_tail_mask(labels, role)
        for mechanism_index, mechanism_name in [(-1, "all")] + list(
            enumerate(MECHANISM_NAMES)
        ):
            mask = role_mask.clone()
            if mechanism_index >= 0:
                mask = mask & (mechanisms == mechanism_index)
            count = int(mask.sum().item())
            if count == 0:
                continue
            values = residual.view(-1)[mask]
            errors = torch.abs(values)
            row = {
                "split": split,
                "role": role,
                "mechanism": mechanism_name,
                "count": count,
                "anchor_mae": float(errors.mean().item()),
                "mean_residual": float(values.mean().item()),
                "median_residual": float(values.median().item()),
                "positive_residual_rate": float((values > 0).float().mean().item()),
                "negative_residual_rate": float((values < 0).float().mean().item()),
                "mae_contribution": float(errors.sum().item()),
            }
            if fold_index is not None and mechanism_name == "all":
                fold_values = fold_index.view(-1)[mask]
                row["fold_count"] = int(torch.unique(fold_values).numel())
            rows.append(row)
    return rows


def direction_agreement_summary(
    train_anchor: torch.Tensor,
    train_labels: torch.Tensor,
    valid_anchor: torch.Tensor,
    valid_labels: torch.Tensor,
):
    rows = []
    for role in TAIL_ROLE_NAMES:
        train_mask = exact_tail_mask(train_labels, role)
        valid_mask = exact_tail_mask(valid_labels, role)
        train_mean = float(
            (train_labels.view(-1)[train_mask] - train_anchor.view(-1)[train_mask])
            .mean()
            .item()
        )
        valid_mean = float(
            (valid_labels.view(-1)[valid_mask] - valid_anchor.view(-1)[valid_mask])
            .mean()
            .item()
        )
        rows.append(
            {
                "role": role,
                "oof_train_mean_residual": train_mean,
                "valid_mean_residual": valid_mean,
                "same_direction": bool(
                    train_mean == 0
                    or valid_mean == 0
                    or train_mean * valid_mean > 0
                ),
                "absolute_direction_gap": abs(train_mean - valid_mean),
            }
        )
    return rows
