"""Core objectives, teacher fitting, and diagnostics for V9 experts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)
REGION_CENTERS = (-2.25, -1.0, 0.0, 1.0, 2.25)
REGION_SIGMAS = (0.85, 0.65, 0.55, 0.65, 0.85)


@dataclass(frozen=True)
class LossWeights:
    global_mae: float = 0.55
    role_mae: float = 1.00
    global_distill: float = 0.15
    role_distill: float = 0.55
    gain_margin: float = 0.30
    outside_retention: float = 0.20
    region_ce: float = 0.16
    region_emd: float = 0.08
    region_consistency: float = 0.06
    role_mechanism: float = 0.15
    auxiliary_backbone: float = 0.05
    correction_shrink: float = 0.015
    risk: float = 0.04


def region_index(labels: torch.Tensor) -> torch.Tensor:
    """Map continuous MOSI/MOSEI labels into five fixed semantic regions."""

    values = labels.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def soft_role_membership(
    labels: torch.Tensor,
    role: int,
    floor: float = 0.25,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    if not 0 <= int(role) < len(REGION_NAMES):
        raise ValueError(f"role must be in [0, 4], got {role}")
    if not 0.0 <= float(floor) < 1.0:
        raise ValueError("membership floor must be in [0, 1).")
    values = labels.view(-1)
    center = float(REGION_CENTERS[int(role)])
    sigma = max(1e-4, float(REGION_SIGMAS[int(role)]) * float(sigma_scale))
    gaussian = torch.exp(-0.5 * ((values - center) / sigma).square())
    return float(floor) + (1.0 - float(floor)) * gaussian


def normalized_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    flat_values = values.view(-1)
    flat_weights = weights.view(-1).to(flat_values)
    denominator = flat_weights.sum().clamp_min(1e-8)
    return (flat_values * flat_weights).sum() / denominator


def stable_group_ids(sample_ids: Sequence[object], groups: int = 3) -> torch.Tensor:
    if int(groups) < 2:
        raise ValueError("groups must be at least 2.")
    return torch.tensor(
        [
            int(hashlib.sha1(str(value).encode("utf-8")).hexdigest(), 16)
            % int(groups)
            for value in sample_ids
        ],
        dtype=torch.long,
    )


def apply_teacher_weights(
    predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if predictions.dim() != 3 or predictions.size(-1) != 1:
        raise ValueError("teacher predictions must have shape [N, K, 1].")
    if weights.dim() != 1 or weights.numel() != predictions.size(1):
        raise ValueError("teacher weights must have shape [K].")
    return (predictions * weights.to(predictions).view(1, -1, 1)).sum(dim=1)


def _simplex_objective(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    prior: torch.Tensor,
    groups: torch.Tensor,
    regularization: float,
    stability_weight: float,
) -> torch.Tensor:
    mae = torch.abs(prediction - labels).mean()
    group_losses = []
    for group in torch.unique(groups):
        mask = groups == group
        if mask.any():
            group_losses.append(torch.abs(prediction[mask] - labels[mask]).mean())
    stability = (
        torch.stack(group_losses).std(unbiased=False)
        if len(group_losses) > 1
        else mae.new_zeros(())
    )
    kl = (
        weights
        * (
            weights.clamp_min(1e-8)
            / prior.to(weights).clamp_min(1e-8)
        ).log()
    ).sum()
    return mae + float(stability_weight) * stability + float(regularization) * kl


def fit_simplex(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    prior: Optional[torch.Tensor] = None,
    regularization: float = 0.02,
    stability_weight: float = 0.35,
    steps: int = 500,
    lr: float = 0.05,
) -> torch.Tensor:
    predictions = predictions.detach().cpu().float()
    labels = labels.detach().cpu().float()
    groups = groups.detach().cpu().long()
    teacher_count = predictions.size(1)
    if predictions.size(0) != labels.size(0) or groups.numel() != labels.size(0):
        raise ValueError("simplex inputs have inconsistent sample counts.")
    if predictions.size(0) < 2:
        raise ValueError("at least two samples are required to fit a simplex.")
    if prior is None:
        prior = torch.full((teacher_count,), 1.0 / teacher_count)
    prior = prior.detach().cpu().float()
    if prior.numel() != teacher_count:
        raise ValueError("simplex prior has the wrong number of teachers.")
    prior = prior.clamp_min(1e-6)
    prior = prior / prior.sum()

    logits = prior.log().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=float(lr))
    best = None
    for _ in range(int(steps)):
        weights = torch.softmax(logits, dim=0)
        prediction = apply_teacher_weights(predictions, weights)
        objective = _simplex_objective(
            prediction,
            labels,
            weights,
            prior,
            groups,
            regularization,
            stability_weight,
        )
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        row = (float(objective.item()), weights.detach().clone())
        if best is None or row[0] < best[0]:
            best = row
    if best is None:
        raise RuntimeError("simplex fitting did not produce a candidate.")
    return best[1]


def fit_role_teacher_weights(
    valid_predictions: torch.Tensor,
    valid_labels: torch.Tensor,
    sample_ids: Sequence[object],
    regularizations: Tuple[float, ...] = (0.03, 0.08, 0.16, 0.32, 0.64),
    folds: int = 3,
    steps: int = 450,
    min_region_samples: int = 12,
) -> Dict[str, object]:
    """Validation-only fitting of global and role-specific teacher targets.

    Each role-specific simplex is shrunk toward the global simplex. The amount
    of shrinkage is selected by deterministic hash-fold validation within each
    true label region. Sparse regions fall back to the global teacher target.
    """

    predictions = valid_predictions.detach().cpu().float()
    labels = valid_labels.detach().cpu().float()
    groups = stable_group_ids(sample_ids, folds)
    regions = region_index(labels)
    uniform = torch.full((predictions.size(1),), 1.0 / predictions.size(1))
    global_weights = fit_simplex(
        predictions,
        labels,
        groups,
        prior=uniform,
        regularization=0.01,
        steps=steps,
    )

    cv_rows = []
    selected_regularizations = []
    final_role_weights = []
    for role in range(len(REGION_NAMES)):
        role_mask = regions == role
        role_count = int(role_mask.sum().item())
        if role_count < int(min_region_samples):
            selected_regularizations.append(None)
            final_role_weights.append(global_weights.clone())
            cv_rows.append(
                {
                    "role": REGION_NAMES[role],
                    "fold": None,
                    "regularization": None,
                    "count": role_count,
                    "mae": None,
                    "status": "global_fallback_sparse_region",
                }
            )
            continue

        candidates: Dict[float, list] = {
            float(value): [] for value in regularizations
        }
        for held_out in range(int(folds)):
            train_mask = role_mask & (groups != held_out)
            held_mask = role_mask & (groups == held_out)
            if int(train_mask.sum().item()) < max(6, predictions.size(1)) or not held_mask.any():
                continue
            for regularization in regularizations:
                weights = fit_simplex(
                    predictions[train_mask],
                    labels[train_mask],
                    groups[train_mask],
                    prior=global_weights,
                    regularization=float(regularization),
                    steps=steps,
                )
                held_prediction = apply_teacher_weights(
                    predictions[held_mask], weights
                )
                mae = float(
                    torch.abs(held_prediction - labels[held_mask]).mean().item()
                )
                candidates[float(regularization)].append(mae)
                cv_rows.append(
                    {
                        "role": REGION_NAMES[role],
                        "fold": int(held_out),
                        "regularization": float(regularization),
                        "count": int(held_mask.sum().item()),
                        "mae": mae,
                        "status": "cv",
                    }
                )

        scored = [
            (sum(values) / len(values), regularization)
            for regularization, values in candidates.items()
            if values
        ]
        if not scored:
            selected_regularizations.append(None)
            final_role_weights.append(global_weights.clone())
            continue
        _, selected = min(scored)
        selected_regularizations.append(float(selected))
        final_role_weights.append(
            fit_simplex(
                predictions[role_mask],
                labels[role_mask],
                groups[role_mask],
                prior=global_weights,
                regularization=float(selected),
                steps=steps,
            )
        )

    role_weights = torch.stack(final_role_weights, dim=0)
    return {
        "global_weights": global_weights,
        "role_weights": role_weights,
        "selected_regularizations": selected_regularizations,
        "cv_rows": cv_rows,
        "region_counts": [
            int((regions == role).sum().item())
            for role in range(len(REGION_NAMES))
        ],
    }


def region_emd_loss(region_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(targets.long(), num_classes=5).to(region_probs)
    predicted_cdf = torch.cumsum(region_probs, dim=1)
    target_cdf = torch.cumsum(one_hot, dim=1)
    return torch.abs(predicted_cdf - target_cdf).mean(dim=1)


def _pairwise_tail_ranking(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    role: int,
    margin: float = 0.08,
) -> torch.Tensor:
    values = labels.view(-1)
    predicted = prediction.view(-1)
    if role == 0:
        mask = values < -1.5
    elif role == 4:
        mask = values > 1.5
    else:
        return prediction.new_zeros(())
    values = values[mask]
    predicted = predicted[mask]
    if values.numel() < 2:
        return prediction.new_zeros(())
    true_difference = torch.abs(values).unsqueeze(1) - torch.abs(values).unsqueeze(0)
    pair_mask = true_difference > 0.25
    if not pair_mask.any():
        return prediction.new_zeros(())
    predicted_difference = (
        torch.abs(predicted).unsqueeze(1) - torch.abs(predicted).unsqueeze(0)
    )
    return F.relu(float(margin) - predicted_difference[pair_mask]).mean()


def role_mechanism_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    role: int,
) -> torch.Tensor:
    flat_prediction = prediction.view(-1)
    flat_labels = labels.view(-1)
    if role in (0, 4):
        sign_penalty = (
            F.softplus(flat_prediction)
            if role == 0
            else F.softplus(-flat_prediction)
        )
        under_strength = F.relu(torch.abs(flat_labels) - torch.abs(flat_prediction))
        local = normalized_weighted_mean(sign_penalty + under_strength, membership)
        return local + 0.5 * _pairwise_tail_ranking(prediction, labels, role)
    if role == 1:
        return normalized_weighted_mean(F.softplus(flat_prediction), membership)
    if role == 3:
        return normalized_weighted_mean(F.softplus(-flat_prediction), membership)
    boundary_distance = torch.abs(torch.abs(flat_prediction) - torch.abs(flat_labels))
    overshoot = F.relu(torch.abs(flat_prediction) - 0.75)
    return normalized_weighted_mean(boundary_distance + 0.5 * overshoot, membership)


def role_conditioned_loss(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    anchor_prediction: torch.Tensor,
    global_teacher_prediction: torch.Tensor,
    role_teacher_prediction: torch.Tensor,
    role: int,
    membership_floor: float,
    membership_sigma_scale: float,
    gain_margin: float,
    gain_fraction: float,
    weights: LossWeights,
) -> Dict[str, torch.Tensor]:
    prediction = outputs["prediction"]
    labels = labels.view_as(prediction)
    anchor_prediction = anchor_prediction.view_as(prediction)
    global_teacher_prediction = global_teacher_prediction.view_as(prediction)
    role_teacher_prediction = role_teacher_prediction.view_as(prediction)
    membership = soft_role_membership(
        labels,
        role,
        floor=membership_floor,
        sigma_scale=membership_sigma_scale,
    )
    outside = (1.0 - membership).clamp_min(0.0)

    absolute_error = torch.abs(prediction - labels)
    anchor_error = torch.abs(anchor_prediction - labels).detach()
    global_mae = absolute_error.mean()
    role_mae = normalized_weighted_mean(absolute_error, membership)

    global_distill = F.smooth_l1_loss(prediction, global_teacher_prediction)
    role_distill = normalized_weighted_mean(
        F.smooth_l1_loss(
            prediction, role_teacher_prediction, reduction="none"
        ),
        membership,
    )
    required_gain = torch.minimum(
        anchor_error * float(gain_fraction),
        anchor_error.new_full(anchor_error.shape, float(gain_margin)),
    )
    gain_loss = normalized_weighted_mean(
        F.relu(absolute_error - anchor_error + required_gain), membership
    )
    outside_retention = normalized_weighted_mean(
        F.smooth_l1_loss(prediction, anchor_prediction, reduction="none"),
        outside + 1e-3,
    )

    targets = region_index(labels)
    ce = F.cross_entropy(outputs["region_logits"], targets, reduction="none")
    region_ce = 0.35 * ce.mean() + 0.65 * normalized_weighted_mean(ce, membership)
    emd_values = region_emd_loss(outputs["region_probs"], targets)
    region_emd = 0.35 * emd_values.mean() + 0.65 * normalized_weighted_mean(
        emd_values, membership
    )
    region_consistency = torch.abs(
        outputs["region_expected"] - prediction.detach()
    ).mean()

    mechanism = role_mechanism_loss(prediction, labels, membership, role)
    backbone = outputs["backbone"]
    auxiliary = sum(
        F.l1_loss(backbone[key], labels)
        for key in (
            "logits_c",
            "logits_l_hetero",
            "logits_a_hetero",
            "logits_v_hetero",
        )
    ) / 4.0
    correction_shrink = outputs["correction"].square().mean()
    risk_target = absolute_error.detach()
    risk_loss = F.smooth_l1_loss(outputs["predicted_abs_error"], risk_target)

    components = {
        "global_mae": global_mae,
        "role_mae": role_mae,
        "global_distill": global_distill,
        "role_distill": role_distill,
        "gain_margin": gain_loss,
        "outside_retention": outside_retention,
        "region_ce": region_ce,
        "region_emd": region_emd,
        "region_consistency": region_consistency,
        "role_mechanism": mechanism,
        "auxiliary_backbone": auxiliary,
        "correction_shrink": correction_shrink,
        "risk": risk_loss,
    }
    total = sum(
        getattr(weights, name) * value for name, value in components.items()
    )
    return {"total": total, **components}


def safe_corr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    x = prediction.view(-1).double()
    y = target.view(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(1e-12)
    return float((x * y).sum().div(denominator).item())


def region_metrics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    regions = region_index(labels)
    result: Dict[str, float] = {
        "global_mae": float(torch.abs(prediction - labels).mean().item()),
        "global_corr": safe_corr(prediction, labels),
    }
    for role, name in enumerate(REGION_NAMES):
        mask = regions == role
        result[f"{name}_count"] = int(mask.sum().item())
        result[f"{name}_mae"] = (
            float(torch.abs(prediction[mask] - labels[mask]).mean().item())
            if mask.any()
            else math.nan
        )
    return result
