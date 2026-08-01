"""Ordinal five-region classifier and fixed region-to-expert routing for V9.11."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SemanticCostCoachV99 import (
    ACTION_NAMES,
    SIGNATURE_DIM,
    global_context_features,
)

REGION_ROUTER_VERSION = "predicted_region_router_v1"
REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)
REGION_BOUNDARIES = (-1.5, -0.5, 0.5, 1.5)
# ACTION_NAMES = anchor, strong_negative, boundary, positive, strong_positive.
SEMANTIC_REGION_ACTION_MAP = (1, 0, 2, 3, 4)


def region_index(labels: torch.Tensor) -> torch.Tensor:
    values = labels.detach().view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def cumulative_targets(labels: torch.Tensor) -> torch.Tensor:
    values = labels.view(-1)
    return torch.stack(
        (
            values >= -1.5,
            values >= -0.5,
            values > 0.5,
            values > 1.5,
        ),
        dim=1,
    ).to(values.dtype)


def cumulative_to_region_probs(cumulative: torch.Tensor) -> torch.Tensor:
    if cumulative.dim() != 2 or cumulative.size(1) != 4:
        raise ValueError("cumulative probabilities must have shape [N,4]")
    q0, q1, q2, q3 = cumulative.unbind(dim=1)
    probabilities = torch.stack(
        (1.0 - q0, q0 - q1, q1 - q2, q2 - q3, q3),
        dim=1,
    ).clamp_min(0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def region_router_features(
    function_space: torch.Tensor,
    actions: torch.Tensor,
    specialist_signatures: torch.Tensor,
) -> torch.Tensor:
    """Build fold-aligned region features from candidate geometry and semantics."""
    if function_space.dim() != 2 or function_space.size(1) != 4:
        raise ValueError("function_space must have shape [N,4]")
    if actions.dim() != 3 or actions.shape[1:] != (len(ACTION_NAMES), 1):
        raise ValueError("actions must have shape [N,5,1]")
    if specialist_signatures.dim() != 3 or specialist_signatures.shape[1:] != (
        4,
        SIGNATURE_DIM,
    ):
        raise ValueError(
            f"specialist_signatures must have shape [N,4,{SIGNATURE_DIM}]"
        )
    context = global_context_features(function_space, actions)
    signatures = specialist_signatures.to(context)
    values = actions.squeeze(-1).to(context)
    boundaries = values.new_tensor(REGION_BOUNDARIES).view(1, 1, -1)
    boundary_offsets = (values.unsqueeze(-1) - boundaries).reshape(len(values), -1)

    signature_mean = signatures.mean(dim=1)
    signature_std = signatures.std(dim=1, unbiased=False)
    # Fixed semantic subspaces from semantic_expert_signature_v1.
    region_profiles = signatures[:, :, 12:17]
    region_mean = region_profiles.mean(dim=1)
    region_std = region_profiles.std(dim=1, unbiased=False)
    confidence = signatures[:, :, 3]
    self_error = signatures[:, :, 5]
    entropy = signatures[:, :, 7:9].reshape(len(values), -1)

    return torch.cat(
        (
            context,
            signatures.reshape(len(values), -1),
            signature_mean,
            signature_std,
            boundary_offsets,
            region_mean,
            region_std,
            confidence,
            self_error,
            entropy,
        ),
        dim=1,
    )


class PredictedRegionRouterV911(nn.Module):
    """Predict a sentiment region using a fixed ordinal/categorical hybrid head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        residual_max: float = 2.0,
        ordinal_temperature: float = 0.45,
        ordinal_blend: float = 0.50,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.residual_max = float(residual_max)
        self.ordinal_temperature = float(ordinal_temperature)
        self.ordinal_blend = float(ordinal_blend)
        if not 0.0 <= self.ordinal_blend <= 1.0:
            raise ValueError("ordinal_blend must be in [0,1]")
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.score_residual = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, len(REGION_NAMES))
        self.confidence_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score_residual.weight)
        nn.init.zeros_(self.score_residual.bias)
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        anchor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if features.dim() != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"features must have shape [N,{self.input_dim}], got {tuple(features.shape)}"
            )
        anchor = anchor.view(-1, 1).to(features)
        latent = self.encoder(features)
        score = anchor + self.residual_max * torch.tanh(self.score_residual(latent))
        boundaries = score.new_tensor(REGION_BOUNDARIES).view(1, -1)
        ordinal_logits = (score - boundaries) / max(self.ordinal_temperature, 1e-6)
        ordinal_probs = cumulative_to_region_probs(torch.sigmoid(ordinal_logits))
        class_logits = self.class_head(latent)
        class_probs = torch.softmax(class_logits, dim=1)
        region_probs = (
            self.ordinal_blend * ordinal_probs
            + (1.0 - self.ordinal_blend) * class_probs
        )
        region_probs = region_probs / region_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        confidence = torch.sigmoid(self.confidence_head(latent))
        return {
            "score": score,
            "ordinal_logits": ordinal_logits,
            "ordinal_probs": ordinal_probs,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "region_probs": region_probs,
            "confidence": confidence,
            "latent": latent,
        }


def predicted_region_router_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    ordinal_weight: float = 0.70,
    class_weight: float = 1.00,
    regression_weight: float = 0.35,
    expected_region_weight: float = 0.20,
    confidence_weight: float = 0.05,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    target_region = region_index(labels).to(output["class_logits"].device)
    target_cumulative = cumulative_targets(labels).to(output["ordinal_logits"])
    ordinal_bce = F.binary_cross_entropy_with_logits(
        output["ordinal_logits"], target_cumulative
    )
    class_ce = F.cross_entropy(
        output["class_logits"],
        target_region,
        weight=None if class_weights is None else class_weights.to(output["class_logits"]),
    )
    regression_huber = F.smooth_l1_loss(
        output["score"], labels.view(-1, 1).to(output["score"]), beta=0.20
    )
    region_values = output["region_probs"].new_tensor(
        (-2.0, -1.0, 0.0, 1.0, 2.0)
    ).view(1, -1)
    expected_region = (output["region_probs"] * region_values).sum(dim=1)
    expected_region_huber = F.smooth_l1_loss(
        expected_region,
        region_values.view(-1)[target_region],
        beta=0.50,
    )
    correctness = (
        output["region_probs"].detach().argmax(dim=1) == target_region
    ).to(output["confidence"])
    confidence_bce = F.binary_cross_entropy(
        output["confidence"].view(-1), correctness.view(-1)
    )
    total = (
        float(ordinal_weight) * ordinal_bce
        + float(class_weight) * class_ce
        + float(regression_weight) * regression_huber
        + float(expected_region_weight) * expected_region_huber
        + float(confidence_weight) * confidence_bce
    )
    return {
        "total": total,
        "ordinal_bce": ordinal_bce,
        "class_ce": class_ce,
        "regression_huber": regression_huber,
        "expected_region_huber": expected_region_huber,
        "confidence_bce": confidence_bce,
    }


def region_classification_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    target = region_index(labels).cpu()
    predicted = probabilities.argmax(dim=1).cpu()
    f1_values = []
    recalls = []
    for category in range(len(REGION_NAMES)):
        true_positive = int(((predicted == category) & (target == category)).sum())
        false_positive = int(((predicted == category) & (target != category)).sum())
        false_negative = int(((predicted != category) & (target == category)).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        f1_values.append(f1)
        recalls.append(recall)
    region_axis = torch.arange(len(REGION_NAMES), dtype=torch.float32).view(1, -1)
    expected = (probabilities.cpu() * region_axis).sum(dim=1)
    result = {
        "accuracy": float((predicted == target).float().mean()),
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "ordinal_index_mae": float(torch.abs(expected - target.float()).mean()),
        "mean_max_probability": float(probabilities.max(dim=1).values.mean()),
    }
    for index, name in enumerate(REGION_NAMES):
        result[f"recall_{name}"] = float(recalls[index])
    return result


def action_indices_from_regions(
    region_indices: torch.Tensor,
    mapping: Sequence[int] = SEMANTIC_REGION_ACTION_MAP,
) -> torch.Tensor:
    if len(mapping) != len(REGION_NAMES):
        raise ValueError("region-action mapping must have five entries")
    table = region_indices.new_tensor(tuple(int(value) for value in mapping))
    return table[region_indices.view(-1)]


def route_by_regions(
    actions: torch.Tensor,
    region_indices: torch.Tensor,
    mapping: Sequence[int] = SEMANTIC_REGION_ACTION_MAP,
) -> Dict[str, torch.Tensor]:
    action_indices = action_indices_from_regions(region_indices, mapping)
    prediction = actions.squeeze(-1).gather(1, action_indices.view(-1, 1))
    return {
        "action_indices": action_indices,
        "prediction": prediction,
    }
