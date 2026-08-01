"""Selective ordinal category coach used with the frozen V9.3 expert pool."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)
REGION_BOUNDARIES = (-1.5, -0.5, 0.5, 1.5)
REGION_CENTERS = (-2.25, -1.0, 0.0, 1.0, 2.25)
SPECIALIST_NAMES = (
    "strong_negative",
    "boundary",
    "positive",
    "strong_positive",
)
SPECIALIST_TO_REGION = {
    "strong_negative": 0,
    "boundary": 2,
    "positive": 3,
    "strong_positive": 4,
}
REGION_TO_SPECIALIST = {value: key for key, value in SPECIALIST_TO_REGION.items()}


def region_index(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def cumulative_targets(labels: torch.Tensor) -> torch.Tensor:
    values = labels.view(-1)
    return torch.stack(
        [
            values >= -1.5,
            values >= -0.5,
            values > 0.5,
            values > 1.5,
        ],
        dim=1,
    ).to(labels.dtype)


def cumulative_to_region_probs(cumulative: torch.Tensor) -> torch.Tensor:
    if cumulative.dim() != 2 or cumulative.size(1) != 4:
        raise ValueError("cumulative probabilities must have shape [N, 4]")
    q0, q1, q2, q3 = cumulative.unbind(dim=1)
    result = torch.stack(
        [1.0 - q0, q0 - q1, q1 - q2, q2 - q3, q3], dim=1
    ).clamp_min(0.0)
    return result / result.sum(dim=1, keepdim=True).clamp_min(1e-8)


def semantic_region_probabilities(score: torch.Tensor, temperature: float) -> torch.Tensor:
    score = score.view(-1, 1)
    boundaries = score.new_tensor(REGION_BOUNDARIES).view(1, -1)
    cumulative = torch.sigmoid((score - boundaries) / max(float(temperature), 1e-5))
    return cumulative_to_region_probs(cumulative)


def boundary_distance(score: torch.Tensor) -> torch.Tensor:
    score = score.view(-1, 1)
    boundaries = score.new_tensor(REGION_BOUNDARIES).view(1, -1)
    return torch.abs(score - boundaries).amin(dim=1)


def coach_input_features(function_space: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    if function_space.dim() != 2 or function_space.size(1) != 4:
        raise ValueError("function_space must have shape [N,4]")
    anchor = anchor.view(-1, 1).to(function_space)
    mean = function_space.mean(dim=1, keepdim=True)
    std = function_space.std(dim=1, keepdim=True, unbiased=False)
    value_range = function_space.amax(dim=1, keepdim=True) - function_space.amin(
        dim=1, keepdim=True
    )
    disagreement = torch.mean(torch.abs(function_space - anchor), dim=1, keepdim=True)
    return torch.cat(
        [
            function_space,
            anchor,
            anchor.abs(),
            torch.tanh(anchor),
            mean,
            std,
            value_range,
            disagreement,
        ],
        dim=1,
    )


class SelectiveOrdinalCalibratorV95(nn.Module):
    """A strongly anchor-regularized ordinal score calibrator.

    The semantic boundaries remain fixed. The network may only make a bounded
    correction to the anchor score, which prevents a high-capacity region model
    from replacing the already competitive anchor-threshold baseline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.10,
        residual_max: float = 0.50,
        temperature: float = 0.45,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.residual_max = float(residual_max)
        self.temperature = float(temperature)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, features: torch.Tensor, anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"features must have shape [N,{self.input_dim}], got {tuple(features.shape)}"
            )
        anchor = anchor.view(-1, 1).to(features)
        latent = self.encoder(features)
        correction = self.residual_max * torch.tanh(self.residual_head(latent))
        score = anchor + correction
        boundaries = score.new_tensor(REGION_BOUNDARIES).view(1, -1)
        logits = (score - boundaries) / max(self.temperature, 1e-5)
        cumulative = torch.sigmoid(logits)
        return {
            "score": score,
            "correction": correction,
            "ordinal_logits": logits,
            "cumulative_probs": cumulative,
            "region_probs": cumulative_to_region_probs(cumulative),
        }


def selective_ordinal_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    anchor: torch.Tensor,
    ordinal_weight: float = 1.0,
    regression_weight: float = 0.35,
    retention_weight: float = 0.30,
) -> Dict[str, torch.Tensor]:
    labels = labels.view(-1, 1).to(output["score"])
    anchor = anchor.view(-1, 1).to(output["score"])
    target = cumulative_targets(labels).to(output["ordinal_logits"])
    ordinal = F.binary_cross_entropy_with_logits(output["ordinal_logits"], target)
    regression = F.smooth_l1_loss(output["score"], labels, beta=0.25)
    retention = F.smooth_l1_loss(output["score"], anchor, beta=0.10)
    total = (
        float(ordinal_weight) * ordinal
        + float(regression_weight) * regression
        + float(retention_weight) * retention
    )
    return {
        "total": total,
        "ordinal_bce": ordinal,
        "regression_huber": regression,
        "anchor_retention": retention,
    }


def specialist_center_progress(
    anchor: torch.Tensor,
    expert: torch.Tensor,
    region: int,
) -> torch.Tensor:
    center = anchor.new_tensor(float(REGION_CENTERS[int(region)]))
    return torch.abs(expert.view(-1) - center) < torch.abs(anchor.view(-1) - center)
