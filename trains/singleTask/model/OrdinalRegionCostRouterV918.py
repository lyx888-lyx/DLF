"""Constrained ordinal region-probability model for V9.18."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_VERSION = "ordinal_region_cost_router_v918_v1"
REGION_THRESHOLDS = (-1.5, -0.5, 0.5, 1.5)


class FeatureStandardizer(nn.Module):
    """Frozen feature normalization stored inside each router checkpoint."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        mean = mean.detach().float().view(1, -1)
        scale = scale.detach().float().view(1, -1).clamp_min(1e-4)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.scale


class OrdinalRegionCostRouterV918(nn.Module):
    """Small CORAL-style model with guaranteed ordered cumulative probabilities."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        input_dim = int(feature_mean.numel())
        self.standardizer = FeatureStandardizer(feature_mean, feature_scale)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), max(16, int(hidden_dim) // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
        )
        latent_dim = max(16, int(hidden_dim) // 2)
        self.score_head = nn.Linear(latent_dim, 1)
        self.cutpoint_start = nn.Parameter(torch.tensor([-1.5], dtype=torch.float32))
        initial_gap = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        self.cutpoint_raw_gaps = nn.Parameter(
            torch.log(torch.expm1(initial_gap).clamp_min(1e-4))
        )
        nn.init.zeros_(self.score_head.bias)

    def ordered_cutpoints(self) -> torch.Tensor:
        gaps = F.softplus(self.cutpoint_raw_gaps).clamp_min(1e-3)
        return torch.cat(
            [self.cutpoint_start, self.cutpoint_start + torch.cumsum(gaps, dim=0)],
            dim=0,
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encoder(self.standardizer(features))
        score = self.score_head(latent)
        cutpoints = self.ordered_cutpoints()
        logits = score - cutpoints.view(1, -1)
        return {
            "score": score,
            "logits": logits,
            "cutpoints": cutpoints,
            "latent": latent,
        }


def cumulative_targets(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.view(-1, 1)
    thresholds = labels.new_tensor(REGION_THRESHOLDS).view(1, -1)
    return (labels > thresholds).to(labels.dtype)


def region_probabilities_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    temperature = max(1e-4, float(temperature))
    cumulative = torch.sigmoid(logits / temperature)
    probabilities = torch.cat(
        [
            1.0 - cumulative[:, 0:1],
            cumulative[:, 0:1] - cumulative[:, 1:2],
            cumulative[:, 1:2] - cumulative[:, 2:3],
            cumulative[:, 2:3] - cumulative[:, 3:4],
            cumulative[:, 3:4],
        ],
        dim=1,
    )
    probabilities = probabilities.clamp_min(0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def ordinal_region_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    positive_weight: torch.Tensor,
    score_penalty: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    targets = cumulative_targets(labels).to(output["logits"])
    positive_weight = positive_weight.to(output["logits"]).view(1, -1)
    raw = F.binary_cross_entropy_with_logits(
        output["logits"], targets, reduction="none"
    )
    weights = torch.where(targets > 0.5, positive_weight, torch.ones_like(raw))
    ordinal_bce = (raw * weights).mean()
    score_l2 = output["score"].square().mean()
    total = ordinal_bce + float(score_penalty) * score_l2
    return {
        "total": total,
        "ordinal_bce": ordinal_bce,
        "score_l2": score_l2,
    }


__all__ = [
    "MODEL_VERSION",
    "REGION_THRESHOLDS",
    "FeatureStandardizer",
    "OrdinalRegionCostRouterV918",
    "cumulative_targets",
    "region_probabilities_from_logits",
    "ordinal_region_loss",
]
