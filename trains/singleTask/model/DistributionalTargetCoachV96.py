"""Distributional target-value coach for nearest-expert routing in V9.6."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
SPECIALIST_NAMES = (
    "strong_negative",
    "boundary",
    "positive",
    "strong_positive",
)
ACTION_NAMES = ("anchor", *SPECIALIST_NAMES)
REGION_BOUNDARIES = (-1.5, -0.5, 0.5, 1.5)


def coach_input_features(function_space: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Build label-free target-coach features aligned across OOF and deployment."""
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
            anchor.square(),
            torch.tanh(anchor),
            mean,
            std,
            value_range,
            disagreement,
        ],
        dim=1,
    )


class DistributionalTargetCoachV96(nn.Module):
    """Predict monotone conditional target quantiles around the Anchor score."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        dropout: float = 0.10,
        residual_max: float = 1.50,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.residual_max = float(residual_max)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.median_head = nn.Linear(hidden_dim, 1)
        self.gap_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.median_head.weight)
        nn.init.zeros_(self.median_head.bias)
        nn.init.zeros_(self.gap_head.weight)
        nn.init.constant_(self.gap_head.bias, -2.0)

    def forward(self, features: torch.Tensor, anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"features must have shape [N,{self.input_dim}], got {tuple(features.shape)}"
            )
        anchor = anchor.view(-1, 1).to(features)
        latent = self.encoder(features)
        correction = self.residual_max * torch.tanh(self.median_head(latent))
        median = anchor + correction
        gaps = F.softplus(self.gap_head(latent))
        lower_inner, lower_outer, upper_inner, upper_outer = gaps.split(1, dim=1)
        q25 = median - lower_inner
        q10 = q25 - lower_outer
        q75 = median + upper_inner
        q90 = q75 + upper_outer
        quantiles = torch.cat([q10, q25, median, q75, q90], dim=1)
        return {
            "quantiles": quantiles,
            "median": median,
            "correction": correction,
            "interval_width_80": q90 - q10,
            "interval_width_50": q75 - q25,
            "latent": latent,
        }


def pinball_loss(
    quantiles: torch.Tensor,
    labels: torch.Tensor,
    levels: Iterable[float] = QUANTILE_LEVELS,
) -> torch.Tensor:
    labels = labels.view(-1, 1).to(quantiles)
    tau = quantiles.new_tensor(tuple(float(value) for value in levels)).view(1, -1)
    if quantiles.size(1) != tau.size(1):
        raise ValueError("quantile count does not match levels")
    error = labels - quantiles
    return torch.maximum(tau * error, (tau - 1.0) * error).mean()


def distributional_target_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    anchor: torch.Tensor,
    quantile_weight: float = 1.0,
    median_weight: float = 0.40,
    retention_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    labels = labels.view(-1, 1).to(output["median"])
    anchor = anchor.view(-1, 1).to(output["median"])
    quantile = pinball_loss(output["quantiles"], labels)
    median = F.smooth_l1_loss(output["median"], labels, beta=0.25)
    retention = F.smooth_l1_loss(output["median"], anchor, beta=0.25)
    total = (
        float(quantile_weight) * quantile
        + float(median_weight) * median
        + float(retention_weight) * retention
    )
    return {
        "total": total,
        "pinball": quantile,
        "median_huber": median,
        "anchor_retention": retention,
    }


def stack_action_predictions(
    anchor: torch.Tensor,
    expert_predictions: torch.Tensor,
) -> torch.Tensor:
    """Return [N,5,1] predictions in ACTION_NAMES order."""
    anchor = anchor.view(-1, 1, 1)
    if expert_predictions.dim() != 3 or expert_predictions.size(1) != 4:
        raise ValueError("expert_predictions must have shape [N,4,1]")
    if expert_predictions.size(0) != anchor.size(0):
        raise ValueError("action prediction sample count mismatch")
    return torch.cat([anchor, expert_predictions], dim=1)


def action_risk(
    actions: torch.Tensor,
    target_output: Dict[str, torch.Tensor],
    mode: str,
) -> torch.Tensor:
    """Estimate absolute-error risk of each action from the predicted target distribution."""
    if actions.dim() != 3 or actions.size(1) != len(ACTION_NAMES):
        raise ValueError("actions must have shape [N,5,1]")
    action_values = actions.squeeze(-1)
    if mode == "median_distance":
        return torch.abs(action_values - target_output["median"].view(-1, 1))
    if mode == "quantile_risk":
        quantiles = target_output["quantiles"].to(action_values)
        return torch.abs(
            action_values.unsqueeze(-1) - quantiles.unsqueeze(1)
        ).mean(dim=2)
    raise ValueError(f"unknown target-risk mode: {mode}")


def target_region_index(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result
