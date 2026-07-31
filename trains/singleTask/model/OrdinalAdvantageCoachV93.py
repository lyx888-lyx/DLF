"""Ordinal region classifier and low-capacity expert-advantage heads for V9.3."""

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
        [
            values >= -1.5,
            values >= -0.5,
            values > 0.5,
            values > 1.5,
        ],
        dim=1,
    ).to(values.dtype)


def cumulative_to_region_probs(cumulative: torch.Tensor) -> torch.Tensor:
    """Convert monotone P(y > boundary_j) values into five region probabilities."""
    if cumulative.dim() != 2 or cumulative.size(1) != 4:
        raise ValueError("cumulative probabilities must have shape [N, 4]")
    q0, q1, q2, q3 = cumulative.unbind(dim=1)
    probabilities = torch.stack(
        [1.0 - q0, q0 - q1, q1 - q2, q2 - q3, q3], dim=1
    )
    probabilities = probabilities.clamp_min(0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def coach_input_features(function_space: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Build a fold-aligned compact feature vector from task-space predictions."""
    if function_space.dim() != 2 or function_space.size(1) != 4:
        raise ValueError("function_space must have shape [N, 4]")
    anchor = anchor.view(-1, 1).to(function_space)
    branch_mean = function_space.mean(dim=1, keepdim=True)
    branch_std = function_space.std(dim=1, keepdim=True, unbiased=False)
    branch_range = function_space.amax(dim=1, keepdim=True) - function_space.amin(
        dim=1, keepdim=True
    )
    return torch.cat(
        [
            function_space,
            anchor,
            torch.abs(anchor),
            anchor.square(),
            torch.tanh(anchor),
            branch_mean,
            branch_std,
            branch_range,
        ],
        dim=1,
    )


class OrdinalRegionCoachV93(nn.Module):
    """Predict a continuous sentiment score and monotone five-region probabilities."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
        residual_max: float = 2.0,
        ordinal_temperature: float = 0.45,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.residual_max = float(residual_max)
        self.ordinal_temperature = float(ordinal_temperature)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.score_residual = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score_residual.weight)
        nn.init.zeros_(self.score_residual.bias)

    def forward(self, features: torch.Tensor, anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"features must have shape [N, {self.input_dim}], got {tuple(features.shape)}"
            )
        anchor = anchor.view(-1, 1).to(features)
        latent = self.adapter(features)
        score = anchor + self.residual_max * torch.tanh(self.score_residual(latent))
        boundaries = score.new_tensor(REGION_BOUNDARIES).view(1, -1)
        ordinal_logits = (score - boundaries) / max(self.ordinal_temperature, 1e-6)
        cumulative = torch.sigmoid(ordinal_logits)
        region_probs = cumulative_to_region_probs(cumulative)
        confidence = torch.sigmoid(self.confidence_head(latent))
        return {
            "score": score,
            "ordinal_logits": ordinal_logits,
            "cumulative_probs": cumulative,
            "region_probs": region_probs,
            "region_confidence": confidence,
            "latent": latent,
        }


class AdvantageHeadV93(nn.Module):
    """Estimate specialist win probability and expected MAE gain over the anchor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        dropout: float = 0.10,
        gain_max: float = 1.5,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.gain_max = float(gain_max)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.win_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.zeros_(self.win_head.weight)
        nn.init.zeros_(self.win_head.bias)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"features must have shape [N, {self.input_dim}], got {tuple(features.shape)}"
            )
        latent = self.adapter(features)
        predicted_gain = self.gain_max * torch.tanh(self.gain_head(latent))
        win_logit = self.win_head(latent)
        return {
            "predicted_gain": predicted_gain,
            "win_logit": win_logit,
            "win_probability": torch.sigmoid(win_logit),
        }


def advantage_input_features(
    coach_features: torch.Tensor,
    region_probs: torch.Tensor,
    ordinal_score: torch.Tensor,
    anchor: torch.Tensor,
    expert_prediction: torch.Tensor,
    specialist_confidence: torch.Tensor,
    designated_region: int,
) -> torch.Tensor:
    """Create expert-specific meta-features without using ground truth."""
    anchor = anchor.view(-1, 1).to(coach_features)
    expert_prediction = expert_prediction.view(-1, 1).to(coach_features)
    specialist_confidence = specialist_confidence.view(-1, 1).to(coach_features)
    ordinal_score = ordinal_score.view(-1, 1).to(coach_features)
    correction = expert_prediction - anchor
    designated = region_probs[:, int(designated_region)].view(-1, 1)
    safe_probs = region_probs.clamp_min(1e-8)
    entropy = -(safe_probs * torch.log(safe_probs)).sum(dim=1, keepdim=True)
    return torch.cat(
        [
            coach_features,
            region_probs,
            ordinal_score,
            designated,
            entropy,
            expert_prediction,
            correction,
            torch.abs(correction),
            specialist_confidence,
        ],
        dim=1,
    )


def ordinal_region_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    ordinal_weight: float = 1.0,
    regression_weight: float = 0.5,
    confidence_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    targets = cumulative_targets(labels).to(output["ordinal_logits"])
    ordinal = F.binary_cross_entropy_with_logits(output["ordinal_logits"], targets)
    regression = F.smooth_l1_loss(output["score"], labels.view(-1, 1).to(output["score"]))
    predicted_region = output["region_probs"].argmax(dim=1)
    true_region = region_index(labels).to(predicted_region.device)
    correctness = (predicted_region.detach() == true_region).to(output["region_confidence"])
    confidence = F.binary_cross_entropy(
        output["region_confidence"].view(-1), correctness.view(-1)
    )
    total = (
        float(ordinal_weight) * ordinal
        + float(regression_weight) * regression
        + float(confidence_weight) * confidence
    )
    return {
        "total": total,
        "ordinal_bce": ordinal,
        "regression_huber": regression,
        "confidence_bce": confidence,
    }


def advantage_loss(
    output: Dict[str, torch.Tensor],
    realized_gain: torch.Tensor,
    win_margin: float = 0.02,
    gain_weight: float = 1.0,
    win_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    realized_gain = realized_gain.view(-1, 1).to(output["predicted_gain"])
    win_target = (realized_gain > float(win_margin)).to(output["win_logit"])
    gain = F.smooth_l1_loss(output["predicted_gain"], realized_gain)
    positive = float(win_target.sum().item())
    negative = float(win_target.numel() - positive)
    pos_weight = output["win_logit"].new_tensor(
        min(10.0, max(0.25, negative / max(positive, 1.0)))
    )
    win = F.binary_cross_entropy_with_logits(
        output["win_logit"], win_target, pos_weight=pos_weight
    )
    total = float(gain_weight) * gain + float(win_weight) * win
    return {"total": total, "gain_huber": gain, "win_bce": win}
