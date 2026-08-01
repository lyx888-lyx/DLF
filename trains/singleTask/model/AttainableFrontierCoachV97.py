"""Attainable-value coach for routing over the frozen V9.3 expert pool."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_NAMES = (
    "anchor",
    "strong_negative",
    "boundary",
    "positive",
    "strong_positive",
)
SPECIALIST_NAMES = ACTION_NAMES[1:]


def stack_action_predictions(
    anchor: torch.Tensor,
    expert_predictions: torch.Tensor,
) -> torch.Tensor:
    """Return candidate predictions with shape [N, 5, 1]."""
    anchor = anchor.view(-1, 1, 1)
    if expert_predictions.dim() != 3 or expert_predictions.shape[1:] != (4, 1):
        raise ValueError("expert_predictions must have shape [N,4,1]")
    if expert_predictions.size(0) != anchor.size(0):
        raise ValueError("candidate sample count mismatch")
    return torch.cat((anchor, expert_predictions), dim=1)


def attainable_frontier_targets(
    actions: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Construct the best value actually present in the candidate pool."""
    labels = labels.view(-1, 1, 1).to(actions)
    costs = torch.abs(actions - labels).squeeze(-1)
    sorted_costs, sorted_indices = costs.sort(dim=1)
    oracle_index = sorted_indices[:, 0]
    oracle_value = actions.squeeze(-1).gather(
        1, oracle_index.view(-1, 1)
    )
    anchor_cost = costs[:, :1]
    oracle_cost = sorted_costs[:, :1]
    oracle_gain = anchor_cost - oracle_cost
    cost_margin = sorted_costs[:, 1:2] - sorted_costs[:, :1]
    return {
        "costs": costs,
        "oracle_index": oracle_index,
        "oracle_value": oracle_value,
        "oracle_gain": oracle_gain,
        "cost_margin": cost_margin,
    }


def frontier_input_features(
    function_space: torch.Tensor,
    actions: torch.Tensor,
    expert_confidences: torch.Tensor,
) -> torch.Tensor:
    """Build deployment-aligned features that explicitly include all candidates."""
    if function_space.dim() != 2 or function_space.size(1) != 4:
        raise ValueError("function_space must have shape [N,4]")
    if actions.dim() != 3 or actions.shape[1:] != (5, 1):
        raise ValueError("actions must have shape [N,5,1]")
    if expert_confidences.dim() != 3 or expert_confidences.shape[1:] != (4, 1):
        raise ValueError("expert_confidences must have shape [N,4,1]")
    values = actions.squeeze(-1).to(function_space)
    anchor = values[:, :1]
    corrections = values[:, 1:] - anchor
    confidences = expert_confidences.squeeze(-1).to(function_space)
    sorted_values = values.sort(dim=1).values
    sorted_gaps = sorted_values[:, 1:] - sorted_values[:, :-1]
    action_mean = values.mean(dim=1, keepdim=True)
    action_std = values.std(dim=1, keepdim=True, unbiased=False)
    action_range = values.amax(dim=1, keepdim=True) - values.amin(
        dim=1, keepdim=True
    )
    function_mean = function_space.mean(dim=1, keepdim=True)
    function_std = function_space.std(dim=1, keepdim=True, unbiased=False)
    function_range = function_space.amax(dim=1, keepdim=True) - function_space.amin(
        dim=1, keepdim=True
    )
    function_disagreement = torch.mean(
        torch.abs(function_space - anchor), dim=1, keepdim=True
    )
    return torch.cat(
        (
            function_space,
            anchor,
            anchor.abs(),
            anchor.square(),
            values,
            corrections,
            corrections.abs(),
            confidences,
            sorted_values,
            sorted_gaps,
            action_mean,
            action_std,
            action_range,
            function_mean,
            function_std,
            function_range,
            function_disagreement,
        ),
        dim=1,
    )


class AttainableFrontierCoachV97(nn.Module):
    """Predict the best attainable candidate value and its useful gain."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        gain_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.gain_max = float(gain_max)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.position_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.useful_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Linear(hidden_dim, len(ACTION_NAMES))
        nn.init.zeros_(self.position_head.weight)
        nn.init.zeros_(self.position_head.bias)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.constant_(self.gain_head.bias, -2.0)
        nn.init.zeros_(self.useful_head.weight)
        nn.init.constant_(self.useful_head.bias, -1.0)

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if features.dim() != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"features must have shape [N,{self.input_dim}], got {tuple(features.shape)}"
            )
        if actions.dim() != 3 or actions.shape[1:] != (5, 1):
            raise ValueError("actions must have shape [N,5,1]")
        latent = self.encoder(features)
        action_values = actions.squeeze(-1).to(features)
        low = action_values.amin(dim=1, keepdim=True)
        high = action_values.amax(dim=1, keepdim=True)
        position = torch.sigmoid(self.position_head(latent))
        frontier_value = low + position * (high - low)
        predicted_gain = self.gain_max * torch.sigmoid(self.gain_head(latent))
        useful_logit = self.useful_head(latent)
        return {
            "frontier_value": frontier_value,
            "predicted_gain": predicted_gain,
            "useful_logit": useful_logit,
            "useful_probability": torch.sigmoid(useful_logit),
            "action_logits": self.action_head(latent),
            "latent": latent,
        }


def nearest_action_from_frontier(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    values = actions.squeeze(-1).to(output["frontier_value"])
    distances = torch.abs(values - output["frontier_value"])
    sorted_distances, sorted_indices = distances.sort(dim=1)
    selected = sorted_indices[:, 0]
    nearest_margin = sorted_distances[:, 1] - sorted_distances[:, 0]
    selected_value = values.gather(1, selected.view(-1, 1))
    return {
        "distances": distances,
        "selected_index": selected,
        "selected_value": selected_value,
        "nearest_margin": nearest_margin.view(-1, 1),
    }


def attainable_frontier_loss(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    labels: torch.Tensor,
    useful_margin: float = 0.02,
    value_weight: float = 1.0,
    gain_weight: float = 0.45,
    useful_weight: float = 0.35,
    action_weight: float = 0.15,
    cost_weight: float = 0.25,
    distance_temperature: float = 0.05,
) -> Dict[str, torch.Tensor]:
    target = attainable_frontier_targets(actions, labels)
    gain_scale = (target["oracle_gain"] / 0.10).clamp(0.0, 1.0)
    margin_scale = (target["cost_margin"] / 0.05).clamp(0.0, 1.0)
    sample_weight = 0.25 + 0.75 * gain_scale * margin_scale

    value_per_sample = F.smooth_l1_loss(
        output["frontier_value"],
        target["oracle_value"],
        beta=0.10,
        reduction="none",
    )
    value = (value_per_sample * sample_weight).mean()
    gain = F.smooth_l1_loss(
        output["predicted_gain"],
        target["oracle_gain"],
        beta=0.05,
    )
    useful_target = (target["oracle_gain"] > float(useful_margin)).to(
        output["useful_logit"]
    )
    positive = float(useful_target.sum().item())
    negative = float(useful_target.numel() - positive)
    pos_weight = output["useful_logit"].new_tensor(
        min(8.0, max(0.5, negative / max(positive, 1.0)))
    )
    useful = F.binary_cross_entropy_with_logits(
        output["useful_logit"], useful_target, pos_weight=pos_weight
    )
    action_per_sample = F.cross_entropy(
        output["action_logits"], target["oracle_index"], reduction="none"
    ).view(-1, 1)
    action = (action_per_sample * sample_weight).mean()

    distances = torch.abs(
        actions.squeeze(-1).to(output["frontier_value"])
        - output["frontier_value"]
    )
    action_probs = torch.softmax(
        -distances / max(float(distance_temperature), 1e-5), dim=1
    )
    expected_cost = (
        action_probs * target["costs"].to(action_probs)
    ).sum(dim=1).mean()
    total = (
        float(value_weight) * value
        + float(gain_weight) * gain
        + float(useful_weight) * useful
        + float(action_weight) * action
        + float(cost_weight) * expected_cost
    )
    return {
        "total": total,
        "frontier_value_huber": value,
        "oracle_gain_huber": gain,
        "useful_bce": useful,
        "action_ce": action,
        "soft_expected_cost": expected_cost,
    }
