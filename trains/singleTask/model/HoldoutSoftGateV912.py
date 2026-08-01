"""Low-capacity convex soft gate for the V9.12 single-holdout expert stack."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SemanticCostCoachV99 import ACTION_NAMES, SIGNATURE_DIM, SPECIALIST_NAMES

GATE_VERSION = "holdout_soft_gate_v1"


class HoldoutSoftGateV912(nn.Module):
    """Score Anchor and four specialists, then form a convex prediction mixture."""

    def __init__(
        self,
        context_dim: int,
        signature_dim: int = SIGNATURE_DIM,
        hidden_dim: int = 48,
        action_embedding_dim: int = 8,
        dropout: float = 0.10,
        temperature: float = 1.0,
        anchor_bias: float = 1.50,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.signature_dim = int(signature_dim)
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_embedding = nn.Embedding(
            len(SPECIALIST_NAMES), int(action_embedding_dim)
        )
        expert_input = (
            self.signature_dim + int(hidden_dim) + int(action_embedding_dim)
        )
        self.expert_encoder = nn.Sequential(
            nn.LayerNorm(expert_input),
            nn.Linear(expert_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
        )
        self.anchor_head = nn.Linear(hidden_dim, 1)
        self.expert_head = nn.Linear(max(16, hidden_dim // 2), 1)

        nn.init.zeros_(self.anchor_head.weight)
        nn.init.constant_(self.anchor_head.bias, float(anchor_bias))
        nn.init.zeros_(self.expert_head.weight)
        nn.init.zeros_(self.expert_head.bias)

    def forward(
        self,
        context: torch.Tensor,
        specialist_signatures: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if context.dim() != 2 or context.size(1) != self.context_dim:
            raise ValueError(
                f"context must have shape [N,{self.context_dim}], "
                f"got {tuple(context.shape)}"
            )
        expected = (len(SPECIALIST_NAMES), self.signature_dim)
        if (
            specialist_signatures.dim() != 3
            or specialist_signatures.shape[1:] != expected
        ):
            raise ValueError(
                "specialist_signatures must have shape "
                f"[N,{expected[0]},{expected[1]}]"
            )

        shared = self.context_encoder(context)
        n = len(context)
        action_index = torch.arange(
            len(SPECIALIST_NAMES), device=context.device
        )
        action_embed = self.action_embedding(action_index).unsqueeze(0).expand(
            n, -1, -1
        )
        repeated = shared.unsqueeze(1).expand(-1, len(SPECIALIST_NAMES), -1)
        latent = self.expert_encoder(
            torch.cat(
                (
                    specialist_signatures.to(context),
                    repeated,
                    action_embed,
                ),
                dim=2,
            )
        )
        anchor_logit = self.anchor_head(shared)
        expert_logits = self.expert_head(latent).squeeze(-1)
        logits = torch.cat((anchor_logit, expert_logits), dim=1)
        weights = torch.softmax(logits / self.temperature, dim=1)
        entropy = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        return {
            "logits": logits,
            "weights": weights,
            "anchor_weight": weights[:, :1],
            "specialist_mass": 1.0 - weights[:, :1],
            "entropy": entropy,
        }


def soft_gate_prediction(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if actions.dim() != 3 or actions.shape[1:] != (len(ACTION_NAMES), 1):
        raise ValueError("actions must have shape [N,5,1]")
    weights = output["weights"].to(actions)
    mixture = (weights * actions.squeeze(-1)).sum(dim=1, keepdim=True)
    anchor = actions[:, 0]
    prediction = anchor + float(beta) * (mixture - anchor)
    return {
        "prediction": prediction,
        "mixture": mixture,
        "weights": weights,
    }


def holdout_soft_gate_loss(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    labels: torch.Tensor,
    harm_margin: float = 0.05,
    expected_cost_weight: float = 0.10,
    harm_weight: float = 0.20,
    specialist_mass_weight: float = 0.005,
) -> Dict[str, torch.Tensor]:
    routed = soft_gate_prediction(output, actions, beta=1.0)
    prediction = routed["prediction"]
    labels = labels.view(-1, 1).to(prediction)
    anchor = actions[:, 0].to(prediction)

    smooth = F.smooth_l1_loss(
        prediction,
        labels,
        beta=0.05,
    )
    mae = torch.abs(prediction - labels).mean()
    action_costs = torch.abs(actions.squeeze(-1).to(prediction) - labels)
    expected_action_cost = (
        output["weights"].to(action_costs) * action_costs
    ).sum(dim=1).mean()

    routed_error = torch.abs(prediction - labels)
    anchor_error = torch.abs(anchor - labels)
    harm_excess = F.relu(
        routed_error - anchor_error - float(harm_margin)
    ).mean()
    specialist_mass = output["specialist_mass"].mean()

    total = (
        smooth
        + float(expected_cost_weight) * expected_action_cost
        + float(harm_weight) * harm_excess
        + float(specialist_mass_weight) * specialist_mass
    )
    return {
        "total": total,
        "smooth_l1": smooth,
        "mae": mae,
        "expected_action_cost": expected_action_cost,
        "harm_excess": harm_excess,
        "specialist_mass": specialist_mass,
        "mean_entropy": output["entropy"].mean(),
    }
