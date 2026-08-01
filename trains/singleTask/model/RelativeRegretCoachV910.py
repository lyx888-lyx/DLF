"""Semantic-signature relative-regret coach for V9.10."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SemanticCostCoachV99 import (
    ACTION_NAMES,
    SIGNATURE_DIM,
    SPECIALIST_NAMES,
)

REGRET_VERSION = "relative_regret_v1"


def relative_regret_targets(
    actions: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return specialist error deltas relative to Anchor.

    A negative delta means that the specialist is better than Anchor.
    """
    if actions.dim() != 3 or actions.shape[1:] != (len(ACTION_NAMES), 1):
        raise ValueError("actions must have shape [N,5,1]")
    values = actions.squeeze(-1)
    labels = labels.view(-1, 1).to(values)
    costs = torch.abs(values - labels)
    anchor_cost = costs[:, :1]
    expert_cost = costs[:, 1:]
    delta = expert_cost - anchor_cost
    full_delta = torch.cat((torch.zeros_like(anchor_cost), delta), dim=1)
    oracle_index = full_delta.argmin(dim=1)
    return {
        "costs": costs,
        "anchor_cost": anchor_cost,
        "expert_cost": expert_cost,
        "delta": delta,
        "beat_anchor": (delta < 0.0).to(delta),
        "full_delta": full_delta,
        "oracle_index": oracle_index,
    }


class RelativeRegretCoachV910(nn.Module):
    """Predict each specialist's regret relative to Anchor."""

    def __init__(
        self,
        context_dim: int,
        signature_dim: int = SIGNATURE_DIM,
        hidden_dim: int = 128,
        action_embedding_dim: int = 12,
        dropout: float = 0.15,
        minimum_scale: float = 0.01,
        regret_max: float = 1.75,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.signature_dim = int(signature_dim)
        self.minimum_scale = float(minimum_scale)
        self.regret_max = float(regret_max)
        self.action_embedding = nn.Embedding(
            len(SPECIALIST_NAMES), int(action_embedding_dim)
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        expert_input = self.signature_dim + hidden_dim + int(action_embedding_dim)
        self.expert_encoder = nn.Sequential(
            nn.LayerNorm(expert_input),
            nn.Linear(expert_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.delta_head = nn.Linear(hidden_dim // 2, 1)
        self.scale_head = nn.Linear(hidden_dim // 2, 1)
        self.beat_head = nn.Linear(hidden_dim // 2, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, -1.75)
        nn.init.zeros_(self.beat_head.weight)
        nn.init.zeros_(self.beat_head.bias)

    def forward(
        self,
        context: torch.Tensor,
        specialist_signatures: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if context.dim() != 2 or context.size(1) != self.context_dim:
            raise ValueError(
                f"context must have shape [N,{self.context_dim}]"
            )
        if specialist_signatures.dim() != 3 or specialist_signatures.shape[1:] != (
            len(SPECIALIST_NAMES),
            self.signature_dim,
        ):
            raise ValueError(
                "specialist_signatures must have shape "
                f"[N,{len(SPECIALIST_NAMES)},{self.signature_dim}]"
            )
        n = len(context)
        shared = self.context_encoder(context).unsqueeze(1).expand(-1, 4, -1)
        indices = torch.arange(4, device=context.device)
        action_embed = self.action_embedding(indices).unsqueeze(0).expand(n, -1, -1)
        latent = self.expert_encoder(
            torch.cat(
                (specialist_signatures.to(context), shared, action_embed),
                dim=2,
            )
        )
        predicted_delta = self.regret_max * torch.tanh(
            self.delta_head(latent).squeeze(-1)
        )
        predicted_scale = (
            F.softplus(self.scale_head(latent).squeeze(-1))
            + self.minimum_scale
        )
        beat_logit = self.beat_head(latent).squeeze(-1)
        return {
            "predicted_delta": predicted_delta,
            "predicted_scale": predicted_scale,
            "beat_logit": beat_logit,
            "beat_probability": torch.sigmoid(beat_logit),
            "latent": latent,
        }


def relative_regret_loss(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    labels: torch.Tensor,
    sign_margin: float = 0.02,
    rank_margin: float = 0.02,
    rank_temperature: float = 0.05,
    selection_temperature: float = 0.08,
    delta_weight: float = 1.0,
    nll_weight: float = 0.10,
    beat_weight: float = 0.45,
    sign_weight: float = 0.30,
    rank_weight: float = 0.40,
    action_weight: float = 0.20,
    expected_regret_weight: float = 0.30,
) -> Dict[str, torch.Tensor]:
    target = relative_regret_targets(actions, labels)
    delta = target["delta"]
    predicted = output["predicted_delta"]
    scale = output["predicted_scale"].clamp_min(1e-4)

    importance = 1.0 + (delta.abs() / 0.10).clamp(0.0, 3.0)
    delta_per = F.smooth_l1_loss(
        predicted,
        delta,
        beta=0.05,
        reduction="none",
    )
    delta_huber = (delta_per * importance).mean()

    absolute_residual = torch.abs(predicted - delta)
    laplace_nll = (absolute_residual / scale + scale.log()).mean()

    beat_target = target["beat_anchor"]
    positive = beat_target.sum(dim=0)
    negative = beat_target.size(0) - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(0.5, 8.0)
    beat_bce = F.binary_cross_entropy_with_logits(
        output["beat_logit"],
        beat_target,
        pos_weight=pos_weight,
    )

    meaningful = delta.abs() > float(sign_margin)
    desired_sign = delta.sign()
    sign_terms = F.softplus(
        (
            float(sign_margin)
            - desired_sign * predicted
        )
        / max(float(rank_temperature), 1e-5)
    )
    sign_consistency = (
        sign_terms[meaningful].mean()
        if meaningful.any()
        else predicted.new_zeros(())
    )

    target_full = target["full_delta"]
    predicted_full = torch.cat(
        (predicted.new_zeros((len(predicted), 1)), predicted),
        dim=1,
    )
    true_difference = target_full.unsqueeze(2) - target_full.unsqueeze(1)
    predicted_difference = predicted_full.unsqueeze(2) - predicted_full.unsqueeze(1)
    pair_mask = torch.abs(true_difference) > float(rank_margin)
    pair_mask &= torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    if pair_mask.any():
        desired = true_difference.sign()
        rank_terms = F.softplus(
            (
                float(rank_margin)
                - desired * predicted_difference
            )
            / max(float(rank_temperature), 1e-5)
        )
        pair_weight = (
            torch.abs(true_difference) / 0.15
        ).clamp(0.25, 4.0)
        pairwise_rank = (
            rank_terms[pair_mask] * pair_weight[pair_mask]
        ).mean()
    else:
        pairwise_rank = predicted.new_zeros(())

    oracle_index = target["oracle_index"]
    action_ce = F.cross_entropy(
        -predicted_full / max(float(selection_temperature), 1e-5),
        oracle_index,
    )
    selection_prob = torch.softmax(
        -predicted_full / max(float(selection_temperature), 1e-5),
        dim=1,
    )
    soft_expected_regret = (
        selection_prob * target_full
    ).sum(dim=1).mean()

    total = (
        float(delta_weight) * delta_huber
        + float(nll_weight) * laplace_nll
        + float(beat_weight) * beat_bce
        + float(sign_weight) * sign_consistency
        + float(rank_weight) * pairwise_rank
        + float(action_weight) * action_ce
        + float(expected_regret_weight) * soft_expected_regret
    )
    return {
        "total": total,
        "delta_huber": delta_huber,
        "laplace_nll": laplace_nll,
        "beat_bce": beat_bce,
        "sign_consistency": sign_consistency,
        "pairwise_rank": pairwise_rank,
        "action_ce": action_ce,
        "soft_expected_regret": soft_expected_regret,
    }


def select_action_from_regret(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    risk_aversion: float = 0.0,
    confidence_z: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Select among Anchor and four specialists with Anchor fixed at zero regret."""
    mean = output["predicted_delta"]
    scale = output["predicted_scale"]
    risk = mean + float(risk_aversion) * scale
    full_risk = torch.cat((risk.new_zeros((len(risk), 1)), risk), dim=1)
    sorted_risk, sorted_index = full_risk.sort(dim=1)
    selected_index = sorted_index[:, 0]
    specialist_index = (selected_index - 1).clamp_min(0)
    gather_index = specialist_index.view(-1, 1)
    selected_delta = mean.gather(1, gather_index)
    selected_scale = scale.gather(1, gather_index)
    selected_probability = output["beat_probability"].gather(1, gather_index)
    is_anchor = selected_index == 0
    selected_delta = torch.where(
        is_anchor.view(-1, 1),
        torch.zeros_like(selected_delta),
        selected_delta,
    )
    selected_scale = torch.where(
        is_anchor.view(-1, 1),
        torch.zeros_like(selected_scale),
        selected_scale,
    )
    selected_probability = torch.where(
        is_anchor.view(-1, 1),
        torch.ones_like(selected_probability),
        selected_probability,
    )
    selected_value = actions.squeeze(-1).gather(
        1, selected_index.view(-1, 1)
    )
    return {
        "decision_regret": full_risk,
        "selected_index": selected_index,
        "selected_value": selected_value,
        "selected_delta": selected_delta,
        "selected_scale": selected_scale,
        "beat_probability": selected_probability,
        "predicted_gain": -selected_delta,
        "lower_gain": -(
            selected_delta + float(confidence_z) * selected_scale
        ),
        "regret_margin": sorted_risk[:, 1:2] - sorted_risk[:, :1],
    }


def regret_soft_mixture(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    temperature: float = 0.08,
    risk_aversion: float = 0.0,
) -> Dict[str, torch.Tensor]:
    risk = output["predicted_delta"] + float(risk_aversion) * output[
        "predicted_scale"
    ]
    full_risk = torch.cat((risk.new_zeros((len(risk), 1)), risk), dim=1)
    weights = torch.softmax(
        -full_risk / max(float(temperature), 1e-5),
        dim=1,
    )
    prediction = (
        weights * actions.squeeze(-1).to(weights)
    ).sum(dim=1, keepdim=True)
    return {
        "prediction": prediction,
        "weights": weights,
        "decision_regret": full_risk,
    }
