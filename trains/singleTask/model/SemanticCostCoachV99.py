"""Semantic-signature per-action cost coach for V9.9."""

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
SIGNATURE_VERSION = "semantic_expert_signature_v1"
SIGNATURE_FIELDS = (
    "prediction",
    "correction",
    "abs_correction",
    "confidence",
    "raw_correction",
    "predicted_abs_error",
    "region_expected",
    "region_entropy",
    "mechanism_entropy",
    "role_available",
    "tail_available",
    "risk_available",
    "region_prob_strong_negative",
    "region_prob_negative",
    "region_prob_boundary",
    "region_prob_positive",
    "region_prob_strong_positive",
    "mechanism_prob_0",
    "mechanism_prob_1",
    "mechanism_prob_2",
    "mechanism_prob_3",
)
SIGNATURE_DIM = len(SIGNATURE_FIELDS)


def stack_action_predictions(
    anchor: torch.Tensor, expert_predictions: torch.Tensor
) -> torch.Tensor:
    anchor = anchor.view(-1, 1, 1)
    if expert_predictions.dim() != 3 or expert_predictions.shape[1:] != (4, 1):
        raise ValueError("expert_predictions must have shape [N,4,1]")
    if expert_predictions.size(0) != anchor.size(0):
        raise ValueError("candidate sample count mismatch")
    return torch.cat((anchor, expert_predictions), dim=1)


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1, keepdim=True)


def role_signature(
    prediction: torch.Tensor,
    correction: torch.Tensor,
    confidence: torch.Tensor,
    region_probs: torch.Tensor,
    region_expected: torch.Tensor,
    predicted_abs_error: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.view(-1, 1)
    correction = correction.view(-1, 1)
    confidence = confidence.view(-1, 1)
    region_expected = region_expected.view(-1, 1)
    predicted_abs_error = predicted_abs_error.view(-1, 1)
    if region_probs.dim() != 2 or region_probs.size(1) != 5:
        raise ValueError("region_probs must have shape [N,5]")
    zeros4 = prediction.new_zeros((len(prediction), 4))
    zeros1 = prediction.new_zeros((len(prediction), 1))
    ones1 = prediction.new_ones((len(prediction), 1))
    signature = torch.cat(
        (
            prediction,
            correction,
            correction.abs(),
            confidence,
            correction,
            predicted_abs_error,
            region_expected,
            entropy_from_probs(region_probs),
            zeros1,
            ones1,
            zeros1,
            ones1,
            region_probs,
            zeros4,
        ),
        dim=1,
    )
    if signature.shape[1] != SIGNATURE_DIM:
        raise RuntimeError("role signature dimension mismatch")
    return signature


def tail_signature(
    prediction: torch.Tensor,
    correction: torch.Tensor,
    raw_correction: torch.Tensor,
    confidence: torch.Tensor,
    mechanism_probs: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.view(-1, 1)
    correction = correction.view(-1, 1)
    raw_correction = raw_correction.view(-1, 1)
    confidence = confidence.view(-1, 1)
    if mechanism_probs.dim() != 2 or mechanism_probs.size(1) != 4:
        raise ValueError("mechanism_probs must have shape [N,4]")
    zeros1 = prediction.new_zeros((len(prediction), 1))
    zeros5 = prediction.new_zeros((len(prediction), 5))
    ones1 = prediction.new_ones((len(prediction), 1))
    signature = torch.cat(
        (
            prediction,
            correction,
            correction.abs(),
            confidence,
            raw_correction,
            zeros1,
            zeros1,
            zeros1,
            entropy_from_probs(mechanism_probs),
            zeros1,
            ones1,
            zeros1,
            zeros5,
            mechanism_probs,
        ),
        dim=1,
    )
    if signature.shape[1] != SIGNATURE_DIM:
        raise RuntimeError("tail signature dimension mismatch")
    return signature


def stack_action_signatures(
    anchor: torch.Tensor, expert_signatures: torch.Tensor
) -> torch.Tensor:
    if expert_signatures.dim() != 3 or expert_signatures.shape[1:] != (
        4,
        SIGNATURE_DIM,
    ):
        raise ValueError(
            f"expert_signatures must have shape [N,4,{SIGNATURE_DIM}]"
        )
    anchor = anchor.view(-1, 1)
    if len(anchor) != len(expert_signatures):
        raise ValueError("signature sample count mismatch")
    base = anchor.new_zeros((len(anchor), SIGNATURE_DIM))
    base[:, 0] = anchor.view(-1)
    base[:, 3] = 1.0
    return torch.cat((base.unsqueeze(1), expert_signatures), dim=1)


def global_context_features(
    function_space: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    if function_space.dim() != 2 or function_space.size(1) != 4:
        raise ValueError("function_space must have shape [N,4]")
    if actions.dim() != 3 or actions.shape[1:] != (5, 1):
        raise ValueError("actions must have shape [N,5,1]")
    values = actions.squeeze(-1).to(function_space)
    anchor = values[:, :1]
    corrections = values[:, 1:] - anchor
    sorted_values = values.sort(dim=1).values
    sorted_gaps = sorted_values[:, 1:] - sorted_values[:, :-1]
    action_stats = torch.cat(
        (
            values.mean(dim=1, keepdim=True),
            values.std(dim=1, keepdim=True, unbiased=False),
            values.amax(dim=1, keepdim=True)
            - values.amin(dim=1, keepdim=True),
        ),
        dim=1,
    )
    function_stats = torch.cat(
        (
            function_space.mean(dim=1, keepdim=True),
            function_space.std(dim=1, keepdim=True, unbiased=False),
            function_space.amax(dim=1, keepdim=True)
            - function_space.amin(dim=1, keepdim=True),
            torch.mean(
                torch.abs(function_space - anchor), dim=1, keepdim=True
            ),
        ),
        dim=1,
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
            sorted_values,
            sorted_gaps,
            action_stats,
            function_stats,
        ),
        dim=1,
    )


class SemanticCostCoachV99(nn.Module):
    """Predict an absolute-error distribution for every candidate action."""

    def __init__(
        self,
        context_dim: int,
        signature_dim: int = SIGNATURE_DIM,
        hidden_dim: int = 128,
        action_embedding_dim: int = 12,
        dropout: float = 0.15,
        minimum_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.signature_dim = int(signature_dim)
        self.minimum_scale = float(minimum_scale)
        self.action_embedding = nn.Embedding(
            len(ACTION_NAMES), int(action_embedding_dim)
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
        self.cost_head = nn.Linear(hidden_dim // 2, 1)
        self.scale_head = nn.Linear(hidden_dim // 2, 1)
        nn.init.zeros_(self.cost_head.weight)
        nn.init.constant_(self.cost_head.bias, -0.5)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, -1.5)

    def forward(
        self, context: torch.Tensor, signatures: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if context.dim() != 2 or context.size(1) != self.context_dim:
            raise ValueError(
                f"context must have shape [N,{self.context_dim}]"
            )
        if signatures.dim() != 3 or signatures.shape[1:] != (
            len(ACTION_NAMES),
            self.signature_dim,
        ):
            raise ValueError(
                "signatures must have shape "
                f"[N,{len(ACTION_NAMES)},{self.signature_dim}]"
            )
        n = len(context)
        shared = self.context_encoder(context).unsqueeze(1).expand(-1, 5, -1)
        indices = torch.arange(5, device=context.device)
        action_embed = self.action_embedding(indices).unsqueeze(0).expand(n, -1, -1)
        latent = self.expert_encoder(
            torch.cat((signatures.to(context), shared, action_embed), dim=2)
        )
        predicted_cost = F.softplus(self.cost_head(latent)).squeeze(-1)
        predicted_scale = (
            F.softplus(self.scale_head(latent)).squeeze(-1)
            + self.minimum_scale
        )
        return {
            "predicted_cost": predicted_cost,
            "predicted_scale": predicted_scale,
            "latent": latent,
        }


def actual_action_costs(
    actions: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return torch.abs(actions.squeeze(-1) - labels.view(-1, 1).to(actions))


def semantic_cost_loss(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    labels: torch.Tensor,
    rank_margin: float = 0.02,
    rank_temperature: float = 0.05,
    selection_temperature: float = 0.08,
    cost_weight: float = 1.0,
    nll_weight: float = 0.15,
    gain_weight: float = 0.40,
    rank_weight: float = 0.50,
    action_weight: float = 0.20,
    expected_cost_weight: float = 0.30,
) -> Dict[str, torch.Tensor]:
    target_cost = actual_action_costs(actions, labels)
    predicted = output["predicted_cost"]
    scale = output["predicted_scale"].clamp_min(1e-4)
    cost_huber = F.smooth_l1_loss(predicted, target_cost, beta=0.10)
    absolute_residual = torch.abs(predicted - target_cost)
    laplace_nll = (absolute_residual / scale + scale.log()).mean()

    target_gain = target_cost[:, :1] - target_cost
    predicted_gain = predicted[:, :1] - predicted
    gain_huber = F.smooth_l1_loss(predicted_gain, target_gain, beta=0.05)

    true_delta = target_cost.unsqueeze(1) - target_cost.unsqueeze(2)
    predicted_delta = predicted.unsqueeze(1) - predicted.unsqueeze(2)
    pair_mask = torch.abs(true_delta) > float(rank_margin)
    pair_mask &= torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    if pair_mask.any():
        signs = true_delta.sign()
        weighted_margin = (
            torch.abs(true_delta) / 0.25
        ).clamp(0.25, 4.0)
        ranking = F.softplus(
            (
                float(rank_margin)
                - signs * predicted_delta
            )
            / max(float(rank_temperature), 1e-5)
        )
        pairwise_rank = (
            ranking[pair_mask] * weighted_margin[pair_mask]
        ).mean()
    else:
        pairwise_rank = predicted.new_zeros(())

    oracle_index = target_cost.argmin(dim=1)
    action_ce = F.cross_entropy(
        -predicted / max(float(selection_temperature), 1e-5),
        oracle_index,
    )
    selection_probs = torch.softmax(
        -predicted / max(float(selection_temperature), 1e-5), dim=1
    )
    soft_expected_cost = (selection_probs * target_cost).sum(dim=1).mean()
    total = (
        float(cost_weight) * cost_huber
        + float(nll_weight) * laplace_nll
        + float(gain_weight) * gain_huber
        + float(rank_weight) * pairwise_rank
        + float(action_weight) * action_ce
        + float(expected_cost_weight) * soft_expected_cost
    )
    return {
        "total": total,
        "cost_huber": cost_huber,
        "laplace_nll": laplace_nll,
        "gain_huber": gain_huber,
        "pairwise_rank": pairwise_rank,
        "action_ce": action_ce,
        "soft_expected_cost": soft_expected_cost,
    }


def select_action_from_cost(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    risk_aversion: float = 0.0,
    confidence_z: float = 1.0,
) -> Dict[str, torch.Tensor]:
    mean = output["predicted_cost"]
    scale = output["predicted_scale"]
    decision_cost = mean + float(risk_aversion) * scale
    sorted_cost, sorted_index = decision_cost.sort(dim=1)
    selected = sorted_index[:, 0]
    selected_value = actions.squeeze(-1).gather(
        1, selected.view(-1, 1)
    )
    selected_mean = mean.gather(1, selected.view(-1, 1))
    selected_scale = scale.gather(1, selected.view(-1, 1))
    predicted_gain = mean[:, :1] - selected_mean
    lower_gain = (
        mean[:, :1]
        - float(confidence_z) * scale[:, :1]
        - selected_mean
        - float(confidence_z) * selected_scale
    )
    return {
        "decision_cost": decision_cost,
        "selected_index": selected,
        "selected_value": selected_value,
        "selected_cost": selected_mean,
        "selected_scale": selected_scale,
        "predicted_gain": predicted_gain,
        "lower_gain": lower_gain,
        "cost_margin": (sorted_cost[:, 1:2] - sorted_cost[:, :1]),
    }


def cost_soft_mixture(
    output: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    temperature: float = 0.08,
    risk_aversion: float = 0.0,
) -> Dict[str, torch.Tensor]:
    decision = output["predicted_cost"] + float(risk_aversion) * output[
        "predicted_scale"
    ]
    weights = torch.softmax(
        -decision / max(float(temperature), 1e-5), dim=1
    )
    prediction = (
        weights * actions.squeeze(-1).to(weights)
    ).sum(dim=1, keepdim=True)
    return {"prediction": prediction, "weights": weights}
