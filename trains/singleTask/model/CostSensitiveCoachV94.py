"""Models and objectives for the V9.4 cost-sensitive OOF coach."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIALIST_NAMES = (
    "strong_negative",
    "boundary",
    "positive",
    "strong_positive",
)
ACTION_NAMES = ("anchor",) + SPECIALIST_NAMES
SPECIALIST_TO_REGION = {
    "strong_negative": 0,
    "boundary": 2,
    "positive": 3,
    "strong_positive": 4,
}
REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)
REGION_BOUNDARIES = (-1.5, -0.5, 0.5, 1.5)


def region_index(labels: torch.Tensor) -> torch.Tensor:
    values = labels.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def exact_specialist_mask(labels: torch.Tensor, role: str) -> torch.Tensor:
    values = labels.view(-1)
    if role == "strong_negative":
        return values < -1.5
    if role == "boundary":
        return values.abs() <= 0.5
    if role == "positive":
        return (values > 0.5) & (values <= 1.5)
    if role == "strong_positive":
        return values > 1.5
    raise ValueError(f"unknown specialist role: {role}")


def smooth_specialist_membership(
    labels: torch.Tensor,
    role: str,
    temperature: float = 0.25,
) -> torch.Tensor:
    values = labels.view(-1)
    tau = max(float(temperature), 1e-4)
    if role == "strong_negative":
        return torch.sigmoid((-1.5 - values) / tau)
    if role == "boundary":
        return torch.sigmoid((0.5 - values.abs()) / tau)
    if role == "positive":
        lower = torch.sigmoid((values - 0.5) / tau)
        upper = torch.sigmoid((1.5 - values) / tau)
        return (4.0 * lower * upper).clamp(max=1.0)
    if role == "strong_positive":
        return torch.sigmoid((values - 1.5) / tau)
    raise ValueError(f"unknown specialist role: {role}")


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values = values.view(-1)
    weights = weights.view(-1).to(values)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class CachedSpecialistResidualHeadV94(nn.Module):
    """Low-capacity specialist over an immutable OOF anchor and task-space logits."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        residual_max: float = 1.25,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.residual_max = float(residual_max)
        input_dim = self.feature_dim + 1
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.applicability_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.applicability_head.weight)
        nn.init.zeros_(self.applicability_head.bias)

    def forward(self, feature: torch.Tensor, anchor: torch.Tensor):
        if feature.dim() != 2 or feature.size(1) != self.feature_dim:
            raise ValueError(
                f"feature must have shape [N,{self.feature_dim}], got {tuple(feature.shape)}"
            )
        anchor = anchor.view(-1, 1).to(feature)
        latent = self.adapter(torch.cat([feature, anchor], dim=1))
        raw_correction = self.residual_max * torch.tanh(self.residual_head(latent))
        applicability_logit = self.applicability_head(latent)
        applicability_prob = torch.sigmoid(applicability_logit)
        correction = applicability_prob.detach() * raw_correction
        return {
            "anchor": anchor,
            "prediction": anchor + correction,
            "raw_correction": raw_correction,
            "correction": correction,
            "applicability_logit": applicability_logit,
            "applicability_prob": applicability_prob,
        }


def specialist_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    role: str,
    membership_temperature: float = 0.25,
    gain_margin: float = 0.04,
):
    labels = labels.view(-1, 1).to(output["prediction"])
    anchor = output["anchor"]
    prediction = output["prediction"]
    membership = smooth_specialist_membership(
        labels, role, membership_temperature
    ).to(prediction)
    exact = exact_specialist_mask(labels, role).float().to(prediction)
    outside = 1.0 - exact
    anchor_error = torch.abs(anchor - labels)
    prediction_error = torch.abs(prediction - labels)
    target_residual = labels - anchor

    global_mae = prediction_error.mean()
    role_mae = _weighted_mean(prediction_error, membership)
    residual_huber = _weighted_mean(
        F.smooth_l1_loss(
            output["raw_correction"], target_residual, reduction="none", beta=0.20
        ),
        membership,
    )
    realized_gain = anchor_error - prediction_error
    gain_penalty = _weighted_mean(
        F.relu(float(gain_margin) - realized_gain), membership
    )
    outside_retention = _weighted_mean(
        torch.abs(prediction - anchor), outside
    )
    positives = exact.sum().clamp_min(1.0)
    negatives = outside.sum().clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(1.0, 8.0)
    gate_bce = F.binary_cross_entropy_with_logits(
        output["applicability_logit"].view(-1),
        exact.view(-1),
        pos_weight=pos_weight,
    )
    correction_shrink = output["raw_correction"].square().mean()
    total = (
        0.45 * global_mae
        + 1.00 * role_mae
        + 0.70 * residual_huber
        + 0.15 * gain_penalty
        + 0.45 * outside_retention
        + 0.15 * gate_bce
        + 0.02 * correction_shrink
    )
    return {
        "total": total,
        "global_mae": global_mae,
        "role_mae": role_mae,
        "residual_huber": residual_huber,
        "gain_penalty": gain_penalty,
        "outside_retention": outside_retention,
        "applicability_bce": gate_bce,
        "correction_shrink": correction_shrink,
    }


def coach_input_features(
    function_space: torch.Tensor,
    anchor: torch.Tensor,
    expert_predictions: torch.Tensor,
    expert_confidences: torch.Tensor,
) -> torch.Tensor:
    """Construct label-free features shared by OOF and deployable coach runs."""
    if function_space.dim() != 2:
        raise ValueError("function_space must be rank two")
    anchor = anchor.view(-1, 1).to(function_space)
    if expert_predictions.shape != (anchor.size(0), len(SPECIALIST_NAMES), 1):
        raise ValueError("expert_predictions must have shape [N,4,1]")
    if expert_confidences.shape != expert_predictions.shape:
        raise ValueError("expert_confidences shape mismatch")
    predictions = expert_predictions.squeeze(-1).to(function_space)
    confidences = expert_confidences.squeeze(-1).to(function_space)
    corrections = predictions - anchor
    branch_mean = function_space.mean(dim=1, keepdim=True)
    branch_std = function_space.std(dim=1, keepdim=True, unbiased=False)
    branch_range = (
        function_space.max(dim=1, keepdim=True).values
        - function_space.min(dim=1, keepdim=True).values
    )
    features = torch.cat(
        [
            function_space,
            anchor,
            anchor.abs(),
            branch_mean,
            branch_std,
            branch_range,
            predictions,
            corrections,
            confidences,
        ],
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise FloatingPointError("coach features contain non-finite values")
    return features


class CostSensitiveCoachV94(nn.Module):
    """Predict five action scores while retaining ordinal strength as an auxiliary task."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        ordinal_residual_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.ordinal_residual_max = float(ordinal_residual_max)
        self.encoder = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.action_head = nn.Linear(hidden_dim, len(ACTION_NAMES))
        self.ordinal_residual_head = nn.Linear(hidden_dim, 1)
        self.log_temperature = nn.Parameter(torch.tensor(-0.2))
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        with torch.no_grad():
            self.action_head.bias[0] = 0.5
        nn.init.zeros_(self.ordinal_residual_head.weight)
        nn.init.zeros_(self.ordinal_residual_head.bias)

    def forward(self, features: torch.Tensor, anchor: torch.Tensor):
        anchor = anchor.view(-1, 1).to(features)
        latent = self.encoder(features)
        action_logits = self.action_head(latent)
        action_probs = torch.softmax(action_logits, dim=1)
        ordinal_score = anchor + self.ordinal_residual_max * torch.tanh(
            self.ordinal_residual_head(latent)
        )
        temperature = F.softplus(self.log_temperature) + 0.05
        boundaries = ordinal_score.new_tensor(REGION_BOUNDARIES).view(1, -1)
        cumulative = torch.sigmoid((ordinal_score - boundaries) / temperature)
        region_probs = torch.cat(
            [
                1.0 - cumulative[:, 0:1],
                cumulative[:, 0:1] - cumulative[:, 1:2],
                cumulative[:, 1:2] - cumulative[:, 2:3],
                cumulative[:, 2:3] - cumulative[:, 3:4],
                cumulative[:, 3:4],
            ],
            dim=1,
        ).clamp_min(0.0)
        region_probs = region_probs / region_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "ordinal_score": ordinal_score,
            "cumulative_probs": cumulative,
            "region_probs": region_probs,
            "ordinal_temperature": temperature,
        }


def action_costs(action_predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if action_predictions.dim() != 3 or action_predictions.size(2) != 1:
        raise ValueError("action_predictions must have shape [N,5,1]")
    labels = labels.view(-1, 1, 1).to(action_predictions)
    return torch.abs(action_predictions - labels).squeeze(-1)


def cost_sensitive_coach_loss(
    output: Dict[str, torch.Tensor],
    action_predictions: torch.Tensor,
    labels: torch.Tensor,
    cost_temperature: float = 0.10,
    abstain_margin: float = 0.02,
):
    labels = labels.view(-1, 1).to(output["action_logits"])
    costs = action_costs(action_predictions, labels)
    probabilities = output["action_probs"]
    expected_cost = (probabilities * costs).sum(dim=1).mean()

    soft_target = torch.softmax(-costs / max(float(cost_temperature), 1e-4), dim=1)
    cost_distill = F.kl_div(
        F.log_softmax(output["action_logits"], dim=1),
        soft_target,
        reduction="batchmean",
    )
    best_action = costs.argmin(dim=1)
    regret_scale = (costs.max(dim=1).values - costs.min(dim=1).values).detach()
    hard_ce = (
        F.cross_entropy(output["action_logits"], best_action, reduction="none")
        * (0.25 + regret_scale)
    ).mean()

    specialist_best = costs[:, 1:].min(dim=1).values
    specialist_gain = costs[:, 0] - specialist_best
    no_clear_gain = specialist_gain <= float(abstain_margin)
    abstain_penalty = probabilities[no_clear_gain, 1:].sum(dim=1).mean() if no_clear_gain.any() else probabilities.sum() * 0.0

    target_region = region_index(labels)
    target_cumulative = torch.stack(
        [target_region >= boundary for boundary in (1, 2, 3, 4)], dim=1
    ).float().to(labels)
    ordinal_bce = F.binary_cross_entropy(
        output["cumulative_probs"].clamp(1e-6, 1.0 - 1e-6),
        target_cumulative,
    )
    ordinal_l1 = F.smooth_l1_loss(output["ordinal_score"], labels, beta=0.25)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=1).mean()

    total = (
        1.00 * expected_cost
        + 0.40 * cost_distill
        + 0.20 * hard_ce
        + 0.25 * abstain_penalty
        + 0.12 * ordinal_bce
        + 0.08 * ordinal_l1
        + 0.01 * entropy
    )
    return {
        "total": total,
        "expected_cost": expected_cost,
        "cost_distill": cost_distill,
        "hard_ce": hard_ce,
        "abstain_penalty": abstain_penalty,
        "ordinal_bce": ordinal_bce,
        "ordinal_l1": ordinal_l1,
        "entropy": entropy,
    }


def stack_action_predictions(
    anchor: torch.Tensor,
    expert_predictions: torch.Tensor,
) -> torch.Tensor:
    anchor = anchor.view(-1, 1, 1).to(expert_predictions)
    if expert_predictions.shape != (anchor.size(0), len(SPECIALIST_NAMES), 1):
        raise ValueError("expert prediction shape mismatch")
    return torch.cat([anchor, expert_predictions], dim=1)


def infer_hidden_dim(state: Dict[str, torch.Tensor], key: str = "adapter.1.weight") -> int:
    if key not in state or state[key].dim() != 2:
        raise ValueError(f"cannot infer hidden dimension from {key}")
    return int(state[key].size(0))


def ensure_action_names(names: Sequence[str]) -> None:
    if tuple(names) != ACTION_NAMES:
        raise ValueError(f"action schema mismatch: {tuple(names)}")
