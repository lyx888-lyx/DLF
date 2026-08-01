"""Action-cost matrices and safe expert policy for V9.18."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch

from .model.SemanticCostCoachV99 import ACTION_NAMES
from .role_conditioned_experts_v9 import REGION_NAMES
from .region_probability_model_training_v918 import PolicyGridV918, SafetyConfigV918


def action_cost_matrix(
    actions: torch.Tensor,
    labels: torch.Tensor,
    region_targets: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    errors = torch.abs(actions.index_select(0, indices) - labels.index_select(0, indices).view(-1, 1))
    local_regions = region_targets.index_select(0, indices)
    global_cost = errors.mean(dim=0)
    matrix = []
    counts = []
    for region in range(len(REGION_NAMES)):
        mask = local_regions == region
        count = int(mask.sum().item())
        counts.append(count)
        matrix.append(errors[mask].mean(dim=0) if count else global_cost)
    return torch.stack(matrix, dim=0), torch.tensor(counts, dtype=torch.long)


def shrink_cost_matrix(
    prior_matrix: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
    region_targets: torch.Tensor,
    indices: torch.Tensor,
    prior_strength: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    empirical, counts = action_cost_matrix(actions, labels, region_targets, indices)
    weights = counts.float().view(-1, 1)
    matrix = (
        weights * empirical + float(prior_strength) * prior_matrix
    ) / (weights + float(prior_strength))
    return matrix, counts


def expected_action_costs(region_probabilities: torch.Tensor, cost_matrix: torch.Tensor) -> torch.Tensor:
    if region_probabilities.shape[1] != cost_matrix.shape[0]:
        raise ValueError("region/cost matrix dimension mismatch")
    return region_probabilities @ cost_matrix


def apply_policy(
    region_probabilities: torch.Tensor,
    expected_costs: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
    gain_margin: float,
    min_region_confidence: float,
) -> Dict[str, object]:
    anchor_cost = expected_costs[:, 0]
    specialist_cost, specialist_offset = expected_costs[:, 1:].min(dim=1)
    proposed_action = specialist_offset + 1
    predicted_gain = anchor_cost - specialist_cost
    region_confidence = region_probabilities.max(dim=1).values
    trigger = (predicted_gain >= float(gain_margin)) & (region_confidence >= float(min_region_confidence))
    selected_action = torch.where(trigger, proposed_action, torch.zeros_like(proposed_action))
    selected_prediction = actions.gather(1, selected_action.view(-1, 1)).view(-1, 1)
    labels = labels.view(-1, 1)
    anchor_prediction = actions[:, 0:1]
    anchor_error = torch.abs(anchor_prediction - labels)
    selected_error = torch.abs(selected_prediction - labels)
    gain = anchor_error - selected_error
    return {
        "selected_action": selected_action,
        "selected_prediction": selected_prediction,
        "predicted_gain": predicted_gain,
        "region_confidence": region_confidence,
        "trigger": trigger,
        "anchor_mae": float(anchor_error.mean().item()),
        "mae": float(selected_error.mean().item()),
        "gain_vs_anchor": float(gain.mean().item()),
        "harm_over_010_rate": float((gain.view(-1) < -0.10).float().mean().item()),
        "large_gain_rate_010": float((gain.view(-1) > 0.10).float().mean().item()),
        "coverage": float(trigger.float().mean().item()),
        "trigger_precision": float((gain.view(-1)[trigger] > 0).float().mean().item()) if trigger.any() else 0.0,
        "mean_trigger_gain": float(gain.view(-1)[trigger].mean().item()) if trigger.any() else 0.0,
        "action_counts": {name: int((selected_action == index).sum().item()) for index, name in enumerate(ACTION_NAMES)},
    }


def policy_candidates(
    region_probabilities_by_temperature: Mapping[float, torch.Tensor],
    cost_matrices: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
    grid: PolicyGridV918,
) -> Sequence[Dict[str, object]]:
    rows = []
    for temperature in grid.temperatures:
        probabilities = region_probabilities_by_temperature[float(temperature)]
        expected = torch.einsum("nr,nra->na", probabilities, cost_matrices)
        for margin in grid.gain_margins:
            for confidence in grid.min_region_confidences:
                result = apply_policy(probabilities, expected, actions, labels, margin, confidence)
                rows.append({
                    "temperature": float(temperature),
                    "gain_margin": float(margin),
                    "min_region_confidence": float(confidence),
                    **{key: value for key, value in result.items() if key not in {"selected_action", "selected_prediction", "predicted_gain", "region_confidence", "trigger", "action_counts"}},
                    **{f"count_{name}": result["action_counts"][name] for name in ACTION_NAMES},
                })
    return rows


def select_policy(rows: Sequence[Mapping[str, object]], safety: SafetyConfigV918) -> Dict[str, object]:
    safe = [
        dict(row) for row in rows
        if float(row["gain_vs_anchor"]) >= float(safety.min_oof_gain)
        and float(row["harm_over_010_rate"]) <= float(safety.max_oof_harm_over_010_rate)
        and float(row["coverage"]) > 0.0
    ]
    if not safe:
        return {
            "temperature": 1.0,
            "gain_margin": 1e9,
            "min_region_confidence": 1.0,
            "fallback": True,
            "reason": "no_oof_candidate_passed_safety",
        }
    selected = min(
        safe,
        key=lambda row: (
            float(row["mae"]),
            float(row["harm_over_010_rate"]),
            float(row["coverage"]),
            -float(row["gain_margin"]),
            -float(row["min_region_confidence"]),
        ),
    )
    selected["fallback"] = False
    selected["reason"] = "strict_oof_selected"
    return selected


__all__ = [
    "action_cost_matrix", "shrink_cost_matrix", "expected_action_costs",
    "apply_policy", "policy_candidates", "select_policy",
]
