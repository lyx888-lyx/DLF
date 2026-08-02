"""Pre-registered nested router protocol for fully same-stack V9.19."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Dict, Mapping, Sequence

import torch

from .model.OrdinalRegionCostRouterV918 import region_probabilities_from_logits
from .model.SemanticCostCoachV99 import ACTION_NAMES
from .region_action_cost_policy_v918 import (
    action_cost_matrix,
    expected_action_costs,
)
from .region_probability_model_training_v918 import (
    RegionModelConfigV918,
    inner_split_indices,
    normalize_router_pool,
    predict_logits,
    region_metrics,
    select_temperature,
    train_fixed_epochs,
    train_region_model,
)

PROTOCOL_VERSION = "full_same_stack_nested_crossfit_v919_v1"


@dataclass(frozen=True)
class FixedPolicyV919:
    gain_margin: float = 0.03
    min_region_confidence: float = 0.55
    max_coverage: float = 0.25
    temperature_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)
    allowed_specialists: tuple[str, ...] = (
        "strong_negative",
        "boundary",
        "positive",
        "strong_positive",
    )


@dataclass(frozen=True)
class InnerGateV919:
    min_gain: float = 0.005
    max_harm_over_010_rate: float = 0.05
    min_positive_fold_fraction: float = 2.0 / 3.0
    min_trigger_precision: float = 0.55


def merge_pools(pools: Sequence[Mapping[str, object]], expected_indices: Sequence[int]):
    """Merge disjoint same-stack holdout pools into global-index order."""
    if not pools:
        raise ValueError("no same-stack pools to merge")
    by_index = {}
    for pool in pools:
        indices = [int(value) for value in pool["sample_indices"]]
        for local, index in enumerate(indices):
            if index in by_index:
                raise RuntimeError(f"duplicate merged sample index {index}")
            by_index[index] = (pool, local)
    ordered = sorted(int(value) for value in expected_indices)
    if set(by_index) != set(ordered):
        raise RuntimeError("merged pool index set does not match development set")

    def stack_field(name: str):
        return torch.stack(
            [by_index[index][0][name][by_index[index][1]] for index in ordered], dim=0
        )

    result = {
        "version": PROTOCOL_VERSION,
        "sample_indices": ordered,
        "sample_ids": [by_index[index][0]["sample_ids"][by_index[index][1]] for index in ordered],
        "group_ids": [by_index[index][0]["group_ids"][by_index[index][1]] for index in ordered],
        "labels": stack_field("labels"),
        "anchor": stack_field("anchor"),
        "function_space": stack_field("function_space"),
        "expert_predictions": stack_field("expert_predictions"),
        "expert_confidences": stack_field("expert_confidences"),
        "expert_corrections": stack_field("expert_corrections"),
        "action_names": pools[0]["action_names"],
        "feature_space": pools[0]["feature_space"],
        "fold_index": torch.tensor(
            [
                int(by_index[index][0].get("inner_fold", -1))
                for index in ordered
            ],
            dtype=torch.long,
        ),
        "provenance": {
            "all_rows_are_unseen_by_their_expert_stack": True,
            "same_stack_recipe_for_every_inner_fold": True,
        },
    }
    return result


def apply_fixed_policy(
    probabilities: torch.Tensor,
    expected_costs: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
    policy: FixedPolicyV919,
):
    action_to_index = {name: index for index, name in enumerate(ACTION_NAMES)}
    allowed = [action_to_index[name] for name in policy.allowed_specialists]
    specialist_costs = expected_costs[:, allowed]
    local_best_cost, local_offset = specialist_costs.min(dim=1)
    proposed_action = torch.tensor(allowed, dtype=torch.long).index_select(0, local_offset)
    predicted_gain = expected_costs[:, 0] - local_best_cost
    confidence = probabilities.max(dim=1).values
    eligible = (
        (predicted_gain >= float(policy.gain_margin))
        & (confidence >= float(policy.min_region_confidence))
    )
    selected_action = torch.zeros(len(actions), dtype=torch.long)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).view(-1)
    max_trigger = int(math.floor(float(policy.max_coverage) * len(actions)))
    if policy.max_coverage > 0 and max_trigger == 0 and len(actions) > 0:
        max_trigger = 1
    if len(eligible_indices) > max_trigger >= 0:
        order = torch.argsort(predicted_gain.index_select(0, eligible_indices), descending=True)
        eligible_indices = eligible_indices.index_select(0, order[:max_trigger])
    selected_action[eligible_indices] = proposed_action.index_select(0, eligible_indices)
    selected_prediction = actions.gather(1, selected_action.view(-1, 1)).view(-1, 1)
    labels = labels.view(-1, 1)
    anchor_prediction = actions[:, 0:1]
    anchor_error = torch.abs(anchor_prediction - labels)
    selected_error = torch.abs(selected_prediction - labels)
    gain = anchor_error - selected_error
    triggered = selected_action != 0
    return {
        "selected_action": selected_action,
        "selected_prediction": selected_prediction,
        "predicted_gain": predicted_gain,
        "region_confidence": confidence,
        "trigger": triggered,
        "anchor_mae": float(anchor_error.mean().item()),
        "mae": float(selected_error.mean().item()),
        "gain_vs_anchor": float(gain.mean().item()),
        "harm_over_010_rate": float((gain.view(-1) < -0.10).float().mean().item()),
        "large_gain_rate_010": float((gain.view(-1) > 0.10).float().mean().item()),
        "coverage": float(triggered.float().mean().item()),
        "trigger_precision": float((gain.view(-1)[triggered] > 0).float().mean().item()) if triggered.any() else 0.0,
        "mean_trigger_gain": float(gain.view(-1)[triggered].mean().item()) if triggered.any() else 0.0,
        "action_counts": {
            name: int((selected_action == index).sum().item())
            for index, name in enumerate(ACTION_NAMES)
        },
    }


def crossfit_router_diagnostic(
    merged_pool: Mapping[str, object],
    device,
    model_config: RegionModelConfigV918,
    policy: FixedPolicyV919,
    gate: InnerGateV919,
    seed: int,
):
    pool = normalize_router_pool(merged_pool, "fully_same_stack_inner_oof")
    fold_index = merged_pool["fold_index"].view(-1)
    unique_folds = sorted(int(value) for value in torch.unique(fold_index).tolist())
    logits_oof = torch.full((len(pool["labels"]), 4), float("nan"))
    probabilities_oof = torch.full((len(pool["labels"]), 5), float("nan"))
    expected_oof = torch.full((len(pool["labels"]), len(ACTION_NAMES)), float("nan"))
    fold_rows = []
    histories = []
    best_epochs = []
    temperatures = []
    for fold in unique_folds:
        holdout = torch.nonzero(fold_index == fold, as_tuple=False).view(-1)
        development = torch.nonzero(fold_index != fold, as_tuple=False).view(-1)
        train_idx, valid_idx = inner_split_indices(
            development, pool["group_ids"], seed + 1009 * (fold + 1)
        )
        trained = train_region_model(
            pool["router_features"],
            pool["labels"],
            train_idx,
            valid_idx,
            device,
            model_config,
            seed + 17011 * (fold + 1),
        )
        valid_logits = predict_logits(
            trained["model"],
            pool["router_features"].index_select(0, valid_idx),
            device,
            model_config.batch_size,
        )
        temperature = select_temperature(
            valid_logits,
            pool["region_index"].index_select(0, valid_idx),
            policy.temperature_grid,
        )
        final_model = train_fixed_epochs(
            pool["router_features"],
            pool["labels"],
            development,
            device,
            model_config,
            seed + 23003 * (fold + 1),
            trained["best_epoch"],
        )
        holdout_logits = predict_logits(
            final_model,
            pool["router_features"].index_select(0, holdout),
            device,
            model_config.batch_size,
        )
        matrix, _ = action_cost_matrix(
            pool["actions_2d"],
            pool["labels"],
            pool["region_index"],
            development,
        )
        probabilities = region_probabilities_from_logits(holdout_logits, temperature)
        expected = expected_action_costs(probabilities, matrix)
        fold_result = apply_fixed_policy(
            probabilities,
            expected,
            pool["actions_2d"].index_select(0, holdout),
            pool["labels"].index_select(0, holdout),
            policy,
        )
        logits_oof[holdout] = holdout_logits
        probabilities_oof[holdout] = probabilities
        expected_oof[holdout] = expected
        fold_rows.append(
            {
                "inner_router_fold": fold,
                "holdout_count": len(holdout),
                "best_epoch": int(trained["best_epoch"]),
                "temperature": float(temperature),
                **{key: value for key, value in fold_result.items() if key not in {
                    "selected_action", "selected_prediction", "predicted_gain",
                    "region_confidence", "trigger", "action_counts"
                }},
                **{f"count_{name}": fold_result["action_counts"][name] for name in ACTION_NAMES},
            }
        )
        histories.extend({"inner_router_fold": fold, **row} for row in trained["history"])
        best_epochs.append(int(trained["best_epoch"]))
        temperatures.append(float(temperature))
        del trained, final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if (
        not torch.isfinite(logits_oof).all()
        or not torch.isfinite(probabilities_oof).all()
        or not torch.isfinite(expected_oof).all()
    ):
        raise FloatingPointError("V9.19 router crossfit outputs incomplete")
    overall = apply_fixed_policy(
        probabilities_oof,
        expected_oof,
        pool["actions_2d"],
        pool["labels"],
        policy,
    )
    positive_fold_fraction = float(
        sum(float(row["gain_vs_anchor"]) > 0.0 for row in fold_rows)
        / max(1, len(fold_rows))
    )
    accepted = (
        overall["gain_vs_anchor"] >= float(gate.min_gain)
        and overall["harm_over_010_rate"] <= float(gate.max_harm_over_010_rate)
        and positive_fold_fraction >= float(gate.min_positive_fold_fraction)
        and overall["trigger_precision"] >= float(gate.min_trigger_precision)
        and overall["coverage"] > 0.0
    )
    return {
        "pool": pool,
        "logits": logits_oof,
        "probabilities": probabilities_oof,
        "expected_costs": expected_oof,
        "overall": overall,
        "fold_rows": fold_rows,
        "history_rows": histories,
        "positive_fold_fraction": positive_fold_fraction,
        "accepted": bool(accepted),
        "median_best_epoch": int(round(median(best_epochs))),
        "median_temperature": float(median(temperatures)),
        "policy": asdict(policy),
        "gate": asdict(gate),
    }


def train_calibrated_outer_router(
    merged_pool: Mapping[str, object],
    device,
    model_config: RegionModelConfigV918,
    policy: FixedPolicyV919,
    seed: int,
    epochs: int,
):
    """Train on disjoint groups and calibrate temperature on held-out groups."""
    pool = normalize_router_pool(merged_pool, "fully_same_stack_inner_oof")
    all_indices = torch.arange(len(pool["labels"]), dtype=torch.long)
    training_idx, calibration_idx = inner_split_indices(
        all_indices, pool["group_ids"], seed + 9191
    )
    model = train_fixed_epochs(
        pool["router_features"],
        pool["labels"],
        training_idx,
        device,
        model_config,
        seed + 31013,
        epochs,
    )
    calibration_logits = predict_logits(
        model,
        pool["router_features"].index_select(0, calibration_idx),
        device,
        model_config.batch_size,
    )
    temperature = select_temperature(
        calibration_logits,
        pool["region_index"].index_select(0, calibration_idx),
        policy.temperature_grid,
    )
    matrix, counts = action_cost_matrix(
        pool["actions_2d"],
        pool["labels"],
        pool["region_index"],
        torch.arange(len(pool["labels"]), dtype=torch.long),
    )
    return {
        "model": model,
        "temperature": float(temperature),
        "cost_matrix": matrix,
        "cost_region_counts": counts,
        "training_indices": training_idx,
        "calibration_indices": calibration_idx,
    }


def evaluate_outer_pool(
    trained_router: Mapping[str, object],
    outer_pool: Mapping[str, object],
    device,
    model_config: RegionModelConfigV918,
    policy: FixedPolicyV919,
    accepted: bool,
):
    pool = normalize_router_pool(outer_pool, "fully_same_stack_outer_holdout")
    logits = predict_logits(
        trained_router["model"], pool["router_features"], device, model_config.batch_size
    )
    probabilities = region_probabilities_from_logits(
        logits, trained_router["temperature"]
    )
    expected = expected_action_costs(probabilities, trained_router["cost_matrix"])
    if accepted:
        result = apply_fixed_policy(
            probabilities,
            expected,
            pool["actions_2d"],
            pool["labels"],
            policy,
        )
    else:
        labels = pool["labels"].view(-1, 1)
        anchor = pool["actions_2d"][:, 0:1]
        anchor_error = torch.abs(anchor - labels)
        selected_action = torch.zeros(len(labels), dtype=torch.long)
        result = {
            "selected_action": selected_action,
            "selected_prediction": anchor,
            "predicted_gain": expected[:, 0] - expected[:, 1:].min(dim=1).values,
            "region_confidence": probabilities.max(dim=1).values,
            "trigger": torch.zeros(len(labels), dtype=torch.bool),
            "anchor_mae": float(anchor_error.mean().item()),
            "mae": float(anchor_error.mean().item()),
            "gain_vs_anchor": 0.0,
            "harm_over_010_rate": 0.0,
            "large_gain_rate_010": 0.0,
            "coverage": 0.0,
            "trigger_precision": 0.0,
            "mean_trigger_gain": 0.0,
            "action_counts": {name: len(labels) if index == 0 else 0 for index, name in enumerate(ACTION_NAMES)},
        }
    return {
        "pool": pool,
        "logits": logits,
        "region_probabilities": probabilities,
        "expected_costs": expected,
        "region_metrics": region_metrics(probabilities, pool["region_index"]),
        **result,
    }


__all__ = [
    "PROTOCOL_VERSION",
    "FixedPolicyV919",
    "InnerGateV919",
    "merge_pools",
    "apply_fixed_policy",
    "crossfit_router_diagnostic",
    "train_calibrated_outer_router",
    "evaluate_outer_pool",
]
