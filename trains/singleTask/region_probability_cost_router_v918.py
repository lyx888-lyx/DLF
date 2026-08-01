"""Nested OOF integration and deployment helpers for V9.18."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Dict, Mapping

import torch

from .model.OrdinalRegionCostRouterV918 import (
    MODEL_VERSION,
    OrdinalRegionCostRouterV918,
    region_probabilities_from_logits,
)
from .region_probability_model_training_v918 import (
    ROUTER_VERSION,
    RegionModelConfigV918,
    PolicyGridV918,
    SafetyConfigV918,
    normalize_router_pool,
    inner_split_indices,
    train_region_model,
    train_fixed_epochs,
    predict_logits,
    select_temperature,
    region_metrics,
    group_bucket,
)
from .region_action_cost_policy_v918 import (
    action_cost_matrix,
    shrink_cost_matrix,
    expected_action_costs,
    apply_policy,
    policy_candidates,
    select_policy,
)


def nested_oof_region_probabilities(
    pool: Mapping[str, object],
    device,
    config: RegionModelConfigV918,
    grid: PolicyGridV918,
    seed: int,
) -> Dict[str, object]:
    features = pool["router_features"]
    labels = pool["labels"].view(-1, 1)
    actions = pool["actions_2d"]
    regions = pool["region_index"]
    fold_index = pool.get("fold_index")
    if fold_index is None:
        raise ValueError("strict OOF pool has no fold_index")
    unique_folds = sorted(int(value) for value in torch.unique(fold_index).tolist())
    n = len(labels)
    logits_oof = torch.full((n, 4), float("nan"))
    cost_oof = torch.full((n, 5, 5), float("nan"))
    fold_rows = []
    histories = []
    for fold in unique_folds:
        holdout = torch.nonzero(fold_index == fold, as_tuple=False).view(-1)
        development = torch.nonzero(fold_index != fold, as_tuple=False).view(-1)
        inner_train, inner_valid = inner_split_indices(
            development, pool["group_ids"], seed + 1009 * (fold + 1)
        )
        trained = train_region_model(
            features,
            labels,
            inner_train,
            inner_valid,
            device,
            config,
            seed + 17011 * (fold + 1),
        )
        valid_logits = predict_logits(
            trained["model"],
            features.index_select(0, inner_valid),
            device,
            config.batch_size,
        )
        valid_regions = regions.index_select(0, inner_valid)
        temperature = select_temperature(valid_logits, valid_regions, grid.temperatures)
        final_model = train_fixed_epochs(
            features,
            labels,
            development,
            device,
            config,
            seed + 23003 * (fold + 1),
            trained["best_epoch"],
        )
        holdout_logits = predict_logits(
            final_model,
            features.index_select(0, holdout),
            device,
            config.batch_size,
        )
        logits_oof[holdout] = holdout_logits
        matrix, counts = action_cost_matrix(actions, labels, regions, development)
        cost_oof[holdout] = matrix.view(1, 5, 5)
        fold_probs = region_probabilities_from_logits(holdout_logits, temperature)
        fold_metrics = region_metrics(fold_probs, regions.index_select(0, holdout))
        fold_rows.append({
            "outer_fold": fold,
            "development_count": len(development),
            "inner_train_count": len(inner_train),
            "inner_valid_count": len(inner_valid),
            "holdout_count": len(holdout),
            "best_epoch": int(trained["best_epoch"]),
            "temperature": float(temperature),
            "cost_region_counts": counts.tolist(),
            **fold_metrics,
        })
        histories.extend({"outer_fold": fold, **row} for row in trained["history"])
        del trained, final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not torch.isfinite(logits_oof).all() or not torch.isfinite(cost_oof).all():
        raise FloatingPointError("nested OOF router outputs incomplete")
    probabilities_by_temperature = {
        float(temp): region_probabilities_from_logits(logits_oof, temp)
        for temp in grid.temperatures
    }
    candidates = policy_candidates(
        probabilities_by_temperature, cost_oof, actions, labels, grid
    )
    return {
        "logits": logits_oof,
        "cost_matrices": cost_oof,
        "candidate_rows": candidates,
        "fold_rows": fold_rows,
        "history_rows": histories,
        "median_best_epoch": int(round(median([row["best_epoch"] for row in fold_rows]))),
        "median_temperature": float(median([row["temperature"] for row in fold_rows])),
    }


def save_model_checkpoint(
    path: Path,
    model,
    model_config: RegionModelConfigV918,
    epochs: int,
    temperature: float,
    feature_dim: int,
) -> None:
    torch.save({
        "version": ROUTER_VERSION,
        "model_version": MODEL_VERSION,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": asdict(model_config),
        "epochs": int(epochs),
        "temperature": float(temperature),
        "feature_dim": int(feature_dim),
    }, Path(path))


def load_model_checkpoint(path: Path, device):
    payload = torch.load(path, map_location="cpu")
    if payload.get("version") != ROUTER_VERSION or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("unexpected V9.18 router checkpoint version")
    state = payload["model_state_dict"]
    mean = state["standardizer.mean"].view(-1)
    scale = state["standardizer.scale"].view(-1)
    config = RegionModelConfigV918(**payload["model_config"])
    model = OrdinalRegionCostRouterV918(
        mean, scale, config.hidden_dim, config.dropout
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def policy_result_for_pool(
    model,
    pool: Mapping[str, object],
    device,
    batch_size: int,
    temperature: float,
    cost_matrix: torch.Tensor,
    gain_margin: float,
    min_region_confidence: float,
) -> Dict[str, object]:
    logits = predict_logits(model, pool["router_features"], device, batch_size)
    probabilities = region_probabilities_from_logits(logits, temperature)
    expected = expected_action_costs(probabilities, cost_matrix)
    result = apply_policy(
        probabilities,
        expected,
        pool["actions_2d"],
        pool["labels"],
        gain_margin,
        min_region_confidence,
    )
    return {
        "logits": logits,
        "region_probabilities": probabilities,
        "expected_costs": expected,
        **result,
    }


__all__ = [
    "ROUTER_VERSION", "RegionModelConfigV918", "PolicyGridV918", "SafetyConfigV918",
    "normalize_router_pool", "nested_oof_region_probabilities", "select_policy",
    "train_fixed_epochs", "action_cost_matrix", "shrink_cost_matrix",
    "expected_action_costs", "apply_policy", "policy_result_for_pool",
    "save_model_checkpoint", "load_model_checkpoint", "region_metrics", "group_bucket",
]
