"""Strict-OOF ordinal region probabilities and action-cost routing for V9.18."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .fixed_expert_region_audit_v917 import normalize_expert_pool
from .model.OrdinalRegionCostRouterV918 import (
    OrdinalRegionCostRouterV918,
    ordinal_region_loss,
    region_probabilities_from_logits,
)
from .model.SemanticCostCoachV99 import ACTION_NAMES, SPECIALIST_NAMES
from .role_conditioned_experts_v9 import region_index

ROUTER_VERSION = "region_probability_action_cost_router_v918_v1"


@dataclass(frozen=True)
class RegionModelConfigV918:
    hidden_dim: int = 64
    dropout: float = 0.10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    early_stop: int = 10
    gradient_clip: float = 2.0
    score_penalty: float = 1e-4


@dataclass(frozen=True)
class PolicyGridV918:
    gain_margins: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
    min_region_confidences: Tuple[float, ...] = (0.0, 0.35, 0.45, 0.55, 0.65, 0.75)
    temperatures: Tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)


@dataclass(frozen=True)
class SafetyConfigV918:
    min_oof_gain: float = 0.001
    max_oof_harm_over_010_rate: float = 0.10
    min_validation_gain: float = 0.0
    max_validation_harm_over_010_rate: float = 0.10
    cost_prior_strength: float = 30.0


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _column(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.detach().cpu().float()
    if value.dim() == 1:
        value = value.view(-1, 1)
    if value.dim() != 2 or value.size(1) != 1:
        raise ValueError(f"{name} must have shape [N,1]")
    return value


def _matrix(value: torch.Tensor, name: str, columns: int) -> torch.Tensor:
    value = value.detach().cpu().float()
    if value.dim() == 3 and value.size(-1) == 1:
        value = value.squeeze(-1)
    if value.dim() != 2 or value.size(1) != int(columns):
        raise ValueError(f"{name} must have shape [N,{columns}]")
    return value


def normalize_router_pool(pool: Mapping[str, object], source_protocol: str) -> Dict[str, object]:
    """Create aligned router features from strict-OOF or frozen deployment pools."""
    normalized = normalize_expert_pool(pool, source_protocol)
    actions = normalized["actions"].detach().cpu().float().squeeze(-1)
    if actions.shape[1] != len(ACTION_NAMES):
        raise ValueError("unexpected action count")
    function_space = pool.get("function_space")
    if function_space is None:
        raise ValueError("router pool has no function_space")
    function_space = function_space.detach().cpu().float()
    if function_space.dim() != 2 or len(function_space) != len(actions):
        raise ValueError("invalid function_space shape")

    if "expert_confidences" in pool:
        confidences = _matrix(pool["expert_confidences"], "expert_confidences", 4)
        corrections = _matrix(pool["expert_corrections"], "expert_corrections", 4)
    elif "experts" in pool:
        confidences = torch.cat(
            [_column(pool["experts"][name]["confidence"], f"{name}_confidence") for name in SPECIALIST_NAMES],
            dim=1,
        )
        corrections = torch.cat(
            [_column(pool["experts"][name]["correction"], f"{name}_correction") for name in SPECIALIST_NAMES],
            dim=1,
        )
    else:
        raise ValueError("router pool has no expert confidence/correction data")

    anchor = actions[:, 0:1]
    specialists = actions[:, 1:]
    consensus = torch.cat(
        [
            actions.mean(dim=1, keepdim=True),
            actions.std(dim=1, unbiased=False, keepdim=True),
            actions.amin(dim=1, keepdim=True),
            actions.amax(dim=1, keepdim=True),
            actions.median(dim=1, keepdim=True).values,
            specialists.mean(dim=1, keepdim=True) - anchor,
            specialists.std(dim=1, unbiased=False, keepdim=True),
        ],
        dim=1,
    )
    features = torch.cat([function_space, actions, corrections, confidences, consensus], dim=1)
    if not torch.isfinite(features).all():
        raise FloatingPointError("router features contain non-finite values")
    return {
        **normalized,
        "actions_2d": actions,
        "function_space": function_space,
        "expert_confidences": confidences,
        "expert_corrections": corrections,
        "router_features": features,
        "region_index": region_index(normalized["labels"]).long(),
    }


def fit_feature_statistics(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-4)
    return mean, scale


def positive_weights(labels: torch.Tensor) -> torch.Tensor:
    thresholds = labels.new_tensor((-1.5, -0.5, 0.5, 1.5)).view(1, -1)
    targets = (labels.view(-1, 1) > thresholds).float()
    positives = targets.sum(dim=0).clamp_min(1.0)
    negatives = (len(targets) - targets.sum(dim=0)).clamp_min(1.0)
    return (negatives / positives).clamp(0.25, 4.0)


def group_bucket(group_id: object, seed: int, buckets: int) -> int:
    digest = hashlib.sha1(f"{seed}|{group_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % int(buckets)


def inner_split_indices(
    development_indices: torch.Tensor,
    group_ids: Sequence[object],
    seed: int,
    buckets: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dev = development_indices.view(-1).tolist()
    valid = [idx for idx in dev if group_bucket(group_ids[idx], seed, buckets) == 0]
    valid_set = set(valid)
    train = [idx for idx in dev if idx not in valid_set]
    if len(valid) < 20 or len(train) < 100:
        unique = sorted(set(str(group_ids[idx]) for idx in dev))
        valid_groups = set(unique[::max(2, len(unique) // 5)])
        valid = [idx for idx in dev if str(group_ids[idx]) in valid_groups]
        train = [idx for idx in dev if str(group_ids[idx]) not in valid_groups]
    if not valid or not train:
        raise RuntimeError("failed to build inner group split")
    return torch.tensor(train, dtype=torch.long), torch.tensor(valid, dtype=torch.long)


def make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(features.index_select(0, indices), labels.index_select(0, indices))
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), generator=generator)


def region_nll(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    return float(
        -torch.log(probabilities.clamp_min(1e-8))
        .gather(1, targets.view(-1, 1))
        .mean()
        .item()
    )


def region_metrics(probabilities: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    predicted = probabilities.argmax(dim=1)
    return {
        "region_nll": region_nll(probabilities, targets),
        "region_accuracy": float((predicted == targets).float().mean().item()),
        "region_adjacent_accuracy": float(((predicted - targets).abs() <= 1).float().mean().item()),
        "region_mae": float((predicted.float() - targets.float()).abs().mean().item()),
        "mean_max_probability": float(probabilities.max(dim=1).values.mean().item()),
    }


def train_region_model(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    valid_indices: torch.Tensor,
    device,
    config: RegionModelConfigV918,
    seed: int,
) -> Dict[str, object]:
    seed_all(seed)
    mean, scale = fit_feature_statistics(features.index_select(0, train_indices))
    model = OrdinalRegionCostRouterV918(
        mean, scale, hidden_dim=config.hidden_dim, dropout=config.dropout
    ).to(device)
    pos_weight = positive_weights(labels.index_select(0, train_indices)).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    loader = make_loader(features, labels, train_indices, config.batch_size, True, seed)
    valid_features = features.index_select(0, valid_indices).to(device)
    valid_regions = region_index(labels.index_select(0, valid_indices)).long()
    best_state = None
    best_epoch = 0
    best_score = float("inf")
    stale = 0
    history = []
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        totals: Dict[str, float] = {}
        for batch_features, batch_labels in loader:
            output = model(batch_features.to(device))
            losses = ordinal_region_loss(
                output,
                batch_labels.to(device),
                pos_weight,
                score_penalty=config.score_penalty,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        model.eval()
        with torch.no_grad():
            logits = model(valid_features)["logits"].detach().cpu()
            probs = region_probabilities_from_logits(logits, 1.0)
        metrics = region_metrics(probs, valid_regions)
        score = metrics["region_nll"] + 0.05 * metrics["region_mae"]
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value / max(1, len(loader)) for key, value in totals.items()},
            **metrics,
        })
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config.early_stop):
                break
    if best_state is None:
        raise RuntimeError("region model produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    return {"model": model, "best_epoch": best_epoch, "history": history}


def train_fixed_epochs(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    device,
    config: RegionModelConfigV918,
    seed: int,
    epochs: int,
):
    seed_all(seed)
    mean, scale = fit_feature_statistics(features.index_select(0, indices))
    model = OrdinalRegionCostRouterV918(mean, scale, config.hidden_dim, config.dropout).to(device)
    pos_weight = positive_weights(labels.index_select(0, indices)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = make_loader(features, labels, indices, config.batch_size, True, seed)
    for _ in range(max(1, int(epochs))):
        model.train()
        for batch_features, batch_labels in loader:
            output = model(batch_features.to(device))
            loss = ordinal_region_loss(output, batch_labels.to(device), pos_weight, config.score_penalty)["total"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model, features: torch.Tensor, device, batch_size: int) -> torch.Tensor:
    outputs = []
    for start in range(0, len(features), int(batch_size)):
        outputs.append(model(features[start:start + batch_size].to(device))["logits"].detach().cpu())
    return torch.cat(outputs, dim=0)


def select_temperature(logits: torch.Tensor, targets: torch.Tensor, grid: Sequence[float]) -> float:
    rows = []
    for temperature in grid:
        probabilities = region_probabilities_from_logits(logits, temperature)
        rows.append((region_nll(probabilities, targets), float(temperature)))
    return min(rows, key=lambda row: (row[0], abs(row[1] - 1.0)))[1]


__all__ = [
    "ROUTER_VERSION", "RegionModelConfigV918", "PolicyGridV918", "SafetyConfigV918",
    "normalize_router_pool", "group_bucket", "inner_split_indices", "region_metrics",
    "train_region_model", "train_fixed_epochs", "predict_logits", "select_temperature",
]
