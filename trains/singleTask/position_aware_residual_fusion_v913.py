"""Validation-only fusion for the frozen five-model MOSI expert pool.

V9.13 deliberately avoids a learned neural router.  It treats the anchor and
four frozen specialists as fixed candidate functions, selects only low-
dimensional convex rules on the official Validation split, and applies the
selected rule once to Test.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from .model.SemanticCostCoachV99 import ACTION_NAMES, SPECIALIST_NAMES

FUSION_VERSION = "position_aware_residual_fusion_v913_v1"
EXPERT_CENTERS = {
    "strong_negative": -2.25,
    "boundary": 0.0,
    "positive": 1.0,
    "strong_positive": 2.25,
}


def _column(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.detach().cpu().float()
    if value.dim() == 1:
        value = value.view(-1, 1)
    if value.dim() != 2 or value.size(1) != 1:
        raise ValueError(f"{name} must have shape [N,1], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def pool_tensors(pool: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    """Validate one collected expert pool and return aligned CPU tensors."""
    anchor = _column(pool["anchor"], "anchor")
    labels = _column(pool["labels"], "labels")
    if len(anchor) != len(labels):
        raise ValueError("anchor/label length mismatch")
    sample_ids = list(pool.get("sample_ids", []))
    if len(sample_ids) != len(labels):
        raise ValueError("sample-id/label length mismatch")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample IDs in expert pool")

    predictions = [anchor]
    confidences = []
    experts = pool.get("experts", {})
    for name in SPECIALIST_NAMES:
        if name not in experts:
            raise KeyError(f"missing specialist: {name}")
        values = experts[name]
        prediction = _column(values["prediction"], f"{name}.prediction")
        confidence = _column(values["confidence"], f"{name}.confidence")
        if len(prediction) != len(anchor) or len(confidence) != len(anchor):
            raise ValueError(f"{name} sample count mismatch")
        predictions.append(prediction)
        confidences.append(confidence.clamp(0.0, 1.0))

    actions = torch.cat(predictions, dim=1)
    confidence = torch.cat(confidences, dim=1)
    if actions.shape != (len(labels), len(ACTION_NAMES)):
        raise RuntimeError("unexpected action matrix shape")
    return {
        "actions": actions,
        "confidence": confidence,
        "labels": labels,
    }


def mae(prediction: torch.Tensor, labels: torch.Tensor) -> float:
    prediction = _column(prediction, "prediction")
    labels = _column(labels, "labels")
    return float(torch.mean(torch.abs(prediction - labels)).item())


def harm_over_margin_rate(
    prediction: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.10,
) -> float:
    candidate_error = torch.abs(_column(prediction, "prediction") - labels)
    anchor_error = torch.abs(_column(anchor, "anchor") - labels)
    return float(torch.mean((candidate_error - anchor_error > margin).float()).item())


def evaluate_prediction(
    prediction: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.10,
) -> Dict[str, float]:
    candidate_mae = mae(prediction, labels)
    anchor_mae = mae(actions[:, :1], labels)
    return {
        "mae": candidate_mae,
        "anchor_mae": anchor_mae,
        "gain_vs_anchor": anchor_mae - candidate_mae,
        "harm_over_010_rate": harm_over_margin_rate(
            prediction, actions[:, :1], labels, margin=margin
        ),
    }


def position_aware_weights(
    actions: torch.Tensor,
    confidence: torch.Tensor,
    position_source: str,
    tau: float,
    rho: float,
    confidence_power: float,
) -> torch.Tensor:
    """Return four specialist weights whose row sums never exceed ``rho``."""
    if actions.dim() != 2 or actions.size(1) != len(ACTION_NAMES):
        raise ValueError("actions must have shape [N,5]")
    if confidence.shape != (len(actions), len(SPECIALIST_NAMES)):
        raise ValueError("confidence must have shape [N,4]")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0,1]")
    if confidence_power < 0.0:
        raise ValueError("confidence_power must be non-negative")

    if position_source == "anchor":
        position = actions[:, 0]
    elif position_source == "median":
        position = actions.median(dim=1).values
    else:
        raise ValueError(f"unknown position source: {position_source}")

    centers = actions.new_tensor(
        [EXPERT_CENTERS[name] for name in SPECIALIST_NAMES]
    )
    scaled_distance = (position.view(-1, 1) - centers.view(1, -1)) / float(tau)
    activation = torch.exp(-0.5 * scaled_distance.square())
    if confidence_power > 0.0:
        activation = activation * confidence.clamp_min(1e-6).pow(
            float(confidence_power)
        )

    normalized = activation / activation.sum(dim=1, keepdim=True).clamp_min(1e-12)
    # The strongest location match controls how much of the maximum budget is
    # actually spent.  This permits a strict fallback toward Anchor.
    budget = float(rho) * activation.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)
    weights = normalized * budget
    if bool((weights < -1e-8).any()):
        raise RuntimeError("negative specialist weight")
    if bool((weights.sum(dim=1) > float(rho) + 1e-6).any()):
        raise RuntimeError("position-aware residual budget exceeded")
    return weights


def position_aware_prediction(
    actions: torch.Tensor,
    confidence: torch.Tensor,
    position_source: str,
    tau: float,
    rho: float,
    confidence_power: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = position_aware_weights(
        actions,
        confidence,
        position_source=position_source,
        tau=tau,
        rho=rho,
        confidence_power=confidence_power,
    )
    anchor = actions[:, :1]
    residuals = actions[:, 1:] - anchor
    prediction = anchor + (weights * residuals).sum(dim=1, keepdim=True)
    return prediction, weights


def simplex_integer_compositions(total: int, parts: int) -> Iterable[Tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in simplex_integer_compositions(total - first, parts - 1):
            yield (first, *tail)


def simplex_grid(step: float, parts: int = 5) -> List[Tuple[float, ...]]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("simplex step must be in (0,1]")
    units = round(1.0 / float(step))
    if abs(units * float(step) - 1.0) > 1e-8:
        raise ValueError("simplex step must divide one exactly")
    return [
        tuple(value / units for value in composition)
        for composition in simplex_integer_compositions(units, parts)
    ]


def apply_config(
    pool: Mapping[str, object],
    config: Mapping[str, object],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    values = pool_tensors(pool)
    actions = values["actions"]
    confidence = values["confidence"]
    family = str(config["family"])
    extras: Dict[str, torch.Tensor] = {}

    if family == "anchor":
        prediction = actions[:, :1]
    elif family == "single":
        index = ACTION_NAMES.index(str(config["action"]))
        prediction = actions[:, index : index + 1]
    elif family == "pairwise":
        index = ACTION_NAMES.index(str(config["expert"]))
        beta = float(config["beta"])
        prediction = (1.0 - beta) * actions[:, :1] + beta * actions[:, index : index + 1]
    elif family == "simplex":
        weights = actions.new_tensor(list(config["weights"])).view(1, -1)
        if weights.shape[1] != len(ACTION_NAMES):
            raise ValueError("simplex config must provide five weights")
        if abs(float(weights.sum().item()) - 1.0) > 1e-6 or bool((weights < 0).any()):
            raise ValueError("invalid simplex weights")
        prediction = (actions * weights).sum(dim=1, keepdim=True)
        extras["action_weights"] = weights.expand(len(actions), -1)
    elif family == "position_aware":
        prediction, weights = position_aware_prediction(
            actions,
            confidence,
            position_source=str(config["position_source"]),
            tau=float(config["tau"]),
            rho=float(config["rho"]),
            confidence_power=float(config["confidence_power"]),
        )
        extras["specialist_weights"] = weights
    else:
        raise ValueError(f"unknown fusion family: {family}")
    return prediction, extras


def _candidate_row(
    candidate_id: str,
    config: Mapping[str, object],
    prediction: torch.Tensor,
    actions: torch.Tensor,
    labels: torch.Tensor,
    complexity: int,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "candidate_id": candidate_id,
        "family": str(config["family"]),
        "complexity": int(complexity),
        "config": dict(config),
    }
    result.update(evaluate_prediction(prediction, actions, labels))
    return result


def _better(row: Mapping[str, object], best: Mapping[str, object] | None) -> bool:
    if best is None:
        return True
    left = (
        float(row["mae"]),
        float(row["harm_over_010_rate"]),
        int(row["complexity"]),
        str(row["candidate_id"]),
    )
    right = (
        float(best["mae"]),
        float(best["harm_over_010_rate"]),
        int(best["complexity"]),
        str(best["candidate_id"]),
    )
    return left < right


def fit_validation_fusion(
    valid_pool: Mapping[str, object],
    beta_grid: Sequence[float],
    simplex_step: float,
    position_sources: Sequence[str],
    tau_grid: Sequence[float],
    rho_grid: Sequence[float],
    confidence_powers: Sequence[float],
    minimum_validation_gain: float = 0.0005,
    maximum_validation_harm: float = 0.05,
) -> Dict[str, object]:
    """Select a deployable rule using Validation tensors only."""
    values = pool_tensors(valid_pool)
    actions = values["actions"]
    labels = values["labels"]
    rows: List[Dict[str, object]] = []

    anchor_config = {"family": "anchor"}
    anchor_row = _candidate_row(
        "anchor", anchor_config, actions[:, :1], actions, labels, complexity=0
    )
    rows.append(anchor_row)

    best_single = None
    for name in SPECIALIST_NAMES:
        config = {"family": "single", "action": name}
        index = ACTION_NAMES.index(name)
        prediction = actions[:, index : index + 1]
        row = _candidate_row(f"single__{name}", config, prediction, actions, labels, 1)
        rows.append(row)
        if _better(row, best_single):
            best_single = row

    best_pairwise = None
    best_pairwise_by_expert: Dict[str, Dict[str, object]] = {}
    for name, beta in product(SPECIALIST_NAMES, beta_grid):
        beta = float(beta)
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta grid values must be in [0,1]")
        config = {"family": "pairwise", "expert": name, "beta": beta}
        index = ACTION_NAMES.index(name)
        prediction = (1.0 - beta) * actions[:, :1] + beta * actions[:, index : index + 1]
        candidate_id = f"pairwise__{name}__beta_{beta:.4f}"
        row = _candidate_row(candidate_id, config, prediction, actions, labels, 2)
        rows.append(row)
        current = best_pairwise_by_expert.get(name)
        if _better(row, current):
            best_pairwise_by_expert[name] = row
        if _better(row, best_pairwise):
            best_pairwise = row

    best_simplex = None
    simplex_count = 0
    for weights in simplex_grid(simplex_step, parts=len(ACTION_NAMES)):
        simplex_count += 1
        config = {"family": "simplex", "weights": list(weights)}
        weight_tensor = actions.new_tensor(weights).view(1, -1)
        prediction = (actions * weight_tensor).sum(dim=1, keepdim=True)
        row = _candidate_row(
            "simplex__" + "_".join(f"{value:.4f}" for value in weights),
            config,
            prediction,
            actions,
            labels,
            3,
        )
        if _better(row, best_simplex):
            best_simplex = row
    if best_simplex is None:
        raise RuntimeError("simplex search produced no candidates")
    rows.append(best_simplex)

    best_position = None
    for source, tau, rho, confidence_power in product(
        position_sources, tau_grid, rho_grid, confidence_powers
    ):
        config = {
            "family": "position_aware",
            "position_source": str(source),
            "tau": float(tau),
            "rho": float(rho),
            "confidence_power": float(confidence_power),
        }
        prediction, _ = position_aware_prediction(
            actions,
            values["confidence"],
            position_source=str(source),
            tau=float(tau),
            rho=float(rho),
            confidence_power=float(confidence_power),
        )
        candidate_id = (
            f"position__{source}__tau_{float(tau):.4f}__"
            f"rho_{float(rho):.4f}__cp_{float(confidence_power):.4f}"
        )
        row = _candidate_row(candidate_id, config, prediction, actions, labels, 4)
        rows.append(row)
        if _better(row, best_position):
            best_position = row

    family_best = [anchor_row]
    family_best.extend(best_pairwise_by_expert.values())
    for value in (best_single, best_pairwise, best_simplex, best_position):
        if value is not None:
            family_best.append(value)

    unique = {str(row["candidate_id"]): row for row in family_best}
    eligible = [anchor_row]
    for row in unique.values():
        if row["candidate_id"] == "anchor":
            continue
        if (
            float(row["gain_vs_anchor"]) >= float(minimum_validation_gain)
            and float(row["harm_over_010_rate"]) <= float(maximum_validation_harm)
        ):
            eligible.append(row)
    selected = min(
        eligible,
        key=lambda row: (
            float(row["mae"]),
            int(row["complexity"]),
            float(row["harm_over_010_rate"]),
            str(row["candidate_id"]),
        ),
    )

    return {
        "version": FUSION_VERSION,
        "anchor": anchor_row,
        "best_single": best_single,
        "best_pairwise": best_pairwise,
        "best_pairwise_by_expert": best_pairwise_by_expert,
        "best_simplex": best_simplex,
        "best_position_aware": best_position,
        "selected": selected,
        "candidate_rows": rows,
        "simplex_candidate_count": simplex_count,
        "minimum_validation_gain": float(minimum_validation_gain),
        "maximum_validation_harm": float(maximum_validation_harm),
        "selected_by_validation_only": True,
    }


def selected_family_configs(fit: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    result: Dict[str, Mapping[str, object]] = {
        "anchor": fit["anchor"]["config"],
        "validation_best_single": fit["best_single"]["config"],
        "validation_best_pairwise": fit["best_pairwise"]["config"],
        "validation_best_simplex": fit["best_simplex"]["config"],
        "validation_best_position_aware": fit["best_position_aware"]["config"],
        "validation_selected_deployable": fit["selected"]["config"],
    }
    for name, row in fit["best_pairwise_by_expert"].items():
        result[f"validation_pairwise_{name}"] = row["config"]
    return result
