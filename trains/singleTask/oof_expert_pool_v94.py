"""Fully nested OOF specialist pool construction for the V9.4 coach."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .grouped_oof_function_space_v92 import run_nested_oof_cfcompat
from .model.CostSensitiveCoachV94 import (
    ACTION_NAMES,
    SPECIALIST_NAMES,
    CachedSpecialistResidualHeadV94,
    CostSensitiveCoachV94,
    coach_input_features,
    cost_sensitive_coach_loss,
    specialist_loss,
    stack_action_predictions,
)
from .oof_group_splits_v92 import (
    StageLimits,
    build_nested_group_folds,
    canonical_sample_id,
    conversation_group_id,
)

logger = logging.getLogger("MMSA")
POOL_VERSION = "fully_nested_oof_action_cost_pool_v9_4"


@dataclass(frozen=True)
class SpecialistConfigV94:
    hidden_dim: int = 64
    dropout: float = 0.10
    residual_max: float = 1.25
    epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    membership_temperature: float = 0.25
    gain_margin: float = 0.04


@dataclass(frozen=True)
class CoachConfigV94:
    hidden_dim: int = 96
    dropout: float = 0.15
    ordinal_residual_max: float = 2.0
    max_epochs: int = 30
    early_stop: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    cost_temperature: float = 0.10
    abstain_margin: float = 0.02


class ReindexedDatasetView(Dataset):
    """Expose a subset as a contiguous dataset for recursive grouped OOF runs."""

    def __init__(self, base: Dataset, global_indices: Sequence[int]) -> None:
        self.base = base
        self.global_indices = tuple(int(value) for value in global_indices)
        self.ids = [base.ids[index] for index in self.global_indices]
        self.labels = {
            key: np.asarray(value)[np.asarray(self.global_indices, dtype=int)]
            for key, value in base.labels.items()
        }

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, local_index):
        item = dict(self.base[self.global_indices[int(local_index)]])
        item["index"] = int(local_index)
        return item


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _tensor_loader(
    tensors: Sequence[torch.Tensor],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        drop_last=False,
    )


def _new_specialist(feature_dim, config: SpecialistConfigV94, device):
    return CachedSpecialistResidualHeadV94(
        feature_dim=int(feature_dim),
        hidden_dim=int(config.hidden_dim),
        dropout=float(config.dropout),
        residual_max=float(config.residual_max),
    ).to(device)


def fit_specialist_fixed(
    feature: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor,
    role: str,
    config: SpecialistConfigV94,
    device,
    seed: int,
    epochs: int | None = None,
):
    """Fit a specialist for a pre-registered epoch count; no sample selects itself."""
    _seed_all(seed)
    model = _new_specialist(feature.size(1), config, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loader = _tensor_loader(
        (feature.float(), anchor.float(), labels.float()),
        config.batch_size,
        True,
        seed,
    )
    history = []
    epoch_count = int(config.epochs if epochs is None else epochs)
    for epoch in range(1, epoch_count + 1):
        model.train()
        totals: Dict[str, float] = {}
        for batch_feature, batch_anchor, batch_labels in loader:
            output = model(batch_feature.to(device), batch_anchor.to(device))
            losses = specialist_loss(
                output,
                batch_labels.to(device),
                role=role,
                membership_temperature=config.membership_temperature,
                gain_margin=config.gain_margin,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        history.append(
            {
                "epoch": epoch,
                **{key: value / max(1, len(loader)) for key, value in totals.items()},
            }
        )
    model.eval()
    return model, history


@torch.no_grad()
def apply_specialist(model, feature, anchor, device):
    model.eval()
    output = model(feature.float().to(device), anchor.float().to(device))
    return {
        key: output[key].detach().cpu()
        for key in ("prediction", "correction", "applicability_prob")
    }


def crossfit_specialists(
    feature: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor,
    fold_index: torch.Tensor,
    config: SpecialistConfigV94,
    device,
    seed: int,
    save_dir: Path | None = None,
):
    """Create OOF specialist actions inside one top-level development set."""
    sample_count = labels.size(0)
    predictions = torch.full(
        (sample_count, len(SPECIALIST_NAMES), 1), float("nan")
    )
    confidences = torch.full_like(predictions, float("nan"))
    corrections = torch.full_like(predictions, float("nan"))
    histories = []
    unique_folds = sorted(int(value) for value in fold_index.unique().tolist())
    for heldout_fold in unique_folds:
        train_mask = fold_index != heldout_fold
        holdout_mask = fold_index == heldout_fold
        if not train_mask.any() or not holdout_mask.any():
            raise RuntimeError("specialist cross-fit produced an empty partition")
        for role_index, role in enumerate(SPECIALIST_NAMES):
            role_seed = int(seed) + 1009 * (heldout_fold + 1) + 53 * (role_index + 1)
            model, history = fit_specialist_fixed(
                feature[train_mask],
                anchor[train_mask],
                labels[train_mask],
                role,
                config,
                device,
                role_seed,
            )
            collected = apply_specialist(
                model, feature[holdout_mask], anchor[holdout_mask], device
            )
            predictions[holdout_mask, role_index] = collected["prediction"]
            confidences[holdout_mask, role_index] = collected[
                "applicability_prob"
            ]
            corrections[holdout_mask, role_index] = collected["correction"]
            for row in history:
                histories.append(
                    {
                        "heldout_fold": heldout_fold,
                        "role": role,
                        **row,
                    }
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not torch.isfinite(predictions).all() or not torch.isfinite(confidences).all():
        raise FloatingPointError("cross-fitted specialist pool contains non-finite values")
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(histories).to_csv(
            save_dir / "v94_inner_specialist_crossfit_history.csv", index=False
        )
    return {
        "predictions": predictions,
        "confidences": confidences,
        "corrections": corrections,
        "history": histories,
    }


def fit_final_specialists(
    feature,
    anchor,
    labels,
    config: SpecialistConfigV94,
    device,
    seed,
    save_dir: Path | None = None,
):
    models = {}
    histories = []
    for role_index, role in enumerate(SPECIALIST_NAMES):
        model, history = fit_specialist_fixed(
            feature,
            anchor,
            labels,
            role,
            config,
            device,
            int(seed) + 79 * (role_index + 1),
        )
        models[role] = model
        for row in history:
            histories.append({"role": role, **row})
        if save_dir is not None:
            path = Path(save_dir) / f"specialist_{role}_v94.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "method": "oof_cached_specialist_v9_4",
                    "role": role,
                    "feature_dim": int(feature.size(1)),
                    "config": config.__dict__,
                    "state_dict": _cpu_state_dict(model),
                },
                path,
            )
    if save_dir is not None:
        pd.DataFrame(histories).to_csv(
            Path(save_dir) / "v94_final_specialist_history.csv", index=False
        )
    return models, histories


def collect_specialist_models(models, feature, anchor, device):
    predictions = []
    confidences = []
    corrections = []
    for role in SPECIALIST_NAMES:
        output = apply_specialist(models[role], feature, anchor, device)
        predictions.append(output["prediction"])
        confidences.append(output["applicability_prob"])
        corrections.append(output["correction"])
    return {
        "predictions": torch.stack(predictions, dim=1),
        "confidences": torch.stack(confidences, dim=1),
        "corrections": torch.stack(corrections, dim=1),
    }


def _new_coach(input_dim, config: CoachConfigV94, device):
    return CostSensitiveCoachV94(
        input_dim=int(input_dim),
        hidden_dim=int(config.hidden_dim),
        dropout=float(config.dropout),
        ordinal_residual_max=float(config.ordinal_residual_max),
    ).to(device)


def _coach_validation_row(model, features, anchor, actions, labels, device):
    model.eval()
    with torch.no_grad():
        output = model(features.to(device), anchor.to(device))
        costs = torch.abs(actions.to(device) - labels.view(-1, 1, 1).to(device)).squeeze(-1)
        expected = (output["action_probs"] * costs).sum(dim=1).mean()
        hard_index = output["action_probs"].argmax(dim=1)
        hard_prediction = actions.to(device)[
            torch.arange(actions.size(0), device=device), hard_index
        ]
        hard_mae = torch.abs(hard_prediction - labels.to(device)).mean()
        anchor_mae = torch.abs(anchor.to(device) - labels.to(device)).mean()
        objective = 0.65 * expected + 0.35 * hard_mae
        return {
            "objective": float(objective.item()),
            "expected_cost": float(expected.item()),
            "hard_mae": float(hard_mae.item()),
            "anchor_mae": float(anchor_mae.item()),
            "hard_gain": float((anchor_mae - hard_mae).item()),
        }


def fit_coach_selected_epoch(
    function_space,
    anchor,
    labels,
    expert_predictions,
    expert_confidences,
    fold_index,
    config: CoachConfigV94,
    device,
    seed,
):
    """Select coach epoch inside a top-level development set, then refit all data."""
    features = coach_input_features(
        function_space, anchor, expert_predictions, expert_confidences
    )
    actions = stack_action_predictions(anchor, expert_predictions)
    unique_folds = sorted(int(value) for value in fold_index.unique().tolist())
    selection_fold = unique_folds[int(seed) % len(unique_folds)]
    train_mask = fold_index != selection_fold
    valid_mask = fold_index == selection_fold
    _seed_all(seed)
    model = _new_coach(features.size(1), config, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _tensor_loader(
        (
            features[train_mask].float(),
            anchor[train_mask].float(),
            actions[train_mask].float(),
            labels[train_mask].float(),
        ),
        config.batch_size,
        True,
        seed,
    )
    history = []
    best = None
    no_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        totals: Dict[str, float] = {}
        for batch_features, batch_anchor, batch_actions, batch_labels in loader:
            output = model(batch_features.to(device), batch_anchor.to(device))
            losses = cost_sensitive_coach_loss(
                output,
                batch_actions.to(device),
                batch_labels.to(device),
                cost_temperature=config.cost_temperature,
                abstain_margin=config.abstain_margin,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        valid_row = _coach_validation_row(
            model,
            features[valid_mask],
            anchor[valid_mask],
            actions[valid_mask],
            labels[valid_mask],
            device,
        )
        row = {
            "epoch": epoch,
            "selection_fold": selection_fold,
            **{f"train_{key}": value / max(1, len(loader)) for key, value in totals.items()},
            **{f"valid_{key}": value for key, value in valid_row.items()},
        }
        history.append(row)
        if best is None or valid_row["objective"] < best["objective"] - 1e-6:
            best = {
                "epoch": epoch,
                "objective": valid_row["objective"],
                "valid": valid_row,
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= config.early_stop:
            break
    if best is None:
        raise RuntimeError("coach failed to select an epoch")

    _seed_all(seed + 500003)
    final_model = _new_coach(features.size(1), config, device)
    optimizer = optim.AdamW(
        final_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    full_loader = _tensor_loader(
        (features.float(), anchor.float(), actions.float(), labels.float()),
        config.batch_size,
        True,
        seed + 500003,
    )
    for _ in range(int(best["epoch"])):
        final_model.train()
        for batch_features, batch_anchor, batch_actions, batch_labels in full_loader:
            output = final_model(batch_features.to(device), batch_anchor.to(device))
            losses = cost_sensitive_coach_loss(
                output,
                batch_actions.to(device),
                batch_labels.to(device),
                cost_temperature=config.cost_temperature,
                abstain_margin=config.abstain_margin,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 2.0)
            optimizer.step()
    final_model.eval()
    return final_model, best, history


@torch.no_grad()
def apply_coach(model, function_space, anchor, expert_predictions, expert_confidences, device):
    features = coach_input_features(
        function_space, anchor, expert_predictions, expert_confidences
    )
    output = model(features.to(device), anchor.to(device))
    return {key: value.detach().cpu() for key, value in output.items()}


def _outer_specs_from_top_cache(top_cache):
    specs, manifest = build_nested_group_folds(
        top_cache["sample_ids"],
        top_cache["labels"].view(-1).tolist(),
        outer_folds=int(top_cache["outer_folds"]),
        inner_valid_fraction=float(top_cache["inner_valid_fraction"]),
        seed=int(top_cache["seed"]),
    )
    reconstructed = torch.full_like(top_cache["fold_index"], -1)
    for spec in specs:
        reconstructed[list(spec.outer_holdout_indices)] = int(spec.outer_fold)
    if not torch.equal(reconstructed.cpu(), top_cache["fold_index"].cpu()):
        raise RuntimeError("V9.2 OOF fold reconstruction does not match the cache")
    return specs, manifest


def build_fully_nested_expert_pool(
    args,
    train_dataset,
    top_oof_cache_path: Path,
    output_dir: Path,
    inner_folds: int,
    inner_valid_fraction: float,
    num_workers: int,
    limits: StageLimits,
    specialist_config: SpecialistConfigV94,
    coach_config: CoachConfigV94,
    resume: bool = True,
):
    """Build top-level OOF expert actions and coach predictions without label leakage."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_cache = torch.load(top_oof_cache_path, map_location="cpu")
    required = {
        "sample_ids", "group_ids", "labels", "oof_prediction", "oof_feature",
        "fold_index", "outer_folds", "inner_valid_fraction", "seed",
    }
    if not required.issubset(top_cache):
        raise ValueError("top-level V9.2 OOF cache is incomplete")
    specs, manifest = _outer_specs_from_top_cache(top_cache)
    manifest.to_csv(output_dir / "v94_top_outer_manifest.csv", index=False)

    n = len(top_cache["sample_ids"])
    expert_predictions = torch.full((n, len(SPECIALIST_NAMES), 1), float("nan"))
    expert_confidences = torch.full_like(expert_predictions, float("nan"))
    expert_corrections = torch.full_like(expert_predictions, float("nan"))
    coach_action_probs = torch.full((n, len(ACTION_NAMES)), float("nan"))
    coach_region_probs = torch.full((n, 5), float("nan"))
    coach_ordinal_score = torch.full((n, 1), float("nan"))
    fold_metadata = []

    for spec in specs:
        fold_dir = output_dir / f"outer_fold_{spec.outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_cache_path = fold_dir / "outer_expert_holdout_v94.pth"
        if resume and fold_cache_path.is_file():
            cached = torch.load(fold_cache_path, map_location="cpu")
            if cached.get("version") != POOL_VERSION:
                raise RuntimeError(f"stale V9.4 fold cache: {fold_cache_path}")
            holdout_indices = torch.tensor(cached["global_holdout_indices"], dtype=torch.long)
            expert_predictions[holdout_indices] = cached["expert_predictions"]
            expert_confidences[holdout_indices] = cached["expert_confidences"]
            expert_corrections[holdout_indices] = cached["expert_corrections"]
            coach_action_probs[holdout_indices] = cached["coach_action_probs"]
            coach_region_probs[holdout_indices] = cached["coach_region_probs"]
            coach_ordinal_score[holdout_indices] = cached["coach_ordinal_score"]
            fold_metadata.append(cached["metadata"])
            logger.info("V9.4 outer fold=%d resumed", spec.outer_fold)
            continue

        development_indices = tuple(
            sorted((*spec.inner_train_indices, *spec.inner_valid_indices))
        )
        view = ReindexedDatasetView(train_dataset, development_indices)
        inner_seed = int(top_cache["seed"]) + 100003 * (spec.outer_fold + 1)
        inner_dir = fold_dir / "development_inner_oof"
        inner_cache = run_nested_oof_cfcompat(
            args=args,
            train_dataset=view,
            output_dir=inner_dir,
            seed=inner_seed,
            outer_folds=int(inner_folds),
            inner_valid_fraction=float(inner_valid_fraction),
            num_workers=int(num_workers),
            limits=limits,
            resume=resume,
        )
        if inner_cache.get("feature_space") != top_cache.get("feature_space"):
            raise RuntimeError("inner/top function-space schema mismatch")

        dev_specialists = crossfit_specialists(
            inner_cache["oof_feature"],
            inner_cache["oof_prediction"],
            inner_cache["labels"],
            inner_cache["fold_index"],
            specialist_config,
            args.device,
            inner_seed + 7001,
            fold_dir,
        )
        final_specialists, final_history = fit_final_specialists(
            inner_cache["oof_feature"],
            inner_cache["oof_prediction"],
            inner_cache["labels"],
            specialist_config,
            args.device,
            inner_seed + 9001,
            None,
        )
        fold_coach, coach_best, coach_history = fit_coach_selected_epoch(
            inner_cache["oof_feature"],
            inner_cache["oof_prediction"],
            inner_cache["labels"],
            dev_specialists["predictions"],
            dev_specialists["confidences"],
            inner_cache["fold_index"],
            coach_config,
            args.device,
            inner_seed + 11003,
        )
        pd.DataFrame(coach_history).to_csv(
            fold_dir / "v94_inner_coach_history.csv", index=False
        )
        pd.DataFrame(final_history).to_csv(
            fold_dir / "v94_fold_final_specialist_history.csv", index=False
        )

        global_holdout = torch.tensor(spec.outer_holdout_indices, dtype=torch.long)
        holdout_feature = top_cache["oof_feature"][global_holdout]
        holdout_anchor = top_cache["oof_prediction"][global_holdout]
        holdout_specialists = collect_specialist_models(
            final_specialists,
            holdout_feature,
            holdout_anchor,
            args.device,
        )
        holdout_coach = apply_coach(
            fold_coach,
            holdout_feature,
            holdout_anchor,
            holdout_specialists["predictions"],
            holdout_specialists["confidences"],
            args.device,
        )
        expert_predictions[global_holdout] = holdout_specialists["predictions"]
        expert_confidences[global_holdout] = holdout_specialists["confidences"]
        expert_corrections[global_holdout] = holdout_specialists["corrections"]
        coach_action_probs[global_holdout] = holdout_coach["action_probs"]
        coach_region_probs[global_holdout] = holdout_coach["region_probs"]
        coach_ordinal_score[global_holdout] = holdout_coach["ordinal_score"]

        metadata = {
            "outer_fold": int(spec.outer_fold),
            "development_count": len(development_indices),
            "outer_holdout_count": len(spec.outer_holdout_indices),
            "inner_folds": int(inner_folds),
            "inner_seed": int(inner_seed),
            "coach_selected_epoch": int(coach_best["epoch"]),
            "coach_selected_valid": coach_best["valid"],
            "inner_oof_mae": float(
                torch.abs(inner_cache["oof_prediction"] - inner_cache["labels"]).mean().item()
            ),
            "inner_oof_cache": str(
                inner_dir / "nested_grouped_oof_cfcompat_cache_v92.pth"
            ),
        }
        fold_metadata.append(metadata)
        torch.save(
            {
                "version": POOL_VERSION,
                "global_holdout_indices": list(spec.outer_holdout_indices),
                "expert_predictions": holdout_specialists["predictions"],
                "expert_confidences": holdout_specialists["confidences"],
                "expert_corrections": holdout_specialists["corrections"],
                "coach_action_probs": holdout_coach["action_probs"],
                "coach_region_probs": holdout_coach["region_probs"],
                "coach_ordinal_score": holdout_coach["ordinal_score"],
                "metadata": metadata,
            },
            fold_cache_path,
        )
        del fold_coach
        final_specialists.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tensors = (
        expert_predictions,
        expert_confidences,
        expert_corrections,
        coach_action_probs,
        coach_region_probs,
        coach_ordinal_score,
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("V9.4 nested pool has incomplete or non-finite values")

    payload = {
        "version": POOL_VERSION,
        "method": "fully_nested_grouped_oof_specialists_and_cost_coach",
        "sample_ids": [canonical_sample_id(value) for value in top_cache["sample_ids"]],
        "group_ids": [conversation_group_id(value) for value in top_cache["sample_ids"]],
        "labels": top_cache["labels"].float(),
        "anchor": top_cache["oof_prediction"].float(),
        "function_space": top_cache["oof_feature"].float(),
        "feature_space": top_cache.get("feature_space"),
        "feature_keys": list(top_cache.get("feature_keys", [])),
        "expert_names": list(SPECIALIST_NAMES),
        "action_names": list(ACTION_NAMES),
        "expert_predictions": expert_predictions,
        "expert_confidences": expert_confidences,
        "expert_corrections": expert_corrections,
        "coach_action_probs": coach_action_probs,
        "coach_region_probs": coach_region_probs,
        "coach_ordinal_score": coach_ordinal_score,
        "fold_index": top_cache["fold_index"].long(),
        "fold_metadata": fold_metadata,
        "specialist_config": specialist_config.__dict__,
        "coach_config": coach_config.__dict__,
        "protocol": (
            "For each top-level holdout video group, all development action costs are "
            "built from an inner grouped OOF CFCompat cache. Specialist heads and the "
            "fold coach train only on that development cache. The top holdout is never "
            "used by any backbone, specialist, coach, checkpoint, or epoch selection."
        ),
    }
    cache_path = output_dir / "fully_nested_oof_expert_pool_v94.pth"
    torch.save(payload, cache_path)

    frame = {
        "sample_id": payload["sample_ids"],
        "group_id": payload["group_ids"],
        "outer_fold": payload["fold_index"].tolist(),
        "label": payload["labels"].view(-1).tolist(),
        "anchor": payload["anchor"].view(-1).tolist(),
        "nested_coach_action": payload["coach_action_probs"].argmax(dim=1).tolist(),
        "nested_coach_confidence": payload["coach_action_probs"].max(dim=1).values.tolist(),
    }
    for role_index, role in enumerate(SPECIALIST_NAMES):
        frame[f"{role}_prediction"] = expert_predictions[:, role_index, 0].tolist()
        frame[f"{role}_confidence"] = expert_confidences[:, role_index, 0].tolist()
        frame[f"{role}_correction"] = expert_corrections[:, role_index, 0].tolist()
    pd.DataFrame(frame).to_csv(
        output_dir / "fully_nested_oof_expert_pool_v94.csv", index=False
    )

    actions = stack_action_predictions(payload["anchor"], expert_predictions)
    costs = torch.abs(actions - payload["labels"].view(-1, 1, 1)).squeeze(-1)
    summary = {
        "version": POOL_VERSION,
        "sample_count": n,
        "group_count": len(set(payload["group_ids"])),
        "outer_folds": int(top_cache["outer_folds"]),
        "inner_folds": int(inner_folds),
        "anchor_oof_mae": float(costs[:, 0].mean().item()),
        "sample_oracle_oof_mae": float(costs.min(dim=1).values.mean().item()),
        "best_action_counts": {
            ACTION_NAMES[index]: int((costs.argmin(dim=1) == index).sum().item())
            for index in range(len(ACTION_NAMES))
        },
        "fold_metadata": fold_metadata,
        "cache_path": str(cache_path),
        "protocol": payload["protocol"],
    }
    (output_dir / "fully_nested_oof_expert_pool_v94_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return payload


def median_selected_coach_epoch(pool: Mapping[str, object]) -> int:
    values = [
        int(row["coach_selected_epoch"])
        for row in pool.get("fold_metadata", [])
        if int(row.get("coach_selected_epoch", 0)) > 0
    ]
    if not values:
        return 1
    return max(1, int(round(median(values))))
