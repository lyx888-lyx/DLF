"""Group-cross-fitted V9.3-architecture candidate pool for the V9.7 coach."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from .expert_analysis import normalize_batch_ids
from .model.AttainableFrontierCoachV97 import (
    ACTION_NAMES,
    SPECIALIST_NAMES,
    attainable_frontier_targets,
    stack_action_predictions,
)
from .model.CachedTailResidualHeadV92 import CachedTailResidualHeadV92
from .model.RoleConditionedDLF import RoleConditionedDLF
from .oof_group_splits_v92 import canonical_sample_id, conversation_group_id
from .oof_tail_residual_v92 import OOFTailLossWeights, oof_tail_residual_loss
from .role_conditioned_experts_v9 import (
    LossWeights,
    apply_teacher_weights,
    role_conditioned_loss,
)

logger = logging.getLogger("MMSA")
POOL_VERSION = "v97_crossfit_v93_architecture_frontier_pool"
ROLE_INDEX = {"boundary": 2, "positive": 3}


@dataclass(frozen=True)
class RoleCrossfitConfigV97:
    hidden_dim: int = 192
    dropout: float = 0.15
    residual_max: float = 0.45
    head_epochs: int = 4
    tail_epochs: int = 12
    head_lr: float = 3e-4
    tail_lr: float = 2e-5
    backbone_lr: float = 5e-6
    weight_decay: float = 1e-3
    membership_floor: float = 0.25
    membership_sigma_scale: float = 1.0
    gain_margin: float = 0.08
    gain_fraction: float = 0.20
    batch_size: int = 32


@dataclass(frozen=True)
class TailCrossfitConfigV97:
    hidden_dim: int = 96
    dropout: float = 0.15
    residual_max: float = 1.50
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    membership_temperature: float = 0.25
    gain_margin: float = 0.08
    gain_fraction: float = 0.20
    batch_size: int = 64


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _canonical_ids(values: Sequence[object]) -> list[str]:
    return [canonical_sample_id(value) for value in values]


def _dataset_ids(dataset) -> list[str]:
    if not hasattr(dataset, "ids"):
        raise AttributeError("MMDataset must expose ids for V9.7 alignment")
    return _canonical_ids(list(dataset.ids))


def _index_map(values: Sequence[object], name: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for index, value in enumerate(values):
        key = canonical_sample_id(value)
        if key in result:
            raise RuntimeError(f"duplicate {name} sample id: {key}")
        result[key] = int(index)
    return result


def _load_teacher_artifacts(v9_root: Path):
    cache_path = Path(v9_root) / "role_conditioned_experts_v9_teacher_cache.pth"
    weights_path = Path(v9_root) / "v9_role_teacher_weights.json"
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    cache = torch.load(cache_path, map_location="cpu")
    teacher_paths = [Path(value) for value in cache.get("teacher_paths", [])]
    if len(teacher_paths) < 3 or any(not path.is_file() for path in teacher_paths):
        raise RuntimeError("V9 teacher cache has missing teacher checkpoints")
    valid = cache["splits"]["valid"]
    valid_mae = torch.abs(
        valid["predictions"].float() - valid["labels"].float().unsqueeze(1)
    ).mean(dim=(0, 2))
    anchor_index = int(valid_mae.argmin().item())
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    global_weights = torch.tensor(weights["global_weights"], dtype=torch.float32)
    role_weights = torch.tensor(weights["role_weights"], dtype=torch.float32)
    if role_weights.shape != (5, len(teacher_paths)):
        raise RuntimeError("V9 role teacher weights do not match teacher count")
    return {
        "cache": cache,
        "teacher_paths": teacher_paths,
        "anchor_index": anchor_index,
        "global_weights": global_weights,
        "role_weights": role_weights,
        "cache_path": str(cache_path),
        "weights_path": str(weights_path),
    }


def _loader(dataset, dataset_indices, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        Subset(dataset, [int(value) for value in dataset_indices]),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def _role_batch_teacher_indices(batch_ids, teacher_map):
    values = normalize_batch_ids(batch_ids)
    try:
        return torch.tensor(
            [teacher_map[canonical_sample_id(value)] for value in values],
            dtype=torch.long,
        )
    except KeyError as error:
        raise KeyError(f"V9.7 sample missing from V9 teacher cache: {error}")


def _fit_role_expert(
    args,
    train_dataset,
    train_dataset_indices,
    role: str,
    teacher,
    config: RoleCrossfitConfigV97,
    seed: int,
    num_workers: int,
):
    _seed_all(seed)
    role_index = ROLE_INDEX[role]
    model = RoleConditionedDLF(
        args,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        residual_max=config.residual_max,
    ).to(args.device)
    model.load_backbone_checkpoint(
        teacher["teacher_paths"][teacher["anchor_index"]],
        map_location=args.device,
    )
    teacher_train = teacher["cache"]["splits"]["train"]
    teacher_map = _index_map(teacher_train["sample_ids"], "teacher-train")
    loader = _loader(
        train_dataset,
        train_dataset_indices,
        config.batch_size,
        True,
        seed,
        num_workers,
    )
    history = []
    stages = (("heads", config.head_epochs), ("tail", config.tail_epochs))
    for stage, epochs in stages:
        if int(epochs) <= 0:
            continue
        model.set_stage(stage)
        optimizer = optim.AdamW(
            model.parameter_groups(
                head_lr=config.head_lr,
                tail_lr=config.tail_lr,
                backbone_lr=config.backbone_lr,
            ),
            weight_decay=config.weight_decay,
        )
        for epoch in range(1, int(epochs) + 1):
            model.set_train_mode()
            totals: Dict[str, float] = {}
            for batch in loader:
                cache_indices = _role_batch_teacher_indices(
                    batch.get("id"), teacher_map
                )
                predictions = teacher_train["predictions"][cache_indices].float()
                anchor = predictions[:, teacher["anchor_index"]].to(args.device)
                global_teacher = apply_teacher_weights(
                    predictions, teacher["global_weights"]
                ).to(args.device)
                role_teacher = apply_teacher_weights(
                    predictions, teacher["role_weights"][role_index]
                ).to(args.device)
                labels = batch["labels"]["M"].to(args.device).view(-1, 1)
                output = model(
                    batch["text"].to(args.device),
                    batch["audio"].to(args.device),
                    batch["vision"].to(args.device),
                )
                losses = role_conditioned_loss(
                    output,
                    labels,
                    anchor,
                    global_teacher,
                    role_teacher,
                    role=role_index,
                    membership_floor=config.membership_floor,
                    membership_sigma_scale=config.membership_sigma_scale,
                    gain_margin=config.gain_margin,
                    gain_fraction=config.gain_fraction,
                    weights=LossWeights(),
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 2.0
                )
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().item()
                    )
            history.append(
                {
                    "role": role,
                    "stage": stage,
                    "epoch": epoch,
                    **{
                        key: value / max(1, len(loader))
                        for key, value in totals.items()
                    },
                }
            )
    model.eval()
    return model, history


@torch.no_grad()
def _collect_role_expert(
    model,
    args,
    dataset,
    dataset_indices,
    role: str,
    top_id_map,
    batch_size,
    num_workers,
):
    loader = _loader(
        dataset,
        dataset_indices,
        batch_size,
        False,
        0,
        num_workers,
    )
    rows = []
    role_index = ROLE_INDEX[role]
    model.eval()
    for batch in loader:
        output = model(
            batch["text"].to(args.device),
            batch["audio"].to(args.device),
            batch["vision"].to(args.device),
        )
        ids = normalize_batch_ids(batch.get("id"))
        for offset, value in enumerate(ids):
            sample_id = canonical_sample_id(value)
            rows.append(
                {
                    "top_index": top_id_map[sample_id],
                    "prediction": output["prediction"][offset].detach().cpu(),
                    "confidence": output["region_probs"][
                        offset, role_index
                    ].detach().cpu(),
                    "correction": output["correction"][offset].detach().cpu(),
                }
            )
    rows.sort(key=lambda row: row["top_index"])
    return rows


def _fit_tail_expert(
    feature,
    anchor,
    labels,
    role,
    config: TailCrossfitConfigV97,
    device,
    seed,
):
    _seed_all(seed)
    model = CachedTailResidualHeadV92(
        feature_dim=int(feature.size(1)),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        residual_max=config.residual_max,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameter_groups(config.learning_rate),
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(feature.float(), anchor.float(), labels.float()),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals: Dict[str, float] = {}
        for batch_feature, batch_anchor, batch_labels in loader:
            output = model(
                batch_feature.to(device), batch_anchor.to(device)
            )
            losses = oof_tail_residual_loss(
                output,
                batch_labels.to(device),
                role=role,
                membership_temperature=config.membership_temperature,
                gain_margin=config.gain_margin,
                gain_fraction=config.gain_fraction,
                weights=OOFTailLossWeights(),
            )
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(
                    value.detach().item()
                )
        history.append(
            {
                "role": role,
                "stage": "oof_tail_head",
                "epoch": epoch,
                **{
                    key: value / max(1, len(loader))
                    for key, value in totals.items()
                },
            }
        )
    model.eval()
    return model, history


@torch.no_grad()
def _collect_tail_expert(model, feature, anchor, device, batch_size):
    buffers = {key: [] for key in ("prediction", "correction", "applicability_prob")}
    model.eval()
    for start in range(0, len(feature), int(batch_size)):
        output = model(
            feature[start : start + batch_size].float().to(device),
            anchor[start : start + batch_size].float().to(device),
        )
        for key in buffers:
            buffers[key].append(output[key].detach().cpu())
    return {key: torch.cat(value, dim=0) for key, value in buffers.items()}


def build_crossfit_v93_frontier_pool(
    args,
    train_dataset,
    top_oof_cache_path: Path,
    v9_root: Path,
    output_dir: Path,
    role_config: RoleCrossfitConfigV97,
    tail_config: TailCrossfitConfigV97,
    num_workers: int = 1,
    resume: bool = True,
):
    """Build label-isolated outputs using the original V9.3 architectures.

    Holdout labels are excluded from every fold-local specialist optimization.
    The pre-existing V9 teacher checkpoints/cache are treated as frozen artifacts;
    they are not regenerated inside each fold, so this is architecture-matched
    group cross-fitting rather than a fully nested retraining of the teacher stack.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "crossfit_v93_frontier_pool_v97.pth"
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu")
        if (
            payload.get("version") == POOL_VERSION
            and payload.get("role_config") == role_config.__dict__
            and payload.get("tail_config") == tail_config.__dict__
        ):
            logger.info("Reusing V9.7 frontier pool: %s", final_path)
            return payload

    top = torch.load(top_oof_cache_path, map_location="cpu")
    required = {
        "sample_ids",
        "group_ids",
        "labels",
        "oof_prediction",
        "oof_feature",
        "fold_index",
        "feature_space",
    }
    if not required.issubset(top):
        raise ValueError(f"V9.2 OOF cache missing: {sorted(required - set(top))}")
    if top.get("feature_space") != "auxiliary_prediction_logits_v1":
        raise ValueError("V9.7 requires aligned V9.2 function-space features")

    top_ids = _canonical_ids(top["sample_ids"])
    top_id_map = _index_map(top_ids, "top-OOF")
    dataset_ids = _dataset_ids(train_dataset)
    dataset_id_map = _index_map(dataset_ids, "train-dataset")
    missing = sorted(set(top_ids) - set(dataset_id_map))
    if missing:
        raise RuntimeError(f"Train dataset misses {len(missing)} top OOF ids")
    top_to_dataset = [dataset_id_map[value] for value in top_ids]

    teacher = _load_teacher_artifacts(Path(v9_root))
    teacher_train_ids = set(
        _canonical_ids(teacher["cache"]["splits"]["train"]["sample_ids"])
    )
    if set(top_ids) != teacher_train_ids:
        raise RuntimeError("V9 teacher Train cache and V9.2 OOF sample sets differ")

    n = len(top_ids)
    predictions = torch.full((n, 4, 1), float("nan"))
    confidences = torch.full_like(predictions, float("nan"))
    corrections = torch.full_like(predictions, float("nan"))
    histories = []
    fold_rows = []
    unique_folds = sorted(int(value) for value in top["fold_index"].unique().tolist())
    name_to_column = {name: index for index, name in enumerate(SPECIALIST_NAMES)}

    for fold in unique_folds:
        fold_dir = output_dir / f"outer_fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_path = fold_dir / "crossfit_v93_holdout_v97.pth"
        holdout_mask = top["fold_index"] == fold
        train_mask = ~holdout_mask
        holdout_top = torch.nonzero(holdout_mask, as_tuple=False).view(-1)
        train_top = torch.nonzero(train_mask, as_tuple=False).view(-1)
        if resume and fold_path.is_file():
            cached = torch.load(fold_path, map_location="cpu")
            if (
                cached.get("version") == POOL_VERSION
                and cached.get("role_config") == role_config.__dict__
                and cached.get("tail_config") == tail_config.__dict__
            ):
                predictions[holdout_top] = cached["predictions"]
                confidences[holdout_top] = cached["confidences"]
                corrections[holdout_top] = cached["corrections"]
                histories.extend(cached.get("history", []))
                fold_rows.append(cached["metadata"])
                logger.info("V9.7 frontier outer fold=%d resumed", fold)
                continue

        fold_predictions = torch.full((len(holdout_top), 4, 1), float("nan"))
        fold_confidences = torch.full_like(fold_predictions, float("nan"))
        fold_corrections = torch.full_like(fold_predictions, float("nan"))
        train_dataset_indices = [top_to_dataset[index] for index in train_top.tolist()]
        holdout_dataset_indices = [
            top_to_dataset[index] for index in holdout_top.tolist()
        ]
        holdout_position = {
            int(top_index): offset
            for offset, top_index in enumerate(holdout_top.tolist())
        }
        fold_history = []

        for offset, role in enumerate(("boundary", "positive")):
            role_seed = int(args.seed) + 10007 * (fold + 1) + 101 * (offset + 1)
            model, history = _fit_role_expert(
                args,
                train_dataset,
                train_dataset_indices,
                role,
                teacher,
                role_config,
                role_seed,
                num_workers,
            )
            rows = _collect_role_expert(
                model,
                args,
                train_dataset,
                holdout_dataset_indices,
                role,
                top_id_map,
                role_config.batch_size,
                num_workers,
            )
            column = name_to_column[role]
            for row in rows:
                local = holdout_position[int(row["top_index"])]
                fold_predictions[local, column] = row["prediction"].view(1)
                fold_confidences[local, column] = row["confidence"].view(1)
                fold_corrections[local, column] = row["correction"].view(1)
            tagged = [{"outer_fold": fold, **row} for row in history]
            histories.extend(tagged)
            fold_history.extend(tagged)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for offset, role in enumerate(("strong_negative", "strong_positive")):
            tail_seed = int(args.seed) + 20011 * (fold + 1) + 131 * (offset + 1)
            model, history = _fit_tail_expert(
                top["oof_feature"][train_top],
                top["oof_prediction"][train_top],
                top["labels"][train_top],
                role,
                tail_config,
                args.device,
                tail_seed,
            )
            collected = _collect_tail_expert(
                model,
                top["oof_feature"][holdout_top],
                top["oof_prediction"][holdout_top],
                args.device,
                tail_config.batch_size,
            )
            column = name_to_column[role]
            fold_predictions[:, column] = collected["prediction"]
            fold_confidences[:, column] = collected["applicability_prob"]
            fold_corrections[:, column] = collected["correction"]
            tagged = [{"outer_fold": fold, **row} for row in history]
            histories.extend(tagged)
            fold_history.extend(tagged)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if any(
            not torch.isfinite(value).all()
            for value in (fold_predictions, fold_confidences, fold_corrections)
        ):
            raise FloatingPointError(f"V9.7 fold {fold} candidate pool incomplete")
        predictions[holdout_top] = fold_predictions
        confidences[holdout_top] = fold_confidences
        corrections[holdout_top] = fold_corrections
        metadata = {
            "outer_fold": fold,
            "train_count": int(train_mask.sum().item()),
            "holdout_count": int(holdout_mask.sum().item()),
            "holdout_groups": len(
                {conversation_group_id(top_ids[index]) for index in holdout_top.tolist()}
            ),
        }
        fold_rows.append(metadata)
        torch.save(
            {
                "version": POOL_VERSION,
                "role_config": role_config.__dict__,
                "tail_config": tail_config.__dict__,
                "predictions": fold_predictions,
                "confidences": fold_confidences,
                "corrections": fold_corrections,
                "history": fold_history,
                "metadata": metadata,
            },
            fold_path,
        )
        logger.info(
            "V9.7 frontier outer fold=%d complete train=%d holdout=%d",
            fold,
            metadata["train_count"],
            metadata["holdout_count"],
        )

    if any(
        not torch.isfinite(value).all()
        for value in (predictions, confidences, corrections)
    ):
        raise FloatingPointError("V9.7 cross-fitted candidate pool is incomplete")
    actions = stack_action_predictions(top["oof_prediction"], predictions)
    targets = attainable_frontier_targets(actions, top["labels"])
    payload = {
        "version": POOL_VERSION,
        "method": "group_crossfit_v93_architecture_attainable_frontier_pool",
        "sample_ids": top_ids,
        "group_ids": [conversation_group_id(value) for value in top_ids],
        "labels": top["labels"].float(),
        "anchor": top["oof_prediction"].float(),
        "function_space": top["oof_feature"].float(),
        "feature_space": top["feature_space"],
        "fold_index": top["fold_index"].long(),
        "expert_predictions": predictions,
        "expert_confidences": confidences,
        "expert_corrections": corrections,
        "action_names": ACTION_NAMES,
        "oracle_action_index": targets["oracle_index"],
        "oracle_value": targets["oracle_value"],
        "oracle_gain": targets["oracle_gain"],
        "oracle_cost_margin": targets["cost_margin"],
        "role_config": role_config.__dict__,
        "tail_config": tail_config.__dict__,
        "fold_metadata": fold_rows,
        "provenance": {
            "top_oof_cache": str(top_oof_cache_path),
            "v9_teacher_cache": teacher["cache_path"],
            "v9_teacher_weights": teacher["weights_path"],
            "v9_anchor_teacher": str(
                teacher["teacher_paths"][teacher["anchor_index"]]
            ),
            "is_fully_nested_teacher_stack": False,
            "holdout_label_isolation": True,
            "note": (
                "Fold-local V9.3-architecture specialists exclude each holdout "
                "group's labels. Existing V9 teacher checkpoints/cache are frozen "
                "artifacts and are not regenerated inside every fold."
            ),
        },
    }
    torch.save(payload, final_path)
    pd.DataFrame(histories).to_csv(
        output_dir / "v97_crossfit_specialist_history.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "v97_crossfit_fold_summary.csv", index=False
    )
    diagnostic = pd.DataFrame(
        {
            "sample_id": top_ids,
            "label": top["labels"].view(-1).tolist(),
            "anchor": top["oof_prediction"].view(-1).tolist(),
            "oracle_action": [
                ACTION_NAMES[index]
                for index in targets["oracle_index"].tolist()
            ],
            "oracle_value": targets["oracle_value"].view(-1).tolist(),
            "oracle_gain": targets["oracle_gain"].view(-1).tolist(),
            "oracle_cost_margin": targets["cost_margin"].view(-1).tolist(),
        }
    )
    for index, name in enumerate(SPECIALIST_NAMES):
        diagnostic[f"{name}_prediction"] = predictions[:, index, 0].tolist()
        diagnostic[f"{name}_confidence"] = confidences[:, index, 0].tolist()
    diagnostic.to_csv(
        output_dir / "v97_crossfit_frontier_pool.csv", index=False
    )
    summary = {
        "sample_count": n,
        "group_count": len(set(payload["group_ids"])),
        "fold_count": len(unique_folds),
        "anchor_oof_mae": float(
            torch.abs(top["oof_prediction"] - top["labels"]).mean().item()
        ),
        "sample_oracle_oof_mae": float(
            torch.abs(targets["oracle_value"] - top["labels"]).mean().item()
        ),
        "mean_oracle_gain": float(targets["oracle_gain"].mean().item()),
        "meaningful_gain_rate": float(
            (targets["oracle_gain"].view(-1) > 0.02).float().mean().item()
        ),
        "oracle_action_counts": {
            name: int((targets["oracle_index"] == index).sum().item())
            for index, name in enumerate(ACTION_NAMES)
        },
        "provenance": payload["provenance"],
    }
    (output_dir / "v97_crossfit_frontier_pool_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return payload
