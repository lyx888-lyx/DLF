"""Strict OOF soft-teacher construction and residual Student training for V9.14."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from .expert_analysis import normalize_batch_ids
from .model.SemanticCostCoachV99 import SPECIALIST_NAMES
from .model.StrictOOFExpertStudentV914 import (
    STUDENT_VERSION,
    StrictOOFExpertStudentV914,
    strict_oof_distillation_loss,
)
from .oof_group_splits_v92 import canonical_sample_id

TEACHER_VERSION = "strict_oof_soft_teacher_v914_v1"


@dataclass(frozen=True)
class TeacherConfigV914:
    temperature: float = 0.15
    gain_margin: float = 0.05
    gain_scale: float = 0.15
    max_alpha: float = 0.50


@dataclass(frozen=True)
class StudentConfigV914:
    hidden_dim: int = 192
    dropout: float = 0.15
    residual_max: float = 0.75
    auxiliary_residual_max: float = 1.50
    max_epochs: int = 40
    early_stop: int = 7
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    auxiliary_weight: float = 0.10
    correction_penalty: float = 0.01
    gradient_clip: float = 2.0


def _seed_all(seed: int) -> None:
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
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def build_soft_teacher_v914(
    strict_pool: Mapping[str, object],
    config: TeacherConfigV914,
) -> Dict[str, object]:
    """Construct a privileged but bounded training target from strict OOF outputs."""
    required = {
        "sample_ids",
        "group_ids",
        "labels",
        "anchor",
        "fold_index",
        "expert_predictions",
        "action_names",
        "provenance",
    }
    if not required.issubset(strict_pool):
        raise ValueError(
            f"strict OOF pool missing {sorted(required - set(strict_pool))}"
        )
    provenance = strict_pool["provenance"]
    if provenance.get("is_fully_nested_teacher_stack") is not True:
        raise ValueError("V9.14 requires the fully nested V9.8 shadow pool")
    if provenance.get("holdout_label_isolation") is not True:
        raise ValueError("strict OOF holdout isolation is not guaranteed")
    if provenance.get("historical_full_train_teacher_reuse") is not False:
        raise ValueError("historical full-train experts entered strict OOF targets")

    sample_ids = [
        canonical_sample_id(value) for value in strict_pool["sample_ids"]
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("duplicate strict OOF sample id")
    labels = _column(strict_pool["labels"], "labels")
    anchor = _column(strict_pool["anchor"], "anchor")
    experts = strict_pool["expert_predictions"].detach().cpu().float()
    if experts.dim() != 3 or experts.shape[1:] != (
        len(SPECIALIST_NAMES),
        1,
    ):
        raise ValueError("expert_predictions must have shape [N,4,1]")
    if len(experts) != len(anchor) or len(labels) != len(anchor):
        raise ValueError("strict OOF tensor length mismatch")
    if not torch.isfinite(experts).all():
        raise FloatingPointError("strict OOF experts contain non-finite values")

    temperature = max(float(config.temperature), 1e-6)
    gain_scale = max(float(config.gain_scale), 1e-6)
    expert_errors = torch.abs(experts.squeeze(-1) - labels)
    expert_relevance = torch.softmax(
        -expert_errors / temperature,
        dim=1,
    )
    proposal = (
        expert_relevance.unsqueeze(-1) * experts
    ).sum(dim=1)
    anchor_error = torch.abs(anchor - labels)
    proposal_error = torch.abs(proposal - labels)
    proposal_gain = anchor_error - proposal_error
    alpha = (
        (proposal_gain - float(config.gain_margin)) / gain_scale
    ).clamp(0.0, 1.0)
    alpha = alpha * float(config.max_alpha)
    teacher_prediction = anchor + alpha * (proposal - anchor)
    teacher_correction = teacher_prediction - anchor
    expert_corrections = experts.squeeze(-1) - anchor
    teacher_error = torch.abs(teacher_prediction - labels)
    teacher_gain = anchor_error - teacher_error

    return {
        "version": TEACHER_VERSION,
        "method": "strict_oof_label_privileged_bounded_residual_teacher_v9_14",
        "config": asdict(config),
        "sample_ids": sample_ids,
        "group_ids": list(strict_pool["group_ids"]),
        "fold_index": strict_pool["fold_index"].detach().cpu().long(),
        "labels": labels,
        "anchor": anchor,
        "expert_predictions": experts.squeeze(-1),
        "expert_corrections": expert_corrections,
        "expert_relevance": expert_relevance,
        "proposal": proposal,
        "proposal_gain": proposal_gain,
        "alpha": alpha,
        "teacher_prediction": teacher_prediction,
        "teacher_correction": teacher_correction,
        "teacher_gain": teacher_gain,
        "primary_expert_index": expert_relevance.argmax(dim=1),
        "action_names": tuple(strict_pool["action_names"]),
        "specialist_names": tuple(SPECIALIST_NAMES),
        "diagnostic": {
            "sample_count": len(sample_ids),
            "anchor_oof_mae": float(anchor_error.mean().item()),
            "proposal_oof_mae": float(proposal_error.mean().item()),
            "teacher_oof_mae": float(teacher_error.mean().item()),
            "teacher_gain": float(teacher_gain.mean().item()),
            "active_rate": float((alpha.view(-1) > 0.0).float().mean().item()),
            "mean_alpha": float(alpha.mean().item()),
            "max_alpha": float(alpha.max().item()),
            "harm_over_010_rate": float(
                ((teacher_error - anchor_error).view(-1) > 0.10)
                .float()
                .mean()
                .item()
            ),
            "primary_expert_counts": {
                name: int(
                    (expert_relevance.argmax(dim=1) == index).sum().item()
                )
                for index, name in enumerate(SPECIALIST_NAMES)
            },
        },
        "provenance": {
            "source_pool_version": strict_pool.get("version"),
            "strict_oof_experts": True,
            "each_training_target_excludes_its_sample": True,
            "train_labels_used_to_construct_privileged_target": True,
            "validation_labels_used_to_construct_teacher": False,
            "test_labels_used_to_construct_teacher": False,
        },
    }


def align_teacher_to_dataset(
    teacher: Mapping[str, object],
    dataset,
) -> Dict[str, object]:
    if not hasattr(dataset, "ids"):
        raise AttributeError("MMDataset must expose ids for V9.14 alignment")
    dataset_ids = [canonical_sample_id(value) for value in list(dataset.ids)]
    source_ids = [
        canonical_sample_id(value) for value in teacher["sample_ids"]
    ]
    source_map = {}
    for index, sample_id in enumerate(source_ids):
        if sample_id in source_map:
            raise RuntimeError(f"duplicate teacher id: {sample_id}")
        source_map[sample_id] = index
    missing = [sample_id for sample_id in dataset_ids if sample_id not in source_map]
    extra = sorted(set(source_ids) - set(dataset_ids))
    if missing or extra:
        raise RuntimeError(
            f"Train/teacher ID mismatch missing={len(missing)} extra={len(extra)}"
        )
    order = torch.tensor(
        [source_map[sample_id] for sample_id in dataset_ids],
        dtype=torch.long,
    )
    tensor_keys = (
        "fold_index",
        "labels",
        "anchor",
        "expert_predictions",
        "expert_corrections",
        "expert_relevance",
        "proposal",
        "proposal_gain",
        "alpha",
        "teacher_prediction",
        "teacher_correction",
        "teacher_gain",
        "primary_expert_index",
    )
    aligned = dict(teacher)
    aligned["sample_ids"] = dataset_ids
    aligned["group_ids"] = [
        teacher["group_ids"][int(index)] for index in order.tolist()
    ]
    for key in tensor_keys:
        aligned[key] = teacher[key].index_select(0, order)
    aligned["sample_id_alignment_checked"] = True
    return aligned


def teacher_index_map(teacher: Mapping[str, object]) -> Dict[str, int]:
    result = {}
    for index, value in enumerate(teacher["sample_ids"]):
        sample_id = canonical_sample_id(value)
        if sample_id in result:
            raise RuntimeError(f"duplicate aligned teacher id: {sample_id}")
        result[sample_id] = int(index)
    return result


def _loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
):
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def new_student(
    args,
    anchor_checkpoint: Path,
    config: StudentConfigV914,
) -> StrictOOFExpertStudentV914:
    model = StrictOOFExpertStudentV914(
        args,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        residual_max=config.residual_max,
        auxiliary_residual_max=config.auxiliary_residual_max,
    ).to(args.device)
    model.load_backbone_checkpoint(
        anchor_checkpoint,
        map_location=args.device,
    )
    model.freeze_backbone()
    return model


def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }


def load_trainable_state(model, state: Mapping[str, torch.Tensor]) -> None:
    result = model.load_state_dict(dict(state), strict=False)
    unexpected = list(result.unexpected_keys)
    invalid_missing = [
        key for key in result.missing_keys
        if not key.startswith("backbone.")
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            f"invalid V9.14 student state missing={invalid_missing} "
            f"unexpected={unexpected}"
        )


@torch.no_grad()
def evaluate_student(
    model,
    dataset,
    device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    model.eval()
    sample_ids = []
    labels = []
    anchor = []
    prediction = []
    correction = []
    loader = _loader(dataset, batch_size, num_workers, False, 0)
    for batch in loader:
        output = model(
            batch["text"].to(device),
            batch["audio"].to(device),
            batch["vision"].to(device),
        )
        ids = normalize_batch_ids(batch.get("id"))
        if len(ids) != len(output["prediction"]):
            raise RuntimeError("evaluation ID/prediction length mismatch")
        sample_ids.extend(canonical_sample_id(value) for value in ids)
        labels.append(batch["labels"]["M"].detach().cpu().float().view(-1, 1))
        anchor.append(output["base_prediction"].detach().cpu().float())
        prediction.append(output["prediction"].detach().cpu().float())
        correction.append(output["correction"].detach().cpu().float())
    labels_tensor = torch.cat(labels, dim=0)
    anchor_tensor = torch.cat(anchor, dim=0)
    prediction_tensor = torch.cat(prediction, dim=0)
    correction_tensor = torch.cat(correction, dim=0)
    anchor_error = torch.abs(anchor_tensor - labels_tensor)
    selected_error = torch.abs(prediction_tensor - labels_tensor)
    return {
        "sample_ids": sample_ids,
        "labels": labels_tensor,
        "anchor": anchor_tensor,
        "prediction": prediction_tensor,
        "correction": correction_tensor,
        "anchor_mae": float(anchor_error.mean().item()),
        "mae": float(selected_error.mean().item()),
        "gain_vs_anchor": float(
            (anchor_error.mean() - selected_error.mean()).item()
        ),
        "harm_over_010_rate": float(
            ((selected_error - anchor_error).view(-1) > 0.10)
            .float()
            .mean()
            .item()
        ),
    }


def _checkpoint_payload(
    model,
    anchor_checkpoint: Path,
    student_config: StudentConfigV914,
    teacher_config: Mapping[str, object],
    distill_weight: float,
    epoch: int,
    validation: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "version": STUDENT_VERSION,
        "student_state_dict": trainable_state_dict(model),
        "anchor_checkpoint": str(anchor_checkpoint),
        "student_config": asdict(student_config),
        "teacher_config": dict(teacher_config),
        "distill_weight": float(distill_weight),
        "epoch": int(epoch),
        "validation_anchor_mae": float(validation["anchor_mae"]),
        "validation_mae": float(validation["mae"]),
        "validation_gain": float(validation["gain_vs_anchor"]),
    }


def train_student_variant(
    args,
    train_dataset,
    valid_dataset,
    teacher: Mapping[str, object],
    anchor_checkpoint: Path,
    output_path: Path,
    student_config: StudentConfigV914,
    distill_weight: float,
    seed: int,
    num_workers: int,
) -> Dict[str, object]:
    """Train one low-capacity Student variant and early-stop on Validation."""
    _seed_all(seed)
    model = new_student(args, anchor_checkpoint, student_config)
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=float(student_config.learning_rate),
        weight_decay=float(student_config.weight_decay),
    )
    index_map = teacher_index_map(teacher)
    loader = _loader(
        train_dataset,
        student_config.batch_size,
        num_workers,
        True,
        seed,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial = evaluate_student(
        model,
        valid_dataset,
        args.device,
        student_config.batch_size,
        num_workers,
    )
    best_mae = float(initial["mae"])
    best_epoch = 0
    stale = 0
    torch.save(
        _checkpoint_payload(
            model,
            anchor_checkpoint,
            student_config,
            teacher["config"],
            distill_weight,
            0,
            initial,
        ),
        output_path,
    )
    history = [
        {
            "distill_weight": float(distill_weight),
            "epoch": 0,
            "validation_anchor_mae": initial["anchor_mae"],
            "validation_mae": initial["mae"],
            "validation_gain": initial["gain_vs_anchor"],
            "validation_harm_over_010_rate": initial[
                "harm_over_010_rate"
            ],
        }
    ]

    for epoch in range(1, int(student_config.max_epochs) + 1):
        model.set_train_mode()
        totals: Dict[str, float] = {}
        for batch in loader:
            ids = normalize_batch_ids(batch.get("id"))
            try:
                indices = torch.tensor(
                    [
                        index_map[canonical_sample_id(value)]
                        for value in ids
                    ],
                    dtype=torch.long,
                )
            except KeyError as error:
                raise KeyError(
                    f"V9.14 training sample missing from strict OOF teacher: {error}"
                )
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)
            cached_labels = teacher["labels"].index_select(0, indices)
            if not torch.allclose(
                cached_labels.float(),
                labels.detach().cpu().float(),
                atol=1e-5,
                rtol=0.0,
            ):
                raise RuntimeError("V9.14 teacher/train labels are misaligned")
            output = model(
                batch["text"].to(args.device),
                batch["audio"].to(args.device),
                batch["vision"].to(args.device),
            )
            losses = strict_oof_distillation_loss(
                output,
                labels,
                teacher["teacher_correction"].index_select(0, indices),
                teacher["expert_corrections"].index_select(0, indices),
                teacher["expert_relevance"].index_select(0, indices),
                distill_weight=float(distill_weight),
                auxiliary_weight=float(student_config.auxiliary_weight),
                correction_penalty=float(student_config.correction_penalty),
            )
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                model.trainable_parameters(),
                float(student_config.gradient_clip),
            )
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(
                    value.detach().item()
                )

        validation = evaluate_student(
            model,
            valid_dataset,
            args.device,
            student_config.batch_size,
            num_workers,
        )
        history.append(
            {
                "distill_weight": float(distill_weight),
                "epoch": int(epoch),
                **{
                    f"train_{key}": value / max(1, len(loader))
                    for key, value in totals.items()
                },
                "validation_anchor_mae": validation["anchor_mae"],
                "validation_mae": validation["mae"],
                "validation_gain": validation["gain_vs_anchor"],
                "validation_harm_over_010_rate": validation[
                    "harm_over_010_rate"
                ],
            }
        )
        if float(validation["mae"]) < best_mae - 1e-7:
            best_mae = float(validation["mae"])
            best_epoch = int(epoch)
            stale = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    anchor_checkpoint,
                    student_config,
                    teacher["config"],
                    distill_weight,
                    epoch,
                    validation,
                ),
                output_path,
            )
        else:
            stale += 1
            if stale >= int(student_config.early_stop):
                break

    payload = torch.load(output_path, map_location="cpu")
    load_trainable_state(model, payload["student_state_dict"])
    selected = evaluate_student(
        model,
        valid_dataset,
        args.device,
        student_config.batch_size,
        num_workers,
    )
    return {
        "candidate_id": f"student__distill_{float(distill_weight):.4f}",
        "distill_weight": float(distill_weight),
        "checkpoint": str(output_path),
        "best_epoch": int(best_epoch),
        "validation_anchor_mae": selected["anchor_mae"],
        "validation_mae": selected["mae"],
        "validation_gain": selected["gain_vs_anchor"],
        "validation_harm_over_010_rate": selected[
            "harm_over_010_rate"
        ],
        "history": history,
    }


def load_student_checkpoint(
    args,
    checkpoint_path: Path,
    anchor_checkpoint: Path,
    student_config: StudentConfigV914,
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("version") != STUDENT_VERSION:
        raise ValueError("unexpected V9.14 student checkpoint version")
    model = new_student(args, anchor_checkpoint, student_config)
    load_trainable_state(model, payload["student_state_dict"])
    model.eval()
    return model, payload
