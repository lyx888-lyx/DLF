"""Direct deployment-scale strict-OOF Student training for V9.16."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from .expert_analysis import normalize_batch_ids
from .model.SmallScaleExpertStudentV916 import (
    STUDENT_VERSION,
    SmallScaleExpertStudentV916,
    small_scale_absolute_distillation_loss,
)
from .oof_group_splits_v92 import canonical_sample_id
from .strict_oof_expert_distillation_v914 import (
    TEACHER_VERSION,
    TeacherConfigV914,
    align_teacher_to_dataset,
    build_soft_teacher_v914,
    teacher_index_map,
)


@dataclass(frozen=True)
class StudentConfigV916:
    hidden_dim: int = 64
    dropout: float = 0.10
    residual_max: float = 0.03
    auxiliary_residual_max: float = 0.15
    max_epochs: int = 60
    early_stop: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 1e-3
    batch_size: int = 32
    auxiliary_weight: float = 0.02
    correction_penalty: float = 0.01
    gradient_clip: float = 1.0


def config_with_scale(
    config: StudentConfigV916,
    residual_max: float,
) -> StudentConfigV916:
    return replace(config, residual_max=float(residual_max))


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
    config: StudentConfigV916,
) -> SmallScaleExpertStudentV916:
    model = SmallScaleExpertStudentV916(
        args,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        residual_max=config.residual_max,
        auxiliary_residual_max=config.auxiliary_residual_max,
    ).to(args.device)
    model.load_backbone_checkpoint(anchor_checkpoint, map_location=args.device)
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
            f"invalid V9.16 student state missing={invalid_missing} "
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
    if not torch.allclose(
        prediction_tensor,
        anchor_tensor + correction_tensor,
        atol=1e-6,
    ):
        raise RuntimeError("V9.16 prediction is not anchor plus correction")
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
        "mean_abs_correction": float(correction_tensor.abs().mean().item()),
        "max_abs_correction": float(correction_tensor.abs().max().item()),
        "saturation_rate": float(
            (
                correction_tensor.abs()
                >= float(model.residual_max) - 1e-6
            )
            .float()
            .mean()
            .item()
        ),
    }


def _checkpoint_payload(
    model,
    anchor_checkpoint: Path,
    student_config: StudentConfigV916,
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
        "deployment_beta": 1.0,
        "validation_anchor_mae": float(validation["anchor_mae"]),
        "validation_mae": float(validation["mae"]),
        "validation_gain": float(validation["gain_vs_anchor"]),
        "validation_harm_over_010_rate": float(
            validation["harm_over_010_rate"]
        ),
        "validation_mean_abs_correction": float(
            validation["mean_abs_correction"]
        ),
        "validation_max_abs_correction": float(
            validation["max_abs_correction"]
        ),
        "validation_saturation_rate": float(validation["saturation_rate"]),
    }


def train_student_variant(
    args,
    train_dataset,
    valid_dataset,
    teacher: Mapping[str, object],
    anchor_checkpoint: Path,
    output_path: Path,
    student_config: StudentConfigV916,
    distill_weight: float,
    seed: int,
    num_workers: int,
) -> Dict[str, object]:
    """Train one fixed-scale Student; Validation selects epoch only."""
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
    best_score = (
        float(initial["mae"]),
        float(initial["harm_over_010_rate"]),
    )
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
            "residual_max": float(student_config.residual_max),
            "distill_weight": float(distill_weight),
            "epoch": 0,
            "deployment_beta": 1.0,
            "validation_anchor_mae": initial["anchor_mae"],
            "validation_mae": initial["mae"],
            "validation_gain": initial["gain_vs_anchor"],
            "validation_harm_over_010_rate": initial[
                "harm_over_010_rate"
            ],
            "validation_mean_abs_correction": initial[
                "mean_abs_correction"
            ],
            "validation_max_abs_correction": initial[
                "max_abs_correction"
            ],
            "validation_saturation_rate": initial["saturation_rate"],
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
                    f"V9.16 training sample missing from strict OOF teacher: {error}"
                )
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)
            cached_labels = teacher["labels"].index_select(0, indices)
            if not torch.allclose(
                cached_labels.float(),
                labels.detach().cpu().float(),
                atol=1e-5,
                rtol=0.0,
            ):
                raise RuntimeError("V9.16 teacher/train labels are misaligned")

            output = model(
                batch["text"].to(args.device),
                batch["audio"].to(args.device),
                batch["vision"].to(args.device),
            )
            losses = small_scale_absolute_distillation_loss(
                output,
                labels,
                teacher["teacher_prediction"].index_select(0, indices),
                teacher["expert_predictions"].index_select(0, indices),
                teacher["expert_relevance"].index_select(0, indices),
                distill_weight=float(distill_weight),
                auxiliary_weight=float(student_config.auxiliary_weight),
                residual_max=float(student_config.residual_max),
                auxiliary_residual_max=float(
                    student_config.auxiliary_residual_max
                ),
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
                "residual_max": float(student_config.residual_max),
                "distill_weight": float(distill_weight),
                "epoch": int(epoch),
                "deployment_beta": 1.0,
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
                "validation_mean_abs_correction": validation[
                    "mean_abs_correction"
                ],
                "validation_max_abs_correction": validation[
                    "max_abs_correction"
                ],
                "validation_saturation_rate": validation[
                    "saturation_rate"
                ],
            }
        )
        score = (
            float(validation["mae"]),
            float(validation["harm_over_010_rate"]),
        )
        if score < best_score:
            best_score = score
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
        "candidate_id": (
            f"student__scale_{float(student_config.residual_max):.3f}"
            f"__distill_{float(distill_weight):.4f}"
        ),
        "residual_max": float(student_config.residual_max),
        "distill_weight": float(distill_weight),
        "deployment_beta": 1.0,
        "checkpoint": str(output_path),
        "best_epoch": int(best_epoch),
        "validation_anchor_mae": selected["anchor_mae"],
        "validation_mae": selected["mae"],
        "validation_gain": selected["gain_vs_anchor"],
        "validation_harm_over_010_rate": selected[
            "harm_over_010_rate"
        ],
        "validation_mean_abs_correction": selected[
            "mean_abs_correction"
        ],
        "validation_max_abs_correction": selected[
            "max_abs_correction"
        ],
        "validation_saturation_rate": selected["saturation_rate"],
        "history": history,
    }


def load_student_checkpoint(
    args,
    checkpoint_path: Path,
    anchor_checkpoint: Path,
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("version") != STUDENT_VERSION:
        raise ValueError("unexpected V9.16 student checkpoint version")
    config = StudentConfigV916(**payload["student_config"])
    model = new_student(args, anchor_checkpoint, config)
    load_trainable_state(model, payload["student_state_dict"])
    model.eval()
    return model, payload, config


__all__ = [
    "STUDENT_VERSION",
    "TEACHER_VERSION",
    "TeacherConfigV914",
    "StudentConfigV916",
    "align_teacher_to_dataset",
    "build_soft_teacher_v914",
    "config_with_scale",
    "new_student",
    "trainable_state_dict",
    "evaluate_student",
    "train_student_variant",
    "load_student_checkpoint",
]
