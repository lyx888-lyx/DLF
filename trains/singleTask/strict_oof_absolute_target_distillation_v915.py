"""Absolute-target strict-OOF Student training and beta calibration for V9.15."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from .expert_analysis import normalize_batch_ids
from .model.AbsoluteTargetExpertStudentV915 import (
    STUDENT_VERSION,
    AbsoluteTargetExpertStudentV915,
    absolute_target_distillation_loss,
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
class StudentConfigV915:
    hidden_dim: int = 64
    dropout: float = 0.10
    residual_max: float = 0.15
    auxiliary_residual_max: float = 1.50
    max_epochs: int = 60
    early_stop: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 1e-3
    batch_size: int = 32
    auxiliary_weight: float = 0.05
    correction_penalty: float = 0.10
    gradient_clip: float = 1.0


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
    config: StudentConfigV915,
) -> AbsoluteTargetExpertStudentV915:
    model = AbsoluteTargetExpertStudentV915(
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
            f"invalid V9.15 student state missing={invalid_missing} "
            f"unexpected={unexpected}"
        )


@torch.no_grad()
def collect_student_outputs(
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
    raw_prediction = []
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
        raw_prediction.append(output["prediction"].detach().cpu().float())
        correction.append(output["correction"].detach().cpu().float())
    labels_tensor = torch.cat(labels, dim=0)
    anchor_tensor = torch.cat(anchor, dim=0)
    correction_tensor = torch.cat(correction, dim=0)
    raw_tensor = torch.cat(raw_prediction, dim=0)
    if not torch.allclose(raw_tensor, anchor_tensor + correction_tensor, atol=1e-6):
        raise RuntimeError("raw Student prediction is not anchor plus correction")
    return {
        "sample_ids": sample_ids,
        "labels": labels_tensor,
        "anchor": anchor_tensor,
        "raw_prediction": raw_tensor,
        "correction": correction_tensor,
    }


def metrics_from_prediction(
    prediction: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    anchor_error = torch.abs(anchor - labels)
    selected_error = torch.abs(prediction - labels)
    return {
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


def apply_beta(outputs: Mapping[str, object], beta: float) -> Dict[str, object]:
    beta = float(beta)
    prediction = outputs["anchor"] + beta * outputs["correction"]
    metrics = metrics_from_prediction(
        prediction,
        outputs["anchor"],
        outputs["labels"],
    )
    return {
        **metrics,
        "beta": beta,
        "prediction": prediction,
    }


def calibrate_beta(
    outputs: Mapping[str, object],
    beta_grid: Sequence[float],
) -> Dict[str, object]:
    candidates = [apply_beta(outputs, float(beta)) for beta in beta_grid]
    if not candidates:
        raise ValueError("beta grid cannot be empty")
    return min(
        candidates,
        key=lambda row: (
            float(row["mae"]),
            float(row["harm_over_010_rate"]),
            float(row["beta"]),
        ),
    )


def evaluate_student(
    model,
    dataset,
    device,
    batch_size: int,
    num_workers: int,
    beta: float,
) -> Dict[str, object]:
    outputs = collect_student_outputs(
        model,
        dataset,
        device,
        batch_size,
        num_workers,
    )
    calibrated = apply_beta(outputs, beta)
    return {
        **outputs,
        "prediction": calibrated["prediction"],
        "beta": float(beta),
        "anchor_mae": calibrated["anchor_mae"],
        "mae": calibrated["mae"],
        "gain_vs_anchor": calibrated["gain_vs_anchor"],
        "harm_over_010_rate": calibrated["harm_over_010_rate"],
    }


def _checkpoint_payload(
    model,
    anchor_checkpoint: Path,
    student_config: StudentConfigV915,
    teacher_config: Mapping[str, object],
    distill_weight: float,
    epoch: int,
    beta: float,
    raw_validation: Mapping[str, object],
    calibrated_validation: Mapping[str, object],
) -> Dict[str, object]:
    raw_metrics = metrics_from_prediction(
        raw_validation["raw_prediction"],
        raw_validation["anchor"],
        raw_validation["labels"],
    )
    return {
        "version": STUDENT_VERSION,
        "student_state_dict": trainable_state_dict(model),
        "anchor_checkpoint": str(anchor_checkpoint),
        "student_config": asdict(student_config),
        "teacher_config": dict(teacher_config),
        "distill_weight": float(distill_weight),
        "epoch": int(epoch),
        "selected_beta": float(beta),
        "validation_anchor_mae": float(calibrated_validation["anchor_mae"]),
        "validation_raw_mae": float(raw_metrics["mae"]),
        "validation_mae": float(calibrated_validation["mae"]),
        "validation_gain": float(calibrated_validation["gain_vs_anchor"]),
        "validation_harm_over_010_rate": float(
            calibrated_validation["harm_over_010_rate"]
        ),
    }


def train_student_variant(
    args,
    train_dataset,
    valid_dataset,
    teacher: Mapping[str, object],
    anchor_checkpoint: Path,
    output_path: Path,
    student_config: StudentConfigV915,
    distill_weight: float,
    beta_grid: Sequence[float],
    seed: int,
    num_workers: int,
) -> Dict[str, object]:
    """Train one conservative Student and select epoch/beta on Validation."""
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

    initial_outputs = collect_student_outputs(
        model,
        valid_dataset,
        args.device,
        student_config.batch_size,
        num_workers,
    )
    initial_calibrated = calibrate_beta(initial_outputs, beta_grid)
    best_score = (
        float(initial_calibrated["mae"]),
        float(initial_calibrated["harm_over_010_rate"]),
        float(initial_calibrated["beta"]),
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
            initial_calibrated["beta"],
            initial_outputs,
            initial_calibrated,
        ),
        output_path,
    )
    initial_raw = metrics_from_prediction(
        initial_outputs["raw_prediction"],
        initial_outputs["anchor"],
        initial_outputs["labels"],
    )
    history = [
        {
            "distill_weight": float(distill_weight),
            "epoch": 0,
            "selected_beta": float(initial_calibrated["beta"]),
            "validation_anchor_mae": initial_calibrated["anchor_mae"],
            "validation_raw_mae": initial_raw["mae"],
            "validation_mae": initial_calibrated["mae"],
            "validation_gain": initial_calibrated["gain_vs_anchor"],
            "validation_harm_over_010_rate": initial_calibrated[
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
                    f"V9.15 training sample missing from strict OOF teacher: {error}"
                )
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)
            cached_labels = teacher["labels"].index_select(0, indices)
            if not torch.allclose(
                cached_labels.float(),
                labels.detach().cpu().float(),
                atol=1e-5,
                rtol=0.0,
            ):
                raise RuntimeError("V9.15 teacher/train labels are misaligned")
            output = model(
                batch["text"].to(args.device),
                batch["audio"].to(args.device),
                batch["vision"].to(args.device),
            )
            losses = absolute_target_distillation_loss(
                output,
                labels,
                teacher["teacher_prediction"].index_select(0, indices),
                teacher["expert_predictions"].index_select(0, indices),
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

        validation_outputs = collect_student_outputs(
            model,
            valid_dataset,
            args.device,
            student_config.batch_size,
            num_workers,
        )
        calibrated = calibrate_beta(validation_outputs, beta_grid)
        raw_metrics = metrics_from_prediction(
            validation_outputs["raw_prediction"],
            validation_outputs["anchor"],
            validation_outputs["labels"],
        )
        history.append(
            {
                "distill_weight": float(distill_weight),
                "epoch": int(epoch),
                "selected_beta": float(calibrated["beta"]),
                **{
                    f"train_{key}": value / max(1, len(loader))
                    for key, value in totals.items()
                },
                "validation_anchor_mae": calibrated["anchor_mae"],
                "validation_raw_mae": raw_metrics["mae"],
                "validation_mae": calibrated["mae"],
                "validation_gain": calibrated["gain_vs_anchor"],
                "validation_harm_over_010_rate": calibrated[
                    "harm_over_010_rate"
                ],
            }
        )
        score = (
            float(calibrated["mae"]),
            float(calibrated["harm_over_010_rate"]),
            float(calibrated["beta"]),
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
                    calibrated["beta"],
                    validation_outputs,
                    calibrated,
                ),
                output_path,
            )
        else:
            stale += 1
            if stale >= int(student_config.early_stop):
                break

    payload = torch.load(output_path, map_location="cpu")
    load_trainable_state(model, payload["student_state_dict"])
    selected_outputs = collect_student_outputs(
        model,
        valid_dataset,
        args.device,
        student_config.batch_size,
        num_workers,
    )
    selected = apply_beta(selected_outputs, float(payload["selected_beta"]))
    raw_metrics = metrics_from_prediction(
        selected_outputs["raw_prediction"],
        selected_outputs["anchor"],
        selected_outputs["labels"],
    )
    return {
        "candidate_id": f"student__absolute_distill_{float(distill_weight):.4f}",
        "distill_weight": float(distill_weight),
        "checkpoint": str(output_path),
        "best_epoch": int(best_epoch),
        "selected_beta": float(payload["selected_beta"]),
        "validation_anchor_mae": selected["anchor_mae"],
        "validation_raw_mae": raw_metrics["mae"],
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
    student_config: StudentConfigV915,
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("version") != STUDENT_VERSION:
        raise ValueError("unexpected V9.15 student checkpoint version")
    model = new_student(args, anchor_checkpoint, student_config)
    load_trainable_state(model, payload["student_state_dict"])
    model.eval()
    return model, payload


__all__ = [
    "STUDENT_VERSION",
    "TEACHER_VERSION",
    "TeacherConfigV914",
    "StudentConfigV915",
    "align_teacher_to_dataset",
    "build_soft_teacher_v914",
    "new_student",
    "trainable_state_dict",
    "collect_student_outputs",
    "metrics_from_prediction",
    "apply_beta",
    "calibrate_beta",
    "evaluate_student",
    "train_student_variant",
    "load_student_checkpoint",
]
