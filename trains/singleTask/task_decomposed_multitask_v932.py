"""V9.32 task-decomposed multitask DLF.

The deployed prediction is always the original DLF regression output. Intensity
and ordinal branches are training-only auxiliary tasks: every sample follows the
same path at train and test time, and no label-defined expert, winner router,
scalar residual correction, or hard task-output composition is used.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .HingeLoss import HingeLoss
from .cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _batch_to_device,
    _load_state,
    compute_full_dlf_loss,
    mode_to_mask,
)
from .model.FSC_DLF import _FusionFeatureCapture
from .oof_group_splits_v92 import canonical_sample_id

logger = logging.getLogger("MMSA")

VERSION = "task_decomposed_multitask_v932_v1"
ORDINAL_THRESHOLDS = (-2.0, -1.0, 0.0, 1.0, 2.0)
VARIANT_NAMES = (
    "regression_only",
    "ordinal_only",
    "intensity_only",
    "ordinal_intensity",
)


@dataclass(frozen=True)
class TaskDecomposedConfigV932:
    batch_size: int = 128
    num_workers: int = 2
    max_epochs: int = 12
    early_stop: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    update_epochs: int = 1
    use_amp: bool = True
    trainable_scope: str = "fusion_tail"
    auxiliary_hidden_dim: int = 128
    auxiliary_dropout: float = 0.15
    ordinal_weight: float = 0.20
    intensity_weight: float = 0.10
    ordinal_monotonic_weight: float = 0.05
    intensity_max: float = 3.0
    min_delta: float = 1e-6
    required_gain_vs_control: float = 0.005
    required_nondegrading_folds: int = 4
    max_worst_fold_degradation: float = 0.005

    def validate(self) -> None:
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("invalid loader configuration")
        if self.max_epochs < 1 or self.early_stop < 1:
            raise ValueError("invalid epoch configuration")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.update_epochs < 1:
            raise ValueError("update_epochs must be positive")
        if self.trainable_scope not in {"fusion_tail", "full"}:
            raise ValueError("trainable_scope must be fusion_tail or full")
        if self.auxiliary_hidden_dim < 4:
            raise ValueError("auxiliary_hidden_dim is too small")
        if not 0 <= self.auxiliary_dropout < 1:
            raise ValueError("auxiliary_dropout must be in [0, 1)")
        if min(
            self.ordinal_weight,
            self.intensity_weight,
            self.ordinal_monotonic_weight,
        ) < 0:
            raise ValueError("auxiliary loss weights must be nonnegative")
        if self.intensity_max <= 0:
            raise ValueError("intensity_max must be positive")
        if not 1 <= self.required_nondegrading_folds <= 5:
            raise ValueError("required_nondegrading_folds must be in [1, 5]")


def variant_loss_weights(
    variant: str, config: TaskDecomposedConfigV932
) -> Dict[str, float]:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unsupported V9.32 variant: {variant}")
    return {
        "ordinal": (
            float(config.ordinal_weight)
            if variant in {"ordinal_only", "ordinal_intensity"}
            else 0.0
        ),
        "intensity": (
            float(config.intensity_weight)
            if variant in {"intensity_only", "ordinal_intensity"}
            else 0.0
        ),
    }


def ordinal_targets(
    labels: torch.Tensor,
    thresholds: Sequence[float] = ORDINAL_THRESHOLDS,
) -> torch.Tensor:
    values = labels.view(-1, 1)
    cuts = torch.as_tensor(
        tuple(float(value) for value in thresholds),
        dtype=values.dtype,
        device=values.device,
    ).view(1, -1)
    return (values > cuts).to(values.dtype)


def ordinal_monotonic_penalty(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.size(1) < 2:
        raise ValueError("ordinal logits must have shape [batch, thresholds]")
    # P(y > t) must not increase as t increases.
    return F.relu(logits[:, 1:] - logits[:, :-1]).mean()


def auxiliary_losses(
    intensity_prediction: torch.Tensor,
    ordinal_logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    intensity_target = labels.abs().view_as(intensity_prediction)
    target = ordinal_targets(labels)
    intensity = F.smooth_l1_loss(intensity_prediction, intensity_target)
    ordinal = F.binary_cross_entropy_with_logits(ordinal_logits, target)
    monotonic = ordinal_monotonic_penalty(ordinal_logits)
    return {
        "intensity": intensity,
        "ordinal": ordinal,
        "ordinal_monotonic": monotonic,
    }


class TaskDecomposedDLFV932(nn.Module):
    """CFCompat DLF with training-only intensity and ordinal branches."""

    def __init__(
        self,
        args,
        hidden_dim: int = 128,
        dropout: float = 0.15,
        intensity_max: float = 3.0,
        thresholds: Sequence[float] = ORDINAL_THRESHOLDS,
    ):
        super().__init__()
        self.backbone = MissingModalityWrapper(
            DLF(args),
            int(args.feature_dims[1]),
            int(args.feature_dims[2]),
        )
        feature_dim = int(self.backbone.backbone.out_layer.in_features)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.intensity_max = float(intensity_max)
        self.auxiliary_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.intensity_head = nn.Linear(int(hidden_dim), 1)
        self.ordinal_head = nn.Linear(int(hidden_dim), len(self.thresholds))
        self._trainable_scope = "full"

    def load_anchor_checkpoint(self, path: Path, map_location=None) -> None:
        state = _load_state(Path(path), map_location)
        self.backbone.load_state_dict(state, strict=True)

    def configure_trainable(self, scope: str) -> int:
        if scope not in {"fusion_tail", "full"}:
            raise ValueError(f"unsupported trainable scope: {scope}")
        self._trainable_scope = scope
        for parameter in self.parameters():
            parameter.requires_grad_(scope == "full")

        if scope == "fusion_tail":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            modules = (
                self.backbone.backbone.projector_l,
                self.backbone.backbone.projector_a,
                self.backbone.backbone.projector_v,
                self.backbone.backbone.projector_c,
                self.backbone.backbone.proj1,
                self.backbone.backbone.proj2,
                self.backbone.backbone.out_layer,
                self.auxiliary_adapter,
                self.intensity_head,
                self.ordinal_head,
            )
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

        return int(
            sum(parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad)
        )

    def set_train_mode(self) -> None:
        self.train()
        if self._trainable_scope == "fusion_tail":
            # Frozen Transformer/BERT dropout remains deterministic. Linear tail
            # modules still receive gradients in eval mode.
            self.backbone.eval()
            self.auxiliary_adapter.train()
            self.intensity_head.train()
            self.ordinal_head.train()

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        batch_size = int(audio.size(0))
        mask = mode_to_mask(
            "LAV", batch_size, audio.device, audio.dtype
        )
        capture = _FusionFeatureCapture(self.backbone.backbone)
        try:
            backbone_output = self.backbone(text, audio, vision, mask)
            if capture.value is None:
                raise RuntimeError("V9.32 fusion feature was not captured")
            feature = capture.value
        finally:
            capture.close()

        latent = self.auxiliary_adapter(feature)
        intensity = self.intensity_max * torch.sigmoid(
            self.intensity_head(latent)
        )
        ordinal_logits = self.ordinal_head(latent)
        return {
            "backbone": backbone_output,
            "feature": feature,
            "auxiliary_latent": latent,
            "regression": backbone_output["output_logit"],
            "intensity": intensity,
            "ordinal_logits": ordinal_logits,
        }


def multitask_loss(
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    variant: str,
    config: TaskDecomposedConfigV932,
    criterion: nn.Module,
    cosine: nn.Module,
    hinge: nn.Module,
) -> Dict[str, torch.Tensor]:
    weights = variant_loss_weights(variant, config)
    base = compute_full_dlf_loss(
        output["backbone"], labels, criterion, cosine, hinge
    )
    aux = auxiliary_losses(
        output["intensity"], output["ordinal_logits"], labels
    )
    total = (
        base
        + weights["intensity"] * aux["intensity"]
        + weights["ordinal"]
        * (
            aux["ordinal"]
            + float(config.ordinal_monotonic_weight)
            * aux["ordinal_monotonic"]
        )
    )
    return {
        "total": total,
        "base": base,
        "regression_mae": F.l1_loss(output["regression"], labels),
        **aux,
    }


@torch.no_grad()
def predict_task_model(model, loader, device) -> Dict[str, object]:
    model.eval()
    sample_indices = []
    sample_ids = []
    group_labels = []
    regression = []
    intensity = []
    ordinal_logits = []

    for batch in loader:
        text, audio, vision, labels = _batch_to_device(batch, device)
        output = model(text, audio, vision)
        sample_indices.extend(
            int(value) for value in batch["index"].view(-1).cpu().tolist()
        )
        sample_ids.extend(
            canonical_sample_id(value) for value in list(batch["id"])
        )
        group_labels.append(labels.detach().cpu())
        regression.append(output["regression"].detach().cpu())
        intensity.append(output["intensity"].detach().cpu())
        ordinal_logits.append(output["ordinal_logits"].detach().cpu())

    if not sample_indices:
        raise RuntimeError("V9.32 prediction loader is empty")
    labels = torch.cat(group_labels, dim=0)
    prediction = torch.cat(regression, dim=0)
    intensity_prediction = torch.cat(intensity, dim=0)
    logits = torch.cat(ordinal_logits, dim=0)
    probabilities = torch.sigmoid(logits)
    targets = ordinal_targets(labels)
    return {
        "sample_indices": sample_indices,
        "sample_ids": sample_ids,
        "labels": labels,
        "regression": prediction,
        "intensity": intensity_prediction,
        "ordinal_logits": logits,
        "ordinal_probabilities": probabilities,
        "ordinal_targets": targets,
    }


def prediction_diagnostics(payload: Mapping[str, object]) -> Dict[str, float]:
    labels = torch.as_tensor(payload["labels"]).view(-1, 1).float()
    regression = torch.as_tensor(payload["regression"]).view(-1, 1).float()
    intensity = torch.as_tensor(payload["intensity"]).view(-1, 1).float()
    logits = torch.as_tensor(payload["ordinal_logits"]).float()
    targets = ordinal_targets(labels)
    probabilities = torch.sigmoid(logits)
    violations = (
        probabilities[:, 1:] > probabilities[:, :-1] + 1e-7
    ).float()
    return {
        "regression_mae": float(
            torch.abs(regression - labels).mean().item()
        ),
        "intensity_mae": float(
            torch.abs(intensity - labels.abs()).mean().item()
        ),
        "ordinal_bce": float(
            F.binary_cross_entropy_with_logits(logits, targets).item()
        ),
        "ordinal_accuracy": float(
            ((probabilities >= 0.5) == (targets >= 0.5))
            .float()
            .mean()
            .item()
        ),
        "ordinal_monotonic_violation_rate": float(
            violations.mean().item()
        ),
    }


def _checkpoint_is_reusable(
    path: Path,
    source_manifest: Mapping[str, object],
    variant: str,
    config: TaskDecomposedConfigV932,
) -> bool:
    if not Path(path).is_file():
        return False
    payload = torch.load(path, map_location="cpu")
    return (
        payload.get("version") == VERSION
        and payload.get("variant") == variant
        and payload.get("source_manifest") == dict(source_manifest)
        and payload.get("config") == asdict(config)
    )


def train_task_model(
    args,
    train_loader,
    valid_loader,
    anchor_checkpoint: Path,
    variant: str,
    config: TaskDecomposedConfigV932,
    checkpoint: Path,
    source_manifest: Mapping[str, object],
    seed: int,
    resume: bool = True,
):
    """Fine-tune one pre-registered variant and early-stop on regression MAE."""
    config.validate()
    variant_loss_weights(variant, config)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model = TaskDecomposedDLFV932(
        args,
        hidden_dim=config.auxiliary_hidden_dim,
        dropout=config.auxiliary_dropout,
        intensity_max=config.intensity_max,
    ).to(args.device)
    model.load_anchor_checkpoint(anchor_checkpoint, args.device)
    trainable_parameters = model.configure_trainable(config.trainable_scope)

    if resume and _checkpoint_is_reusable(
        checkpoint, source_manifest, variant, config
    ):
        cached = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(cached["state_dict"], strict=True)
        return {
            "model": model,
            "history": cached.get("history", []),
            "best_epoch": int(cached["best_epoch"]),
            "best_valid_mae": float(cached["best_valid_mae"]),
            "trainable_parameters": int(
                cached.get("trainable_parameters", trainable_parameters)
            ),
            "reused": True,
        }

    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, config.early_stop // 2)
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    amp_enabled = bool(
        config.use_amp
        and torch.cuda.is_available()
        and str(args.device).startswith("cuda")
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_valid_mae = float("inf")
    best_epoch = 0
    history = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, int(config.max_epochs) + 1):
        model.set_train_mode()
        epoch_values: Dict[str, list[float]] = {
            "total": [],
            "base": [],
            "regression_mae": [],
            "intensity": [],
            "ordinal": [],
            "ordinal_monotonic": [],
        }
        for step, batch in enumerate(train_loader, start=1):
            text, audio, vision, labels = _batch_to_device(
                batch, args.device
            )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(text, audio, vision)
                losses = multitask_loss(
                    output,
                    labels,
                    variant,
                    config,
                    criterion,
                    cosine,
                    hinge,
                )
                scaled_loss = losses["total"] / max(
                    1, int(config.update_epochs)
                )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    f"non-finite V9.32 loss for {variant}"
                )
            scaler.scale(scaled_loss).backward()
            should_step = (
                step % max(1, int(config.update_epochs)) == 0
                or step == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                if float(config.grad_clip_norm) > 0:
                    nn.utils.clip_grad_norm_(
                        parameters, float(config.grad_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for name in epoch_values:
                epoch_values[name].append(
                    float(losses[name].detach().item())
                )

        valid_payload = predict_task_model(model, valid_loader, args.device)
        valid_metrics = prediction_diagnostics(valid_payload)
        valid_mae = float(valid_metrics["regression_mae"])
        scheduler.step(valid_mae)
        row = {
            "epoch": int(epoch),
            **{
                f"train_{name}": float(np.mean(values))
                for name, values in epoch_values.items()
            },
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        logger.info(
            "V9.32 variant=%s epoch=%d valid_mae=%.6f",
            variant,
            epoch,
            valid_mae,
        )

        if valid_mae < best_valid_mae - float(config.min_delta):
            best_valid_mae = valid_mae
            best_epoch = epoch
            torch.save(
                {
                    "version": VERSION,
                    "variant": variant,
                    "config": asdict(config),
                    "thresholds": list(ORDINAL_THRESHOLDS),
                    "source_manifest": dict(source_manifest),
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "history": history,
                    "best_epoch": int(best_epoch),
                    "best_valid_mae": float(best_valid_mae),
                    "trainable_parameters": int(trainable_parameters),
                    "deployment_output": "backbone_regression_head",
                    "auxiliary_outputs_used_at_inference": False,
                    "winner_router_present": False,
                    "scalar_residual_correction_present": False,
                },
                checkpoint,
            )
        if epoch - best_epoch >= int(config.early_stop):
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"V9.32 did not save checkpoint: {checkpoint}")
    best = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(best["state_dict"], strict=True)
    return {
        "model": model,
        "history": history,
        "best_epoch": int(best["best_epoch"]),
        "best_valid_mae": float(best["best_valid_mae"]),
        "trainable_parameters": int(trainable_parameters),
        "reused": False,
    }


def paired_prediction_metrics(
    prediction: Sequence[float],
    labels: Sequence[float],
    baseline: Sequence[float],
) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if not (
        prediction.shape == labels.shape == baseline.shape
        and np.isfinite(prediction).all()
        and np.isfinite(labels).all()
        and np.isfinite(baseline).all()
    ):
        raise ValueError("invalid paired prediction arrays")
    error = np.abs(prediction - labels)
    baseline_error = np.abs(baseline - labels)
    gain = baseline_error - error
    return {
        "mae": float(error.mean()),
        "gain_vs_baseline": float(gain.mean()),
        "win_rate_vs_baseline": float(np.mean(gain > 0)),
        "nondegrade_rate_vs_baseline": float(np.mean(gain >= -1e-12)),
        "large_harm_rate_010": float(np.mean(gain < -0.10)),
    }


def success_gate(
    fold_gains: Sequence[float],
    aggregate_gain: float,
    config: TaskDecomposedConfigV932,
) -> Dict[str, object]:
    gains = np.asarray(fold_gains, dtype=np.float64).reshape(-1)
    nondegrading = int(np.sum(gains >= -1e-12))
    worst_degradation = float(max(0.0, -float(gains.min())))
    passed = (
        float(aggregate_gain) >= float(config.required_gain_vs_control)
        and nondegrading >= int(config.required_nondegrading_folds)
        and worst_degradation <= float(config.max_worst_fold_degradation)
    )
    return {
        "passed": bool(passed),
        "aggregate_gain": float(aggregate_gain),
        "required_gain": float(config.required_gain_vs_control),
        "nondegrading_outer_folds": nondegrading,
        "required_nondegrading_outer_folds": int(
            config.required_nondegrading_folds
        ),
        "worst_fold_degradation": worst_degradation,
        "max_worst_fold_degradation": float(
            config.max_worst_fold_degradation
        ),
    }


__all__ = [
    "VERSION",
    "ORDINAL_THRESHOLDS",
    "VARIANT_NAMES",
    "TaskDecomposedConfigV932",
    "TaskDecomposedDLFV932",
    "variant_loss_weights",
    "ordinal_targets",
    "ordinal_monotonic_penalty",
    "auxiliary_losses",
    "multitask_loss",
    "predict_task_model",
    "prediction_diagnostics",
    "train_task_model",
    "paired_prediction_metrics",
    "success_gate",
]
