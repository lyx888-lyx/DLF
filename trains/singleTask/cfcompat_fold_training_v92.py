"""Fold-local clean DLF, ModDrop, and CFCompatKD training for V9.2."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .HingeLoss import HingeLoss
from .model.DLF import DLF
from .model.FSC_DLF import _FusionFeatureCapture
from .oof_group_splits_v92 import canonical_sample_id

logger = logging.getLogger("MMSA")

MODALITY_MASKS = {
    "LAV": (1.0, 1.0, 1.0),
    "LA": (1.0, 1.0, 0.0),
    "LV": (1.0, 0.0, 1.0),
    "L": (1.0, 0.0, 0.0),
}
MISSING_MODES = ("LA", "LV", "L")


def mode_to_mask(
    mode: str,
    batch_size: Optional[int] = None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    if mode not in MODALITY_MASKS:
        raise ValueError(f"unsupported modality mode: {mode}")
    value = torch.tensor(MODALITY_MASKS[mode], dtype=dtype, device=device)
    if batch_size is not None:
        value = value.unsqueeze(0).expand(int(batch_size), -1)
    return value


def sample_missing_masks(
    batch_size: int,
    generator: torch.Generator,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    choices = torch.randint(
        len(MISSING_MODES), (int(batch_size),), generator=generator
    )
    candidates = torch.stack([mode_to_mask(mode) for mode in MISSING_MODES])
    return candidates.index_select(0, choices).to(device=device, dtype=dtype)


def modes_from_masks(mask: torch.Tensor) -> List[str]:
    mapping = {(1, 1, 0): "LA", (1, 0, 1): "LV", (1, 0, 0): "L"}
    result = []
    for row in mask.detach().cpu().to(torch.int64).tolist():
        key = tuple(row)
        if key not in mapping:
            raise ValueError(f"unexpected missing mask: {key}")
        result.append(mapping[key])
    return result


def _present_column(mask: torch.Tensor, index: int, values: torch.Tensor):
    shape = [values.size(0)] + [1] * (values.ndim - 1)
    return mask[:, index].to(values).view(*shape)


class MissingModalityWrapper(nn.Module):
    """DLF wrapper matching feature/cf-compat-kd-v1 without editing DLF.py."""

    def __init__(self, backbone: nn.Module, audio_dim: int, vision_dim: int):
        super().__init__()
        self.backbone = backbone
        feature_dim = int(backbone.out_layer.in_features)
        self.missing_audio_token = nn.Parameter(
            torch.zeros(1, 1, int(audio_dim))
        )
        self.missing_vision_token = nn.Parameter(
            torch.zeros(1, 1, int(vision_dim))
        )
        self.mask_adapter = nn.Linear(3, feature_dim, bias=False)
        nn.init.zeros_(self.mask_adapter.weight)

    def forward(self, text, audio, vision, modality_mask):
        mask = modality_mask.to(device=audio.device, dtype=audio.dtype)
        audio_present = _present_column(mask, 1, audio)
        vision_present = _present_column(mask, 2, vision)
        masked_audio = (
            audio_present * audio
            + (1.0 - audio_present) * self.missing_audio_token
        )
        masked_vision = (
            vision_present * vision
            + (1.0 - vision_present) * self.missing_vision_token
        )
        fusion_residual = self.mask_adapter(1.0 - mask)

        # Algebraically reproduce the historical optional fusion_residual path:
        # add it before proj1 and again before out_layer so both the nonlinear
        # branch and the residual connection see last_hs + fusion_residual.
        def add_to_proj1(module, inputs):
            del module
            return (inputs[0] + fusion_residual,)

        def add_to_output_feature(module, inputs):
            del module
            return (inputs[0] + fusion_residual,)

        proj1_handle = self.backbone.proj1.register_forward_pre_hook(add_to_proj1)
        output_handle = self.backbone.out_layer.register_forward_pre_hook(
            add_to_output_feature
        )
        try:
            return self.backbone(text, masked_audio, masked_vision)
        finally:
            proj1_handle.remove()
            output_handle.remove()


def compute_task_loss(output, labels, criterion):
    return (
        criterion(output["output_logit"], labels)
        + criterion(output["logits_c"], labels)
        + 3.0 * criterion(output["logits_l_hetero"], labels)
        + criterion(output["logits_v_hetero"], labels)
        + criterion(output["logits_a_hetero"], labels)
    )


def compute_full_dlf_loss(
    output,
    labels,
    criterion,
    cosine: Optional[nn.Module] = None,
    hinge: Optional[nn.Module] = None,
):
    cosine = cosine if cosine is not None else nn.CosineEmbeddingLoss()
    hinge = hinge if hinge is not None else HingeLoss()
    task = compute_task_loss(output, labels, criterion)
    reconstruction = (
        F.mse_loss(output["recon_l"], output["origin_l"])
        + F.mse_loss(output["recon_v"], output["origin_v"])
        + F.mse_loss(output["recon_a"], output["origin_a"])
    )
    specific = (
        F.mse_loss(output["s_l"].permute(1, 2, 0), output["s_l_r"])
        + F.mse_loss(output["s_v"].permute(1, 2, 0), output["s_v_r"])
        + F.mse_loss(output["s_a"].permute(1, 2, 0), output["s_a_r"])
    )
    feature_dim = int(output["s_l"].shape[-1])
    target = torch.full(
        (output["s_l"].reshape(-1, feature_dim).size(0),),
        -1.0,
        dtype=output["s_l"].dtype,
        device=output["s_l"].device,
    )
    orthogonality = (
        cosine(
            output["s_l"].reshape(-1, feature_dim),
            output["c_l"].reshape(-1, feature_dim),
            target,
        )
        + cosine(
            output["s_v"].reshape(-1, feature_dim),
            output["c_v"].reshape(-1, feature_dim),
            target,
        )
        + cosine(
            output["s_a"].reshape(-1, feature_dim),
            output["c_a"].reshape(-1, feature_dim),
            target,
        )
    )
    shared = (
        output["c_l_sim"],
        output["c_v_sim"],
        output["c_a_sim"],
    )
    features = torch.cat(
        [
            feature[index].view(1, -1)
            for index in range(labels.size(0))
            for feature in shared
        ],
        dim=0,
    )
    ids = torch.cat(
        [
            labels[index].view(1, -1).repeat(3, 1)
            for index in range(labels.size(0))
        ],
        dim=0,
    )
    similarity = hinge(ids, features)
    return task + 0.1 * (
        specific + reconstruction + 0.1 * (similarity + orthogonality)
    )


def _batch_to_device(batch, device):
    return (
        batch["text"].to(device),
        batch["audio"].to(device),
        batch["vision"].to(device),
        batch["labels"]["M"].to(device).view(-1, 1),
    )


def _mae(prediction: torch.Tensor, labels: torch.Tensor) -> float:
    return float(
        torch.abs(prediction.view(-1) - labels.view(-1)).mean().item()
    )


@torch.no_grad()
def predict_plain(model, loader, device, capture_features: bool = False):
    model.eval()
    rows = []
    for batch in loader:
        text, audio, vision, labels = _batch_to_device(batch, device)
        capture = _FusionFeatureCapture(model) if capture_features else None
        try:
            output = model(text, audio, vision)
            feature = (
                capture.value.detach().cpu() if capture is not None else None
            )
        finally:
            if capture is not None:
                capture.close()
        ids = [canonical_sample_id(value) for value in list(batch["id"])]
        indices = batch["index"].view(-1).cpu().tolist()
        for offset, index in enumerate(indices):
            row = {
                "sample_index": int(index),
                "sample_id": ids[offset],
                "label": float(labels[offset].item()),
                "prediction": float(output["output_logit"][offset].item()),
            }
            if feature is not None:
                row["feature"] = feature[offset].clone()
            rows.append(row)
    return rows


@torch.no_grad()
def predict_wrapper_lav(model, loader, device, capture_features: bool = False):
    model.eval()
    rows = []
    for batch in loader:
        text, audio, vision, labels = _batch_to_device(batch, device)
        mask = mode_to_mask("LAV", labels.size(0), device, audio.dtype)
        capture = (
            _FusionFeatureCapture(model.backbone)
            if capture_features
            else None
        )
        try:
            output = model(text, audio, vision, mask)
            feature = (
                capture.value.detach().cpu() if capture is not None else None
            )
        finally:
            if capture is not None:
                capture.close()
        ids = [canonical_sample_id(value) for value in list(batch["id"])]
        indices = batch["index"].view(-1).cpu().tolist()
        for offset, index in enumerate(indices):
            row = {
                "sample_index": int(index),
                "sample_id": ids[offset],
                "label": float(labels[offset].item()),
                "prediction": float(output["output_logit"][offset].item()),
            }
            if feature is not None:
                row["feature"] = feature[offset].clone()
            rows.append(row)
    return rows


@torch.no_grad()
def evaluate_wrapper_modes(model, loader, device, criterion):
    model.eval()
    collected = {
        mode: {"prediction": [], "labels": [], "loss": []}
        for mode in MODALITY_MASKS
    }
    for batch in loader:
        text, audio, vision, labels = _batch_to_device(batch, device)
        for mode in MODALITY_MASKS:
            mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
            output = model(text, audio, vision, mask)
            collected[mode]["prediction"].append(
                output["output_logit"].detach().cpu()
            )
            collected[mode]["labels"].append(labels.detach().cpu())
            collected[mode]["loss"].append(
                float(criterion(output["output_logit"], labels).item())
            )
    result = {}
    for mode, values in collected.items():
        prediction = torch.cat(values["prediction"])
        labels = torch.cat(values["labels"])
        result[mode] = {
            "MAE": _mae(prediction, labels),
            "Loss": float(np.mean(values["loss"])),
        }
    return result


def validation_objective(metrics_by_mode) -> float:
    return 0.5 * float(metrics_by_mode["LAV"]["MAE"]) + 0.5 * float(
        np.mean([metrics_by_mode[mode]["MAE"] for mode in MISSING_MODES])
    )


def stable_average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def compatibility_from_deltas(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    ranks = stable_average_ranks(array)
    q = (ranks - 0.5) / float(len(array))
    return 1.0 - q


@torch.no_grad()
def build_compatibility_cache(evaluator, loader, device):
    evaluator.eval()
    raw = []
    for batch in loader:
        text, audio, vision, labels = _batch_to_device(batch, device)
        predictions: Dict[str, np.ndarray] = {}
        for mode in ("LAV",) + MISSING_MODES:
            mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
            predictions[mode] = (
                evaluator(text, audio, vision, mask)["output_logit"]
                .detach()
                .view(-1)
                .cpu()
                .numpy()
            )
        indices = batch["index"].view(-1).cpu().numpy().astype(int)
        for offset, index in enumerate(indices):
            raw.append(
                {
                    "sample_index": int(index),
                    **{
                        f"delta_{mode}": float(
                            abs(
                                predictions["LAV"][offset]
                                - predictions[mode][offset]
                            )
                        )
                        for mode in MISSING_MODES
                    },
                }
            )
    frame = pd.DataFrame(raw).sort_values("sample_index", kind="mergesort")
    if frame.empty or frame.sample_index.duplicated().any():
        raise RuntimeError("compatibility cache is empty or duplicated")
    for mode in MISSING_MODES:
        frame[f"compat_{mode}"] = compatibility_from_deltas(
            frame[f"delta_{mode}"]
        )
    return {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }


def _load_state(path: Path, device):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    return payload


def train_clean_dlf(args, train_loader, valid_loader, checkpoint, limits):
    model = DLF(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(args.patience)
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    best_mae = float("inf")
    best_epoch = 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(limits.clean_max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        totals = []
        for step, batch in enumerate(train_loader, start=1):
            text, audio, vision, labels = _batch_to_device(batch, args.device)
            loss = compute_full_dlf_loss(
                model(text, audio, vision),
                labels,
                criterion,
                cosine,
                hinge,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite clean DLF loss")
            (loss / max(1, int(args.update_epochs))).backward()
            if (
                step % max(1, int(args.update_epochs)) == 0
                or step == len(train_loader)
            ):
                if float(args.grad_clip) != -1.0:
                    nn.utils.clip_grad_value_(
                        model.parameters(), float(args.grad_clip)
                    )
                optimizer.step()
                optimizer.zero_grad()
            totals.append(float(loss.detach().item()))

        valid_rows = predict_plain(model, valid_loader, args.device)
        valid_prediction = torch.tensor(
            [row["prediction"] for row in valid_rows]
        )
        valid_labels = torch.tensor([row["label"] for row in valid_rows])
        valid_mae = _mae(valid_prediction, valid_labels)
        scheduler.step(valid_mae)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(totals)),
                "valid_mae": valid_mae,
            }
        )
        logger.info("V9.2 clean epoch=%d valid_mae=%.6f", epoch, valid_mae)
        if valid_mae < best_mae - 1e-6:
            best_mae = valid_mae
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        if epoch - best_epoch >= int(limits.early_stop):
            break
    if not checkpoint.is_file():
        raise RuntimeError("clean DLF did not save a checkpoint")
    return {
        "best_epoch": best_epoch,
        "best_valid_mae": best_mae,
        "history": history,
    }


def train_moddrop_evaluator(
    args,
    train_loader,
    valid_loader,
    clean_checkpoint,
    checkpoint,
    limits,
    seed,
):
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(
        _load_state(clean_checkpoint, args.device), strict=True
    )
    model = MissingModalityWrapper(
        backbone, int(args.feature_dims[1]), int(args.feature_dims[2])
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(args.patience)
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    generator = torch.Generator().manual_seed(int(seed) + 104729)
    best_j = float("inf")
    best_epoch = 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(limits.moddrop_max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        totals = []
        for step, batch in enumerate(train_loader, start=1):
            text, audio, vision, labels = _batch_to_device(batch, args.device)
            full_mask = mode_to_mask(
                "LAV", labels.size(0), args.device, audio.dtype
            )
            full_loss = compute_full_dlf_loss(
                model(text, audio, vision, full_mask),
                labels,
                criterion,
                cosine,
                hinge,
            )
            missing_mask = sample_missing_masks(
                labels.size(0), generator, args.device, audio.dtype
            )
            missing_loss = compute_task_loss(
                model(text, audio, vision, missing_mask), labels, criterion
            )
            loss = full_loss + missing_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite ModDrop loss")
            (loss / max(1, int(args.update_epochs))).backward()
            if (
                step % max(1, int(args.update_epochs)) == 0
                or step == len(train_loader)
            ):
                if float(args.grad_clip) != -1.0:
                    nn.utils.clip_grad_value_(
                        model.parameters(), float(args.grad_clip)
                    )
                optimizer.step()
                optimizer.zero_grad()
            totals.append(float(loss.detach().item()))

        metrics = evaluate_wrapper_modes(
            model, valid_loader, args.device, criterion
        )
        j_value = validation_objective(metrics)
        scheduler.step(j_value)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(totals)),
                "valid_j": j_value,
                **{
                    f"valid_{mode}_mae": metrics[mode]["MAE"]
                    for mode in MODALITY_MASKS
                },
            }
        )
        logger.info("V9.2 moddrop epoch=%d valid_j=%.6f", epoch, j_value)
        if j_value < best_j - 1e-6:
            best_j = j_value
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        if epoch - best_epoch >= int(limits.early_stop):
            break
    if not checkpoint.is_file():
        raise RuntimeError("ModDrop evaluator did not save a checkpoint")
    return {
        "best_epoch": best_epoch,
        "best_valid_j": best_j,
        "history": history,
    }


def train_cfcompat_student(
    args,
    train_loader,
    valid_loader,
    clean_checkpoint,
    evaluator_checkpoint,
    checkpoint,
    limits,
    seed,
):
    teacher = DLF(args).to(args.device)
    teacher.load_state_dict(
        _load_state(clean_checkpoint, args.device), strict=True
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()

    evaluator = MissingModalityWrapper(
        DLF(args).to(args.device),
        int(args.feature_dims[1]),
        int(args.feature_dims[2]),
    ).to(args.device)
    evaluator.load_state_dict(
        _load_state(evaluator_checkpoint, args.device), strict=True
    )
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)
    evaluator.eval()

    student_backbone = DLF(args).to(args.device)
    student_backbone.load_state_dict(
        _load_state(clean_checkpoint, args.device), strict=True
    )
    student = MissingModalityWrapper(
        student_backbone,
        int(args.feature_dims[1]),
        int(args.feature_dims[2]),
    ).to(args.device)
    compatibility = build_compatibility_cache(
        evaluator, train_loader, args.device
    )

    optimizer = optim.Adam(student.parameters(), lr=float(args.learning_rate))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(args.patience)
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    generator = torch.Generator().manual_seed(int(seed) + 104729)
    best_j = float("inf")
    best_epoch = 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(limits.cfcompat_max_epochs) + 1):
        student.train()
        teacher.eval()
        optimizer.zero_grad()
        totals = []
        gate_values = []
        for step, batch in enumerate(train_loader, start=1):
            text, audio, vision, labels = _batch_to_device(batch, args.device)
            full_mask = mode_to_mask(
                "LAV", labels.size(0), args.device, audio.dtype
            )
            full_loss = compute_full_dlf_loss(
                student(text, audio, vision, full_mask),
                labels,
                criterion,
                cosine,
                hinge,
            )
            missing_mask = sample_missing_masks(
                labels.size(0), generator, args.device, audio.dtype
            )
            modes = modes_from_masks(missing_mask)
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss = compute_task_loss(
                missing_output, labels, criterion
            )
            with torch.no_grad():
                teacher_prediction = teacher(
                    text, audio, vision
                )["output_logit"].detach()

            indices = batch["index"].view(-1).cpu().tolist()
            gates = torch.tensor(
                [
                    compatibility[int(index)][f"compat_{mode}"]
                    for index, mode in zip(indices, modes)
                ],
                dtype=labels.dtype,
                device=args.device,
            )
            kd_each = F.smooth_l1_loss(
                missing_output["output_logit"].view(-1),
                teacher_prediction.view(-1),
                reduction="none",
            )
            kd_loss = (
                (gates * kd_each).sum() / gates.sum().clamp_min(1e-8)
            )
            loss = full_loss + missing_loss + kd_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CFCompatKD loss")
            (loss / max(1, int(args.update_epochs))).backward()
            if (
                step % max(1, int(args.update_epochs)) == 0
                or step == len(train_loader)
            ):
                if float(args.grad_clip) != -1.0:
                    nn.utils.clip_grad_value_(
                        student.parameters(), float(args.grad_clip)
                    )
                optimizer.step()
                optimizer.zero_grad()
            totals.append(float(loss.detach().item()))
            gate_values.extend(gates.detach().cpu().tolist())

        metrics = evaluate_wrapper_modes(
            student, valid_loader, args.device, criterion
        )
        j_value = validation_objective(metrics)
        scheduler.step(j_value)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(totals)),
                "mean_gate": float(np.mean(gate_values)),
                "valid_j": j_value,
                **{
                    f"valid_{mode}_mae": metrics[mode]["MAE"]
                    for mode in MODALITY_MASKS
                },
            }
        )
        logger.info("V9.2 cfcompat epoch=%d valid_j=%.6f", epoch, j_value)
        if j_value < best_j - 1e-6:
            best_j = j_value
            best_epoch = epoch
            torch.save(student.state_dict(), checkpoint)
        if epoch - best_epoch >= int(limits.early_stop):
            break
    if not checkpoint.is_file():
        raise RuntimeError("CFCompatKD student did not save a checkpoint")
    return {
        "best_epoch": best_epoch,
        "best_valid_j": best_j,
        "history": history,
    }


def _checkpoint_payload(path: Path):
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
    }
