"""Function-space consensus student and frozen DLF teachers for V7."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import DLF as DLFModel


class _FusionFeatureCapture:
    """Capture the tensor immediately before the scalar DLF output layer."""

    def __init__(self, backbone: nn.Module):
        self.value = None
        self.handle = backbone.out_layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        del module
        self.value = inputs[0]

    def close(self):
        self.handle.remove()


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "backbone"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    state = {}
    for key, value in payload.items():
        new_key = str(key)
        for prefix in ("module.", "student.", "model.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        state[new_key] = value
    return state


def load_dlf_checkpoint(backbone: nn.Module, path, map_location=None):
    payload = torch.load(path, map_location=map_location)
    state = _extract_state_dict(payload)
    return backbone.load_state_dict(state, strict=False)


class FrozenTeacherDLF(nn.Module):
    """A frozen DLF teacher that also exposes its final fusion feature."""

    def __init__(self, args):
        super().__init__()
        self.backbone = getattr(DLFModel, "DLF")(args)

    def load_checkpoint(self, path, map_location=None):
        return load_dlf_checkpoint(self.backbone, path, map_location)

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self

    @torch.no_grad()
    def forward(self, text, audio, vision):
        capture = _FusionFeatureCapture(self.backbone)
        try:
            output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError("Teacher fusion feature hook did not capture a tensor.")
            feature = capture.value
        finally:
            capture.close()
        return {
            "prediction": output["output_logit"],
            "feature": feature,
        }


class FunctionConsensusStudent(nn.Module):
    """DLF student with a zero-initialized residual adapter and risk heads."""

    def __init__(
        self,
        args,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        residual_max: float = 0.50,
    ):
        super().__init__()
        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        self.residual_max = float(residual_max)
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.log_scale_head = nn.Linear(hidden_dim, 1)
        self.ordinal_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.constant_(self.log_scale_head.bias, -1.0)

    def load_backbone_checkpoint(self, path, map_location=None):
        return load_dlf_checkpoint(self.backbone, path, map_location)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def unfreeze_backbone_tail(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        modules: Iterable[nn.Module] = (
            self.backbone.projector_l,
            self.backbone.projector_a,
            self.backbone.projector_v,
            self.backbone.projector_c,
            self.backbone.proj1,
            self.backbone.proj2,
            self.backbone.out_layer,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if hasattr(self.backbone, "text_model"):
            self.backbone.text_model.eval()

    def set_train_mode(self):
        self.train()
        # Tail linear parameters still receive gradients in eval mode while
        # dropout in the frozen Transformer/BERT path stays deterministic.
        self.backbone.eval()
        if hasattr(self.backbone, "text_model"):
            self.backbone.text_model.eval()

    def forward(self, text, audio, vision):
        capture = _FusionFeatureCapture(self.backbone)
        try:
            output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError("Student fusion feature hook did not capture a tensor.")
            feature = capture.value
        finally:
            capture.close()
        latent = self.adapter(feature)
        correction = self.residual_max * torch.tanh(self.residual_head(latent))
        base_prediction = output["output_logit"]
        prediction = base_prediction + correction
        return {
            "backbone": output,
            "feature": feature,
            "latent": latent,
            "base_prediction": base_prediction,
            "prediction": prediction,
            "correction": correction,
            "log_scale": self.log_scale_head(latent),
            "ordinal_logits": self.ordinal_head(latent),
        }


def robust_committee_summary(
    predictions: torch.Tensor,
    features: torch.Tensor,
    dispersion_temperature: float = 0.12,
) -> Dict[str, torch.Tensor]:
    """Summarize cached teacher predictions and coordinate-free features.

    Args:
        predictions: [batch, teachers, 1]
        features: [teachers, batch, feature_dim]
    """

    if predictions.dim() != 3 or predictions.size(-1) != 1:
        raise ValueError("predictions must have shape [batch, teachers, 1].")
    teacher_count = predictions.size(1)
    sorted_predictions = predictions.sort(dim=1).values
    mean = predictions.mean(dim=1)
    median = predictions.median(dim=1).values
    if teacher_count >= 5:
        trim = max(1, int(round(teacher_count * 0.20)))
        trimmed = sorted_predictions[:, trim: teacher_count - trim].mean(dim=1)
    elif teacher_count >= 3:
        trimmed = sorted_predictions[:, 1:-1].mean(dim=1)
    else:
        trimmed = mean
    consensus = 0.50 * trimmed + 0.50 * mean
    deviation = torch.abs(predictions - median.unsqueeze(1))
    mad = deviation.median(dim=1).values
    std = predictions.std(dim=1, unbiased=False)
    if teacher_count > 1:
        leave_one_out = (
            predictions.sum(dim=1, keepdim=True) - predictions
        ) / float(teacher_count - 1)
        jackknife_std = leave_one_out.std(dim=1, unbiased=False)
    else:
        jackknife_std = torch.zeros_like(mean)
    dispersion = mad + 0.50 * std + jackknife_std
    agreement = torch.exp(
        -dispersion / max(float(dispersion_temperature), 1e-6)
    ).clamp_min(1e-4)
    agreement = agreement / agreement.mean().clamp_min(1e-6)

    normalized = F.normalize(features, dim=2)
    teacher_kernels = torch.einsum("tbd,tcd->tbc", normalized, normalized)
    kernel = teacher_kernels.mean(dim=0)
    kernel_variance = teacher_kernels.var(dim=0, unbiased=False).mean(dim=1, keepdim=True)

    return {
        "mean": mean,
        "median": median,
        "trimmed_mean": trimmed,
        "consensus": consensus,
        "mad": mad,
        "std": std,
        "jackknife_std": jackknife_std,
        "dispersion": dispersion,
        "agreement": agreement.detach(),
        "kernel": kernel.detach(),
        "kernel_variance": kernel_variance.detach(),
    }
