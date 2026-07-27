"""Complementarity-preserving DLF student for V7.1."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .FSC_DLF import _FusionFeatureCapture, load_dlf_checkpoint


REGION_CENTERS = (-2.25, -1.0, 0.0, 1.0, 2.25)


def soft_region_membership(
    prediction: torch.Tensor,
    temperature: float = 0.55,
) -> torch.Tensor:
    """Deployable soft sentiment-region membership from the anchor prediction."""
    centers = prediction.new_tensor(REGION_CENTERS).view(1, -1)
    distance = torch.abs(prediction.view(-1, 1) - centers)
    return torch.softmax(-distance / max(float(temperature), 1e-6), dim=1)


class ComplementarityResidualStudent(nn.Module):
    """Frozen DLF plus zero-initialized global/region residual heads."""

    def __init__(
        self,
        args,
        hidden_dim: int = 128,
        dropout: float = 0.18,
        residual_max: float = 0.35,
        region_temperature: float = 0.55,
    ):
        super().__init__()
        from . import DLF as DLFModel

        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        self.residual_max = float(residual_max)
        self.region_temperature = float(region_temperature)

        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.global_head = nn.Linear(hidden_dim, 1)
        self.region_heads = nn.Linear(hidden_dim, 5)
        self.log_scale_head = nn.Linear(hidden_dim, 1)
        self.ordinal_head = nn.Linear(hidden_dim, 4)

        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        nn.init.zeros_(self.region_heads.weight)
        nn.init.zeros_(self.region_heads.bias)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.constant_(self.log_scale_head.bias, -1.5)

    def load_backbone_checkpoint(self, path, map_location=None):
        return load_dlf_checkpoint(self.backbone, path, map_location)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def set_train_mode(self):
        self.train()
        # The DLF anchor must remain deterministic and fixed.
        self.backbone.eval()
        if hasattr(self.backbone, "text_model"):
            self.backbone.text_model.eval()

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        capture = _FusionFeatureCapture(self.backbone)
        try:
            output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError("Student fusion feature hook did not capture a tensor.")
            feature = capture.value
        finally:
            capture.close()

        base_prediction = output["output_logit"]
        latent = self.adapter(feature)
        region_probs = soft_region_membership(
            base_prediction.detach(), self.region_temperature
        )
        regional_raw = (
            self.region_heads(latent) * region_probs
        ).sum(dim=1, keepdim=True)
        global_raw = self.global_head(latent)
        raw_correction = 0.35 * global_raw + 0.65 * regional_raw
        correction = self.residual_max * torch.tanh(raw_correction)
        prediction = base_prediction + correction

        return {
            "backbone": output,
            "feature": feature,
            "latent": latent,
            "base_prediction": base_prediction,
            "prediction": prediction,
            "correction": correction,
            "global_correction_raw": global_raw,
            "regional_correction_raw": regional_raw,
            "region_probs": region_probs,
            "log_scale": self.log_scale_head(latent),
            "ordinal_logits": self.ordinal_head(latent),
        }
