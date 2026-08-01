"""Conservative unified Student for absolute-target OOF distillation (V9.15)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FSC_DLF import _FusionFeatureCapture, load_dlf_checkpoint
from .SemanticCostCoachV99 import SPECIALIST_NAMES

STUDENT_VERSION = "absolute_target_expert_student_v915_v1"


class AbsoluteTargetExpertStudentV915(nn.Module):
    """Frozen DLF anchor plus one deployable residual and training-only heads."""

    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        residual_max: float = 0.15,
        auxiliary_residual_max: float = 1.50,
    ) -> None:
        super().__init__()
        from . import DLF as DLFModel

        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        self.residual_max = float(residual_max)
        self.auxiliary_residual_max = float(auxiliary_residual_max)

        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
        )
        self.unified_head = nn.Linear(int(hidden_dim), 1)
        self.specialist_heads = nn.Linear(
            int(hidden_dim), len(SPECIALIST_NAMES)
        )
        nn.init.zeros_(self.unified_head.weight)
        nn.init.zeros_(self.unified_head.bias)
        nn.init.zeros_(self.specialist_heads.weight)
        nn.init.zeros_(self.specialist_heads.bias)
        self.freeze_backbone()

    def load_backbone_checkpoint(self, path, map_location=None):
        return load_dlf_checkpoint(self.backbone, path, map_location)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        if hasattr(self.backbone, "text_model"):
            self.backbone.text_model.eval()

    def set_train_mode(self) -> None:
        self.train()
        self.backbone.eval()
        if hasattr(self.backbone, "text_model"):
            self.backbone.text_model.eval()

    def trainable_parameters(self):
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        capture = _FusionFeatureCapture(self.backbone)
        try:
            backbone_output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError(
                    "V9.15 student fusion feature hook captured nothing"
                )
            feature = capture.value
        finally:
            capture.close()

        latent = self.adapter(feature)
        base_prediction = backbone_output["output_logit"]
        correction = self.residual_max * torch.tanh(
            self.unified_head(latent)
        )
        auxiliary_corrections = self.auxiliary_residual_max * torch.tanh(
            self.specialist_heads(latent)
        )
        prediction = base_prediction + correction
        auxiliary_predictions = base_prediction + auxiliary_corrections
        return {
            "backbone": backbone_output,
            "feature": feature,
            "latent": latent,
            "base_prediction": base_prediction,
            "correction": correction,
            "prediction": prediction,
            "auxiliary_corrections": auxiliary_corrections,
            "auxiliary_predictions": auxiliary_predictions,
        }


def absolute_target_distillation_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    teacher_prediction: torch.Tensor,
    expert_predictions: torch.Tensor,
    expert_relevance: torch.Tensor,
    distill_weight: float,
    auxiliary_weight: float,
    correction_penalty: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Distill absolute predictions so OOF and deployment anchors share coordinates."""
    labels = labels.view(-1, 1).to(output["prediction"])
    teacher_prediction = teacher_prediction.view(-1, 1).to(
        output["prediction"]
    )
    expert_predictions = expert_predictions.to(
        output["auxiliary_predictions"]
    )
    expert_relevance = expert_relevance.to(
        output["auxiliary_predictions"]
    )
    if expert_predictions.shape != output["auxiliary_predictions"].shape:
        raise ValueError("expert absolute-prediction shape mismatch")
    if expert_relevance.shape != expert_predictions.shape:
        raise ValueError("expert relevance shape mismatch")

    label_mae = F.l1_loss(output["prediction"], labels)
    unified_distill = F.smooth_l1_loss(
        output["prediction"],
        teacher_prediction,
        beta=0.05,
    )
    auxiliary_raw = F.smooth_l1_loss(
        output["auxiliary_predictions"],
        expert_predictions,
        beta=0.10,
        reduction="none",
    )
    relevance = expert_relevance.clamp_min(0.0)
    auxiliary_distill = (
        auxiliary_raw * relevance
    ).sum() / relevance.sum().clamp_min(1e-6)
    correction_l2 = output["correction"].square().mean()

    effective_auxiliary_weight = (
        float(auxiliary_weight) if float(distill_weight) > 0.0 else 0.0
    )
    total = (
        label_mae
        + float(distill_weight) * unified_distill
        + effective_auxiliary_weight * auxiliary_distill
        + float(correction_penalty) * correction_l2
    )
    return {
        "total": total,
        "label_mae": label_mae,
        "absolute_distill": unified_distill,
        "auxiliary_absolute_distill": auxiliary_distill,
        "correction_l2": correction_l2,
    }
