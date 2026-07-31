"""Frozen-anchor DLF residual expert for tail-sentiment specialization V9.1.

The original DLF predictor remains immutable.  A lightweight residual branch
learns bidirectional corrections relative to that fixed anchor.  A detached,
role-specific applicability gate limits corrections outside the candidate's
training domain, and an auxiliary mechanism head describes which residual
repair is needed (sign repair, magnitude increase, magnitude decrease, or
near-correct).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FSC_DLF import _FusionFeatureCapture, load_dlf_checkpoint


class TailResidualDLF(nn.Module):
    """Immutable DLF anchor plus a bounded, gated residual specialist."""

    def __init__(
        self,
        args,
        hidden_dim: int = 192,
        dropout: float = 0.15,
        residual_max: float = 1.50,
    ) -> None:
        super().__init__()
        from . import DLF as DLFModel

        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        hidden_dim = int(hidden_dim)
        self.residual_max = float(residual_max)

        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
        )
        head_input_dim = hidden_dim + 1
        self.residual_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, max(64, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(64, hidden_dim // 2), 1),
        )
        self.applicability_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, max(32, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, hidden_dim // 4), 1),
        )
        self.mechanism_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, max(64, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(64, hidden_dim // 2), 4),
        )

        # All candidates start as the exact anchor.  The final applicability
        # bias is deliberately neutral; only its supervised BCE target can
        # change the gate because the regression path uses a detached gate.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.applicability_head[-1].weight)
        nn.init.zeros_(self.applicability_head[-1].bias)

        self.freeze_anchor()

    def load_anchor_checkpoint(self, path, map_location=None):
        return load_dlf_checkpoint(self.backbone, path, map_location)

    def freeze_anchor(self) -> None:
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

    def parameter_groups(self, learning_rate: float) -> List[Dict[str, object]]:
        parameters = [
            parameter
            for module in (
                self.adapter,
                self.residual_head,
                self.applicability_head,
                self.mechanism_head,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("TailResidualDLF configured no trainable parameters.")
        return [{"params": parameters, "lr": float(learning_rate)}]

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        capture = _FusionFeatureCapture(self.backbone)
        try:
            with torch.no_grad():
                backbone_output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError("V9.1 fusion feature hook did not capture a tensor.")
            feature = capture.value.detach()
        finally:
            capture.close()

        anchor_prediction = backbone_output["output_logit"].detach()
        latent = self.adapter(feature)
        head_input = torch.cat([latent, anchor_prediction], dim=1)

        raw_correction = self.residual_max * torch.tanh(
            self.residual_head(head_input)
        )
        applicability_logit = self.applicability_head(head_input)
        applicability_prob = torch.sigmoid(applicability_logit)

        # The gate is trained only by its explicit applicability target.  This
        # avoids the residual objective trivially driving gate probability to
        # one on every sample while still enforcing deployable internal gating.
        correction = applicability_prob.detach() * raw_correction
        prediction = anchor_prediction + correction
        mechanism_logits = self.mechanism_head(head_input)

        return {
            "backbone": backbone_output,
            "feature": feature,
            "latent": latent,
            "anchor_prediction": anchor_prediction,
            "prediction": prediction,
            "raw_correction": raw_correction,
            "correction": correction,
            "applicability_logit": applicability_logit,
            "applicability_prob": applicability_prob,
            "mechanism_logits": mechanism_logits,
            "mechanism_probs": F.softmax(mechanism_logits, dim=1),
        }
