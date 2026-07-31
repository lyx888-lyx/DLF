"""Low-capacity residual head trained on grouped OOF CFCompatKD features."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CachedTailResidualHeadV92(nn.Module):
    """Predict a gated signed correction from a cached fusion feature and anchor.

    The anchor/backbone is external and immutable. This module therefore cannot
    accidentally convert an in-sample train prediction into a moving baseline.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        residual_max: float = 1.50,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.residual_max = float(residual_max)
        input_dim = self.feature_dim + 1
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.applicability_head = nn.Linear(hidden_dim, 1)
        self.mechanism_head = nn.Linear(hidden_dim, 4)

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.applicability_head.weight)
        nn.init.zeros_(self.applicability_head.bias)

    def parameter_groups(self, learning_rate: float) -> List[Dict[str, object]]:
        return [{"params": self.parameters(), "lr": float(learning_rate)}]

    def forward(self, feature: torch.Tensor, anchor: torch.Tensor):
        if feature.dim() != 2 or feature.size(1) != self.feature_dim:
            raise ValueError(
                f"feature must have shape [N, {self.feature_dim}], got {tuple(feature.shape)}"
            )
        anchor = anchor.view(-1, 1).to(feature)
        latent = self.adapter(torch.cat([feature, anchor], dim=1))
        raw_correction = self.residual_max * torch.tanh(
            self.residual_head(latent)
        )
        applicability_logit = self.applicability_head(latent)
        applicability_prob = torch.sigmoid(applicability_logit)
        correction = applicability_prob.detach() * raw_correction
        prediction = anchor + correction
        mechanism_logits = self.mechanism_head(latent)
        return {
            "prediction": prediction,
            "anchor": anchor,
            "raw_correction": raw_correction,
            "correction": correction,
            "applicability_logit": applicability_logit,
            "applicability_prob": applicability_prob,
            "mechanism_logits": mechanism_logits,
            "mechanism_probs": F.softmax(mechanism_logits, dim=1),
        }
