"""Ordinal-consistent extension of the frozen V7.1 complementarity student."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .CPFD_DLF import soft_region_membership
from .FSC_DLF import _FusionFeatureCapture


class MonotonicOrdinalHead(nn.Module):
    """A rank-consistent cumulative ordinal head with fixed class boundaries."""

    def __init__(
        self,
        input_dim: int,
        thresholds: Sequence[float],
        low_value: float,
        high_value: float,
    ):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.log_temperature = nn.Parameter(torch.tensor(0.55))
        self.low_value = float(low_value)
        self.high_value = float(high_value)
        self.register_buffer(
            "thresholds",
            torch.tensor(tuple(float(value) for value in thresholds)).view(1, -1),
        )

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        radius = max(abs(self.low_value), abs(self.high_value))
        score = radius * torch.tanh(self.score(feature) / max(radius, 1e-6))
        temperature = F.softplus(self.log_temperature) + 0.50
        logits = temperature * (score - self.thresholds)
        probabilities = torch.sigmoid(logits)
        value = self.low_value + probabilities.sum(dim=1, keepdim=True)
        return {
            "score": score,
            "temperature": temperature,
            "logits": logits,
            "probabilities": probabilities,
            "value": value,
        }


class OrdinalComplementarityStudent(nn.Module):
    """Preserve the V7.1 predictor and learn a small ordinal correction path.

    The DLF backbone and all legacy V7.1 residual modules stay frozen. Only the
    new ordinal adapter, the two monotonic ordinal heads and one bounded global
    blending coefficient are trained. This makes the original V7.1 prediction
    an explicit deployment-safe reference rather than a moving target.
    """

    def __init__(
        self,
        args,
        hidden_dim: int = 128,
        ordinal_hidden_dim: int = 128,
        dropout: float = 0.18,
        residual_max: float = 0.35,
        region_temperature: float = 0.55,
        max_ordinal_blend: float = 0.30,
        initial_ordinal_blend: float = 0.02,
    ):
        super().__init__()
        from . import DLF as DLFModel

        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        self.residual_max = float(residual_max)
        self.region_temperature = float(region_temperature)
        self.max_ordinal_blend = float(max_ordinal_blend)

        # Legacy V7.1 modules. Their names intentionally match the frozen
        # checkpoint so the old Student can be restored exactly.
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

        self.ordinal_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, ordinal_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ordinal_hidden_dim, ordinal_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.ordinal7_head = MonotonicOrdinalHead(
            ordinal_hidden_dim,
            thresholds=(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5),
            low_value=-3.0,
            high_value=3.0,
        )
        self.ordinal5_head = MonotonicOrdinalHead(
            ordinal_hidden_dim,
            thresholds=(-1.5, -0.5, 0.5, 1.5),
            low_value=-2.0,
            high_value=2.0,
        )

        ratio = min(
            max(float(initial_ordinal_blend) / max(self.max_ordinal_blend, 1e-6), 1e-4),
            1.0 - 1e-4,
        )
        self.ordinal_blend_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def load_v71_checkpoint(self, path, map_location=None):
        payload = torch.load(path, map_location=map_location)
        state = payload.get("state", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError("V7.1 checkpoint does not contain a state dictionary.")
        return self.load_state_dict(state, strict=False)

    def freeze_legacy(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        modules: Iterable[nn.Module] = (
            self.ordinal_adapter,
            self.ordinal7_head,
            self.ordinal5_head,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.ordinal_blend_logit.requires_grad_(True)
        self.eval()

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def set_train_mode(self):
        # Keep the complete legacy prediction deterministic while training only
        # the new ordinal path.
        self.eval()
        self.ordinal_adapter.train()
        self.ordinal7_head.train()
        self.ordinal5_head.train()

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        capture = _FusionFeatureCapture(self.backbone)
        try:
            backbone_output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError("Ordinal Student fusion hook captured no feature.")
            feature = capture.value
        finally:
            capture.close()

        base_prediction = backbone_output["output_logit"]
        legacy_latent = self.adapter(feature)
        region_probs = soft_region_membership(
            base_prediction.detach(), self.region_temperature
        )
        regional_raw = (self.region_heads(legacy_latent) * region_probs).sum(
            dim=1, keepdim=True
        )
        global_raw = self.global_head(legacy_latent)
        raw_correction = 0.35 * global_raw + 0.65 * regional_raw
        correction = self.residual_max * torch.tanh(raw_correction)
        legacy_prediction = base_prediction + correction

        ordinal_feature = self.ordinal_adapter(feature.detach())
        ordinal7 = self.ordinal7_head(ordinal_feature)
        ordinal5 = self.ordinal5_head(ordinal_feature)
        ordinal_consensus = 0.70 * ordinal7["value"] + 0.30 * ordinal5["value"]
        ordinal_blend = self.max_ordinal_blend * torch.sigmoid(
            self.ordinal_blend_logit
        )
        prediction = legacy_prediction + ordinal_blend * (
            ordinal_consensus - legacy_prediction
        )

        return {
            "backbone": backbone_output,
            "feature": feature,
            "base_prediction": base_prediction,
            "legacy_prediction": legacy_prediction,
            "prediction": prediction,
            "correction": correction,
            "region_probs": region_probs,
            "ordinal_feature": ordinal_feature,
            "ordinal7_logits": ordinal7["logits"],
            "ordinal7_value": ordinal7["value"],
            "ordinal5_logits": ordinal5["logits"],
            "ordinal5_value": ordinal5["value"],
            "ordinal_consensus": ordinal_consensus,
            "ordinal_blend": ordinal_blend.view(1),
        }
