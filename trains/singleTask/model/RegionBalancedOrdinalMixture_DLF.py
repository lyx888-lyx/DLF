"""Region-balanced ordinal soft-mixture extension of the frozen V7.1 Student."""

from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .CPFD_DLF import soft_region_membership
from .FSC_DLF import _FusionFeatureCapture
from .OrdinalCPFD_DLF import MonotonicOrdinalHead


class RegionBalancedOrdinalMixtureStudent(nn.Module):
    """Keep the V7.1 predictor frozen and learn a guarded soft mixture head.

    The new path contains three polarity-specific regression experts, a soft
    polarity gate and two monotonic ordinal heads.  Its output is blended into
    the frozen V7.1 prediction with a bounded scalar, so the original model is
    always available as a deployment-safe reference.
    """

    def __init__(
        self,
        args,
        hidden_dim: int = 160,
        legacy_hidden_dim: int = 128,
        dropout: float = 0.18,
        residual_max: float = 0.35,
        region_temperature: float = 0.55,
        neutral_expert_range: float = 0.75,
        max_mixture_blend: float = 0.35,
        initial_mixture_blend: float = 0.02,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        from . import DLF as DLFModel

        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        self.residual_max = float(residual_max)
        self.region_temperature = float(region_temperature)
        self.neutral_expert_range = float(neutral_expert_range)
        self.max_mixture_blend = float(max_mixture_blend)
        self.gate_temperature = max(float(gate_temperature), 1e-4)

        # Legacy V7.1 modules.  Names and shapes intentionally match the source
        # checkpoint so the historical Student is restored exactly.
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, legacy_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(legacy_hidden_dim, legacy_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.global_head = nn.Linear(legacy_hidden_dim, 1)
        self.region_heads = nn.Linear(legacy_hidden_dim, 5)
        self.log_scale_head = nn.Linear(legacy_hidden_dim, 1)
        self.ordinal_head = nn.Linear(legacy_hidden_dim, 4)

        self.mixture_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.polarity_head = nn.Linear(hidden_dim, 3)
        self.expert_head = nn.Linear(hidden_dim, 3)
        self.ordinal7_head = MonotonicOrdinalHead(
            hidden_dim,
            thresholds=(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5),
            low_value=-3.0,
            high_value=3.0,
        )
        self.ordinal5_head = MonotonicOrdinalHead(
            hidden_dim,
            thresholds=(-1.5, -0.5, 0.5, 1.5),
            low_value=-2.0,
            high_value=2.0,
        )

        # Start with a neutral gate and mild experts.  The bounded blend is tiny,
        # so the initial prediction remains extremely close to V7.1.
        nn.init.zeros_(self.polarity_head.weight)
        nn.init.zeros_(self.polarity_head.bias)
        nn.init.zeros_(self.expert_head.weight)
        nn.init.constant_(self.expert_head.bias[0], -1.0986123)
        nn.init.zeros_(self.expert_head.bias[1])
        nn.init.constant_(self.expert_head.bias[2], -1.0986123)

        ratio = min(
            max(float(initial_mixture_blend) / max(self.max_mixture_blend, 1e-6), 1e-4),
            1.0 - 1e-4,
        )
        self.mixture_blend_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def load_v71_checkpoint(self, path, map_location=None):
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        state = payload.get("state", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError("V7.1 checkpoint does not contain a state dictionary.")
        return self.load_state_dict(state, strict=False)

    def freeze_legacy(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        modules: Iterable[nn.Module] = (
            self.mixture_adapter,
            self.polarity_head,
            self.expert_head,
            self.ordinal7_head,
            self.ordinal5_head,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.mixture_blend_logit.requires_grad_(True)
        self.eval()

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def set_train_mode(self):
        self.eval()
        self.mixture_adapter.train()
        self.polarity_head.train()
        self.expert_head.train()
        self.ordinal7_head.train()
        self.ordinal5_head.train()

    def _legacy_prediction(self, feature, base_prediction):
        latent = self.adapter(feature)
        region_probs = soft_region_membership(
            base_prediction.detach(), self.region_temperature
        )
        regional_raw = (self.region_heads(latent) * region_probs).sum(
            dim=1, keepdim=True
        )
        global_raw = self.global_head(latent)
        raw_correction = 0.35 * global_raw + 0.65 * regional_raw
        correction = self.residual_max * torch.tanh(raw_correction)
        return base_prediction + correction, correction, region_probs

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        capture = _FusionFeatureCapture(self.backbone)
        try:
            backbone_output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError("Mixture Student fusion hook captured no feature.")
            feature = capture.value
        finally:
            capture.close()

        base_prediction = backbone_output["output_logit"]
        legacy_prediction, legacy_correction, region_probs = self._legacy_prediction(
            feature, base_prediction
        )

        mixture_feature = self.mixture_adapter(feature.detach())
        polarity_logits = self.polarity_head(mixture_feature) / self.gate_temperature
        polarity_probs = torch.softmax(polarity_logits, dim=1)
        expert_raw = self.expert_head(mixture_feature)
        expert_values = torch.stack(
            (
                -3.0 * torch.sigmoid(expert_raw[:, 0]),
                self.neutral_expert_range * torch.tanh(expert_raw[:, 1]),
                3.0 * torch.sigmoid(expert_raw[:, 2]),
            ),
            dim=1,
        )
        mixture_value = (polarity_probs * expert_values).sum(dim=1, keepdim=True)

        ordinal7 = self.ordinal7_head(mixture_feature)
        ordinal5 = self.ordinal5_head(mixture_feature)
        ordinal_consensus = 0.70 * ordinal7["value"] + 0.30 * ordinal5["value"]

        mixture_blend = self.max_mixture_blend * torch.sigmoid(
            self.mixture_blend_logit
        )
        prediction = legacy_prediction + mixture_blend * (
            mixture_value - legacy_prediction
        )

        return {
            "backbone": backbone_output,
            "feature": feature,
            "base_prediction": base_prediction,
            "legacy_prediction": legacy_prediction,
            "legacy_correction": legacy_correction,
            "region_probs": region_probs,
            "mixture_feature": mixture_feature,
            "polarity_logits": polarity_logits,
            "polarity_probs": polarity_probs,
            "expert_values": expert_values,
            "mixture_value": mixture_value,
            "ordinal7_logits": ordinal7["logits"],
            "ordinal7_value": ordinal7["value"],
            "ordinal5_logits": ordinal5["logits"],
            "ordinal5_value": ordinal5["value"],
            "ordinal_consensus": ordinal_consensus,
            "mixture_blend": mixture_blend.view(1),
            "prediction": prediction,
        }
