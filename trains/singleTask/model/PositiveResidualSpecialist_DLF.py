"""Constrained positive-residual specialist on top of the frozen V7.1 Student."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn

from .CPFD_DLF import soft_region_membership
from .FSC_DLF import _FusionFeatureCapture


class PositiveResidualSpecialist(nn.Module):
    """Learn a bounded non-negative correction while preserving V7.1 exactly.

    The frozen V7.1 path remains the reference predictor.  The new path receives
    a stop-gradient fusion feature and predicts only a positive correction:

        correction(x) = gate(x) * magnitude(x),  0 <= correction <= max_correction

    It cannot directly push any prediction downward and therefore has a clear
    no-harm interpretation for the ordinary-positive under-prediction problem.
    """

    def __init__(
        self,
        args,
        hidden_dim: int = 160,
        legacy_hidden_dim: int = 128,
        dropout: float = 0.18,
        residual_max: float = 0.35,
        region_temperature: float = 0.55,
        max_correction: float = 0.50,
        initial_gate_probability: float = 0.20,
        initial_magnitude_fraction: float = 0.30,
    ):
        super().__init__()
        from . import DLF as DLFModel

        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        self.residual_max = float(residual_max)
        self.region_temperature = float(region_temperature)
        self.max_correction = float(max_correction)

        # Legacy V7.1 modules. Their names and shapes intentionally match the
        # historical checkpoint so the original Student is restored exactly.
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

        self.specialist_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.magnitude_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.magnitude_head.weight)

        gate_probability = min(max(float(initial_gate_probability), 1e-4), 1.0 - 1e-4)
        gate_bias = torch.log(torch.tensor(gate_probability / (1.0 - gate_probability)))
        nn.init.constant_(self.gate_head.bias, float(gate_bias.item()))

        magnitude_fraction = min(
            max(float(initial_magnitude_fraction), 1e-4), 1.0 - 1e-4
        )
        magnitude_bias = torch.log(
            torch.tensor(magnitude_fraction / (1.0 - magnitude_fraction))
        )
        nn.init.constant_(self.magnitude_head.bias, float(magnitude_bias.item()))

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
            self.specialist_adapter,
            self.gate_head,
            self.magnitude_head,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.eval()

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def set_train_mode(self):
        # Keep every historical component deterministic.
        self.eval()
        self.specialist_adapter.train()
        self.gate_head.train()
        self.magnitude_head.train()

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
        legacy_correction = self.residual_max * torch.tanh(raw_correction)
        legacy_prediction = base_prediction + legacy_correction
        return legacy_prediction, legacy_correction, region_probs

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        capture = _FusionFeatureCapture(self.backbone)
        try:
            backbone_output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError(
                    "Positive-residual specialist captured no fusion feature."
                )
            feature = capture.value
        finally:
            capture.close()

        base_prediction = backbone_output["output_logit"]
        legacy_prediction, legacy_correction, region_probs = self._legacy_prediction(
            feature, base_prediction
        )

        specialist_feature = self.specialist_adapter(feature.detach())
        gate_logit = self.gate_head(specialist_feature)
        gate_probability = torch.sigmoid(gate_logit)
        magnitude_fraction = torch.sigmoid(self.magnitude_head(specialist_feature))
        magnitude = self.max_correction * magnitude_fraction
        positive_correction = gate_probability * magnitude

        return {
            "backbone": backbone_output,
            "feature": feature,
            "base_prediction": base_prediction,
            "legacy_prediction": legacy_prediction,
            "legacy_correction": legacy_correction,
            "region_probs": region_probs,
            "specialist_feature": specialist_feature,
            "gate_logit": gate_logit,
            "gate_probability": gate_probability,
            "magnitude": magnitude,
            "positive_correction": positive_correction,
            "specialist_prediction": legacy_prediction + positive_correction,
        }
