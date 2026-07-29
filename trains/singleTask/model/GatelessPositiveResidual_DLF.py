"""Gateless bounded positive-residual specialist for V9.3."""

from __future__ import annotations

from typing import Dict

import torch

from .PositiveResidualSpecialist_DLF import PositiveResidualSpecialist


class GatelessPositiveResidualSpecialist(PositiveResidualSpecialist):
    """Learn only a bounded non-negative correction magnitude.

    V9.2 multiplied the magnitude by a learned gate and collapsed to an almost
    always-off solution.  V9.3 removes that bottleneck: the trainable path
    directly predicts

        0 <= correction(x) <= max_correction.

    The complete V7.1 predictor remains frozen and the final contribution scale
    is selected only on Validation with gamma=0 as a legal fallback.
    """

    def freeze_legacy(self):
        super().freeze_legacy()
        for parameter in self.gate_head.parameters():
            parameter.requires_grad_(False)
        self.gate_head.eval()

    def set_train_mode(self):
        self.eval()
        self.specialist_adapter.train()
        self.magnitude_head.train()
        self.gate_head.eval()

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        output = super().forward(text, audio, vision)
        positive_correction = output["magnitude"]
        output["gate_probability"] = torch.ones_like(positive_correction)
        output["positive_correction"] = positive_correction
        output["specialist_prediction"] = (
            output["legacy_prediction"] + positive_correction
        )
        return output
