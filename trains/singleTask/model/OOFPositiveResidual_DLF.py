"""Two-stage positive residual specialist for cross-fitted V9.4."""

from __future__ import annotations

from typing import Dict

import torch

from .PositiveResidualSpecialist_DLF import PositiveResidualSpecialist


class OOFPositiveResidualSpecialist(PositiveResidualSpecialist):
    """Keep V7.1 frozen and train magnitude and activation in separate stages."""

    def _freeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def freeze_for_magnitude(self):
        """Train the representation and magnitude without a multiplicative gate."""
        self._freeze_all()
        for module in (self.specialist_adapter, self.magnitude_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.specialist_adapter.train()
        self.magnitude_head.train()

    def freeze_for_gate(self):
        """Freeze the learned magnitude representation and train only the gate."""
        self._freeze_all()
        for parameter in self.gate_head.parameters():
            parameter.requires_grad_(True)
        self.gate_head.train()

    def set_magnitude_train_mode(self):
        self.eval()
        self.specialist_adapter.train()
        self.magnitude_head.train()
        self.gate_head.eval()

    def set_gate_train_mode(self):
        self.eval()
        self.gate_head.train()

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        output = super().forward(text, audio, vision)
        # Expose the ungated branch explicitly for staged training and attribution.
        output["magnitude_only_correction"] = output["magnitude"]
        output["gated_correction"] = (
            output["gate_probability"] * output["magnitude"]
        )
        output["positive_correction"] = output["gated_correction"]
        output["specialist_prediction"] = (
            output["legacy_prediction"] + output["positive_correction"]
        )
        return output
