"""Static dense expert consensus used by the V9.21 audit.

The module has no router and no sample-dependent gate.  A single non-negative
simplex weight vector is applied to every sample and all actions participate.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

MODEL_VERSION = "global_dense_expert_consensus_v921_v1"


class GlobalDenseExpertConsensusV921(nn.Module):
    """Apply one fixed convex weight vector to all action predictions."""

    def __init__(self, weights: Sequence[float] | torch.Tensor):
        super().__init__()
        value = torch.as_tensor(weights, dtype=torch.float32).view(-1)
        if value.numel() < 2:
            raise ValueError("consensus requires at least two actions")
        if not torch.isfinite(value).all():
            raise ValueError("consensus weights contain non-finite values")
        if bool((value < -1e-8).any()):
            raise ValueError("consensus weights must be non-negative")
        total = float(value.sum().item())
        if total <= 0.0:
            raise ValueError("consensus weights must have positive mass")
        value = value.clamp_min(0.0) / value.clamp_min(0.0).sum()
        self.register_buffer("weights", value)

    def forward(self, action_predictions: torch.Tensor) -> torch.Tensor:
        """Return [N,1] consensus predictions from [N,A] or [N,A,1]."""
        values = action_predictions
        if values.dim() == 3 and values.size(-1) == 1:
            values = values.squeeze(-1)
        if values.dim() != 2 or values.size(1) != self.weights.numel():
            raise ValueError(
                f"action_predictions must have shape [N,{self.weights.numel()}]"
            )
        if not torch.isfinite(values).all():
            raise ValueError("action predictions contain non-finite values")
        return (values * self.weights.to(values)).sum(dim=1, keepdim=True)


__all__ = ["MODEL_VERSION", "GlobalDenseExpertConsensusV921"]
