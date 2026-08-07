"""Sample-conditioned frozen-S0 residual utilities for CFCompatKD v8.

The complete historical Student S0 path is immutable.  Three independent
missing-mode residual heads consume only detached S0 features and add a bounded
scalar correction to S0's missing-modality output.  LAV is never corrected.
"""
from __future__ import annotations

import hashlib
import io
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .missing_utils import MISSING_MODES


VERSION = "cfcompat_sample_conditioned_residual_valid_screen_v8"
METHOD = "DLF-Frozen-S0-Sample-Conditioned-Residual-CFCompatKD-v8"
OUTPUT_TAG = "cfcompat_sample_conditioned_residual_v8"
DEV_SEED = 1113
RUN = "frozen_s0_sample_residual_cfcompat"
RUNS = (RUN,)

RESIDUAL_HIDDEN_DIM = 64
MAX_ABS_RESIDUAL = 1.0

# Frozen before the single v8 trajectory.  v8 should recover complementary
# non-beneficial cases without giving back v7's beneficial-Teacher safety.
J_MAX_DEGRADATION_VS_V7 = 0.002
BENEFICIAL_NTR_MAX_DEGRADATION_VS_V7 = 0.03
NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V7 = 0.10
OVERALL_NTR_MAX_DEGRADATION_VS_V4 = 0.01


def jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def module_state_sha256(module: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _mask_for_mode(modality_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if modality_mask.ndim != 2 or modality_mask.size(1) != 3:
        raise ValueError("modality_mask must have shape [batch, 3].")
    expected = {
        "LA": (1.0, 1.0, 0.0),
        "LV": (1.0, 0.0, 1.0),
        "L": (1.0, 0.0, 0.0),
        "LAV": (1.0, 1.0, 1.0),
    }[mode]
    target = torch.as_tensor(expected, device=modality_mask.device, dtype=modality_mask.dtype)
    return torch.isclose(modality_mask, target.view(1, 3), atol=0.0, rtol=0.0).all(dim=1)


class FrozenS0SampleResidual(nn.Module):
    """Immutable S0 plus three bounded sample-conditioned residual heads.

    Feature construction deliberately uses outputs already produced by the
    frozen S0 path.  No trainable gate or label-dependent inference feature is
    introduced.  Parameter-free LayerNorm reduces sensitivity to global scale
    drift between splits.
    """

    def __init__(
        self,
        s0_student: nn.Module,
        hidden_dim: int = RESIDUAL_HIDDEN_DIM,
        max_abs_residual: float = MAX_ABS_RESIDUAL,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if max_abs_residual <= 0:
            raise ValueError("max_abs_residual must be positive.")
        self.s0 = s0_student
        self.hidden_dim = int(hidden_dim)
        self.max_abs_residual = float(max_abs_residual)

        for parameter in self.s0.parameters():
            parameter.requires_grad_(False)
        self.s0.eval()

        backbone = getattr(self.s0, "backbone", None)
        if backbone is None or not hasattr(backbone, "d_l"):
            raise ValueError("S0 Student must expose backbone.d_l.")
        # c_l_sim/c_v_sim/c_a_sim each have d_l dimensions; append five scalar
        # frozen predictions (c, l/v/a hetero, final S0 output).
        self.feature_dim = int(3 * backbone.d_l + 5)
        self.residual_heads = nn.ModuleDict(
            {
                mode: nn.Sequential(
                    nn.Linear(self.feature_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, 1),
                )
                for mode in MISSING_MODES
            }
        )
        # Exact functional identity at initialization: v8 epoch-0 == S0.
        for head in self.residual_heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        # super().train would otherwise toggle the frozen S0 path.
        self.s0.eval()
        self.residual_heads.train(mode)
        return self

    def _frozen_features(self, base: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required = (
            "c_l_sim",
            "c_v_sim",
            "c_a_sim",
            "logits_c",
            "logits_l_hetero",
            "logits_v_hetero",
            "logits_a_hetero",
            "output_logit",
        )
        missing = [name for name in required if name not in base]
        if missing:
            raise KeyError("Frozen S0 output lacks residual features: {}".format(missing))
        parts = [
            base["c_l_sim"].reshape(base["output_logit"].size(0), -1),
            base["c_v_sim"].reshape(base["output_logit"].size(0), -1),
            base["c_a_sim"].reshape(base["output_logit"].size(0), -1),
            base["logits_c"].reshape(-1, 1),
            base["logits_l_hetero"].reshape(-1, 1),
            base["logits_v_hetero"].reshape(-1, 1),
            base["logits_a_hetero"].reshape(-1, 1),
            base["output_logit"].reshape(-1, 1),
        ]
        features = torch.cat(parts, dim=1).detach()
        if features.size(1) != self.feature_dim:
            raise RuntimeError(
                "Residual feature dimension changed: {} != {}".format(
                    features.size(1), self.feature_dim
                )
            )
        if not torch.isfinite(features).all():
            raise FloatingPointError("Residual features contain NaN/Inf.")
        # No fitted centering/scaling statistics: normalization is within-sample.
        return F.layer_norm(features, (self.feature_dim,))

    def forward(self, text, audio, vision, modality_mask):
        self.s0.eval()
        with torch.no_grad():
            base = self.s0(text, audio, vision, modality_mask)
        features = self._frozen_features(base)
        raw_delta = torch.zeros_like(base["output_logit"].detach())
        # Evaluate only the head that corresponds to each realized missing mode.
        for mode in MISSING_MODES:
            select = _mask_for_mode(modality_mask, mode)
            if bool(select.any()):
                mode_delta = self.residual_heads[mode](features[select])
                raw_delta = raw_delta.index_put((select,), mode_delta)
        # LAV has no selected missing head, hence exactly zero correction.
        delta = self.max_abs_residual * torch.tanh(
            raw_delta / self.max_abs_residual
        )
        output = {key: value.detach() for key, value in base.items()}
        output["s0_output_logit"] = base["output_logit"].detach()
        output["residual_delta"] = delta
        output["output_logit"] = base["output_logit"].detach() + delta
        return output


def residual_parameter_summary(model: FrozenS0SampleResidual) -> dict:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    names = [name for name, _ in trainable]
    expected_prefixes = tuple("residual_heads.{}.".format(m) for m in MISSING_MODES)
    if not names or any(not name.startswith(expected_prefixes) for name in names):
        raise RuntimeError("v8 optimizer scope escaped residual heads: {}".format(names))
    count = int(sum(p.numel() for _, p in trainable))
    total = int(sum(p.numel() for _, p in trainable + frozen))
    return {
        "trainable_names": names,
        "trainable_parameter_count": count,
        "total_parameter_count": total,
        "trainable_parameter_fraction": float(count / total),
        "feature_dim": int(model.feature_dim),
        "hidden_dim": int(model.hidden_dim),
        "max_abs_residual": float(model.max_abs_residual),
    }


def assert_s0_no_gradients(model: FrozenS0SampleResidual):
    offenders = [
        name
        for name, parameter in model.s0.named_parameters()
        if parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError("Frozen S0 received gradients: {}".format(offenders[:8]))


def residual_diagnostic_frame(raw_events: pd.DataFrame, s0_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in s0_predictions.itertuples(index=False):
        for mode in MISSING_MODES:
            rows.append(
                {
                    "sample_index": int(record.sample_index),
                    "Mode": mode,
                    "s0_prediction": float(getattr(record, mode + "_pred")),
                }
            )
    frame = raw_events.loc[raw_events.Mode.astype(str).isin(MISSING_MODES)].merge(
        pd.DataFrame(rows), on=["sample_index", "Mode"], validate="one_to_one"
    )
    frame["residual_delta"] = frame.candidate_prediction - frame.s0_prediction
    frame["abs_residual_delta"] = frame.residual_delta.abs()
    return frame


def development_signal_gate(
    candidate_j: float,
    v7_j: float,
    candidate_transfer: Mapping,
    v7_transfer: Mapping,
    v4_transfer: Mapping,
) -> dict:
    beneficial_degradation = (
        candidate_transfer["teacher_beneficial"]["negative_transfer_rate"]
        - v7_transfer["teacher_beneficial"]["negative_transfer_rate"]
    )
    nonbeneficial_reduction = (
        v7_transfer["teacher_nonbeneficial"]["negative_transfer_rate"]
        - candidate_transfer["teacher_nonbeneficial"]["negative_transfer_rate"]
    )
    overall_degradation_vs_v4 = (
        candidate_transfer["all_missing"]["negative_transfer_rate"]
        - v4_transfer["all_missing"]["negative_transfer_rate"]
    )
    checks = {
        "valid_J_noninferior_to_v7": candidate_j - v7_j <= J_MAX_DEGRADATION_VS_V7,
        "beneficial_teacher_NTR_retained": beneficial_degradation
        <= BENEFICIAL_NTR_MAX_DEGRADATION_VS_V7,
        "nonbeneficial_teacher_NTR_reduced": nonbeneficial_reduction
        >= NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V7,
        "overall_NTR_not_materially_worse_than_v4": overall_degradation_vs_v4
        <= OVERALL_NTR_MAX_DEGRADATION_VS_V4,
    }
    return {
        "candidate_J": float(candidate_j),
        "v7_J": float(v7_j),
        "delta_J_candidate_minus_v7": float(candidate_j - v7_j),
        "beneficial_teacher_NTR_degradation_vs_v7": float(beneficial_degradation),
        "nonbeneficial_teacher_NTR_reduction_vs_v7": float(nonbeneficial_reduction),
        "overall_NTR_degradation_vs_v4": float(overall_degradation_vs_v4),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "thresholds": {
            "J_max_degradation_vs_v7": J_MAX_DEGRADATION_VS_V7,
            "beneficial_NTR_max_degradation_vs_v7": BENEFICIAL_NTR_MAX_DEGRADATION_VS_V7,
            "nonbeneficial_NTR_reduction_required_vs_v7": NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V7,
            "overall_NTR_max_degradation_vs_v4": OVERALL_NTR_MAX_DEGRADATION_VS_V4,
        },
    }
