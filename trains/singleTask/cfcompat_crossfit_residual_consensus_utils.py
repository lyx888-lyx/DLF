"""Cross-fit residual consensus utilities for CFCompatKD v9.

v9 keeps the entire S0 function immutable and reuses the exact v8 residual
architecture. Five residual head banks are trained on video-grouped Train
subsets. At inference a correction is applied only when at least four of five
independently trained banks agree on its sign; the robust median correction is
used. No labels, fitted split statistics, or learned gate are used at inference.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .missing_utils import MISSING_MODES
from .cfcompat_sample_residual_utils import (
    MAX_ABS_RESIDUAL,
    RESIDUAL_HIDDEN_DIM,
    module_state_sha256,
)


VERSION = "cfcompat_crossfit_residual_consensus_valid_screen_v9"
METHOD = "DLF-Frozen-S0-Crossfit-Residual-Consensus-CFCompatKD-v9"
OUTPUT_TAG = "cfcompat_crossfit_residual_consensus_v9"
DEV_SEED = 1113
RUN = "frozen_s0_crossfit_residual_consensus"
N_FOLDS = 5
CONSENSUS_MIN_AGREE = 4
RESIDUAL_INIT_SEED = 20260807

# Frozen before the v9 run.
J_MAX_DEGRADATION_VS_V8 = 0.002
BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8 = 0.03
NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8 = 0.05
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


def sample_video_id(sample_id: str) -> str:
    value = str(sample_id)
    return value.split("$_$", 1)[0] if "$_$" in value else value


def deterministic_video_group_folds(
    sample_ids: Sequence[str], n_folds: int = N_FOLDS
) -> pd.DataFrame:
    """Assign complete videos to deterministic approximately balanced folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    groups: Dict[str, list] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[sample_video_id(sample_id)].append(index)
    if len(groups) < n_folds:
        raise ValueError("Not enough video groups for cross-fit folds.")

    def tie_key(video_id: str) -> str:
        return hashlib.sha1(video_id.encode("utf-8")).hexdigest()

    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), tie_key(item[0])))
    fold_sizes = [0 for _ in range(n_folds)]
    fold_groups = [0 for _ in range(n_folds)]
    assignment = {}
    for video_id, indices in ordered:
        fold = min(range(n_folds), key=lambda f: (fold_sizes[f], fold_groups[f], f))
        fold_sizes[fold] += len(indices)
        fold_groups[fold] += 1
        assignment[video_id] = fold

    rows = []
    for index, sample_id in enumerate(sample_ids):
        video_id = sample_video_id(sample_id)
        rows.append(
            {
                "sample_index": int(index),
                "sample_id": str(sample_id),
                "video_id": video_id,
                "fold": int(assignment[video_id]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.sample_index.nunique() != len(sample_ids):
        raise RuntimeError("Cross-fit fold assignment lost samples.")
    if frame.groupby("video_id").fold.nunique().max() != 1:
        raise RuntimeError("A video leaked across cross-fit folds.")
    if set(frame.fold.astype(int)) != set(range(n_folds)):
        raise RuntimeError("Cross-fit produced an empty fold.")
    return frame


def _mask_for_mode(modality_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if modality_mask.ndim != 2 or modality_mask.size(1) != 3:
        raise ValueError("modality_mask must have shape [batch, 3].")
    expected = {
        "LA": (1.0, 1.0, 0.0),
        "LV": (1.0, 0.0, 1.0),
        "L": (1.0, 0.0, 0.0),
        "LAV": (1.0, 1.0, 1.0),
    }[mode]
    target = torch.as_tensor(
        expected, device=modality_mask.device, dtype=modality_mask.dtype
    )
    return torch.isclose(
        modality_mask, target.view(1, 3), atol=0.0, rtol=0.0
    ).all(dim=1)


def frozen_feature_dim(s0_student: nn.Module) -> int:
    backbone = getattr(s0_student, "backbone", None)
    if backbone is None or not hasattr(backbone, "d_l"):
        raise ValueError("S0 Student must expose backbone.d_l.")
    return int(3 * backbone.d_l + 5)


def frozen_features_from_base(
    base: Mapping[str, torch.Tensor], feature_dim: int
) -> torch.Tensor:
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
    batch = base["output_logit"].size(0)
    parts = [
        base["c_l_sim"].reshape(batch, -1),
        base["c_v_sim"].reshape(batch, -1),
        base["c_a_sim"].reshape(batch, -1),
        base["logits_c"].reshape(-1, 1),
        base["logits_l_hetero"].reshape(-1, 1),
        base["logits_v_hetero"].reshape(-1, 1),
        base["logits_a_hetero"].reshape(-1, 1),
        base["output_logit"].reshape(-1, 1),
    ]
    features = torch.cat(parts, dim=1).detach()
    if features.size(1) != int(feature_dim):
        raise RuntimeError(
            "Residual feature dimension changed: {} != {}".format(
                features.size(1), feature_dim
            )
        )
    if not torch.isfinite(features).all():
        raise FloatingPointError("Residual features contain NaN/Inf.")
    return F.layer_norm(features, (int(feature_dim),))


class ResidualHeadBank(nn.Module):
    """One v8-compatible bank of LA/LV/L residual heads."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = RESIDUAL_HIDDEN_DIM,
        max_abs_residual: float = MAX_ABS_RESIDUAL,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_abs_residual = float(max_abs_residual)
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
        for head in self.residual_heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def delta(
        self, features: torch.Tensor, modality_mask: torch.Tensor
    ) -> torch.Tensor:
        raw = torch.zeros(
            (features.size(0), 1), device=features.device, dtype=features.dtype
        )
        for mode in MISSING_MODES:
            select = _mask_for_mode(modality_mask, mode)
            if bool(select.any()):
                raw = raw.index_put((select,), self.residual_heads[mode](features[select]))
        return self.max_abs_residual * torch.tanh(raw / self.max_abs_residual)


class FrozenS0FoldResidual(nn.Module):
    """Training wrapper: immutable S0 plus one residual head bank."""

    def __init__(
        self,
        s0_student: nn.Module,
        hidden_dim: int = RESIDUAL_HIDDEN_DIM,
        max_abs_residual: float = MAX_ABS_RESIDUAL,
    ):
        super().__init__()
        self.s0 = s0_student
        for parameter in self.s0.parameters():
            parameter.requires_grad_(False)
        self.s0.eval()
        self.feature_dim = frozen_feature_dim(self.s0)
        self.bank = ResidualHeadBank(
            self.feature_dim, hidden_dim=hidden_dim, max_abs_residual=max_abs_residual
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.s0.eval()
        self.bank.train(mode)
        return self

    def forward(self, text, audio, vision, modality_mask):
        self.s0.eval()
        with torch.no_grad():
            base = self.s0(text, audio, vision, modality_mask)
        features = frozen_features_from_base(base, self.feature_dim)
        delta = self.bank.delta(features, modality_mask)
        output = {key: value.detach() for key, value in base.items()}
        output["s0_output_logit"] = base["output_logit"].detach()
        output["residual_delta"] = delta
        output["output_logit"] = base["output_logit"].detach() + delta
        return output


class FrozenS0CrossfitConsensus(nn.Module):
    """Inference wrapper over K frozen residual banks with 4-of-5 sign consensus."""

    def __init__(
        self,
        s0_student: nn.Module,
        bank_state_dicts: Sequence[Mapping[str, torch.Tensor]],
        hidden_dim: int = RESIDUAL_HIDDEN_DIM,
        max_abs_residual: float = MAX_ABS_RESIDUAL,
        min_agree: int = CONSENSUS_MIN_AGREE,
    ):
        super().__init__()
        if len(bank_state_dicts) != N_FOLDS:
            raise ValueError("v9 requires exactly {} fold banks.".format(N_FOLDS))
        if min_agree <= N_FOLDS // 2 or min_agree > N_FOLDS:
            raise ValueError("min_agree must represent a strict fold majority.")
        self.s0 = s0_student
        for parameter in self.s0.parameters():
            parameter.requires_grad_(False)
        self.s0.eval()
        self.feature_dim = frozen_feature_dim(self.s0)
        self.min_agree = int(min_agree)
        self.max_abs_residual = float(max_abs_residual)
        self.fold_banks = nn.ModuleList()
        for state in bank_state_dicts:
            bank = ResidualHeadBank(
                self.feature_dim,
                hidden_dim=hidden_dim,
                max_abs_residual=max_abs_residual,
            )
            bank.load_state_dict(dict(state), strict=True)
            for parameter in bank.parameters():
                parameter.requires_grad_(False)
            bank.eval()
            self.fold_banks.append(bank)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.s0.eval()
        self.fold_banks.eval()
        return self

    def forward(self, text, audio, vision, modality_mask):
        self.s0.eval()
        self.fold_banks.eval()
        with torch.no_grad():
            base = self.s0(text, audio, vision, modality_mask)
            features = frozen_features_from_base(base, self.feature_dim)
            fold_deltas = torch.stack(
                [bank.delta(features, modality_mask) for bank in self.fold_banks],
                dim=0,
            )
            positive = (fold_deltas > 0.0).sum(dim=0)
            negative = (fold_deltas < 0.0).sum(dim=0)
            agree_count = torch.maximum(positive, negative)
            applied = agree_count >= self.min_agree
            median_delta = torch.median(fold_deltas, dim=0).values
            consensus_delta = torch.where(
                applied, median_delta, torch.zeros_like(median_delta)
            )
            fold_std = torch.std(fold_deltas, dim=0, unbiased=False)
            sign_agreement = agree_count.to(median_delta.dtype) / float(N_FOLDS)

        output = {key: value.detach() for key, value in base.items()}
        output["s0_output_logit"] = base["output_logit"].detach()
        output["fold_residual_deltas"] = fold_deltas.detach()
        output["median_residual_delta"] = median_delta.detach()
        output["consensus_applied"] = applied.detach()
        output["consensus_sign_agreement"] = sign_agreement.detach()
        output["consensus_fold_std"] = fold_std.detach()
        output["residual_delta"] = consensus_delta.detach()
        output["output_logit"] = base["output_logit"].detach() + consensus_delta.detach()
        return output


def bank_state_cpu(model: FrozenS0FoldResidual) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.bank.state_dict().items()
    }


def assert_s0_no_gradients(model: FrozenS0FoldResidual):
    offenders = [
        name
        for name, parameter in model.s0.named_parameters()
        if parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError("Frozen S0 received gradients: {}".format(offenders[:8]))


def consensus_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("Consensus diagnostic frame is empty.")
    missing = frame.loc[frame.Mode.astype(str).isin(MISSING_MODES)].copy()
    if missing.empty:
        raise ValueError("No missing-mode rows in consensus diagnostic frame.")
    return {
        "N": int(len(missing)),
        "consensus_applied_rate": float(missing.consensus_applied.mean()),
        "mean_sign_agreement": float(missing.consensus_sign_agreement.mean()),
        "mean_fold_std": float(missing.consensus_fold_std.mean()),
        "mean_abs_consensus_delta": float(missing.abs_consensus_delta.mean()),
        "p95_abs_consensus_delta": float(
            np.quantile(missing.abs_consensus_delta.to_numpy(float), 0.95)
        ),
        "max_abs_consensus_delta": float(missing.abs_consensus_delta.max()),
    }


def development_signal_gate(
    candidate_j: float,
    v8_j: float,
    candidate_transfer: Mapping,
    v8_transfer: Mapping,
    v4_transfer: Mapping,
) -> dict:
    beneficial_degradation = (
        candidate_transfer["teacher_beneficial"]["negative_transfer_rate"]
        - v8_transfer["teacher_beneficial"]["negative_transfer_rate"]
    )
    nonbeneficial_reduction = (
        v8_transfer["teacher_nonbeneficial"]["negative_transfer_rate"]
        - candidate_transfer["teacher_nonbeneficial"]["negative_transfer_rate"]
    )
    overall_degradation_vs_v4 = (
        candidate_transfer["all_missing"]["negative_transfer_rate"]
        - v4_transfer["all_missing"]["negative_transfer_rate"]
    )
    checks = {
        "valid_J_noninferior_to_v8": candidate_j - v8_j <= J_MAX_DEGRADATION_VS_V8,
        "beneficial_teacher_NTR_retained": beneficial_degradation
        <= BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
        "nonbeneficial_teacher_NTR_reduced": nonbeneficial_reduction
        >= NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
        "overall_NTR_not_materially_worse_than_v4": overall_degradation_vs_v4
        <= OVERALL_NTR_MAX_DEGRADATION_VS_V4,
    }
    return {
        "candidate_J": float(candidate_j),
        "v8_J": float(v8_j),
        "delta_J_candidate_minus_v8": float(candidate_j - v8_j),
        "beneficial_teacher_NTR_degradation_vs_v8": float(beneficial_degradation),
        "nonbeneficial_teacher_NTR_reduction_vs_v8": float(nonbeneficial_reduction),
        "overall_NTR_degradation_vs_v4": float(overall_degradation_vs_v4),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "thresholds": {
            "J_max_degradation_vs_v8": J_MAX_DEGRADATION_VS_V8,
            "beneficial_NTR_max_degradation_vs_v8": BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
            "nonbeneficial_NTR_reduction_required_vs_v8": NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
            "overall_NTR_max_degradation_vs_v4": OVERALL_NTR_MAX_DEGRADATION_VS_V4,
        },
    }
