"""Frozen-S0 functional trust-region utilities for CFCompatKD v6.

v6 is a single-seed, Valid-only mechanism test motivated by the v5.2 failure
analysis.  The frozen initial Student (S0) is treated as a functional anchor.
On Train events where S0 already beats the frozen ModDrop baseline by a fixed
margin, a one-sided safe trust loss prevents later shared-parameter updates
from making the current Student worse than S0.  The original v4
DISTILL/PRESERVE/ABSTAIN rule is otherwise unchanged.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cfcompat_regret_preserve_utils import (
    NEGATIVE_TRANSFER_MARGIN,
    SEVERE_NEGATIVE_TRANSFER_MARGIN,
    regret_projection_summary,
)
from .cfcompat_student_safe_abstain_utils import student_safe_project_teacher
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_frozen_s0_trust_region_valid_screen_v6"
METHOD = "DLF-Frozen-S0-Trust-Region-CFCompatKD-v6"
OUTPUT_TAG = "cfcompat_frozen_s0_trust_region_v6"
DEV_SEED = 1113
RUN = "frozen_s0_trust_regret_cfcompat"
RUNS = (RUN,)

# Frozen before the v6 single-seed mechanism test.
S0_PROTECT_MARGIN = 0.02
LAMBDA_S0_TRUST = 1.0

# Mechanism-signal checks.  These are not a model-selection grid; v6 trains one
# trajectory only.  They encode the v5.2 hypothesis that useful-S0 cases should
# be protected without giving back the v4 safety gain on the complementary set.
J_MAX_DEGRADATION_VS_V4 = 0.002
BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED = 0.05
NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION = 0.03
OVERALL_NTR_MAX_DEGRADATION = 0.01


def jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def s0_trust_decision(
    current_student: torch.Tensor,
    s0_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Build a one-sided functional trust target around frozen S0.

    S0 is protected only when it beats the frozen baseline by at least the
    pre-registered margin.  The S0 target is projected onto the interval from
    the *current* Student to the label.  Therefore:

    * if current Student is already at least as good as S0, trust loss is zero;
    * if current Student regresses behind S0, S0 becomes a safe corrective target;
    * if S0 lies beyond the label, the target is clipped to the label;
    * non-protected S0 events receive zero trust weight.

    All decisions and targets are detached.  Gradients flow only through the
    caller's current Student output.
    """
    current = current_student.detach().view(-1, 1)
    s0 = s0_prediction.detach().view(-1, 1)
    baseline = baseline_prediction.detach().view(-1, 1)
    target = labels.detach().view(-1, 1)
    if not (current.shape == s0.shape == baseline.shape == target.shape):
        raise ValueError("S0 trust tensors must have identical shapes.")
    if not all(torch.isfinite(value).all() for value in (current, s0, baseline, target)):
        raise FloatingPointError("S0 trust decision received NaN/Inf.")

    current_error = torch.abs(current - target).view(-1)
    s0_error = torch.abs(s0 - target).view(-1)
    baseline_error = torch.abs(baseline - target).view(-1)
    s0_advantage = baseline_error - s0_error
    current_regret_vs_s0 = current_error - s0_error

    s0_protected = s0_advantage >= S0_PROTECT_MARGIN
    safe_target, projection = student_safe_project_teacher(current, s0, target)
    safe_active = projection["active"].view(-1)
    trust_active = s0_protected & safe_active
    trust_gate = trust_active.to(dtype=current.dtype, device=current.device)

    # Diagnostic geometry: did current cross the baseline to the opposite side
    # from S0?  This is the clip-level failure pattern seen in v5.2.
    s0_side = (s0 - baseline).view(-1)
    current_side = (current - baseline).view(-1)
    crossed_baseline_from_s0 = (s0_side * current_side) < 0.0

    return {
        "s0_safe_target": safe_target.detach().view(-1, 1),
        "s0_trust_gate": trust_gate.detach(),
        "s0_trust_active": trust_active.detach(),
        "s0_protected": s0_protected.detach(),
        "s0_error": s0_error.detach(),
        "s0_advantage_vs_baseline": s0_advantage.detach(),
        "current_regret_vs_s0": current_regret_vs_s0.detach(),
        "crossed_baseline_from_s0": crossed_baseline_from_s0.detach(),
        "s0_projection": projection,
    }


def s0_regret_projection_summary(records: Sequence[Mapping]) -> dict:
    """Extend the v4 projection summary with frozen-S0 trust diagnostics."""
    summary = regret_projection_summary(records)
    frame = pd.DataFrame(records)
    required = {
        "mode",
        "s0_prediction",
        "s0_safe_target",
        "s0_error",
        "s0_advantage_vs_baseline",
        "current_regret_vs_s0",
        "s0_protected",
        "s0_trust_active",
        "s0_trust_gate",
        "s0_trust_loss_each",
        "crossed_baseline_from_s0",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("S0 trust records lack columns: {}".format(sorted(missing)))
    numeric_columns = [
        "s0_prediction",
        "s0_safe_target",
        "s0_error",
        "s0_advantage_vs_baseline",
        "current_regret_vs_s0",
        "s0_trust_gate",
        "s0_trust_loss_each",
    ]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("S0 trust records contain NaN/Inf.")

    protected = frame["s0_protected"].astype(bool)
    active = frame["s0_trust_active"].astype(bool)
    crossed = frame["crossed_baseline_from_s0"].astype(bool)
    summary.update(
        {
            "s0_protected_fraction": float(protected.mean()),
            "s0_trust_active_fraction": float(active.mean()),
            "mean_s0_error": float(frame["s0_error"].mean()),
            "mean_s0_advantage_vs_baseline": float(frame["s0_advantage_vs_baseline"].mean()),
            "mean_current_regret_vs_s0": float(frame["current_regret_vs_s0"].mean()),
            "mean_s0_trust_gate": float(frame["s0_trust_gate"].mean()),
            "mean_s0_trust_loss_each": float(frame["s0_trust_loss_each"].mean()),
            "protected_crossed_baseline_fraction": float(
                crossed[protected].mean() if protected.any() else 0.0
            ),
        }
    )
    for mode in MISSING_MODES:
        local = frame.loc[frame["mode"].astype(str).eq(mode)]
        if local.empty:
            raise RuntimeError("S0 trust records contain no {} events.".format(mode))
        local_protected = local["s0_protected"].astype(bool)
        summary.update(
            {
                "{}_s0_protected_fraction".format(mode): float(local_protected.mean()),
                "{}_s0_trust_active_fraction".format(mode): float(
                    local["s0_trust_active"].astype(bool).mean()
                ),
                "{}_protected_crossed_baseline_fraction".format(mode): float(
                    local.loc[local_protected, "crossed_baseline_from_s0"].astype(bool).mean()
                    if local_protected.any()
                    else 0.0
                ),
            }
        )
    return summary


def _subset_transfer(local: pd.DataFrame) -> dict:
    if local.empty:
        raise ValueError("Transfer summary subset is empty.")
    gain = local["gain_vs_dlf"].to_numpy(dtype=np.float64)
    return {
        "N": int(len(local)),
        "mean_gain_vs_baseline": float(gain.mean()),
        "positive_transfer_rate": float((gain > NEGATIVE_TRANSFER_MARGIN).mean()),
        "negative_transfer_rate": float((gain < -NEGATIVE_TRANSFER_MARGIN).mean()),
        "severe_negative_transfer_rate": float(
            (gain < -SEVERE_NEGATIVE_TRANSFER_MARGIN).mean()
        ),
        "not_improved_rate": float((gain <= 0.0).mean()),
    }


def mechanism_transfer_summary(events: pd.DataFrame, run: str) -> dict:
    """Summarize missing-mode transfer by true Teacher usefulness on Valid.

    Valid labels are diagnostic only.  This function never feeds labels into a
    trained gate or changes checkpoint selection.
    """
    required = {"Seed", "Run", "Mode", "teacher_advantage", "gain_vs_dlf"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Valid events lack mechanism columns: {}".format(sorted(missing)))
    local = events.loc[
        events["Seed"].astype(int).eq(DEV_SEED)
        & events["Run"].astype(str).eq(str(run))
        & events["Mode"].astype(str).isin(MISSING_MODES)
    ].copy()
    expected = 229 * len(MISSING_MODES)
    if len(local) != expected:
        raise RuntimeError("Expected {} Seed1113 missing events, found {}.".format(expected, len(local)))
    beneficial = local["teacher_advantage"].to_numpy(dtype=np.float64) >= S0_PROTECT_MARGIN
    return {
        "all_missing": _subset_transfer(local),
        "teacher_beneficial": _subset_transfer(local.loc[beneficial]),
        "teacher_nonbeneficial": _subset_transfer(local.loc[~beneficial]),
        "teacher_beneficial_prevalence": float(beneficial.mean()),
    }


def development_signal_gate(
    candidate_j: float,
    v4_j: float,
    candidate_transfer: Mapping,
    v4_transfer: Mapping,
) -> dict:
    """Apply the one-shot v6 mechanism-signal checks against frozen v4."""
    candidate_b = candidate_transfer["teacher_beneficial"]
    v4_b = v4_transfer["teacher_beneficial"]
    candidate_n = candidate_transfer["teacher_nonbeneficial"]
    v4_n = v4_transfer["teacher_nonbeneficial"]
    candidate_all = candidate_transfer["all_missing"]
    v4_all = v4_transfer["all_missing"]

    beneficial_reduction = (
        float(v4_b["negative_transfer_rate"])
        - float(candidate_b["negative_transfer_rate"])
    )
    nonbeneficial_degradation = (
        float(candidate_n["negative_transfer_rate"])
        - float(v4_n["negative_transfer_rate"])
    )
    overall_degradation = (
        float(candidate_all["negative_transfer_rate"])
        - float(v4_all["negative_transfer_rate"])
    )
    checks = {
        "valid_J_noninferior_to_v4": float(candidate_j)
        <= float(v4_j) + J_MAX_DEGRADATION_VS_V4,
        "beneficial_teacher_NTR_reduction": beneficial_reduction
        >= BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED,
        "nonbeneficial_teacher_NTR_retained": nonbeneficial_degradation
        <= NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION,
        "overall_NTR_not_worse": overall_degradation <= OVERALL_NTR_MAX_DEGRADATION,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_J": float(candidate_j),
        "v4_J": float(v4_j),
        "delta_J_candidate_minus_v4": float(candidate_j) - float(v4_j),
        "beneficial_teacher_NTR_reduction_vs_v4": beneficial_reduction,
        "nonbeneficial_teacher_NTR_degradation_vs_v4": nonbeneficial_degradation,
        "overall_NTR_degradation_vs_v4": overall_degradation,
        "thresholds": {
            "J_max_degradation_vs_v4": J_MAX_DEGRADATION_VS_V4,
            "beneficial_teacher_NTR_reduction_required": BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED,
            "nonbeneficial_teacher_NTR_max_degradation": NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION,
            "overall_NTR_max_degradation": OVERALL_NTR_MAX_DEGRADATION,
        },
    }
