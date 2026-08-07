"""Utilities for Regret-Aware Preserve-or-Distill CFCompatKD v4.1.

v4 improved Seed-1113 overall Valid-J and negative transfer, but lost a large
fraction of the gain on the ``better_and_correct`` Teacher subgroup.  v4.1
keeps the successful preservation mechanism and makes beneficial-Teacher usage
less binary:

* STRONG_DISTILL: Teacher beats the frozen ModDrop baseline by >= 0.02 and the
  Teacher target is current-Student safe.
* WEAK_DISTILL: Teacher beats the frozen baseline by a positive but < 0.02
  margin, moves in the correct baseline-to-label direction, and is current-
  Student safe.  Its KD contribution is fixed at 0.25 of a strong event.
* BENEFICIAL_PAUSE: Teacher beats the frozen baseline, but the current Student
  has already matched/surpassed the Teacher, so KD is paused rather than pulling
  the Student backward.  Because routing is recomputed every batch, the same
  Teacher automatically re-enters STRONG/WEAK_DISTILL if the Student later
  regresses behind it.
* PRESERVE: Teacher is not beneficial and the current Student has regressed
  from the frozen ModDrop baseline by >= 0.02.
* ABSTAIN: none of the above.

A custom two-tier KD reduction is required.  The original gated KD divides by
``sum(gate)``; multiplying every weak gate by 0.25 would therefore cancel when
a batch contains only weak events.  v4.1 divides by the *unscaled* eligible
CFCompat mass while applying the weak scale only in the numerator.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cfcompat_regret_preserve_utils import (
    LAMBDA_PRESERVE,
    MILD_CFCOMPAT_BASE,
    MILD_CFCOMPAT_SCALE,
    NEGATIVE_TRANSFER_MARGIN,
    PRESERVE_MARGIN,
    SEVERE_NEGATIVE_TRANSFER_MARGIN,
    mild_cfcompat,
    negative_transfer_summary,
)
from .cfcompat_student_safe_abstain_utils import student_projection_summary, student_safe_project_teacher
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_regret_preserve_beneficial_guard_valid_screen_v4p1"
METHOD = "DLF-Regret-Aware-Preserve-or-Distill-CFCompatKD-v4.1"
OUTPUT_TAG = "cfcompat_regret_preserve_guard_v4p1"
DEV_SEED = 1113
RUN = "regret_preserve_guard_cfcompat"
RUNS = (RUN,)

STRONG_DISTILL_MARGIN = 0.02
WEAK_DISTILL_SCALE = 0.25
# Preserve the successful v4 values exactly.
PRESERVE_MARGIN_V4P1 = PRESERVE_MARGIN
LAMBDA_PRESERVE_V4P1 = LAMBDA_PRESERVE
MILD_CFCOMPAT_BASE_V4P1 = MILD_CFCOMPAT_BASE
MILD_CFCOMPAT_SCALE_V4P1 = MILD_CFCOMPAT_SCALE

# Frozen post-v4 single-seed development criteria.  v4.1 is specifically a
# targeted repair of good-Teacher utilization; it must recover that subgroup
# without giving back the main v4 protection gains.
J_MAX_DEGRADATION_VS_V4 = 0.002
BETTER_CORRECT_GAIN_IMPROVEMENT_REQUIRED = 0.010
Q1_Q4_MAX_GAIN_DEGRADATION_VS_V4 = 0.005
NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4 = 0.010
POSITIVE_TRANSFER_MAX_DECREASE_VS_V4 = 0.010
HARMFUL_IMITATION_MAX_INCREASE_VS_V4 = 0.010
SUPPORTING_EPOCHS = 2


def tiered_kd_loss(
    student_prediction: torch.Tensor,
    target_prediction: torch.Tensor,
    strong_gate: torch.Tensor,
    weak_gate: torch.Tensor,
):
    """Return KD with true weak-event attenuation.

    ``strong_gate`` and ``weak_gate`` already contain the mild CFCompat prior.
    The denominator uses their *unscaled* eligible mass.  The numerator applies
    ``WEAK_DISTILL_SCALE`` only to weak events.  Therefore all-strong batches
    reproduce the v4 normalization, while all-weak batches are genuinely 0.25x.
    """
    student = student_prediction.view(-1)
    target = target_prediction.detach().view(-1)
    strong = strong_gate.detach().view(-1).to(student)
    weak = weak_gate.detach().view(-1).to(student)
    if not (student.shape == target.shape == strong.shape == weak.shape):
        raise ValueError("Tiered KD tensors must have identical shapes.")
    if (
        not torch.isfinite(strong).all()
        or not torch.isfinite(weak).all()
        or torch.any(strong < 0.0)
        or torch.any(weak < 0.0)
        or torch.any((strong > 0.0) & (weak > 0.0))
    ):
        raise FloatingPointError("Tiered KD gates are invalid.")
    each = F.smooth_l1_loss(student, target, reduction="none")
    eligible_mass = strong + weak
    effective = strong + WEAK_DISTILL_SCALE * weak
    loss = torch.sum(effective * each) / (torch.sum(eligible_mass) + 1e-8)
    if not torch.isfinite(loss):
        raise FloatingPointError("Tiered KD loss is non-finite.")
    return loss, each, eligible_mass.detach(), effective.detach()


def regret_preserve_guard_decision(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    labels: torch.Tensor,
    compatibility: torch.Tensor,
) -> dict:
    """Compute exclusive v4.1 routing and detached targets/gates."""
    student = student_prediction.detach().view(-1, 1)
    teacher = teacher_prediction.detach().view(-1, 1)
    baseline = baseline_prediction.detach().view(-1, 1)
    target = labels.detach().view(-1, 1)
    compat = compatibility.detach().view(-1)
    if not (student.shape == teacher.shape == baseline.shape == target.shape):
        raise ValueError("v4.1 decision tensors must have identical shapes.")
    if len(compat) != len(student):
        raise ValueError("Compatibility length does not match predictions.")
    if not all(torch.isfinite(value).all() for value in (student, teacher, baseline, target, compat)):
        raise FloatingPointError("v4.1 decision received NaN/Inf.")

    teacher_safe, teacher_projection = student_safe_project_teacher(student, teacher, target)
    preserve_safe, _ = student_safe_project_teacher(student, baseline, target)

    baseline_error = torch.abs(baseline - target).view(-1)
    teacher_error = torch.abs(teacher - target).view(-1)
    current_error = torch.abs(student - target).view(-1)
    teacher_advantage = baseline_error - teacher_error
    current_regret = current_error - baseline_error
    baseline_direction_correct = (
        ((teacher - baseline) * (target - baseline)).view(-1) > 0.0
    )
    safe_teacher_active = teacher_projection["active"].view(-1)

    teacher_better_any = teacher_advantage > 0.0
    strong_candidate = teacher_advantage >= STRONG_DISTILL_MARGIN
    weak_candidate = (
        teacher_better_any
        & (teacher_advantage < STRONG_DISTILL_MARGIN)
        & baseline_direction_correct
    )
    strong_distill = strong_candidate & safe_teacher_active
    weak_distill = weak_candidate & safe_teacher_active
    beneficial_pause = teacher_better_any & (~safe_teacher_active)
    current_regressed = current_regret >= PRESERVE_MARGIN_V4P1
    # Preserve only when the frozen Teacher is not genuinely better than the
    # frozen baseline.  Beneficial Teachers are paused/re-enabled dynamically,
    # never replaced by a worse baseline anchor.
    preserve = (~teacher_better_any) & current_regressed
    abstain = ~(strong_distill | weak_distill | beneficial_pause | preserve)

    stacked = torch.stack(
        [strong_distill, weak_distill, beneficial_pause, preserve, abstain], dim=0
    ).to(torch.int64)
    if not torch.all(stacked.sum(dim=0) == 1):
        raise AssertionError("v4.1 routing is not mutually exclusive/exhaustive.")

    mild = mild_cfcompat(compat)
    strong_gate = strong_distill.to(compat) * mild
    weak_gate = weak_distill.to(compat) * mild
    preserve_gate = preserve.to(compat)

    return {
        "teacher_safe_target": teacher_safe.detach().view(-1, 1),
        "preserve_safe_target": preserve_safe.detach().view(-1, 1),
        "strong_gate": strong_gate.detach(),
        "weak_gate": weak_gate.detach(),
        "preserve_gate": preserve_gate.detach(),
        "strong_distill": strong_distill.detach(),
        "weak_distill": weak_distill.detach(),
        "beneficial_pause": beneficial_pause.detach(),
        "preserve": preserve.detach(),
        "abstain": abstain.detach(),
        "teacher_better_any": teacher_better_any.detach(),
        "strong_candidate": strong_candidate.detach(),
        "weak_candidate": weak_candidate.detach(),
        "baseline_direction_correct": baseline_direction_correct.detach(),
        "current_regressed": current_regressed.detach(),
        "baseline_error": baseline_error.detach(),
        "teacher_error": teacher_error.detach(),
        "current_error": current_error.detach(),
        "teacher_advantage_vs_baseline": teacher_advantage.detach(),
        "current_regret_vs_baseline": current_regret.detach(),
        "mild_compatibility": mild.detach(),
        "teacher_projection": teacher_projection,
    }


def regret_guard_projection_summary(records: Sequence[Mapping]) -> dict:
    if not records:
        raise ValueError("v4.1 projection summary requires records.")
    summary = student_projection_summary(records)
    frame = pd.DataFrame(records)
    required = {
        "mode", "strong_distill", "weak_distill", "beneficial_pause",
        "preserve", "guard_abstain", "teacher_better_any",
        "strong_candidate", "weak_candidate", "baseline_direction_correct",
        "current_regressed", "baseline_error", "teacher_error", "current_error",
        "teacher_advantage_vs_baseline", "current_regret_vs_baseline",
        "compatibility", "mild_compatibility", "strong_gate", "weak_gate",
        "eligible_distill_mass", "effective_distill_gate", "preserve_gate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("v4.1 records lack columns: {}".format(sorted(missing)))
    numeric_cols = [
        "baseline_error", "teacher_error", "current_error",
        "teacher_advantage_vs_baseline", "current_regret_vs_baseline",
        "compatibility", "mild_compatibility", "strong_gate", "weak_gate",
        "eligible_distill_mass", "effective_distill_gate", "preserve_gate",
    ]
    if not np.isfinite(frame[numeric_cols].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("v4.1 records contain non-finite values.")

    states = frame[[
        "strong_distill", "weak_distill", "beneficial_pause", "preserve", "guard_abstain"
    ]].astype(bool).to_numpy(dtype=np.int64)
    if np.any(states.sum(axis=1) != 1):
        raise RuntimeError("v4.1 recorded state accounting is invalid.")

    summary.update({
        "strong_distill_fraction": float(frame.strong_distill.astype(bool).mean()),
        "weak_distill_fraction": float(frame.weak_distill.astype(bool).mean()),
        "beneficial_pause_fraction": float(frame.beneficial_pause.astype(bool).mean()),
        "preserve_fraction": float(frame.preserve.astype(bool).mean()),
        "guard_abstain_fraction": float(frame.guard_abstain.astype(bool).mean()),
        "teacher_better_any_fraction": float(frame.teacher_better_any.astype(bool).mean()),
        "strong_candidate_fraction": float(frame.strong_candidate.astype(bool).mean()),
        "weak_candidate_fraction": float(frame.weak_candidate.astype(bool).mean()),
        "mean_teacher_advantage_vs_baseline": float(frame.teacher_advantage_vs_baseline.mean()),
        "mean_current_regret_vs_baseline": float(frame.current_regret_vs_baseline.mean()),
        "mean_eligible_distill_mass": float(frame.eligible_distill_mass.mean()),
        "mean_effective_distill_gate": float(frame.effective_distill_gate.mean()),
        "mean_preserve_gate": float(frame.preserve_gate.mean()),
    })
    total = sum(summary[key] for key in (
        "strong_distill_fraction", "weak_distill_fraction", "beneficial_pause_fraction",
        "preserve_fraction", "guard_abstain_fraction"
    ))
    if abs(total - 1.0) > 1e-12:
        raise RuntimeError("v4.1 state fractions do not sum to one.")

    for mode in MISSING_MODES:
        local = frame.loc[frame["mode"].astype(str).eq(mode)]
        if local.empty:
            raise RuntimeError("v4.1 records contain no {} events.".format(mode))
        for column, suffix in (
            ("strong_distill", "strong_distill_fraction"),
            ("weak_distill", "weak_distill_fraction"),
            ("beneficial_pause", "beneficial_pause_fraction"),
            ("preserve", "preserve_fraction"),
            ("guard_abstain", "abstain_fraction"),
        ):
            summary["{}_{}".format(mode, suffix)] = float(local[column].astype(bool).mean())
    return summary


def _group_value(groups: pd.DataFrame, run: str, group_type: str, group_value: str, metric: str) -> float:
    selected = groups.loc[
        groups.Seed.astype(str).eq(str(DEV_SEED))
        & groups.Run.astype(str).eq(str(run))
        & groups.GroupType.astype(str).eq(str(group_type))
        & groups.GroupValue.astype(str).eq(str(group_value))
    ]
    if len(selected) != 1:
        raise RuntimeError("Group metric is not unique for {} {}={}.".format(run, group_type, group_value))
    return float(selected.iloc[0][metric])


def _negative_row(frame: pd.DataFrame, run: str) -> Mapping:
    selected = frame.loc[
        frame.Seed.astype(int).eq(DEV_SEED)
        & frame.Run.astype(str).eq(str(run))
        & frame.Mode.astype(str).eq("MISSING_ALL")
    ]
    if len(selected) != 1:
        raise RuntimeError("Missing unique MISSING_ALL transfer row for {}.".format(run))
    return selected.iloc[0].to_dict()


def dev_candidate_gate(candidate_row, v4_grid, v2_grid, groups, transfer, epoch_rows) -> dict:
    """Frozen post-v4 Seed-1113 development gate for the targeted repair."""
    def grid_row(frame, run):
        selected = frame.loc[
            frame.Seed.astype(int).eq(DEV_SEED) & frame.Run.astype(str).eq(run)
        ]
        if len(selected) != 1:
            raise RuntimeError("No unique grid row for {}.".format(run))
        return selected.iloc[0].to_dict()

    v4 = grid_row(v4_grid, "regret_preserve_cfcompat")
    uniform = grid_row(v2_grid, "student_safe_uniform")
    replay = grid_row(v2_grid, "cfcompat_replay")
    candidate_j = float(candidate_row["J_valid"])
    v4_j = float(v4["J_valid"])

    better_v4 = _group_value(groups, "regret_preserve_cfcompat", "teacher_condition", "better_and_correct", "gain_vs_DLF")
    better_candidate = _group_value(groups, RUN, "teacher_condition", "better_and_correct", "gain_vs_DLF")
    q1_v4 = _group_value(groups, "regret_preserve_cfcompat", "baseline_error_quartile", "Q1_easy", "gain_vs_DLF")
    q1_candidate = _group_value(groups, RUN, "baseline_error_quartile", "Q1_easy", "gain_vs_DLF")
    q4_v4 = _group_value(groups, "regret_preserve_cfcompat", "baseline_error_quartile", "Q4_hard", "gain_vs_DLF")
    q4_candidate = _group_value(groups, RUN, "baseline_error_quartile", "Q4_hard", "gain_vs_DLF")
    harm_v4 = _group_value(groups, "regret_preserve_cfcompat", "all", "ALL", "harmful_imitation_rate")
    harm_candidate = _group_value(groups, RUN, "all", "ALL", "harmful_imitation_rate")

    transfer_v4 = _negative_row(transfer, "regret_preserve_cfcompat")
    transfer_candidate = _negative_row(transfer, RUN)
    supporting = int(sum(
        int(row["Seed"]) == DEV_SEED
        and str(row["Run"]) == RUN
        and float(row["J_valid"]) <= v4_j + J_MAX_DEGRADATION_VS_V4
        for row in epoch_rows
    ))

    fractions = {
        "strong_distill": float(candidate_row["projection_strong_distill_fraction"]),
        "weak_distill": float(candidate_row["projection_weak_distill_fraction"]),
        "beneficial_pause": float(candidate_row["projection_beneficial_pause_fraction"]),
        "preserve": float(candidate_row["projection_preserve_fraction"]),
        "abstain": float(candidate_row["projection_guard_abstain_fraction"]),
    }
    checks = {
        "J_not_materially_worse_than_v4": candidate_j - v4_j <= J_MAX_DEGRADATION_VS_V4,
        "better_correct_gain_improves_vs_v4_by_0p01": better_candidate - better_v4 >= BETTER_CORRECT_GAIN_IMPROVEMENT_REQUIRED,
        "Q1_not_materially_degraded_vs_v4": q1_candidate >= q1_v4 - Q1_Q4_MAX_GAIN_DEGRADATION_VS_V4,
        "Q4_not_materially_degraded_vs_v4": q4_candidate >= q4_v4 - Q1_Q4_MAX_GAIN_DEGRADATION_VS_V4,
        "negative_transfer_not_worse_than_v4_by_0p01": float(transfer_candidate["negative_transfer_rate"]) <= float(transfer_v4["negative_transfer_rate"]) + NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4,
        "positive_transfer_not_worse_than_v4_by_0p01": float(transfer_candidate["positive_transfer_rate"]) >= float(transfer_v4["positive_transfer_rate"]) - POSITIVE_TRANSFER_MAX_DECREASE_VS_V4,
        "harmful_imitation_not_worse_than_v4_by_0p01": harm_candidate <= harm_v4 + HARMFUL_IMITATION_MAX_INCREASE_VS_V4,
        "two_noninferior_epochs_vs_v4": supporting >= SUPPORTING_EPOCHS,
        "strong_and_weak_distill_exercised": fractions["strong_distill"] > 0.0 and fractions["weak_distill"] > 0.0,
        "preserve_and_non_distill_exercised": fractions["preserve"] > 0.0 and (fractions["beneficial_pause"] + fractions["abstain"]) > 0.0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "candidate_J": candidate_j,
        "v4_J": v4_j,
        "uniform_J": float(uniform["J_valid"]),
        "replay_J": float(replay["J_valid"]),
        "J_delta_vs_v4": candidate_j - v4_j,
        "better_correct_gain_v4": better_v4,
        "better_correct_gain_candidate": better_candidate,
        "better_correct_gain_improvement": better_candidate - better_v4,
        "Q1_gain_v4": q1_v4,
        "Q1_gain_candidate": q1_candidate,
        "Q4_gain_v4": q4_v4,
        "Q4_gain_candidate": q4_candidate,
        "v4_negative_transfer_rate": float(transfer_v4["negative_transfer_rate"]),
        "candidate_negative_transfer_rate": float(transfer_candidate["negative_transfer_rate"]),
        "v4_positive_transfer_rate": float(transfer_v4["positive_transfer_rate"]),
        "candidate_positive_transfer_rate": float(transfer_candidate["positive_transfer_rate"]),
        "v4_harmful_imitation_rate": harm_v4,
        "candidate_harmful_imitation_rate": harm_candidate,
        "supporting_epoch_count": supporting,
        "decision_fractions": fractions,
    }


def frozen_thresholds() -> dict:
    return {
        "strong_distill_margin": STRONG_DISTILL_MARGIN,
        "weak_distill_scale": WEAK_DISTILL_SCALE,
        "preserve_margin": PRESERVE_MARGIN_V4P1,
        "lambda_preserve": LAMBDA_PRESERVE_V4P1,
        "mild_cfcompat_base": MILD_CFCOMPAT_BASE_V4P1,
        "mild_cfcompat_scale": MILD_CFCOMPAT_SCALE_V4P1,
        "negative_transfer_margin": NEGATIVE_TRANSFER_MARGIN,
        "severe_negative_transfer_margin": SEVERE_NEGATIVE_TRANSFER_MARGIN,
        "J_max_degradation_vs_v4": J_MAX_DEGRADATION_VS_V4,
        "better_correct_gain_improvement_required": BETTER_CORRECT_GAIN_IMPROVEMENT_REQUIRED,
        "Q1_Q4_max_gain_degradation_vs_v4": Q1_Q4_MAX_GAIN_DEGRADATION_VS_V4,
        "negative_transfer_max_increase_vs_v4": NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4,
        "positive_transfer_max_decrease_vs_v4": POSITIVE_TRANSFER_MAX_DECREASE_VS_V4,
        "harmful_imitation_max_increase_vs_v4": HARMFUL_IMITATION_MAX_INCREASE_VS_V4,
        "supporting_epochs": SUPPORTING_EPOCHS,
    }


__all__ = [
    "DEV_SEED", "METHOD", "OUTPUT_TAG", "RUN", "RUNS", "VERSION",
    "STRONG_DISTILL_MARGIN", "WEAK_DISTILL_SCALE", "PRESERVE_MARGIN_V4P1",
    "LAMBDA_PRESERVE_V4P1", "frozen_thresholds", "dev_candidate_gate",
    "negative_transfer_summary", "regret_guard_projection_summary",
    "regret_preserve_guard_decision", "tiered_kd_loss",
]
