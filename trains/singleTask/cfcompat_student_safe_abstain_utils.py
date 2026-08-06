"""Frozen utilities for the current-Student safe-abstention CFCompatKD v2 screen.

The candidate KD target is the frozen full-modality Teacher prediction clipped
to the closed interval between the current missing-modality Student prediction
(detached) and the training label.  Samples whose clipped target equals the
current Student prediction explicitly abstain from KD by receiving gate zero.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cfcompat_safe_projection_utils import projection_summary
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_student_safe_abstain_valid_screen_v2"
METHOD = "DLF-Student-Safe-Abstain-CFCompat-v2"
OUTPUT_TAG = "cfcompat_student_safe_abstain_v2"
FORMAL_SEEDS = (1112, 1113, 1115)
RUNS = (
    "cfcompat_replay",
    "student_safe_uniform",
    "student_safe_cfcompat",
)
CANDIDATE_RUNS = RUNS[1:]
REPLAY_TOLERANCE = 1e-4

# Exploratory v2 non-inferiority/safety rules, frozen before v2 training.
MEAN_J_MAX_DEGRADATION = 0.003
PER_SEED_J_MAX_DEGRADATION = 0.005
PER_MODE_MAX_DEGRADATION = 0.005
MEAN_HARMFUL_REDUCTION_REQUIRED = 0.02
Q1_MAX_DEGRADATION = 0.002
Q4_GAIN_RETENTION_FRACTION = 0.80
BETTER_CORRECT_MAX_DEGRADATION = 0.002
SUPPORTING_EPOCHS = 2
PROJECTION_EPS = 1e-12


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


def student_safe_project_teacher(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    labels: torch.Tensor,
):
    """Clip Teacher to the current-Student-to-label interval and mark abstention.

    The Student endpoint is detached, so the target and gate never create a
    gradient path through the interval construction.  ``active`` is true only
    when the clipped target differs from the current Student prediction.
    Wrong-direction, equal-target, and zero-width cases therefore abstain.
    """
    student = student_prediction.detach().view(-1, 1)
    teacher = teacher_prediction.detach().view(-1, 1)
    target = labels.detach().view(-1, 1)
    if not (student.shape == teacher.shape == target.shape):
        raise ValueError("Student-safe projection tensors must have identical shapes.")
    if not (
        torch.isfinite(student).all()
        and torch.isfinite(teacher).all()
        and torch.isfinite(target).all()
    ):
        raise FloatingPointError("Student-safe projection received NaN or Inf.")

    lower = torch.minimum(student, target)
    upper = torch.maximum(student, target)
    projected = torch.maximum(torch.minimum(teacher, upper), lower)

    student_to_label = target - student
    student_to_teacher = teacher - student
    directional_product = student_to_label * student_to_teacher
    wrong_direction = directional_product < 0.0
    overshoot = (
        (directional_product > 0.0)
        & (student_to_teacher.abs() > student_to_label.abs())
    )
    zero_width = student_to_label.abs() <= PROJECTION_EPS
    unchanged = torch.isclose(projected, teacher, atol=PROJECTION_EPS, rtol=0.0)
    projected_to_student = torch.isclose(
        projected, student, atol=PROJECTION_EPS, rtol=0.0
    )
    projected_to_label = torch.isclose(
        projected, target, atol=PROJECTION_EPS, rtol=0.0
    )
    active = ~projected_to_student.view(-1)
    raw_teacher_better = (
        (teacher - target).abs() < (student - target).abs()
    ).view(-1)

    diagnostics = {
        "wrong_direction": wrong_direction.view(-1),
        "overshoot": overshoot.view(-1),
        "zero_width": zero_width.view(-1),
        "unchanged": unchanged.view(-1),
        # Alias retained so the frozen v1 projection accounting can be reused.
        "projected_to_baseline": projected_to_student.view(-1),
        "projected_to_student": projected_to_student.view(-1),
        "projected_to_label": projected_to_label.view(-1),
        "teacher_target_abs_shift": (projected - teacher).abs().view(-1),
        "safe_interval_width": (upper - lower).view(-1),
        "active": active,
        "abstained": ~active,
        "raw_teacher_better": raw_teacher_better,
    }
    return projected.detach(), diagnostics


def student_projection_summary(records: Sequence[Mapping]) -> dict:
    """Extend the frozen projection accounting with explicit abstention rates."""
    summary = projection_summary(records)
    frame = pd.DataFrame(records)
    required = {"active", "abstained", "raw_teacher_better"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Student-safe projection records lack columns: {}".format(
                sorted(missing)
            )
        )
    summary.update(
        {
            "active_fraction": float(frame.active.mean()),
            "abstain_fraction": float(frame.abstained.mean()),
            "raw_teacher_better_fraction": float(frame.raw_teacher_better.mean()),
        }
    )
    if abs(summary["active_fraction"] + summary["abstain_fraction"] - 1.0) > 1e-12:
        raise RuntimeError("Active and abstain fractions do not sum to one.")
    return summary


def _row(rows, seed, run):
    selected = [
        row
        for row in rows
        if int(row["Seed"]) == int(seed) and str(row["Run"]) == str(run)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "Grid row is not unique: seed={} run={}.".format(seed, run)
        )
    return selected[0]


def _group_value(groups, seed, run, group_type, group_value, metric):
    selected = groups.loc[
        groups.Seed.astype(str).eq(str(seed))
        & groups.Run.astype(str).eq(str(run))
        & groups.GroupType.astype(str).eq(str(group_type))
        & groups.GroupValue.astype(str).eq(str(group_value))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "Group metric is not unique: seed={} run={} {}={}.".format(
                seed, run, group_type, group_value
            )
        )
    return float(selected.iloc[0][metric])


def _q4_pass(candidate_gain, baseline_gain):
    if baseline_gain > 0.0:
        required = Q4_GAIN_RETENTION_FRACTION * baseline_gain
        return candidate_gain >= required, required
    required = baseline_gain - Q1_MAX_DEGRADATION
    return candidate_gain >= required, required


def candidate_gate(candidate_run, grid_rows, epoch_rows, groups):
    """Apply the frozen v2 non-inferiority and safety decision rule."""
    if candidate_run not in CANDIDATE_RUNS:
        raise ValueError("Unknown v2 candidate run: {}".format(candidate_run))

    per_seed = {}
    for seed in FORMAL_SEEDS:
        baseline = _row(grid_rows, seed, "cfcompat_replay")
        candidate = _row(grid_rows, seed, candidate_run)
        j_degradation = float(candidate["J_valid"]) - float(baseline["J_valid"])
        mode_degradations = {
            mode: float(candidate[f"valid_{mode}_MAE"])
            - float(baseline[f"valid_{mode}_MAE"])
            for mode in ("LAV",) + MISSING_MODES
        }

        baseline_harm = _group_value(
            groups, seed, "cfcompat_replay", "all", "ALL",
            "harmful_imitation_rate"
        )
        candidate_harm = _group_value(
            groups, seed, candidate_run, "all", "ALL",
            "harmful_imitation_rate"
        )
        q1_baseline = _group_value(
            groups, seed, "cfcompat_replay", "baseline_error_quartile",
            "Q1_easy", "gain_vs_DLF"
        )
        q1_candidate = _group_value(
            groups, seed, candidate_run, "baseline_error_quartile",
            "Q1_easy", "gain_vs_DLF"
        )
        q4_baseline = _group_value(
            groups, seed, "cfcompat_replay", "baseline_error_quartile",
            "Q4_hard", "gain_vs_DLF"
        )
        q4_candidate = _group_value(
            groups, seed, candidate_run, "baseline_error_quartile",
            "Q4_hard", "gain_vs_DLF"
        )
        better_baseline = _group_value(
            groups, seed, "cfcompat_replay", "teacher_condition",
            "better_and_correct", "gain_vs_DLF"
        )
        better_candidate = _group_value(
            groups, seed, candidate_run, "teacher_condition",
            "better_and_correct", "gain_vs_DLF"
        )
        q4_pass, q4_required = _q4_pass(q4_candidate, q4_baseline)
        supporting = sum(
            int(row["Seed"]) == int(seed)
            and str(row["Run"]) == candidate_run
            and float(row["J_valid"])
            <= float(baseline["J_valid"]) + PER_SEED_J_MAX_DEGRADATION
            for row in epoch_rows
        )

        per_seed[str(seed)] = {
            "J_degradation_vs_CFCompatKD": j_degradation,
            "J_noninferior": j_degradation <= PER_SEED_J_MAX_DEGRADATION,
            "mode_degradations_vs_CFCompatKD": mode_degradations,
            "no_mode_degradation_over_limit": all(
                value <= PER_MODE_MAX_DEGRADATION
                for value in mode_degradations.values()
            ),
            "CFCompat_harmful_imitation_rate": baseline_harm,
            "candidate_harmful_imitation_rate": candidate_harm,
            "harmful_reduction": baseline_harm - candidate_harm,
            "harmful_not_increased": candidate_harm <= baseline_harm,
            "Q1_CFCompat_gain_vs_DLF": q1_baseline,
            "Q1_candidate_gain_vs_DLF": q1_candidate,
            "Q1_required_gain": q1_baseline - Q1_MAX_DEGRADATION,
            "Q1_not_materially_degraded": (
                q1_candidate >= q1_baseline - Q1_MAX_DEGRADATION
            ),
            "Q4_CFCompat_gain_vs_DLF": q4_baseline,
            "Q4_candidate_gain_vs_DLF": q4_candidate,
            "Q4_required_gain": q4_required,
            "Q4_retains_required_gain": bool(q4_pass),
            "CFCompat_better_correct_gain_vs_DLF": better_baseline,
            "candidate_better_correct_gain_vs_DLF": better_candidate,
            "better_correct_not_materially_degraded": (
                better_candidate
                >= better_baseline - BETTER_CORRECT_MAX_DEGRADATION
            ),
            "supporting_epoch_count": int(supporting),
            "at_least_two_noninferior_epochs": supporting >= SUPPORTING_EPOCHS,
            "projection_active_fraction": float(
                candidate.get("projection_active_fraction", float("nan"))
            ),
            "projection_abstain_fraction": float(
                candidate.get("projection_abstain_fraction", float("nan"))
            ),
        }

    mean_j_degradation = float(
        np.mean(
            [item["J_degradation_vs_CFCompatKD"] for item in per_seed.values()]
        )
    )
    mean_harmful_reduction = float(
        np.mean([item["harmful_reduction"] for item in per_seed.values()])
    )
    checks = {
        "mean_J_degradation_le_0p003": (
            mean_j_degradation <= MEAN_J_MAX_DEGRADATION
        ),
        "each_seed_J_degradation_le_0p005": all(
            item["J_noninferior"] for item in per_seed.values()
        ),
        "no_seed_or_mode_degradation_over_0p005": all(
            item["no_mode_degradation_over_limit"]
            for item in per_seed.values()
        ),
        "harmful_imitation_not_increased_each_seed": all(
            item["harmful_not_increased"] for item in per_seed.values()
        ),
        "mean_harmful_imitation_reduction_ge_0p02": (
            mean_harmful_reduction >= MEAN_HARMFUL_REDUCTION_REQUIRED
        ),
        "Q1_not_materially_degraded_each_seed": all(
            item["Q1_not_materially_degraded"] for item in per_seed.values()
        ),
        "Q4_retains_80pct_gain_each_seed": all(
            item["Q4_retains_required_gain"] for item in per_seed.values()
        ),
        "better_correct_not_materially_degraded_each_seed": all(
            item["better_correct_not_materially_degraded"]
            for item in per_seed.values()
        ),
        "two_noninferior_epochs_each_seed": all(
            item["at_least_two_noninferior_epochs"]
            for item in per_seed.values()
        ),
    }
    return {
        "candidate_run": candidate_run,
        "passed": bool(all(checks.values())),
        "mean_J_degradation_vs_CFCompatKD": mean_j_degradation,
        "mean_J_gain_vs_CFCompatKD": -mean_j_degradation,
        "mean_harmful_imitation_reduction": mean_harmful_reduction,
        "checks": checks,
        "per_seed": per_seed,
        "thresholds": {
            "mean_J_max_degradation": MEAN_J_MAX_DEGRADATION,
            "per_seed_J_max_degradation": PER_SEED_J_MAX_DEGRADATION,
            "per_mode_max_degradation": PER_MODE_MAX_DEGRADATION,
            "mean_harmful_reduction_required": MEAN_HARMFUL_REDUCTION_REQUIRED,
            "Q1_max_degradation": Q1_MAX_DEGRADATION,
            "Q4_gain_retention_fraction": Q4_GAIN_RETENTION_FRACTION,
            "better_correct_max_degradation": BETTER_CORRECT_MAX_DEGRADATION,
            "supporting_epochs": SUPPORTING_EPOCHS,
        },
        "official_test_authorized": False,
    }
