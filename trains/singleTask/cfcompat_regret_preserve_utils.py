"""Frozen utilities for Regret-Aware Preserve-or-Distill CFCompatKD v4.

The v4 development screen addresses a specific failure mode observed after
CFCompatKD: some missing-modality predictions improve after distillation while
others are pushed away from an already-strong frozen ModDrop baseline.

For each Train sample/missing-mode event, v4 makes an exclusive three-way
choice using only frozen models, the current detached Student prediction, and
Train labels:

* DISTILL: the frozen full-modality Teacher beats the frozen ModDrop baseline
  by a fixed margin and its current-Student-safe target is active;
* PRESERVE: DISTILL is unavailable and the current Student has regressed from
  the frozen ModDrop baseline by a fixed margin;
* ABSTAIN: neither condition holds.

CFCompat is retained only as a mild prior on DISTILL events via
``0.75 + 0.25 * compatibility``.  PRESERVE events use a safe projection of the
frozen ModDrop prediction onto the current-Student-to-label interval.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cfcompat_student_safe_abstain_utils import (
    PROJECTION_EPS,
    jsonable,
    student_projection_summary,
    student_safe_project_teacher,
)
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_regret_preserve_distill_valid_screen_v4"
METHOD = "DLF-Regret-Aware-Preserve-or-Distill-CFCompatKD-v4"
OUTPUT_TAG = "cfcompat_regret_preserve_v4"
DEV_SEED = 1113
RUN = "regret_preserve_cfcompat"
RUNS = (RUN,)

# Frozen before the single-seed development run.
DISTILL_MARGIN = 0.02
PRESERVE_MARGIN = 0.02
LAMBDA_PRESERVE = 0.25
MILD_CFCOMPAT_BASE = 0.75
MILD_CFCOMPAT_SCALE = 0.25
NEGATIVE_TRANSFER_MARGIN = 0.02
SEVERE_NEGATIVE_TRANSFER_MARGIN = 0.10

# Single-seed promotion rule.  The primary objective is to lower sample-level
# negative transfer without materially sacrificing the already-strong v2
# student_safe_uniform Valid result.
J_MAX_DEGRADATION_VS_UNIFORM = 0.002
J_MAX_DEGRADATION_VS_REPLAY = 0.005
NEGATIVE_TRANSFER_REDUCTION_REQUIRED = 0.02
GROUP_MAX_DEGRADATION_VS_UNIFORM = 0.002
SUPPORTING_EPOCHS = 2


def mild_cfcompat(compatibility: torch.Tensor) -> torch.Tensor:
    """Return a bounded, detached CFCompat prior in (0.75, 1.0)."""
    compat = compatibility.detach().view(-1)
    if (
        not torch.isfinite(compat).all()
        or torch.any(compat <= 0.0)
        or torch.any(compat >= 1.0)
    ):
        raise FloatingPointError("Compatibility must remain strictly in (0,1).")
    result = MILD_CFCOMPAT_BASE + MILD_CFCOMPAT_SCALE * compat
    if (
        not torch.isfinite(result).all()
        or torch.any(result <= MILD_CFCOMPAT_BASE)
        or torch.any(result >= 1.0)
    ):
        raise AssertionError("Mild CFCompat prior escaped its frozen range.")
    return result.detach()


def regret_preserve_decision(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    labels: torch.Tensor,
    compatibility: torch.Tensor,
) -> dict:
    """Compute the exclusive DISTILL / PRESERVE / ABSTAIN v4 decision.

    All branch decisions and targets are detached.  Gradients flow only through
    the caller's Student output when the returned targets are used in losses.
    """
    student = student_prediction.detach().view(-1, 1)
    teacher = teacher_prediction.detach().view(-1, 1)
    baseline = baseline_prediction.detach().view(-1, 1)
    target = labels.detach().view(-1, 1)
    compat = compatibility.detach().view(-1)
    if not (student.shape == teacher.shape == baseline.shape == target.shape):
        raise ValueError("Regret-aware decision tensors must have identical shapes.")
    if len(compat) != len(student):
        raise ValueError("Compatibility length does not match predictions.")
    if not all(
        torch.isfinite(value).all()
        for value in (student, teacher, baseline, target, compat)
    ):
        raise FloatingPointError("Regret-aware decision received NaN/Inf.")

    teacher_safe, teacher_projection = student_safe_project_teacher(
        student, teacher, target
    )
    preserve_safe, _ = student_safe_project_teacher(student, baseline, target)

    baseline_error = torch.abs(baseline - target).view(-1)
    teacher_error = torch.abs(teacher - target).view(-1)
    current_error = torch.abs(student - target).view(-1)
    teacher_advantage = baseline_error - teacher_error
    current_regret = current_error - baseline_error

    teacher_beneficial = teacher_advantage >= DISTILL_MARGIN
    current_regressed = current_regret >= PRESERVE_MARGIN
    safe_teacher_active = teacher_projection["active"].view(-1)

    distill = teacher_beneficial & safe_teacher_active
    preserve = (~distill) & current_regressed
    abstain = ~(distill | preserve)
    if torch.any(distill & preserve) or torch.any(distill & abstain) or torch.any(preserve & abstain):
        raise AssertionError("Three-way decisions are not mutually exclusive.")
    if not torch.all(distill | preserve | abstain):
        raise AssertionError("Three-way decisions are not exhaustive.")

    mild = mild_cfcompat(compat)
    distill_gate = distill.to(dtype=compat.dtype, device=compat.device) * mild
    preserve_gate = preserve.to(dtype=compat.dtype, device=compat.device)

    # A PRESERVE target is also current-Student safe.  If the frozen baseline
    # lies across the label, it is clipped to the label rather than allowing an
    # overshooting preservation force.
    preserve_target = preserve_safe.detach().view(-1, 1)

    return {
        "teacher_safe_target": teacher_safe.detach().view(-1, 1),
        "preserve_safe_target": preserve_target,
        "distill_gate": distill_gate.detach(),
        "preserve_gate": preserve_gate.detach(),
        "distill": distill.detach(),
        "preserve": preserve.detach(),
        "abstain": abstain.detach(),
        "teacher_beneficial": teacher_beneficial.detach(),
        "current_regressed": current_regressed.detach(),
        "baseline_error": baseline_error.detach(),
        "teacher_error": teacher_error.detach(),
        "current_error": current_error.detach(),
        "teacher_advantage_vs_baseline": teacher_advantage.detach(),
        "current_regret_vs_baseline": current_regret.detach(),
        "mild_compatibility": mild.detach(),
        "teacher_projection": teacher_projection,
    }


def regret_projection_summary(records: Sequence[Mapping]) -> dict:
    """Extend current-Student projection accounting with v4 branch statistics."""
    if not records:
        raise ValueError("Regret-aware projection summary requires records.")
    summary = student_projection_summary(records)
    frame = pd.DataFrame(records)
    required = {
        "mode",
        "distill",
        "preserve",
        "decision_abstain",
        "teacher_beneficial",
        "current_regressed",
        "baseline_error",
        "teacher_error",
        "current_error",
        "teacher_advantage_vs_baseline",
        "current_regret_vs_baseline",
        "compatibility",
        "mild_compatibility",
        "distill_gate",
        "preserve_gate",
        "teacher_safe_target",
        "preserve_safe_target",
        "student_prediction",
        "label",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Regret-aware records lack columns: {}".format(sorted(missing)))

    numeric_columns = [
        "baseline_error",
        "teacher_error",
        "current_error",
        "teacher_advantage_vs_baseline",
        "current_regret_vs_baseline",
        "compatibility",
        "mild_compatibility",
        "distill_gate",
        "preserve_gate",
        "teacher_safe_target",
        "preserve_safe_target",
        "student_prediction",
        "label",
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Regret-aware records contain non-finite values.")

    distill = frame.distill.astype(bool).to_numpy()
    preserve = frame.preserve.astype(bool).to_numpy()
    abstain = frame.decision_abstain.astype(bool).to_numpy()
    if np.any((distill.astype(int) + preserve.astype(int) + abstain.astype(int)) != 1):
        raise RuntimeError("Three-way decision accounting is invalid.")

    summary.update(
        {
            "distill_fraction": float(distill.mean()),
            "preserve_fraction": float(preserve.mean()),
            "decision_abstain_fraction": float(abstain.mean()),
            "teacher_beneficial_fraction": float(frame.teacher_beneficial.astype(bool).mean()),
            "current_regressed_fraction": float(frame.current_regressed.astype(bool).mean()),
            "mean_baseline_error": float(frame.baseline_error.mean()),
            "mean_teacher_error": float(frame.teacher_error.mean()),
            "mean_current_error": float(frame.current_error.mean()),
            "mean_teacher_advantage_vs_baseline": float(frame.teacher_advantage_vs_baseline.mean()),
            "mean_current_regret_vs_baseline": float(frame.current_regret_vs_baseline.mean()),
            "mean_compatibility": float(frame.compatibility.mean()),
            "mean_mild_compatibility": float(frame.mild_compatibility.mean()),
            "mean_distill_gate": float(frame.distill_gate.mean()),
            "mean_preserve_gate": float(frame.preserve_gate.mean()),
        }
    )
    if abs(
        summary["distill_fraction"]
        + summary["preserve_fraction"]
        + summary["decision_abstain_fraction"]
        - 1.0
    ) > 1e-12:
        raise RuntimeError("DISTILL/PRESERVE/ABSTAIN fractions do not sum to one.")

    for mode in MISSING_MODES:
        local = frame.loc[frame["mode"].astype(str).eq(mode)]
        if local.empty:
            raise RuntimeError("Regret records contain no {} events.".format(mode))
        summary.update(
            {
                "{}_distill_fraction".format(mode): float(local.distill.astype(bool).mean()),
                "{}_preserve_fraction".format(mode): float(local.preserve.astype(bool).mean()),
                "{}_abstain_fraction".format(mode): float(local.decision_abstain.astype(bool).mean()),
            }
        )
    return summary


def negative_transfer_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Measure sample-level Valid regression relative to the frozen baseline.

    The principal row used by the v4 decision is ``Mode=MISSING_ALL`` so the
    metric directly reflects LA/LV/L, where the selective KD mechanism acts.
    """
    required = {
        "Seed",
        "Run",
        "Mode",
        "sample_index",
        "baseline_error",
        "candidate_error",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Valid events lack negative-transfer columns: {}".format(sorted(missing)))

    rows = []
    for (seed, run), local_run in events.groupby(["Seed", "Run"], sort=True):
        selections = [(mode, local_run.loc[local_run.Mode.astype(str).eq(mode)]) for mode in ("LAV",) + MISSING_MODES]
        selections.extend(
            [
                ("MISSING_ALL", local_run.loc[local_run.Mode.astype(str).isin(MISSING_MODES)]),
                ("ALL", local_run),
            ]
        )
        for mode, local in selections:
            if local.empty:
                raise RuntimeError("Negative-transfer group {} is empty.".format(mode))
            regret = local.candidate_error.to_numpy(dtype=np.float64) - local.baseline_error.to_numpy(dtype=np.float64)
            rows.append(
                {
                    "Seed": int(seed),
                    "Run": str(run),
                    "Mode": str(mode),
                    "N": int(len(local)),
                    "negative_transfer_margin": NEGATIVE_TRANSFER_MARGIN,
                    "negative_transfer_rate": float((regret > NEGATIVE_TRANSFER_MARGIN).mean()),
                    "severe_negative_transfer_margin": SEVERE_NEGATIVE_TRANSFER_MARGIN,
                    "severe_negative_transfer_rate": float((regret > SEVERE_NEGATIVE_TRANSFER_MARGIN).mean()),
                    "positive_transfer_rate": float((regret < -NEGATIVE_TRANSFER_MARGIN).mean()),
                    "mean_regret_vs_baseline": float(regret.mean()),
                    "median_regret_vs_baseline": float(np.median(regret)),
                    "mean_gain_vs_baseline": float((-regret).mean()),
                }
            )
    return pd.DataFrame(rows)


def _grid_row(frame: pd.DataFrame, run: str) -> Mapping:
    selected = frame.loc[
        frame.Seed.astype(int).eq(DEV_SEED) & frame.Run.astype(str).eq(str(run))
    ]
    if len(selected) != 1:
        raise RuntimeError("Reference grid has no unique seed={} run={}.".format(DEV_SEED, run))
    return selected.iloc[0].to_dict()


def _group_value(groups: pd.DataFrame, run: str, group_type: str, group_value: str, metric: str) -> float:
    selected = groups.loc[
        groups.Seed.astype(str).eq(str(DEV_SEED))
        & groups.Run.astype(str).eq(str(run))
        & groups.GroupType.astype(str).eq(str(group_type))
        & groups.GroupValue.astype(str).eq(str(group_value))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "Group metric is not unique for {} {}={}.".format(run, group_type, group_value)
        )
    return float(selected.iloc[0][metric])


def _negative_rate(frame: pd.DataFrame, run: str) -> float:
    selected = frame.loc[
        frame.Seed.astype(int).eq(DEV_SEED)
        & frame.Run.astype(str).eq(str(run))
        & frame.Mode.astype(str).eq("MISSING_ALL")
    ]
    if len(selected) != 1:
        raise RuntimeError("Missing unique MISSING_ALL negative-transfer row for {}.".format(run))
    return float(selected.iloc[0].negative_transfer_rate)


def dev_candidate_gate(
    candidate_row: Mapping,
    reference_grid: pd.DataFrame,
    groups: pd.DataFrame,
    negative_transfer: pd.DataFrame,
    epoch_rows: Sequence[Mapping],
) -> dict:
    """Frozen Seed-1113 promotion gate for the v4 development candidate."""
    replay = _grid_row(reference_grid, "cfcompat_replay")
    uniform = _grid_row(reference_grid, "student_safe_uniform")
    candidate_j = float(candidate_row["J_valid"])
    uniform_j = float(uniform["J_valid"])
    replay_j = float(replay["J_valid"])

    uniform_harm = _group_value(groups, "student_safe_uniform", "all", "ALL", "harmful_imitation_rate")
    candidate_harm = _group_value(groups, RUN, "all", "ALL", "harmful_imitation_rate")

    group_checks = {}
    group_values = {}
    for group_type, group_value, name in (
        ("baseline_error_quartile", "Q1_easy", "Q1"),
        ("baseline_error_quartile", "Q4_hard", "Q4"),
        ("teacher_condition", "better_and_correct", "better_correct"),
    ):
        uniform_gain = _group_value(groups, "student_safe_uniform", group_type, group_value, "gain_vs_DLF")
        candidate_gain = _group_value(groups, RUN, group_type, group_value, "gain_vs_DLF")
        threshold = uniform_gain - GROUP_MAX_DEGRADATION_VS_UNIFORM
        group_values[name] = {
            "uniform_gain": uniform_gain,
            "candidate_gain": candidate_gain,
            "minimum_allowed": threshold,
        }
        group_checks["{}_not_materially_degraded_vs_uniform".format(name)] = bool(candidate_gain >= threshold)

    uniform_negative = _negative_rate(negative_transfer, "student_safe_uniform")
    candidate_negative = _negative_rate(negative_transfer, RUN)
    negative_reduction = uniform_negative - candidate_negative

    supporting = int(
        sum(
            int(row["Seed"]) == DEV_SEED
            and str(row["Run"]) == RUN
            and float(row["J_valid"]) <= uniform_j + J_MAX_DEGRADATION_VS_UNIFORM
            for row in epoch_rows
        )
    )

    fractions = {
        "distill": float(candidate_row["projection_distill_fraction"]),
        "preserve": float(candidate_row["projection_preserve_fraction"]),
        "abstain": float(candidate_row["projection_decision_abstain_fraction"]),
    }
    checks = {
        "J_not_materially_worse_than_uniform": bool(candidate_j - uniform_j <= J_MAX_DEGRADATION_VS_UNIFORM),
        "J_not_materially_worse_than_replay": bool(candidate_j - replay_j <= J_MAX_DEGRADATION_VS_REPLAY),
        "negative_transfer_reduction_ge_0p02": bool(negative_reduction >= NEGATIVE_TRANSFER_REDUCTION_REQUIRED),
        "harmful_imitation_not_increased_vs_uniform": bool(candidate_harm <= uniform_harm + 1e-12),
        "two_noninferior_epochs_vs_uniform": bool(supporting >= SUPPORTING_EPOCHS),
        "three_way_policy_exercised": bool(all(value > 0.0 for value in fractions.values())),
        **group_checks,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_J": candidate_j,
        "uniform_J": uniform_j,
        "replay_J": replay_j,
        "J_degradation_vs_uniform": candidate_j - uniform_j,
        "J_degradation_vs_replay": candidate_j - replay_j,
        "uniform_harmful_imitation_rate": uniform_harm,
        "candidate_harmful_imitation_rate": candidate_harm,
        "uniform_negative_transfer_rate": uniform_negative,
        "candidate_negative_transfer_rate": candidate_negative,
        "negative_transfer_reduction": negative_reduction,
        "supporting_epoch_count": supporting,
        "decision_fractions": fractions,
        "group_values": group_values,
    }


def frozen_thresholds() -> dict:
    return {
        "distill_margin": DISTILL_MARGIN,
        "preserve_margin": PRESERVE_MARGIN,
        "lambda_preserve": LAMBDA_PRESERVE,
        "mild_cfcompat_base": MILD_CFCOMPAT_BASE,
        "mild_cfcompat_scale": MILD_CFCOMPAT_SCALE,
        "negative_transfer_margin": NEGATIVE_TRANSFER_MARGIN,
        "J_max_degradation_vs_uniform": J_MAX_DEGRADATION_VS_UNIFORM,
        "J_max_degradation_vs_replay": J_MAX_DEGRADATION_VS_REPLAY,
        "negative_transfer_reduction_required": NEGATIVE_TRANSFER_REDUCTION_REQUIRED,
        "group_max_degradation_vs_uniform": GROUP_MAX_DEGRADATION_VS_UNIFORM,
        "supporting_epochs": SUPPORTING_EPOCHS,
    }


__all__ = [
    "DEV_SEED",
    "DISTILL_MARGIN",
    "GROUP_MAX_DEGRADATION_VS_UNIFORM",
    "J_MAX_DEGRADATION_VS_REPLAY",
    "J_MAX_DEGRADATION_VS_UNIFORM",
    "LAMBDA_PRESERVE",
    "METHOD",
    "MILD_CFCOMPAT_BASE",
    "MILD_CFCOMPAT_SCALE",
    "NEGATIVE_TRANSFER_MARGIN",
    "NEGATIVE_TRANSFER_REDUCTION_REQUIRED",
    "OUTPUT_TAG",
    "PRESERVE_MARGIN",
    "RUN",
    "RUNS",
    "SUPPORTING_EPOCHS",
    "VERSION",
    "dev_candidate_gate",
    "frozen_thresholds",
    "jsonable",
    "mild_cfcompat",
    "negative_transfer_summary",
    "regret_preserve_decision",
    "regret_projection_summary",
]
