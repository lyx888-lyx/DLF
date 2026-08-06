"""Frozen utilities for student-safe dynamic-utility CFCompatKD v3.

The v3 candidates retain current-Student interval projection and unsafe-KD
abstention.  Active samples are optionally weighted by two dynamic quantities:

* utility: the fraction of the current absolute label error removed by the
  projected Teacher target;
* difficulty: the current absolute label error normalized by a Train-only,
  per-missing-mode scale frozen from the initial Student.

The residual-CFCompat candidate then applies ``0.5 + 0.5 * compatibility`` so
that the original static compatibility prior can modulate, but never suppress,
a dynamically useful safe target below half of its dynamic weight.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cfcompat_student_safe_abstain_utils import (
    BETTER_CORRECT_MAX_DEGRADATION,
    FORMAL_SEEDS,
    MEAN_HARMFUL_REDUCTION_REQUIRED,
    MEAN_J_MAX_DEGRADATION,
    PER_MODE_MAX_DEGRADATION,
    PER_SEED_J_MAX_DEGRADATION,
    Q1_MAX_DEGRADATION,
    Q4_GAIN_RETENTION_FRACTION,
    SUPPORTING_EPOCHS,
    candidate_gate as v2_candidate_gate,
    jsonable,
    student_projection_summary,
    student_safe_project_teacher,
)
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_student_safe_utility_residual_valid_screen_v3"
METHOD = "DLF-Student-Safe-Utility-Residual-CFCompat-v3"
OUTPUT_TAG = "cfcompat_student_safe_utility_v3"
RUNS = (
    "cfcompat_replay",
    "student_safe_uniform",
    "student_safe_utility",
    "student_safe_utility_residual_cfcompat",
)
CANDIDATE_RUNS = RUNS[1:]
PRIMARY_RUN = "student_safe_utility_residual_cfcompat"
UTILITY_RUN = "student_safe_utility"
UNIFORM_RUN = "student_safe_uniform"
RESIDUAL_CFCOMPAT_ALPHA = 0.5
UTILITY_EPS = 1e-8


def dynamic_utility(
    student_prediction: torch.Tensor,
    safe_target: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return the fraction of current absolute error removed by ``safe_target``."""
    student = student_prediction.detach().view(-1)
    safe = safe_target.detach().view(-1)
    target = labels.detach().view(-1)
    if not (student.shape == safe.shape == target.shape):
        raise ValueError("Dynamic utility tensors must have identical shapes.")
    current_error = torch.abs(student - target)
    safe_error = torch.abs(safe - target)
    utility = (current_error - safe_error) / (current_error + UTILITY_EPS)
    utility = torch.clamp(utility, min=0.0, max=1.0)
    if not torch.isfinite(utility).all():
        raise FloatingPointError("Dynamic utility is non-finite.")
    return utility.detach()


def dynamic_difficulty(
    student_prediction: torch.Tensor,
    labels: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Normalize current absolute error by a positive frozen Train-only scale."""
    student = student_prediction.detach().view(-1)
    target = labels.detach().view(-1)
    local_tau = tau.detach().view(-1).to(student)
    if not (student.shape == target.shape == local_tau.shape):
        raise ValueError("Dynamic difficulty tensors must have identical shapes.")
    if not torch.isfinite(local_tau).all() or torch.any(local_tau <= 0.0):
        raise FloatingPointError("Difficulty scales must be positive and finite.")
    error = torch.abs(student - target)
    difficulty = error / (error + local_tau)
    if not torch.isfinite(difficulty).all():
        raise FloatingPointError("Dynamic difficulty is non-finite.")
    if torch.any(difficulty < 0.0) or torch.any(difficulty >= 1.0):
        raise AssertionError("Dynamic difficulty must remain in [0,1).")
    return difficulty.detach()


def residual_compatibility(
    compatibility: torch.Tensor,
    alpha: float = RESIDUAL_CFCOMPAT_ALPHA,
) -> torch.Tensor:
    """Apply a fixed lower-bound residual transform to compatibility."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("Residual CFCompat alpha must lie strictly in (0,1).")
    compatibility = compatibility.detach().view(-1)
    if (
        not torch.isfinite(compatibility).all()
        or torch.any(compatibility <= 0.0)
        or torch.any(compatibility >= 1.0)
    ):
        raise FloatingPointError("Compatibility must remain in (0,1).")
    residual = float(alpha) + (1.0 - float(alpha)) * compatibility
    if (
        not torch.isfinite(residual).all()
        or torch.any(residual <= float(alpha))
        or torch.any(residual >= 1.0)
    ):
        raise AssertionError("Residual compatibility must remain in (alpha,1).")
    return residual.detach()


def gate_effective_sample_size(values) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Gate ESS requires finite non-negative values.")
    denominator = float(np.square(values).sum())
    if denominator <= 0.0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def _correlation(first, second, ranks=False) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(first) != len(second) or len(first) < 2:
        return 0.0
    if ranks:
        first = pd.Series(first).rank(method="average").to_numpy(dtype=np.float64)
        second = pd.Series(second).rank(method="average").to_numpy(dtype=np.float64)
    if float(first.std()) == 0.0 or float(second.std()) == 0.0:
        return 0.0
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else 0.0


def utility_projection_summary(records: Sequence[Mapping]) -> dict:
    """Extend v2 projection accounting with dynamic-weight diagnostics."""
    summary = student_projection_summary(records)
    frame = pd.DataFrame(records)
    required = {
        "mode",
        "utility",
        "difficulty",
        "compatibility",
        "residual_compatibility",
        "final_gate",
        "active",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Student-safe utility records lack columns: {}".format(sorted(missing))
        )

    numeric = frame[
        [
            "utility",
            "difficulty",
            "compatibility",
            "residual_compatibility",
            "final_gate",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Student-safe utility records contain NaN/Inf.")
    if np.any(numeric[:, 0] < 0.0) or np.any(numeric[:, 0] > 1.0):
        raise AssertionError("Utility must remain in [0,1].")
    if np.any(numeric[:, 1] < 0.0) or np.any(numeric[:, 1] >= 1.0):
        raise AssertionError("Difficulty must remain in [0,1).")
    if np.any(numeric[:, 2] <= 0.0) or np.any(numeric[:, 2] >= 1.0):
        raise AssertionError("Compatibility must remain in (0,1).")
    if (
        np.any(numeric[:, 3] <= RESIDUAL_CFCOMPAT_ALPHA)
        or np.any(numeric[:, 3] >= 1.0)
    ):
        raise AssertionError("Residual compatibility is outside its frozen range.")
    if np.any(numeric[:, 4] < 0.0) or np.any(numeric[:, 4] > 1.0):
        raise AssertionError("Final gate must remain in [0,1].")

    active = frame.active.astype(bool).to_numpy()
    active_frame = frame.loc[active]
    summary.update(
        {
            "mean_utility": float(frame.utility.mean()),
            "mean_active_utility": (
                float(active_frame.utility.mean()) if len(active_frame) else 0.0
            ),
            "mean_difficulty": float(frame.difficulty.mean()),
            "mean_compatibility": float(frame.compatibility.mean()),
            "mean_residual_compatibility": float(
                frame.residual_compatibility.mean()
            ),
            "mean_final_gate": float(frame.final_gate.mean()),
            "gate_effective_sample_size": gate_effective_sample_size(
                frame.final_gate.to_numpy(dtype=np.float64)
            ),
            "active_compatibility_utility_pearson": _correlation(
                active_frame.compatibility, active_frame.utility
            ) if len(active_frame) else 0.0,
            "active_compatibility_utility_spearman": _correlation(
                active_frame.compatibility, active_frame.utility, ranks=True
            ) if len(active_frame) else 0.0,
        }
    )
    for mode in MISSING_MODES:
        local = frame.loc[frame["mode"].astype(str).eq(mode)]
        if local.empty:
            raise RuntimeError("Projection records contain no {} events.".format(mode))
        summary.update(
            {
                "{}_active_fraction".format(mode): float(
                    local.active.astype(bool).mean()
                ),
                "{}_mean_utility".format(mode): float(local.utility.mean()),
                "{}_mean_difficulty".format(mode): float(local.difficulty.mean()),
                "{}_mean_final_gate".format(mode): float(local.final_gate.mean()),
                "{}_gate_effective_sample_size".format(mode): (
                    gate_effective_sample_size(
                        local.final_gate.to_numpy(dtype=np.float64)
                    )
                ),
            }
        )
    return summary


def candidate_gate(candidate_run, grid_rows, epoch_rows, groups):
    """Reuse the frozen v2 safety/non-inferiority gate for any v3 candidate."""
    if candidate_run not in CANDIDATE_RUNS:
        raise ValueError("Unknown v3 candidate run: {}".format(candidate_run))

    alias = "student_safe_uniform"
    local_grid = []
    for row in grid_rows:
        run = str(row["Run"])
        if run not in ("cfcompat_replay", candidate_run):
            continue
        copied = dict(row)
        if run == candidate_run:
            copied["Run"] = alias
        local_grid.append(copied)

    local_epochs = []
    for row in epoch_rows:
        run = str(row["Run"])
        if run not in ("cfcompat_replay", candidate_run):
            continue
        copied = dict(row)
        if run == candidate_run:
            copied["Run"] = alias
        local_epochs.append(copied)

    local_groups = groups.loc[
        groups.Run.astype(str).isin(["cfcompat_replay", candidate_run])
    ].copy()
    local_groups.loc[
        local_groups.Run.astype(str).eq(candidate_run), "Run"
    ] = alias

    result = v2_candidate_gate(alias, local_grid, local_epochs, local_groups)
    result["candidate_run"] = candidate_run
    return result


def frozen_thresholds() -> dict:
    return {
        "mean_J_max_degradation": MEAN_J_MAX_DEGRADATION,
        "per_seed_J_max_degradation": PER_SEED_J_MAX_DEGRADATION,
        "per_mode_max_degradation": PER_MODE_MAX_DEGRADATION,
        "mean_harmful_reduction_required": MEAN_HARMFUL_REDUCTION_REQUIRED,
        "Q1_max_degradation": Q1_MAX_DEGRADATION,
        "Q4_gain_retention_fraction": Q4_GAIN_RETENTION_FRACTION,
        "better_correct_max_degradation": BETTER_CORRECT_MAX_DEGRADATION,
        "supporting_epochs": SUPPORTING_EPOCHS,
    }


__all__ = [
    "BETTER_CORRECT_MAX_DEGRADATION",
    "CANDIDATE_RUNS",
    "FORMAL_SEEDS",
    "METHOD",
    "OUTPUT_TAG",
    "PRIMARY_RUN",
    "RESIDUAL_CFCOMPAT_ALPHA",
    "RUNS",
    "UNIFORM_RUN",
    "UTILITY_RUN",
    "VERSION",
    "candidate_gate",
    "dynamic_difficulty",
    "dynamic_utility",
    "frozen_thresholds",
    "jsonable",
    "residual_compatibility",
    "student_safe_project_teacher",
    "utility_projection_summary",
]
