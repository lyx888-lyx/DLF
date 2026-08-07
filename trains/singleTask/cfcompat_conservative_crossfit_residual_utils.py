"""Conservative cross-fit residual selection utilities for CFCompatKD v10.

v10 keeps the v9 architecture, video-grouped folds, and 4-of-5 median
consensus unchanged.  The only mechanism change is fold checkpoint selection:
after a Train-video-holdout trajectory has finished, select the earliest epoch
whose holdout J is within 1% of that fold's absolute best holdout J.
Official Valid and Test are never used by this selector.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .cfcompat_crossfit_residual_consensus_utils import (
    BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
    CONSENSUS_MIN_AGREE,
    DEV_SEED,
    J_MAX_DEGRADATION_VS_V8,
    N_FOLDS,
    NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
    OVERALL_NTR_MAX_DEGRADATION_VS_V4,
    RESIDUAL_INIT_SEED,
)


VERSION = "cfcompat_conservative_crossfit_residual_valid_screen_v10"
METHOD = "DLF-Frozen-S0-Conservative-Crossfit-Residual-CFCompatKD-v10"
OUTPUT_TAG = "cfcompat_conservative_crossfit_residual_v10"
RUN = "frozen_s0_conservative_crossfit_residual"

# Frozen before the single v10 mechanism run.  A 1% Train-holdout-J band is
# deliberately scale-relative and uses no Official Valid information.
NEAR_OPTIMAL_REL_TOL = 0.01


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


def select_earliest_near_optimal(
    epoch_records: Sequence[Mapping],
    rel_tol: float = NEAR_OPTIMAL_REL_TOL,
) -> dict:
    """Select earliest epoch within ``rel_tol`` of minimum Train-holdout J.

    Lower J is better.  The selector is deterministic and depends only on the
    fold's Train-video holdout trajectory.  ``epoch_records`` must contain
    ``Epoch`` and ``HoldoutJ``.
    """
    if not epoch_records:
        raise ValueError("No epoch records supplied to conservative selector.")
    if rel_tol < 0:
        raise ValueError("rel_tol must be non-negative.")
    cleaned = []
    for row in epoch_records:
        epoch = int(row["Epoch"])
        j = float(row["HoldoutJ"])
        if epoch <= 0 or not np.isfinite(j):
            raise ValueError("Invalid epoch/J in conservative selector.")
        cleaned.append({"Epoch": epoch, "HoldoutJ": j})
    if len({row["Epoch"] for row in cleaned}) != len(cleaned):
        raise ValueError("Duplicate epochs in conservative selector.")

    absolute_best = min(cleaned, key=lambda row: (row["HoldoutJ"], row["Epoch"]))
    cutoff = float(absolute_best["HoldoutJ"] * (1.0 + rel_tol))
    eligible = [row for row in cleaned if row["HoldoutJ"] <= cutoff + 1e-12]
    selected = min(eligible, key=lambda row: row["Epoch"])
    return {
        "absolute_best_epoch": int(absolute_best["Epoch"]),
        "absolute_best_holdout_J": float(absolute_best["HoldoutJ"]),
        "near_optimal_cutoff_J": cutoff,
        "relative_tolerance": float(rel_tol),
        "eligible_epoch_count": int(len(eligible)),
        "selected_epoch": int(selected["Epoch"]),
        "selected_holdout_J": float(selected["HoldoutJ"]),
        "selected_minus_best_J": float(
            selected["HoldoutJ"] - absolute_best["HoldoutJ"]
        ),
        "selected_relative_J_degradation": float(
            selected["HoldoutJ"] / absolute_best["HoldoutJ"] - 1.0
        ),
        "epoch_reduction_vs_absolute_best": int(
            absolute_best["Epoch"] - selected["Epoch"]
        ),
    }


def development_signal_gate(
    candidate_j: float,
    v8_j: float,
    candidate_transfer: Mapping,
    v8_transfer: Mapping,
    v4_transfer: Mapping,
) -> dict:
    """Keep the v9 frozen success criteria so only selector changes."""
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


__all__ = [
    "BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8",
    "CONSENSUS_MIN_AGREE",
    "DEV_SEED",
    "J_MAX_DEGRADATION_VS_V8",
    "METHOD",
    "NEAR_OPTIMAL_REL_TOL",
    "NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8",
    "N_FOLDS",
    "OUTPUT_TAG",
    "OVERALL_NTR_MAX_DEGRADATION_VS_V4",
    "RESIDUAL_INIT_SEED",
    "RUN",
    "VERSION",
    "development_signal_gate",
    "jsonable",
    "select_earliest_near_optimal",
]
