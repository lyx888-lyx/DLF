"""Aggregate-only utilities for the CFCompatKD exploratory MOSI Test probe.

This module deliberately contains no model selection or tuning logic.  It is
used after all compared checkpoints have already been frozen on non-Test data.
No sample-level Test artifact is written by the caller.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .cfcompat_regret_preserve_utils import (
    DISTILL_MARGIN,
    NEGATIVE_TRANSFER_MARGIN,
    SEVERE_NEGATIVE_TRANSFER_MARGIN,
)
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_exploratory_test_viability_v13"
METHOD_ORDER = (
    "original_cfcompat_v1",
    "regret_preserve_v4",
    "sample_residual_v8",
    "gradient_surgery_v12",
    "adam_step_safety_v13",
)
PRIMARY_REFERENCE = "original_cfcompat_v1"


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


def build_missing_events(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Create an in-memory Test event frame; callers must not persist it."""
    required_candidate = {"sample_index", "label"}.union(
        {"{}_pred".format(mode) for mode in ("LAV",) + MISSING_MODES}
    )
    required_reference = {"sample_index", "label", "teacher_prediction"}.union(
        {"baseline_{}_pred".format(mode) for mode in ("LAV",) + MISSING_MODES}
    )
    missing_candidate = required_candidate.difference(candidate.columns)
    missing_reference = required_reference.difference(reference.columns)
    if missing_candidate or missing_reference:
        raise ValueError(
            "Missing Test columns candidate={} reference={}".format(
                sorted(missing_candidate), sorted(missing_reference)
            )
        )
    left = candidate.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    right = reference.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(left) != len(right) or len(left) == 0:
        raise RuntimeError("Candidate/reference Test sample counts differ.")
    if left.sample_index.duplicated().any() or right.sample_index.duplicated().any():
        raise RuntimeError("Test sample_index is not unique.")
    if not np.array_equal(
        left.sample_index.to_numpy(dtype=np.int64),
        right.sample_index.to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Candidate/reference Test sample order binding failed.")
    if np.max(np.abs(left.label.to_numpy(float) - right.label.to_numpy(float))) > 1e-7:
        raise RuntimeError("Candidate/reference Test label binding failed.")

    rows = []
    for candidate_row, reference_row in zip(
        left.itertuples(index=False), right.itertuples(index=False)
    ):
        label = float(candidate_row.label)
        teacher_prediction = float(reference_row.teacher_prediction)
        for mode in MISSING_MODES:
            baseline_prediction = float(
                getattr(reference_row, "baseline_{}_pred".format(mode))
            )
            candidate_prediction = float(
                getattr(candidate_row, "{}_pred".format(mode))
            )
            baseline_error = abs(baseline_prediction - label)
            candidate_error = abs(candidate_prediction - label)
            teacher_error = abs(teacher_prediction - label)
            gain = baseline_error - candidate_error
            rows.append(
                {
                    "Method": str(method),
                    "Mode": str(mode),
                    "sample_index": int(candidate_row.sample_index),
                    "gain_vs_baseline": float(gain),
                    "teacher_advantage": float(baseline_error - teacher_error),
                    "negative_transfer": bool(gain < -NEGATIVE_TRANSFER_MARGIN),
                    "severe_negative_transfer": bool(
                        gain < -SEVERE_NEGATIVE_TRANSFER_MARGIN
                    ),
                    "positive_transfer": bool(gain > NEGATIVE_TRANSFER_MARGIN),
                    "not_improved": bool(gain <= 0.0),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != len(left) * len(MISSING_MODES):
        raise RuntimeError("Unexpected Test missing-event count.")
    numeric = frame[["gain_vs_baseline", "teacher_advantage"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Non-finite Test transfer event.")
    return frame


def _subset_summary(local: pd.DataFrame) -> dict:
    if local.empty:
        raise ValueError("Cannot summarize an empty Test transfer subset.")
    return {
        "N": int(len(local)),
        "mean_gain_vs_baseline": float(local.gain_vs_baseline.mean()),
        "positive_transfer_rate": float(local.positive_transfer.astype(bool).mean()),
        "negative_transfer_rate": float(local.negative_transfer.astype(bool).mean()),
        "severe_negative_transfer_rate": float(
            local.severe_negative_transfer.astype(bool).mean()
        ),
        "not_improved_rate": float(local.not_improved.astype(bool).mean()),
    }


def transfer_summary(events: pd.DataFrame) -> dict:
    beneficial = events.teacher_advantage.astype(float) >= float(DISTILL_MARGIN)
    return {
        "teacher_beneficial_prevalence": float(beneficial.mean()),
        "all_missing": _subset_summary(events),
        "teacher_beneficial": _subset_summary(events.loc[beneficial]),
        "teacher_nonbeneficial": _subset_summary(events.loc[~beneficial]),
    }


def transfer_rows(method: str, summary: Mapping) -> list[dict]:
    result = []
    for group in ("all_missing", "teacher_beneficial", "teacher_nonbeneficial"):
        result.append(
            {
                "Method": str(method),
                "Group": str(group),
                **dict(summary[group]),
            }
        )
    return result


def comparison_row(method: str, metrics: Mapping, transfer: Mapping) -> dict:
    return {
        "Method": str(method),
        "TestJ": float(metrics["TestJ"]),
        "MissingMacroMAE": float(metrics["MissingMacroMAE"]),
        "LAV_MAE": float(metrics["LAV_MAE"]),
        "LA_MAE": float(metrics["LA_MAE"]),
        "LV_MAE": float(metrics["LV_MAE"]),
        "L_MAE": float(metrics["L_MAE"]),
        "OverallMeanGain": float(transfer["all_missing"]["mean_gain_vs_baseline"]),
        "OverallNTR": float(transfer["all_missing"]["negative_transfer_rate"]),
        "OverallSevereNTR": float(
            transfer["all_missing"]["severe_negative_transfer_rate"]
        ),
        "OverallPositiveTransfer": float(
            transfer["all_missing"]["positive_transfer_rate"]
        ),
        "BeneficialNTR": float(
            transfer["teacher_beneficial"]["negative_transfer_rate"]
        ),
        "BeneficialSevereNTR": float(
            transfer["teacher_beneficial"]["severe_negative_transfer_rate"]
        ),
        "BeneficialPositiveTransfer": float(
            transfer["teacher_beneficial"]["positive_transfer_rate"]
        ),
        "NonbeneficialNTR": float(
            transfer["teacher_nonbeneficial"]["negative_transfer_rate"]
        ),
        "NonbeneficialSevereNTR": float(
            transfer["teacher_nonbeneficial"]["severe_negative_transfer_rate"]
        ),
    }


def add_reference_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    if set(comparison.Method.astype(str)) != set(METHOD_ORDER):
        raise RuntimeError("Exploratory Test comparison method set is incomplete.")
    reference = comparison.loc[
        comparison.Method.astype(str).eq(PRIMARY_REFERENCE)
    ]
    if len(reference) != 1:
        raise RuntimeError("Primary Original-CFCompat reference is not unique.")
    ref = reference.iloc[0]
    result = comparison.copy()
    result["DeltaJVsOriginal"] = result.TestJ.astype(float) - float(ref.TestJ)
    result["OverallNTRReductionVsOriginal"] = (
        float(ref.OverallNTR) - result.OverallNTR.astype(float)
    )
    result["BeneficialNTRReductionVsOriginal"] = (
        float(ref.BeneficialNTR) - result.BeneficialNTR.astype(float)
    )
    result["SevereNTRReductionVsOriginal"] = (
        float(ref.OverallSevereNTR) - result.OverallSevereNTR.astype(float)
    )
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    result["_order"] = result.Method.astype(str).map(order)
    return result.sort_values("_order", kind="mergesort").drop(columns="_order")


def v13_route_decision(comparison: pd.DataFrame) -> dict:
    """Sign-only route decision fixed before Test access; not a tuning gate."""
    indexed = comparison.set_index(comparison.Method.astype(str), drop=False)
    if PRIMARY_REFERENCE not in indexed.index or "adam_step_safety_v13" not in indexed.index:
        raise RuntimeError("Original/v13 Test rows are required for route decision.")
    original = indexed.loc[PRIMARY_REFERENCE]
    v13 = indexed.loc["adam_step_safety_v13"]
    checks = {
        "test_J_improves_original": bool(float(v13.TestJ) < float(original.TestJ)),
        "beneficial_NTR_improves_original": bool(
            float(v13.BeneficialNTR) < float(original.BeneficialNTR)
        ),
        "overall_NTR_not_worse_than_original": bool(
            float(v13.OverallNTR) <= float(original.OverallNTR)
        ),
    }
    if all(checks.values()):
        verdict = "CLEAR_POSITIVE_GENERALIZATION_SIGNAL"
    elif not any(checks.values()):
        verdict = "CLEAR_NEGATIVE_GENERALIZATION_SIGNAL"
    else:
        verdict = "MIXED_GENERALIZATION_SIGNAL"
    return {
        "verdict": verdict,
        "checks": checks,
        "delta_J_v13_minus_original": float(v13.TestJ) - float(original.TestJ),
        "beneficial_NTR_reduction_v13_vs_original": float(original.BeneficialNTR)
        - float(v13.BeneficialNTR),
        "overall_NTR_reduction_v13_vs_original": float(original.OverallNTR)
        - float(v13.OverallNTR),
        "severe_NTR_reduction_v13_vs_original": float(original.OverallSevereNTR)
        - float(v13.OverallSevereNTR),
    }


__all__ = [
    "METHOD_ORDER",
    "PRIMARY_REFERENCE",
    "VERSION",
    "add_reference_deltas",
    "build_missing_events",
    "comparison_row",
    "jsonable",
    "transfer_rows",
    "transfer_summary",
    "v13_route_decision",
]
