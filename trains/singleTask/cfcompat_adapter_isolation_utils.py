"""Frozen-backbone / missing-adapter isolation utilities for CFCompatKD v7.

v7 is a Seed1113, Valid-only mechanism test.  It keeps the v4
DISTILL/PRESERVE/ABSTAIN objective unchanged while hard-freezing the shared DLF
backbone (including buffers by keeping it in eval mode).  Only the existing
missing-modality parameters are trainable: two missing tokens and mask_adapter.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cfcompat_regret_preserve_utils import (
    DISTILL_MARGIN,
    NEGATIVE_TRANSFER_MARGIN,
    SEVERE_NEGATIVE_TRANSFER_MARGIN,
)
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_frozen_backbone_adapter_isolation_valid_screen_v7"
METHOD = "DLF-Frozen-Backbone-Adapter-Isolation-CFCompatKD-v7"
OUTPUT_TAG = "cfcompat_frozen_backbone_adapter_isolation_v7"
DEV_SEED = 1113
RUN = "frozen_backbone_adapter_isolation_cfcompat"
RUNS = (RUN,)

# Frozen before the single-seed v7 mechanism test.
J_MAX_DEGRADATION_VS_V4 = 0.002
BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED = 0.05
NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION = 0.03
OVERALL_NTR_MAX_DEGRADATION = 0.01
S0_PROTECT_MARGIN = 0.02

EXPECTED_TRAINABLE_NAMES = (
    "missing_audio_token",
    "missing_vision_token",
    "mask_adapter.weight",
)


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


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash parameters and buffers so frozen-backbone drift is detectable."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def freeze_shared_backbone(student: torch.nn.Module) -> dict:
    """Hard-freeze the shared backbone and expose only missing-specific params."""
    for attr in ("backbone", "missing_audio_token", "missing_vision_token", "mask_adapter"):
        if not hasattr(student, attr):
            raise AttributeError("Student lacks required isolation attribute: {}".format(attr))

    for parameter in student.parameters():
        parameter.requires_grad_(False)
    student.missing_audio_token.requires_grad_(True)
    student.missing_vision_token.requires_grad_(True)
    for parameter in student.mask_adapter.parameters():
        parameter.requires_grad_(True)

    trainable_names = tuple(
        name for name, parameter in student.named_parameters() if parameter.requires_grad
    )
    if trainable_names != EXPECTED_TRAINABLE_NAMES:
        raise RuntimeError(
            "Unexpected v7 trainables: {} != {}".format(
                trainable_names, EXPECTED_TRAINABLE_NAMES
            )
        )
    if any(parameter.requires_grad for parameter in student.backbone.parameters()):
        raise RuntimeError("A shared-backbone parameter remained trainable.")

    trainable_count = int(
        sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
    )
    total_count = int(sum(parameter.numel() for parameter in student.parameters()))
    return {
        "trainable_names": list(trainable_names),
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "trainable_parameter_fraction": float(trainable_count / max(total_count, 1)),
    }


def enforce_isolation_train_mode(student: torch.nn.Module) -> None:
    """Train missing-specific modules while keeping the shared function frozen."""
    student.train()
    student.backbone.eval()
    student.mask_adapter.train()
    if student.backbone.training:
        raise RuntimeError("Frozen backbone entered train mode.")
    if not student.mask_adapter.training:
        raise RuntimeError("mask_adapter did not enter train mode.")


def assert_no_backbone_gradients(student: torch.nn.Module) -> None:
    offenders = [
        name
        for name, parameter in student.backbone.named_parameters()
        if parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError("Frozen backbone received gradients: {}".format(offenders[:8]))


def prediction_frame_to_long(
    frame: pd.DataFrame,
    epoch: int,
    was_best_when_observed: bool = False,
) -> pd.DataFrame:
    required = {"sample_index", "sample_id", "label"}.union(
        {"{}_pred".format(mode) for mode in ("LAV",) + MISSING_MODES}
    )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Prediction frame lacks columns: {}".format(sorted(missing)))
    rows = []
    for record in frame.itertuples(index=False):
        for mode in ("LAV",) + MISSING_MODES:
            rows.append(
                {
                    "Epoch": int(epoch),
                    "WasBestWhenObserved": bool(was_best_when_observed),
                    "Mode": str(mode),
                    "sample_index": int(record.sample_index),
                    "sample_id": str(record.sample_id),
                    "label": float(record.label),
                    "prediction": float(getattr(record, "{}_pred".format(mode))),
                }
            )
    return pd.DataFrame(rows)


def _reference_long(reference: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_index", "label", "teacher_prediction"}.union(
        {"baseline_{}_pred".format(mode) for mode in ("LAV",) + MISSING_MODES}
    )
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError("Reference frame lacks columns: {}".format(sorted(missing)))
    rows = []
    for record in reference.itertuples(index=False):
        sample_id = str(getattr(record, "sample_id", ""))
        for mode in ("LAV",) + MISSING_MODES:
            rows.append(
                {
                    "sample_index": int(record.sample_index),
                    "sample_id_reference": sample_id,
                    "label": float(record.label),
                    "Mode": str(mode),
                    "baseline_prediction": float(
                        getattr(record, "baseline_{}_pred".format(mode))
                    ),
                    "teacher_prediction": float(record.teacher_prediction),
                }
            )
    return pd.DataFrame(rows)


def build_epoch_event_trajectory(
    epoch_predictions: pd.DataFrame,
    reference: pd.DataFrame,
    selected_best_epoch: int,
) -> pd.DataFrame:
    """Attach baseline/Teacher/S0 geometry to every Valid epoch prediction."""
    ref = _reference_long(reference)
    merged = epoch_predictions.merge(
        ref,
        on=["sample_index", "label", "Mode"],
        how="inner",
        validate="many_to_one",
    )
    if len(merged) != len(epoch_predictions):
        raise RuntimeError("Epoch/reference prediction binding lost rows.")
    s0 = merged.loc[merged.Epoch.astype(int).eq(0), [
        "sample_index", "Mode", "prediction"
    ]].rename(columns={"prediction": "s0_prediction"})
    if s0.duplicated(["sample_index", "Mode"]).any():
        raise RuntimeError("S0 epoch predictions are not unique.")
    merged = merged.merge(s0, on=["sample_index", "Mode"], how="left", validate="many_to_one")
    if merged.s0_prediction.isna().any():
        raise RuntimeError("S0 prediction binding failed.")

    merged["SelectedBestValid"] = merged.Epoch.astype(int).eq(int(selected_best_epoch))
    merged["baseline_error"] = np.abs(merged.baseline_prediction - merged.label)
    merged["teacher_error"] = np.abs(merged.teacher_prediction - merged.label)
    merged["s0_error"] = np.abs(merged.s0_prediction - merged.label)
    merged["prediction_error"] = np.abs(merged.prediction - merged.label)
    merged["teacher_advantage"] = merged.baseline_error - merged.teacher_error
    merged["s0_gain_vs_baseline"] = merged.baseline_error - merged.s0_error
    merged["gain_vs_baseline"] = merged.baseline_error - merged.prediction_error
    merged["abs_drift_from_s0"] = np.abs(merged.prediction - merged.s0_prediction)
    merged["teacher_beneficial"] = merged.teacher_advantage >= DISTILL_MARGIN
    merged["s0_beneficial"] = merged.s0_gain_vs_baseline >= S0_PROTECT_MARGIN
    merged["positive_transfer"] = merged.gain_vs_baseline > NEGATIVE_TRANSFER_MARGIN
    merged["negative_transfer"] = merged.gain_vs_baseline < -NEGATIVE_TRANSFER_MARGIN
    merged["severe_negative_transfer"] = (
        merged.gain_vs_baseline < -SEVERE_NEGATIVE_TRANSFER_MARGIN
    )
    merged["worse_than_s0_by_margin"] = (
        merged.prediction_error - merged.s0_error >= S0_PROTECT_MARGIN
    )
    merged["crossed_baseline_from_s0"] = (
        (merged.s0_prediction - merged.baseline_prediction)
        * (merged.prediction - merged.baseline_prediction)
        < 0.0
    )
    numeric = [
        "prediction", "baseline_prediction", "teacher_prediction", "s0_prediction",
        "baseline_error", "teacher_error", "s0_error", "prediction_error",
        "teacher_advantage", "s0_gain_vs_baseline", "gain_vs_baseline",
        "abs_drift_from_s0",
    ]
    if not np.isfinite(merged[numeric].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("Epoch trajectory contains NaN/Inf.")
    return merged.sort_values(
        ["Epoch", "sample_index", "Mode"], kind="mergesort"
    ).reset_index(drop=True)


def _trajectory_group_row(epoch: int, group: str, local: pd.DataFrame) -> dict:
    if local.empty:
        raise ValueError("Empty trajectory group: {}".format(group))
    return {
        "Epoch": int(epoch),
        "Group": str(group),
        "N": int(len(local)),
        "mean_gain_vs_baseline": float(local.gain_vs_baseline.mean()),
        "positive_transfer_rate": float(local.positive_transfer.mean()),
        "negative_transfer_rate": float(local.negative_transfer.mean()),
        "severe_negative_transfer_rate": float(local.severe_negative_transfer.mean()),
        "mean_abs_drift_from_s0": float(local.abs_drift_from_s0.mean()),
        "crossed_baseline_from_s0_rate": float(local.crossed_baseline_from_s0.mean()),
        "worse_than_s0_by_margin_rate": float(local.worse_than_s0_by_margin.mean()),
    }


def epoch_transfer_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    missing = trajectory.loc[trajectory.Mode.astype(str).isin(MISSING_MODES)].copy()
    rows = []
    for epoch, local in missing.groupby("Epoch", sort=True):
        specs = (
            ("ALL_MISSING", np.ones(len(local), dtype=bool)),
            ("TEACHER_BENEFICIAL", local.teacher_beneficial.to_numpy(dtype=bool)),
            ("TEACHER_NONBENEFICIAL", ~local.teacher_beneficial.to_numpy(dtype=bool)),
            ("S0_BENEFICIAL", local.s0_beneficial.to_numpy(dtype=bool)),
            (
                "S0_AND_TEACHER_BENEFICIAL",
                local.s0_beneficial.to_numpy(dtype=bool)
                & local.teacher_beneficial.to_numpy(dtype=bool),
            ),
        )
        for group, mask in specs:
            if mask.any():
                rows.append(_trajectory_group_row(int(epoch), group, local.loc[mask]))
    return pd.DataFrame(rows)


def failure_onset_table(trajectory: pd.DataFrame) -> pd.DataFrame:
    local = trajectory.loc[
        trajectory.Mode.astype(str).isin(MISSING_MODES)
        & trajectory.Epoch.astype(int).ge(1)
    ].copy()
    rows = []
    for (sample_index, mode), group in local.groupby(["sample_index", "Mode"], sort=True):
        group = group.sort_values("Epoch", kind="mergesort")

        def first_epoch(column):
            hit = group.loc[group[column].astype(bool), "Epoch"]
            return int(hit.iloc[0]) if len(hit) else np.nan

        first = group.iloc[0]
        rows.append(
            {
                "sample_index": int(sample_index),
                "sample_id": str(first.sample_id),
                "Mode": str(mode),
                "label": float(first.label),
                "teacher_beneficial": bool(first.teacher_beneficial),
                "s0_beneficial": bool(first.s0_beneficial),
                "s0_gain_vs_baseline": float(first.s0_gain_vs_baseline),
                "teacher_advantage": float(first.teacher_advantage),
                "first_negative_transfer_epoch": first_epoch("negative_transfer"),
                "first_severe_negative_transfer_epoch": first_epoch("severe_negative_transfer"),
                "first_cross_baseline_from_s0_epoch": first_epoch("crossed_baseline_from_s0"),
                "first_worse_than_s0_by_margin_epoch": first_epoch("worse_than_s0_by_margin"),
            }
        )
    return pd.DataFrame(rows)


def clip_failure_epoch_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    local = trajectory.loc[
        trajectory.Mode.astype(str).isin(MISSING_MODES)
        & trajectory.Epoch.astype(int).ge(1)
    ].copy()
    rows = []
    for epoch, epoch_frame in local.groupby("Epoch", sort=True):
        clip = epoch_frame.groupby("sample_index", sort=False).agg(
            negative_mode_count=("negative_transfer", "sum"),
            severe_mode_count=("severe_negative_transfer", "sum"),
            cross_mode_count=("crossed_baseline_from_s0", "sum"),
            teacher_beneficial_mode_count=("teacher_beneficial", "sum"),
            s0_beneficial_mode_count=("s0_beneficial", "sum"),
        )
        rows.append(
            {
                "Epoch": int(epoch),
                "clip_count": int(len(clip)),
                "mean_negative_modes_per_clip": float(clip.negative_mode_count.mean()),
                "three_mode_negative_clip_count": int((clip.negative_mode_count == 3).sum()),
                "three_mode_severe_clip_count": int((clip.severe_mode_count == 3).sum()),
                "three_mode_cross_baseline_clip_count": int((clip.cross_mode_count == 3).sum()),
                "all3_teacher_beneficial_clip_count": int(
                    (clip.teacher_beneficial_mode_count == 3).sum()
                ),
                "all3_teacher_beneficial_and_all3_negative_clip_count": int(
                    (
                        (clip.teacher_beneficial_mode_count == 3)
                        & (clip.negative_mode_count == 3)
                    ).sum()
                ),
                "all3_s0_beneficial_and_all3_negative_clip_count": int(
                    (
                        (clip.s0_beneficial_mode_count == 3)
                        & (clip.negative_mode_count == 3)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _subset_transfer(local: pd.DataFrame) -> dict:
    gain = local.gain_vs_dlf.to_numpy(dtype=np.float64)
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
    required = {"Seed", "Run", "Mode", "teacher_advantage", "gain_vs_dlf"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Valid events lack mechanism columns: {}".format(sorted(missing)))
    local = events.loc[
        events.Seed.astype(int).eq(DEV_SEED)
        & events.Run.astype(str).eq(str(run))
        & events.Mode.astype(str).isin(MISSING_MODES)
    ].copy()
    if len(local) != 229 * len(MISSING_MODES):
        raise RuntimeError("Expected exactly 687 Seed1113 missing-mode Valid events.")
    beneficial = local.teacher_advantage >= DISTILL_MARGIN
    return {
        "teacher_beneficial_prevalence": float(beneficial.mean()),
        "all_missing": _subset_transfer(local),
        "teacher_beneficial": _subset_transfer(local.loc[beneficial]),
        "teacher_nonbeneficial": _subset_transfer(local.loc[~beneficial]),
    }


def development_signal_gate(candidate_j, v4_j, candidate_transfer, v4_transfer) -> dict:
    beneficial_reduction = (
        float(v4_transfer["teacher_beneficial"]["negative_transfer_rate"])
        - float(candidate_transfer["teacher_beneficial"]["negative_transfer_rate"])
    )
    nonbeneficial_degradation = (
        float(candidate_transfer["teacher_nonbeneficial"]["negative_transfer_rate"])
        - float(v4_transfer["teacher_nonbeneficial"]["negative_transfer_rate"])
    )
    overall_degradation = (
        float(candidate_transfer["all_missing"]["negative_transfer_rate"])
        - float(v4_transfer["all_missing"]["negative_transfer_rate"])
    )
    delta_j = float(candidate_j) - float(v4_j)
    checks = {
        "valid_J_noninferior_to_v4": bool(delta_j <= J_MAX_DEGRADATION_VS_V4),
        "beneficial_teacher_NTR_reduction": bool(
            beneficial_reduction >= BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED
        ),
        "nonbeneficial_teacher_NTR_retained": bool(
            nonbeneficial_degradation <= NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION
        ),
        "overall_NTR_not_worse": bool(overall_degradation <= OVERALL_NTR_MAX_DEGRADATION),
    }
    return {
        "passed": bool(all(checks.values())),
        "candidate_J": float(candidate_j),
        "v4_J": float(v4_j),
        "delta_J_candidate_minus_v4": delta_j,
        "beneficial_teacher_NTR_reduction_vs_v4": beneficial_reduction,
        "nonbeneficial_teacher_NTR_degradation_vs_v4": nonbeneficial_degradation,
        "overall_NTR_degradation_vs_v4": overall_degradation,
        "checks": checks,
        "thresholds": {
            "J_max_degradation_vs_v4": J_MAX_DEGRADATION_VS_V4,
            "beneficial_teacher_NTR_reduction_required": BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED,
            "nonbeneficial_teacher_NTR_max_degradation": NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION,
            "overall_NTR_max_degradation": OVERALL_NTR_MAX_DEGRADATION,
        },
    }
