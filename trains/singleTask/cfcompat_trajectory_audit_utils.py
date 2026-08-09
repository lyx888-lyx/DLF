"""Trajectory-resolved Train-OOF diagnostics for CFCompatKD v12.2.

v12.2 is diagnostic only.  It does not define a new training method.  The
formal v12 numerical-hotfix training function is replayed unchanged while a
read-only hook captures the residual-bank state that v12 already snapshots at
each epoch.  All OOF prediction/gradient diagnostics are run *after* the full
five-fold replay, so diagnostic passes cannot perturb the training RNG path.
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


VERSION = "cfcompat_trajectory_train_oof_audit_v12p2"
METHOD = "DLF-v12-Exact-Replay-Trajectory-Train-OOF-Audit-v12.2"
OUTPUT_TAG = "cfcompat_trajectory_audit_v12p2"
DEV_SEED = 1113
N_FOLDS = 5

# Frozen before running v12.2.  Selected/absolute-best epochs are also audited
# because they are determined by the already-frozen Train-video selector, not
# by any OOF gradient result from this diagnostic.
FIXED_GRADIENT_MILESTONES = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64)

TRAJECTORY_GROUPS = (
    "ALL_MISSING",
    "TEACHER_BENEFICIAL",
    "TEACHER_NONBENEFICIAL",
    "S0_BENEFICIAL",
    "S0_NONBENEFICIAL",
    "S0_AND_TEACHER_BENEFICIAL",
)


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


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> dict:
    return {str(k): v.detach().cpu().clone() for k, v in state.items()}


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Content hash independent of torch.save container metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def states_exactly_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key].detach().cpu(), right[key].detach().cpu()) for key in left)


def enrich_transfer_flags(events: pd.DataFrame) -> pd.DataFrame:
    """Attach the exact transfer margins used by the historical mechanism audit."""
    required = {
        "current_gain_vs_baseline",
        "current_error",
        "s0_error",
        "teacher_beneficial",
        "s0_beneficial",
        "residual_crossed_label_from_s0",
        "residual_crossed_baseline_from_s0",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Trajectory events lack columns: {}".format(sorted(missing)))
    result = events.copy()
    gain = result.current_gain_vs_baseline.astype(float)
    result["positive_transfer"] = gain > NEGATIVE_TRANSFER_MARGIN
    result["negative_transfer"] = gain < -NEGATIVE_TRANSFER_MARGIN
    result["severe_negative_transfer"] = gain < -SEVERE_NEGATIVE_TRANSFER_MARGIN
    result["worse_than_s0_by_margin"] = (
        result.current_error.astype(float) - result.s0_error.astype(float) >= DISTILL_MARGIN
    )
    return result


def _group_mask(frame: pd.DataFrame, group: str) -> np.ndarray:
    teacher = frame.teacher_beneficial.astype(bool).to_numpy()
    s0 = frame.s0_beneficial.astype(bool).to_numpy()
    if group == "ALL_MISSING":
        return np.ones(len(frame), dtype=bool)
    if group == "TEACHER_BENEFICIAL":
        return teacher
    if group == "TEACHER_NONBENEFICIAL":
        return ~teacher
    if group == "S0_BENEFICIAL":
        return s0
    if group == "S0_NONBENEFICIAL":
        return ~s0
    if group == "S0_AND_TEACHER_BENEFICIAL":
        return s0 & teacher
    raise KeyError(group)


def epoch_group_summary(events: pd.DataFrame) -> pd.DataFrame:
    required = {"Fold", "Epoch", "current_gain_vs_baseline", "current_gain_vs_s0", "residual_abs_delta"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Trajectory events lack summary columns: {}".format(sorted(missing)))
    rows = []
    for (fold, epoch), local in events.groupby(["Fold", "Epoch"], sort=True):
        for group in TRAJECTORY_GROUPS:
            mask = _group_mask(local, group)
            if not mask.any():
                continue
            subset = local.loc[mask]
            rows.append(
                {
                    "Fold": int(fold),
                    "Epoch": int(epoch),
                    "Group": group,
                    "N": int(len(subset)),
                    "mean_gain_vs_baseline": float(subset.current_gain_vs_baseline.mean()),
                    "mean_gain_vs_s0": float(subset.current_gain_vs_s0.mean()),
                    "positive_transfer_rate": float(subset.positive_transfer.astype(bool).mean()),
                    "negative_transfer_rate": float(subset.negative_transfer.astype(bool).mean()),
                    "severe_negative_transfer_rate": float(subset.severe_negative_transfer.astype(bool).mean()),
                    "mean_abs_residual": float(subset.residual_abs_delta.mean()),
                    "crossed_label_from_s0_rate": float(subset.residual_crossed_label_from_s0.astype(bool).mean()),
                    "crossed_baseline_from_s0_rate": float(subset.residual_crossed_baseline_from_s0.astype(bool).mean()),
                    "worse_than_s0_by_margin_rate": float(subset.worse_than_s0_by_margin.astype(bool).mean()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_epoch_groups(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (epoch, group), local in per_fold.groupby(["Epoch", "Group"], sort=True):
        weights = local.N.to_numpy(dtype=np.float64)
        total = float(weights.sum())
        row = {"Epoch": int(epoch), "Group": str(group), "FoldCount": int(local.Fold.nunique()), "N": int(total)}
        for column in (
            "mean_gain_vs_baseline",
            "mean_gain_vs_s0",
            "positive_transfer_rate",
            "negative_transfer_rate",
            "severe_negative_transfer_rate",
            "mean_abs_residual",
            "crossed_label_from_s0_rate",
            "crossed_baseline_from_s0_rate",
            "worse_than_s0_by_margin_rate",
        ):
            row[column] = float(np.average(local[column].to_numpy(float), weights=weights))
        rows.append(row)
    return pd.DataFrame(rows)


def failure_onset_table(events: pd.DataFrame) -> pd.DataFrame:
    """First epoch at which each OOF sample/mode enters each harmful state."""
    local = events.loc[events.Epoch.astype(int).ge(1)].copy()
    rows = []
    for (fold, sample_index, mode), group in local.groupby(
        ["Fold", "sample_index", "mode"], sort=True
    ):
        group = group.sort_values("Epoch", kind="mergesort")

        def first_epoch(column: str):
            hit = group.loc[group[column].astype(bool), "Epoch"]
            return int(hit.iloc[0]) if len(hit) else np.nan

        first = group.iloc[0]
        rows.append(
            {
                "Fold": int(fold),
                "sample_index": int(sample_index),
                "sample_id": str(first.sample_id),
                "video_id": str(first.video_id),
                "mode": str(mode),
                "teacher_beneficial": bool(first.teacher_beneficial),
                "s0_beneficial": bool(first.s0_beneficial),
                "s0_gain_vs_baseline": float(first.s0_gain_vs_baseline),
                "first_negative_transfer_epoch": first_epoch("negative_transfer"),
                "first_severe_negative_transfer_epoch": first_epoch("severe_negative_transfer"),
                "first_cross_label_epoch": first_epoch("residual_crossed_label_from_s0"),
                "first_cross_baseline_epoch": first_epoch("residual_crossed_baseline_from_s0"),
                "first_worse_than_s0_epoch": first_epoch("worse_than_s0_by_margin"),
            }
        )
    return pd.DataFrame(rows)


def clip_sync_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Track synchronized three-mode OOF failures as the trajectory evolves."""
    rows = []
    local = events.loc[events.Epoch.astype(int).ge(0)].copy()
    for (fold, epoch), epoch_frame in local.groupby(["Fold", "Epoch"], sort=True):
        clip = epoch_frame.groupby("sample_index", sort=False).agg(
            negative_mode_count=("negative_transfer", "sum"),
            severe_mode_count=("severe_negative_transfer", "sum"),
            cross_label_mode_count=("residual_crossed_label_from_s0", "sum"),
            teacher_beneficial_mode_count=("teacher_beneficial", "sum"),
            s0_beneficial_mode_count=("s0_beneficial", "sum"),
        )
        rows.append(
            {
                "Fold": int(fold),
                "Epoch": int(epoch),
                "ClipN": int(len(clip)),
                "three_mode_negative_clip_count": int((clip.negative_mode_count == 3).sum()),
                "three_mode_severe_clip_count": int((clip.severe_mode_count == 3).sum()),
                "three_mode_cross_label_clip_count": int((clip.cross_label_mode_count == 3).sum()),
                "all3_teacher_beneficial_clip_count": int((clip.teacher_beneficial_mode_count == 3).sum()),
                "all3_teacher_beneficial_and_all3_negative_clip_count": int(
                    ((clip.teacher_beneficial_mode_count == 3) & (clip.negative_mode_count == 3)).sum()
                ),
                "all3_s0_beneficial_and_all3_negative_clip_count": int(
                    ((clip.s0_beneficial_mode_count == 3) & (clip.negative_mode_count == 3)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def milestone_roles(available_epochs: Sequence[int], selected_epoch: int, best_epoch: int) -> dict:
    available = {int(x) for x in available_epochs}
    roles = {}
    for epoch in FIXED_GRADIENT_MILESTONES:
        if int(epoch) in available:
            roles.setdefault(int(epoch), []).append("FIXED_MILESTONE")
    if int(selected_epoch) in available:
        roles.setdefault(int(selected_epoch), []).append("CONSERVATIVE_SELECTED")
    if int(best_epoch) in available:
        roles.setdefault(int(best_epoch), []).append("ABSOLUTE_BEST")
    return {epoch: "+".join(names) for epoch, names in sorted(roles.items())}


__all__ = [
    "DEV_SEED",
    "FIXED_GRADIENT_MILESTONES",
    "METHOD",
    "N_FOLDS",
    "OUTPUT_TAG",
    "TRAJECTORY_GROUPS",
    "VERSION",
    "aggregate_epoch_groups",
    "clip_sync_summary",
    "clone_state_dict",
    "enrich_transfer_flags",
    "epoch_group_summary",
    "failure_onset_table",
    "jsonable",
    "milestone_roles",
    "state_dict_sha256",
    "states_exactly_equal",
]
