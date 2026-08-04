"""Utilities for the CFCompatKD seed-mechanism audit.

This module is analysis-only. It never constructs Test, trains a model, changes
an optimizer, or selects a new method.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


VERSION = "cfcompat_seed_mechanism_audit_v1"
METHOD = "DLF-CFCompatKD-SeedMechanismAudit-v1"
OUTPUT_TAG = "cfcompat_seed_mechanism_audit_v1"
FORMAL_SEEDS = (1111, 1114)
PROBE_SAMPLE_COUNT = 64
MAX_SAMPLES_PER_VIDEO = 2
MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
OBJECTIVE_PAIRS = (
    ("full", "missing"),
    ("full", "kd"),
    ("missing", "kd"),
)
PARAMETER_GROUPS = (
    "mask_path",
    "fusion_head",
    "multimodal_body",
    "bert_top",
)


def parse_video_id(sample_id: object) -> str:
    """Parse the source video from standard CMU-MOSI/MOSEI segment IDs."""
    value = str(sample_id)
    if "$_$" in value:
        return value.split("$_$", 1)[0]
    match = re.match(r"^(.*?)(?:\[[0-9]+\])$", value)
    if match:
        return match.group(1)
    match = re.match(r"^(.*?)[_-](?:seg(?:ment)?[_-]?)?[0-9]+$", value, re.I)
    if match:
        return match.group(1)
    raise ValueError("Unable to parse source video from sample id: {}".format(value))


def label_bin(value: float) -> str:
    value = float(value)
    if value < -0.5:
        return "negative"
    if value > 0.5:
        return "positive"
    return "near_zero"


def intensity_bin(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude <= 0.5:
        return "abs_le_0p5"
    if magnitude <= 1.5:
        return "abs_0p5_to_1p5"
    return "abs_gt_1p5"


def stable_hash(value: object, salt: str = "cfcompat-mechanism-v1") -> str:
    return hashlib.sha256((salt + "|" + str(value)).encode("utf-8")).hexdigest()


def select_probe_positions(
    metadata: pd.DataFrame,
    target_count: int = PROBE_SAMPLE_COUNT,
    max_per_video: int = MAX_SAMPLES_PER_VIDEO,
) -> pd.DataFrame:
    """Select a deterministic, label-balanced, video-diverse train probe."""
    required = {"dataset_position", "sample_index", "sample_id", "video_id", "label"}
    if not required.issubset(metadata.columns):
        raise ValueError("Probe metadata is missing columns: {}".format(
            sorted(required - set(metadata.columns))
        ))
    if metadata.sample_index.nunique() != len(metadata):
        raise ValueError("Probe source sample_index values are not unique.")
    frame = metadata.copy()
    frame["label_bin"] = frame.label.map(label_bin)
    frame["stable_key"] = frame.sample_id.map(stable_hash)
    frame = frame.sort_values(
        ["stable_key", "sample_index"], kind="mergesort"
    ).reset_index(drop=True)

    bins = ("negative", "near_zero", "positive")
    base = int(target_count) // len(bins)
    quotas = {name: base for name in bins}
    for name in bins[: int(target_count) - base * len(bins)]:
        quotas[name] += 1

    selected = []
    video_counts = defaultdict(int)
    bin_counts = defaultdict(int)

    for row in frame.itertuples(index=False):
        bucket = str(row.label_bin)
        video = str(row.video_id)
        if bin_counts[bucket] >= quotas[bucket]:
            continue
        if video_counts[video] >= int(max_per_video):
            continue
        selected.append(row._asdict())
        video_counts[video] += 1
        bin_counts[bucket] += 1
        if len(selected) == int(target_count):
            break

    if len(selected) < int(target_count):
        chosen = {int(row["sample_index"]) for row in selected}
        for row in frame.itertuples(index=False):
            video = str(row.video_id)
            if int(row.sample_index) in chosen:
                continue
            if video_counts[video] >= int(max_per_video):
                continue
            selected.append(row._asdict())
            chosen.add(int(row.sample_index))
            video_counts[video] += 1
            if len(selected) == int(target_count):
                break

    result = pd.DataFrame(selected)
    if len(result) != int(target_count):
        raise RuntimeError(
            "Unable to select {} probe samples with max_per_video={}; got {}."
            .format(target_count, max_per_video, len(result))
        )
    if result.sample_index.nunique() != len(result):
        raise RuntimeError("Probe selection contains duplicate samples.")
    if int(result.groupby("video_id").size().max()) > int(max_per_video):
        raise RuntimeError("Probe selection violates max_per_video.")
    result["selection_rank"] = np.arange(len(result), dtype=int)
    return result.sort_values("selection_rank", kind="mergesort")


def rank_compatibility_proxy(deltas: Sequence[float]) -> np.ndarray:
    """Return a descriptive within-Valid compatibility proxy, not the train gate."""
    values = pd.Series(np.asarray(deltas, dtype=np.float64))
    if len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Compatibility proxy requires finite non-negative deltas.")
    ranks = values.rank(method="average", ascending=True).to_numpy(dtype=float)
    q = (ranks - 0.5) / float(len(values))
    return 1.0 - q


def parameter_groups(model: torch.nn.Module) -> dict:
    """Return fixed, non-overlapping parameter groups for exact cosine audits."""
    groups = {name: [] for name in PARAMETER_GROUPS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            name.startswith("missing_audio_token")
            or name.startswith("missing_vision_token")
            or name.startswith("mask_adapter.")
        ):
            groups["mask_path"].append((name, parameter))
            continue
        if (
            name.startswith("backbone.proj1.")
            or name.startswith("backbone.proj2.")
            or name.startswith("backbone.out_layer.")
        ):
            groups["fusion_head"].append((name, parameter))
            continue
        if "backbone.text_model.model.encoder.layer.11." in name:
            groups["bert_top"].append((name, parameter))
            continue
        if name.startswith("backbone.") and not name.startswith(
            "backbone.text_model."
        ):
            groups["multimodal_body"].append((name, parameter))

    missing = [name for name, values in groups.items() if not values]
    if missing:
        raise RuntimeError("Empty mechanism parameter groups: {}".format(missing))
    all_ids = []
    for values in groups.values():
        all_ids.extend(id(parameter) for _, parameter in values)
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Mechanism parameter groups overlap.")
    return groups


def gradient_pair_stats(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> dict:
    if len(left) != len(right):
        raise ValueError("Gradient lists differ in length.")
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    finite = True
    active = 0
    for first, second in zip(left, right):
        if first is None and second is None:
            continue
        if first is None:
            first = torch.zeros_like(second)
        if second is None:
            second = torch.zeros_like(first)
        finite = finite and bool(torch.isfinite(first).all()) and bool(
            torch.isfinite(second).all()
        )
        first_value = first.detach().float()
        second_value = second.detach().float()
        dot += float((first_value * second_value).sum().cpu())
        left_sq += float(first_value.pow(2).sum().cpu())
        right_sq += float(second_value.pow(2).sum().cpu())
        active += int(first.numel())
    left_norm = math.sqrt(max(left_sq, 0.0))
    right_norm = math.sqrt(max(right_sq, 0.0))
    if left_norm > 0.0 and right_norm > 0.0:
        cosine = dot / (left_norm * right_norm)
        cosine = max(-1.0, min(1.0, float(cosine)))
    else:
        cosine = float("nan")
    return {
        "dot": float(dot),
        "left_norm": float(left_norm),
        "right_norm": float(right_norm),
        "cosine": cosine,
        "conflict": bool(math.isfinite(cosine) and cosine < 0.0),
        "finite": bool(finite),
        "active_numel": int(active),
    }


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def pearson(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    left = pd.Series(np.asarray(first, dtype=np.float64)).rank(method="average")
    right = pd.Series(np.asarray(second, dtype=np.float64)).rank(method="average")
    return pearson(left, right)


def calibration_stats(labels: Sequence[float], predictions: Sequence[float]) -> dict:
    label = np.asarray(labels, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if len(label) == 0 or not np.isfinite(label).all() or not np.isfinite(pred).all():
        raise ValueError("Calibration stats require finite non-empty arrays.")
    variance = float(np.var(label))
    slope = float(np.cov(label, pred, ddof=0)[0, 1] / variance) if variance > 0 else 0.0
    intercept = float(pred.mean() - slope * label.mean())
    return {
        "count": int(len(label)),
        "label_mean": float(label.mean()),
        "label_std": float(label.std()),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "prediction_abs_mean": float(np.abs(pred).mean()),
        "mae": float(np.abs(pred - label).mean()),
        "slope_pred_on_label": slope,
        "intercept_pred_on_label": intercept,
        "corr": pearson(label, pred),
    }


def weighted_sample_error(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return (
        0.5 * frame[f"{prefix}_LAV_abs_error"].to_numpy(dtype=float)
        + (1.0 / 6.0)
        * sum(
            frame[f"{prefix}_{mode}_abs_error"].to_numpy(dtype=float)
            for mode in MISSING_MODES
        )
    )


def summarize_mode_deltas(mode_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"Seed", "Mode", "delta_abs_error"}
    if not required.issubset(mode_rows.columns):
        raise ValueError("Mode delta rows are incomplete.")
    rows = []
    for keys, local in mode_rows.groupby(["Seed", "Mode"], sort=True):
        delta = local.delta_abs_error.to_numpy(dtype=float)
        rows.append({
            "Seed": int(keys[0]),
            "Mode": str(keys[1]),
            "count": int(len(local)),
            "mean_delta_abs_error": float(delta.mean()),
            "median_delta_abs_error": float(np.median(delta)),
            "improved_fraction": float(np.mean(delta < 0)),
            "worsened_fraction": float(np.mean(delta > 0)),
            "unchanged_fraction": float(np.mean(delta == 0)),
        })
    return pd.DataFrame(rows)


def infer_mechanism_flags(
    cross_seed: Mapping,
    shrinkage: pd.DataFrame,
    gradient_delta: pd.DataFrame,
) -> dict:
    std_ratios = shrinkage["prediction_std_ratio_sam_to_baseline"].to_numpy(dtype=float)
    abs_ratios = shrinkage[
        "prediction_abs_mean_ratio_sam_to_baseline"
    ].to_numpy(dtype=float)
    shrinkage_supported = bool(
        np.median(std_ratios) < 0.95 and np.median(abs_ratios) < 0.95
    )
    sample_consistent = bool(
        float(cross_seed["spearman_delta_J_proxy"]) >= 0.30
        and float(cross_seed["same_direction_fraction"]) >= 0.60
    )
    if gradient_delta.empty:
        conflict_consistent = False
    else:
        pivot = gradient_delta.groupby(
            ["Seed", "Pair", "ParameterGroup"], sort=True
        ).conflict_fraction_change_sam_minus_baseline.mean().reset_index()
        directions = []
        for _, local in pivot.groupby(["Pair", "ParameterGroup"], sort=True):
            if set(local.Seed.astype(int)) == set(FORMAL_SEEDS):
                values = local.set_index("Seed").conflict_fraction_change_sam_minus_baseline
                directions.append(float(values.loc[1111]) * float(values.loc[1114]) > 0)
        conflict_consistent = bool(directions and np.mean(directions) >= 0.60)
    return {
        "sample_effect_consistent_across_seeds": sample_consistent,
        "prediction_shrinkage_supported": shrinkage_supported,
        "sam_gradient_conflict_change_consistent": conflict_consistent,
    }
