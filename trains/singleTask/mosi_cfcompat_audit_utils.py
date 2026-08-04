"""Utilities for a frozen MOSI dataset and CFCompatKD mechanism audit."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "mosi_cfcompat_dataset_mechanism_audit_v1"
METHOD = "DLF-MOSI-CFCompat-DatasetMechanismAudit-v1"
OUTPUT_TAG = "mosi_cfcompat_dataset_mechanism_audit_v1"
FORMAL_SEEDS = (1111, 1114)
MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
NEUTRAL_TAU = 0.5
SENTIMENT_BINS = (-3, -2, -1, 0, 1, 2, 3)
SENTIMENT_EDGES = (-3.000001, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.000001)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260804
MIN_GROUP_PER_SEED = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def canonical_id(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "|".join(canonical_id(item) for item in value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_video_segment(value) -> Tuple[str, str]:
    """Parse common MOSI id encodings without assuming one pickle schema."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value):
        video = canonical_id(value[0])
        segment = canonical_id(value[-1]) if len(value) > 1 else ""
        return video, segment
    text = canonical_id(value)
    for delimiter in ("$_$", "|", "[", "::"):
        if delimiter in text:
            left, right = text.split(delimiter, 1)
            return left, right.rstrip("]")
    match = re.match(r"^(.*?)[_-](\d+)$", text)
    if match:
        return match.group(1), match.group(2)
    return text, ""


def segment_order(value, fallback: int) -> float:
    text = canonical_id(value)
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return float(fallback)


def sentiment_bin(labels: Sequence[float]) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    result = pd.cut(
        values,
        bins=list(SENTIMENT_EDGES),
        labels=list(SENTIMENT_BINS),
        include_lowest=True,
        right=False,
    )
    if pd.isna(result).any():
        raise ValueError("Sentiment values fall outside the fixed MOSI range.")
    return result.astype(int).to_numpy()


def polarity_label(labels: Sequence[float]) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    return np.where(values < -NEUTRAL_TAU, "negative", np.where(values > NEUTRAL_TAU, "positive", "neutral"))


def intensity_label(labels: Sequence[float]) -> np.ndarray:
    values = np.abs(np.asarray(labels, dtype=np.float64).reshape(-1))
    return pd.cut(
        values,
        bins=[-np.inf, 0.5, 1.5, 2.5, np.inf],
        labels=["neutral", "weak", "medium", "strong"],
        right=False,
    ).astype(str).to_numpy()


def safe_pearson(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 2 or left.std() == 0.0 or right.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Ranks require finite non-empty values.")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 3:
        return 0.0
    return safe_pearson(average_ranks(left), average_ranks(right))


def jensen_shannon(first: Sequence[float], second: Sequence[float]) -> float:
    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("Jensen-Shannon inputs must be aligned vectors.")
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(left, right):
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def effective_group_count(counts: Sequence[int]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    if len(values) == 0 or values.sum() <= 0:
        return 0.0
    probabilities = values / values.sum()
    return float(1.0 / np.square(probabilities).sum())


def empirical_compatibility(reference_delta: Sequence[float], query_delta: Sequence[float]) -> np.ndarray:
    """Map Valid evaluator shifts through a train-only empirical CDF.

    This is a descriptive Valid proxy, not the training cache value. The open
    interval convention avoids exact zero/one weights for out-of-sample values.
    """
    reference = np.sort(np.asarray(reference_delta, dtype=np.float64).reshape(-1))
    query = np.asarray(query_delta, dtype=np.float64).reshape(-1)
    if len(reference) == 0 or not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("Empirical compatibility requires finite values.")
    lower = np.searchsorted(reference, query, side="left")
    upper = np.searchsorted(reference, query, side="right")
    percentile = (lower + 0.5 * (upper - lower) + 0.5) / float(len(reference) + 1)
    compatibility = 1.0 - percentile
    epsilon = 0.5 / float(len(reference) + 1)
    return np.clip(compatibility, epsilon, 1.0 - epsilon)


def _frame_activity(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("Feature array must contain sample and feature axes.")
    flat = values.reshape(values.shape[0], -1)
    finite_fraction = np.isfinite(flat).mean(axis=1)
    clean = np.where(np.isfinite(values), values, 0.0)
    if clean.ndim == 2:
        frames = clean[:, :, None]
    else:
        frames = clean.reshape(clean.shape[0], clean.shape[1], -1)
    active = np.any(np.abs(frames) > 1e-12, axis=2)
    effective_length = active.sum(axis=1).astype(float)
    zero_fraction = 1.0 - active.mean(axis=1)
    sample_std = flat.std(axis=1)
    return finite_fraction, effective_length, zero_fraction, sample_std


def feature_quality_rows(split: str, modality: str, array: np.ndarray) -> pd.DataFrame:
    values = np.asarray(array)
    finite_fraction, effective_length, zero_fraction, sample_std = _frame_activity(values)
    clean = np.where(np.isfinite(values), values, 0.0).reshape(values.shape[0], -1)
    return pd.DataFrame(
        {
            "Split": split,
            "Modality": modality,
            "sample_index": np.arange(values.shape[0], dtype=int),
            "finite_fraction": finite_fraction,
            "effective_length": effective_length,
            "zero_fraction": zero_fraction,
            "mean_abs_value": np.mean(np.abs(clean), axis=1),
            "sample_std": sample_std,
            "all_zero": (np.max(np.abs(clean), axis=1) <= 1e-12).astype(int),
            "near_constant": (sample_std <= 1e-8).astype(int),
        }
    )


def text_statistics(raw_text: Sequence[object]) -> pd.DataFrame:
    normalized = [" ".join(str(value).strip().lower().split()) for value in raw_text]
    return pd.DataFrame(
        {
            "normalized_text": normalized,
            "char_count": [len(value) for value in normalized],
            "token_count": [len(value.split()) if value else 0 for value in normalized],
        }
    )


def dataset_sample_frame(dataset, split: str) -> pd.DataFrame:
    labels = np.asarray(dataset.labels["M"], dtype=np.float64).reshape(-1)
    rows = []
    text_stats = text_statistics(dataset.raw_text)
    for index in range(len(labels)):
        sample_id = canonical_id(dataset.ids[index])
        video_id, segment_id = parse_video_segment(dataset.ids[index])
        rows.append(
            {
                "Split": split,
                "sample_index": int(index),
                "sample_id": sample_id,
                "video_id": video_id,
                "segment_id": segment_id,
                "segment_order": segment_order(segment_id, index),
                "label": float(labels[index]),
                "sentiment_bin": int(sentiment_bin([labels[index]])[0]),
                "polarity": str(polarity_label([labels[index]])[0]),
                "intensity": str(intensity_label([labels[index]])[0]),
                "normalized_text": text_stats.iloc[index].normalized_text,
                "char_count": int(text_stats.iloc[index].char_count),
                "token_count": int(text_stats.iloc[index].token_count),
            }
        )
    return pd.DataFrame(rows)


def dataset_split_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, local in samples.groupby("Split", sort=True):
        video_counts = local.video_id.astype(str).value_counts()
        bin_counts = local.sentiment_bin.value_counts()
        rows.append(
            {
                "Split": str(split),
                "sample_count": int(len(local)),
                "video_count": int(local.video_id.astype(str).nunique()),
                "effective_video_count": effective_group_count(video_counts.to_numpy()),
                "segments_per_video_mean": float(video_counts.mean()),
                "segments_per_video_max": int(video_counts.max()),
                "top1_video_share": float(video_counts.iloc[:1].sum() / len(local)),
                "top3_video_share": float(video_counts.iloc[:3].sum() / len(local)),
                "label_mean": float(local.label.mean()),
                "label_std": float(local.label.std(ddof=0)),
                "mean_abs_label": float(local.label.abs().mean()),
                "near_neutral_fraction": float((local.label.abs() <= NEUTRAL_TAU).mean()),
                "label_imbalance_ratio": float(bin_counts.max() / max(1, bin_counts.min())),
                "exact_text_duplicate_fraction": float(local.normalized_text.duplicated(keep=False).mean()),
            }
        )
    return pd.DataFrame(rows)


def video_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, video), local in samples.groupby(["Split", "video_id"], sort=True):
        ordered = local.sort_values(["segment_order", "sample_index"], kind="mergesort")
        labels = ordered.label.to_numpy(dtype=float)
        adjacent_left = labels[:-1]
        adjacent_right = labels[1:]
        rows.append(
            {
                "Split": str(split),
                "video_id": str(video),
                "segment_count": int(len(local)),
                "label_mean": float(labels.mean()),
                "label_std": float(labels.std()),
                "label_min": float(labels.min()),
                "label_max": float(labels.max()),
                "adjacent_label_abs_change": float(np.abs(adjacent_right - adjacent_left).mean()) if len(labels) > 1 else 0.0,
                "adjacent_label_correlation": safe_pearson(adjacent_left, adjacent_right) if len(labels) > 2 else 0.0,
                "adjacent_same_polarity_rate": float(
                    np.mean(polarity_label(adjacent_left) == polarity_label(adjacent_right))
                ) if len(labels) > 1 else 1.0,
            }
        )
    return pd.DataFrame(rows)


def split_shift_summary(samples: pd.DataFrame) -> Dict[str, object]:
    train = samples.loc[samples.Split.eq("train")]
    valid = samples.loc[samples.Split.eq("valid")]
    train_counts = train.sentiment_bin.value_counts().reindex(SENTIMENT_BINS, fill_value=0)
    valid_counts = valid.sentiment_bin.value_counts().reindex(SENTIMENT_BINS, fill_value=0)
    train_text = set(train.normalized_text.astype(str)) - {""}
    valid_text = set(valid.normalized_text.astype(str)) - {""}
    return {
        "train_valid_video_overlap_count": int(len(set(train.video_id) & set(valid.video_id))),
        "train_valid_exact_text_overlap_count": int(len(train_text & valid_text)),
        "label_mean_shift_valid_minus_train": float(valid.label.mean() - train.label.mean()),
        "label_std_shift_valid_minus_train": float(valid.label.std(ddof=0) - train.label.std(ddof=0)),
        "label_bin_jensen_shannon": jensen_shannon(train_counts.to_numpy(), valid_counts.to_numpy()),
        "near_neutral_fraction_shift": float(
            (valid.label.abs() <= NEUTRAL_TAU).mean() - (train.label.abs() <= NEUTRAL_TAU).mean()
        ),
    }


def prediction_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Seed", "Mode", "sample_index", "video_id", "label",
        "baseline_prediction", "cfcompat_prediction", "teacher_prediction",
        "evaluator_shift", "compatibility_proxy",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Prediction table lacks columns: {}".format(sorted(missing)))
    local = frame.copy()
    local["baseline_error"] = np.abs(local.baseline_prediction - local.label)
    local["cfcompat_error"] = np.abs(local.cfcompat_prediction - local.label)
    local["teacher_error"] = np.abs(local.teacher_prediction - local.label)
    local["cfcompat_gain"] = local.baseline_error - local.cfcompat_error
    local["teacher_advantage"] = local.baseline_error - local.teacher_error
    local["teacher_better"] = (local.teacher_advantage > 0.0).astype(int)
    local["direction_correct"] = (
        (local.teacher_prediction - local.baseline_prediction)
        * (local.label - local.baseline_prediction)
        > 0.0
    ).astype(int)
    local["teacher_approach"] = (
        np.abs(local.baseline_prediction - local.teacher_prediction)
        - np.abs(local.cfcompat_prediction - local.teacher_prediction)
    )
    toward = local.teacher_approach > 0.0
    improved = local.cfcompat_gain > 0.0
    local["transfer_quadrant"] = np.select(
        [toward & improved, toward & ~improved, ~toward & improved, ~toward & ~improved],
        ["Q1_helpful_imitation", "Q2_harmful_imitation", "Q3_improve_without_imitation", "Q4_double_failure"],
        default="boundary",
    )
    local["sentiment_bin"] = sentiment_bin(local.label)
    local["polarity"] = polarity_label(local.label)
    local["intensity"] = intensity_label(local.label)
    local["baseline_error_quartile"] = (
        local.groupby(["Seed", "Mode"], sort=False).baseline_error.transform(
            lambda values: pd.qcut(values.rank(method="first"), 4, labels=["Q1_easy", "Q2", "Q3", "Q4_hard"])
        ).astype(str)
    )
    local["compatibility_decile"] = (
        local.groupby(["Seed", "Mode"], sort=False).compatibility_proxy.transform(
            lambda values: pd.qcut(values.rank(method="first"), 10, labels=False) + 1
        ).astype(int)
    )
    local["teacher_condition"] = np.select(
        [
            (local.teacher_better == 1) & (local.direction_correct == 1),
            (local.teacher_better == 1) & (local.direction_correct == 0),
            (local.teacher_better == 0) & (local.direction_correct == 1),
        ],
        ["better_and_correct", "better_wrong_direction", "not_better_but_correct"],
        default="not_better_wrong_direction",
    )
    return local


def j_from_events(events: pd.DataFrame, prediction: str) -> float:
    maes = {
        mode: float(np.abs(events.loc[events.Mode.eq(mode), prediction] - events.loc[events.Mode.eq(mode), "label"]).mean())
        for mode in MODES
    }
    return float(0.5 * maes["LAV"] + 0.5 * np.mean([maes[mode] for mode in MISSING_MODES]))


def overall_prediction_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, seed_frame in events.groupby("Seed", sort=True):
        mode_values = {}
        for mode, local in seed_frame.groupby("Mode", sort=True):
            row = {
                "Seed": int(seed),
                "Mode": str(mode),
                "N": int(len(local)),
                "baseline_MAE": float(local.baseline_error.mean()),
                "cfcompat_MAE": float(local.cfcompat_error.mean()),
                "cfcompat_gain": float(local.cfcompat_gain.mean()),
                "win_rate": float((local.cfcompat_gain > 0).mean()),
                "teacher_better_rate": float(local.teacher_better.mean()),
                "direction_correct_rate": float(local.direction_correct.mean()),
                "helpful_imitation_rate": float((local.transfer_quadrant == "Q1_helpful_imitation").mean()),
                "harmful_imitation_rate": float((local.transfer_quadrant == "Q2_harmful_imitation").mean()),
                "mean_compatibility": float(local.compatibility_proxy.mean()),
            }
            rows.append(row)
            mode_values[str(mode)] = row
        rows.append(
            {
                "Seed": int(seed),
                "Mode": "J",
                "N": int(len(seed_frame) // len(MODES)),
                "baseline_MAE": j_from_events(seed_frame, "baseline_prediction"),
                "cfcompat_MAE": j_from_events(seed_frame, "cfcompat_prediction"),
                "cfcompat_gain": j_from_events(seed_frame, "baseline_prediction") - j_from_events(seed_frame, "cfcompat_prediction"),
                "win_rate": float((seed_frame.cfcompat_gain > 0).mean()),
                "teacher_better_rate": float(seed_frame.teacher_better.mean()),
                "direction_correct_rate": float(seed_frame.direction_correct.mean()),
                "helpful_imitation_rate": float((seed_frame.transfer_quadrant == "Q1_helpful_imitation").mean()),
                "harmful_imitation_rate": float((seed_frame.transfer_quadrant == "Q2_harmful_imitation").mean()),
                "mean_compatibility": float(seed_frame.compatibility_proxy.mean()),
            }
        )
    return pd.DataFrame(rows)


def _group_row(frame: pd.DataFrame, group_type: str, group_value: str, seed) -> Dict[str, object]:
    return {
        "Seed": seed,
        "GroupType": group_type,
        "GroupValue": str(group_value),
        "N": int(len(frame)),
        "video_count": int(frame.video_id.astype(str).nunique()),
        "baseline_MAE": float(frame.baseline_error.mean()),
        "cfcompat_MAE": float(frame.cfcompat_error.mean()),
        "cfcompat_gain": float(frame.cfcompat_gain.mean()),
        "win_rate": float((frame.cfcompat_gain > 0).mean()),
        "teacher_MAE": float(frame.teacher_error.mean()),
        "teacher_advantage": float(frame.teacher_advantage.mean()),
        "teacher_better_rate": float(frame.teacher_better.mean()),
        "direction_correct_rate": float(frame.direction_correct.mean()),
        "teacher_approach_mean": float(frame.teacher_approach.mean()),
        "helpful_imitation_rate": float((frame.transfer_quadrant == "Q1_helpful_imitation").mean()),
        "harmful_imitation_rate": float((frame.transfer_quadrant == "Q2_harmful_imitation").mean()),
        "mean_compatibility": float(frame.compatibility_proxy.mean()),
        "mean_evaluator_shift": float(frame.evaluator_shift.mean()),
    }


def group_mechanism_summary(events: pd.DataFrame) -> pd.DataFrame:
    group_columns = (
        ("mode", "Mode"),
        ("sentiment_bin", "sentiment_bin"),
        ("polarity", "polarity"),
        ("intensity", "intensity"),
        ("compatibility_decile", "compatibility_decile"),
        ("baseline_error_quartile", "baseline_error_quartile"),
        ("teacher_condition", "teacher_condition"),
        ("video", "video_id"),
    )
    rows = []
    for seed, seed_frame in events.groupby("Seed", sort=True):
        rows.append(_group_row(seed_frame, "all", "ALL", int(seed)))
        for group_type, column in group_columns:
            for value, local in seed_frame.groupby(column, sort=True):
                rows.append(_group_row(local, group_type, value, int(seed)))
    rows.append(_group_row(events, "all", "ALL", "POOLED"))
    for group_type, column in group_columns:
        for value, local in events.groupby(column, sort=True):
            rows.append(_group_row(local, group_type, value, "POOLED"))
    return pd.DataFrame(rows)


def modality_marginal_value(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, seed_frame in events.groupby("Seed", sort=True):
        for method, column in (("DLF_ModDrop", "baseline_prediction"), ("CFCompatKD", "cfcompat_prediction")):
            wide = seed_frame.pivot(index=["sample_index", "video_id", "label"], columns="Mode", values=column).reset_index()
            for name, richer, poorer in (
                ("audio_given_text", "LA", "L"),
                ("vision_given_text", "LV", "L"),
                ("audio_given_text_vision", "LAV", "LV"),
                ("vision_given_text_audio", "LAV", "LA"),
                ("all_nontext_given_text", "LAV", "L"),
            ):
                gain = np.abs(wide[poorer] - wide.label) - np.abs(wide[richer] - wide.label)
                rows.append(
                    {
                        "Seed": int(seed),
                        "Method": method,
                        "Comparison": name,
                        "N": int(len(wide)),
                        "mean_MAE_gain": float(gain.mean()),
                        "median_MAE_gain": float(np.median(gain)),
                        "improvement_rate": float((gain > 0).mean()),
                        "degradation_rate": float((gain < 0).mean()),
                        "gain_std": float(gain.std()),
                    }
                )
    return pd.DataFrame(rows)


def joint_video_bootstrap(events: pd.DataFrame, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    if int(replicates) < 100:
        raise ValueError("Video bootstrap requires at least 100 replicates.")
    videos = sorted(events.video_id.astype(str).unique())
    if len(videos) < 2:
        raise RuntimeError("Video bootstrap requires at least two videos.")
    generator = np.random.default_rng(int(seed))
    rows = []
    for replicate in range(int(replicates)):
        sampled = generator.choice(videos, size=len(videos), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        local = events.copy()
        local["bootstrap_weight"] = local.video_id.astype(str).map(lambda value: int(counts.get(value, 0)))
        local = local.loc[local.bootstrap_weight > 0]
        seed_gains = []
        for seed_value, seed_frame in local.groupby("Seed", sort=True):
            mode_maes = {}
            for mode, mode_frame in seed_frame.groupby("Mode", sort=True):
                weight = mode_frame.bootstrap_weight.to_numpy(dtype=float)
                baseline = np.average(mode_frame.baseline_error, weights=weight)
                cfcompat = np.average(mode_frame.cfcompat_error, weights=weight)
                mode_maes[str(mode)] = float(baseline - cfcompat)
            if set(mode_maes) == set(MODES):
                seed_gains.append(0.5 * mode_maes["LAV"] + 0.5 * np.mean([mode_maes[mode] for mode in MISSING_MODES]))
        if len(seed_gains) != len(FORMAL_SEEDS):
            raise RuntimeError("Bootstrap lost a formal seed or mode.")
        high = local.loc[local.compatibility_decile >= 8]
        low = local.loc[local.compatibility_decile <= 3]
        rows.append(
            {
                "Replicate": int(replicate),
                "mean_J_gain": float(np.mean(seed_gains)),
                "high_compat_gain": float(np.average(high.cfcompat_gain, weights=high.bootstrap_weight)),
                "low_compat_gain": float(np.average(low.cfcompat_gain, weights=low.bootstrap_weight)),
                "high_minus_low_gain": float(
                    np.average(high.cfcompat_gain, weights=high.bootstrap_weight)
                    - np.average(low.cfcompat_gain, weights=low.bootstrap_weight)
                ),
                "unique_videos": int(len(counts)),
            }
        )
    return pd.DataFrame(rows)


def interval(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Interval requires finite values.")
    low, high = np.quantile(array, [0.025, 0.975])
    return {"mean": float(array.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def opportunity_ranking(groups: pd.DataFrame) -> pd.DataFrame:
    candidates = groups.loc[
        groups.GroupType.isin(
            ["mode", "sentiment_bin", "polarity", "intensity", "compatibility_decile", "baseline_error_quartile", "teacher_condition", "video"]
        )
        & ~groups.Seed.astype(str).eq("POOLED")
    ].copy()
    rows = []
    for (group_type, group_value), local in candidates.groupby(["GroupType", "GroupValue"], sort=True):
        if set(local.Seed.astype(int)) != set(FORMAL_SEEDS):
            continue
        if int(local.N.min()) < MIN_GROUP_PER_SEED:
            continue
        min_gain = float(local.cfcompat_gain.min())
        mean_gain = float(local.cfcompat_gain.mean())
        remaining = float(local.cfcompat_MAE.mean())
        teacher_advantage = float(local.teacher_advantage.mean())
        direction = float(local.direction_correct_rate.mean())
        compatibility = float(local.mean_compatibility.mean())
        recoverable_score = max(0.0, teacher_advantage) * direction * compatibility
        rows.append(
            {
                "GroupType": group_type,
                "GroupValue": str(group_value),
                "min_seed_N": int(local.N.min()),
                "mean_cfcompat_gain": mean_gain,
                "min_seed_cfcompat_gain": min_gain,
                "both_seeds_positive_gain": bool((local.cfcompat_gain > 0).all()),
                "mean_remaining_CFCompat_MAE": remaining,
                "mean_teacher_advantage": teacher_advantage,
                "mean_direction_correct_rate": direction,
                "mean_compatibility": compatibility,
                "recoverable_opportunity_score": recoverable_score,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["both_seeds_positive_gain", "min_seed_cfcompat_gain", "recoverable_opportunity_score", "min_seed_N"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def dataset_limitation_flags(
    split_summary: pd.DataFrame,
    shift: Mapping[str, object],
    modality_value: pd.DataFrame,
    overall: pd.DataFrame,
) -> Dict[str, object]:
    train = split_summary.loc[split_summary.Split.eq("train")].iloc[0]
    valid = split_summary.loc[split_summary.Split.eq("valid")].iloc[0]
    baseline_j = overall.loc[overall.Mode.eq("J")].baseline_MAE.astype(float)
    nontext = modality_value.loc[
        modality_value.Method.eq("DLF_ModDrop")
        & modality_value.Comparison.eq("all_nontext_given_text")
    ]
    return {
        "small_valid_video_support": bool(valid.video_count < 20),
        "valid_effective_video_count": float(valid.effective_video_count),
        "valid_top3_video_share": float(valid.top3_video_share),
        "train_label_long_tail": bool(train.label_imbalance_ratio >= 3.0),
        "train_near_neutral_concentration": float(train.near_neutral_fraction),
        "train_valid_label_shift_nontrivial": bool(
            abs(float(shift["label_mean_shift_valid_minus_train"])) >= 0.1
            or float(shift["label_bin_jensen_shannon"]) >= 0.03
        ),
        "text_dominance_indicator": bool(
            len(nontext) == len(FORMAL_SEEDS)
            and float(nontext.mean_MAE_gain.mean()) <= 0.03
        ),
        "nontext_help_mean_MAE_gain": float(nontext.mean_MAE_gain.mean()) if len(nontext) else None,
        "baseline_seed_J_gap": float(baseline_j.max() - baseline_j.min()) if len(baseline_j) else None,
        "seed_instability_indicator": bool(len(baseline_j) and baseline_j.max() - baseline_j.min() >= 0.005),
    }


def mechanism_assessment(events: pd.DataFrame, overall: pd.DataFrame, bootstrap: pd.DataFrame) -> Dict[str, object]:
    j_rows = overall.loc[overall.Mode.eq("J")].sort_values("Seed")
    both_positive = bool(len(j_rows) == len(FORMAL_SEEDS) and (j_rows.cfcompat_gain > 0).all())
    pooled = group_mechanism_summary(events)
    condition = pooled.loc[
        pooled.Seed.astype(str).eq("POOLED")
        & pooled.GroupType.eq("teacher_condition")
    ].set_index("GroupValue")
    good_gain = float(condition.loc["better_and_correct", "cfcompat_gain"]) if "better_and_correct" in condition.index else float("nan")
    bad_values = condition.loc[condition.index != "better_and_correct", "cfcompat_gain"]
    bad_gain = float(bad_values.mean()) if len(bad_values) else float("nan")
    high = events.loc[events.compatibility_decile >= 8]
    low = events.loc[events.compatibility_decile <= 3]
    compatibility_gain_difference = float(high.cfcompat_gain.mean() - low.cfcompat_gain.mean())
    compatibility_harm_difference = float(
        (high.transfer_quadrant == "Q2_harmful_imitation").mean()
        - (low.transfer_quadrant == "Q2_harmful_imitation").mean()
    )
    boot = interval(bootstrap.mean_J_gain)
    checks = {
        "cfcompat_J_gain_positive_both_seeds": both_positive,
        "joint_video_bootstrap_J_gain_ci_low_positive": bool(boot["ci95_low"] > 0.0),
        "teacher_better_correct_subset_gain_exceeds_other_conditions": bool(
            math.isfinite(good_gain) and math.isfinite(bad_gain) and good_gain > bad_gain
        ),
        "high_compatibility_gain_exceeds_low_compatibility": bool(compatibility_gain_difference > 0.0),
        "high_compatibility_not_more_harmful": bool(compatibility_harm_difference <= 0.0),
    }
    if not both_positive:
        verdict = "CFCompat_GAIN_NOT_REPRODUCED"
    elif all(checks.values()):
        verdict = "CFCompat_IMPROVEMENT_MECHANISM_SUPPORTED"
    else:
        verdict = "CFCompat_GAIN_REPRODUCED_MECHANISM_PARTIAL"
    return {
        "verdict": verdict,
        "checks": checks,
        "mean_two_seed_J_gain": float(j_rows.cfcompat_gain.mean()),
        "per_seed_J_gain": {
            str(int(row.Seed)): float(row.cfcompat_gain)
            for row in j_rows.itertuples(index=False)
        },
        "video_bootstrap_J_gain": boot,
        "better_and_correct_gain": good_gain,
        "other_teacher_conditions_mean_gain": bad_gain,
        "high_minus_low_compatibility_gain": compatibility_gain_difference,
        "high_minus_low_harmful_imitation_rate": compatibility_harm_difference,
    }
