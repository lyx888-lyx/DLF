"""Shared deterministic utilities for the Stage 21A source-relative audit.

This module deliberately has no Test entry point.  Source IDs are used only to
construct audit splits and relative rows; they are never model/probe features.
"""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.missing_utils import regression_metrics
from trains.utils.metricsTop import MetricsTop


ALLOWED_SPLITS = ("train", "valid")
MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
DELTA_CANDIDATES = (1.0, 0.75, 0.5)
RIDGE_ALPHA = 1e-3
RELATIVE_MASS = 0.5


def require_allowed_split(split):
    if split not in ALLOWED_SPLITS:
        raise RuntimeError(
            "Stage 21A permits only Official Train and Official Valid; "
            "split=test is hard locked."
        )
    return split


def canonical_id(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    return str(value)


def canonical_ids(values):
    return [canonical_id(value) for value in values]


def parse_sample_id(sample_id):
    sample_id = canonical_id(sample_id)
    parts = sample_id.split("$_$")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise RuntimeError("Malformed MOSEI sample ID: {}".format(sample_id))
    return parts[0], parts[1]


def assert_unique_complete_ids(ids):
    canonical = canonical_ids(ids)
    if len(canonical) != len(set(canonical)):
        raise RuntimeError("Duplicate sample ID hard failure.")
    parsed = [parse_sample_id(sample_id) for sample_id in canonical]
    if any(not video_id for video_id, _ in parsed):
        raise RuntimeError("Missing video_id hard failure.")
    if any(not clip_id for _, clip_id in parsed):
        raise RuntimeError("Missing clip_id hard failure.")
    return parsed


def assert_order_sha(ids, expected):
    actual = ordered_id_sha(ids)
    if actual != str(expected):
        raise RuntimeError("Sample-order SHA mismatch hard failure.")
    return actual


def ordered_id_sha(ids):
    return hashlib.sha256("\n".join(canonical_ids(ids)).encode("utf-8")).hexdigest()


def unordered_id_sha(ids):
    return ordered_id_sha(sorted(canonical_ids(ids)))


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_frame(path, rows, sep="\t"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, sep=sep, index=False)
    os.replace(str(temporary), str(path))


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def group_indices(video_ids, indices=None):
    if indices is None:
        indices = np.arange(len(video_ids), dtype=np.int64)
    groups = defaultdict(list)
    for index in np.asarray(indices, dtype=np.int64):
        groups[str(video_ids[index])].append(int(index))
    return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}


def label_histogram(labels):
    edges = np.asarray([-3, -2, -1, 0, 1, 2, 3.000001], dtype=np.float64)
    counts, _ = np.histogram(np.asarray(labels, dtype=np.float64), bins=edges)
    return {
        "edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
    }


def size_summary(sizes):
    sizes = np.asarray(sizes, dtype=np.float64)
    quantiles = np.quantile(sizes, [0.5, 0.75, 0.9])
    return {
        "min": int(sizes.min()),
        "median": float(quantiles[0]),
        "mean": float(sizes.mean()),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "max": int(sizes.max()),
    }


def pair_candidates_for_group(indices, labels, delta):
    indices = np.asarray(indices, dtype=np.int64)
    result = []
    for left_position in range(len(indices)):
        left = int(indices[left_position])
        for right_position in range(left_position + 1, len(indices)):
            right = int(indices[right_position])
            gap = float(labels[left] - labels[right])
            if abs(gap) >= float(delta):
                result.append((left, right, gap))
    return result


def coverage_for_delta(video_ids, labels, delta):
    groups = group_indices(video_ids)
    covered = set()
    eligible_videos = 0
    total_pairs = 0
    multi = sum(len(indices) >= 2 for indices in groups.values())
    for indices in groups.values():
        pairs = pair_candidates_for_group(indices, labels, delta)
        if pairs:
            eligible_videos += 1
            total_pairs += len(pairs)
            for left, right, _ in pairs:
                covered.add(left)
                covered.add(right)
    return {
        "delta": float(delta),
        "sample_coverage_count": len(covered),
        "sample_coverage_fraction": float(len(covered) / len(video_ids)),
        "eligible_video_count": int(eligible_videos),
        "multi_clip_video_count": int(multi),
        "eligible_video_fraction": float(eligible_videos / max(multi, 1)),
        "raw_pair_count": int(total_pairs),
        "positive_pair_exists": bool(total_pairs),
        "negative_pair_exists": bool(total_pairs),
        "passed": bool(
            len(covered) / len(video_ids) >= 0.60
            and eligible_videos / max(multi, 1) >= 0.50
            and total_pairs > 0
        ),
    }


def select_delta(video_ids, labels):
    rows = [coverage_for_delta(video_ids, labels, delta) for delta in DELTA_CANDIDATES]
    selected = next((row["delta"] for row in rows if row["passed"]), None)
    return selected, rows


def balanced_within_pairs(video_ids, labels, delta, seed=2100, max_pairs=64, indices=None):
    """Select at most 64 deterministic pairs per real source.

    Each returned pair has one of two deterministic orientations.  Per-video
    raw weights sum to one, preventing long sources from dominating.
    """
    rng = np.random.RandomState(int(seed))
    groups = group_indices(video_ids, indices)
    rows = []
    for video_id in sorted(groups):
        candidates = pair_candidates_for_group(groups[video_id], labels, delta)
        if not candidates:
            continue
        # Round-robin over fixed absolute-gap bins before random tie breaking.
        bins = defaultdict(list)
        for pair in candidates:
            gap_bin = int(min(abs(pair[2]) // 0.5, 12))
            bins[gap_bin].append(pair)
        for values in bins.values():
            rng.shuffle(values)
        selected = []
        while len(selected) < min(int(max_pairs), len(candidates)):
            changed = False
            for gap_bin in sorted(bins, reverse=True):
                if bins[gap_bin]:
                    selected.append(bins[gap_bin].pop())
                    changed = True
                    if len(selected) >= min(int(max_pairs), len(candidates)):
                        break
            if not changed:
                break
        per_pair_weight = 1.0 / len(selected)
        for position, (left, right, gap) in enumerate(selected):
            desired_positive = (position % 2) == 0
            if (gap > 0) != desired_positive:
                left, right, gap = right, left, -gap
            rows.append(
                {
                    "video_id": video_id,
                    "left_index": int(left),
                    "right_index": int(right),
                    "target_difference": float(labels[left] - labels[right]),
                    "absolute_gap": float(abs(labels[left] - labels[right])),
                    "raw_video_balanced_weight": float(per_pair_weight),
                }
            )
    return rows


def pseudo_membership(video_ids, indices, seed):
    groups = group_indices(video_ids, indices)
    sizes = [len(groups[key]) for key in sorted(groups)]
    rng = np.random.RandomState(int(seed))
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    pseudo = np.empty(len(video_ids), dtype=object)
    cursor = 0
    for position, size in enumerate(sizes):
        chosen = shuffled[cursor : cursor + size]
        pseudo[chosen] = "pseudo_{:05d}".format(position)
        cursor += size
    if cursor != len(shuffled):
        raise AssertionError("Pseudo membership did not consume every sample.")
    if sorted(sizes) != sorted(len(value) for value in group_indices(pseudo, indices).values()):
        raise AssertionError("Pseudo group-size multiset changed.")
    return pseudo


def nearest_cross_source_pairs(target_rows, indices, video_ids, labels, seed):
    """Greedily match each target signed gap with a cross-source pair."""
    rng = np.random.RandomState(int(seed))
    pool = np.asarray(indices, dtype=np.int64)
    sorted_pool = pool[np.argsort(labels[pool], kind="mergesort")]
    sorted_labels = labels[sorted_pool]
    result = []
    for row in target_rows:
        target = float(row["target_difference"])
        best = None
        # Multiple deterministic random anchors make exact discrete-label matches common.
        anchors = pool[rng.randint(0, len(pool), size=min(32, len(pool)))]
        for left in anchors:
            desired = float(labels[left] - target)
            insertion = int(np.searchsorted(sorted_labels, desired))
            for offset in range(-4, 5):
                position = insertion + offset
                if position < 0 or position >= len(sorted_pool):
                    continue
                right = int(sorted_pool[position])
                if right == int(left) or video_ids[right] == video_ids[int(left)]:
                    continue
                actual = float(labels[int(left)] - labels[right])
                error = abs(actual - target)
                candidate = (error, int(left), right, actual)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise RuntimeError("Unable to construct a cross-source control pair.")
        _, left, right, actual = best
        result.append(
            {
                "video_id": "cross_source",
                "left_index": left,
                "right_index": right,
                "target_difference": actual,
                "absolute_gap": abs(actual),
                "raw_video_balanced_weight": float(row["raw_video_balanced_weight"]),
            }
        )
    return result


def gap_match_from_candidates(target_rows, candidate_rows):
    if not candidate_rows:
        raise RuntimeError("Pseudo-source candidates are empty.")
    positive = sorted(
        [row for row in candidate_rows if row["target_difference"] > 0],
        key=lambda row: row["target_difference"],
    )
    negative = sorted(
        [row for row in candidate_rows if row["target_difference"] < 0],
        key=lambda row: row["target_difference"],
    )
    result = []
    for target_row in target_rows:
        target = float(target_row["target_difference"])
        pool = positive if target > 0 else negative
        if not pool:
            raise RuntimeError("Pseudo-source control lacks one signed direction.")
        values = np.asarray([row["target_difference"] for row in pool])
        position = int(np.searchsorted(values, target))
        positions = [max(0, min(len(pool) - 1, position + offset)) for offset in (-1, 0)]
        chosen = min((pool[p] for p in positions), key=lambda row: abs(row["target_difference"] - target))
        copied = dict(chosen)
        copied["raw_video_balanced_weight"] = float(target_row["raw_video_balanced_weight"])
        result.append(copied)
    return result


def pair_matching_diagnostics(first, second):
    from scipy.stats import ks_2samp, wasserstein_distance

    left = np.asarray([row["absolute_gap"] for row in first], dtype=np.float64)
    right = np.asarray([row["absolute_gap"] for row in second], dtype=np.float64)
    quantiles = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "pair_count_first": int(len(left)),
        "pair_count_second": int(len(right)),
        "positive_fraction_first": float(np.mean([row["target_difference"] > 0 for row in first])),
        "positive_fraction_second": float(np.mean([row["target_difference"] > 0 for row in second])),
        "ks_statistic": float(ks_2samp(left, right).statistic),
        "wasserstein_distance": float(wasserstein_distance(left, right)),
        "quantiles_first": np.quantile(left, quantiles).tolist(),
        "quantiles_second": np.quantile(right, quantiles).tolist(),
        "maximum_quantile_difference": float(
            np.max(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles)))
        ),
    }


def one_way_icc_from_groups(groups):
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in groups if len(value)]
    count = sum(len(value) for value in values)
    group_count = len(values)
    if group_count < 2 or count <= group_count:
        return {
            "icc": 0.0,
            "between_video_variance": 0.0,
            "within_video_variance": 0.0,
            "ms_between": 0.0,
            "ms_within": 0.0,
            "effective_group_size": 0.0,
        }
    sizes = np.asarray([len(value) for value in values], dtype=np.float64)
    means = np.asarray([value.mean() for value in values], dtype=np.float64)
    grand = float(np.sum(sizes * means) / count)
    ss_between = float(np.sum(sizes * np.square(means - grand)))
    ss_within = float(sum(np.square(value - value.mean()).sum() for value in values))
    ms_between = ss_between / (group_count - 1)
    ms_within = ss_within / (count - group_count)
    n0 = (count - float(np.square(sizes).sum()) / count) / (group_count - 1)
    denominator = ms_between + (n0 - 1.0) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator else 0.0
    between = (ms_between - ms_within) / n0 if n0 else 0.0
    return {
        "icc": float(icc),
        "between_video_variance": float(between),
        "within_video_variance": float(ms_within),
        "ms_between": float(ms_between),
        "ms_within": float(ms_within),
        "effective_group_size": float(n0),
    }


def residual_structure(video_ids, labels, predictions):
    residuals = np.asarray(labels, dtype=np.float64) - np.asarray(predictions, dtype=np.float64)
    groups = group_indices(video_ids)
    group_values = [residuals[groups[key]] for key in sorted(groups)]
    result = one_way_icc_from_groups(group_values)
    source_rows = []
    for video_id in sorted(groups):
        index = groups[video_id]
        source_rows.append(
            {
                "video_id": video_id,
                "clip_count": int(len(index)),
                "mean_residual": float(residuals[index].mean()),
                "mean_absolute_residual": float(np.abs(residuals[index]).mean()),
                "label_mean": float(np.asarray(labels)[index].mean()),
            }
        )
    means = np.asarray([row["mean_residual"] for row in source_rows])
    sizes = np.asarray([row["clip_count"] for row in source_rows])
    label_means = np.asarray([row["label_mean"] for row in source_rows])
    result.update(
        {
            "source_mean_residual_std": float(means.std()),
            "source_mean_mae": float(np.mean([row["mean_absolute_residual"] for row in source_rows])),
            "source_mean_signed_error": float(means.mean()),
            "top_worst_sources": sorted(source_rows, key=lambda row: row["mean_absolute_residual"], reverse=True)[:10],
            "corr_source_size_residual_mean": safe_corr(sizes, means),
            "corr_label_mean_residual_mean": safe_corr(label_means, means),
        }
    )
    return result, source_rows, residuals


def safe_corr(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def icc_permutation_and_bootstrap(video_ids, residuals, permutations=2000, seed=2103):
    groups = group_indices(video_ids)
    ordered_keys = sorted(groups)
    sizes = np.asarray([len(groups[key]) for key in ordered_keys], dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    permuted = []
    for _ in range(int(permutations)):
        shuffled = np.asarray(residuals, dtype=np.float64)[rng.permutation(len(residuals))]
        cursor = 0
        values = []
        for size in sizes:
            values.append(shuffled[cursor : cursor + size])
            cursor += int(size)
        permuted.append(one_way_icc_from_groups(values)["icc"])
    original_values = [np.asarray(residuals)[groups[key]] for key in ordered_keys]
    bootstrap = []
    for _ in range(int(permutations)):
        chosen = rng.randint(0, len(original_values), size=len(original_values))
        bootstrap.append(one_way_icc_from_groups([original_values[index] for index in chosen])["icc"])
    return np.asarray(permuted), np.asarray(bootstrap)


def make_source_split(video_ids, labels, seed, valid_fraction=0.2):
    """One deterministic, label-stratified source split; no result search."""
    from sklearn.model_selection import StratifiedShuffleSplit

    groups = group_indices(video_ids)
    keys = np.asarray(sorted(groups), dtype=object)
    means = np.asarray([np.asarray(labels)[groups[key]].mean() for key in keys])
    sizes = np.asarray([len(groups[key]) for key in keys])
    mean_bin = np.digitize(means, [-1.5, -0.5, 0.5, 1.5])
    size_bin = np.digitize(sizes, [3, 6, 10, 20])
    strata = np.asarray(["{}_{}".format(a, b) for a, b in zip(mean_bin, size_bin)])
    # Collapse sparse joint strata deterministically to label-only strata.
    counts = {key: int(np.sum(strata == key)) for key in set(strata)}
    strata = np.asarray([key if counts[key] >= 2 else "label_{}".format(a) for key, a in zip(strata, mean_bin)])
    split = StratifiedShuffleSplit(n_splits=1, test_size=float(valid_fraction), random_state=int(seed))
    train_position, valid_position = next(split.split(np.zeros(len(keys)), strata))
    train_sources = set(keys[train_position].tolist())
    valid_sources = set(keys[valid_position].tolist())
    if train_sources & valid_sources:
        raise AssertionError("Source-disjoint split overlap.")
    train_index = np.asarray([i for i, value in enumerate(video_ids) if value in train_sources], dtype=np.int64)
    valid_index = np.asarray([i for i, value in enumerate(video_ids) if value in valid_sources], dtype=np.int64)
    return {
        "seed": int(seed),
        "inner_train_sources": sorted(train_sources),
        "inner_valid_sources": sorted(valid_sources),
        "inner_train_indices": train_index,
        "inner_valid_indices": valid_index,
    }


def standardization(representations, labels, train_indices):
    stacked = np.concatenate(
        [np.asarray(representations[mode])[train_indices] for mode in MODES], axis=0
    ).astype(np.float64)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-8] = 1.0
    label_values = np.asarray(labels, dtype=np.float64)[train_indices]
    label_mean = float(label_values.mean())
    label_std = float(label_values.std())
    if label_std < 1e-8:
        raise RuntimeError("Inner-train labels have zero variance.")
    return mean, std, label_mean, label_std


def fit_shared_ridge(representations, labels, train_indices, pairs, standard, alpha=RIDGE_ALPHA):
    feature_mean, feature_std, label_mean, label_std = standard
    dimension = len(feature_mean)
    xtx = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
    xty = np.zeros(dimension + 1, dtype=np.float64)
    absolute_total = 0.0
    for mode in MODES:
        features = (np.asarray(representations[mode])[train_indices] - feature_mean) / feature_std
        target = (np.asarray(labels)[train_indices] - label_mean) / label_std
        augmented = np.concatenate([features, np.ones((len(features), 1))], axis=1)
        xtx += augmented.T @ augmented
        xty += augmented.T @ target
        absolute_total += len(features)
    if pairs:
        raw = np.asarray([row["raw_video_balanced_weight"] for row in pairs], dtype=np.float64)
        if np.any(raw <= 0):
            raise RuntimeError("Relative pair weights must be positive.")
        per_mode_scale = RELATIVE_MASS * absolute_total / (len(MODES) * raw.sum())
        weights = raw * per_mode_scale
        left = np.asarray([row["left_index"] for row in pairs], dtype=np.int64)
        right = np.asarray([row["right_index"] for row in pairs], dtype=np.int64)
        target = (np.asarray(labels)[left] - np.asarray(labels)[right]) / label_std
        for mode in MODES:
            difference = (
                np.asarray(representations[mode])[left]
                - np.asarray(representations[mode])[right]
            ) / feature_std
            augmented = np.concatenate([difference, np.zeros((len(difference), 1))], axis=1)
            weighted = augmented * weights[:, None]
            xtx += augmented.T @ weighted
            xty += augmented.T @ (weights * target)
    regularizer = np.eye(dimension + 1, dtype=np.float64) * float(alpha)
    regularizer[-1, -1] = 0.0
    coefficient = np.linalg.solve(xtx + regularizer, xty)
    return {
        "coefficient": coefficient,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "label_mean": label_mean,
        "label_std": label_std,
        "ridge_alpha": float(alpha),
        "relative_mass": float(RELATIVE_MASS if pairs else 0.0),
        "source_features_in_design": False,
        "relative_bias_column_max": 0.0,
    }


def ridge_predict(model, representation):
    features = (np.asarray(representation, dtype=np.float64) - model["feature_mean"]) / model["feature_std"]
    prediction_standard = features @ model["coefficient"][:-1] + model["coefficient"][-1]
    return prediction_standard * model["label_std"] + model["label_mean"]


def project_metrics(prediction, labels):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    project = MetricsTop("regression").getMetics("MOSEI")(
        torch.as_tensor(prediction, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    precise = regression_metrics(
        torch.as_tensor(prediction, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    parity = max(abs(float(project[key]) - float(precise[key])) for key in project)
    precise.update(
        {
            "PredictionStd": float(prediction.std()),
            "PredictionMin": float(prediction.min()),
            "PredictionMax": float(prediction.max()),
            "ProjectEvaluatorRoundingParityMax": float(parity),
        }
    )
    return precise


def evaluate_modes(predictions, labels, indices):
    indices = np.asarray(indices, dtype=np.int64)
    result = {
        mode: project_metrics(np.asarray(predictions[mode])[indices], np.asarray(labels)[indices])
        for mode in MODES
    }
    result["MissingMacro"] = {
        metric: float(np.mean([result[mode][metric] for mode in MISSING_MODES]))
        for metric in result["LA"]
    }
    result["J"] = float(
        0.5 * result["LAV"]["MAE"] + 0.5 * result["MissingMacro"]["MAE"]
    )
    return result


def metric_delta(left, right):
    result = {"J": float(left["J"] - right["J"])}
    for mode in MODES + ("MissingMacro",):
        result[mode] = {
            key: float(left[mode][key] - right[mode][key])
            for key in left[mode]
            if key in right[mode]
        }
    return result


def bootstrap_j_delta(prediction_a, prediction_b, labels, video_ids, indices, iterations=2000, seed=2104):
    indices = np.asarray(indices, dtype=np.int64)
    groups = group_indices(video_ids, indices)
    keys = sorted(groups)
    count = np.asarray([len(groups[key]) for key in keys], dtype=np.float64)
    sums = {"a": {}, "b": {}}
    for name, prediction in (("a", prediction_a), ("b", prediction_b)):
        for mode in MODES:
            error = np.abs(np.asarray(prediction[mode]) - np.asarray(labels))
            sums[name][mode] = np.asarray([error[groups[key]].sum() for key in keys])
    rng = np.random.RandomState(int(seed))
    deltas = []
    for _ in range(int(iterations)):
        chosen = rng.randint(0, len(keys), size=len(keys))
        denominator = count[chosen].sum()
        j = {}
        for name in ("a", "b"):
            maes = {
                mode: float(sums[name][mode][chosen].sum() / denominator)
                for mode in MODES
            }
            j[name] = 0.5 * maes["LAV"] + 0.5 * np.mean([maes[mode] for mode in MISSING_MODES])
        deltas.append(j["a"] - j["b"])
    values = np.asarray(deltas)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_a_better": float(np.mean(values < 0)),
        "iterations": int(iterations),
    }, values
