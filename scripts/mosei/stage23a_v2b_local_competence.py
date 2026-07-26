#!/usr/bin/env python
"""Stage23A-v2b source-aware local competence audit.

``develop`` reads Direction inner-train/inner-valid only and freezes all
choices. ``outer`` refuses to run before that manifest exists, freezes
label-free predictions first, then opens the Train-OOF outer labels once for
audit metrics. No Expert, neural Judge, Official Valid, Test, or Student is
constructed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from stage23a_v2_common import (
    EXPERTS,
    MODES,
    ROOT,
    regression_metrics,
    sha256_file,
)


V2A = ROOT / "result" / "arbiter_audit_v2" / "mosei"
V2B = ROOT / "result" / "arbiter_audit_v2b" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23a_v2b"
PRED_SLICE = slice(4, 9)
CONTENT_SLICE = slice(22, 54)
AVAIL_SLICE = slice(54, 57)
CONTENT_BLOCKS = ((0, 16, 0, "Text"), (16, 24, 1, "Audio"), (24, 32, 2, "Vision"))
KS = (15, 30, 60)
WEIGHTINGS = ("reciprocal_epsilon_1e-6", "exponential_query_neighbor_median_bandwidth")
TAUS = (0.02, 0.05, 0.10, 0.20)
MARGINS = (0.02, 0.05, 0.10)
COVERAGES = (0.10, 0.20, 0.30, 0.50, 1.00)
CONTROL_SEEDS = {
    "N0_random_same_mode_source": (23401, 23402, 23403),
    "N1_shuffled_content": (23411, 23412, 23413),
    "N4_decision_randomization": (23421, 23422, 23423),
}
EXPECTED = {
    "A": {"inner_train": 29752, "inner_valid": 4376, "outer_evaluation": 31176},
    "B": {"inner_train": 27732, "inner_valid": 3444, "outer_evaluation": 34128},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=lambda item: item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def atomic_tsv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, float_format="%.10g")
    os.replace(str(temporary), str(path))


def atomic_gzip_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as stream:
            frame.to_csv(stream, index=False, float_format="%.10g")
    os.replace(str(temporary), str(path))


def finite_spearman(left, right):
    value = spearmanr(np.asarray(left), np.asarray(right)).correlation
    return float(value) if np.isfinite(value) else 0.0


def overall_j(frame, prediction):
    maes = {}
    values = np.asarray(prediction, dtype=np.float64)
    for mode in MODES:
        mask = frame["mode"].to_numpy() == mode
        maes[mode] = float(np.mean(np.abs(values[mask] - frame.loc[mask, "label"].to_numpy())))
    return 0.5 * maes["LAV"] + 0.5 * np.mean([maes[value] for value in ("LA", "LV", "L")])


def load_role(direction, role, allow_outer=False, with_labels=True):
    if role == "outer_evaluation" and not allow_outer:
        raise RuntimeError("Outer role loader is locked during development.")
    root = V2A / "features" / "meta57" / "direction_{}".format(direction)
    values = np.load(root / "{}_features_57d.npz".format(role))
    sidecar_path = root / "{}_targets_and_audit.csv".format(role)
    sidecar_columns = [
        "meta_row_id", "sample_id", "video_id", "train_index", "mode",
        "expert_fold", "role", "strong_static",
    ]
    if with_labels:
        sidecar_columns.append("label")
    sidecar = pd.read_csv(
        sidecar_path,
        usecols=sidecar_columns,
        dtype={"sample_id": str, "video_id": str, "meta_row_id": str},
    )
    X = values["X"].astype(np.float64)
    if X.shape != (EXPECTED[direction][role], 57):
        raise RuntimeError("Unexpected {} {} shape {}".format(direction, role, X.shape))
    if sidecar["meta_row_id"].tolist() != values["meta_row_id"].astype(str).tolist():
        raise RuntimeError("Feature/sidecar binding mismatch.")
    predictions = X[:, PRED_SLICE]
    result = {
        "X": X,
        "sidecar": sidecar,
        "predictions": predictions,
    }
    if with_labels:
        labels = sidecar["label"].to_numpy(dtype=np.float64)
        errors = np.abs(predictions - labels[:, None])
        result.update(
            {
                "errors": errors,
                "squared_errors": np.square(predictions - labels[:, None]),
                "regrets": errors - errors.min(axis=1, keepdims=True),
                "best": (errors == errors.min(axis=1, keepdims=True)).astype(np.float64),
                "static_errors": np.abs(sidecar["strong_static"].to_numpy(dtype=np.float64) - labels),
            }
        )
    return result


def attach_outer_labels(direction, data):
    root = V2A / "features" / "meta57" / "direction_{}".format(direction)
    labels = pd.read_csv(
        root / "outer_evaluation_targets_and_audit.csv",
        usecols=["meta_row_id", "label"],
        dtype={"meta_row_id": str},
    )
    if labels["meta_row_id"].tolist() != data["sidecar"]["meta_row_id"].tolist():
        raise RuntimeError("Outer label binding mismatch.")
    data["sidecar"]["label"] = labels["label"].to_numpy(dtype=np.float64)
    error = np.abs(data["predictions"] - data["sidecar"]["label"].to_numpy()[:, None])
    data["errors"] = error
    data["squared_errors"] = np.square(data["predictions"] - data["sidecar"]["label"].to_numpy()[:, None])
    data["regrets"] = error - error.min(axis=1, keepdims=True)
    data["best"] = (error == error.min(axis=1, keepdims=True)).astype(np.float64)
    data["static_errors"] = np.abs(
        data["sidecar"]["strong_static"].to_numpy(dtype=float)
        - data["sidecar"]["label"].to_numpy(dtype=float)
    )


def decision_raw(X):
    predictions = X[:, PRED_SLICE]
    ranks = np.argsort(np.argsort(predictions, axis=1), axis=1).astype(np.float64) / 4.0
    return np.concatenate([predictions, ranks, X[:, 9:12]], axis=1)


def fit_stats(train):
    stats = {}
    frame = train["sidecar"]
    for mode in MODES:
        mask = frame["mode"].to_numpy() == mode
        content = train["X"][mask, CONTENT_SLICE].copy()
        availability = train["X"][mask, AVAIL_SLICE].astype(np.int8)
        content_mean = np.zeros(32)
        content_std = np.ones(32)
        for start, stop, availability_column, _ in CONTENT_BLOCKS:
            active = availability[:, availability_column] == 1
            if np.any(active):
                content_mean[start:stop] = content[active, start:stop].mean(axis=0)
                content_std[start:stop] = np.maximum(
                    content[active, start:stop].std(axis=0), 1e-8
                )
            else:
                # Structurally unavailable blocks are never used by the
                # content distance. Keep their frozen transform finite and
                # deterministic so that missingness does not introduce NaNs.
                content_mean[start:stop] = 0.0
                content_std[start:stop] = 1.0
        decision = decision_raw(train["X"][mask])
        decision_mean = decision.mean(axis=0)
        decision_std = np.maximum(decision.std(axis=0), 1e-8)
        transformed_content = (content - content_mean) / content_std
        for start, stop, availability_column, _ in CONTENT_BLOCKS:
            transformed_content[availability[:, availability_column] == 0, start:stop] = 0
        transformed_decision = (decision - decision_mean) / decision_std
        rng = np.random.RandomState(23500 + MODES.index(mode))
        n_pairs = min(20000, len(content) * 3)
        left = rng.randint(0, len(content), size=n_pairs)
        right = rng.randint(0, len(content), size=n_pairs)
        content_scales = {}
        decision_scales = {}
        for metric in ("standardized_euclidean", "cosine"):
            c = content_paired_distance(
                transformed_content[left], availability[left],
                transformed_content[right], availability[right], metric,
            )
            d = decision_paired_distance(
                transformed_decision[left], transformed_decision[right], metric
            )
            content_scales[metric] = max(float(np.median(c[c > 0])), 1e-6)
            decision_scales[metric] = max(float(np.median(d[d > 0])), 1e-6)
        stats[mode] = {
            "content_mean": content_mean,
            "content_std": content_std,
            "decision_mean": decision_mean,
            "decision_std": decision_std,
            "content_distance_scale": content_scales,
            "decision_distance_scale": decision_scales,
        }
    return stats


def transform_mode(data, mode, stats):
    mask = data["sidecar"]["mode"].to_numpy() == mode
    content = data["X"][mask, CONTENT_SLICE].copy()
    availability = data["X"][mask, AVAIL_SLICE].astype(np.int8)
    local = stats[mode]
    content = (content - local["content_mean"]) / local["content_std"]
    for start, stop, availability_column, _ in CONTENT_BLOCKS:
        content[availability[:, availability_column] == 0, start:stop] = 0
    decision = (decision_raw(data["X"][mask]) - local["decision_mean"]) / local["decision_std"]
    return {
        "row_indices": np.flatnonzero(mask),
        "content": content,
        "availability": availability,
        "decision": decision,
        "source": data["sidecar"].loc[mask, "video_id"].to_numpy(dtype=str),
        "sample_id": data["sidecar"].loc[mask, "sample_id"].to_numpy(dtype=str),
        "meta_row_id": data["sidecar"].loc[mask, "meta_row_id"].to_numpy(dtype=str),
    }


def content_paired_distance(left, left_avail, right, right_avail, metric):
    total = np.zeros(len(left), dtype=np.float64)
    active_blocks = np.zeros(len(left), dtype=np.float64)
    for start, stop, availability_column, _ in CONTENT_BLOCKS:
        active = (left_avail[:, availability_column] == 1) & (right_avail[:, availability_column] == 1)
        if metric == "standardized_euclidean":
            distance = np.sqrt(np.mean(np.square(left[:, start:stop] - right[:, start:stop]), axis=1))
        else:
            numerator = np.sum(left[:, start:stop] * right[:, start:stop], axis=1)
            denominator = np.linalg.norm(left[:, start:stop], axis=1) * np.linalg.norm(right[:, start:stop], axis=1)
            distance = 1.0 - np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
        total[active] += distance[active]
        active_blocks[active] += 1
    return total / np.maximum(active_blocks, 1)


def decision_paired_distance(left, right, metric):
    if metric == "standardized_euclidean":
        return np.sqrt(np.mean(np.square(left - right), axis=1))
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return 1.0 - np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def content_distance_matrix(query, query_avail, reference, reference_avail, metric):
    total = np.zeros((len(query), len(reference)), dtype=np.float32)
    count = np.zeros_like(total)
    for start, stop, availability_column, _ in CONTENT_BLOCKS:
        if metric == "standardized_euclidean":
            distance = pairwise_distances(
                query[:, start:stop], reference[:, start:stop], metric="euclidean", n_jobs=1
            ) / math.sqrt(stop - start)
        else:
            distance = pairwise_distances(
                query[:, start:stop], reference[:, start:stop], metric="cosine", n_jobs=1
            )
        active = np.outer(
            query_avail[:, availability_column] == 1,
            reference_avail[:, availability_column] == 1,
        )
        total += np.where(active, distance, 0).astype(np.float32)
        count += active.astype(np.float32)
    return total / np.maximum(count, 1)


def decision_distance_matrix(query, reference, metric):
    if metric == "standardized_euclidean":
        return (
            pairwise_distances(query, reference, metric="euclidean", n_jobs=1)
            / math.sqrt(query.shape[1])
        ).astype(np.float32)
    return pairwise_distances(query, reference, metric="cosine", n_jobs=1).astype(np.float32)


def config_id(space, metric, k, weighting, beta=None):
    value = "{}__{}__K{}__{}".format(space, metric, k, weighting)
    if beta is not None:
        value += "__beta{}".format(str(beta).replace(".", "p"))
    return value


def candidate_configs():
    rows = []
    for metric in ("standardized_euclidean", "cosine"):
        for k in KS:
            for weighting in WEIGHTINGS:
                rows.append({"space": "C", "distance": metric, "K": k, "weighting": weighting, "beta": None})
                rows.append({"space": "D", "distance": metric, "K": k, "weighting": weighting, "beta": None})
                for beta in (0.25, 0.50, 0.75):
                    rows.append({"space": "H", "distance": metric, "K": k, "weighting": weighting, "beta": beta})
    for row in rows:
        row["config_id"] = config_id(row["space"], row["distance"], row["K"], row["weighting"], row["beta"])
    return rows


def source_topk(distances, reference_sources, query_sources, kmax):
    unique_sources, inverse = np.unique(reference_sources, return_inverse=True)
    available = len(unique_sources)
    k = min(kmax, available)
    source_min = np.empty((len(distances), available), dtype=np.float32)
    source_arg = np.empty((len(distances), available), dtype=np.int32)
    for source_index in range(available):
        indices = np.flatnonzero(inverse == source_index)
        local = distances[:, indices]
        position = np.argmin(local, axis=1)
        source_min[:, source_index] = local[np.arange(len(local)), position]
        source_arg[:, source_index] = indices[position]
    query_to_source = {value: index for index, value in enumerate(unique_sources)}
    for query_index, source in enumerate(query_sources):
        if source in query_to_source:
            source_min[query_index, query_to_source[source]] = np.inf
    # Resolve exact (or float32-identical) distance ties by the frozen,
    # lexicographically sorted source index.  The 1e-12 key perturbation is
    # far below float32 distance resolution and is used only for ordering;
    # all reported distances and weights retain the unmodified values.
    tie_key = source_min.astype(np.float64) + (
        np.arange(available, dtype=np.float64)[None, :] * 1e-12
    )
    positions = np.argpartition(tie_key, kth=k - 1, axis=1)[:, :k]
    values = np.take_along_axis(source_min, positions, axis=1)
    selected_keys = np.take_along_axis(tie_key, positions, axis=1)
    order = np.argsort(selected_keys, axis=1)
    positions = np.take_along_axis(positions, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    reference_indices = np.take_along_axis(source_arg, positions, axis=1)
    return reference_indices, values, unique_sources[positions]


def neighbor_weights(distances, weighting):
    if weighting == "reciprocal_epsilon_1e-6":
        weights = 1.0 / (np.asarray(distances, dtype=np.float64) + 1e-6)
    else:
        bandwidth = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-6)
        weights = np.exp(-np.asarray(distances, dtype=np.float64) / bandwidth)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def competence(reference, neighbor_indices, neighbor_distances, weighting):
    weights = neighbor_weights(neighbor_distances, weighting)
    errors = reference["errors"][neighbor_indices]
    regrets = reference["regrets"][neighbor_indices]
    best = reference["best"][neighbor_indices]
    static_errors = reference["static_errors"][neighbor_indices]
    risk = np.sum(weights[:, :, None] * errors, axis=1)
    local_regret = np.sum(weights[:, :, None] * regrets, axis=1)
    centered = errors - risk[:, None, :]
    return {
        "risk": risk,
        "median_error": np.median(errors, axis=1),
        "regret": local_regret,
        "soft_win_rate": np.sum(weights[:, :, None] * best, axis=1),
        "error_variance": np.sum(weights[:, :, None] * np.square(centered), axis=1),
        "effective_neighbor_count": np.repeat(
            (1.0 / np.maximum(np.sum(np.square(weights), axis=1), 1e-12))[:, None],
            len(EXPERTS),
            axis=1,
        ),
        "static_risk": np.sum(weights * static_errors, axis=1),
    }


def regularity_metrics(estimated_risk, estimated_regret, actual_errors, actual_regrets):
    true_best = np.argmin(actual_errors, axis=1)
    order = np.argsort(estimated_risk, axis=1)
    q_true = softmax(-actual_regrets / 0.05, axis=1)
    q_est = softmax(-estimated_risk / 0.05, axis=1)
    kl = np.sum(q_true * (np.log(np.clip(q_true, 1e-12, 1)) - np.log(np.clip(q_est, 1e-12, 1))), axis=1)
    return {
        "spearman_local_risk_vs_error": finite_spearman(estimated_risk.ravel(), actual_errors.ravel()),
        "spearman_local_regret_vs_regret": finite_spearman(estimated_regret.ravel(), actual_regrets.ravel()),
        "top1_accuracy": float(np.mean(order[:, 0] == true_best)),
        "top2_coverage": float(np.mean(np.any(order[:, :2] == true_best[:, None], axis=1))),
        "soft_regret_KL": float(kl.mean()),
    }


def method_prediction(risk, expert_predictions, method, tau=None, margin=None):
    if method == "Local-DS":
        selected = np.argmin(risk, axis=1)
        weights = np.zeros_like(risk)
        weights[np.arange(len(risk)), selected] = 1
    else:
        logits = -risk / float(tau)
        if method == "Local-DWS":
            keep = risk <= (risk.min(axis=1, keepdims=True) + float(margin))
            logits = np.where(keep, logits, -1e9)
        weights = softmax(logits, axis=1)
    prediction = np.sum(weights * expert_predictions, axis=1)
    estimated_risk = np.sum(weights * risk, axis=1)
    return prediction, estimated_risk, weights


def method_grid(risk, query):
    rows = []
    outputs = {}
    candidates = [("Local-DS", None, None)]
    candidates.extend(("Local-DW", tau, None) for tau in TAUS)
    candidates.extend(("Local-DWS", tau, margin) for tau in TAUS for margin in MARGINS)
    for method, tau, margin in candidates:
        prediction, estimated, weights = method_prediction(
            risk, query["predictions"], method, tau, margin
        )
        identifier = "{}__tau{}__margin{}".format(method, tau, margin)
        j = overall_j(query["sidecar"], prediction)
        rows.append({"method_config_id": identifier, "DRS": method, "tau": tau, "margin": margin, "inner_valid_J": j})
        outputs[identifier] = (prediction, estimated, weights)
    selected = min(rows, key=lambda row: (row["inner_valid_J"], row["method_config_id"]))
    return pd.DataFrame(rows), selected, outputs[selected["method_config_id"]]


def prediction_metric_rows(frame, methods):
    rows = []
    for name, prediction in methods.items():
        for mode in list(MODES) + ["Overall"]:
            mask = np.ones(len(frame), dtype=bool) if mode == "Overall" else frame["mode"].to_numpy() == mode
            metrics = regression_metrics(np.asarray(prediction)[mask], frame.loc[mask, "label"])
            metrics["J"] = overall_j(frame, prediction) if mode == "Overall" else metrics["MAE"]
            rows.append({"method": name, "mode": mode, **metrics})
    return rows


def fixed_coverage_predictions(static_prediction, dynamic_prediction, estimated_gain):
    outputs = {}
    triggers = {}
    for coverage in COVERAGES:
        count = max(1, int(round(len(estimated_gain) * coverage)))
        order = np.argsort(-estimated_gain)
        trigger = np.zeros(len(estimated_gain), dtype=bool)
        trigger[order[:count]] = True
        prediction = np.where(trigger, dynamic_prediction, static_prediction)
        outputs["coverage_{:.0f}".format(coverage * 100)] = prediction
        triggers[coverage] = trigger
    return outputs, triggers


def derange_indices(sources, seed):
    rng = np.random.RandomState(seed)
    sources = np.asarray(sources)
    unique_sources = np.unique(sources)
    rng.shuffle(unique_sources)
    groups = []
    for source in unique_sources:
        group = np.flatnonzero(sources == source)
        rng.shuffle(group)
        groups.append(group)
    source_grouped_rows = np.concatenate(groups)
    maximum_group_size = max(len(group) for group in groups)
    valid_shifts = np.arange(
        maximum_group_size, len(sources) - maximum_group_size + 1
    )
    rng.shuffle(valid_shifts)
    for shift in valid_shifts:
        candidate = np.empty(len(sources), dtype=np.int64)
        candidate[source_grouped_rows] = np.roll(source_grouped_rows, int(shift))
        if not np.any(sources == sources[candidate]):
            return candidate
    raise RuntimeError("Could not create cross-source derangement.")


def distance_for_config(query_mode, reference_mode, stats_mode, config, content_permutation=None, decision_permutation=None):
    reference_content = reference_mode["content"] if content_permutation is None else reference_mode["content"][content_permutation]
    reference_availability = reference_mode["availability"] if content_permutation is None else reference_mode["availability"][content_permutation]
    reference_decision = reference_mode["decision"] if decision_permutation is None else reference_mode["decision"][decision_permutation]
    metric = config["distance"]
    content = None
    decision = None
    if config["space"] in ("C", "H"):
        content = content_distance_matrix(
            query_mode["content"], query_mode["availability"],
            reference_content, reference_availability, metric,
        ) / stats_mode["content_distance_scale"][metric]
    if config["space"] in ("D", "H"):
        decision = decision_distance_matrix(
            query_mode["decision"], reference_decision, metric
        ) / stats_mode["decision_distance_scale"][metric]
    if config["space"] == "C":
        return content
    if config["space"] == "D":
        return decision
    return float(config["beta"]) * content + (1.0 - float(config["beta"])) * decision


def compute_config(reference, query, stats, config, control=None, seed=None, save_neighbors=False, chunk_size=512):
    result = {
        key: np.zeros((len(query["sidecar"]), 5), dtype=np.float64)
        for key in ("risk", "median_error", "regret", "soft_win_rate", "error_variance")
    }
    result["effective_neighbor_count"] = np.zeros((len(query["sidecar"]), 5), dtype=np.float64)
    result["static_risk"] = np.zeros(len(query["sidecar"]), dtype=np.float64)
    neighbor_rows = []
    density_rows = []
    for mode in MODES:
        ref_mode = transform_mode(reference, mode, stats)
        query_mode = transform_mode(query, mode, stats)
        overlap = set(ref_mode["source"]) & set(query_mode["source"])
        if overlap:
            raise RuntimeError(
                "Reference/query source overlap in {}: {}".format(
                    mode, len(overlap)
                )
            )
        content_permutation = None
        decision_permutation = None
        if control == "N1_shuffled_content":
            content_permutation = derange_indices(ref_mode["source"], seed)
        if control == "N4_decision_randomization":
            decision_permutation = derange_indices(ref_mode["source"], seed)
        mode_output = {key: [] for key in result}
        for start in range(0, len(query_mode["row_indices"]), chunk_size):
            stop = min(start + chunk_size, len(query_mode["row_indices"]))
            local_query = {
                key: value[start:stop] if isinstance(value, np.ndarray) else value
                for key, value in query_mode.items()
            }
            if control == "N0_random_same_mode_source":
                rng = np.random.RandomState(int(seed) + MODES.index(mode) * 100000 + start)
                unique_sources = np.unique(ref_mode["source"])
                k = min(int(config["K"]), len(unique_sources))
                indices = np.zeros((stop - start, k), dtype=np.int32)
                source_values = np.empty((stop - start, k), dtype=object)
                for query_index in range(stop - start):
                    candidates = unique_sources[unique_sources != local_query["source"][query_index]]
                    chosen = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
                    for neighbor_index, source in enumerate(chosen):
                        clips = np.flatnonzero(ref_mode["source"] == source)
                        indices[query_index, neighbor_index] = rng.choice(clips)
                        source_values[query_index, neighbor_index] = source
                distances = np.ones_like(indices, dtype=np.float32)
            else:
                distances_full = distance_for_config(
                    local_query, ref_mode, stats[mode], config,
                    content_permutation, decision_permutation,
                )
                indices, distances, source_values = source_topk(
                    distances_full,
                    ref_mode["source"],
                    local_query["source"],
                    int(config["K"]),
                )
            global_ref_indices = ref_mode["row_indices"][indices]
            comp = competence(reference, global_ref_indices, distances, config["weighting"])
            for key in result:
                mode_output[key].append(comp[key])
            density_rows.append(
                {
                    "mode": mode,
                    "query_start": int(start),
                    "queries": int(stop - start),
                    "mean_nearest_distance": float(distances[:, 0].mean()),
                    "mean_kth_distance": float(distances[:, -1].mean()),
                    "mean_neighbor_sources": float(distances.shape[1]),
                    "available_reference_sources": int(np.unique(ref_mode["source"]).size),
                }
            )
            if save_neighbors:
                for query_position in range(stop - start):
                    for rank in range(indices.shape[1]):
                        ref_position = indices[query_position, rank]
                        neighbor_rows.append(
                            {
                                "query_meta_row_id": local_query["meta_row_id"][query_position],
                                "query_sample_id": local_query["sample_id"][query_position],
                                "query_video_id": local_query["source"][query_position],
                                "mode": mode,
                                "neighbor_rank": rank + 1,
                                "neighbor_sample_id": ref_mode["sample_id"][ref_position],
                                "neighbor_video_id": ref_mode["source"][ref_position],
                                "distance": float(distances[query_position, rank]),
                            }
                        )
        target_rows = query_mode["row_indices"]
        for key in result:
            result[key][target_rows] = np.concatenate(mode_output[key], axis=0)
    return result, pd.DataFrame(neighbor_rows), pd.DataFrame(density_rows)


def stats_to_json(stats):
    output = {}
    for mode, values in stats.items():
        output[mode] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in values.items()
        }
    return output


def risks_ledger(direction, roles):
    rows = []
    for role, data in roles.items():
        frame = data["sidecar"][["meta_row_id", "sample_id", "video_id", "mode", "expert_fold", "role"]].copy()
        for expert_index, expert in enumerate(EXPERTS):
            local = frame.copy()
            local["expert_id"] = expert
            local["absolute_error"] = data["errors"][:, expert_index]
            local["squared_error"] = data["squared_errors"][:, expert_index]
            local["regret"] = data["regrets"][:, expert_index]
            local["best_expert_indicator"] = data["best"][:, expert_index].astype(int)
            rows.append(local)
    output = pd.concat(rows, ignore_index=True)
    path = V2B / "data" / "direction_{}_expert_risk_ledger.csv.gz".format(direction)
    atomic_gzip_csv(output, path)
    return path


def candidate_development(direction, train, valid, stats):
    configs = candidate_configs()
    store = {}
    for config in configs:
        store[config["config_id"]] = {
            "risk": np.zeros((len(valid["sidecar"]), 5)),
            "regret": np.zeros((len(valid["sidecar"]), 5)),
        }
    for mode in MODES:
        ref_mode = transform_mode(train, mode, stats)
        query_mode = transform_mode(valid, mode, stats)
        base = {}
        for metric in ("standardized_euclidean", "cosine"):
            base[("C", metric)] = content_distance_matrix(
                query_mode["content"], query_mode["availability"],
                ref_mode["content"], ref_mode["availability"], metric,
            ) / stats[mode]["content_distance_scale"][metric]
            base[("D", metric)] = decision_distance_matrix(
                query_mode["decision"], ref_mode["decision"], metric
            ) / stats[mode]["decision_distance_scale"][metric]
        top_cache = {}
        for metric in ("standardized_euclidean", "cosine"):
            variants = [("C", None, base[("C", metric)]), ("D", None, base[("D", metric)])]
            variants.extend(
                ("H", beta, beta * base[("C", metric)] + (1 - beta) * base[("D", metric)])
                for beta in (0.25, 0.50, 0.75)
            )
            for space, beta, distances in variants:
                indices, values, _ = source_topk(
                    distances, ref_mode["source"], query_mode["source"], max(KS)
                )
                top_cache[(space, metric, beta)] = (indices, values)
        for config in configs:
            indices, values = top_cache[(config["space"], config["distance"], config["beta"])]
            k = int(config["K"])
            comp = competence(
                train,
                ref_mode["row_indices"][indices[:, :k]],
                values[:, :k],
                config["weighting"],
            )
            target = query_mode["row_indices"]
            store[config["config_id"]]["risk"][target] = comp["risk"]
            store[config["config_id"]]["regret"][target] = comp["regret"]
        del base, top_cache

    rows = []
    detail_rows = []
    global_mode_risk = np.zeros_like(valid["errors"])
    for mode in MODES:
        train_mask = train["sidecar"]["mode"].to_numpy() == mode
        valid_mask = valid["sidecar"]["mode"].to_numpy() == mode
        global_mode_risk[valid_mask] = train["errors"][train_mask].mean(axis=0)
    baseline = regularity_metrics(
        global_mode_risk,
        global_mode_risk - global_mode_risk.min(axis=1, keepdims=True),
        valid["errors"],
        valid["regrets"],
    )
    for config in configs:
        values = store[config["config_id"]]
        metrics = regularity_metrics(values["risk"], values["regret"], valid["errors"], valid["regrets"])
        score = 0.5 * (
            metrics["spearman_local_risk_vs_error"]
            + metrics["spearman_local_regret_vs_regret"]
        )
        rows.append(
            {
                "direction": direction,
                **config,
                **metrics,
                "selection_score": score,
                "delta_risk_spearman_vs_mode_only": metrics["spearman_local_risk_vs_error"] - baseline["spearman_local_risk_vs_error"],
                "delta_regret_spearman_vs_mode_only": metrics["spearman_local_regret_vs_regret"] - baseline["spearman_local_regret_vs_regret"],
            }
        )
        for mode in MODES:
            mode_mask = valid["sidecar"]["mode"].to_numpy() == mode
            for expert_index, expert in enumerate(EXPERTS):
                detail_rows.append(
                    {
                        "direction": direction,
                        "config_id": config["config_id"],
                        "mode": mode,
                        "expert_id": expert,
                        "spearman_risk_error": finite_spearman(
                            values["risk"][mode_mask, expert_index],
                            valid["errors"][mode_mask, expert_index],
                        ),
                        "spearman_regret": finite_spearman(
                            values["regret"][mode_mask, expert_index],
                            valid["regrets"][mode_mask, expert_index],
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    selected_row = result.sort_values(
        ["selection_score", "top1_accuracy", "config_id"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()
    selected_config = {
        key: selected_row[key] if not (isinstance(selected_row[key], float) and np.isnan(selected_row[key])) else None
        for key in ("space", "distance", "K", "weighting", "beta", "config_id")
    }
    selected_config["K"] = int(selected_config["K"])
    return result, pd.DataFrame(detail_rows), selected_config, store[selected_config["config_id"]]


def control_outputs(direction, train, query, stats, selected_config, method_selection, safety, role):
    rows = []
    predictions = {}
    risk_estimates = {}
    regularity = []
    method = method_selection["DRS"]
    tau = method_selection.get("tau")
    margin = method_selection.get("margin")
    threshold = safety["estimated_gain_threshold"]
    controls = []
    for control, seeds in CONTROL_SEEDS.items():
        for seed in seeds:
            controls.append((control, seed))
    for control, seed in controls:
        comp, _, density = compute_config(
            train, query, stats, selected_config, control=control, seed=seed, save_neighbors=False
        )
        dynamic, dynamic_risk, _ = method_prediction(
            comp["risk"], query["predictions"], method, tau, margin
        )
        estimated_gain = comp["static_risk"] - dynamic_risk
        trigger = estimated_gain >= threshold if np.isfinite(threshold) else np.zeros(len(dynamic), dtype=bool)
        safe = np.where(trigger, dynamic, query["sidecar"]["strong_static"].to_numpy(dtype=float))
        identifier = "{}__seed{}".format(control, seed)
        predictions[identifier] = safe
        risk_estimates[identifier] = (comp["risk"], comp["regret"])
        if role == "inner_valid":
            j = overall_j(query["sidecar"], safe)
        else:
            j = np.nan
        reg = regularity_metrics(comp["risk"], comp["regret"], query["errors"], query["regrets"]) if role == "inner_valid" else {}
        rows.append(
            {
                "direction": direction,
                "role": role,
                "control": control,
                "seed": seed,
                "J": j,
                "trigger_rate": float(trigger.mean()),
                **reg,
            }
        )
    # N2 global and N3 mode-only competence.
    global_risk = np.tile(train["errors"].mean(axis=0), (len(query["sidecar"]), 1))
    mode_risk = np.zeros_like(global_risk)
    global_static = float(train["static_errors"].mean())
    mode_static = np.zeros(len(query["sidecar"]))
    for mode_name in MODES:
        train_mask = train["sidecar"]["mode"].to_numpy() == mode_name
        query_mask = query["sidecar"]["mode"].to_numpy() == mode_name
        mode_risk[query_mask] = train["errors"][train_mask].mean(axis=0)
        mode_static[query_mask] = train["static_errors"][train_mask].mean()
    for name, risk, static_risk in (
        ("N2_global_competence", global_risk, np.full(len(query["sidecar"]), global_static)),
        ("N3_mode_only_competence", mode_risk, mode_static),
    ):
        dynamic, dynamic_risk, _ = method_prediction(risk, query["predictions"], method, tau, margin)
        estimated_gain = static_risk - dynamic_risk
        trigger = estimated_gain >= threshold if np.isfinite(threshold) else np.zeros(len(dynamic), dtype=bool)
        safe = np.where(trigger, dynamic, query["sidecar"]["strong_static"].to_numpy(dtype=float))
        predictions[name] = safe
        risk_estimates[name] = (risk, risk - risk.min(axis=1, keepdims=True))
        rows.append(
            {
                "direction": direction,
                "role": role,
                "control": name,
                "seed": -1,
                "J": overall_j(query["sidecar"], safe) if role == "inner_valid" else np.nan,
                "trigger_rate": float(trigger.mean()),
                **(regularity_metrics(risk, risk - risk.min(axis=1, keepdims=True), query["errors"], query["regrets"]) if role == "inner_valid" else {}),
            }
        )
    return pd.DataFrame(rows), predictions, risk_estimates


def development():
    protocol_path = V2B / "protocol" / "frozen_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["outer_evaluation_access_count"] != 0:
        raise RuntimeError("Outer was already accessed.")
    analysis = V2B / "development"
    selection = {"stage": "Stage23A-v2b frozen development selection", "directions": {}, "outer_evaluation_used": False}
    all_candidate, all_detail, all_method, all_coverage, all_controls = [], [], [], [], []
    schema = {
        "risk_fields": ["absolute_error", "squared_error", "regret", "best_expert_indicator"],
        "query_features": {
            "C": "32D Direction-specific content PCA plus effective availability; block-wise joint-effective distance",
            "D": "5 predictions + 5 within-row ranks + committee std/range/polarity; inner-train standardized",
            "H": "beta*C_distance+(1-beta)*D_distance after inner-train distance-scale normalization",
        },
        "forbidden_query_fields": ["label", "absolute_error", "squared_error", "regret", "best_expert_indicator", "video_id", "expert_fold", "raw_hierarchical_consistency"],
        "source_rule": "same mode; query source excluded; one nearest clip per reference source",
    }
    atomic_json(V2B / "protocol" / "reference_bank_schema.json", schema)
    fold_identity = pd.read_csv(
        V2A / "analysis_v2a" / "probe_b_fold_identity.tsv", sep="\t"
    )
    identity_auc = float(
        fold_identity[
            (fold_identity["role"] == "inner_valid")
            & (fold_identity["probe"] == "content_plus_consistency")
        ]["AUROC"].iloc[0]
    )
    atomic_json(
        analysis / "consistency_ablation_decision.json",
        {
            "status": "STOPPED_NOT_ENTERED_INTO_ROUTING",
            "v2a_fold_identity_AUROC": identity_auc,
            "clearly_above_random": identity_auc >= 0.70,
            "raw_consistency_in_main_method": False,
            "reason": "Frozen protocol forbids the ablation when fold identity remains clearly above random.",
        },
    )

    for direction in ("A", "B"):
        train = load_role(direction, "inner_train")
        valid = load_role(direction, "inner_valid")
        risk_path = risks_ledger(direction, {"inner_train": train, "inner_valid": valid})
        stats = fit_stats(train)
        stats_path = V2B / "protocol" / "direction_{}_retrieval_stats.json".format(direction)
        atomic_json(stats_path, stats_to_json(stats))
        candidates, details, selected_config, selected_store = candidate_development(
            direction, train, valid, stats
        )
        all_candidate.append(candidates)
        all_detail.append(details)
        selected_comp, neighbor_ledger, density = compute_config(
            train,
            valid,
            stats,
            selected_config,
            save_neighbors=True,
            chunk_size=len(valid["sidecar"]),
        )
        recomputation_delta = np.abs(selected_comp["risk"] - selected_store["risk"])
        if (
            not np.isfinite(selected_store["risk"]).all()
            or not np.isfinite(selected_comp["risk"]).all()
            or float(np.max(recomputation_delta)) > 1e-6
        ):
            raise RuntimeError(
                "Selected candidate recomputation mismatch: config={} max_delta={} "
                "mean_delta={} changed={}".format(
                    selected_config,
                    float(np.max(recomputation_delta)),
                    float(np.mean(recomputation_delta)),
                    int(np.count_nonzero(recomputation_delta > 1e-6)),
                )
            )
        method_table, method_selected, method_output = method_grid(selected_comp["risk"], valid)
        method_table["direction"] = direction
        all_method.append(method_table)
        family_best = {}
        for family in ("Local-DS", "Local-DW", "Local-DWS"):
            row = (
                method_table[method_table["DRS"] == family]
                .sort_values(["inner_valid_J", "method_config_id"])
                .iloc[0]
                .to_dict()
            )
            family_best[family] = {
                "DRS": family,
                "tau": None if pd.isna(row["tau"]) else float(row["tau"]),
                "margin": None if pd.isna(row["margin"]) else float(row["margin"]),
                "inner_valid_J": float(row["inner_valid_J"]),
            }
        dynamic_prediction, dynamic_risk, dynamic_weights = method_output
        estimated_gain = selected_comp["static_risk"] - dynamic_risk
        fixed_predictions, fixed_triggers = fixed_coverage_predictions(
            valid["sidecar"]["strong_static"].to_numpy(dtype=float),
            dynamic_prediction,
            estimated_gain,
        )
        coverage_rows = []
        for coverage, trigger in fixed_triggers.items():
            name = "coverage_{:.0f}".format(coverage * 100)
            prediction = fixed_predictions[name]
            coverage_rows.append(
                {
                    "direction": direction,
                    "coverage": coverage,
                    "inner_valid_J": overall_j(valid["sidecar"], prediction),
                    "estimated_gain_threshold": float(np.min(estimated_gain[trigger])),
                    "trigger_rate": float(trigger.mean()),
                }
            )
        zero_j = overall_j(valid["sidecar"], valid["sidecar"]["strong_static"].to_numpy(dtype=float))
        coverage_rows.append(
            {"direction": direction, "coverage": 0.0, "inner_valid_J": zero_j, "estimated_gain_threshold": float("inf"), "trigger_rate": 0.0}
        )
        coverage_table = pd.DataFrame(coverage_rows)
        chosen_coverage = coverage_table.sort_values(["inner_valid_J", "coverage"]).iloc[0].to_dict()
        all_coverage.append(coverage_table)
        safety = {
            "coverage": float(chosen_coverage["coverage"]),
            "inner_valid_J": float(chosen_coverage["inner_valid_J"]),
            "strong_static_inner_valid_J": zero_j,
            "estimated_gain_threshold": float(chosen_coverage["estimated_gain_threshold"]),
        }
        control_table, _, _ = control_outputs(
            direction, train, valid, stats, selected_config, method_selected, safety, "inner_valid"
        )
        all_controls.append(control_table)
        common_config = dict(
            json.loads(
                (V2B / "protocol" / "candidate_table.json").read_text(
                    encoding="utf-8"
                )
            )["common_configuration"]
        )
        common_region = {
            "space": common_config["space"],
            "distance": common_config["distance"],
            "K": int(common_config["K"]),
            "weighting": common_config["weighting"],
            "beta": None,
            "config_id": config_id(
                common_config["space"],
                common_config["distance"],
                int(common_config["K"]),
                common_config["weighting"],
                None,
            ),
        }
        common_comp, _, _ = compute_config(
            train, valid, stats, common_region, save_neighbors=False
        )
        common_dynamic, _, _ = method_prediction(
            common_comp["risk"],
            valid["predictions"],
            "Local-DW",
            float(common_config["tau"]),
            None,
        )
        common_outputs, _ = fixed_coverage_predictions(
            valid["sidecar"]["strong_static"].to_numpy(dtype=float),
            common_dynamic,
            common_comp["static_risk"]
            - np.sum(
                softmax(-common_comp["risk"] / float(common_config["tau"]), axis=1)
                * common_comp["risk"],
                axis=1,
            ),
        )
        common_valid_prediction = common_outputs[
            "coverage_{:.0f}".format(float(common_config["safe_coverage"]) * 100)
        ]
        neighbor_path = analysis / "direction_{}_selected_inner_valid_neighbors.csv.gz".format(direction)
        atomic_gzip_csv(neighbor_ledger, neighbor_path)
        atomic_tsv(density, analysis / "direction_{}_inner_valid_density.tsv".format(direction))
        valid_frequency, valid_source_summary = source_audit(
            neighbor_ledger, density, direction, "inner_valid"
        )
        atomic_tsv(
            valid_frequency,
            analysis
            / "direction_{}_inner_valid_source_frequency.tsv".format(direction),
        )
        atomic_json(
            analysis
            / "direction_{}_inner_valid_source_summary.json".format(direction),
            valid_source_summary,
        )
        local_ledger = valid["sidecar"][["meta_row_id", "sample_id", "video_id", "mode"]].copy()
        for expert_index, expert in enumerate(EXPERTS):
            for key in ("risk", "median_error", "regret", "soft_win_rate", "error_variance", "effective_neighbor_count"):
                local_ledger["{}__{}".format(key, expert)] = selected_comp[key][:, expert_index]
        local_ledger["estimated_static_local_risk"] = selected_comp["static_risk"]
        local_ledger["estimated_dynamic_local_risk"] = dynamic_risk
        local_ledger["estimated_gain"] = estimated_gain
        atomic_gzip_csv(local_ledger, analysis / "direction_{}_inner_valid_local_competence.csv.gz".format(direction))
        selection["directions"][direction] = {
            "selected_region": selected_config,
            "selected_method": {
                "method_config_id": method_selected["method_config_id"],
                "DRS": method_selected["DRS"],
                "tau": None if pd.isna(method_selected["tau"]) else float(method_selected["tau"]),
                "margin": None if pd.isna(method_selected["margin"]) else float(method_selected["margin"]),
                "inner_valid_J": float(method_selected["inner_valid_J"]),
            },
            "family_best_methods": family_best,
            "safety": safety,
            "common_configuration": {
                **common_config,
                "inner_valid_J": overall_j(
                    valid["sidecar"], common_valid_prediction
                ),
            },
            "strong_static": "per_mode_constrained_fixed_stacking",
            "risk_ledger_path": str(risk_path.resolve()),
            "risk_ledger_sha256": sha256_file(risk_path),
            "retrieval_stats_path": str(stats_path.resolve()),
            "retrieval_stats_sha256": sha256_file(stats_path),
            "neighbor_ledger_path": str(neighbor_path.resolve()),
            "neighbor_ledger_sha256": sha256_file(neighbor_path),
            "development_samples": {"inner_train": int(len(train["sidecar"])), "inner_valid": int(len(valid["sidecar"]))},
        }
    candidate_frame = pd.concat(all_candidate, ignore_index=True)
    detail_frame = pd.concat(all_detail, ignore_index=True)
    method_frame = pd.concat(all_method, ignore_index=True)
    coverage_frame = pd.concat(all_coverage, ignore_index=True)
    control_frame = pd.concat(all_controls, ignore_index=True)
    atomic_tsv(candidate_frame, analysis / "local_regularity_candidate_metrics.tsv")
    atomic_tsv(detail_frame, analysis / "local_regularity_per_mode_expert.tsv")
    atomic_tsv(method_frame, analysis / "drs_inner_valid_metrics.tsv")
    atomic_tsv(coverage_frame, analysis / "safe_coverage_inner_valid.tsv")
    atomic_tsv(control_frame, analysis / "negative_controls_inner_valid.tsv")
    selection["candidate_metrics_sha256"] = sha256_file(analysis / "local_regularity_candidate_metrics.tsv")
    selection["selected_at"] = utc_now()
    selection["selection_used_roles"] = ["inner_train", "inner_valid"]
    selection["outer_evaluation_access_count"] = 0
    selection["common_configuration"] = json.loads(
        (V2B / "protocol" / "candidate_table.json").read_text(encoding="utf-8")
    )["common_configuration"]
    selection_path = V2B / "protocol" / "frozen_development_selection.json"
    atomic_json(selection_path, selection)
    state = {
        "stage": "Stage23A-v2b",
        "status": "DEVELOPMENT_SELECTION_FROZEN_OUTER_NOT_ACCESSED",
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "outer_evaluation_access_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "feature_rich_judge_trained": False,
        "student_trained": False,
        "updated_at": utc_now(),
    }
    atomic_json(RUNTIME / "state.json", state)
    print(json.dumps(state, indent=2))


def safe_prediction_from_selection(comp, query, selection):
    method = selection["selected_method"]
    dynamic, dynamic_risk, weights = method_prediction(
        comp["risk"], query["predictions"], method["DRS"], method["tau"], method["margin"]
    )
    estimated_gain = comp["static_risk"] - dynamic_risk
    threshold = selection["safety"]["estimated_gain_threshold"]
    trigger = estimated_gain >= threshold if np.isfinite(threshold) else np.zeros(len(dynamic), dtype=bool)
    safe = np.where(trigger, dynamic, query["sidecar"]["strong_static"].to_numpy(dtype=float))
    return dynamic, safe, dynamic_risk, estimated_gain, trigger, weights


def risk_coverage_rows(direction, outer, static, dynamic, estimated_gain):
    rows = []
    outputs, triggers = fixed_coverage_predictions(static, dynamic, estimated_gain)
    actual_static_error = np.abs(static - outer["sidecar"]["label"].to_numpy(dtype=float))
    for coverage, trigger in triggers.items():
        name = "coverage_{:.0f}".format(coverage * 100)
        prediction = outputs[name]
        actual_dynamic_error = np.abs(dynamic - outer["sidecar"]["label"].to_numpy(dtype=float))
        gain = actual_static_error - actual_dynamic_error
        aggregate_metrics = regression_metrics(
            prediction, outer["sidecar"]["label"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "direction": direction,
                "coverage": coverage,
                "J": overall_j(outer["sidecar"], prediction),
                "triggered_true_gain_mean": float(gain[trigger].mean()),
                "untriggered_counterfactual_gain_mean": float(gain[~trigger].mean()) if (~trigger).any() else np.nan,
                "wrong_trigger_fraction": float(np.mean(gain[trigger] < 0)),
                "fallback_rate": float((~trigger).mean()),
                "Corr": aggregate_metrics["Corr"],
                "Acc7": aggregate_metrics["Acc7"],
                "Acc5": aggregate_metrics["Acc5"],
                "Acc2": aggregate_metrics["Acc2"],
                "F1": aggregate_metrics["F1"],
                **{
                    "{}_MAE".format(mode): float(
                        np.mean(
                            np.abs(
                                prediction[outer["sidecar"]["mode"].to_numpy() == mode]
                                - outer["sidecar"].loc[outer["sidecar"]["mode"] == mode, "label"].to_numpy()
                            )
                        )
                    )
                    for mode in MODES
                },
            }
        )
    return pd.DataFrame(rows), outputs


def source_audit(neighbor_ledger, density, direction, role):
    frequency = (
        neighbor_ledger.groupby("neighbor_video_id")
        .agg(neighbor_uses=("query_meta_row_id", "size"), queries_reached=("query_meta_row_id", "nunique"))
        .reset_index()
        .sort_values("neighbor_uses", ascending=False)
    )
    total = max(int(frequency["neighbor_uses"].sum()), 1)
    frequency["share_of_neighbor_slots"] = frequency["neighbor_uses"] / total
    frequency["direction"] = direction
    frequency["role"] = role
    summary = {
        "direction": direction,
        "role": role,
        "queries": int(neighbor_ledger["query_meta_row_id"].nunique()),
        "neighbor_rows": int(len(neighbor_ledger)),
        "unique_neighbor_sources": int(frequency["neighbor_video_id"].nunique()),
        "top1_source_share": float(frequency["share_of_neighbor_slots"].iloc[0]),
        "top10_source_share": float(frequency["share_of_neighbor_slots"].head(10).sum()),
        "mean_nearest_distance": float(density["mean_nearest_distance"].mean()),
        "mean_kth_distance": float(density["mean_kth_distance"].mean()),
    }
    return frequency, summary


def svg_risk_coverage(frame, path):
    width, height, left, top, right, bottom = 720, 420, 70, 50, 30, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = frame["J"].min(), frame["J"].max()
    span = max(ymax - ymin, 1e-6)
    colors = {"A": "#4C78A8", "B": "#F58518"}
    lines = []
    for direction in ("A", "B"):
        local = frame[frame["direction"] == direction].sort_values("coverage")
        points = []
        for row in local.itertuples():
            x = left + row.coverage * plot_w
            y = top + (ymax - row.J) / span * plot_h
            points.append("{:.2f},{:.2f}".format(x, y))
        lines.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="3"/>'.format(" ".join(points), colors[direction]))
    text = """<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect width="100%" height="100%" fill="white"/><text x="{cx}" y="26" text-anchor="middle" font-family="Arial" font-size="18">Stage23A-v2b outer risk–coverage</text>
<line x1="{l}" y1="{t}" x2="{l}" y2="{yb}" stroke="black"/><line x1="{l}" y1="{yb}" x2="{xr}" y2="{yb}" stroke="black"/>
{lines}<text x="{cx}" y="{hy}" text-anchor="middle" font-family="Arial">dynamic coverage</text>
<text x="18" y="{cy}" transform="rotate(-90 18 {cy})" text-anchor="middle" font-family="Arial">J (lower is better)</text>
<text x="{l}" y="{ly}">0%</text><text x="{xr}" y="{ly}" text-anchor="end">100%</text>
</svg>""".format(w=width, h=height, cx=width/2, cy=height/2, l=left, t=top, yb=top+plot_h, xr=left+plot_w, lines="\n".join(lines), hy=height-12, ly=height-42)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def outer():
    selection_path = V2B / "protocol" / "frozen_development_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    state_path = RUNTIME / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "DEVELOPMENT_SELECTION_FROZEN_OUTER_NOT_ACCESSED":
        raise RuntimeError("Outer phase is not authorized from current state.")
    if state["outer_evaluation_access_count"] != 0:
        raise RuntimeError("Outer evaluation already used.")
    output = V2B / "outer"
    prediction_pieces = []
    neighbor_frequency_pieces = []
    source_summaries = []
    density_pieces = []
    coverage_pieces = []
    local_comp_pieces = []
    outer_cache = {}
    control_prediction_names = defaultdict(list)
    selected_outputs = {}
    for direction in ("A", "B"):
        train = load_role(direction, "inner_train")
        outer_data = load_role(
            direction, "outer_evaluation", allow_outer=True, with_labels=False
        )
        outer_cache[direction] = outer_data
        stats_raw = json.loads((V2B / "protocol" / "direction_{}_retrieval_stats.json".format(direction)).read_text())
        stats = {}
        for mode, values in stats_raw.items():
            stats[mode] = {
                key: np.asarray(value, dtype=float) if isinstance(value, list) else value
                for key, value in values.items()
            }
        selected = selection["directions"][direction]
        config = selected["selected_region"]
        comp, neighbors, density = compute_config(
            train, outer_data, stats, config, save_neighbors=True
        )
        dynamic, safe, dynamic_risk, estimated_gain, trigger, dynamic_weights = safe_prediction_from_selection(
            comp, outer_data, selected
        )
        family_predictions = {}
        for family, family_config in selected["family_best_methods"].items():
            family_predictions[family], _, _ = method_prediction(
                comp["risk"],
                outer_data["predictions"],
                family,
                family_config["tau"],
                family_config["margin"],
            )
        common = selected["common_configuration"]
        common_region = {
            "space": common["space"],
            "distance": common["distance"],
            "K": int(common["K"]),
            "weighting": common["weighting"],
            "beta": None,
            "config_id": config_id(
                common["space"],
                common["distance"],
                int(common["K"]),
                common["weighting"],
                None,
            ),
        }
        if common_region["config_id"] == config["config_id"]:
            common_comp = comp
        else:
            common_comp, _, _ = compute_config(
                train, outer_data, stats, common_region, save_neighbors=False
            )
        common_dynamic, common_dynamic_risk, _ = method_prediction(
            common_comp["risk"],
            outer_data["predictions"],
            "Local-DW",
            float(common["tau"]),
            None,
        )
        common_estimated_gain = common_comp["static_risk"] - common_dynamic_risk
        common_coverage_outputs, _ = fixed_coverage_predictions(
            outer_data["sidecar"]["strong_static"].to_numpy(dtype=float),
            common_dynamic,
            common_estimated_gain,
        )
        common_safe = common_coverage_outputs[
            "coverage_{:.0f}".format(float(common["safe_coverage"]) * 100)
        ]
        selected_outputs[direction] = {
            "dynamic": dynamic,
            "safe": safe,
            "estimated_gain": estimated_gain,
            "trigger": trigger,
            "comp": comp,
        }
        neighbor_path = output / "direction_{}_selected_outer_neighbors.csv.gz".format(direction)
        atomic_gzip_csv(neighbors, neighbor_path)
        frequency, source_summary = source_audit(neighbors, density, direction, "outer_evaluation")
        neighbor_frequency_pieces.append(frequency)
        source_summaries.append(source_summary)
        density["direction"] = direction
        density["role"] = "outer_evaluation"
        density_pieces.append(density)
        coverage_predictions, _ = fixed_coverage_predictions(
            outer_data["sidecar"]["strong_static"].to_numpy(dtype=float),
            dynamic,
            estimated_gain,
        )
        control_table, control_predictions, control_risks = control_outputs(
            direction, train, outer_data, stats, config,
            selected["selected_method"], selected["safety"], "outer_evaluation"
        )
        pred = outer_data["sidecar"][["meta_row_id", "sample_id", "video_id", "mode", "expert_fold"]].copy()
        pred["direction"] = direction
        pred["strong_static"] = outer_data["sidecar"]["strong_static"].to_numpy(dtype=float)
        pred["selected_local_dynamic"] = dynamic
        pred["selected_safe_local"] = safe
        pred["estimated_static_local_risk"] = comp["static_risk"]
        pred["estimated_dynamic_local_risk"] = dynamic_risk
        pred["estimated_gain"] = estimated_gain
        pred["selected_safe_trigger"] = trigger.astype(int)
        for family, values in family_predictions.items():
            pred["drs__{}".format(family)] = values
        pred["common_fixed_safe"] = common_safe
        for name, values in coverage_predictions.items():
            pred["selected_{}".format(name)] = values
        for name, values in control_predictions.items():
            column = "control__{}".format(name)
            pred[column] = values
            control_prediction_names[direction].append(column)
        selected_outputs[direction]["control_risks"] = control_risks
        prediction_pieces.append(pred)
        local = pred[["meta_row_id", "sample_id", "video_id", "mode", "direction"]].copy()
        for expert_index, expert in enumerate(EXPERTS):
            for key in ("risk", "median_error", "regret", "soft_win_rate", "error_variance", "effective_neighbor_count"):
                local["{}__{}".format(key, expert)] = comp[key][:, expert_index]
        local_comp_pieces.append(local)
    frozen_predictions = pd.concat(prediction_pieces, ignore_index=True)
    frozen_path = output / "outer_predictions_frozen_before_label_evaluation.csv.gz"
    atomic_gzip_csv(frozen_predictions, frozen_path)
    prediction_sha = sha256_file(frozen_path)
    access_audit = {
        "stage": "Stage23A-v2b outer one-shot access",
        "status": "PREDICTIONS_FROZEN_BEFORE_LABEL_METRICS",
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "prediction_path": str(frozen_path.resolve()),
        "prediction_sha256": prediction_sha,
        "prediction_rows": int(len(frozen_predictions)),
        "outer_evaluation_access_count": 1,
        "outer_labels_used_for_retrieval_or_prediction": False,
        "outer_labels_used_after_prediction_freeze_for_metrics_only": True,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "frozen_at": utc_now(),
    }
    atomic_json(output / "outer_one_shot_access_audit.json", access_audit)
    state["status"] = "OUTER_PREDICTIONS_FROZEN_METRICS_AUDIT_IN_PROGRESS"
    state["outer_evaluation_access_count"] = 1
    state["outer_prediction_sha256"] = prediction_sha
    atomic_json(state_path, state)

    # Labels are consulted only below this line, after every prediction column is frozen.
    metric_rows = []
    regularity_rows = []
    control_rows = []
    for direction in ("A", "B"):
        outer_data = outer_cache[direction]
        attach_outer_labels(direction, outer_data)
        pred = frozen_predictions[frozen_predictions["direction"] == direction].reset_index(drop=True)
        if pred["meta_row_id"].tolist() != outer_data["sidecar"]["meta_row_id"].tolist():
            raise RuntimeError("Frozen prediction/outer label binding mismatch.")
        methods = {
            "strong_static": pred["strong_static"].to_numpy(),
            "selected_local_dynamic": pred["selected_local_dynamic"].to_numpy(),
            "selected_safe_local": pred["selected_safe_local"].to_numpy(),
            "common_fixed_safe": pred["common_fixed_safe"].to_numpy(),
        }
        for family in ("Local-DS", "Local-DW", "Local-DWS"):
            methods["drs__{}".format(family)] = pred[
                "drs__{}".format(family)
            ].to_numpy()
        for coverage in COVERAGES:
            name = "selected_coverage_{:.0f}".format(coverage * 100)
            methods[name] = pred[name].to_numpy()
        for column in control_prediction_names[direction]:
            methods[column] = pred[column].to_numpy()
        local_metrics = prediction_metric_rows(outer_data["sidecar"], methods)
        for row in local_metrics:
            row["direction"] = direction
        metric_rows.extend(local_metrics)
        local_coverage, _ = risk_coverage_rows(
            direction,
            outer_data,
            pred["strong_static"].to_numpy(dtype=float),
            pred["selected_local_dynamic"].to_numpy(dtype=float),
            pred["estimated_gain"].to_numpy(dtype=float),
        )
        coverage_pieces.append(local_coverage)
        comp = selected_outputs[direction]["comp"]
        regularity = regularity_metrics(
            comp["risk"], comp["regret"], outer_data["errors"], outer_data["regrets"]
        )
        regularity_rows.append({"direction": direction, "method": "selected_true_neighborhood", **regularity})
        for control_name, (risk, regret) in selected_outputs[direction][
            "control_risks"
        ].items():
            regularity_rows.append(
                {
                    "direction": direction,
                    "method": control_name,
                    **regularity_metrics(
                        risk, regret, outer_data["errors"], outer_data["regrets"]
                    ),
                }
            )
        for control_column in control_prediction_names[direction]:
            overall = next(
                row for row in local_metrics
                if row["method"] == control_column and row["mode"] == "Overall"
            )
            control_rows.append({"direction": direction, "control": control_column, "J": overall["J"]})
    metrics = pd.DataFrame(metric_rows)
    regularity = pd.DataFrame(regularity_rows)
    controls = pd.DataFrame(control_rows)
    coverage = pd.concat(coverage_pieces, ignore_index=True)
    atomic_tsv(metrics, output / "outer_method_metrics.tsv")
    atomic_tsv(regularity, output / "outer_local_regularity.tsv")
    atomic_tsv(controls, output / "outer_negative_control_metrics.tsv")
    atomic_tsv(coverage, output / "outer_risk_coverage.tsv")
    atomic_tsv(pd.concat(neighbor_frequency_pieces, ignore_index=True), output / "source_neighbor_frequency.tsv")
    atomic_tsv(pd.DataFrame(source_summaries), output / "source_dominance_summary.tsv")
    atomic_tsv(pd.concat(density_pieces, ignore_index=True), output / "outer_neighborhood_density.tsv")
    atomic_gzip_csv(pd.concat(local_comp_pieces, ignore_index=True), output / "outer_local_competence_ledger.csv.gz")
    svg_risk_coverage(coverage, output / "risk_coverage.svg")

    gate, conclusion = promotion_gate(metrics, regularity, controls, coverage)
    final = {
        "stage": "Stage23A-v2b Source-Aware Local Competence Audit",
        "status": conclusion,
        "promotion_gate": gate,
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "outer_prediction_path": str(frozen_path.resolve()),
        "outer_prediction_sha256": prediction_sha,
        "outer_evaluation_access_count": 1,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "feature_rich_judge_trained": False,
        "student_trained": False,
        "completed_at": utc_now(),
    }
    final_dir = V2B / "final"
    atomic_json(final_dir / "stage23a_v2b_audit.json", final)
    write_report(final, selection, metrics, regularity, controls, coverage, source_summaries)
    lock = {
        "stage": "Stage23A-v2b",
        "status": conclusion,
        "post_hoc_judge_route_permanently_closed": conclusion == "LOCAL_COMPETENCE_FAIL",
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "outer_evaluation_access_count": 1,
        "feature_rich_judge_trained": False,
        "student_trained": False,
        "expert_retrained_or_modified": False,
        "next_route": "jointly trained structurally specialized Experts" if conclusion == "LOCAL_COMPETENCE_FAIL" else "await explicit user decision",
    }
    atomic_json(final_dir / "TEST_LOCK_STATUS.json", lock)
    state.update(
        {
            "status": conclusion,
            "outer_evaluation_access_count": 1,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "feature_rich_judge_trained": False,
            "student_trained": False,
            "next_action": lock["next_route"],
            "updated_at": utc_now(),
        }
    )
    atomic_json(state_path, state)
    print(json.dumps(final, indent=2))


def promotion_gate(metrics, regularity, controls, coverage):
    overall = metrics[metrics["mode"] == "Overall"]
    selected_regularity = regularity[
        regularity["method"] == "selected_true_neighborhood"
    ].copy()
    deltas = {}
    for direction in ("A", "B"):
        local = overall[overall["direction"] == direction].set_index("method")
        deltas[direction] = float(local.loc["selected_safe_local", "J"] - local.loc["strong_static", "J"])
    mean_delta = float(np.mean(list(deltas.values())))
    worst = float(max(deltas.values()))
    strongest_control = {}
    for direction in ("A", "B"):
        local_random = controls[
            (controls["direction"] == direction)
            & controls["control"].str.contains("N0_random|N1_shuffled")
        ]
        strongest_control[direction] = float(local_random["J"].min())
    selected_j = {
        direction: float(
            overall[(overall["direction"] == direction) & (overall["method"] == "selected_safe_local")]["J"].iloc[0]
        )
        for direction in ("A", "B")
    }
    control_delta = float(np.mean([selected_j[d] - strongest_control[d] for d in ("A", "B")]))
    missing_improved = 0
    for mode in ("LA", "LV", "L"):
        mode_deltas = []
        for direction in ("A", "B"):
            local = metrics[(metrics["direction"] == direction) & (metrics["mode"] == mode)].set_index("method")
            mode_deltas.append(local.loc["selected_safe_local", "MAE"] - local.loc["strong_static", "MAE"])
        if np.mean(mode_deltas) < 0:
            missing_improved += 1
    corr_deltas = []
    classification_deltas = defaultdict(list)
    for direction in ("A", "B"):
        local = overall[overall["direction"] == direction].set_index("method")
        corr_deltas.append(local.loc["selected_safe_local", "Corr"] - local.loc["strong_static", "Corr"])
        for metric in ("Acc7", "Acc5", "Acc2", "F1"):
            classification_deltas[metric].append(local.loc["selected_safe_local", metric] - local.loc["strong_static", metric])
    top10 = coverage[np.isclose(coverage["coverage"], 0.10)]
    high_confidence = bool((top10["triggered_true_gain_mean"] > 0).all())
    regularity_consistent = bool(
        len(selected_regularity) == 2
        and (selected_regularity["spearman_local_risk_vs_error"] > 0).all()
        and (selected_regularity["spearman_local_regret_vs_regret"] > 0).all()
        and all(value <= 0.001 for value in deltas.values())
    )
    global_mode_best = {}
    for direction in ("A", "B"):
        local_controls = controls[
            (controls["direction"] == direction)
            & controls["control"].str.contains("N2_global|N3_mode")
        ]
        global_mode_best[direction] = float(local_controls["J"].min())
    not_global_mode = bool(np.mean([selected_j[d] - global_mode_best[d] for d in ("A", "B")]) < 0)
    gate = {
        "mean_delta_J_vs_strong_static": mean_delta,
        "mean_delta_J_gate_pass": mean_delta <= -0.003,
        "direction_delta_J": deltas,
        "worst_direction_delta_J": worst,
        "worst_direction_gate_pass": worst <= 0.001,
        "mean_delta_J_vs_strongest_random_or_shuffled": control_delta,
        "control_gate_pass": control_delta <= -0.002,
        "missing_modes_improved": missing_improved,
        "missing_mode_gate_pass": missing_improved >= 2,
        "corr_delta_by_direction": corr_deltas,
        "corr_gate_pass": min(corr_deltas) >= -0.02 and np.mean(corr_deltas) >= -0.01,
        "classification_mean_deltas": {key: float(np.mean(value)) for key, value in classification_deltas.items()},
        "classification_gate_pass": sum(np.mean(value) < -0.005 for value in classification_deltas.values()) <= 1,
        "high_confidence_trigger_gate_pass": high_confidence,
        "direction_consistency_gate_pass": regularity_consistent,
        "not_global_or_mode_only_gate_pass": not_global_mode,
        "outer_one_shot_gate_pass": True,
    }
    required = [value for key, value in gate.items() if key.endswith("_gate_pass")]
    if all(required):
        conclusion = "LOCAL_COMPETENCE_PASS"
    elif (
        mean_delta >= 0
        or control_delta >= 0
        or len(selected_regularity) != 2
        or not (selected_regularity["spearman_local_risk_vs_error"] > 0).all()
    ):
        conclusion = "LOCAL_COMPETENCE_FAIL"
    else:
        conclusion = "LOCAL_COMPETENCE_WEAK"
    return gate, conclusion


def markdown_table(frame, columns):
    display = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else "{:.6f}".format(value))
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        + ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    )


def write_report(final, selection, metrics, regularity, controls, coverage, source_summaries):
    overall = metrics[
        (metrics["mode"] == "Overall")
        & metrics["method"].isin(["strong_static", "selected_local_dynamic", "selected_safe_local"])
    ][["direction", "method", "J", "MAE", "Corr", "Acc7", "Acc5", "Acc2", "F1"]]
    report = """# Stage23A-v2b Source-Aware Local Competence Audit

## Conclusion

**{status}**

Stage23A-v2a remains `SIGNAL_AUDIT_FAIL`; this audit does not revise it.
The formal Feature-Rich Judge, Official Valid, Test, Experts and Student all
remained locked.

## Frozen choices

```json
{choices}
```

## Outer one-shot core metrics

{metrics}

## Local regularity

{regularity}

## Promotion gate

```json
{gate}
```

## Source-aware neighborhood summary

```json
{sources}
```

## Plain-language questions

1. Local regularity is judged by held-out risk/regret Spearman and ranking, not
   by Oracle space.
2. The winning region is recorded independently for Directions A/B above.
3. True neighborhoods are compared against all random/shuffled controls; the
   gate uses the strongest negative control.
4. DS/DW/DWS were selected only by inner-valid J.
5. Safe deferral is audited at 10/20/30/50/100% coverage and by one frozen
   inner-valid threshold.
6. Direction agreement is an explicit gate.
7. Final promotion status is `{status}`.
8. Density and source dominance distinguish lack of regularity from sparse or
   monopolized neighborhoods.
9. If FAIL, the frozen protocol permanently closes post-hoc Judge work on this
   Expert pool and redirects work to structurally specialized joint training.
""".format(
        status=final["status"],
        choices=json.dumps(selection["directions"], indent=2, ensure_ascii=False),
        metrics=markdown_table(overall, list(overall.columns)),
        regularity=markdown_table(regularity, list(regularity.columns)),
        gate=json.dumps(
            final["promotion_gate"],
            indent=2,
            ensure_ascii=False,
            default=lambda item: item.item()
            if isinstance(item, np.generic)
            else str(item),
        ),
        sources=json.dumps(
            source_summaries,
            indent=2,
            ensure_ascii=False,
            default=lambda item: item.item()
            if isinstance(item, np.generic)
            else str(item),
        ),
    )
    path = V2B / "final" / "stage23a_v2b_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def finalize_from_saved_outer_metrics():
    """Finalize after the one-shot predictions/metrics were already frozen.

    This recovery path intentionally reads only saved aggregate outputs.  It
    does not construct an outer loader or reopen any query Ground Truth.
    """
    state_path = RUNTIME / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state["status"] != "OUTER_PREDICTIONS_FROZEN_METRICS_AUDIT_IN_PROGRESS"
        or state["outer_evaluation_access_count"] != 1
    ):
        raise RuntimeError("Saved-metric finalization is not authorized.")
    output = V2B / "outer"
    required = {
        "metrics": output / "outer_method_metrics.tsv",
        "regularity": output / "outer_local_regularity.tsv",
        "controls": output / "outer_negative_control_metrics.tsv",
        "coverage": output / "outer_risk_coverage.tsv",
        "sources": output / "source_dominance_summary.tsv",
        "predictions": output
        / "outer_predictions_frozen_before_label_evaluation.csv.gz",
        "access_audit": output / "outer_one_shot_access_audit.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError("Missing frozen outer artifacts: {}".format(missing))
    if sha256_file(required["predictions"]) != state["outer_prediction_sha256"]:
        raise RuntimeError("Frozen outer prediction SHA mismatch.")
    access_audit = json.loads(
        required["access_audit"].read_text(encoding="utf-8")
    )
    if (
        access_audit["outer_evaluation_access_count"] != 1
        or access_audit["official_valid_access_count"] != 0
        or access_audit["locked_test_access_count"] != 0
    ):
        raise RuntimeError("Outer one-shot access audit is inconsistent.")
    metrics = pd.read_csv(required["metrics"], sep="\t")
    regularity = pd.read_csv(required["regularity"], sep="\t")
    controls = pd.read_csv(required["controls"], sep="\t")
    coverage = pd.read_csv(required["coverage"], sep="\t")
    source_summaries = pd.read_csv(required["sources"], sep="\t").to_dict(
        orient="records"
    )
    selection_path = V2B / "protocol" / "frozen_development_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    gate, conclusion = promotion_gate(metrics, regularity, controls, coverage)
    final = {
        "stage": "Stage23A-v2b Source-Aware Local Competence Audit",
        "status": conclusion,
        "promotion_gate": gate,
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "outer_prediction_path": str(required["predictions"].resolve()),
        "outer_prediction_sha256": state["outer_prediction_sha256"],
        "outer_evaluation_access_count": 1,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "feature_rich_judge_trained": False,
        "student_trained": False,
        "completed_at": utc_now(),
        "finalized_from_saved_metrics_without_outer_reload": True,
    }
    final_dir = V2B / "final"
    atomic_json(final_dir / "stage23a_v2b_audit.json", final)
    write_report(
        final,
        selection,
        metrics,
        regularity,
        controls,
        coverage,
        source_summaries,
    )
    lock = {
        "stage": "Stage23A-v2b",
        "status": conclusion,
        "post_hoc_judge_route_permanently_closed": conclusion
        == "LOCAL_COMPETENCE_FAIL",
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "outer_evaluation_access_count": 1,
        "feature_rich_judge_trained": False,
        "student_trained": False,
        "expert_retrained_or_modified": False,
        "next_route": "jointly trained structurally specialized Experts"
        if conclusion == "LOCAL_COMPETENCE_FAIL"
        else "await explicit user decision",
    }
    atomic_json(final_dir / "TEST_LOCK_STATUS.json", lock)
    state.update(
        {
            "status": conclusion,
            "outer_evaluation_access_count": 1,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "feature_rich_judge_trained": False,
            "student_trained": False,
            "next_action": lock["next_route"],
            "finalized_from_saved_metrics_without_outer_reload": True,
            "updated_at": utc_now(),
        }
    )
    atomic_json(state_path, state)
    print(json.dumps(final, indent=2, default=lambda item: item.item()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True, choices=("develop", "outer", "finalize-saved")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "develop":
        development()
    elif args.phase == "outer":
        outer()
    else:
        finalize_from_saved_outer_metrics()
