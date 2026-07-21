"""Stage 21A frozen-representation fingerprint and controlled ridge probes."""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage21a_common import (
    MISSING_MODES,
    MODES,
    atomic_frame,
    atomic_json,
    balanced_within_pairs,
    bootstrap_j_delta,
    evaluate_modes,
    fit_shared_ridge,
    gap_match_from_candidates,
    group_indices,
    metric_delta,
    nearest_cross_source_pairs,
    ordered_id_sha,
    pair_matching_diagnostics,
    pseudo_membership,
    ridge_predict,
    sha256_file,
    standardization,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def index_from_manifest(manifest, sample_ids, key):
    by_id = {sample_id: index for index, sample_id in enumerate(sample_ids.tolist())}
    values = manifest[key]
    try:
        result = np.asarray([by_id[value] for value in values], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError("Split manifest sample is absent from frozen cache: {}".format(error))
    if ordered_id_sha(sample_ids[result]) != manifest[key.replace("sample_ids", "sample_order_sha256")]:
        raise RuntimeError("Split manifest order SHA mismatch.")
    return result


def rows_for_indices(frame, allowed):
    allowed = set(np.asarray(allowed, dtype=np.int64).tolist())
    result = []
    for value in frame.to_dict("records"):
        left, right = int(value["left_index"]), int(value["right_index"])
        if left in allowed and right in allowed:
            result.append(
                {
                    "video_id": str(value["video_id"]),
                    "left_index": left,
                    "right_index": right,
                    "target_difference": float(value["target_difference"]),
                    "absolute_gap": float(value["absolute_gap"]),
                    "raw_video_balanced_weight": float(value["raw_video_balanced_weight"]),
                }
            )
    return result


def prediction_bundle(models, representations):
    return {
        probe: {
            mode: ridge_predict(model, representations[mode])
            for mode in MODES
        }
        for probe, model in models.items()
    }


def remove_largest_source_indices(video_ids, indices):
    groups = group_indices(video_ids, indices)
    remove_count = max(1, int(math.ceil(0.05 * len(groups))))
    removed = set(sorted(groups, key=lambda key: len(groups[key]), reverse=True)[:remove_count])
    kept = np.asarray([index for index in indices if video_ids[index] not in removed], dtype=np.int64)
    return kept, sorted(removed)


def pair_features(representation, rows):
    left = np.asarray([row["left_index"] for row in rows], dtype=np.int64)
    right = np.asarray([row["right_index"] for row in rows], dtype=np.int64)
    first = np.asarray(representation, dtype=np.float64)[left]
    second = np.asarray(representation, dtype=np.float64)[right]
    difference = np.abs(first - second)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine_distance = 1.0 - np.sum(first * second, axis=1) / np.maximum(denominator, 1e-12)
    return np.column_stack(
        [
            cosine_distance,
            np.linalg.norm(first - second, axis=1),
            difference.mean(axis=1),
            difference.std(axis=1),
            difference.max(axis=1),
        ]
    )


def fingerprint_for_partition(representation, video_ids, labels, train_index, valid_index, seed):
    same_train = balanced_within_pairs(video_ids, labels, 0.0, seed=seed, max_pairs=8, indices=train_index)
    same_valid = balanced_within_pairs(video_ids, labels, 0.0, seed=seed + 1, max_pairs=8, indices=valid_index)
    rng = np.random.RandomState(seed + 2)
    if len(same_train) > 5000:
        same_train = [same_train[index] for index in rng.choice(len(same_train), 5000, replace=False)]
    if len(same_valid) > 3000:
        same_valid = [same_valid[index] for index in rng.choice(len(same_valid), 3000, replace=False)]
    diff_train = nearest_cross_source_pairs(same_train, train_index, video_ids, labels, seed + 3)
    diff_valid = nearest_cross_source_pairs(same_valid, valid_index, video_ids, labels, seed + 4)
    x_train = np.concatenate([pair_features(representation, same_train), pair_features(representation, diff_train)])
    y_train = np.concatenate([np.ones(len(same_train)), np.zeros(len(diff_train))])
    x_valid = np.concatenate([pair_features(representation, same_valid), pair_features(representation, diff_valid)])
    y_valid = np.concatenate([np.ones(len(same_valid)), np.zeros(len(diff_valid))])
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std
    classifier = LogisticRegression(C=0.1, solver="liblinear", random_state=seed, max_iter=500)
    classifier.fit(x_train, y_train)
    probability = classifier.predict_proba(x_valid)[:, 1]
    prediction = probability >= 0.5
    shuffled = y_train.copy()
    rng.shuffle(shuffled)
    shuffled_classifier = LogisticRegression(C=0.1, solver="liblinear", random_state=seed, max_iter=500)
    shuffled_classifier.fit(x_train, shuffled)
    shuffled_probability = shuffled_classifier.predict_proba(x_valid)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_valid, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, prediction)),
        "shuffled_train_label_auroc": float(roc_auc_score(y_valid, shuffled_probability)),
        "train_pair_count_per_class": int(len(same_train)),
        "valid_pair_count_per_class": int(len(same_valid)),
        "feature_dimension": 5,
        "source_disjoint": True,
    }


def fingerprint_audit(representations, video_ids, labels, split_specs):
    rows = []
    for seed, (train_index, valid_index) in split_specs.items():
        for mode_position, mode in enumerate(MODES):
            result = fingerprint_for_partition(
                representations[mode], video_ids, labels, train_index, valid_index, seed + 100 * mode_position
            )
            rows.append({"split_seed": seed, "mode": mode, **result})
    return rows


def run_one_split(seed, train_index, valid_index, representations, labels, video_ids, p1_rows, iterations):
    standard = standardization(representations, labels, train_index)
    p2_rows = nearest_cross_source_pairs(p1_rows, train_index, video_ids, labels, seed + 200)
    pseudo = pseudo_membership(video_ids, train_index, seed + 300)
    pseudo_candidates = balanced_within_pairs(
        pseudo, labels, min(row["absolute_gap"] for row in p1_rows), seed=seed + 301, max_pairs=64, indices=train_index
    )
    p3_rows = gap_match_from_candidates(p1_rows, pseudo_candidates)
    p2_diagnostic = pair_matching_diagnostics(p1_rows, p2_rows)
    p3_diagnostic = pair_matching_diagnostics(p1_rows, p3_rows)
    models = {
        "P0": fit_shared_ridge(representations, labels, train_index, [], standard),
        "P1": fit_shared_ridge(representations, labels, train_index, p1_rows, standard),
        "P2": fit_shared_ridge(representations, labels, train_index, p2_rows, standard),
        "P3": fit_shared_ridge(representations, labels, train_index, p3_rows, standard),
    }
    predictions = prediction_bundle(models, representations)
    metrics = {probe: evaluate_modes(prediction, labels, valid_index) for probe, prediction in predictions.items()}
    deltas = {
        "P1_minus_P0": metric_delta(metrics["P1"], metrics["P0"]),
        "P1_minus_P2": metric_delta(metrics["P1"], metrics["P2"]),
        "P1_minus_P3": metric_delta(metrics["P1"], metrics["P3"]),
        "P2_minus_P0": metric_delta(metrics["P2"], metrics["P0"]),
        "P3_minus_P0": metric_delta(metrics["P3"], metrics["P0"]),
    }
    bootstraps = {}
    bootstrap_values = {}
    for position, reference in enumerate(("P0", "P2", "P3")):
        key = "P1_minus_{}".format(reference)
        bootstraps[key], bootstrap_values[key] = bootstrap_j_delta(
            predictions["P1"], predictions[reference], labels, video_ids, valid_index,
            iterations=iterations, seed=seed + 400 + position,
        )
    kept_index, removed_sources = remove_largest_source_indices(video_ids, valid_index)
    without_largest = {
        probe: evaluate_modes(predictions[probe], labels, kept_index)
        for probe in ("P0", "P1", "P2", "P3")
    }
    trim_threshold = float(np.quantile([row["absolute_gap"] for row in p1_rows], 0.95))
    trimmed_rows = [row for row in p1_rows if row["absolute_gap"] <= trim_threshold]
    trimmed_model = fit_shared_ridge(representations, labels, train_index, trimmed_rows, standard)
    trimmed_prediction = {mode: ridge_predict(trimmed_model, representations[mode]) for mode in MODES}
    trimmed_metrics = evaluate_modes(trimmed_prediction, labels, valid_index)
    return {
        "seed": seed,
        "train_sample_count": int(len(train_index)),
        "valid_sample_count": int(len(valid_index)),
        "source_overlap_count": 0,
        "pair_counts": {"P1": len(p1_rows), "P2": len(p2_rows), "P3": len(p3_rows)},
        "pair_matching": {"P2": p2_diagnostic, "P3": p3_diagnostic},
        "metrics": metrics,
        "deltas": deltas,
        "bootstrap": bootstraps,
        "bootstrap_values": bootstrap_values,
        "without_largest_sources": {
            "removed_sources": removed_sources,
            "metrics": without_largest,
            "P1_minus_P0": metric_delta(without_largest["P1"], without_largest["P0"]),
        },
        "without_largest_label_gap_pairs": {
            "gap_threshold": trim_threshold,
            "retained_pair_count": len(trimmed_rows),
            "metrics": trimmed_metrics,
            "P1_trimmed_minus_P0": metric_delta(trimmed_metrics, metrics["P0"]),
        },
        "predictions": predictions,
    }, p2_rows, p3_rows


def mean_delta(results, key, field_path):
    values = []
    for result in results:
        value = result["deltas"][key]
        for field in field_path:
            value = value[field]
        values.append(float(value))
    return float(np.mean(values))


def train_gate(results, residual_gate, pair_controls_valid):
    p10 = [result["deltas"]["P1_minus_P0"]["J"] for result in results]
    p12 = [result["deltas"]["P1_minus_P2"]["J"] for result in results]
    p13 = [result["deltas"]["P1_minus_P3"]["J"] for result in results]
    missing_improved = [
        mode
        for mode in MISSING_MODES
        if mean_delta(results, "P1_minus_P0", (mode, "MAE")) < 0
    ]
    classification_safe = all(
        mean_delta(results, "P1_minus_P0", (mode, metric)) >= -0.003
        for mode in MODES + ("MissingMacro",)
        for metric in ("acc_7", "acc_5", "acc_2", "F1_score")
    )
    checks = {
        "mean_P1_minus_P0_J_le_minus_0_003": np.mean(p10) <= -0.003,
        "worst_P1_minus_P0_J_le_plus_0_001": max(p10) <= 0.001,
        "mean_P1_minus_P2_J_le_minus_0_002": np.mean(p12) <= -0.002,
        "mean_P1_minus_P3_J_le_minus_0_002": np.mean(p13) <= -0.002,
        "P1_minus_controls_direction_consistent": all(value < 0 for value in p12 + p13),
        "missing_macro_mae_improved": mean_delta(results, "P1_minus_P0", ("MissingMacro", "MAE")) < 0,
        "at_least_two_missing_modes_mae_improved": len(missing_improved) >= 2,
        "lav_mae_safe": mean_delta(results, "P1_minus_P0", ("LAV", "MAE")) <= 0.003,
        "lav_corr_safe": mean_delta(results, "P1_minus_P0", ("LAV", "Corr")) >= -0.002,
        "missing_macro_corr_safe": mean_delta(results, "P1_minus_P0", ("MissingMacro", "Corr")) >= -0.002,
        "classification_safe": classification_safe,
        "bootstrap_probability_ge_0_90": all(
            result["bootstrap"]["P1_minus_P0"]["probability_a_better"] >= 0.90 for result in results
        ),
        "direction_survives_largest_sources": all(
            result["without_largest_sources"]["P1_minus_P0"]["J"] < 0 for result in results
        ),
        "source_residual_gate_passed": bool(residual_gate),
        "pair_controls_valid": bool(pair_controls_valid),
    }
    return {
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": bool(all(checks.values())),
        "P1_minus_P0_J": p10,
        "P1_minus_P2_J": p12,
        "P1_minus_P3_J": p13,
        "mean_P1_minus_P0_J": float(np.mean(p10)),
        "mean_P1_minus_P2_J": float(np.mean(p12)),
        "mean_P1_minus_P3_J": float(np.mean(p13)),
        "missing_modes_mae_improved": missing_improved,
    }


def full_train_official_valid(output, train, valid, representations, train_pairs, gate, iterations):
    path = output / "probes/official_valid_results.tsv"
    if not gate["passed"]:
        atomic_frame(
            path,
            [{"OfficialValidRun": False, "Reason": "Train-only promotion gate failed; no Official Valid probe evaluation."}],
        )
        return {"run": False, "reason": "Train-only promotion gate failed."}
    train_indices = np.arange(len(train["label"]), dtype=np.int64)
    standard = standardization(representations, train["label"], train_indices)
    p2 = nearest_cross_source_pairs(train_pairs, train_indices, train["video_id"], train["label"], 2191)
    pseudo = pseudo_membership(train["video_id"], train_indices, 2192)
    pseudo_candidates = balanced_within_pairs(pseudo, train["label"], 1.0, seed=2193, max_pairs=64, indices=train_indices)
    p3 = gap_match_from_candidates(train_pairs, pseudo_candidates)
    models = {
        "P0": fit_shared_ridge(representations, train["label"], train_indices, [], standard),
        "P1": fit_shared_ridge(representations, train["label"], train_indices, train_pairs, standard),
        "P2": fit_shared_ridge(representations, train["label"], train_indices, p2, standard),
        "P3": fit_shared_ridge(representations, train["label"], train_indices, p3, standard),
    }
    valid_representations = {mode: valid["primary_{}".format(mode)] for mode in MODES}
    predictions = prediction_bundle(models, valid_representations)
    valid_indices = np.arange(len(valid["label"]), dtype=np.int64)
    metrics = {probe: evaluate_modes(prediction, valid["label"], valid_indices) for probe, prediction in predictions.items()}
    deltas = {
        "P1_minus_P0": metric_delta(metrics["P1"], metrics["P0"]),
        "P1_minus_P2": metric_delta(metrics["P1"], metrics["P2"]),
        "P1_minus_P3": metric_delta(metrics["P1"], metrics["P3"]),
    }
    rows = []
    for probe in ("P0", "P1", "P2", "P3"):
        for mode in MODES + ("MissingMacro",):
            rows.append({"Probe": probe, "Mode": mode, "J": metrics[probe]["J"], **metrics[probe][mode]})
    atomic_frame(path, rows)
    bootstrap = {}
    for position, reference in enumerate(("P0", "P2", "P3")):
        bootstrap["P1_minus_{}".format(reference)], _ = bootstrap_j_delta(
            predictions["P1"], predictions[reference], valid["label"], valid["video_id"], valid_indices,
            iterations=iterations, seed=2194 + position,
        )
    missing_improved = sum(deltas["P1_minus_P0"][mode]["MAE"] < 0 for mode in MISSING_MODES)
    safety = {
        "P1_minus_P0_J_le_minus_0_003": deltas["P1_minus_P0"]["J"] <= -0.003,
        "P1_minus_P2_J_le_minus_0_002": deltas["P1_minus_P2"]["J"] <= -0.002,
        "P1_minus_P3_J_le_minus_0_002": deltas["P1_minus_P3"]["J"] <= -0.002,
        "missing_macro_mae_improved": deltas["P1_minus_P0"]["MissingMacro"]["MAE"] < 0,
        "at_least_two_missing_modes_mae_improved": missing_improved >= 2,
        "corr_safe": deltas["P1_minus_P0"]["LAV"]["Corr"] >= -0.002 and deltas["P1_minus_P0"]["MissingMacro"]["Corr"] >= -0.002,
        "classification_safe": all(
            deltas["P1_minus_P0"][mode][metric] >= -0.003
            for mode in MODES + ("MissingMacro",)
            for metric in ("acc_7", "acc_5", "acc_2", "F1_score")
        ),
        "bootstrap_controls_probability_ge_0_90": all(
            bootstrap[key]["probability_a_better"] >= 0.90 for key in bootstrap
        ),
    }
    return {
        "run": True,
        "metrics": metrics,
        "deltas": deltas,
        "bootstrap": bootstrap,
        "gate_checks": {key: bool(value) for key, value in safety.items()},
        "passed": bool(all(safety.values())),
    }


def main():
    cli = parse_args()
    output = Path(cli.output_root)
    train = load_npz(output / "representations/train_cache.npz")
    representations = {mode: train["primary_{}".format(mode)] for mode in MODES}
    sample_ids = train["sample_id"].astype(str)
    labels = train["label"].astype(np.float64)
    video_ids = train["video_id"].astype(str)
    global_pairs_frame = pd.read_csv(output / "pairs/within_video_pair_manifest.tsv", sep="\t")
    split_specs = {}
    split_manifests = {}
    for seed in (2101, 2102):
        manifest = json.loads((output / "splits/split_{}_manifest.json".format(seed)).read_text())
        train_index = index_from_manifest(manifest, sample_ids, "inner_train_sample_ids")
        valid_index = index_from_manifest(manifest, sample_ids, "inner_valid_sample_ids")
        if set(video_ids[train_index]) & set(video_ids[valid_index]):
            raise RuntimeError("Source-disjoint split was violated.")
        split_specs[seed] = (train_index, valid_index)
        split_manifests[seed] = manifest
    fingerprint_rows = fingerprint_audit(representations, video_ids, labels, split_specs)
    split_results = []
    matching_rows = []
    bootstrap_rows = []
    for seed in (2101, 2102):
        train_index, valid_index = split_specs[seed]
        p1_rows = rows_for_indices(global_pairs_frame, train_index)
        if not p1_rows:
            raise RuntimeError("Inner-train source pair set is empty.")
        result, p2_rows, p3_rows = run_one_split(
            seed, train_index, valid_index, representations, labels, video_ids,
            p1_rows, cli.bootstrap_iterations,
        )
        for control, diagnostic in result["pair_matching"].items():
            matching_rows.append(
                {
                    "SplitSeed": seed,
                    "Control": control,
                    "PairCount": diagnostic["pair_count_second"],
                    "PositiveFractionP1": diagnostic["positive_fraction_first"],
                    "PositiveFractionControl": diagnostic["positive_fraction_second"],
                    "KS": diagnostic["ks_statistic"],
                    "Wasserstein": diagnostic["wasserstein_distance"],
                    "MaxQuantileDifference": diagnostic["maximum_quantile_difference"],
                }
            )
        for comparison, values in result.pop("bootstrap_values").items():
            for iteration, value in enumerate(values):
                bootstrap_rows.append(
                    {"SplitSeed": seed, "Comparison": comparison, "Iteration": iteration, "DeltaJ": float(value)}
                )
        # Predictions are intentionally not serialized in the metric JSON.
        result.pop("predictions")
        split_results.append(result)
    pair_controls_valid = all(
        row["PairCount"] > 0
        and abs(row["PositiveFractionP1"] - row["PositiveFractionControl"]) <= 1e-12
        and row["KS"] <= 0.05
        and row["Wasserstein"] <= 0.05
        and row["MaxQuantileDifference"] <= 0.05
        for row in matching_rows
    )
    residual = json.loads((output / "residual/source_residual_icc.json").read_text())
    gate = train_gate(split_results, residual["source_residual_gate_passed"], pair_controls_valid)
    valid = None
    if gate["passed"]:
        valid = load_npz(output / "representations/valid_cache.npz")
    train_pairs = rows_for_indices(global_pairs_frame, np.arange(len(labels)))
    official = full_train_official_valid(
        output, train, valid, representations, train_pairs, gate, cli.bootstrap_iterations
    ) if gate["passed"] else full_train_official_valid(
        output, train, {}, representations, train_pairs, gate, cli.bootstrap_iterations
    )
    mean_p20 = float(np.mean([result["deltas"]["P2_minus_P0"]["J"] for result in split_results]))
    mean_p30 = float(np.mean([result["deltas"]["P3_minus_P0"]["J"] for result in split_results]))
    generic = bool(
        not gate["passed"]
        and gate["mean_P1_minus_P0_J"] < 0
        and mean_p20 < 0
        and mean_p30 < 0
        and (
            gate["mean_P1_minus_P2_J"] > -0.002
            or gate["mean_P1_minus_P3_J"] > -0.002
        )
    )
    if gate["passed"] and official.get("passed"):
        status = "STAGE21A_SOURCE_RELATIVE_AUDIT_PASSED"
    elif gate["passed"]:
        status = "STAGE21A_FAILED_OFFICIAL_VALID_CONFIRMATION"
    elif not residual["source_residual_gate_passed"]:
        status = "STAGE21A_FAILED_NO_SOURCE_RESIDUAL_STRUCTURE"
    elif generic:
        status = "STAGE21A_GENERIC_PAIRWISE_ONLY_SOURCE_CLAIM_UNSUPPORTED"
    else:
        status = "STAGE21A_FAILED_TRUE_SOURCE_NOT_BETTER_THAN_CONTROLS"
    payload = {
        "created_at": utc_now(),
        "code_commit": git_head(),
        "probe_design": {
            "primary_representation": "actual tensor entering backbone.proj1",
            "shared_head": True,
            "ridge_alpha": 1e-3,
            "relative_mass": 0.5,
            "video_id_in_design_matrix": False,
            "clip_id_in_design_matrix": False,
            "source_embedding": False,
            "context_input": False,
            "J_formula": "0.5*LAV_MAE + 0.5*mean(LA_MAE,LV_MAE,L_MAE)",
        },
        "fingerprint": fingerprint_rows,
        "inner_splits": split_results,
        "train_only_gate": gate,
        "pair_controls_valid": pair_controls_valid,
        "generic_pairwise_signal": generic,
        "official_valid": official,
        "status": status,
        "locked_test_access_count": 0,
    }
    atomic_json(output / "probes/probe_metrics.json", payload)
    atomic_json(
        output / "probes/pair_control_diagnostics.json",
        {"rows": matching_rows, "valid": pair_controls_valid, "locked_test_access_count": 0},
    )
    atomic_frame(output / "pairs/control_pair_matching.tsv", matching_rows)
    atomic_frame(output / "probes/probe_cluster_bootstrap.tsv", bootstrap_rows)
    result_rows = []
    for result in split_results:
        seed = result["seed"]
        for probe, metrics in result["metrics"].items():
            for mode in MODES + ("MissingMacro",):
                result_rows.append({"SplitSeed": seed, "Probe": probe, "Mode": mode, "J": metrics["J"], **metrics[mode]})
    atomic_frame(output / "probes/inner_split_results.tsv", result_rows)
    source_rows = []
    for row in fingerprint_rows:
        source_rows.append(
            {
                "Evidence": "source_fingerprint",
                **row,
                "P1_minus_P0_J": "NA",
                "P1_minus_P2_J": "NA",
                "P1_minus_P3_J": "NA",
            }
        )
    for result in split_results:
        source_rows.append(
            {
                "Evidence": "controlled_probe",
                "split_seed": result["seed"],
                "mode": "ALL_SHARED_HEAD",
                "auroc": "NA",
                "balanced_accuracy": "NA",
                "shuffled_train_label_auroc": "NA",
                "train_pair_count_per_class": "NA",
                "valid_pair_count_per_class": "NA",
                "feature_dimension": "NA",
                "source_disjoint": True,
                "P1_minus_P0_J": result["deltas"]["P1_minus_P0"]["J"],
                "P1_minus_P2_J": result["deltas"]["P1_minus_P2"]["J"],
                "P1_minus_P3_J": result["deltas"]["P1_minus_P3"]["J"],
            }
        )
    atomic_frame(output / "source_evidence.tsv", source_rows)
    print(
        json.dumps(
            {
                "status": status,
                "train_gate_passed": gate["passed"],
                "P1_minus_P0": gate["P1_minus_P0_J"],
                "P1_minus_P2": gate["P1_minus_P2_J"],
                "P1_minus_P3": gate["P1_minus_P3_J"],
                "official_valid_run": official["run"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
