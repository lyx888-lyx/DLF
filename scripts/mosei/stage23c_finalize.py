#!/usr/bin/env python3
"""One-shot outer evaluation and final audit for frozen Stage23C Students."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from stage23a_v2_common import EXPERTS, MODES, MISSING_MODES, regression_metrics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "result" / "oracle_distillation_v1" / "mosei"
ANALYSIS = OUT / "analysis"
AUDIT = OUT / "audit"
FINAL = OUT / "final"
RUNTIME = ROOT / "runtime" / "stage23c"
META = (
    ROOT
    / "result"
    / "arbiter_audit_v2"
    / "mosei"
    / "data"
    / "meta_ledger.csv"
)
SELECTION = OUT / "protocol" / "frozen_student_selection.json"
TARGET_SELECTION = OUT / "protocol" / "frozen_target_selection.json"
CACHE = AUDIT / "outer_evaluation_cache.csv.gz"
FINAL_JSON = ANALYSIS / "stage23c_mosei_audit.json"

METHODS = (
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
)
SHUFFLE_SEEDS = (23611, 23612, 23613)
EXPERT_COLUMNS = [f"prediction__{expert}" for expert in EXPERTS]
LOWER_IS_BETTER = {"J", "MAE"}
METRICS = ("MAE", "Corr", "Acc7", "Acc5", "Acc2", "F1")


def utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_tsv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, float_format="%.10g")
    os.replace(temporary, path)


def atomic_gzip_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "mtime": 0},
    )
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_id(method, selection, direction, shuffle_seed=None):
    if method == "S5":
        ratio = selection["directions"][direction]["selected_hier_ratio"]
        return f"S5_hier_ratio{str(ratio).replace('.', 'p')}"
    if method == "S6":
        return f"S6_shuffle_seed{shuffle_seed}"
    return method


def expected_runs(selection):
    rows = []
    for direction in ("A", "B"):
        for method in METHODS:
            rows.append((direction, method, None, run_id(method, selection, direction)))
        for seed in SHUFFLE_SEEDS:
            rows.append(
                (
                    direction,
                    "S6",
                    seed,
                    run_id("S6", selection, direction, seed),
                )
            )
    return rows


def freeze_predictions(selection):
    inventory = []
    frames = {}
    for direction, method, seed, identifier in expected_runs(selection):
        directory = (
            OUT
            / "training"
            / f"direction_{direction}"
            / "final"
            / identifier
        )
        manifest_path = directory / "run_manifest.json"
        prediction_path = directory / "outer_predictions_label_free.csv.gz"
        if not manifest_path.exists() or not prediction_path.exists():
            raise RuntimeError(f"Missing frozen outer output: {directory}")
        manifest = read_json(manifest_path)
        expected_rows = 7794 * 4 if direction == "A" else 8532 * 4
        checks = {
            "status_complete": manifest.get("status") == "COMPLETED",
            "prediction_written": manifest.get("outer_prediction_written") is True,
            "prediction_pass_once": manifest.get("outer_prediction_pass_count") == 1,
            "outer_label_access_zero": manifest.get("outer_label_access_count") == 0,
            "official_valid_zero": manifest.get("official_valid_access_count") == 0,
            "test_zero": manifest.get("locked_test_access_count") == 0,
            "manifest_rows_correct": manifest.get("outer_prediction_rows")
            == expected_rows,
            "prediction_sha_correct": manifest.get("outer_prediction_sha256")
            == sha256_file(prediction_path),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"Outer manifest integrity failure {manifest_path}: {checks}"
            )
        frame = pd.read_csv(prediction_path)
        frame_checks = {
            "rows_correct": len(frame) == expected_rows,
            "no_label_column": "label" not in frame.columns,
            "unique_binding": not frame.duplicated(["sample_id", "mode"]).any(),
            "modes_exact": set(frame["mode"]) == set(MODES),
            "direction_exact": set(frame["direction"]) == {direction},
            "method_exact": set(frame["method"]) == {identifier},
            "finite_predictions": np.isfinite(frame["prediction"]).all(),
        }
        if not all(frame_checks.values()):
            raise RuntimeError(
                f"Outer prediction integrity failure {prediction_path}: {frame_checks}"
            )
        method_id = method if method != "S6" else f"S6_seed{seed}"
        frames[(direction, method_id)] = frame[
            ["sample_id", "train_index", "mode", "prediction"]
        ].rename(columns={"prediction": method_id})
        inventory.append(
            {
                "direction": direction,
                "method": method_id,
                "run_id": identifier,
                "shuffle_seed": seed,
                "prediction_rows": len(frame),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                **checks,
                **frame_checks,
            }
        )
    inventory_frame = pd.DataFrame(inventory)
    inventory_path = AUDIT / "outer_prediction_inventory.tsv"
    atomic_tsv(inventory_frame, inventory_path)
    freeze = {
        "stage": "Stage23C outer prediction freeze",
        "status": "ALL_OUTER_PREDICTIONS_FROZEN_BEFORE_LABEL_ACCESS",
        "frozen_at": utc_now(),
        "prediction_count": len(inventory_frame),
        "expected_prediction_count": 18,
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "outer_evaluation_access_count": 0,
    }
    atomic_json(AUDIT / "outer_prediction_freeze.json", freeze)
    return frames, inventory_frame, freeze


def open_outer_labels_once(frames):
    state_path = RUNTIME / "state.json"
    state = read_json(state_path)
    count = int(state.get("outer_evaluation_access_count", 0))
    if CACHE.exists():
        if count != 1:
            raise RuntimeError("Outer cache exists but access count is not exactly one")
        return pd.read_csv(CACHE), False
    if count != 0:
        raise RuntimeError(f"Refusing a second outer label opening; count={count}")
    state.update(
        {
            "status": "OUTER_LABELS_OPENED_ANALYSIS_IN_PROGRESS",
            "outer_evaluation_access_count": 1,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "outer_labels_opened_at": utc_now(),
        }
    )
    atomic_json(state_path, state)
    meta = pd.read_csv(META)
    if len(meta) != 65304 or meta.duplicated(["sample_id", "mode"]).any():
        raise RuntimeError("Frozen meta ledger binding changed")
    combined = []
    for direction, fold in (("A", 1), ("B", 0)):
        local = meta.loc[
            meta["expert_fold"] == fold,
            [
                "sample_id",
                "video_id",
                "train_index",
                "mode",
                "label",
                *EXPERT_COLUMNS,
            ],
        ].copy()
        expected = 7794 * 4 if direction == "A" else 8532 * 4
        if len(local) != expected:
            raise RuntimeError(f"Direction {direction} outer label rows changed")
        local.insert(0, "direction", direction)
        for (frame_direction, method), prediction in frames.items():
            if frame_direction != direction:
                continue
            local = local.merge(
                prediction,
                on=["sample_id", "train_index", "mode"],
                how="left",
                validate="one_to_one",
            )
            if local[method].isna().any():
                raise RuntimeError(f"Direction {direction} missing {method} predictions")
        combined.append(local)
    cache = pd.concat(combined, ignore_index=True)
    atomic_gzip_csv(cache, CACHE)
    return cache, True


def evaluate_frame(frame, prediction_column):
    result = {}
    for mode in MODES:
        local = frame.loc[frame["mode"] == mode]
        result[mode] = regression_metrics(
            local[prediction_column].to_numpy(),
            local["label"].to_numpy(),
        )
    result["MissingMacro"] = {
        metric: float(np.mean([result[mode][metric] for mode in MISSING_MODES]))
        for metric in METRICS
    }
    result["Overall"] = {
        "J": float(
            0.5 * result["LAV"]["MAE"]
            + 0.5 * result["MissingMacro"]["MAE"]
        )
    }
    return result


def metrics_table(cache):
    raw_rows = []
    method_ids = list(METHODS) + [f"S6_seed{seed}" for seed in SHUFFLE_SEEDS]
    for direction in ("A", "B"):
        local = cache.loc[cache["direction"] == direction]
        for method in method_ids:
            values = evaluate_frame(local, method)
            for mode, metrics in values.items():
                for metric, value in metrics.items():
                    raw_rows.append(
                        {
                            "direction": direction,
                            "method": method,
                            "mode": mode,
                            "metric": metric,
                            "value": value,
                        }
                    )
    raw = pd.DataFrame(raw_rows)
    mean_rows = (
        raw.groupby(["method", "mode", "metric"], as_index=False)["value"]
        .mean()
        .assign(direction="Mean")
    )
    full = pd.concat([raw, mean_rows], ignore_index=True)
    shuffled = full.loc[full["method"].str.startswith("S6_seed")].copy()
    shuffled_summary = (
        shuffled.groupby(["direction", "mode", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    mean_j = full.loc[
        (full["direction"] == "Mean")
        & (full["mode"] == "Overall")
        & (full["metric"] == "J")
        & full["method"].str.startswith("S6_seed")
    ].sort_values(["value", "method"])
    strongest_seed_method = str(mean_j.iloc[0]["method"])
    strongest_seed = int(strongest_seed_method.rsplit("seed", 1)[1])
    strongest = full.loc[full["method"] == strongest_seed_method].copy()
    strongest["method"] = "S6_strongest"
    shuffled_mean = shuffled_summary.rename(columns={"mean": "value"}).copy()
    shuffled_mean["method"] = "S6_mean"
    shuffled_std = shuffled_summary.rename(columns={"std": "value"}).copy()
    shuffled_std["method"] = "S6_std"
    full = pd.concat(
        [
            full,
            strongest,
            shuffled_mean[["direction", "method", "mode", "metric", "value"]],
            shuffled_std[["direction", "method", "mode", "metric", "value"]],
        ],
        ignore_index=True,
    )
    for index, row in shuffled_summary.iterrows():
        candidates = shuffled.loc[
            (shuffled["direction"] == row["direction"])
            & (shuffled["mode"] == row["mode"])
            & (shuffled["metric"] == row["metric"])
        ].sort_values(
            "value",
            ascending=row["metric"] in LOWER_IS_BETTER,
        )
        shuffled_summary.loc[index, "strongest_seed"] = int(
            candidates.iloc[0]["method"].rsplit("seed", 1)[1]
        )
        shuffled_summary.loc[index, "strongest_value"] = float(
            candidates.iloc[0]["value"]
        )
    shuffled_summary["gate_strongest_seed"] = strongest_seed
    return full, shuffled_summary, strongest_seed_method


def value(full, direction, method, mode, metric):
    selected = full.loc[
        (full["direction"] == direction)
        & (full["method"] == method)
        & (full["mode"] == mode)
        & (full["metric"] == metric),
        "value",
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Metric lookup not unique: {direction}/{method}/{mode}/{metric}"
        )
    return float(selected.iloc[0])


def comparison_table(full, strongest_baseline):
    comparisons = {
        "S5_minus_S1": "S1",
        "S5_minus_S2": "S2",
        "S5_minus_S3": "S3",
        "S5_minus_S4": "S4",
        "S5_minus_strongest_shuffled": "S6_strongest",
        "S5_minus_strongest_baseline": strongest_baseline,
    }
    rows = []
    for name, reference in comparisons.items():
        for direction in ("A", "B", "Mean"):
            for mode in (*MODES, "MissingMacro", "Overall"):
                metrics = ("J",) if mode == "Overall" else METRICS
                for metric in metrics:
                    proposed = value(full, direction, "S5", mode, metric)
                    baseline = value(full, direction, reference, mode, metric)
                    delta = proposed - baseline
                    improved = (
                        delta < -1e-8
                        if metric in LOWER_IS_BETTER
                        else delta > 1e-8
                    )
                    tied = abs(delta) <= 1e-8
                    rows.append(
                        {
                            "comparison": name,
                            "direction": direction,
                            "mode": mode,
                            "metric": metric,
                            "reference_method": reference,
                            "reference": baseline,
                            "S5": proposed,
                            "delta": delta,
                            "verdict": (
                                "Improved"
                                if improved
                                else "Tied"
                                if tied
                                else "Degraded"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def add_outer_diagnostics(cache):
    target_selection = read_json(TARGET_SELECTION)
    result = []
    for direction in ("A", "B"):
        local = cache.loc[cache["direction"] == direction].copy()
        tau = float(
            target_selection["directions"][direction]["selected_tau"]
        )
        predictions = local[EXPERT_COLUMNS].to_numpy(dtype=np.float64)
        labels = local["label"].to_numpy(dtype=np.float64)[:, None]
        errors = np.abs(predictions - labels)
        regret = errors - errors.min(axis=1, keepdims=True)
        logits = -regret / tau
        logits -= logits.max(axis=1, keepdims=True)
        q = np.exp(logits)
        q /= q.sum(axis=1, keepdims=True)
        local["posthoc_q_entropy"] = -np.sum(
            q * np.log(np.maximum(q, 1e-300)), axis=1
        )
        local["posthoc_top1_responsibility"] = q.max(axis=1)
        sorted_error = np.sort(errors, axis=1)
        local["posthoc_best_second_error_margin"] = (
            sorted_error[:, 1] - sorted_error[:, 0]
        )
        local["posthoc_oracle_teacher"] = np.sum(q * predictions, axis=1)
        local["posthoc_equal_teacher"] = predictions.mean(axis=1)
        fixed = pd.read_csv(
            OUT / "targets" / f"direction_{direction}_fixed_weights.tsv",
            sep="\t",
        )
        fixed = fixed.loc[fixed["fit_scope"] == "full_development"]
        fixed_target = np.zeros(len(local), dtype=np.float64)
        for mode in MODES:
            mask = local["mode"].to_numpy() == mode
            weights = (
                fixed.loc[fixed["mode"] == mode]
                .set_index("expert_id")
                .loc[list(EXPERTS), "weight"]
                .to_numpy(dtype=np.float64)
            )
            fixed_target[mask] = predictions[mask] @ weights
        local["posthoc_fixed_teacher"] = fixed_target
        static_error = np.minimum(
            np.abs(local["label"] - local["posthoc_equal_teacher"]),
            np.abs(local["label"] - local["posthoc_fixed_teacher"]),
        )
        oracle_error = np.abs(
            local["label"] - local["posthoc_oracle_teacher"]
        )
        local["posthoc_oracle_gain"] = static_error - oracle_error
        result.append(local)
    return pd.concat(result, ignore_index=True)


def rank_quartile(series):
    return pd.qcut(
        series.rank(method="first"),
        4,
        labels=("Q1_low", "Q2", "Q3", "Q4_high"),
    ).astype(str)


def bucket_analysis(cache, strongest_seed_method):
    frame = add_outer_diagnostics(cache)
    frame["bucket_q_entropy"] = frame.groupby("direction")[
        "posthoc_q_entropy"
    ].transform(rank_quartile)
    frame["bucket_oracle_gain"] = frame.groupby("direction")[
        "posthoc_oracle_gain"
    ].transform(rank_quartile)
    frame["bucket_error_margin"] = frame.groupby("direction")[
        "posthoc_best_second_error_margin"
    ].transform(rank_quartile)
    frame["bucket_polarity"] = np.select(
        [frame["label"] < 0, frame["label"] > 0],
        ["Negative", "Positive"],
        default="Zero",
    )
    frame["bucket_label_magnitude"] = pd.cut(
        np.abs(frame["label"]),
        [-np.inf, 1.0, 2.0, np.inf],
        labels=("[0,1)", "[1,2)", "[2,inf)"),
        right=False,
    ).astype(str)
    methods = ("S1", "S2", "S3", "S4", "S5", strongest_seed_method)
    rows = []
    for family, column in (
        ("q_entropy", "bucket_q_entropy"),
        ("oracle_gain", "bucket_oracle_gain"),
        ("error_margin", "bucket_error_margin"),
        ("polarity", "bucket_polarity"),
        ("label_magnitude", "bucket_label_magnitude"),
    ):
        for (direction, mode, bucket), local in frame.groupby(
            ["direction", "mode", column], observed=True
        ):
            s5_mae = float(np.mean(np.abs(local["S5"] - local["label"])))
            for method in methods:
                method_mae = float(
                    np.mean(np.abs(local[method] - local["label"]))
                )
                rows.append(
                    {
                        "direction": direction,
                        "mode": mode,
                        "bucket_family": family,
                        "bucket": bucket,
                        "method": method,
                        "sample_count": len(local),
                        "MAE": method_mae,
                        "S5_minus_method_MAE": s5_mae - method_mae,
                        "mean_q_entropy": float(
                            local["posthoc_q_entropy"].mean()
                        ),
                        "mean_oracle_gain": float(
                            local["posthoc_oracle_gain"].mean()
                        ),
                        "mean_error_margin": float(
                            local["posthoc_best_second_error_margin"].mean()
                        ),
                    }
                )
    residual_rows = []
    for (direction, mode), local in frame.groupby(["direction", "mode"]):
        oracle_residual = (
            local["posthoc_oracle_teacher"] - local["label"]
        ).to_numpy()
        for method in methods:
            student_residual = (local[method] - local["label"]).to_numpy()
            corr = (
                float(np.corrcoef(student_residual, oracle_residual)[0, 1])
                if student_residual.std() > 0 and oracle_residual.std() > 0
                else 0.0
            )
            residual_rows.append(
                {
                    "direction": direction,
                    "mode": mode,
                    "method": method,
                    "student_oracle_teacher_residual_corr": corr,
                }
            )
    diagnostic_columns = [
        "direction",
        "sample_id",
        "mode",
        "posthoc_q_entropy",
        "posthoc_top1_responsibility",
        "posthoc_best_second_error_margin",
        "posthoc_oracle_gain",
    ]
    atomic_gzip_csv(
        frame[diagnostic_columns],
        ANALYSIS / "outer_posthoc_oracle_diagnostics.csv.gz",
    )
    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


def promotion_gate(full, comparison, strongest_baseline, strongest_seed):
    def delta(comparison_name, direction, mode, metric):
        selected = comparison.loc[
            (comparison["comparison"] == comparison_name)
            & (comparison["direction"] == direction)
            & (comparison["mode"] == mode)
            & (comparison["metric"] == metric),
            "delta",
        ]
        return float(selected.iloc[0])

    main_mean = delta(
        "S5_minus_strongest_baseline", "Mean", "Overall", "J"
    )
    main_directions = [
        delta("S5_minus_strongest_baseline", direction, "Overall", "J")
        for direction in ("A", "B")
    ]
    stable_static = []
    for name in ("S5_minus_S2", "S5_minus_S3"):
        stable_static.append(delta(name, "Mean", "Overall", "J") <= 0)
        stable_static.append(
            max(
                delta(name, direction, "Overall", "J")
                for direction in ("A", "B")
            )
            <= 0.001
        )
    shuffled_delta = delta(
        "S5_minus_strongest_shuffled", "Mean", "Overall", "J"
    )
    missing_deltas = {
        mode: delta(
            "S5_minus_strongest_baseline", "Mean", mode, "MAE"
        )
        for mode in MISSING_MODES
    }
    corr_deltas = [
        delta(
            "S5_minus_strongest_baseline", direction, mode, "Corr"
        )
        for direction in ("A", "B")
        for mode in MODES
    ]
    class_deltas = {
        metric: [
            delta(
                "S5_minus_strongest_baseline", direction, mode, metric
            )
            for direction in ("A", "B")
            for mode in MODES
        ]
        for metric in ("Acc7", "Acc5", "Acc2", "F1")
    }
    hierarchy_mean = delta("S5_minus_S4", "Mean", "Overall", "J")
    hierarchy_worst = max(
        delta("S5_minus_S4", direction, "Overall", "J")
        for direction in ("A", "B")
    )
    parameter_counts = []
    inference_safe = True
    selection = read_json(SELECTION)
    for direction, method, seed, identifier in expected_runs(selection):
        manifest = read_json(
            OUT
            / "training"
            / f"direction_{direction}"
            / "final"
            / identifier
            / "run_manifest.json"
        )
        parameter_counts.append(int(manifest["parameter_count"]))
        inference_safe &= (
            manifest.get("student_inference_inputs")
            == "text,audio,vision,mode mask only"
        )
    gates = [
        (
            "G1_mean_delta_J_vs_strongest_baseline_le_-0.003",
            main_mean <= -0.003,
            main_mean,
            strongest_baseline,
        ),
        (
            "G2_worst_direction_delta_J_le_+0.001",
            max(main_directions) <= 0.001,
            max(main_directions),
            json.dumps(dict(zip(("A", "B"), main_directions))),
        ),
        (
            "G3_stable_advantage_vs_S2_and_S3",
            all(stable_static),
            int(sum(stable_static)),
            "4/4 required: mean<=0 and worst direction<=+0.001 for each",
        ),
        (
            "G4_mean_delta_J_vs_strongest_shuffled_le_-0.002",
            shuffled_delta <= -0.002,
            shuffled_delta,
            strongest_seed,
        ),
        (
            "G5_at_least_2_of_3_missing_modes_MAE_improve",
            sum(value < 0 for value in missing_deltas.values()) >= 2,
            sum(value < 0 for value in missing_deltas.values()),
            json.dumps(missing_deltas, sort_keys=True),
        ),
        (
            "G6_Corr_not_clearly_lower",
            np.mean(corr_deltas) >= -0.002
            and sum(value >= 0 for value in corr_deltas) >= len(corr_deltas) / 2,
            float(np.mean(corr_deltas)),
            "pre-outer rule: mean>=-0.002 and >=50% nonnegative over A/B x 4 modes",
        ),
        (
            "G7_classification_not_systematically_degraded",
            all(
                np.mean(values) >= -0.002
                and sum(value >= 0 for value in values) >= len(values) / 2
                for values in class_deltas.values()
            ),
            min(float(np.mean(values)) for values in class_deltas.values()),
            json.dumps(
                {
                    metric: {
                        "mean": float(np.mean(values)),
                        "nonnegative": int(sum(value >= 0 for value in values)),
                    }
                    for metric, values in class_deltas.items()
                },
                sort_keys=True,
            ),
        ),
        (
            "G8_direction_A_B_improvement_consistent",
            all(value < 0 for value in main_directions),
            int(sum(value < 0 for value in main_directions)),
            json.dumps(dict(zip(("A", "B"), main_directions))),
        ),
        (
            "G9_hierarchical_S5_not_weaker_than_S4",
            hierarchy_mean <= 0 and hierarchy_worst <= 0.001,
            hierarchy_mean,
            f"worst_direction={hierarchy_worst:+.10f}",
        ),
        (
            "G10_single_student_inference_and_equal_parameter_count",
            inference_safe and len(set(parameter_counts)) == 1,
            len(set(parameter_counts)),
            f"parameter_count={parameter_counts[0]}; no q/Expert/label input",
        ),
    ]
    gate = pd.DataFrame(
        gates, columns=["gate", "passed", "observed", "detail"]
    )
    if bool(gate["passed"].all()):
        conclusion = "ORACLE_DISTILLATION_PASS"
    elif main_mean < 0 or (
        delta("S5_minus_S2", "Mean", "Overall", "J") < 0
        and delta("S5_minus_S3", "Mean", "Overall", "J") < 0
    ):
        conclusion = "ORACLE_DISTILLATION_WEAK"
    else:
        conclusion = "ORACLE_DISTILLATION_FAIL"
    details = {
        "conclusion": conclusion,
        "strongest_student_baseline": strongest_baseline,
        "strongest_shuffled_seed": strongest_seed,
        "mean_delta_J_vs_strongest_baseline": main_mean,
        "direction_delta_J_vs_strongest_baseline": dict(
            zip(("A", "B"), main_directions)
        ),
        "mean_delta_J_vs_strongest_shuffled": shuffled_delta,
        "missing_mode_MAE_deltas": missing_deltas,
        "hierarchical_delta_J_vs_S4": {
            "mean": hierarchy_mean,
            "worst_direction": hierarchy_worst,
        },
        "passed_gate_count": int(gate["passed"].sum()),
        "gate_count": len(gate),
    }
    return gate, details


def markdown_table(frame, columns, limit=None):
    local = frame.loc[:, columns]
    if limit is not None:
        local = local.head(limit)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in local.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.6f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, separator, *rows])


def write_report(full, gate, details, comparison, target_sanity):
    core = full.loc[
        full["direction"].isin(["A", "B", "Mean"])
        & full["method"].isin(
            [
                "S0",
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
                "S6_mean",
                "S6_strongest",
            ]
        )
        & (
            ((full["mode"] == "Overall") & (full["metric"] == "J"))
            | (
                full["mode"].isin(["LAV", "LA", "LV", "L", "MissingMacro"])
                & (full["metric"] == "MAE")
            )
        )
    ].copy()
    core["item"] = np.where(
        core["mode"] == "Overall",
        "J",
        core["mode"] + "_" + core["metric"],
    )
    core_wide = core.pivot_table(
        index=["direction", "method"], columns="item", values="value"
    ).reset_index()
    core_columns = [
        column
        for column in (
            "direction",
            "method",
            "J",
            "LAV_MAE",
            "LA_MAE",
            "LV_MAE",
            "L_MAE",
            "MissingMacro_MAE",
        )
        if column in core_wide
    ]
    teacher_valid = target_sanity.loc[
        (target_sanity["role"] == "inner_valid")
        & (target_sanity["mode"] == "Overall")
    ][["direction", "teacher", "MAE", "Corr"]]
    missing = details["missing_mode_MAE_deltas"]
    interpretation = (
        "The Soft-Oracle target is strong on the development-side target audit, "
        "but this single audit cannot separate Student capacity from optimization "
        "or protocol limitations if the Student fails to inherit that advantage."
    )
    report = f"""# Stage23C MOSEI OOF Soft-Oracle Hierarchical Distillation Audit

## Frozen outcome

**{details['conclusion']}**

- Stage23A-v2a remains `SIGNAL_AUDIT_FAIL`.
- Stage23A-v2b remains `LOCAL_COMPETENCE_WEAK`.
- Strongest current Student baseline: `{details['strongest_student_baseline']}`.
- Strongest shuffled control seed: `{details['strongest_shuffled_seed']}`.
- Mean S5 ΔJ versus strongest baseline: {details['mean_delta_J_vs_strongest_baseline']:+.6f}.
- Direction A/B ΔJ: {json.dumps(details['direction_delta_J_vs_strongest_baseline'], sort_keys=True)}.
- Mean S5 ΔJ versus strongest shuffled control: {details['mean_delta_J_vs_strongest_shuffled']:+.6f}.
- Missing-mode MAE deltas: {json.dumps(missing, sort_keys=True)}.
- Official Valid access count: 0.
- Locked Test access count: 0.
- Outer evaluation label opening count: 1, after all 18 prediction ledgers were frozen.

## Development-side Teacher target sanity

{markdown_table(teacher_valid, ['direction', 'teacher', 'MAE', 'Corr'])}

Teacher target results are Train-side diagnostics only and are not Student claims.

## Outer Student core metrics

{markdown_table(core_wide, core_columns)}

## Frozen promotion gates

{markdown_table(gate, ['gate', 'passed', 'observed', 'detail'])}

Passed {details['passed_gate_count']}/{details['gate_count']} gates.

## Main comparisons

The delta convention is Student S5 minus reference. Negative is better for J/MAE;
positive is better for Corr and classification metrics.

{markdown_table(comparison.loc[(comparison['direction'] == 'Mean') & (comparison['mode'].isin(['Overall', 'LAV', 'MissingMacro']))], ['comparison', 'mode', 'metric', 'reference', 'S5', 'delta', 'verdict'])}

## Interpretation boundary

{interpretation}

No outer result was used to change tau, hierarchical loss ratio, epoch, seed,
architecture, or any other configuration. No Official Valid/Test data were read.
S5 inference is one Student and receives only text/audio/vision plus the mode mask.
It receives no q, Expert prediction, Judge output, or Ground Truth.
"""
    path = ANALYSIS / "stage23c_mosei_audit.md"
    path.write_text(report, encoding="utf-8")
    return path


def runtime_report():
    screen_logs = list((OUT / "training").glob("direction_*/screen/*/epoch_metrics.tsv"))
    final_logs = list(
        (OUT / "training").glob("direction_*/final/*/retrain_epoch_metrics.tsv")
    )
    rows = []
    for path in screen_logs + final_logs:
        frame = pd.read_csv(path, sep="\t")
        rows.append(
            {
                "path": str(path.resolve()),
                "phase": "screen" if "screen" in path.parts else "final_retrain",
                "epochs": int(frame["epoch"].max()),
                "wall_seconds": float(frame["wall_seconds"].max()),
            }
        )
    atomic_tsv(pd.DataFrame(rows), OUT / "runtime_report.tsv")
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except Exception as error:  # pragma: no cover - environment diagnostic
        gpu = f"nvidia-smi unavailable: {error}"
    (OUT / "runtime_gpu_report.txt").write_text(
        "Stage23C used GPU IDs 0, 1, and 2 only; GPU 3 was left to the external job.\n"
        + gpu,
        encoding="utf-8",
    )


def artifact_manifest():
    rows = []
    manifest_path = OUT / "artifact_manifest.tsv"
    self_referential = {
        manifest_path,
        FINAL_JSON,
        FINAL / "STAGE23C_CONCLUSION.json",
    }
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path in self_referential:
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    atomic_tsv(frame, manifest_path)
    return manifest_path, len(frame)


def main():
    if FINAL_JSON.exists():
        result = read_json(FINAL_JSON)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    selection = read_json(SELECTION)
    frames, inventory, freeze = freeze_predictions(selection)
    cache, opened_now = open_outer_labels_once(frames)
    full, shuffled, strongest_seed_method = metrics_table(cache)
    strongest_baseline = (
        full.loc[
            (full["direction"] == "Mean")
            & (full["mode"] == "Overall")
            & (full["metric"] == "J")
            & full["method"].isin(["S0", "S1", "S2", "S3", "S4"])
        ]
        .sort_values(["value", "method"])
        .iloc[0]["method"]
    )
    comparison = comparison_table(full, strongest_baseline)
    buckets, residual = bucket_analysis(cache, strongest_seed_method)
    gate, details = promotion_gate(
        full,
        comparison,
        strongest_baseline,
        int(strongest_seed_method.rsplit("seed", 1)[1]),
    )
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    atomic_tsv(full, ANALYSIS / "full_metrics.tsv")
    atomic_json(
        ANALYSIS / "full_metrics.json",
        full.to_dict(orient="records"),
    )
    atomic_tsv(comparison, ANALYSIS / "main_comparisons.tsv")
    atomic_tsv(shuffled, ANALYSIS / "shuffled_control_results.tsv")
    atomic_tsv(buckets, ANALYSIS / "per_mode_bucket_analysis.tsv")
    atomic_tsv(residual, ANALYSIS / "student_oracle_residual_correlation.tsv")
    atomic_tsv(gate, ANALYSIS / "promotion_gate.tsv")
    target_sanity = pd.read_csv(
        OUT / "targets" / "teacher_target_sanity.tsv", sep="\t"
    )
    report_path = write_report(
        full, gate, details, comparison, target_sanity
    )
    result = {
        "stage": "Stage23C OOF Soft-Oracle Hierarchical Distillation Audit",
        "status": "COMPLETED",
        **details,
        "all_outer_predictions_frozen_before_label_access": True,
        "outer_prediction_count": len(inventory),
        "outer_evaluation_access_count": 1,
        "outer_labels_opened_in_this_invocation": opened_now,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "student_second_seed_count": 0,
        "full_data_training_count": 0,
        "judge_training_count": 0,
        "joint_specialized_moe_training_count": 0,
        "single_student_inference": True,
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "completed_at": utc_now(),
    }
    atomic_json(FINAL_JSON, result)
    lock = {
        "stage": "Stage23C",
        "official_valid_authorized": False,
        "official_valid_access_count": 0,
        "test_authorized": False,
        "locked_test_access_count": 0,
        "outer_evaluation_access_count": 1,
        "status": "OFFICIAL_VALID_AND_TEST_LOCKED",
        "updated_at": utc_now(),
    }
    atomic_json(FINAL / "TEST_LOCK_STATUS.json", lock)
    atomic_json(FINAL / "STAGE23C_CONCLUSION.json", result)
    integrity = {
        "prediction_inventory_sha256": sha256_file(
            AUDIT / "outer_prediction_inventory.tsv"
        ),
        "prediction_freeze_sha256": sha256_file(
            AUDIT / "outer_prediction_freeze.json"
        ),
        "outer_evaluation_cache_sha256": sha256_file(CACHE),
        "prediction_count": len(inventory),
        "parameter_count_unique": int(
            read_json(SELECTION)["student_parameter_count"]
        ),
        "all_outer_predictions_frozen_before_label_access": True,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "outer_evaluation_access_count": 1,
    }
    atomic_json(AUDIT / "integrity_audit.json", integrity)
    commands = """# Stage23C reproducible commands

All selection commands use only the corresponding source-disjoint development side.
The finalizer refuses to open outer labels until all 18 label-free prediction
ledgers and their SHA-bound manifests are complete.

```bash
python scripts/mosei/stage23c_authorize.py
python scripts/mosei/stage23c_prepare_targets.py
python scripts/mosei/stage23c_train.py --phase clean-screen --direction A --gpu-id 0
python scripts/mosei/stage23c_train.py --phase clean-screen --direction B --gpu-id 1
# S0-S5 inner-valid screens, then:
python scripts/mosei/stage23c_train.py --phase freeze-pre-s6
# Three fixed S6 shuffle seeds per direction, then:
python scripts/mosei/stage23c_train.py --phase freeze-selection
# clean-final and student-final for every frozen method/direction
python scripts/mosei/stage23c_finalize.py
```
"""
    (OUT / "reproducible_commands.md").write_text(commands, encoding="utf-8")
    runtime_report()
    state_path = RUNTIME / "state.json"
    state = read_json(state_path)
    state.update(
        {
            "status": details["conclusion"],
            "outer_evaluation_access_count": 1,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "final_report_path": str(FINAL_JSON.resolve()),
            "final_report_sha256": sha256_file(FINAL_JSON),
            "updated_at": utc_now(),
        }
    )
    atomic_json(state_path, state)
    manifest_path, artifact_count = artifact_manifest()
    result["artifact_manifest_path"] = str(manifest_path.resolve())
    result["artifact_manifest_sha256"] = sha256_file(manifest_path)
    result["artifact_count"] = artifact_count
    atomic_json(FINAL_JSON, result)
    atomic_json(FINAL / "STAGE23C_CONCLUSION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
