"""Independent artifact audit for the CFCompatKD seed-mechanism analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_mechanism_utils import (
    FORMAL_SEEDS,
    MAX_SAMPLES_PER_VIDEO,
    METHOD,
    MISSING_MODES,
    MODES,
    OBJECTIVE_PAIRS,
    PARAMETER_GROUPS,
    PROBE_SAMPLE_COUNT,
    VERSION,
    calibration_stats,
    infer_mechanism_flags,
    pearson,
    spearman,
    summarize_mode_deltas,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def close(left, right, tolerance=1e-7):
    if pd.isna(left) and pd.isna(right):
        return True
    return abs(float(left) - float(right)) <= tolerance


def frame_equal(actual, expected, keys, tolerance=1e-7):
    left = actual.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = expected.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        if column in keys:
            if not left[column].astype(str).equals(right[column].astype(str)):
                return False
        elif pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            for first, second in zip(left[column], right[column]):
                if not close(first, second, tolerance):
                    return False
        else:
            if not left[column].astype(str).equals(right[column].astype(str)):
                return False
    return True


def recompute_gradient_summary(raw):
    rows = []
    for keys, local in raw.groupby(
        ["Seed", "Variant", "Mode", "ParameterGroup", "Pair"], sort=True
    ):
        finite_cos = local.cosine[np.isfinite(local.cosine.astype(float))]
        rows.append({
            "Seed": int(keys[0]),
            "Variant": str(keys[1]),
            "Mode": str(keys[2]),
            "ParameterGroup": str(keys[3]),
            "Pair": str(keys[4]),
            "batch_count": int(len(local)),
            "finite_cosine_count": int(len(finite_cos)),
            "mean_cosine": (
                float(finite_cos.astype(float).mean())
                if len(finite_cos) else float("nan")
            ),
            "median_cosine": (
                float(finite_cos.astype(float).median())
                if len(finite_cos) else float("nan")
            ),
            "conflict_fraction": (
                float((finite_cos.astype(float) < 0).mean())
                if len(finite_cos) else float("nan")
            ),
            "mean_left_norm": float(local.left_norm.mean()),
            "mean_right_norm": float(local.right_norm.mean()),
            "all_finite": bool(local.finite.all()),
        })
    return pd.DataFrame(rows)


def recompute_gradient_delta(summary):
    baseline = summary.loc[
        summary.Variant.eq("baseline")
    ].drop(columns=["Variant"])
    sam = summary.loc[
        summary.Variant.eq("sam")
    ].drop(columns=["Variant"])
    keys = ["Seed", "Mode", "ParameterGroup", "Pair"]
    delta = baseline.merge(
        sam, on=keys, suffixes=("_baseline", "_sam"), validate="one_to_one"
    )
    delta["mean_cosine_change_sam_minus_baseline"] = (
        delta.mean_cosine_sam - delta.mean_cosine_baseline
    )
    delta["conflict_fraction_change_sam_minus_baseline"] = (
        delta.conflict_fraction_sam - delta.conflict_fraction_baseline
    )
    return delta


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    required = {
        "summary": root / "mechanism_summary.json",
        "source": root / "mechanism_source_manifest.json",
        "report": root / "mechanism_report.md",
        "sample": root / "prediction_sample_deltas.csv",
        "mode": root / "prediction_mode_deltas.csv",
        "mode_summary": root / "prediction_mode_summary.csv",
        "group_summary": root / "prediction_group_summary.csv",
        "video_summary": root / "prediction_video_summary.csv",
        "cross": root / "cross_seed_sample_comparison.csv",
        "context": root / "valid_teacher_evaluator_context.csv",
        "calibration": root / "prediction_calibration.csv",
        "shrinkage": root / "prediction_shrinkage.csv",
        "relations": root / "prediction_context_relations.csv",
        "probe": root / "gradient_probe_manifest.csv",
        "gradient_raw": root / "gradient_pairwise_raw.csv",
        "gradient_summary": root / "gradient_pair_summary.csv",
        "gradient_delta": root / "gradient_sam_minus_baseline.csv",
        "gradient_state": root / "gradient_model_state_audit.csv",
        "gradient_groups": root / "gradient_parameter_groups.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing mechanism artifacts:\n" + "\n".join(missing))

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    source = json.loads(required["source"].read_text(encoding="utf-8"))
    sample = pd.read_csv(required["sample"])
    mode = pd.read_csv(required["mode"])
    mode_summary = pd.read_csv(required["mode_summary"])
    cross = pd.read_csv(required["cross"])
    context = pd.read_csv(required["context"])
    calibration = pd.read_csv(required["calibration"])
    shrinkage = pd.read_csv(required["shrinkage"])
    probe = pd.read_csv(required["probe"])
    gradient_raw = pd.read_csv(required["gradient_raw"])
    gradient_summary = pd.read_csv(required["gradient_summary"])
    gradient_delta = pd.read_csv(required["gradient_delta"])
    gradient_state = pd.read_csv(required["gradient_state"])
    gradient_groups = pd.read_csv(required["gradient_groups"])
    checks = {}

    checks["version_and_method"] = bool(
        summary["version"] == VERSION
        and summary["method"] == METHOD
        and source["version"] == VERSION
        and source["method"] == METHOD
    )
    checks["frozen_scope"] = bool(
        tuple(source["formal_seeds"]) == FORMAL_SEEDS
        and tuple(summary["protocol"]["parameter_groups"]) == PARAMETER_GROUPS
        and summary["protocol"]["gradient_probe_split"] == "train"
        and summary["protocol"]["official_valid_predictions_only"]
    )

    source_hashes = True
    for item in source["sam_source_files"].values():
        path = Path(item["path"])
        source_hashes = (
            source_hashes
            and path.is_file()
            and checkpoint_sha256(path) == item["sha256"]
        )
    checks["sam_source_hashes"] = bool(source_hashes)

    checks["valid_sample_contract"] = bool(
        len(sample) == 229 * len(FORMAL_SEEDS)
        and all(
            len(sample.loc[sample.Seed.astype(int).eq(seed)]) == 229
            and sample.loc[
                sample.Seed.astype(int).eq(seed)
            ].sample_index.nunique() == 229
            for seed in FORMAL_SEEDS
        )
        and len(mode) == len(sample) * len(MODES)
        and set(mode.Mode.astype(str)) == set(MODES)
    )

    error_ok = True
    for mode_name in MODES:
        base_expected = np.abs(
            sample[f"{mode_name}_pred_baseline"].to_numpy(dtype=float)
            - sample.label.to_numpy(dtype=float)
        )
        sam_expected = np.abs(
            sample[f"{mode_name}_pred_sam"].to_numpy(dtype=float)
            - sample.label.to_numpy(dtype=float)
        )
        error_ok = error_ok and np.allclose(
            base_expected,
            sample[f"baseline_{mode_name}_abs_error"].to_numpy(dtype=float),
            atol=1e-8,
        )
        error_ok = error_ok and np.allclose(
            sam_expected,
            sample[f"sam_{mode_name}_abs_error"].to_numpy(dtype=float),
            atol=1e-8,
        )
        error_ok = error_ok and np.allclose(
            sam_expected - base_expected,
            sample[f"delta_{mode_name}_abs_error"].to_numpy(dtype=float),
            atol=1e-8,
        )
    base_j = (
        0.5 * sample.baseline_LAV_abs_error
        + (1.0 / 6.0)
        * sum(sample[f"baseline_{mode_name}_abs_error"] for mode_name in MISSING_MODES)
    )
    sam_j = (
        0.5 * sample.sam_LAV_abs_error
        + (1.0 / 6.0)
        * sum(sample[f"sam_{mode_name}_abs_error"] for mode_name in MISSING_MODES)
    )
    error_ok = error_ok and np.allclose(
        base_j, sample.baseline_J_proxy, atol=1e-8
    )
    error_ok = error_ok and np.allclose(
        sam_j, sample.sam_J_proxy, atol=1e-8
    )
    error_ok = error_ok and np.allclose(
        sam_j - base_j, sample.delta_J_proxy, atol=1e-8
    )
    checks["prediction_deltas_recomputed"] = bool(error_ok)

    recomputed_mode_summary = summarize_mode_deltas(mode)
    checks["mode_summary_recomputed"] = frame_equal(
        mode_summary,
        recomputed_mode_summary,
        keys=["Seed", "Mode"],
    )

    left = cross.delta_J_seed1111.to_numpy(dtype=float)
    right = cross.delta_J_seed1114.to_numpy(dtype=float)
    recomputed_cross = {
        "count": int(len(cross)),
        "pearson_delta_J_proxy": pearson(left, right),
        "spearman_delta_J_proxy": spearman(left, right),
        "same_direction_fraction": float(
            (np.sign(left) == np.sign(right)).mean()
        ),
        "both_improve_fraction": float(((left < 0) & (right < 0)).mean()),
        "opposite_direction_fraction": float(
            (np.sign(left) != np.sign(right)).mean()
        ),
    }
    checks["cross_seed_effect_recomputed"] = bool(all(
        close(recomputed_cross[key], summary["cross_seed_sample_effect"][key])
        for key in recomputed_cross
    ))

    calibration_ok = True
    for row in calibration.to_dict("records"):
        local = sample.loc[sample.Seed.astype(int).eq(int(row["Seed"]))]
        column = (
            f"{row['Mode']}_pred_baseline"
            if row["Variant"] == "baseline"
            else f"{row['Mode']}_pred_sam"
        )
        stats = calibration_stats(local.label, local[column])
        for key, value in stats.items():
            calibration_ok = calibration_ok and close(row[key], value)
    checks["calibration_recomputed"] = bool(calibration_ok)

    checks["valid_context_contract"] = bool(
        len(context) == 229 * len(FORMAL_SEEDS)
        and context.sample_index.nunique() == 229
        and all(
            context[f"valid_compat_proxy_{mode_name}"].between(
                0.0, 1.0, inclusive="neither"
            ).all()
            for mode_name in MISSING_MODES
        )
        and summary["protocol"][
            "valid_compatibility_is_descriptive_proxy_not_training_gate"
        ]
    )

    expected_probe = int(summary["protocol"]["probe_sample_count"])
    checks["probe_contract"] = bool(
        expected_probe == PROBE_SAMPLE_COUNT
        and len(probe) == expected_probe
        and probe.sample_index.nunique() == expected_probe
        and int(probe.groupby("video_id").size().max())
        <= MAX_SAMPLES_PER_VIDEO
        and source["probe_sample_count"] == expected_probe
        and source["probe_max_samples_per_video"] == MAX_SAMPLES_PER_VIDEO
    )

    expected_pairs = {
        f"{left_name}_vs_{right_name}"
        for left_name, right_name in OBJECTIVE_PAIRS
    }
    checks["gradient_raw_contract"] = bool(
        set(gradient_raw.Seed.astype(int)) == set(FORMAL_SEEDS)
        and set(gradient_raw.Variant.astype(str)) == {"baseline", "sam"}
        and set(gradient_raw.Mode.astype(str)) == set(MISSING_MODES)
        and set(gradient_raw.ParameterGroup.astype(str)) == set(PARAMETER_GROUPS)
        and set(gradient_raw.Pair.astype(str)) == expected_pairs
        and gradient_raw.finite.all()
        and np.isfinite(gradient_raw.left_norm.astype(float)).all()
        and np.isfinite(gradient_raw.right_norm.astype(float)).all()
    )

    recomputed_gradient_summary = recompute_gradient_summary(gradient_raw)
    checks["gradient_summary_recomputed"] = frame_equal(
        gradient_summary,
        recomputed_gradient_summary,
        keys=["Seed", "Variant", "Mode", "ParameterGroup", "Pair"],
    )
    recomputed_gradient_delta = recompute_gradient_delta(
        recomputed_gradient_summary
    )
    checks["gradient_delta_recomputed"] = frame_equal(
        gradient_delta,
        recomputed_gradient_delta,
        keys=["Seed", "Mode", "ParameterGroup", "Pair"],
    )

    checks["parameter_groups_bound"] = bool(
        set(gradient_groups.Seed.astype(int)) == set(FORMAL_SEEDS)
        and set(gradient_groups.Variant.astype(str)) == {"baseline", "sam"}
        and set(gradient_groups.ParameterGroup.astype(str))
        == set(PARAMETER_GROUPS)
        and (gradient_groups.parameter_count.astype(int) > 0).all()
        and (gradient_groups.parameter_numel.astype(int) > 0).all()
    )
    state_hash_ok = True
    for row in gradient_state.to_dict("records"):
        path = Path(row["checkpoint"])
        state_hash_ok = (
            state_hash_ok
            and path.is_file()
            and checkpoint_sha256(path) == row["checkpoint_sha256"]
            and bool(row["parameters_unchanged"])
            and row["state_sha_before"] == row["state_sha_after"]
        )
    checks["no_parameter_updates"] = bool(
        state_hash_ok and summary["gradient_models_unchanged"]
    )

    flags = infer_mechanism_flags(
        recomputed_cross, shrinkage, recomputed_gradient_delta
    )
    checks["mechanism_flags_recomputed"] = flags == summary["mechanism_flags"]
    if flags["sample_effect_consistent_across_seeds"]:
        verdict = "SHARED_SAMPLE_EFFECT_FOUND_REVIEW_MECHANISM"
    elif flags["prediction_shrinkage_supported"]:
        verdict = "PREDICTION_SHRINKAGE_FOUND_SEED_EFFECT_STILL_INCONSISTENT"
    elif flags["sam_gradient_conflict_change_consistent"]:
        verdict = "CONSISTENT_GRADIENT_CHANGE_FOUND_REVIEW_MECHANISM"
    else:
        verdict = "SEED_SPECIFIC_EFFECT_NO_SHARED_MECHANISM"
    checks["verdict_recomputed"] = summary["verdict"] == verdict

    forbidden_names = [
        path.name
        for path in root.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    ]
    checks["test_and_training_lock"] = bool(
        not source["official_test_constructed"]
        and source["test_loader_construction_count"] == 0
        and source["test_loader_traversal_count"] == 0
        and not source["new_model_training"]
        and source["optimizer_steps"] == 0
        and not summary["protocol"]["official_test_constructed"]
        and summary["protocol"]["test_loader_construction_count"] == 0
        and summary["protocol"]["test_loader_traversal_count"] == 0
        and not summary["protocol"]["new_model_training"]
        and summary["protocol"]["optimizer_steps"] == 0
        and not summary["protocol"]["training_authorized"]
        and not forbidden_names
    )

    passed = bool(all(checks.values()))
    payload = {
        "passed": passed,
        "verdict": summary["verdict"],
        "checks": checks,
    }
    output = root / "mechanism_audit_check.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("CFCompatKD mechanism audit failed.")
    print("CFCompatKD mechanism independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
