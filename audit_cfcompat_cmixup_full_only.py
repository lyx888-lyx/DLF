"""Independent artifact audit for CFCompatKD full-view-only C-Mixup v2."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cmixup_regression_utils import promotion_gate
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


MODES = ("LAV", "LA", "LV", "L")
EXPECTED_VERSION = "cfcompat_cmixup_full_only_v2"
EXPECTED_METHOD = "DLF-CFCompatKD-FullViewCMixup-v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def metrics_from_predictions(frame):
    labels = frame["label"].to_numpy(dtype=float)
    return {
        mode: float(
            np.mean(
                np.abs(
                    frame["{}_pred".format(mode)].to_numpy(dtype=float)
                    - labels
                )
            )
        )
        for mode in MODES
    }


def require_close(left, right, name, tolerance=1e-7):
    if abs(float(left) - float(right)) > tolerance:
        raise RuntimeError(
            "{} mismatch: {} vs {}".format(name, left, right)
        )


def compare_gate(actual, expected):
    if actual["passed"] != expected["passed"]:
        return False
    if actual["checks"] != expected["checks"]:
        return False
    for key in ("supporting_epoch_count", "required_supporting_epochs"):
        if int(actual[key]) != int(expected[key]):
            return False
    for key in (
        "gain_valid_J",
        "gain_valid_LAV_MAE",
        "missing_macro_degradation",
        "required_gain_valid_J",
        "required_gain_valid_LAV_MAE",
        "max_missing_macro_degradation",
    ):
        if abs(float(actual[key]) - float(expected[key])) > 1e-10:
            return False
    return True


def main():
    args = parse_args()
    root = Path(args.result_dir)
    required = {
        "summary": root / "cmixup_summary.json",
        "source": root / "cmixup_source_manifest.json",
        "epochs": root / "cmixup_epoch_metrics.csv",
        "valid": root / "cmixup_valid_predictions.csv",
        "test": root / "cmixup_test_predictions.csv",
        "comparison": root / "cmixup_comparison.csv",
        "report": root / "cmixup_report.md",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing full-view C-Mixup artifacts:\n" + "\n".join(missing)
        )

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    source = json.loads(required["source"].read_text(encoding="utf-8"))
    epochs = pd.read_csv(required["epochs"])
    valid = pd.read_csv(required["valid"])
    test = pd.read_csv(required["test"])
    comparison = pd.read_csv(required["comparison"])

    checks = {}
    checks["version_and_method"] = (
        summary["version"] == EXPECTED_VERSION
        and summary["method"] == EXPECTED_METHOD
        and source["version"] == EXPECTED_VERSION
        and source["method"] == EXPECTED_METHOD
    )
    checks["candidate_checkpoint_hash"] = (
        checkpoint_sha256(Path(source["candidate_checkpoint"]))
        == source["candidate_checkpoint_sha256"]
    )
    checks["teacher_hash"] = (
        checkpoint_sha256(Path(source["teacher_checkpoint"]))
        == source["teacher_sha256"]
    )
    checks["evaluator_hash"] = (
        checkpoint_sha256(Path(source["evaluator_checkpoint"]))
        == source["evaluator_sha256"]
    )
    checks["cache_hash"] = (
        checkpoint_sha256(Path(source["compatibility_cache"]))
        == source["compatibility_cache_sha256"]
    )
    checks["baseline_result_hash"] = (
        checkpoint_sha256(Path(source["baseline_result"]))
        == source["baseline_result_sha256"]
    )

    candidate = summary["candidate"]
    valid_mae = metrics_from_predictions(valid)
    test_mae = metrics_from_predictions(test)
    for mode in MODES:
        require_close(
            valid_mae[mode],
            candidate["valid_{}_MAE".format(mode)],
            "valid {} MAE".format(mode),
        )
        require_close(
            test_mae[mode],
            candidate["test_at_valid_best_{}_MAE".format(mode)],
            "test {} MAE".format(mode),
        )
    valid_j = 0.5 * valid_mae["LAV"] + 0.5 * np.mean(
        [valid_mae[mode] for mode in ("LA", "LV", "L")]
    )
    test_j = 0.5 * test_mae["LAV"] + 0.5 * np.mean(
        [test_mae[mode] for mode in ("LA", "LV", "L")]
    )
    require_close(valid_j, candidate["J_valid"], "Valid J")
    require_close(test_j, candidate["J_test_at_valid_best"], "Test J")
    checks["prediction_metrics_recomputed"] = True

    baseline_frame = pd.read_csv(source["baseline_result"])
    selected = baseline_frame.loc[
        baseline_frame["Seed"].astype(int).eq(int(summary["seed"]))
    ]
    if len(selected) != 1:
        raise RuntimeError("Baseline seed row is not unique.")
    baseline_row = selected.iloc[0]
    for key, value in summary["baseline"].items():
        if key in baseline_row and isinstance(value, (int, float)):
            require_close(baseline_row[key], value, "baseline {}".format(key))
    checks["baseline_binding_recomputed"] = True

    gate = promotion_gate(
        candidate, summary["baseline"], epochs.to_dict("records")
    )
    if not compare_gate(gate, summary["promotion_gate"]):
        raise RuntimeError("Promotion gate does not recompute.")
    expected_verdict = (
        "PROMOTE_SEED1111_RUN_SEED1114"
        if gate["passed"] and int(summary["seed"]) == 1111
        else (
            "PROMOTE_TWO_SEEDS_RUN_FIVE_SEEDS"
            if gate["passed"] and int(summary["seed"]) == 1114
            else "STOP_CMIXUP_SEED_GATE_FAILED"
        )
    )
    if summary["verdict"] != expected_verdict:
        raise RuntimeError("Verdict does not match the frozen Valid gate.")
    checks["summary_and_gate_recomputed"] = True

    protocol = summary["protocol"]
    checks["full_view_only_contract"] = (
        protocol["full_view_only_cmixup"] is True
        and protocol["whole_batch_label_kde_partner_pool"] is True
        and protocol["mixed_missing_view_loss"] is False
        and protocol["same_partner_full_and_missing"] is False
        and protocol["same_missing_mode_only"] is False
        and candidate["CMixupScope"] == "LAV_fusion_only"
        and candidate["MixedMissingViewLoss"] is False
        and source["cmixup_scope"] == "LAV_fusion_only"
        and source["mixed_missing_view_loss"] is False
    )
    checks["valid_only_selection"] = (
        protocol["decision_split"] == "valid"
        and protocol["test_used_for_selection"] is False
        and protocol["test_loader_traversal_count"] == 1
    )
    checks["no_inference_change"] = (
        protocol["additional_inference_parameters"] == 0
        and source["no_inference_architecture_change"] is True
    )
    checks["no_synthetic_teacher_or_compatibility"] = (
        protocol["mixed_samples_use_teacher_or_compatibility"] is False
    )
    checks["missing_objective_unchanged"] = (
        protocol["original_missing_task_and_cfcompat_losses_unchanged"] is True
    )
    checks["missing_mix_loss_zero"] = bool(
        np.allclose(epochs["mix_missing_loss"].to_numpy(dtype=float), 0.0)
    )
    checks["full_mix_active"] = bool(
        (epochs["mix_active_fraction"].to_numpy(dtype=float) > 0.95).all()
        and (epochs["mix_full_loss"].to_numpy(dtype=float) > 0).all()
    )
    test_flags = comparison.loc[
        comparison["metric"].str.startswith("test_"), "decision_metric"
    ]
    checks["comparison_valid_flags"] = bool(test_flags.eq(False).all())

    passed = all(checks.values())
    payload = {
        "passed": bool(passed),
        "verdict": summary["verdict"],
        "checks": checks,
    }
    (root / "cmixup_audit_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("Full-view C-Mixup independent audit failed.")
    print("CFCompat full-view C-Mixup v2 independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
