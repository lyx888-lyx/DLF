"""Offline benchmark of frozen ADPEP-All and historical Hybrid under missing modalities.

No MOSI DataLoader is constructed and no model forward is run.  The script reads
only already-frozen Test prediction artifacts produced before this benchmark.

ADPEP-All is evaluated exactly from its frozen Stage 9B LAV/LA/LV/L predictions.
The historical Hybrid (V7.1 anchored complementarity, LAV MAE ~= 0.6995) did not
originally define a missing-modality interface.  A missing extension is therefore
reported only when the local frozen V7.1 run proves that its validation-selected
student alpha is exactly zero.  In that case the deployed Hybrid reduces to the
frozen formula

    beta * region_simplex_committee + (1-beta) * anchor,

which can be applied without retraining or Test tuning to the already-frozen
Stage 9A per-seed CFCompatKD LA/LV/L predictions.  The LAV reconstruction must
first reproduce the historical Hybrid sample predictions and metrics; otherwise
the missing extension is marked unavailable and no Hybrid missing result is
reported.

Only aggregate outputs are written.  Existing sample-level Test prediction files
are read in memory but never copied or rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import torch

from trains.singleTask.missing_utils import regression_metrics


SEEDS = (1111, 1112, 1113, 1114, 1115)
MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
EXPECTED_TEST_N = 686
REGION_CENTERS = np.asarray((-2.25, -1.0, 0.0, 1.0, 2.25), dtype=np.float64)
HISTORICAL_HYBRID_EXPECTED = {
    "MAE": 0.6995,
    "Corr": 0.7942,
    "acc_2": 0.8491,
    "F1_score": 0.8484,
    "acc_7": 0.4781,
    "acc_5": 0.5394,
}
HISTORICAL_METRIC_TOLERANCE = 5e-4
LAV_RECONSTRUCTION_MAX_ABS_TOLERANCE = 2e-5
ORIGINAL_REPLAY_TOLERANCE = 2e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline existing-strong-method missing-modality benchmark"
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--hybrid-root", default="result/complementarity_v71/mosi/seed_1111")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_frame(frame: pd.DataFrame, columns) -> None:
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite values in columns {}".format(list(columns)))


def _read_prediction_frame(path: Path, split="test") -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "sample_index",
        "sample_id",
        "label",
        "Split",
        *["{}_pred".format(mode) for mode in MODES],
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Prediction frame {} lacks {}".format(path, sorted(missing)))
    if set(frame.Split.astype(str)) != {split}:
        raise RuntimeError("Split mixing in {}".format(path))
    frame["sample_index"] = frame.sample_index.astype(int)
    frame = frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != EXPECTED_TEST_N or frame.sample_index.nunique() != EXPECTED_TEST_N:
        raise RuntimeError("Unexpected Test binding in {}".format(path))
    _finite_frame(
        frame,
        ["sample_index", "label"] + ["{}_pred".format(mode) for mode in MODES],
    )
    return frame


def _bind_frames(reference: pd.DataFrame, candidate: pd.DataFrame, check_label=True) -> None:
    if not np.array_equal(
        reference.sample_index.to_numpy(np.int64), candidate.sample_index.to_numpy(np.int64)
    ):
        raise RuntimeError("sample_index binding differs")
    if not np.array_equal(
        reference.sample_id.astype(str).to_numpy(), candidate.sample_id.astype(str).to_numpy()
    ):
        raise RuntimeError("sample_id binding differs")
    if check_label and np.max(
        np.abs(reference.label.to_numpy(float) - candidate.label.to_numpy(float))
    ) > 1e-7:
        raise RuntimeError("label binding differs")


def load_stage9_members(result_root: Path, dataset: str):
    root = result_root / "missing_baseline" / "cfcompat_prediction_ensemble_v1" / dataset
    manifest_path = root / "individual_predictions_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Frozen Stage9A manifest missing: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = manifest.get("Members", [])
    if [int(row["Seed"]) for row in members] != list(SEEDS):
        raise RuntimeError("Stage9A member seed order is not frozen 1111..1115")
    frames = {}
    records = {}
    reference = None
    for row in members:
        seed = int(row["Seed"])
        if str(row.get("SelectedBy")) != "validation_J":
            raise RuntimeError("Stage9A member was not validation-selected: seed{}".format(seed))
        pred_record = row["Predictions"]["test"]
        path = root / Path(str(pred_record["Path"])).name
        if sha256(path) != str(pred_record["SHA256"]):
            raise RuntimeError("Stage9A prediction SHA mismatch: {}".format(path))
        frame = _read_prediction_frame(path)
        if reference is None:
            reference = frame
        else:
            _bind_frames(reference, frame, check_label=True)
        frames[seed] = frame
        records[seed] = {
            "checkpoint": str(row.get("Checkpoint", "")),
            "checkpoint_sha256": str(row.get("CheckpointSHA256", "")),
            "prediction_path": str(path.resolve()),
            "prediction_sha256": sha256(path),
        }
    return root, frames, records, reference


def load_adpep_all(result_root: Path, dataset: str, label_source: pd.DataFrame):
    root = result_root / "missing_baseline" / "anchor_decision_preserving_ensemble_v1" / dataset
    manifest_path = root / "output_prediction_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Frozen ADPEP output manifest missing: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("MainMethod") != "adpep_all":
        raise RuntimeError("ADPEP manifest main method is not adpep_all")
    if not manifest.get("PredictionsFrozenBeforeMetricLabelsLoaded", False):
        raise RuntimeError("ADPEP predictions were not frozen before metrics")
    records = [
        row for row in manifest.get("Predictions", [])
        if row.get("Variant") == "adpep_all" and row.get("Split") == "test"
    ]
    if len(records) != 1:
        raise RuntimeError("Unique frozen ADPEP-All Test prediction not found")
    record = records[0]
    path = root / Path(str(record["Path"])).name
    if sha256(path) != str(record["SHA256"]):
        raise RuntimeError("ADPEP-All Test prediction SHA mismatch")
    frame = pd.read_csv(path).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    required = {"sample_index", "sample_id", *["{}_pred".format(mode) for mode in MODES]}
    if required.difference(frame.columns):
        raise ValueError("ADPEP-All Test columns are incomplete")
    if "label" in frame.columns:
        raise RuntimeError("Frozen ADPEP-All prediction unexpectedly contains Test labels")
    if len(frame) != EXPECTED_TEST_N:
        raise RuntimeError("ADPEP-All Test sample count changed")
    frame["sample_index"] = frame.sample_index.astype(int)
    if not np.array_equal(
        frame.sample_index.to_numpy(np.int64), label_source.sample_index.to_numpy(np.int64)
    ) or not np.array_equal(
        frame.sample_id.astype(str).to_numpy(), label_source.sample_id.astype(str).to_numpy()
    ):
        raise RuntimeError("ADPEP-All/Stage9A Test sample binding differs")
    frame.insert(2, "label", label_source.label.to_numpy(dtype=np.float64))
    return frame, {
        "prediction_path": str(path.resolve()),
        "prediction_sha256": sha256(path),
        "manifest": str(manifest_path.resolve()),
        "anchor_seed": int(manifest["AnchorSeed"]),
    }


def metric_bundle(frame: pd.DataFrame) -> dict:
    labels = torch.as_tensor(frame.label.to_numpy(dtype=np.float32)).view(-1, 1)
    by_mode = {}
    for mode in MODES:
        pred = torch.as_tensor(
            frame["{}_pred".format(mode)].to_numpy(dtype=np.float32)
        ).view(-1, 1)
        by_mode[mode] = regression_metrics(pred, labels)
    missing_macro = {
        key: float(np.mean([by_mode[mode][key] for mode in MISSING_MODES]))
        for key in by_mode["LAV"].keys()
    }
    test_j = 0.5 * float(by_mode["LAV"]["MAE"]) + 0.5 * float(missing_macro["MAE"])
    result = {
        "TestJ": test_j,
        "MissingMacroMAE": float(missing_macro["MAE"]),
    }
    for mode in MODES:
        for key, value in by_mode[mode].items():
            result["{}_{}".format(mode, key)] = float(value)
    for key, value in missing_macro.items():
        result["MissingMacro_{}".format(key)] = float(value)
    return result


def paired_missing_vs_original(candidate: pd.DataFrame, original: pd.DataFrame) -> dict:
    _bind_frames(original, candidate, check_label=True)
    labels = original.label.to_numpy(dtype=np.float64)
    gains = []
    for mode in MISSING_MODES:
        original_error = np.abs(original["{}_pred".format(mode)].to_numpy(float) - labels)
        candidate_error = np.abs(candidate["{}_pred".format(mode)].to_numpy(float) - labels)
        gains.append(original_error - candidate_error)
    gain = np.concatenate(gains)
    lav_gain = (
        np.abs(original.LAV_pred.to_numpy(float) - labels)
        - np.abs(candidate.LAV_pred.to_numpy(float) - labels)
    )
    return {
        "PairedMissingN": int(gain.size),
        "PairedMissingMeanGainVsOriginal": float(gain.mean()),
        "PairedMissingWinRateVsOriginal": float((gain > 0.0).mean()),
        "PairedMissingHarmRateVsOriginal": float((gain < 0.0).mean()),
        "PairedMissingGainOver002RateVsOriginal": float((gain > 0.02).mean()),
        "PairedMissingHarmOver002RateVsOriginal": float((gain < -0.02).mean()),
        "PairedMissingSevereHarmOver010RateVsOriginal": float((gain < -0.10).mean()),
        "PairedLAVMeanGainVsOriginal": float(lav_gain.mean()),
        "PairedLAVWinRateVsOriginal": float((lav_gain > 0.0).mean()),
    }


def _soft_region_membership(anchor: np.ndarray, temperature: float) -> np.ndarray:
    logits = -np.abs(anchor.reshape(-1, 1) - REGION_CENTERS.reshape(1, -1)) / max(
        float(temperature), 1e-6
    )
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def _extract_seed(path_text: str) -> int:
    matches = re.findall(r"seed[_-]?(111[1-5])", str(path_text), flags=re.IGNORECASE)
    if not matches:
        raise RuntimeError("Could not extract frozen seed from teacher path: {}".format(path_text))
    return int(matches[-1])


def build_hybrid_missing_extension(hybrid_root: Path, stage9_frames: Dict[int, pd.DataFrame]):
    summary_path = hybrid_root / "complementarity_v71_summary.json"
    weights_path = hybrid_root / "v71_committee_weights.json"
    historical_predictions_path = hybrid_root / "complementarity_v71_predictions.csv"
    required = (summary_path, weights_path, historical_predictions_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return None, {
            "available": False,
            "reason": "missing_historical_hybrid_artifacts",
            "missing": missing,
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    weights_payload = json.loads(weights_path.read_text(encoding="utf-8"))
    alpha = float(summary.get("student_policy", {}).get("alpha", float("nan")))
    policy = summary.get("hybrid_policy", {})
    committee = summary.get("committee", {})
    if not math.isfinite(alpha):
        return None, {"available": False, "reason": "historical_student_alpha_missing"}
    if abs(alpha) > 1e-12:
        return None, {
            "available": False,
            "reason": "historical_hybrid_contains_nonzero_student_residual_and_has_no_frozen_missing_interface",
            "student_alpha": alpha,
        }
    if str(policy.get("committee")) != "region_simplex" or str(committee.get("selected")) != "region_simplex":
        return None, {
            "available": False,
            "reason": "historical_hybrid_is_not_frozen_region_simplex",
            "hybrid_policy": policy,
            "committee_selected": committee.get("selected"),
        }
    beta = float(policy.get("beta", float("nan")))
    if not math.isfinite(beta) or abs(beta - 0.5) > 1e-12:
        return None, {
            "available": False,
            "reason": "historical_hybrid_beta_differs_from_frozen_0p5",
            "beta": beta,
        }
    base_index = int(summary.get("base_teacher_index", -1))
    teacher_paths = list(summary.get("teacher_paths", []))
    teacher_seeds = [_extract_seed(value) for value in teacher_paths]
    if teacher_seeds != list(SEEDS) or base_index != 3:
        return None, {
            "available": False,
            "reason": "historical_teacher_pool_or_anchor_differs_from_frozen_reference",
            "teacher_seeds": teacher_seeds,
            "base_teacher_index": base_index,
        }

    region_weights = np.asarray(committee.get("region_weights"), dtype=np.float64)
    weights_file_region = np.asarray(weights_payload.get("region_weights"), dtype=np.float64)
    if region_weights.shape != (5, 5) or weights_file_region.shape != (5, 5):
        raise RuntimeError("Historical Hybrid region weights must be 5x5")
    if not np.allclose(region_weights, weights_file_region, atol=1e-10, rtol=0.0):
        raise RuntimeError("Hybrid summary/weights-file region weights differ")
    temperature = float(weights_payload.get("region_temperature", 0.55))

    reference = stage9_frames[SEEDS[0]]
    result = reference[["sample_index", "sample_id", "label"]].copy()
    for mode in MODES:
        predictions = np.column_stack(
            [stage9_frames[seed]["{}_pred".format(mode)].to_numpy(np.float64) for seed in SEEDS]
        )
        anchor = predictions[:, base_index]
        membership = _soft_region_membership(anchor, temperature)
        sample_weights = membership @ region_weights
        committee_prediction = np.sum(predictions * sample_weights, axis=1)
        # alpha == 0 => historical calibrated student equals the frozen anchor.
        result["{}_pred".format(mode)] = beta * committee_prediction + (1.0 - beta) * anchor

    historical = pd.read_csv(historical_predictions_path)
    required_hist = {"sample_id", "label", "hybrid_valid_selected"}
    if required_hist.difference(historical.columns):
        raise RuntimeError("Historical Hybrid prediction file is incomplete")
    hist_metric_frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(historical), dtype=np.int64),
            "sample_id": historical.sample_id.astype(str),
            "label": historical.label.astype(float),
            "LAV_pred": historical.hybrid_valid_selected.astype(float),
            "LA_pred": historical.hybrid_valid_selected.astype(float),
            "LV_pred": historical.hybrid_valid_selected.astype(float),
            "L_pred": historical.hybrid_valid_selected.astype(float),
        }
    )
    historical_metrics = metric_bundle(hist_metric_frame)
    observed = {
        "MAE": historical_metrics["LAV_MAE"],
        "Corr": historical_metrics["LAV_Corr"],
        "acc_2": historical_metrics["LAV_acc_2"],
        "F1_score": historical_metrics["LAV_F1_score"],
        "acc_7": historical_metrics["LAV_acc_7"],
        "acc_5": historical_metrics["LAV_acc_5"],
    }
    metric_diffs = {
        key: abs(float(observed[key]) - float(HISTORICAL_HYBRID_EXPECTED[key]))
        for key in HISTORICAL_HYBRID_EXPECTED
    }
    if any(value > HISTORICAL_METRIC_TOLERANCE for value in metric_diffs.values()):
        return None, {
            "available": False,
            "reason": "historical_hybrid_reference_metrics_do_not_reproduce",
            "observed": observed,
            "expected": HISTORICAL_HYBRID_EXPECTED,
            "absolute_differences": metric_diffs,
        }

    historical_binding = historical[["sample_id", "label", "hybrid_valid_selected"]].copy()
    historical_binding["sample_id"] = historical_binding.sample_id.astype(str)
    reconstructed = result[["sample_id", "label", "LAV_pred"]].copy()
    reconstructed["sample_id"] = reconstructed.sample_id.astype(str)
    merged = historical_binding.merge(
        reconstructed,
        on="sample_id",
        how="inner",
        suffixes=("_historical", "_reconstructed"),
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_TEST_N:
        return None, {
            "available": False,
            "reason": "historical_and_stage9_sample_ids_do_not_bind",
            "bound_rows": int(len(merged)),
        }
    label_diff = float(
        np.max(np.abs(merged.label_historical.to_numpy(float) - merged.label_reconstructed.to_numpy(float)))
    )
    max_abs_prediction_diff = float(
        np.max(
            np.abs(
                merged.hybrid_valid_selected.to_numpy(float)
                - merged.LAV_pred.to_numpy(float)
            )
        )
    )
    if label_diff > 1e-7 or max_abs_prediction_diff > LAV_RECONSTRUCTION_MAX_ABS_TOLERANCE:
        return None, {
            "available": False,
            "reason": "frozen_lav_hybrid_formula_does_not_reconstruct_historical_predictions",
            "max_abs_label_difference": label_diff,
            "max_abs_prediction_difference": max_abs_prediction_diff,
            "tolerance": LAV_RECONSTRUCTION_MAX_ABS_TOLERANCE,
        }

    return result, {
        "available": True,
        "definition": "frozen_v71_region_simplex_plus_anchor_missing_extension",
        "historical_student_alpha": alpha,
        "beta": beta,
        "base_teacher_index": base_index,
        "base_teacher_seed": teacher_seeds[base_index],
        "teacher_seeds": teacher_seeds,
        "region_temperature": temperature,
        "historical_reference_metrics": observed,
        "historical_metric_absolute_differences": metric_diffs,
        "lav_reconstruction_max_abs_difference": max_abs_prediction_diff,
        "lav_reconstruction_tolerance": LAV_RECONSTRUCTION_MAX_ABS_TOLERANCE,
        "summary_path": str(summary_path.resolve()),
        "weights_path": str(weights_path.resolve()),
        "historical_predictions_path": str(historical_predictions_path.resolve()),
    }


def _common_row(method: str, metrics: Mapping, provenance: str, paired=None) -> dict:
    row = {
        "Method": method,
        "Provenance": provenance,
        "TestJ": float(metrics["TestJ"]),
        "MissingMacroMAE": float(metrics["MissingMacroMAE"]),
        "LAV_MAE": float(metrics["LAV_MAE"]),
        "LA_MAE": float(metrics["LA_MAE"]),
        "LV_MAE": float(metrics["LV_MAE"]),
        "L_MAE": float(metrics["L_MAE"]),
        "LAV_Corr": float(metrics.get("LAV_Corr", np.nan)),
        "MissingMacro_Corr": float(metrics.get("MissingMacro_Corr", np.nan)),
        "LAV_acc_2": float(metrics.get("LAV_acc_2", np.nan)),
        "MissingMacro_acc_2": float(metrics.get("MissingMacro_acc_2", np.nan)),
        "LAV_F1": float(metrics.get("LAV_F1", metrics.get("LAV_F1_score", np.nan))),
        "MissingMacro_F1": float(
            metrics.get("MissingMacro_F1", metrics.get("MissingMacro_F1_score", np.nan))
        ),
    }
    if paired:
        row.update(paired)
    return row


def previous_method_rows(result_root: Path, dataset: str):
    root = (
        result_root
        / "missing_baseline"
        / "cfcompat_exploratory_test_viability_v13"
        / dataset
        / "exploratory_test"
        / "seed1113"
    )
    summary_path = root / "exploratory_test_viability_v13_summary.json"
    comparison_path = root / "exploratory_test_viability_v13_comparison.csv"
    if not summary_path.is_file() or not comparison_path.is_file():
        raise FileNotFoundError("Previous exploratory aggregate result is required under {}".format(root))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(comparison_path)
    method_metrics = summary.get("method_test_metrics", {})
    rows = []
    for record in comparison.itertuples(index=False):
        method = str(record.Method)
        metrics = dict(method_metrics.get(method, {}))
        # Backfill the common MAE fields from the comparison CSV.
        for key in ("TestJ", "MissingMacroMAE", "LAV_MAE", "LA_MAE", "LV_MAE", "L_MAE"):
            metrics[key] = float(getattr(record, key))
        if all("{}_Corr".format(mode) in metrics for mode in MISSING_MODES):
            metrics["MissingMacro_Corr"] = float(
                np.mean([metrics["{}_Corr".format(mode)] for mode in MISSING_MODES])
            )
        if all("{}_acc_2".format(mode) in metrics for mode in MISSING_MODES):
            metrics["MissingMacro_acc_2"] = float(
                np.mean([metrics["{}_acc_2".format(mode)] for mode in MISSING_MODES])
            )
        if all("{}_F1".format(mode) in metrics for mode in MISSING_MODES):
            metrics["MissingMacro_F1"] = float(
                np.mean([metrics["{}_F1".format(mode)] for mode in MISSING_MODES])
            )
        rows.append(_common_row(method, metrics, "previous_exploratory_probe_aggregate"))
    return rows, summary, comparison


def main():
    cli = parse_args()
    result_root = Path(cli.result_root)
    output_root = (
        result_root
        / "missing_baseline"
        / "adpep_hybrid_missing_benchmark_v1"
        / cli.dataset
        / "offline_frozen_test_predictions"
    )
    if output_root.exists():
        if not cli.overwrite:
            raise FileExistsError("Output exists; inspect it or use --overwrite: {}".format(output_root))
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    previous_rows, previous_summary, previous_comparison = previous_method_rows(
        result_root, cli.dataset
    )
    stage9_root, stage9_frames, stage9_records, label_source = load_stage9_members(
        result_root, cli.dataset
    )

    # Seed1113 Stage9A must replay the Original CFCompat aggregate from the prior probe.
    original_frame = stage9_frames[1113]
    original_metrics = metric_bundle(original_frame)
    previous_original = previous_comparison.loc[
        previous_comparison.Method.astype(str).eq("original_cfcompat_v1")
    ]
    if len(previous_original) != 1:
        raise RuntimeError("Previous Original CFCompat row is not unique")
    previous_original = previous_original.iloc[0]
    original_replay_diffs = {
        key: abs(float(original_metrics[key]) - float(previous_original[key]))
        for key in ("TestJ", "MissingMacroMAE", "LAV_MAE", "LA_MAE", "LV_MAE", "L_MAE")
    }
    if any(value > ORIGINAL_REPLAY_TOLERANCE for value in original_replay_diffs.values()):
        raise RuntimeError(
            "Stage9A seed1113 does not replay previous Original CFCompat aggregate: {}".format(
                original_replay_diffs
            )
        )

    adpep_frame, adpep_binding = load_adpep_all(result_root, cli.dataset, label_source)
    adpep_metrics = metric_bundle(adpep_frame)
    adpep_paired = paired_missing_vs_original(adpep_frame, original_frame)

    # Replay the frozen Stage9B aggregate if present.
    adpep_metric_path = (
        result_root
        / "missing_baseline"
        / "anchor_decision_preserving_ensemble_v1"
        / cli.dataset
        / "adpep_metrics.csv"
    )
    adpep_aggregate_replay = None
    if adpep_metric_path.is_file():
        saved = pd.read_csv(adpep_metric_path)
        saved_j = saved.loc[
            saved.Split.astype(str).eq("test")
            & saved.Method.astype(str).eq("ADPEP-All")
            & saved.Mode.astype(str).eq("LAV"),
            "J",
        ]
        saved_lav = saved.loc[
            saved.Split.astype(str).eq("test")
            & saved.Method.astype(str).eq("ADPEP-All")
            & saved.Mode.astype(str).eq("LAV"),
            "MAE",
        ]
        saved_missing = saved.loc[
            saved.Split.astype(str).eq("test")
            & saved.Method.astype(str).eq("ADPEP-All")
            & saved.Mode.astype(str).eq("MissingMacro"),
            "MAE",
        ]
        if len(saved_j) == len(saved_lav) == len(saved_missing) == 1:
            adpep_aggregate_replay = {
                "delta_J": float(adpep_metrics["TestJ"] - saved_j.iloc[0]),
                "delta_LAV_MAE": float(adpep_metrics["LAV_MAE"] - saved_lav.iloc[0]),
                "delta_MissingMacro_MAE": float(
                    adpep_metrics["MissingMacroMAE"] - saved_missing.iloc[0]
                ),
            }
            if max(abs(value) for value in adpep_aggregate_replay.values()) > 2e-7:
                raise RuntimeError("ADPEP frozen aggregate replay differs: {}".format(adpep_aggregate_replay))

    hybrid_frame, hybrid_status = build_hybrid_missing_extension(
        Path(cli.hybrid_root), stage9_frames
    )

    rows = list(previous_rows)
    rows.append(
        _common_row(
            "adpep_all",
            adpep_metrics,
            "exact_frozen_stage9b_test_predictions",
            adpep_paired,
        )
    )
    hybrid_metrics = None
    hybrid_paired = None
    if hybrid_frame is not None:
        hybrid_metrics = metric_bundle(hybrid_frame)
        hybrid_paired = paired_missing_vs_original(hybrid_frame, original_frame)
        rows.append(
            _common_row(
                "hybrid_v71_missing_extension",
                hybrid_metrics,
                "frozen_v71_formula_extended_to_stage9a_missing_predictions_after_exact_LAV_replay",
                hybrid_paired,
            )
        )

    comparison = pd.DataFrame(rows)
    reference_j = float(previous_original.TestJ)
    reference_missing = float(previous_original.MissingMacroMAE)
    comparison["DeltaJVsOriginal"] = comparison.TestJ.astype(float) - reference_j
    comparison["DeltaMissingMacroMAEVsOriginal"] = (
        comparison.MissingMacroMAE.astype(float) - reference_missing
    )
    comparison["RankByTestJ"] = comparison.TestJ.rank(method="min", ascending=True).astype(int)
    comparison["RankByMissingMacroMAE"] = comparison.MissingMacroMAE.rank(
        method="min", ascending=True
    ).astype(int)
    comparison = comparison.sort_values(["TestJ", "MissingMacroMAE"], kind="mergesort")

    comparison_path = output_root / "adpep_hybrid_missing_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "version": "adpep_hybrid_missing_benchmark_v1",
        "protocol": {
            "dataset": cli.dataset,
            "offline_only": True,
            "new_test_loader_constructed": False,
            "new_model_forward": False,
            "new_training": False,
            "new_checkpoint_selection": False,
            "new_test_tuning": False,
            "sample_level_output_written": False,
            "adpep_definition": "exact_frozen_stage9b_predictions",
            "hybrid_definition": (
                "frozen_v71_formula_extension_only_if_historical_alpha_zero_and_LAV_exactly_replays"
            ),
        },
        "original_cfcompat_stage9a_replay_absolute_differences": original_replay_diffs,
        "stage9a_root": str(stage9_root.resolve()),
        "stage9a_members": stage9_records,
        "adpep_binding": adpep_binding,
        "adpep_metrics": adpep_metrics,
        "adpep_paired_vs_original": adpep_paired,
        "adpep_saved_aggregate_replay": adpep_aggregate_replay,
        "hybrid_status": hybrid_status,
        "hybrid_metrics": hybrid_metrics,
        "hybrid_paired_vs_original": hybrid_paired,
        "comparison_csv": str(comparison_path.resolve()),
    }
    summary_path = output_root / "adpep_hybrid_missing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("ADPEP/Hybrid missing benchmark complete")
    print("offline-only: True")
    print("new Test DataLoader constructed: False")
    print("new model forward: False")
    print("ADPEP-All TestJ: {:.9f}".format(adpep_metrics["TestJ"]))
    print("ADPEP-All MissingMacro MAE: {:.9f}".format(adpep_metrics["MissingMacroMAE"]))
    print("ADPEP-All paired missing mean gain vs Original: {:+.9f}".format(
        adpep_paired["PairedMissingMeanGainVsOriginal"]
    ))
    print("Hybrid missing extension available:", bool(hybrid_status.get("available", False)))
    if hybrid_metrics is not None:
        print("Hybrid-MissingExt TestJ: {:.9f}".format(hybrid_metrics["TestJ"]))
        print("Hybrid-MissingExt MissingMacro MAE: {:.9f}".format(hybrid_metrics["MissingMacroMAE"]))
        print("Hybrid LAV reconstruction max abs diff: {:.3e}".format(
            hybrid_status["lav_reconstruction_max_abs_difference"]
        ))
        print("Hybrid-MissingExt paired missing mean gain vs Original: {:+.9f}".format(
            hybrid_paired["PairedMissingMeanGainVsOriginal"]
        ))
    else:
        print("Hybrid unavailable reason:", hybrid_status.get("reason"))
    print("comparison:", comparison_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
