"""Frozen 50/50 Raw5-PE5 + v13 exploratory MOSI Test transfer check.

This script intentionally performs exactly one new model Test traversal: the
already-frozen v13 consensus model. Raw5-PE5 is read from the five previously
recovered frozen Test prediction CSVs. The blend weight is hard-coded to 0.5/0.5
because it was frozen from Valid before this Test check.

Protocol constraints:
- EXPLORATORY_TEST_FIXED_BLEND_EVALUATION
- TEST_ALREADY_HISTORICALLY_ACCESSED
- NO_TEST_DRIVEN_WEIGHT_TUNING
- FROZEN_WEIGHT_RAW5_0.5_V13_0.5
- no training, checkpoint selection, calibration, or Test weight search
- no sample-level Test artifact is written
- aggregate-only output
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import MMDataLoader
from evaluate_cfcompat_exploratory_test_viability_v13 import (
    DEV_SEED,
    EXPECTED_TEST_N,
    collect_model_once,
    load_consensus_model,
    load_frozen_paths,
    load_v8_model,
)
from train_cf_compat_kd import build_config
from trains.singleTask.cfcompat_crossfit_residual_consensus_utils import module_state_sha256
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES, regression_metrics, validation_objective
from utils.functions import setup_seed


VERSION = "cfcompat_pe5_v13_fixedblend_test_v1"
RUN_LABEL = "EXPLORATORY_TEST_FIXED_BLEND_EVALUATION"
RAW5_SEEDS = (1111, 1112, 1113, 1114, 1115)
MODES = ("LAV",) + MISSING_MODES
FROZEN_RAW5_WEIGHT = 0.5
FROZEN_V13_WEIGHT = 0.5

# Frozen before this Test check from the completed Valid-only audit.
FROZEN_VALID = {
    "raw5_J": 0.666755626598994,
    "v13_J": 0.6675898631413777,
    "fixed_50_50_J": 0.6595411400000255,
}

# Replayed immediately before this Test check from the recovered five frozen
# Raw5 member Test CSVs. This protects against accidentally reading another set.
EXPECTED_RAW5_TEST_J = 0.6976729532082875
RAW5_REPLAY_TOL = 2e-6
V13_REPLAY_TOL = 2e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen 50/50 Raw5-PE5 + v13 exploratory aggregate-only Test check"
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--acknowledge-exploratory-contaminated-test", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("Fixed-blend Test check fixes num_workers=1 to match the historical v13 probe.")
    if not args.acknowledge_exploratory_contaminated_test:
        parser.error(
            "This is an exploratory/contaminated Test check. Pass "
            "--acknowledge-exploratory-contaminated-test explicitly."
        )
    return args


def output_root(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / VERSION
        / cli.dataset
        / "exploratory_test_fixed_blend"
    )
    if root.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "Fixed-blend Test output already exists; inspect it instead of rerunning, "
                "or explicitly pass --overwrite for a conscious technical rerun: {}".format(root)
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def raw5_root(cli):
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_prediction_ensemble_v1"
        / cli.dataset
    )


def raw5_member_path(root: Path, seed: int) -> Path:
    return root / "online_seed{}_test_predictions.csv".format(int(seed))


def load_raw5_test(cli):
    root = raw5_root(cli)
    frames = []
    manifest = []
    reference = None
    for seed in RAW5_SEEDS:
        path = raw5_member_path(root, seed)
        if not path.is_file():
            raise FileNotFoundError("Recovered Raw5 Test member is missing: {}".format(path))
        frame = pd.read_csv(path).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
        required = {
            "sample_index", "sample_id", "label", "Seed", "Method", "Split", "SelectedBy",
            *["{}_pred".format(mode) for mode in MODES],
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError("{} lacks {}".format(path, sorted(missing)))
        if len(frame) != EXPECTED_TEST_N or frame.sample_index.nunique() != EXPECTED_TEST_N:
            raise RuntimeError("Raw5 seed{} Test sample binding differs.".format(seed))
        if set(frame.Seed.astype(int)) != {int(seed)}:
            raise RuntimeError("Raw5 seed binding differs: {}".format(path))
        if set(frame.Method.astype(str)) != {"Online"}:
            raise RuntimeError("Raw5 method binding differs: {}".format(path))
        if set(frame.Split.astype(str)) != {"test"}:
            raise RuntimeError("Raw5 split binding differs: {}".format(path))
        if set(frame.SelectedBy.astype(str)) != {"validation_J"}:
            raise RuntimeError("Raw5 selection binding differs: {}".format(path))
        if reference is None:
            reference = frame
        else:
            if not np.array_equal(
                reference.sample_index.to_numpy(np.int64), frame.sample_index.to_numpy(np.int64)
            ):
                raise RuntimeError("Raw5 sample_index binding differs across seeds.")
            if not np.array_equal(
                reference.sample_id.astype(str).to_numpy(), frame.sample_id.astype(str).to_numpy()
            ):
                raise RuntimeError("Raw5 sample_id binding differs across seeds.")
            if np.max(np.abs(reference.label.to_numpy(float) - frame.label.to_numpy(float))) > 1e-7:
                raise RuntimeError("Raw5 label binding differs across seeds.")
        frames.append(frame)
        manifest.append({
            "seed": int(seed),
            "path": str(path.resolve()),
            "sha256": checkpoint_sha256(path),
        })

    result = reference[["sample_index", "sample_id", "label"]].copy()
    for mode in MODES:
        result["{}_pred".format(mode)] = np.mean(
            np.stack([frame["{}_pred".format(mode)].to_numpy(float) for frame in frames], axis=0),
            axis=0,
        )
    return result, manifest


def frame_metrics(frame: pd.DataFrame) -> dict:
    labels = torch.as_tensor(frame.label.to_numpy(np.float32)).view(-1, 1)
    by_mode = {}
    for mode in MODES:
        pred = torch.as_tensor(frame["{}_pred".format(mode)].to_numpy(np.float32)).view(-1, 1)
        by_mode[mode] = regression_metrics(pred, labels)
    return {
        "TestJ": float(validation_objective(by_mode)),
        "MissingMacroMAE": float(np.mean([by_mode[m]["MAE"] for m in MISSING_MODES])),
        "by_mode": by_mode,
    }


def historical_v13_row(cli):
    path = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_exploratory_test_viability_v13"
        / cli.dataset
        / "exploratory_test"
        / "seed1113"
        / "exploratory_test_viability_v13_comparison.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Previous aggregate v13 exploratory comparison is required for replay binding: {}".format(path)
        )
    frame = pd.read_csv(path)
    row = frame.loc[frame.Method.astype(str).eq("adam_step_safety_v13")]
    if len(row) != 1:
        raise RuntimeError("Historical v13 aggregate row is not unique: {}".format(path))
    return path, row.iloc[0]


def verify_v13_replay(metrics: dict, historical_row: pd.Series):
    pairs = {
        "TestJ": (float(metrics["TestJ"]), float(historical_row.TestJ)),
        "MissingMacroMAE": (
            float(metrics["MissingMacroMAE"]), float(historical_row.MissingMacroMAE)
        ),
    }
    for mode in MODES:
        pairs["{}_MAE".format(mode)] = (
            float(metrics["{}_MAE".format(mode)]),
            float(historical_row["{}_MAE".format(mode)]),
        )
    diffs = {name: abs(actual - expected) for name, (actual, expected) in pairs.items()}
    if max(diffs.values()) > V13_REPLAY_TOL:
        raise RuntimeError(
            "v13 Test forward does not replay the previous aggregate-only exploratory probe: {}".format(diffs)
        )
    return diffs


def blend_frames(raw5: pd.DataFrame, v13: pd.DataFrame) -> pd.DataFrame:
    left = raw5.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    right = v13.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(left.sample_index.to_numpy(np.int64), right.sample_index.to_numpy(np.int64)):
        raise RuntimeError("Raw5/v13 Test sample_index binding failed.")
    if np.max(np.abs(left.label.to_numpy(float) - right.label.to_numpy(float))) > 1e-7:
        raise RuntimeError("Raw5/v13 Test label binding failed.")
    result = left[["sample_index", "label"]].copy()
    for mode in MODES:
        result["{}_pred".format(mode)] = (
            FROZEN_RAW5_WEIGHT * left["{}_pred".format(mode)].to_numpy(float)
            + FROZEN_V13_WEIGHT * right["{}_pred".format(mode)].to_numpy(float)
        )
    return result


def metric_row(method: str, metrics: dict) -> dict:
    row = {
        "Method": method,
        "TestJ": float(metrics["TestJ"]),
        "MissingMacroMAE": float(metrics["MissingMacroMAE"]),
    }
    for mode in MODES:
        for key in ("MAE", "Corr", "acc_2", "F1_score", "acc_7", "acc_5"):
            row["{}_{}".format(mode, key)] = float(metrics["by_mode"][mode][key])
    return row


def main():
    cli = parse_args()
    root = output_root(cli)

    print(RUN_LABEL)
    print("TEST_ALREADY_HISTORICALLY_ACCESSED")
    print("NO_TEST_DRIVEN_WEIGHT_TUNING")
    print("FROZEN_WEIGHT_RAW5_0.5_V13_0.5")

    # Offline Raw5: no new Test model traversal.
    raw5, raw5_manifest = load_raw5_test(cli)
    raw5_metrics = frame_metrics(raw5)
    raw5_diff = abs(float(raw5_metrics["TestJ"]) - EXPECTED_RAW5_TEST_J)
    if raw5_diff > RAW5_REPLAY_TOL:
        raise RuntimeError(
            "Raw5 Test replay failed: actual={} expected={} diff={}".format(
                raw5_metrics["TestJ"], EXPECTED_RAW5_TEST_J, raw5_diff
            )
        )
    print("Raw5 replay: PASS TestJ={:.12f}".format(raw5_metrics["TestJ"]))

    historical_path, historical_v13 = historical_v13_row(cli)

    # Exactly one new frozen-model Test traversal: v13.
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    args.mode = "test"
    args.is_training = False
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"test"}:
        raise RuntimeError("Fixed-blend check must construct exactly one Test split loader.")
    test_loader = loaders["test"]
    if len(test_loader.dataset) != EXPECTED_TEST_N:
        raise RuntimeError("Unexpected MOSI Test sample count: {}".format(len(test_loader.dataset)))

    bindings = load_frozen_paths(cli)
    v8_model = load_v8_model(args, bindings["sample_residual_v8"]["checkpoint"])
    expected_s0_sha = str(
        bindings["v13_summary"].get("parameter_isolation", {}).get("s0_state_sha256_before", "")
    )
    actual_s0_sha = module_state_sha256(v8_model.s0)
    if expected_s0_sha and actual_s0_sha != expected_s0_sha:
        raise RuntimeError("v8-carried S0 hash differs from frozen v13 S0.")
    s0_state = {
        key: value.detach().cpu().clone()
        for key, value in v8_model.s0.state_dict().items()
    }
    del v8_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    v13_model = load_consensus_model(args, s0_state, bindings["adam_step_safety_v13"]["checkpoints"])
    v13_frame, v13_flat_metrics = collect_model_once(v13_model, test_loader, args)
    del v13_model, test_loader, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    replay_diffs = verify_v13_replay(v13_flat_metrics, historical_v13)
    print("v13 historical aggregate replay: PASS TestJ={:.12f}".format(v13_flat_metrics["TestJ"]))

    # Convert the already-verified v13 in-memory frame to the same aggregate metric shape.
    v13_metrics = frame_metrics(v13_frame)
    if abs(v13_metrics["TestJ"] - v13_flat_metrics["TestJ"]) > 1e-7:
        raise RuntimeError("Internal v13 metric calculation mismatch.")

    blend = blend_frames(raw5, v13_frame)
    blend_metrics = frame_metrics(blend)

    # Deliberately do not persist raw5/v13/blend sample-level Test frames.
    comparison = pd.DataFrame([
        metric_row("raw5_pe5", raw5_metrics),
        metric_row("adam_step_safety_v13", v13_metrics),
        metric_row("raw5_0p5_v13_0p5_fixed_blend", blend_metrics),
    ])
    comparison["DeltaJVsRaw5"] = comparison.TestJ.astype(float) - float(raw5_metrics["TestJ"])
    comparison["DeltaMissingMacroMAEVsRaw5"] = (
        comparison.MissingMacroMAE.astype(float) - float(raw5_metrics["MissingMacroMAE"])
    )
    comparison["DeltaLAVMAEVsRaw5"] = (
        comparison.LAV_MAE.astype(float) - float(raw5_metrics["by_mode"]["LAV"]["MAE"])
    )
    comparison_path = root / "fixed_blend_test_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    blend_j = float(blend_metrics["TestJ"])
    raw5_j = float(raw5_metrics["TestJ"])
    verdict = (
        "FIXED_BLEND_GENERALIZATION_SIGNAL_POSITIVE"
        if blend_j < raw5_j
        else "FIXED_BLEND_GENERALIZATION_SIGNAL_NEGATIVE"
    )

    checkpoint_manifest = {
        "raw5_prediction_members": raw5_manifest,
        "v13_fold_manifest": str(bindings["adam_step_safety_v13"]["manifest"].resolve()),
        "v13_fold_checkpoints": [
            {
                "fold": int(i),
                "path": str(Path(path).resolve()),
                "sha256": checkpoint_sha256(path),
            }
            for i, path in enumerate(bindings["adam_step_safety_v13"]["checkpoints"])
        ],
        "v8_s0_source_checkpoint": str(Path(bindings["sample_residual_v8"]["checkpoint"]).resolve()),
        "v8_s0_state_sha256": actual_s0_sha,
        "historical_v13_aggregate_source": str(historical_path.resolve()),
    }

    summary = {
        "version": VERSION,
        "run_label": RUN_LABEL,
        "verdict": verdict,
        "frozen_valid_evidence": FROZEN_VALID,
        "frozen_weights": {
            "raw5": FROZEN_RAW5_WEIGHT,
            "v13": FROZEN_V13_WEIGHT,
        },
        "test_metrics": {
            "raw5_pe5": metric_row("raw5_pe5", raw5_metrics),
            "adam_step_safety_v13": metric_row("adam_step_safety_v13", v13_metrics),
            "fixed_blend": metric_row("raw5_0p5_v13_0p5_fixed_blend", blend_metrics),
        },
        "test_deltas_fixed_blend_vs_raw5": {
            "delta_J": blend_j - raw5_j,
            "delta_LAV_MAE": float(blend_metrics["by_mode"]["LAV"]["MAE"])
            - float(raw5_metrics["by_mode"]["LAV"]["MAE"]),
            "delta_MissingMacro_MAE": float(blend_metrics["MissingMacroMAE"])
            - float(raw5_metrics["MissingMacroMAE"]),
        },
        "replay_gates": {
            "raw5_expected_TestJ": EXPECTED_RAW5_TEST_J,
            "raw5_absolute_difference": raw5_diff,
            "raw5_passed": True,
            "v13_historical_aggregate_absolute_differences": replay_diffs,
            "v13_passed": True,
        },
        "protocol": {
            "exploratory_test_only": True,
            "test_already_historically_accessed": True,
            "this_is_not_final_unbiased_test": True,
            "weight_frozen_on_valid_before_this_test_check": True,
            "weight_search_on_test": False,
            "only_tested_weight_pair": [FROZEN_RAW5_WEIGHT, FROZEN_V13_WEIGHT],
            "training": False,
            "checkpoint_selection": False,
            "calibration": False,
            "raw5_new_test_model_forward": False,
            "v13_new_test_model_forward_count": 1,
            "sample_level_test_artifacts_written": False,
            "aggregate_artifacts_only": True,
            "future_weight_changes_must_not_use_this_test_result": True,
        },
        "checkpoint_manifest": checkpoint_manifest,
        "comparison_csv": str(comparison_path.resolve()),
    }
    summary_path = root / "fixed_blend_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Fixed 50/50 Test transfer check complete")
    print("Raw5 Test J:       {:.12f}".format(raw5_j))
    print("v13 Test J:        {:.12f}".format(v13_metrics["TestJ"]))
    print("FixedBlend Test J: {:.12f}".format(blend_j))
    print("Delta J vs Raw5:   {:+.12f}".format(blend_j - raw5_j))
    print(
        "Delta MissingMacro MAE vs Raw5: {:+.12f}".format(
            float(blend_metrics["MissingMacroMAE"]) - float(raw5_metrics["MissingMacroMAE"])
        )
    )
    print("Verdict:", verdict)
    print("sample-level Test artifacts written: False")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
