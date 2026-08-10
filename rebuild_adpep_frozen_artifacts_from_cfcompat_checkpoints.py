"""Rebuild historical Stage9A/Stage9B input artifacts from frozen CFCompat checkpoints.

This is a recovery utility for a migrated workstation where historical result/
artifacts are absent but the five validation-selected CFCompatKD checkpoints are
still present.  It performs inference only; it does not train, select, or tune a
model.  The recovered Stage9A five-member ensemble must replay the historical
fixed PE5 objectives before any Stage9B ADPEP projection is allowed.

The script intentionally uses the same five checkpoint paths recorded by the
frozen MOSI anchored-complementarity reference.  Diagnostic/best-Test checkpoints
are never considered.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from train_cf_compat_kd import build_config, prediction_rows
from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    EXPECTED_J,
    METRICS,
    PE5_MODES,
    PE5_SEEDS,
    PE5_SPLITS,
    equal_prediction_ensemble,
    metric_rows,
    metrics_from_predictions,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MissingModalityWrapper, build_single_split_loader
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


HISTORICAL_CHECKPOINTS = {
    1111: Path("pt/missing_baseline/cf_compat_kd_v1/DLF_mosi_seed1111_best_valid.pth"),
    1112: Path("pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1112/DLF_mosi_seed1112_best_valid.pth"),
    1113: Path("pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1113/DLF_mosi_seed1113_best_valid.pth"),
    1114: Path("pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1114/DLF_mosi_seed1114_best_valid.pth"),
    1115: Path("pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1115/DLF_mosi_seed1115_best_valid.pth"),
}
EXPECTED_TEST_N = 686
HISTORICAL_PE5_J_TOLERANCE = 1e-4
ORIGINAL_1113_REPLAY_TOLERANCE = 2e-6


def parse_args():
    parser = argparse.ArgumentParser(description="Recover frozen Stage9A inputs from five CFCompat best-valid checkpoints")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_workers != 0:
        parser.error("Recovery fixes num_workers=0 for deterministic inference.")
    return args


def _cli_for_build_config(args):
    return SimpleNamespace(
        dataset=args.dataset,
        config_file=args.config_file,
        gpu_ids=list(args.gpu_ids),
    )


def _assert_checkpoint_set():
    missing = [str(path) for path in HISTORICAL_CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Historical CFCompat best-valid checkpoint(s) missing: {}".format(missing))
    for seed, path in HISTORICAL_CHECKPOINTS.items():
        lowered = str(path).lower()
        if "diagnostic" in lowered or "best_test" in lowered or "smoke" in lowered:
            raise RuntimeError("Forbidden recovery checkpoint for seed{}: {}".format(seed, path))
        if path.name != "DLF_mosi_seed{}_best_valid.pth".format(seed):
            raise RuntimeError("Unexpected frozen checkpoint name for seed{}: {}".format(seed, path))


def _build_student(config, checkpoint: Path):
    state = torch.load(checkpoint, map_location=config.device)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint is not a state_dict: {}".format(checkpoint))
    if any("teacher" in str(key).lower() for key in state):
        raise RuntimeError("Student checkpoint contains Teacher state: {}".format(checkpoint))
    student = MissingModalityWrapper(
        DLF(config).to(config.device),
        config.feature_dims[1],
        config.feature_dims[2],
    ).to(config.device)
    student.load_state_dict(state, strict=True)
    student.eval()
    return student


def _prediction_path(root: Path, seed: int, split: str) -> Path:
    return root / "online_seed{}_{}_predictions.csv".format(seed, split)


def _infer_member(args, seed: int, checkpoint: Path, root: Path):
    setup_seed(seed)
    config = build_config(_cli_for_build_config(args), seed)
    student = _build_student(config, checkpoint)
    frames = {}
    records = {}
    for split in PE5_SPLITS:
        loader = build_single_split_loader(config, split, args.num_workers)
        frame = prediction_rows(student, loader, config.device)
        frame["Seed"] = int(seed)
        frame["Method"] = "Online"
        frame["Split"] = str(split)
        frame["SelectedBy"] = "validation_J"
        path = _prediction_path(root, seed, split)
        frame.to_csv(path, index=False)
        frames[split] = frame
        records[split] = {
            "Path": str(path),
            "SHA256": checkpoint_sha256(path),
            "Rows": int(len(frame)),
        }
        del loader
    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frames, records


def _ensemble_metric_rows(frame: pd.DataFrame):
    by_mode, j_value = metrics_from_predictions(frame)
    split = str(frame.Split.iloc[0])
    rows = []
    for mode in PE5_MODES + ("MissingMacro",):
        for metric in METRICS:
            rows.append(
                {
                    "Split": split,
                    "Mode": mode,
                    "Metric": metric,
                    "EnsembleValue": float(by_mode[mode][metric]),
                    "J": float(j_value),
                }
            )
    return rows


def _comparison_rows(individual: pd.DataFrame, ensembles: dict):
    rows = []
    for split in PE5_SPLITS:
        by_mode, ensemble_j = metrics_from_predictions(ensembles[split])
        for mode in PE5_MODES + ("MissingMacro",):
            local = individual.loc[
                individual.Split.astype(str).eq(split)
                & individual.Mode.astype(str).eq(mode)
            ]
            for metric in METRICS:
                values = local[metric].astype(float)
                ensemble_value = float(by_mode[mode][metric])
                rows.append(
                    {
                        "Split": split,
                        "Mode": mode,
                        "Metric": metric,
                        "EnsembleValue": ensemble_value,
                        "MeanIndividualValue": float(values.mean()),
                        "DeltaVsMeanIndividual": ensemble_value - float(values.mean()),
                    }
                )
        objective = local = individual.loc[
            individual.Split.astype(str).eq(split)
            & individual.Mode.astype(str).eq("LAV")
        ]["J"].astype(float)
        rows.append(
            {
                "Split": split,
                "Mode": "Objective",
                "Metric": "J",
                "EnsembleValue": float(ensemble_j),
                "MeanIndividualValue": float(objective.mean()),
                "DeltaVsMeanIndividual": float(ensemble_j) - float(objective.mean()),
            }
        )
    return rows


def _verify_seed1113_against_recent_probe(result_root: Path, recovered: pd.DataFrame):
    path = (
        result_root
        / "missing_baseline"
        / "cfcompat_exploratory_test_viability_v13"
        / "mosi"
        / "exploratory_test"
        / "seed1113"
        / "exploratory_test_viability_v13_comparison.csv"
    )
    if not path.is_file():
        return {"available": False, "path": str(path)}
    comparison = pd.read_csv(path)
    row = comparison.loc[comparison.Method.astype(str).eq("original_cfcompat_v1")]
    if len(row) != 1:
        raise RuntimeError("Recent exploratory Original-CFCompat row is not unique")
    row = row.iloc[0]
    metrics, j_value = metrics_from_predictions(recovered)
    actual = {
        "TestJ": float(j_value),
        "MissingMacroMAE": float(metrics["MissingMacro"]["MAE"]),
        "LAV_MAE": float(metrics["LAV"]["MAE"]),
        "LA_MAE": float(metrics["LA"]["MAE"]),
        "LV_MAE": float(metrics["LV"]["MAE"]),
        "L_MAE": float(metrics["L"]["MAE"]),
    }
    diffs = {key: abs(actual[key] - float(row[key])) for key in actual}
    if max(diffs.values()) > ORIGINAL_1113_REPLAY_TOLERANCE:
        raise RuntimeError("Recovered seed1113 does not replay recent Original CFCompat aggregate: {}".format(diffs))
    return {
        "available": True,
        "path": str(path),
        "tolerance": ORIGINAL_1113_REPLAY_TOLERANCE,
        "absolute_differences": diffs,
        "passed": True,
    }


def main():
    args = parse_args()
    _assert_checkpoint_set()
    result_root = Path(args.result_root)
    root = result_root / "missing_baseline" / "cfcompat_prediction_ensemble_v1" / args.dataset
    if root.exists():
        if not args.overwrite:
            raise FileExistsError("Recovery output already exists; inspect it or pass --overwrite: {}".format(root))
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    all_frames = {split: [] for split in PE5_SPLITS}
    member_manifest = []
    individual_rows = []
    for seed in PE5_SEEDS:
        checkpoint = HISTORICAL_CHECKPOINTS[int(seed)]
        checkpoint_sha = checkpoint_sha256(checkpoint)
        print("recovering seed{} from {}".format(seed, checkpoint))
        frames, prediction_records = _infer_member(args, int(seed), checkpoint, root)
        for split in PE5_SPLITS:
            all_frames[split].append(frames[split])
            individual_rows.extend(metric_rows(frames[split], "Online", seed=int(seed)))
        member_manifest.append(
            {
                "Seed": int(seed),
                "Checkpoint": str(checkpoint),
                "CheckpointSHA256": checkpoint_sha,
                "SelectedBy": "validation_J",
                "RecoverySource": "frozen_historical_cfcompat_best_valid_checkpoint",
                "Predictions": prediction_records,
            }
        )

    individual = pd.DataFrame(individual_rows)
    individual.to_csv(root / "individual_model_metrics.csv", index=False)

    ensembles = {}
    ensemble_metric_rows = []
    pe5_j = {}
    for split in PE5_SPLITS:
        ensemble = equal_prediction_ensemble(all_frames[split], split, PE5_SEEDS)
        path = root / "ensemble_predictions_{}.csv".format(split)
        ensemble.to_csv(path, index=False)
        ensembles[split] = ensemble
        ensemble_metric_rows.extend(_ensemble_metric_rows(ensemble))
        _, j_value = metrics_from_predictions(ensemble)
        pe5_j[split] = float(j_value)
        expected = float(EXPECTED_J[split])
        difference = abs(float(j_value) - expected)
        print("PE5 {} J={:.9f} expected={:.9f} diff={:.3e}".format(split, j_value, expected, difference))
        if difference > HISTORICAL_PE5_J_TOLERANCE:
            raise RuntimeError(
                "Recovered five-checkpoint PE5 does not replay historical {} J: actual={} expected={} diff={}".format(
                    split, j_value, expected, difference
                )
            )

    pd.DataFrame(ensemble_metric_rows).to_csv(root / "ensemble_metrics.csv", index=False)
    pd.DataFrame(_comparison_rows(individual, ensembles)).to_csv(
        root / "pe5_full_metric_comparison.csv", index=False
    )

    prediction_manifest = {
        "Recovery": True,
        "RecoveryProtocol": "five frozen historical CFCompat validation-best checkpoints; inference only",
        "Members": member_manifest,
    }
    (root / "individual_predictions_manifest.json").write_text(
        json.dumps(prediction_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_manifest = {
        "Method": "CFCompatKD-PE5",
        "Seeds": list(PE5_SEEDS),
        "Weights": [0.2] * len(PE5_SEEDS),
        "Members": member_manifest,
        "OnlyValidationSelectedOnlineCheckpoints": True,
        "NoTestSelectedCheckpoint": True,
        "Recovery": True,
        "NoTraining": True,
    }
    (root / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    seed1113_test = all_frames["test"][list(PE5_SEEDS).index(1113)]
    recent_replay = _verify_seed1113_against_recent_probe(result_root, seed1113_test)

    replay = {
        "Recovery": True,
        "NoTraining": True,
        "NoCheckpointSelection": True,
        "NoTestDrivenSelection": True,
        "HistoricalCheckpointPathsFixed": {str(seed): str(path) for seed, path in HISTORICAL_CHECKPOINTS.items()},
        "ExpectedPE5J": {key: float(value) for key, value in EXPECTED_J.items()},
        "RecoveredPE5J": pe5_j,
        "HistoricalPE5JTolerance": HISTORICAL_PE5_J_TOLERANCE,
        "RecentSeed1113OriginalReplay": recent_replay,
        "Passed": True,
    }
    (root / "ensemble_replay_verification.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Recovered Stage9A-compatible frozen artifacts:", root)
    print("Historical PE5 replay: PASS")
    if recent_replay.get("available"):
        print("Recent seed1113 Original-CFCompat replay: PASS")


if __name__ == "__main__":
    main()
