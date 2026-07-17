"""Formal sequential-inference entry point for Stage 9A CFCompatKD-PE5."""
import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from train_cf_compat_kd import build_config, prediction_rows
from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    EXPECTED_J,
    PE5_MODES,
    PE5_SEEDS,
    PE5_SPLITS,
    equal_prediction_ensemble,
    load_locked_checkpoints,
    max_prediction_difference,
    metric_max_difference,
    metrics_from_predictions,
    require_locked_members,
    stage8_directory,
    stage9_directory,
    validate_prediction_frame,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage9A CFCompatKD five-seed equal prediction ensemble."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PE5_SEEDS))
    parser.add_argument("--split", nargs="+", choices=PE5_SPLITS, default=list(PE5_SPLITS))
    parser.add_argument("--modes", nargs="+", choices=PE5_MODES, default=list(PE5_MODES))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[2])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    try:
        require_locked_members(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if tuple(args.split) != PE5_SPLITS:
        parser.error("Stage9A requires --split valid test in order.")
    if tuple(args.modes) != PE5_MODES:
        parser.error("Stage9A requires --modes LAV LA LV L in order.")
    if args.num_workers != 0:
        parser.error("Stage9A fixes num_workers=0 for deterministic inference.")
    return args


def source_prediction_path(result_root, dataset, seed, split):
    return (
        stage8_directory(result_root, dataset)
        / "seed{}".format(int(seed))
        / "online_{}_predictions.csv".format(split)
    )


def online_prediction_path(output, seed, split):
    return output / "online_seed{}_{}_predictions.csv".format(int(seed), split)


def load_offline_sources(cli):
    frames = {}
    for split in PE5_SPLITS:
        local = []
        for seed in PE5_SEEDS:
            path = source_prediction_path(cli.result_root, cli.dataset, seed, split)
            frame = validate_prediction_frame(
                pd.read_csv(path), split, seed=seed, method="Online"
            )
            local.append(frame)
        frames[split] = local
    return frames


def build_student(config, checkpoint):
    state = torch.load(checkpoint, map_location=config.device)
    if any("teacher" in key.lower() for key in state):
        raise RuntimeError("Student checkpoint contains Teacher state.")
    student = MissingModalityWrapper(
        DLF(config).to(config.device),
        config.feature_dims[1],
        config.feature_dims[2],
    ).to(config.device)
    student.load_state_dict(state, strict=True)
    student.eval()
    return student, len(state)


def infer_one_member(cli, record):
    seed = int(record["Seed"])
    setup_seed(seed)
    config = build_config(cli, seed)
    student, state_key_count = build_student(config, record["Checkpoint"])
    frames = {}
    for split in PE5_SPLITS:
        loader = build_single_split_loader(config, split, cli.num_workers)
        frame = prediction_rows(student, loader, config.device)
        frame["Seed"] = seed
        frame["Method"] = "Online"
        frame["Split"] = split
        frame["SelectedBy"] = "validation_J"
        frames[split] = validate_prediction_frame(
            frame, split, seed=seed, method="Online"
        )
        del loader
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frames, state_key_count, str(config.device)


def run(cli):
    output = stage9_directory(cli.result_root, cli.dataset)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints, stage8_manifest = load_locked_checkpoints(
        cli.result_root, cli.dataset, cli.seeds
    )

    offline_sources = load_offline_sources(cli)
    offline_ensemble = {}
    replay = {
        "ExpectedJ": EXPECTED_J,
        "Tolerance": 1e-4,
        "Offline": {},
        "Online": {},
        "PerMemberOnlineOfflineMaxAbsDifference": {},
        "AllSampleBindingsMatch": True,
        "AllCheckpointsValidationSelectedOnline": True,
        "EqualWeights": [0.2] * 5,
        "NoTestBasedSelection": True,
    }
    for split in PE5_SPLITS:
        offline = equal_prediction_ensemble(
            offline_sources[split], split, PE5_SEEDS
        )
        _, j_value = metrics_from_predictions(offline)
        passed = abs(j_value - EXPECTED_J[split]) <= 1e-4
        replay["Offline"][split] = {
            "J": j_value,
            "ExpectedJ": EXPECTED_J[split],
            "AbsoluteDifference": abs(j_value - EXPECTED_J[split]),
            "Passed": passed,
        }
        if not passed:
            raise RuntimeError("STAGE8 OFFLINE ENSEMBLE REPLAY FAILURE")
        offline.to_csv(
            output / "offline_ensemble_predictions_{}.csv".format(split),
            index=False,
        )
        offline_ensemble[split] = offline

    online_sources = {split: [] for split in PE5_SPLITS}
    member_manifest = []
    for record in checkpoints:
        seed = int(record["Seed"])
        inferred, state_key_count, device = infer_one_member(cli, record)
        item = {
            **record,
            "StudentOnly": True,
            "SequentialInference": True,
            "Device": device,
            "StateKeyCount": state_key_count,
            "Predictions": {},
        }
        for split in PE5_SPLITS:
            source = offline_sources[split][PE5_SEEDS.index(seed)]
            maximum = max_prediction_difference(source, inferred[split], split)
            replay["PerMemberOnlineOfflineMaxAbsDifference"][
                "seed{}_{}".format(seed, split)
            ] = maximum
            if maximum > 1e-6:
                raise RuntimeError(
                    "OFFLINE/ONLINE MEMBER PREDICTION MISMATCH seed{} {}".format(
                        seed, split
                    )
                )
            path = online_prediction_path(output, seed, split)
            inferred[split].to_csv(path, index=False)
            item["Predictions"][split] = {
                "Path": str(path),
                "SHA256": checkpoint_sha256(path),
                "Stage8SourcePath": str(
                    source_prediction_path(
                        cli.result_root, cli.dataset, seed, split
                    )
                ),
                "MaxAbsDifference": maximum,
            }
            online_sources[split].append(inferred[split])
        member_manifest.append(item)

    maximum_prediction_difference = 0.0
    maximum_metric_difference = 0.0
    for split in PE5_SPLITS:
        online = equal_prediction_ensemble(online_sources[split], split, PE5_SEEDS)
        prediction_difference = max_prediction_difference(
            offline_ensemble[split], online, split
        )
        metric_difference = metric_max_difference(offline_ensemble[split], online)
        maximum_prediction_difference = max(
            maximum_prediction_difference, prediction_difference
        )
        maximum_metric_difference = max(maximum_metric_difference, metric_difference)
        if prediction_difference > 1e-6 or metric_difference > 1e-6:
            raise RuntimeError("OFFLINE/ONLINE ENSEMBLE MISMATCH")
        _, j_value = metrics_from_predictions(online)
        passed = abs(j_value - EXPECTED_J[split]) <= 1e-4
        replay["Online"][split] = {
            "J": j_value,
            "ExpectedJ": EXPECTED_J[split],
            "AbsoluteDifference": abs(j_value - EXPECTED_J[split]),
            "Passed": passed,
        }
        if not passed:
            raise RuntimeError("STAGE8 ONLINE ENSEMBLE REPLAY FAILURE")
        online.to_csv(
            output / "ensemble_predictions_{}.csv".format(split), index=False
        )

    replay.update(
        {
            "MaximumOfflineOnlinePredictionDifference": maximum_prediction_difference,
            "MaximumOfflineOnlineMetricDifference": maximum_metric_difference,
            "PredictionTolerance": 1e-6,
            "MetricTolerance": 1e-6,
            "Passed": True,
        }
    )
    checkpoint_manifest = {
        "Method": "CFCompatKD-PE5",
        "Members": member_manifest,
        "Seeds": list(PE5_SEEDS),
        "Weights": [0.2] * 5,
        "Stage8CheckpointManifest": str(stage8_manifest),
        "OnlyValidationSelectedOnlineCheckpoints": True,
        "NoParameterAveraging": True,
        "NoHiddenStateAveraging": True,
        "NoTeacherEvaluatorCacheRequiredForInference": True,
        "SequentialLoading": True,
        "MaximumResidentModels": 1,
    }
    (output / "individual_predictions_manifest.json").write_text(
        json.dumps(
            {
                "Members": [
                    {
                        "Seed": item["Seed"],
                        "Checkpoint": item["Checkpoint"],
                        "CheckpointSHA256": item["CheckpointSHA256"],
                        "BestValidEpoch": item["BestValidEpoch"],
                        "SelectedBy": item["SelectedBy"],
                        "Predictions": item["Predictions"],
                    }
                    for item in member_manifest
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "ensemble_replay_verification.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return replay


def main():
    cli = parse_args()
    result = run(cli)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
