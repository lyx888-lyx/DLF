"""Integrity verification for Stage 8 per-seed outputs."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_stability_utils import STAGE8_SEEDS
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


METHODS = ("Online", "EMA", "Soup-3", "Soup-5")


def seed_directory(result_root, dataset, seed, smoke=False):
    root = (
        Path(result_root) / "missing_baseline" / "cfcompat_stability_v1"
        / dataset
    )
    if smoke:
        root = root / "smoke"
    return root / "seed{}".format(int(seed))


def verify_seed(directory, require_replay=True):
    directory = Path(directory)
    required = (
        "per_seed_all_methods.csv",
        "ema_epoch_metrics.csv",
        "soup_source_checkpoints.csv",
        "seed_manifest.json",
        "online_replay_audit.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage8 seed artifacts: {}".format(missing))
    rows = pd.read_csv(directory / "per_seed_all_methods.csv")
    if rows.Method.tolist() != list(METHODS):
        raise RuntimeError("Per-seed methods/order is not Online, EMA, Soup-3, Soup-5.")
    if rows.Seed.nunique() != 1 or rows.duplicated(["Seed", "Method"]).any():
        raise RuntimeError("Per-seed method rows are not unique.")
    numeric = rows.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite per-seed result.")
    for _, row in rows.iterrows():
        checkpoint = Path(row.Checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError("Checkpoint is absent: {}".format(checkpoint))
        if checkpoint_sha256(checkpoint) != row.CheckpointSHA256:
            raise RuntimeError("Checkpoint SHA mismatch: {}".format(checkpoint))
        if row.SelectedBy != "validation_J" or not bool(row.StudentOnlyEval):
            raise RuntimeError("Checkpoint selection/evaluation declaration is invalid.")
        slug = row.Method.lower().replace("-", "")
        for split in ("valid", "test"):
            path = directory / "{}_{}_predictions.csv".format(slug, split)
            prediction = pd.read_csv(path)
            if (
                set(prediction.Split) != {split}
                or set(prediction.Method) != {row.Method}
                or prediction.sample_index.duplicated().any()
            ):
                raise RuntimeError("Prediction binding is invalid: {}".format(path))
            values = prediction.select_dtypes(include=[np.number]).to_numpy()
            if not np.isfinite(values).all():
                raise RuntimeError("Non-finite prediction artifact: {}".format(path))
    replay = json.loads((directory / "online_replay_audit.json").read_text())
    if require_replay and (
        not replay.get("Passed") or not replay.get("MissingSequenceMatch")
    ):
        raise RuntimeError("Formal Online replay gate failed.")
    manifest = json.loads((directory / "seed_manifest.json").read_text())
    if (
        float(manifest["EMA"]["Decay"]) != 0.999
        or not manifest["EMA"]["NeverInOptimizer"]
        or not manifest["EMA"]["NeverInBackward"]
        or not manifest["EMA"]["RNGPreserved"]
        or not manifest["Soup"]["OnlyOnlineSameSeedSources"]
        or manifest["Soup"]["TestUsedForSourceSelection"]
        or not manifest["NoCrossSeedWeightAveraging"]
    ):
        raise RuntimeError("Stage8 method integrity declaration failed.")
    sources = pd.read_csv(directory / "soup_source_checkpoints.csv")
    if set(sources.SoupMethod) != {"Soup-3", "Soup-5"}:
        raise RuntimeError("Soup source manifest is incomplete.")
    for method, count in (("Soup-3", 3), ("Soup-5", 5)):
        selected = sources.loc[
            sources.SoupMethod.eq(method) & sources.Included.astype(bool)
        ]
        if len(selected) != count:
            raise RuntimeError("{} does not use exactly {} sources.".format(method, count))
    return {
        "Seed": int(rows.Seed.iloc[0]),
        "Directory": str(directory),
        "ReplayPassed": bool(replay.get("Passed")),
        "MissingSequenceMatch": bool(replay.get("MissingSequenceMatch")),
        "Methods": list(rows.Method),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, choices=STAGE8_SEEDS, required=True)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    result = verify_seed(
        seed_directory(
            args.result_root, args.dataset, args.seed, smoke=args.smoke_test
        ),
        require_replay=not args.smoke_test,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
