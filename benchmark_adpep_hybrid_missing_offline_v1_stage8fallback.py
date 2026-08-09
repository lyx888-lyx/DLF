"""Stage8 fallback entry point for the offline ADPEP/Hybrid missing benchmark.

The original benchmark prefers the Stage9A per-member prediction manifest.  Some
local workspaces no longer retain that intermediate directory.  Stage9A itself
was defined as an exact replay of the frozen Stage8 Online prediction files, so
this wrapper falls back to those already-frozen Stage8 artifacts without any
new model inference or Test DataLoader construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import benchmark_adpep_hybrid_missing_offline_v1 as base


_ORIGINAL_STAGE9_LOADER = base.load_stage9_members


def _load_stage8_members(result_root: Path, dataset: str):
    root = Path(result_root) / "missing_baseline" / "cfcompat_stability_v1" / dataset
    frames = {}
    records = {}
    reference = None

    for seed in base.SEEDS:
        seed_dir = root / "seed{}".format(int(seed))
        prediction_path = seed_dir / "online_test_predictions.csv"
        replay_path = seed_dir / "online_replay_audit.json"
        rows_path = seed_dir / "per_seed_all_methods.csv"

        missing = [
            str(path)
            for path in (prediction_path, replay_path, rows_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Stage9A manifest is absent and Stage8 fallback is incomplete for "
                "seed{}: {}".format(seed, missing)
            )

        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if not replay.get("Passed") or not replay.get("MissingSequenceMatch"):
            raise RuntimeError(
                "Stage8 formal Online replay gate failed for seed{}".format(seed)
            )

        rows = pd.read_csv(rows_path)
        online = rows.loc[rows.Method.astype(str).eq("Online")]
        if len(online) != 1:
            raise RuntimeError("Stage8 Online row is not unique for seed{}".format(seed))
        online = online.iloc[0]
        if str(online.SelectedBy) != "validation_J" or not bool(online.StudentOnlyEval):
            raise RuntimeError(
                "Stage8 Online prediction is not a validation-selected Student-only "
                "artifact for seed{}".format(seed)
            )

        frame = base._read_prediction_frame(prediction_path)
        if "Seed" in frame.columns and set(frame.Seed.astype(int)) != {int(seed)}:
            raise RuntimeError("Stage8 prediction seed binding differs for seed{}".format(seed))
        if "Method" in frame.columns and set(frame.Method.astype(str)) != {"Online"}:
            raise RuntimeError("Stage8 prediction method binding differs for seed{}".format(seed))
        if "SelectedBy" in frame.columns and set(frame.SelectedBy.astype(str)) != {"validation_J"}:
            raise RuntimeError("Stage8 prediction selection binding differs for seed{}".format(seed))

        if reference is None:
            reference = frame
        else:
            base._bind_frames(reference, frame, check_label=True)

        frames[int(seed)] = frame
        records[int(seed)] = {
            "source": "stage8_frozen_online_prediction_fallback",
            "checkpoint": str(online.get("Checkpoint", "")),
            "checkpoint_sha256": str(online.get("CheckpointSHA256", "")),
            "best_valid_epoch": int(online.get("BestValidEpoch", -1)),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": base.sha256(prediction_path),
            "online_replay_audit": str(replay_path.resolve()),
            "online_replay_passed": True,
            "missing_sequence_match": True,
        }

    if reference is None:
        raise RuntimeError("Stage8 fallback found no frozen Online prediction frames")
    return root, frames, records, reference


def load_stage9_or_stage8_members(result_root: Path, dataset: str):
    stage9_manifest = (
        Path(result_root)
        / "missing_baseline"
        / "cfcompat_prediction_ensemble_v1"
        / dataset
        / "individual_predictions_manifest.json"
    )
    if stage9_manifest.is_file():
        print("Frozen member source: Stage9A manifest")
        return _ORIGINAL_STAGE9_LOADER(result_root, dataset)

    print("Frozen Stage9A manifest absent; using exact Stage8 Online prediction fallback")
    return _load_stage8_members(result_root, dataset)


def main():
    base.load_stage9_members = load_stage9_or_stage8_members
    base.main()


if __name__ == "__main__":
    main()
