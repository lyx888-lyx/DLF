"""Pre-register the Stage 23A candidate pool and source-disjoint splits."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stage23a_common import (
    EXPERTS,
    RESULT_ROOT,
    atomic_csv,
    atomic_json,
    git_head,
    inner_valid,
    judge_fold,
    ordered_id_sha,
    outer_fold,
    sha256_file,
    source_video,
)


ASSETS = {
    "uniform_kd_seed1111": (
        "/code/DLF-mosei-mgrd-v1/result/missing_baseline/mgrd_v1/mosei/"
        "baseline/uniform_seed1111_fp32/checkpoints/best_epoch002.pth"
    ),
    "moddrop_seed1111": (
        "/code/DLF-mosei-generalization-v1/result/missing_baseline/"
        "mosei_generalization_v1/moddrop/seed1111/DLF_mosei_seed1111_best_valid.pth"
    ),
    "moddrop_seed1114": (
        "/code/DLF-mosei-generalization-v1/result/missing_baseline/"
        "mosei_generalization_v1/moddrop/seed1114/DLF_mosei_seed1114_best_valid.pth"
    ),
    "cfcompat_seed1111": (
        "/code/DLF-mosei-generalization-v1/result/missing_baseline/"
        "mosei_generalization_v1/cfcompat/seed1111/DLF_mosei_seed1111_best_valid.pth"
    ),
    "cfcompat_seed1114": (
        "/code/DLF-mosei-generalization-v1/result/missing_baseline/"
        "mosei_generalization_v1/cfcompat/seed1114/DLF_mosei_seed1114_best_valid.pth"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/data4t/lyx/datasets/MOSEI/Processed/aligned_50.pkl",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    protocol_dir = RESULT_ROOT / "protocol"
    frozen_path = protocol_dir / "preregistered_protocol.json"
    if frozen_path.exists():
        raise RuntimeError("Candidate pool is already frozen: {}".format(frozen_path))

    for expert, value in ASSETS.items():
        if not Path(value).is_file():
            raise FileNotFoundError("Candidate checkpoint missing: {} {}".format(expert, value))

    # The only dataset key read here is train.  Valid and Test are not touched.
    import pickle

    with Path(cli.dataset).open("rb") as handle:
        train = pickle.load(handle)["train"]
    ids = [str(value) for value in train["id"]]
    videos = [source_video(value) for value in ids]
    rows = []
    for index, (sample_id, video_id) in enumerate(zip(ids, videos)):
        rows.append(
            {
                "train_index": index,
                "sample_id": sample_id,
                "video_id": video_id,
                "outer_fold": outer_fold(video_id),
                "judge_fold": judge_fold(video_id),
                "inner_valid_outer0": int(inner_valid(video_id, 0)),
                "inner_valid_outer1": int(inner_valid(video_id, 1)),
            }
        )
    split_frame = pd.DataFrame(rows)
    for column in ("outer_fold", "judge_fold", "inner_valid_outer0", "inner_valid_outer1"):
        if int(split_frame.groupby("video_id")[column].nunique().max()) != 1:
            raise RuntimeError("{} leaks a source video.".format(column))
    split_path = protocol_dir / "source_splits.csv"
    atomic_csv(split_frame, split_path)

    assets = {
        expert: {
            "path": path,
            "sha256": sha256_file(path),
            "bytes": Path(path).stat().st_size,
        }
        for expert, path in ASSETS.items()
    }
    protocol = {
        "stage": "Stage 23A Expert Complementarity and Personalized Teacher Feasibility Audit",
        "status": "PREREGISTERED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_head(),
        "dataset_path": str(cli.dataset),
        "train_sample_count": len(ids),
        "train_source_count": len(set(videos)),
        "train_ordered_sample_id_sha256": ordered_id_sha(ids),
        "source_split_path": str(split_path),
        "source_split_sha256": sha256_file(split_path),
        "expert_ids": list(EXPERTS),
        "candidate_assets": assets,
        "excluded_candidates": {
            "uniform_kd_seed1114": (
                "No complete pre-existing full-train Uniform KD seed1114 checkpoint "
                "and run manifest were present at preregistration."
            )
        },
        "outer_crossfit": "fixed 2-fold source-video-disjoint",
        "inner_selection": "source-disjoint 1/8 of each outer-training source set",
        "training_budget": {
            "max_epochs": 30,
            "early_stop_patience": 6,
            "batch_size": 16,
            "update_epochs": 10,
            "learning_rate": 0.0001,
            "selection": "inner-train-only validation; no Official Valid",
        },
        "judge_models": ["J0_mode_only", "J1_predictions", "J2_predictions_disagreement", "J4_shuffled"],
        "self_confidence_policy": (
            "J3 is unavailable because the frozen experts expose no pre-existing "
            "calibrated self-confidence; no new uncertainty head may be trained."
        ),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
    }
    atomic_json(frozen_path, protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
