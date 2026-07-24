"""MOSI-only protocol adapter for the frozen MOSEI Stage 23A implementation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from stage23a_common import (
    EXPERTS,
    MODES,
    MISSING_MODES,
    atomic_csv,
    atomic_json,
    git_head,
    ordered_id_sha,
    project_simplex_rows,
    regression_metrics,
    sha256_file,
    sha256_json,
    source_video,
    stable_bucket,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "result" / "arbiter_audit_v1" / "mosi"
DATASET_PATH = Path("/code/DLF/dataset/MOSI/Processed/aligned_50.pkl")
N_FOLDS = 5
SPLIT_SEED = 2301
FROZEN_MOSEI_ROOT = Path("/code/DLF-mosei-arbiter-audit-v1")


def judge_fold(video_id):
    return stable_bucket(video_id, "stage23a_judge_v1", 2)


def inner_valid(video_id, outer):
    return stable_bucket(video_id, "stage23a_inner_outer{}".format(outer), 8) == 0


def load_preregistered():
    path = RESULT_ROOT / "protocol" / "preregistered_protocol.json"
    if not path.is_file():
        raise FileNotFoundError("Run MOSI Stage 23A preregistration first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("locked_test_access_count") != 0:
        raise RuntimeError("MOSI Stage 23A test lock is not zero.")
    if tuple(payload["expert_ids"]) != tuple(EXPERTS):
        raise RuntimeError("MOSI candidate pool differs from frozen MOSEI pool.")
    return payload, sha256_file(path)


def load_split_manifest():
    path = RESULT_ROOT / "protocol" / "source_splits.csv"
    frame = pd.read_csv(path, dtype={"sample_id": str, "video_id": str})
    required = {
        "train_index",
        "sample_id",
        "video_id",
        "outer_fold",
        "judge_fold",
        *{"inner_valid_outer{}".format(fold) for fold in range(N_FOLDS)},
    }
    if set(frame.columns) != required:
        raise RuntimeError("Unexpected MOSI source split schema.")
    if frame.sample_id.duplicated().any():
        raise RuntimeError("Duplicate MOSI train sample IDs.")
    for column in ("outer_fold", "judge_fold"):
        if int(frame.groupby("video_id")[column].nunique().max()) != 1:
            raise RuntimeError("{} is not source-disjoint.".format(column))
    return frame, sha256_file(path)


def overall_j(frame, prediction_column="prediction"):
    maes = {}
    for mode in MODES:
        local = frame.loc[frame["mode"] == mode]
        maes[mode] = float(
            (local[prediction_column] - local["label"]).abs().mean()
        )
    return 0.5 * maes["LAV"] + 0.5 * sum(
        maes[mode] for mode in MISSING_MODES
    ) / len(MISSING_MODES)

