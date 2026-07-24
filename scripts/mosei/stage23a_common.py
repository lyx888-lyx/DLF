"""Shared, test-locked utilities for Stage 23A.

This module deliberately exposes only the MOSEI train split.  Official Valid
is handled by a separate, one-shot command after all train-only gates pass.
There is no Test loader or Test evaluation entry point in Stage 23A.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "result" / "arbiter_audit_v1" / "mosei"
MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
EXPERTS = (
    "uniform_kd_seed1111",
    "moddrop_seed1111",
    "moddrop_seed1114",
    "cfcompat_seed1111",
    "cfcompat_seed1114",
)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(str(tmp), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def source_video(sample_id):
    value = str(sample_id)
    if "$_$" not in value:
        raise ValueError("MOSEI sample ID has no source delimiter: {}".format(value))
    return value.rsplit("$_$", 1)[0]


def stable_bucket(value, salt, modulo):
    digest = hashlib.sha256(
        "{}|{}".format(salt, value).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % int(modulo)


def outer_fold(video_id):
    return stable_bucket(video_id, "stage23a_outer_v1", 2)


def judge_fold(video_id):
    return stable_bucket(video_id, "stage23a_judge_v1", 2)


def inner_valid(video_id, outer):
    # About 12.5% of the outer-training sources.  The split is source-disjoint.
    return stable_bucket(video_id, "stage23a_inner_outer{}".format(outer), 8) == 0


def ordered_id_sha(ids):
    return sha256_json([str(value) for value in ids])


def build_subset_loader(
    dataset,
    indices,
    batch_size,
    shuffle,
    seed,
    num_workers=1,
):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        Subset(dataset, [int(value) for value in indices]),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        drop_last=False,
        num_workers=int(num_workers),
    )


def load_preregistered():
    path = RESULT_ROOT / "protocol" / "preregistered_protocol.json"
    if not path.is_file():
        raise FileNotFoundError("Run Stage 23A preregistration first: {}".format(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("locked_test_access_count") != 0:
        raise RuntimeError("Stage 23A test lock is not zero.")
    if tuple(payload["expert_ids"]) != EXPERTS:
        raise RuntimeError("Candidate expert pool changed after preregistration.")
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
        "inner_valid_outer0",
        "inner_valid_outer1",
    }
    if set(frame.columns) != required:
        raise RuntimeError("Unexpected source split schema.")
    if frame.sample_id.duplicated().any():
        raise RuntimeError("Duplicate train sample IDs.")
    for column in ("outer_fold", "judge_fold"):
        by_source = frame.groupby("video_id")[column].nunique()
        if int(by_source.max()) != 1:
            raise RuntimeError("{} is not source-disjoint.".format(column))
    return frame, sha256_file(path)


def project_simplex_rows(values):
    """Euclidean projection of each row onto the probability simplex."""
    values = np.asarray(values, dtype=np.float64)
    ordered = -np.sort(-values, axis=1)
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    steps = np.arange(1, values.shape[1] + 1, dtype=np.float64)
    active = ordered - cumulative / steps > 0
    rho = active.sum(axis=1) - 1
    theta = cumulative[np.arange(len(values)), rho] / (rho + 1.0)
    return np.maximum(values - theta[:, None], 0.0)


def regression_metrics(prediction, label):
    from sklearn.metrics import f1_score

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    if len(prediction) != len(label) or not len(label):
        raise ValueError("Metric arrays are empty or differ.")
    clipped7_pred = np.clip(prediction, -3, 3)
    clipped7_label = np.clip(label, -3, 3)
    clipped5_pred = np.clip(prediction, -2, 2)
    clipped5_label = np.clip(label, -2, 2)
    nonzero = label != 0
    binary_pred = prediction[nonzero] > 0
    binary_label = label[nonzero] > 0
    corr = (
        float(np.corrcoef(prediction, label)[0, 1])
        if len(label) > 1 and prediction.std() > 0 and label.std() > 0
        else 0.0
    )
    return {
        "J": float(np.mean(np.abs(prediction - label))),
        "MAE": float(np.mean(np.abs(prediction - label))),
        "Corr": corr,
        "Acc7": float(np.mean(np.round(clipped7_pred) == np.round(clipped7_label))),
        "Acc5": float(np.mean(np.round(clipped5_pred) == np.round(clipped5_label))),
        "Acc2": float(np.mean(binary_pred == binary_label)) if nonzero.any() else 0.0,
        "F1": (
            float(f1_score(binary_label, binary_pred, average="weighted", zero_division=0))
            if nonzero.any()
            else 0.0
        ),
    }


def overall_j(frame, prediction_column="prediction"):
    maes = {}
    for mode in MODES:
        local = frame.loc[frame["mode"] == mode]
        maes[mode] = float(np.mean(np.abs(local[prediction_column] - local.label)))
    return 0.5 * maes["LAV"] + 0.5 * np.mean([maes[mode] for mode in MISSING_MODES])
