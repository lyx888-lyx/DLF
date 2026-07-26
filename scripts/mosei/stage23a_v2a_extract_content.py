#!/usr/bin/env python
"""Build padding-aware Train-only raw content summaries.

Only the ``train`` partition of the frozen MOSEI aligned pickle is accessed.
The large summary matrices are stored as deterministic NPY files and bound to a
small metadata ledger by ordered sample IDs and SHA256 values.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage23a_v2_common import (
    V1_ROOT,
    V2_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    sha256_json,
)


DATASET_PATH = Path("/data4t/lyx/datasets/MOSEI/Processed/aligned_50.pkl")
DATASET_SHA = "45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791"
MODALITIES = ("text", "audio", "vision")


def decode_id(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.size == 1:
        return decode_id(value.reshape(-1)[0])
    return str(value)


def temporal_summary(values, chunk_size=256):
    values = np.asarray(values)
    count, timesteps, feature_dim = values.shape
    means = np.zeros((count, feature_dim), dtype=np.float32)
    stds = np.zeros((count, feature_dim), dtype=np.float32)
    active_counts = np.zeros(count, dtype=np.int16)
    finite = True
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk = values[start:stop]
        finite = finite and bool(np.isfinite(chunk).all())
        active = np.any(chunk != 0, axis=2)
        local_count = active.sum(axis=1)
        active_counts[start:stop] = local_count
        for position in range(stop - start):
            if local_count[position] == 0:
                continue
            selected = np.asarray(
                chunk[position, active[position]], dtype=np.float64
            )
            means[start + position] = selected.mean(axis=0).astype(np.float32)
            stds[start + position] = selected.std(axis=0, ddof=0).astype(np.float32)
    return means, stds, active_counts, finite, timesteps


def main():
    authorization = json.loads(
        (V2_ROOT / "protocol" / "v2a_authorization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not authorization["feature_extraction_authorized"]:
        raise RuntimeError("Feature extraction is not authorized.")
    if any(
        authorization[key]
        for key in (
            "judge_training_authorized",
            "official_valid_authorized",
            "test_authorized",
            "student_training_authorized",
        )
    ):
        raise RuntimeError("A forbidden authorization flag is open.")
    if sha256_file(DATASET_PATH) != DATASET_SHA:
        raise RuntimeError("MOSEI dataset SHA mismatch.")

    with DATASET_PATH.open("rb") as handle:
        dataset = pickle.load(handle)
    if "train" not in dataset:
        raise RuntimeError("Train partition missing.")
    train = dataset["train"]
    id_key = "id" if "id" in train else "ids"
    sample_ids = [decode_id(value) for value in train[id_key]]
    if len(sample_ids) != 16326 or len(set(sample_ids)) != 16326:
        raise RuntimeError("Train sample ID count/uniqueness mismatch.")

    source_split = pd.read_csv(
        V1_ROOT / "protocol" / "source_splits.csv",
        dtype={"sample_id": str, "video_id": str},
    ).sort_values("train_index")
    expected_ids = source_split["sample_id"].tolist()
    if sample_ids != expected_ids:
        # Some loaders expose IDs with a singleton suffix/container; report the
        # first mismatch rather than silently reordering content.
        mismatch = next(
            index
            for index, (left, right) in enumerate(zip(sample_ids, expected_ids))
            if left != right
        )
        raise RuntimeError(
            "Dataset/source-ledger ordering mismatch at {}: {} != {}".format(
                mismatch, sample_ids[mismatch], expected_ids[mismatch]
            )
        )

    output_dir = V2_ROOT / "features" / "content_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = source_split[
        ["sample_id", "video_id", "train_index", "outer_fold"]
    ].copy()
    artifacts = {}
    modality_reports = {}
    effective_columns = {}
    for modality in MODALITIES:
        if modality not in train:
            raise RuntimeError("Missing Train modality {}".format(modality))
        raw = np.asarray(train[modality])
        means, stds, active_counts, finite, timesteps = temporal_summary(raw)
        if not finite:
            raise RuntimeError("Non-finite raw {} values.".format(modality))
        mean_path = output_dir / "{}_temporal_mean_float32.npy".format(modality)
        std_path = output_dir / "{}_temporal_std_float32.npy".format(modality)
        np.save(str(mean_path), means, allow_pickle=False)
        np.save(str(std_path), stds, allow_pickle=False)
        effective = active_counts > 0
        metadata["{}_valid_timesteps".format(modality)] = active_counts
        metadata["{}_padding_timesteps".format(modality)] = timesteps - active_counts
        metadata["{}_all_zero".format(modality)] = ~effective
        metadata["{}_effective_available".format(modality)] = effective.astype(np.int8)
        metadata["{}_finite".format(modality)] = True
        metadata["{}_summary_l2".format(modality)] = np.sqrt(
            np.square(means, dtype=np.float64).sum(axis=1)
            + np.square(stds, dtype=np.float64).sum(axis=1)
        )
        effective_columns[modality] = effective
        artifacts[modality] = {
            "mean_path": str(mean_path.resolve()),
            "mean_sha256": sha256_file(mean_path),
            "std_path": str(std_path.resolve()),
            "std_sha256": sha256_file(std_path),
        }
        modality_reports[modality] = {
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "summary_shape": [len(raw), 2 * raw.shape[2]],
            "stored_summary_dtype": "float32",
            "finite": finite,
            "valid_timestep_min": int(active_counts.min()),
            "valid_timestep_max": int(active_counts.max()),
            "valid_timestep_mean": float(active_counts.mean()),
            "all_zero_clips": int((~effective).sum()),
            "all_zero_summary_is_zero": bool(
                np.all(means[~effective] == 0) and np.all(stds[~effective] == 0)
            ),
            **artifacts[modality],
        }
        print(
            "{} shape={} zero={} active_mean={:.6f}".format(
                modality, raw.shape, int((~effective).sum()), active_counts.mean()
            ),
            flush=True,
        )
        del raw, means, stds

    if int((~effective_columns["vision"]).sum()) != 482:
        raise RuntimeError("Expected exactly 482 all-zero Vision clips.")
    if int((~effective_columns["text"]).sum()) != 0:
        raise RuntimeError("Unexpected ineffective Text clips.")
    if int((~effective_columns["audio"]).sum()) != 0:
        raise RuntimeError("Unexpected ineffective Audio clips.")
    metadata["sample_binding_sha256"] = [
        sha256_json(
            {
                "dataset_sha256": DATASET_SHA,
                "sample_id": row.sample_id,
                "train_index": int(row.train_index),
            }
        )
        for row in metadata.itertuples(index=False)
    ]
    metadata_path = output_dir / "content_summary_ledger.csv"
    atomic_csv(metadata, metadata_path)
    ordered_id_sha = sha256_json(sample_ids)
    report = {
        "stage": "Stage23A-v2a Train-only fixed input content summaries",
        "status": "PASS",
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": DATASET_SHA,
        "accessed_partitions": ["train"],
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "samples": len(sample_ids),
        "ordered_sample_id_sha256": ordered_id_sha,
        "ordered_sample_ids_match_source_split": True,
        "summary_definition": "padding-aware temporal mean concatenated with population std (ddof=0)",
        "padding_rule": "timestep active iff any feature value is nonzero",
        "stored_summary_dtype": "float32",
        "modalities": modality_reports,
        "effective_availability": {
            "text_ineffective": int((~effective_columns["text"]).sum()),
            "audio_ineffective": int((~effective_columns["audio"]).sum()),
            "vision_ineffective": int((~effective_columns["vision"]).sum()),
            "vision_zero_policy": "raw summary zero; exclude from fit; transformed block forced zero; effective mask vision=0",
        },
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "sample_binding_sha_verified": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge_training_started": False,
        "student_trained": False,
    }
    report_path = output_dir / "content_summary_manifest.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
