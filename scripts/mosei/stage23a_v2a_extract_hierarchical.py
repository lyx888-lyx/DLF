#!/usr/bin/env python
"""Extract frozen Expert hierarchical logits after the final-logit replay gate.

Inference only: no optimizer, no checkpoint writes, no Official Valid/Test, no
Student and no Judge are constructed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage23a_v2_common import (
    EXPERTS,
    MODE_ACTIVE_HEADS,
    MODES,
    ROOT,
    V1_ROOT,
    V2_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    sha256_json,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.missing_utils import MissingModalityWrapper, mode_to_mask
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


HEAD_COLUMNS = (
    "output_logit",
    "logits_c",
    "logits_l_hetero",
    "logits_a_hetero",
    "logits_v_hetero",
)
DATASET_SHA = "45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", required=True, type=int, choices=(0, 1))
    parser.add_argument("--gpu-id", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    return parser.parse_args()


def atomic_gzip_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as compressed:
            frame.to_csv(compressed, index=False, float_format="%.10g")
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(str(temporary), str(path))


def build_args(cli):
    args = get_config_regression("DLF", "mosei", str(ROOT / "config/config.json"))
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False
    args.train_mode = "regression"
    args.batch_size = int(cli.batch_size)
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def main():
    cli = parse_args()
    authorization_path = V2_ROOT / "protocol" / "v2a_authorization_manifest.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    required = {
        "feature_extraction_authorized": True,
        "judge_training_authorized": False,
        "official_valid_authorized": False,
        "test_authorized": False,
        "student_training_authorized": False,
    }
    for key, expected in required.items():
        if authorization.get(key) is not expected:
            raise RuntimeError("Authorization mismatch {}.".format(key))

    replay = json.loads(
        (V2_ROOT / "validation" / "replay_fold{}_report.json".format(cli.outer_fold))
        .read_text(encoding="utf-8")
    )
    if replay["status"] != "PASS" or replay["max_abs_diff"] > 1e-6:
        raise RuntimeError("Preferred final-logit replay gate did not pass.")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    setup_seed(23300 + int(cli.outer_fold))
    args = build_args(cli)
    dataset = MMDataset(args, mode="train")
    args.seq_lens = dataset.get_seq_len()

    split = pd.read_csv(
        V1_ROOT / "protocol" / "source_splits.csv",
        dtype={"sample_id": str, "video_id": str},
    )
    holdout = split.loc[
        split["outer_fold"].astype(int) == int(cli.outer_fold)
    ].copy()
    holdout_indices = holdout["train_index"].astype(int).tolist()
    index_by_id = holdout.set_index("sample_id")["train_index"].astype(int).to_dict()
    video_by_id = holdout.set_index("sample_id")["video_id"].astype(str).to_dict()
    sample_binding = {
        sample_id: sha256_json(
            {
                "dataset_sha256": DATASET_SHA,
                "sample_id": sample_id,
                "train_index": int(train_index),
            }
        )
        for sample_id, train_index in index_by_id.items()
    }
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, holdout_indices),
        batch_size=int(cli.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(cli.num_workers),
    )
    old_path = (
        V1_ROOT
        / "expert_oof"
        / "outer_fold{}".format(cli.outer_fold)
        / "oof_predictions.csv"
    )
    old = pd.read_csv(old_path, dtype={"sample_id": str, "video_id": str})
    label_by_id = (
        old[["sample_id", "label"]]
        .drop_duplicates("sample_id")
        .set_index("sample_id")["label"]
        .astype(float)
        .to_dict()
    )
    expected = old.set_index(["sample_id", "mode", "expert_id"])["prediction"]

    all_records = []
    statistic_rows = []
    maximum_rows = []
    for expert in EXPERTS:
        expert_dir = (
            V1_ROOT
            / "expert_oof"
            / "outer_fold{}".format(cli.outer_fold)
            / "experts"
            / expert
        )
        manifest_path = expert_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(manifest["checkpoint"])
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != manifest["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint SHA mismatch {}".format(expert))
        model = MissingModalityWrapper(
            DLF(args), args.feature_dims[1], args.feature_dims[2]
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu"), strict=True
        )
        model.to(args.device).eval()
        expert_records = []
        with torch.no_grad():
            for batch in loader:
                text = batch["text"].to(args.device)
                audio = batch["audio"].to(args.device)
                vision = batch["vision"].to(args.device)
                identifiers = [str(value) for value in batch["id"]]
                for mode in MODES:
                    mask = mode_to_mask(
                        mode, text.size(0), args.device, audio.dtype
                    )
                    outputs = model(text, audio, vision, mask)
                    head_values = {
                        name: outputs[name].view(-1).detach().cpu().numpy()
                        for name in HEAD_COLUMNS
                    }
                    active = MODE_ACTIVE_HEADS[mode]
                    for position, sample_id in enumerate(identifiers):
                        values = {
                            name: float(head_values[name][position])
                            for name in HEAD_COLUMNS
                        }
                        active_values = np.asarray(
                            [values[name] for name in active], dtype=np.float64
                        )
                        active_median = float(np.median(active_values))
                        consistency = float(
                            np.median(np.abs(active_values - active_median))
                        )
                        row_binding = sha256_json(
                            {
                                "sample_binding_sha256": sample_binding[sample_id],
                                "mode": mode,
                                "expert_id": expert,
                                "checkpoint_sha256": checkpoint_sha,
                            }
                        )
                        expert_records.append(
                            {
                                "sample_id": sample_id,
                                "video_id": video_by_id[sample_id],
                                "train_index": index_by_id[sample_id],
                                "outer_fold": int(cli.outer_fold),
                                "expert_fold": int(cli.outer_fold),
                                "mode": mode,
                                "expert_id": expert,
                                "label": label_by_id[sample_id],
                                **values,
                                "active_head_names": "|".join(active),
                                "active_head_count": len(active),
                                "active_head_median": active_median,
                                "hierarchical_consistency_mad": consistency,
                                "checkpoint_path": str(checkpoint),
                                "checkpoint_sha256": checkpoint_sha,
                                "run_manifest_sha256": sha256_file(manifest_path),
                                "sample_binding_sha256": sample_binding[sample_id],
                                "row_binding_sha256": row_binding,
                            }
                        )
        local = pd.DataFrame(expert_records)
        key = ["sample_id", "mode", "expert_id"]
        replayed = local.set_index(key)["output_logit"].sort_index()
        old_expert = expected.loc[
            expected.index.get_level_values("expert_id") == expert
        ].sort_index()
        if not replayed.index.equals(old_expert.index):
            raise RuntimeError("OOF binding differs for {}".format(expert))
        diff = np.abs(
            replayed.to_numpy(dtype=np.float64)
            - old_expert.to_numpy(dtype=np.float64)
        )
        maximum_position = int(np.argmax(diff))
        maximum_key = replayed.index[maximum_position]
        maximum_rows.append(
            {
                "outer_fold": int(cli.outer_fold),
                "expert_id": expert,
                "sample_id": maximum_key[0],
                "mode": maximum_key[1],
                "old_prediction": float(old_expert.iloc[maximum_position]),
                "replayed_prediction": float(replayed.iloc[maximum_position]),
                "absolute_difference": float(diff[maximum_position]),
            }
        )
        statistic_rows.append(
            {
                "outer_fold": int(cli.outer_fold),
                "expert_id": expert,
                "rows": int(len(local)),
                "checkpoint_sha256": checkpoint_sha,
                "run_manifest_sha256": sha256_file(manifest_path),
                "max_abs_final_replay_diff": float(diff.max()),
                "mean_abs_final_replay_diff": float(diff.mean()),
                "mismatch_gt_1e_6": int((diff > 1e-6).sum()),
                "mismatch_gt_1e_5": int((diff > 1e-5).sum()),
            }
        )
        all_records.extend(expert_records)
        print(
            "fold={} expert={} rows={} max_diff={:.12g}".format(
                cli.outer_fold, expert, len(local), diff.max()
            ),
            flush=True,
        )
        model.cpu()
        del model
        torch.cuda.empty_cache()

    frame = pd.DataFrame(all_records).sort_values(
        ["train_index", "mode", "expert_id"]
    )
    expected_rows = int(len(holdout) * len(MODES) * len(EXPERTS))
    if len(frame) != expected_rows:
        raise RuntimeError("Hierarchical row count mismatch.")
    if frame.duplicated(["sample_id", "mode", "expert_id"]).any():
        raise RuntimeError("Duplicate hierarchical row.")
    if frame.isna().any().any():
        raise RuntimeError("Missing hierarchical value.")
    binding_recomputed = frame.apply(
        lambda row: sha256_json(
            {
                "sample_binding_sha256": row["sample_binding_sha256"],
                "mode": row["mode"],
                "expert_id": row["expert_id"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
        ),
        axis=1,
    )
    if not np.all(binding_recomputed.to_numpy() == frame["row_binding_sha256"].to_numpy()):
        raise RuntimeError("Row binding SHA verification failed.")

    output_dir = V2_ROOT / "features" / "hierarchical"
    output_path = output_dir / "hierarchical_logits_fold{}.csv.gz".format(
        cli.outer_fold
    )
    atomic_gzip_csv(frame, output_path)
    stats = pd.DataFrame(statistic_rows)
    stats_path = output_dir / "hierarchical_replay_fold{}_stats.csv".format(
        cli.outer_fold
    )
    atomic_csv(stats, stats_path)
    maximum_path = output_dir / "hierarchical_replay_fold{}_maximum_cases.csv".format(
        cli.outer_fold
    )
    atomic_csv(pd.DataFrame(maximum_rows), maximum_path)
    report = {
        "stage": "Stage23A-v2a hierarchical frozen Expert extraction",
        "status": "PASS",
        "outer_fold": int(cli.outer_fold),
        "gpu_id": int(cli.gpu_id),
        "dataset_partition": "Train outer-fold held-out samples only",
        "rows": int(len(frame)),
        "expected_rows": expected_rows,
        "samples": int(frame["sample_id"].nunique()),
        "modes": list(MODES),
        "experts": list(EXPERTS),
        "head_columns": list(HEAD_COLUMNS),
        "mode_active_heads": {key: list(value) for key, value in MODE_ACTIVE_HEADS.items()},
        "duplicates": int(frame.duplicated(["sample_id", "mode", "expert_id"]).sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "finite_logits": bool(
            np.isfinite(frame[list(HEAD_COLUMNS) + ["hierarchical_consistency_mad"]])
            .all()
            .all()
        ),
        "max_abs_final_replay_diff": float(
            stats["max_abs_final_replay_diff"].max()
        ),
        "mean_abs_final_replay_diff": float(
            np.average(
                stats["mean_abs_final_replay_diff"], weights=stats["rows"]
            )
        ),
        "preferred_replay_gate_pass": bool(
            stats["max_abs_final_replay_diff"].max() <= 1e-6
        ),
        "hard_replay_gate_pass": bool(
            stats["max_abs_final_replay_diff"].max() <= 1e-5
        ),
        "sample_binding_sha_verified": True,
        "row_binding_sha_verified": True,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "stats_path": str(stats_path.resolve()),
        "stats_sha256": sha256_file(stats_path),
        "maximum_cases_path": str(maximum_path.resolve()),
        "maximum_cases_sha256": sha256_file(maximum_path),
        "expert_retrained": False,
        "judge_training_started": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "student_trained": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "hierarchical_fold{}_manifest.json".format(
        cli.outer_fold
    )
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2))
    if not report["preferred_replay_gate_pass"]:
        raise RuntimeError("Preferred 1e-6 replay gate failed.")


if __name__ == "__main__":
    main()
