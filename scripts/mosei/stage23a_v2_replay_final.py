"""Replay frozen Expert final predictions for the Stage23A-v2 hard gate.

This is inference only. It never trains an Expert or Judge, never constructs
Official Valid/Test, and deliberately does not export hierarchical features
until final-logit replay consistency has passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage23a_v2_common import (
    EXPERTS,
    MODES,
    ROOT,
    V1_ROOT,
    V2_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.missing_utils import MissingModalityWrapper, mode_to_mask
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True, choices=(0, 1))
    parser.add_argument("--gpu-id", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_args(cli):
    args = get_config_regression("DLF", "mosei", str(ROOT / "config/config.json"))
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False
    args.train_mode = "regression"
    args.batch_size = int(cli.batch_size)
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def load_state(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    return model


def make_loader(dataset, indices, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [int(value) for value in indices]),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
    )


def main():
    cli = parse_args()
    draft_path = V2_ROOT / "protocol" / "draft_protocol_manifest.json"
    if not draft_path.is_file():
        raise FileNotFoundError("Run stage23a_v2_prepare_protocol.py first.")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft["status"] != "AWAITING_CHECKPOINT_REPLAY":
        raise RuntimeError("Unexpected v2 draft protocol status.")
    if any(
        (
            draft["official_valid_access_count"],
            draft["locked_test_access_count"],
            draft["test_loader_constructed"],
            draft["student_trained"],
            draft["judge_training_started"],
        )
    ):
        raise RuntimeError("A Stage23A-v2 data lock is open.")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    setup_seed(23200 + int(cli.outer_fold))
    args = build_args(cli)
    dataset = MMDataset(args, mode="train")
    args.seq_lens = dataset.get_seq_len()

    split = pd.read_csv(
        V1_ROOT / "protocol" / "source_splits.csv",
        dtype={"sample_id": str, "video_id": str},
    )
    holdout_indices = split.loc[
        split["outer_fold"].astype(int) == int(cli.outer_fold), "train_index"
    ].tolist()
    loader = make_loader(dataset, holdout_indices, cli.batch_size, cli.num_workers)
    old_path = (
        V1_ROOT
        / "expert_oof"
        / "outer_fold{}".format(cli.outer_fold)
        / "oof_predictions.csv"
    )
    old = pd.read_csv(old_path, dtype={"sample_id": str, "video_id": str})
    old = old.set_index(["sample_id", "mode", "expert_id"]).sort_index()

    statistic_rows = []
    maximum_rows = []
    for expert in EXPERTS:
        component_dir = (
            V1_ROOT
            / "expert_oof"
            / "outer_fold{}".format(cli.outer_fold)
            / "experts"
            / expert
        )
        run_manifest_path = component_dir / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(run_manifest["checkpoint"])
        if sha256_file(checkpoint) != run_manifest["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint SHA mismatch: {}".format(expert))
        model = MissingModalityWrapper(
            DLF(args), args.feature_dims[1], args.feature_dims[2]
        )
        model = load_state(model, checkpoint, "cpu").to(args.device)
        model.eval()
        records = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, 1):
                if cli.max_batches and batch_index > cli.max_batches:
                    break
                text = batch["text"].to(args.device)
                audio = batch["audio"].to(args.device)
                vision = batch["vision"].to(args.device)
                identifiers = [str(value) for value in batch["id"]]
                for mode in MODES:
                    mask = mode_to_mask(
                        mode, text.size(0), args.device, audio.dtype
                    )
                    # Only final prediction is read in this hard-gate pass.
                    prediction = (
                        model(text, audio, vision, mask)["output_logit"]
                        .view(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    for sample_id, value in zip(identifiers, prediction):
                        records.append(
                            {
                                "sample_id": sample_id,
                                "mode": mode,
                                "expert_id": expert,
                                "replayed_prediction": float(value),
                            }
                        )
        replay = pd.DataFrame(records).set_index(
            ["sample_id", "mode", "expert_id"]
        )
        expected = old.loc[old.index.get_level_values("expert_id") == expert]
        if cli.max_batches:
            expected = expected.loc[expected.index.isin(replay.index)]
        if set(replay.index) != set(expected.index):
            raise RuntimeError("Replay sample binding differs for {}.".format(expert))
        comparison = expected[["prediction"]].join(replay, how="inner")
        difference = np.abs(
            comparison["replayed_prediction"].to_numpy(dtype=np.float64)
            - comparison["prediction"].to_numpy(dtype=np.float64)
        )
        maximum_position = int(np.argmax(difference))
        maximum_index = comparison.index[maximum_position]
        statistic_rows.append(
            {
                "outer_fold": int(cli.outer_fold),
                "expert_id": expert,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "rows": int(len(comparison)),
                "max_abs_diff": float(difference.max()),
                "mean_abs_diff": float(difference.mean()),
                "mismatch_gt_1e_6": int((difference > 1e-6).sum()),
                "mismatch_gt_1e_5": int((difference > 1e-5).sum()),
                "preferred_gate_pass": bool(difference.max() <= 1e-6),
                "hard_gate_pass": bool(difference.max() <= 1e-5),
            }
        )
        maximum_rows.append(
            {
                "outer_fold": int(cli.outer_fold),
                "expert_id": expert,
                "sample_id": maximum_index[0],
                "mode": maximum_index[1],
                "old_prediction": float(
                    comparison.iloc[maximum_position]["prediction"]
                ),
                "replayed_prediction": float(
                    comparison.iloc[maximum_position]["replayed_prediction"]
                ),
                "absolute_difference": float(difference[maximum_position]),
            }
        )
        model.cpu()
        del model
        torch.cuda.empty_cache()
        print(
            "fold={} expert={} rows={} max_abs_diff={:.10g}".format(
                cli.outer_fold, expert, len(comparison), difference.max()
            ),
            flush=True,
        )

    stats = pd.DataFrame(statistic_rows)
    maxima = pd.DataFrame(maximum_rows)
    validation = (
        V2_ROOT / "validation" / "smoke"
        if cli.max_batches
        else V2_ROOT / "validation"
    )
    stats_path = validation / "replay_fold{}_statistics.csv".format(cli.outer_fold)
    maxima_path = validation / "replay_fold{}_maximum_cases.csv".format(
        cli.outer_fold
    )
    atomic_csv(stats, stats_path)
    atomic_csv(maxima, maxima_path)
    expected_rows = (
        int(draft["inputs"]["outer_folds"][cli.outer_fold]["samples"])
        * len(MODES)
        * len(EXPERTS)
    )
    final_row_count_matches = int(stats["rows"].sum()) == expected_rows
    report = {
        "stage": "Stage23A-v2 frozen Expert final-prediction replay",
        "status": (
            "SMOKE_PASS_NOT_FINAL"
            if cli.max_batches and bool(stats["hard_gate_pass"].all())
            else (
                "PASS"
                if bool(stats["hard_gate_pass"].all()) and final_row_count_matches
                else "FAIL_REPLAY_OR_ROW_COUNT"
            )
        ),
        "outer_fold": int(cli.outer_fold),
        "gpu_id": int(cli.gpu_id),
        "dataset_partition": "train outer-holdout indices only",
        "old_oof_path": str(old_path),
        "old_oof_sha256": sha256_file(old_path),
        "model_eval": True,
        "dropout_disabled": True,
        "random_augmentation": False,
        "expert_retrained": False,
        "hierarchical_features_extracted": False,
        "rows": int(stats["rows"].sum()),
        "expected_rows": expected_rows,
        "final_row_count_matches": final_row_count_matches,
        "max_abs_diff": float(stats["max_abs_diff"].max()),
        "mean_abs_diff": float(
            np.average(stats["mean_abs_diff"], weights=stats["rows"])
        ),
        "mismatch_gt_1e_6": int(stats["mismatch_gt_1e_6"].sum()),
        "mismatch_gt_1e_5": int(stats["mismatch_gt_1e_5"].sum()),
        "preferred_threshold": 1e-6,
        "hard_stop_threshold": 1e-5,
        "statistics_path": str(stats_path),
        "statistics_sha256": sha256_file(stats_path),
        "maximum_cases_path": str(maxima_path),
        "maximum_cases_sha256": sha256_file(maxima_path),
        "completed_at": utc_now(),
        "judge_training_started": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
    }
    report_path = validation / "replay_fold{}_report.json".format(cli.outer_fold)
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] not in ("PASS", "SMOKE_PASS_NOT_FINAL"):
        raise RuntimeError(report["status"])


if __name__ == "__main__":
    main()
