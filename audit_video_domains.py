"""Audit CMU MOSI/MOSEI sample IDs and video-aware batch viability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import MMDataset
from train_cf_compat_kd import build_config
from trains.singleTask.video_vrex_utils import (
    SAMPLES_PER_VIDEO,
    VideoAwareBatchSampler,
    audit_video_splits,
    batch_video_statistics,
    dataset_video_ids,
)
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Audit source-video domains.")
    parser.add_argument("--dataset", choices=("mosi", "mosei"), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--samples-per-video", type=int, default=SAMPLES_PER_VIDEO)
    return parser.parse_args()


def main():
    cli = parse_args()
    setup_seed(cli.seed)
    args = build_config(cli, cli.seed)
    datasets = {
        split: MMDataset(args, mode=split)
        for split in ("train", "valid", "test")
    }
    summary, counts = audit_video_splits(datasets)

    train_videos = dataset_video_ids(datasets["train"])
    sampler = VideoAwareBatchSampler(
        train_videos,
        batch_size=int(args["batch_size"]),
        seed=cli.seed,
        samples_per_video=cli.samples_per_video,
    )
    preview = sampler.preview(epochs=1)
    batch_rows = []
    for batch_index, indices in enumerate(preview):
        stats = batch_video_statistics([train_videos[index] for index in indices])
        stats["batch_index"] = batch_index
        batch_rows.append(stats)
    batch_frame = pd.DataFrame(batch_rows)
    active = batch_frame.eligible_video_count.ge(2)

    summary.update(
        {
            "dataset": cli.dataset,
            "seed": int(cli.seed),
            "batch_size": int(args["batch_size"]),
            "samples_per_video": int(cli.samples_per_video),
            "train_batch_count": int(len(preview)),
            "sampler_exact_coverage": bool(
                sorted(index for batch in preview for index in batch)
                == list(range(len(train_videos)))
            ),
            "sampler_duplicate_count": int(
                len([index for batch in preview for index in batch])
                - len(set(index for batch in preview for index in batch))
            ),
            "mean_videos_per_batch": float(batch_frame.video_count.mean()),
            "mean_eligible_videos_per_batch": float(
                batch_frame.eligible_video_count.mean()
            ),
            "mean_eligible_sample_fraction": float(
                batch_frame.eligible_sample_fraction.mean()
            ),
            "minimum_eligible_videos_per_batch": int(
                batch_frame.eligible_video_count.min()
            ),
            "vrex_active_batch_fraction": float(active.mean()),
            "inactive_batch_count": int((~active).sum()),
            "vrex_batch_viable": bool(
                summary["passed"]
                and active.mean() >= 0.95
                and batch_frame.eligible_sample_fraction.mean() >= 0.5
                and int(active.sum()) >= 2
            ),
        }
    )
    if not summary["sampler_exact_coverage"] or summary["sampler_duplicate_count"]:
        raise RuntimeError("Video-aware sampler failed exact coverage audit.")
    if not summary["vrex_batch_viable"]:
        raise RuntimeError("Video-aware batches are not viable for V-REx.")

    output = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_video_vrex_v1"
        / cli.dataset
        / "domain_audit"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "video_domain_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts.to_csv(output / "video_domain_counts.csv", index=False)
    batch_frame.to_csv(output / "video_aware_batch_audit.csv", index=False)

    print("Video-domain audit passed")
    print("dataset:", cli.dataset)
    print("split video counts:", summary["split_video_counts"])
    print("mean eligible videos per batch:", "{:.3f}".format(summary["mean_eligible_videos_per_batch"]))
    print("mean eligible sample fraction:", "{:.3f}".format(summary["mean_eligible_sample_fraction"]))
    print("active batch fraction:", "{:.3f}".format(summary["vrex_active_batch_fraction"]))
    print("output:", output)


if __name__ == "__main__":
    main()
