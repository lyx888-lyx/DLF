"""Audit CMU source-video IDs and video-aware batch construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_loader import MMDataset
from train_cf_compat_kd import build_config
from trains.singleTask.video_aware_sampling_utils import (
    OUTPUT_TAG,
    SAMPLES_PER_VIDEO,
    VideoAwareBatchSampler,
    audit_video_splits,
    batch_video_statistics,
    dataset_video_ids,
)
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Audit source-video batches.")
    parser.add_argument("--dataset", choices=("mosi", "mosei"), default="mosi")
    parser.add_argument("--seed", type=int, default=1114)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--result-root", default="result")
    parser.add_argument(
        "--samples-per-video", type=int, default=SAMPLES_PER_VIDEO
    )
    args = parser.parse_args()
    if args.samples_per_video != SAMPLES_PER_VIDEO:
        parser.error("The frozen sampler uses samples_per_video=4.")
    return args


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
        stats = batch_video_statistics(
            [train_videos[index] for index in indices]
        )
        stats["batch_index"] = int(batch_index)
        batch_rows.append(stats)
    frame = pd.DataFrame(batch_rows)

    flattened = [index for batch in preview for index in batch]
    exact_coverage = (
        sorted(flattened) == list(range(len(train_videos)))
        and len(flattened) == len(set(flattened))
    )
    summary.update({
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "batch_size": int(args["batch_size"]),
        "samples_per_video": int(cli.samples_per_video),
        "train_batch_count": int(len(preview)),
        "sampler_exact_coverage": bool(exact_coverage),
        "sampler_duplicate_count": int(len(flattened) - len(set(flattened))),
        "mean_videos_per_batch": float(frame.video_count.mean()),
        "mean_repeated_videos_per_batch": float(
            frame.repeated_video_count.mean()
        ),
        "mean_repeated_sample_fraction": float(
            frame.repeated_sample_fraction.mean()
        ),
        "minimum_videos_per_batch": int(frame.video_count.min()),
        "sampler_viable": bool(
            summary["passed"]
            and exact_coverage
            and len(flattened) == len(train_videos)
            and frame.video_count.mean() >= 2.0
            and frame.repeated_sample_fraction.mean() >= 0.5
        ),
    })

    if not summary["sampler_exact_coverage"]:
        raise RuntimeError("Video-aware sampler failed exact coverage audit.")
    if summary["sampler_duplicate_count"]:
        raise RuntimeError("Video-aware sampler duplicated training samples.")
    if not summary["sampler_viable"]:
        raise RuntimeError("Video-aware batches are not viable.")

    output = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "seed{}".format(cli.seed)
        / "domain_audit"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "video_domain_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts.to_csv(output / "video_domain_counts.csv", index=False)
    frame.to_csv(output / "video_aware_batch_audit.csv", index=False)

    print("Video-aware sampling domain audit passed")
    print("dataset:", cli.dataset)
    print("seed:", cli.seed)
    print("split video counts:", summary["split_video_counts"])
    print(
        "mean videos per batch:",
        "{:.3f}".format(summary["mean_videos_per_batch"]),
    )
    print(
        "mean repeated sample fraction:",
        "{:.3f}".format(summary["mean_repeated_sample_fraction"]),
    )
    print("output:", output)


if __name__ == "__main__":
    main()
