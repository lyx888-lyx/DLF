"""Utilities for deterministic source-video-aware batching in CFCompatKD."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Sampler

from data_loader import MMDataset
from .cf_compat_kd_utils import (
    CACHE_COLUMNS,
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    cache_paths,
)
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_video_aware_sampling_v1"
METHOD = "DLF-CFCompatKD-VideoAwareSampling-v1"
OUTPUT_TAG = "cfcompat_video_aware_sampling_v1"
SAMPLES_PER_VIDEO = 4
REPLAY_TOLERANCE = 1e-4
DISCOVERY_REPLAY_TOLERANCE = 1e-4

_VIDEO_SUFFIX_PATTERNS = (
    re.compile(r"^(?P<video>.+?)\$_\$(?P<segment>[^/]+)$"),
    re.compile(r"^(?P<video>.+?)\[(?P<segment>\d+)\]$"),
)


def normalize_sample_id(sample_id) -> str:
    if isinstance(sample_id, bytes):
        return sample_id.decode("utf-8")
    if isinstance(sample_id, np.bytes_):
        return bytes(sample_id).decode("utf-8")
    if isinstance(sample_id, np.str_):
        return str(sample_id)
    if isinstance(sample_id, (tuple, list)):
        if len(sample_id) == 2 and all(
            isinstance(value, (str, bytes, np.str_, np.bytes_))
            for value in sample_id
        ):
            return "{}$_${}".format(
                normalize_sample_id(sample_id[0]),
                normalize_sample_id(sample_id[1]),
            )
        raise ValueError("Unsupported structured sample ID: {!r}".format(sample_id))
    return str(sample_id)


def parse_video_id(sample_id) -> str:
    value = normalize_sample_id(sample_id).strip()
    if not value:
        raise ValueError("Empty sample ID cannot define a video domain.")
    for pattern in _VIDEO_SUFFIX_PATTERNS:
        match = pattern.match(value)
        if match:
            video = match.group("video").strip()
            if video:
                return video
    raise ValueError(
        "Unsupported CMU segment ID format {!r}; expected VIDEO$_$SEGMENT "
        "or VIDEO[SEGMENT].".format(value)
    )


def dataset_video_ids(dataset: MMDataset) -> List[str]:
    return [parse_video_id(value) for value in list(dataset.ids)]


def video_count_frame(split: str, sample_ids: Sequence) -> pd.DataFrame:
    counts = Counter(parse_video_id(value) for value in sample_ids)
    return pd.DataFrame(
        [
            {"split": split, "video_id": video, "sample_count": int(count)}
            for video, count in sorted(counts.items())
        ]
    )


def audit_video_splits(
    datasets: Mapping[str, MMDataset],
) -> Tuple[dict, pd.DataFrame]:
    split_videos: Dict[str, List[str]] = {}
    frames = []
    examples = {}
    for split, dataset in datasets.items():
        ids = list(dataset.ids)
        videos = [parse_video_id(value) for value in ids]
        if len(videos) != len(dataset):
            raise RuntimeError("Video parsing changed sample count for {}.".format(split))
        split_videos[split] = videos
        frames.append(video_count_frame(split, ids))
        examples[split] = [normalize_sample_id(value) for value in ids[:5]]

    overlaps = {}
    names = sorted(split_videos)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps["{}_{}".format(left, right)] = sorted(
                set(split_videos[left]) & set(split_videos[right])
            )
    if any(overlaps.values()):
        raise RuntimeError("Official splits share source videos: {}".format(overlaps))

    counts = pd.concat(frames, ignore_index=True)
    summary = {
        "passed": True,
        "split_sample_counts": {
            split: int(len(dataset)) for split, dataset in datasets.items()
        },
        "split_video_counts": {
            split: int(len(set(split_videos[split]))) for split in names
        },
        "split_video_overlaps": overlaps,
        "sample_id_examples": examples,
        "min_samples_per_video": {
            split: int(min(Counter(split_videos[split]).values()))
            for split in names
        },
        "median_samples_per_video": {
            split: float(np.median(list(Counter(split_videos[split]).values())))
            for split in names
        },
        "max_samples_per_video": {
            split: int(max(Counter(split_videos[split]).values()))
            for split in names
        },
    }
    return summary, counts


class VideoAwareBatchSampler(Sampler[List[int]]):
    """Deterministic exact-coverage batches grouped by source video.

    Every sample occurs exactly once per epoch. Within a batch the sampler takes
    up to ``samples_per_video`` shuffled samples from each selected video while
    keeping multiple videos in the batch. No oversampling or replacement is used.
    """

    def __init__(
        self,
        video_ids: Sequence[str],
        batch_size: int,
        seed: int,
        samples_per_video: int = SAMPLES_PER_VIDEO,
    ):
        if int(batch_size) < 2:
            raise ValueError("Video-aware batches require batch_size >= 2.")
        if int(samples_per_video) < 2:
            raise ValueError("samples_per_video must be at least 2.")
        if int(samples_per_video) > int(batch_size):
            raise ValueError("samples_per_video cannot exceed batch_size.")
        self.video_ids = [str(value) for value in video_ids]
        if not self.video_ids:
            raise ValueError("Video-aware sampler received no samples.")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.samples_per_video = int(samples_per_video)
        self.epoch = 0

        groups: Dict[str, List[int]] = defaultdict(list)
        for index, video in enumerate(self.video_ids):
            groups[video].append(index)
        if len(groups) < 2:
            raise ValueError("Video-aware sampling requires at least two videos.")
        self.groups = {key: tuple(value) for key, value in groups.items()}

    def __len__(self) -> int:
        return int(math.ceil(len(self.video_ids) / float(self.batch_size)))

    def _epoch_batches(self, epoch: int) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + 1000003 * int(epoch))
        queues: Dict[str, deque] = {}
        for video, indices in self.groups.items():
            shuffled = np.asarray(indices, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            queues[video] = deque(int(value) for value in shuffled.tolist())

        all_videos = np.asarray(sorted(queues), dtype=object)
        rng.shuffle(all_videos)
        cycle = deque(str(value) for value in all_videos.tolist())
        batches: List[List[int]] = []

        while any(queues[video] for video in queues):
            batch: List[int] = []
            selected: List[str] = []
            target_groups = max(
                2,
                int(math.ceil(self.batch_size / float(self.samples_per_video))),
            )

            attempts = 0
            while cycle and len(selected) < target_groups:
                video = cycle[0]
                cycle.rotate(-1)
                attempts += 1
                if queues[video] and video not in selected:
                    selected.append(video)
                if attempts > len(cycle) * 3:
                    break

            for video in selected:
                take = min(self.samples_per_video, len(queues[video]))
                for _ in range(take):
                    batch.append(queues[video].popleft())
                    if len(batch) == self.batch_size:
                        break
                if len(batch) == self.batch_size:
                    break

            fill_order = [video for video in cycle if queues[video]]
            if fill_order:
                rng.shuffle(fill_order)
            fill_cursor = 0
            while len(batch) < self.batch_size and fill_order:
                video = fill_order[fill_cursor % len(fill_order)]
                if queues[video]:
                    batch.append(queues[video].popleft())
                fill_order = [item for item in fill_order if queues[item]]
                fill_cursor += 1

            if not batch:
                raise RuntimeError("Video-aware sampler stalled with samples remaining.")
            batches.append(batch)

        flattened = [index for batch in batches for index in batch]
        expected = list(range(len(self.video_ids)))
        if sorted(flattened) != expected or len(flattened) != len(set(flattened)):
            raise RuntimeError("Video-aware sampler did not provide exact coverage.")
        if any(len(batch) > self.batch_size for batch in batches):
            raise RuntimeError("Video-aware sampler exceeded batch size.")
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._epoch_batches(self.epoch)
        self.epoch += 1
        yield from batches

    def preview(self, epochs: int = 1) -> List[List[int]]:
        rows: List[List[int]] = []
        for epoch in range(int(epochs)):
            rows.extend(self._epoch_batches(epoch))
        return rows


def build_video_aware_train_loader(
    args,
    num_workers: int,
    seed: int,
    samples_per_video: int = SAMPLES_PER_VIDEO,
):
    dataset = MMDataset(args, mode="train")
    videos = dataset_video_ids(dataset)
    sampler = VideoAwareBatchSampler(
        videos,
        batch_size=int(args["batch_size"]),
        seed=int(seed),
        samples_per_video=int(samples_per_video),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(num_workers),
    )
    return loader, sampler


def batch_video_statistics(video_ids: Sequence[str]) -> dict:
    counts = Counter(str(value) for value in video_ids)
    repeated = {key: value for key, value in counts.items() if value >= 2}
    return {
        "batch_size": int(len(video_ids)),
        "video_count": int(len(counts)),
        "repeated_video_count": int(len(repeated)),
        "repeated_sample_count": int(sum(repeated.values())),
        "repeated_sample_fraction": float(
            sum(repeated.values()) / max(len(video_ids), 1)
        ),
        "max_samples_one_video": int(max(counts.values())) if counts else 0,
    }


def aggregate_batch_statistics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("No batch statistics were supplied.")
    keys = (
        "batch_size",
        "video_count",
        "repeated_video_count",
        "repeated_sample_count",
        "repeated_sample_fraction",
        "max_samples_one_video",
    )
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def portable_locate_stage1_evaluator(
    result_root,
    dataset,
    seed,
    multiseed: bool = False,
    smoke: bool = False,
):
    if multiseed:
        source = (
            Path(result_root)
            / "missing_baseline"
            / "moddrop_benchmark_multiseed_v1"
        )
        if smoke:
            source = source / "smoke"
        source = (
            source
            / "seed{}".format(int(seed))
            / "{}_per_seed.csv".format(dataset)
        )
        checkpoint_field, epoch_field = "MainCheckpoint", "BestValidEpoch"
    else:
        source = (
            Path(result_root)
            / "missing_baseline"
            / "moddrop"
            / "train"
            / "{}_per_seed.csv".format(dataset)
        )
        checkpoint_field, epoch_field = "Checkpoint", "BestEpoch"

    if not source.is_file():
        raise FileNotFoundError("Required Stage 1 result CSV absent: {}".format(source))
    rows = pd.read_csv(source)
    selected = rows.loc[rows.Seed.astype(int).eq(int(seed))]
    if len(selected) != 1 or checkpoint_field not in selected:
        raise ValueError("Stage 1 CSV has no unique checkpoint for seed {}.".format(seed))

    checkpoint = Path(str(selected.iloc[0][checkpoint_field]))
    if not checkpoint.is_absolute() and not checkpoint.is_file():
        rooted = Path(result_root).resolve().parent / checkpoint
        if rooted.is_file():
            checkpoint = rooted
    if multiseed and (
        "diagnostic" in str(checkpoint) or "best_test" in str(checkpoint)
    ):
        raise ValueError("Counterfactual evaluator must be validation-best ModDrop.")
    if not checkpoint.is_file():
        raise FileNotFoundError("Stage 1 CSV checkpoint is absent: {}".format(checkpoint))
    return checkpoint, int(selected.iloc[0][epoch_field]), source


def load_locked_counterfactual_cache(
    root,
    dataset,
    version=CACHE_VERSION,
    seed=None,
    expected_evaluator_sha=None,
):
    paths = cache_paths(root, dataset, version=version, seed=seed)
    if not paths["csv"].is_file() or not paths["config"].is_file():
        raise FileNotFoundError("Counterfactual cache has not been built.")

    frame = pd.read_csv(paths["csv"])
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    expected_seed = None if seed is None else int(seed)

    if config.get("version") != version or config.get("seed") != expected_seed:
        raise ValueError("Counterfactual cache version/seed binding is invalid.")
    if (
        expected_evaluator_sha is not None
        and config.get("evaluator_sha256") != expected_evaluator_sha
    ):
        raise ValueError("Counterfactual cache evaluator SHA binding is invalid.")
    if config.get("source") != "train_only":
        raise ValueError("Counterfactual cache is not train-only.")

    legacy_seed1111 = (
        version == CACHE_VERSION
        and expected_seed is None
        and "created_from_train_only" not in config
    )
    if (
        config.get("created_from_train_only") is not True
        and not legacy_seed1111
    ):
        raise ValueError("Counterfactual cache is not train-only.")
    if list(frame.columns) != list(CACHE_COLUMNS):
        raise ValueError("Counterfactual cache schema is malformed.")
    if frame.sample_index.duplicated().any():
        raise ValueError("Counterfactual cache sample indices are duplicated.")
    if dataset == "mosi" and (
        len(frame) != 1284 or frame.sample_index.nunique() != 1284
    ):
        raise ValueError("MOSI train cache must contain 1284 unique samples.")

    for mode in MISSING_MODES:
        values = frame["compat_{}".format(mode)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.all(
            (values > 0) & (values < 1)
        ):
            raise ValueError("Cached compatibility is outside (0,1).")

    by_index = {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }
    return frame, by_index, paths


def candidate_valid_gate(
    candidate: dict, baseline: dict, epoch_rows: Sequence[dict]
) -> dict:
    baseline_missing = float(np.mean([
        baseline["valid_{}_MAE".format(mode)] for mode in MISSING_MODES
    ]))
    candidate_missing = float(np.mean([
        candidate["valid_{}_MAE".format(mode)] for mode in MISSING_MODES
    ]))
    gain_j = float(baseline["J_valid"]) - float(candidate["J_valid"])
    gain_lav = (
        float(baseline["valid_LAV_MAE"])
        - float(candidate["valid_LAV_MAE"])
    )
    gain_missing = baseline_missing - candidate_missing
    supporting = int(sum(
        float(row["J_valid"]) <= float(baseline["J_valid"]) - 0.003
        for row in epoch_rows
    ))
    checks = {
        "valid_J_gain_ge_0p005": gain_j >= 0.005,
        "valid_LAV_not_degraded": gain_lav >= 0.0,
        "valid_MissingMacro_not_degraded": gain_missing >= 0.0,
        "at_least_two_supporting_epochs": supporting >= 2,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "gain_valid_J": gain_j,
        "gain_valid_LAV_MAE": gain_lav,
        "gain_valid_MissingMacro_MAE": gain_missing,
        "supporting_epoch_count": supporting,
        "required_gain_valid_J": 0.005,
        "required_supporting_epochs": 2,
    }


def candidate_test_gate(candidate: dict, baseline: dict) -> dict:
    gain_j = (
        float(baseline["J_test_at_valid_best"])
        - float(candidate["J_test_at_valid_best"])
    )
    mode_degradations = {}
    for mode in ("LAV",) + MISSING_MODES:
        key = "test_at_valid_best_{}_MAE".format(mode)
        mode_degradations[mode] = float(candidate[key]) - float(baseline[key])
    primary = gain_j >= 0.005
    safe_positive = gain_j > 0.0 and max(mode_degradations.values()) <= 0.002
    return {
        "passed": bool(primary or safe_positive),
        "primary_gain_ge_0p005": bool(primary),
        "safe_positive_alternative": bool(safe_positive),
        "gain_test_J": gain_j,
        "mode_degradations": mode_degradations,
        "required_primary_gain_test_J": 0.005,
        "max_mode_degradation_for_alternative": 0.002,
    }
