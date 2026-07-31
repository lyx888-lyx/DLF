"""Grouped nested fold construction for V9.2 OOF training."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class OOFFoldSpec:
    outer_fold: int
    inner_train_indices: Tuple[int, ...]
    inner_valid_indices: Tuple[int, ...]
    outer_holdout_indices: Tuple[int, ...]
    inner_train_groups: Tuple[str, ...]
    inner_valid_groups: Tuple[str, ...]
    outer_holdout_groups: Tuple[str, ...]


@dataclass(frozen=True)
class StageLimits:
    clean_max_epochs: int = 40
    moddrop_max_epochs: int = 30
    cfcompat_max_epochs: int = 30
    early_stop: int = 7


def canonical_sample_id(value: object) -> str:
    """Return a stable readable sample ID without DataLoader assumptions."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return canonical_sample_id(value.item())
        return "|".join(canonical_sample_id(item) for item in value.tolist())
    if isinstance(value, (tuple, list)):
        return "|".join(canonical_sample_id(item) for item in value)
    text = str(value).strip()
    if len(text) >= 3 and text[0] == "b" and text[1] in ("'", '"'):
        text = text[2:-1]
    return text


def conversation_group_id(sample_id: object) -> str:
    """Infer the source-video group from a MOSI/MOSEI segment identifier."""
    text = canonical_sample_id(sample_id)
    if not text:
        raise ValueError("empty sample id")

    for delimiter in ("$_$", "_$_", "|", "::"):
        if delimiter in text:
            head = text.split(delimiter, 1)[0].strip()
            if head:
                return head

    bracket = re.match(r"^(.*?)(?:\[[^\]]+\])$", text)
    if bracket and bracket.group(1).strip():
        return bracket.group(1).strip()

    suffix = re.match(
        r"^(.*?)[_-](?:seg(?:ment)?[_-]?)?\d+$",
        text,
        flags=re.IGNORECASE,
    )
    if suffix and suffix.group(1).strip():
        return suffix.group(1).strip()
    return text


def _tail_vector(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    return np.stack(
        [
            np.ones_like(values),
            (values < -1.5).astype(np.float64),
            (values > 1.5).astype(np.float64),
            np.abs(values),
        ],
        axis=1,
    )


def _stable_hash(value: object, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest(), 16)


def _greedy_group_partition(
    group_to_indices: Mapping[str, Sequence[int]],
    labels: np.ndarray,
    partitions: int,
    seed: int,
) -> List[List[str]]:
    if partitions < 2:
        raise ValueError("partitions must be at least 2")
    if len(group_to_indices) < partitions:
        raise ValueError("number of groups is smaller than partitions")

    group_stats = {
        group: _tail_vector(labels[np.asarray(indices, dtype=int)]).sum(axis=0)
        for group, indices in group_to_indices.items()
    }
    total = np.stack(list(group_stats.values()), axis=0).sum(axis=0)
    target = total / float(partitions)
    scales = np.maximum(target, 1.0)
    order = sorted(
        group_stats,
        key=lambda group: (
            -float(group_stats[group][1] + group_stats[group][2]),
            -float(group_stats[group][0]),
            _stable_hash(group, seed),
        ),
    )

    assignments: List[List[str]] = [[] for _ in range(partitions)]
    loads = np.zeros((partitions, group_stats[order[0]].shape[0]), dtype=np.float64)
    for group in order:
        vector = group_stats[group]
        candidates = []
        for partition in range(partitions):
            proposed = loads.copy()
            proposed[partition] += vector
            imbalance = np.square((proposed - target) / scales).mean()
            empty_bonus = -1e-6 if not assignments[partition] else 0.0
            candidates.append(
                (float(imbalance + empty_bonus), len(assignments[partition]), partition)
            )
        _, _, chosen = min(candidates)
        assignments[chosen].append(group)
        loads[chosen] += vector

    if any(not values for values in assignments):
        raise RuntimeError("group partition produced an empty fold")
    return assignments


def build_nested_group_folds(
    sample_ids: Sequence[object],
    labels: Sequence[float],
    outer_folds: int = 3,
    inner_valid_fraction: float = 0.15,
    seed: int = 1111,
):
    labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(sample_ids) != len(labels_array):
        raise ValueError("sample_ids and labels have different lengths")
    if not 0.05 <= float(inner_valid_fraction) <= 0.40:
        raise ValueError("inner_valid_fraction must be in [0.05, 0.40]")

    group_ids = [conversation_group_id(value) for value in sample_ids]
    group_to_indices: MutableMapping[str, List[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        group_to_indices[group].append(index)
    singleton_fraction = float(
        np.mean([len(indices) == 1 for indices in group_to_indices.values()])
    )
    if len(sample_ids) >= 100 and singleton_fraction > 0.90:
        raise RuntimeError(
            "More than 90% of inferred groups are singletons. The sample-ID "
            "format is probably unrecognized; inspect dataset['train']['id'] "
            "rather than silently using segment-level folds."
        )

    outer_groups = _greedy_group_partition(
        group_to_indices, labels_array, int(outer_folds), int(seed)
    )
    all_groups = set(group_to_indices)
    specs: List[OOFFoldSpec] = []
    manifest_rows = []

    for outer_fold, holdout_groups_list in enumerate(outer_groups):
        holdout_groups = set(holdout_groups_list)
        remaining_groups = sorted(all_groups - holdout_groups)
        inner_partition_count = max(
            2, int(round(1.0 / float(inner_valid_fraction)))
        )
        inner_partition_count = min(inner_partition_count, len(remaining_groups))
        remaining_map = {
            group: group_to_indices[group] for group in remaining_groups
        }
        inner_buckets = _greedy_group_partition(
            remaining_map,
            labels_array,
            inner_partition_count,
            int(seed) + 10007 * (outer_fold + 1),
        )
        inner_valid_bucket = outer_fold % inner_partition_count
        inner_valid_groups = set(inner_buckets[inner_valid_bucket])
        inner_train_groups = set(remaining_groups) - inner_valid_groups

        if holdout_groups & inner_valid_groups or holdout_groups & inner_train_groups:
            raise RuntimeError("outer holdout group leakage")
        if inner_train_groups & inner_valid_groups:
            raise RuntimeError("inner train/validation group leakage")

        def indices_for(groups: Iterable[str]) -> Tuple[int, ...]:
            return tuple(
                sorted(
                    index
                    for group in groups
                    for index in group_to_indices[group]
                )
            )

        spec = OOFFoldSpec(
            outer_fold=int(outer_fold),
            inner_train_indices=indices_for(inner_train_groups),
            inner_valid_indices=indices_for(inner_valid_groups),
            outer_holdout_indices=indices_for(holdout_groups),
            inner_train_groups=tuple(sorted(inner_train_groups)),
            inner_valid_groups=tuple(sorted(inner_valid_groups)),
            outer_holdout_groups=tuple(sorted(holdout_groups)),
        )
        if (
            not spec.inner_train_indices
            or not spec.inner_valid_indices
            or not spec.outer_holdout_indices
        ):
            raise RuntimeError("nested fold contains an empty partition")
        for partition_name, partition_indices in (
            ("inner_train", spec.inner_train_indices),
            ("inner_valid", spec.inner_valid_indices),
            ("outer_holdout", spec.outer_holdout_indices),
        ):
            partition_labels = labels_array[np.asarray(partition_indices, dtype=int)]
            if not np.any(partition_labels < -1.5):
                raise RuntimeError(
                    f"outer fold {outer_fold} {partition_name} has no strong-negative sample"
                )
            if not np.any(partition_labels > 1.5):
                raise RuntimeError(
                    f"outer fold {outer_fold} {partition_name} has no strong-positive sample"
                )
        specs.append(spec)

        partition_by_group = {
            **{group: "inner_train" for group in inner_train_groups},
            **{group: "inner_valid" for group in inner_valid_groups},
            **{group: "outer_holdout" for group in holdout_groups},
        }
        for index, group in enumerate(group_ids):
            manifest_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "sample_index": int(index),
                    "sample_id": canonical_sample_id(sample_ids[index]),
                    "group_id": group,
                    "partition": partition_by_group[group],
                    "label": float(labels_array[index]),
                }
            )

    holdout_counts = Counter(
        index for spec in specs for index in spec.outer_holdout_indices
    )
    if len(holdout_counts) != len(sample_ids) or any(
        value != 1 for value in holdout_counts.values()
    ):
        raise RuntimeError("every sample must appear in exactly one outer holdout")
    return specs, pd.DataFrame(manifest_rows)


def build_subset_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    subset = Subset(dataset, list(map(int, indices)))
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        shuffle=bool(shuffle),
        generator=generator,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
