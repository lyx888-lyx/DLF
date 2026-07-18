"""Deterministic causal MOSI discourse-context indexing for Stage 15."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd


CONTEXT_LENGTH = 3
SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<video_id>.+?)\$_\$(?P<segment_index>[0-9]+)$"
)


@dataclass(frozen=True)
class ParsedSampleID:
    sample_id: str
    video_id: str
    segment_index: int


def parse_mosi_sample_id(sample_id: str) -> ParsedSampleID:
    """Parse the observed MOSI ``video$_$segment`` identifier format."""
    value = str(sample_id)
    match = SAMPLE_ID_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("Unparseable MOSI sample ID: {!r}".format(value))
    video_id = match.group("video_id")
    segment_index = int(match.group("segment_index"))
    if not video_id or segment_index < 0:
        raise ValueError("Invalid MOSI sample ID: {!r}".format(value))
    return ParsedSampleID(value, video_id, segment_index)


def parse_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {"sample_index", "sample_id"}
    if not required.issubset(frame.columns):
        raise ValueError("Frame lacks columns: {}".format(sorted(required - set(frame.columns))))
    local = frame.loc[:, ["sample_index", "sample_id"]].copy()
    local["sample_index"] = local["sample_index"].astype(int)
    if local.sample_index.duplicated().any():
        raise ValueError("Duplicate sample_index in {}.".format(split))
    parsed = [parse_mosi_sample_id(value) for value in local.sample_id]
    local.insert(0, "split", str(split))
    local["sample_id"] = [value.sample_id for value in parsed]
    local["video_id"] = [value.video_id for value in parsed]
    local["segment_index"] = [value.segment_index for value in parsed]
    if local.duplicated(["video_id", "segment_index"]).any():
        raise ValueError("Duplicate video/segment in {}.".format(split))
    return local.sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def build_context_index(parsed: pd.DataFrame, k: int = CONTEXT_LENGTH) -> pd.DataFrame:
    """Build strictly past, same-video, left-padded causal context bindings."""
    if int(k) != CONTEXT_LENGTH:
        raise ValueError("Stage 15 freezes K={}; received {}.".format(CONTEXT_LENGTH, k))
    required = {"split", "sample_index", "sample_id", "video_id", "segment_index"}
    if not required.issubset(parsed.columns):
        raise ValueError("Parsed frame is incomplete.")
    if parsed.split.nunique() != 1:
        raise ValueError("A context index must contain exactly one split.")
    lookup = {
        (str(row.video_id), int(row.segment_index)): str(row.sample_id)
        for row in parsed.itertuples(index=False)
    }
    rows = []
    for row in parsed.itertuples(index=False):
        history = []
        mask = []
        for offset in range(CONTEXT_LENGTH, 0, -1):
            sample = lookup.get((str(row.video_id), int(row.segment_index) - offset))
            history.append("" if sample is None else sample)
            mask.append(0 if sample is None else 1)
        rows.append(
            {
                "split": str(row.split),
                "sample_index": int(row.sample_index),
                "sample_id": str(row.sample_id),
                "video_id": str(row.video_id),
                "segment_index": int(row.segment_index),
                "context_sample_id_1": history[0],
                "context_sample_id_2": history[1],
                "context_sample_id_3": history[2],
                "context_length": int(sum(mask)),
                "context_mask": "".join(map(str, mask)),
            }
        )
    result = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    validate_context_index(result, parsed)
    return result


def validate_context_index(context: pd.DataFrame, parsed: pd.DataFrame) -> Mapping[str, int]:
    """Return binding error counts; raise when the schema itself is malformed."""
    if len(context) != len(parsed):
        raise ValueError("Context and parsed row counts differ.")
    sample_to_identity = {
        str(row.sample_id): (str(row.split), str(row.video_id), int(row.segment_index))
        for row in parsed.itertuples(index=False)
    }
    counts = {
        "missing_current_sample": 0,
        "future_or_current_context": 0,
        "cross_video_context": 0,
        "cross_split_context": 0,
        "mask_binding_mismatch": 0,
        "context_length_mismatch": 0,
        "duplicate_context": 0,
    }
    columns = ["context_sample_id_1", "context_sample_id_2", "context_sample_id_3"]
    for row in context.itertuples(index=False):
        current = sample_to_identity.get(str(row.sample_id))
        if current is None:
            counts["missing_current_sample"] += 1
            continue
        mask = [int(value) for value in str(row.context_mask)]
        values = [str(getattr(row, column)) for column in columns]
        present = [value for value in values if value]
        if len(present) != len(set(present)):
            counts["duplicate_context"] += 1
        if len(mask) != CONTEXT_LENGTH or any(value not in (0, 1) for value in mask):
            counts["mask_binding_mismatch"] += 1
            continue
        if int(row.context_length) != sum(mask):
            counts["context_length_mismatch"] += 1
        for position, value in enumerate(values):
            if bool(value) != bool(mask[position]):
                counts["mask_binding_mismatch"] += 1
                continue
            if not value:
                continue
            identity = sample_to_identity.get(value)
            if identity is None:
                counts["missing_current_sample"] += 1
                continue
            split, video, segment = identity
            if split != current[0]:
                counts["cross_split_context"] += 1
            if video != current[1]:
                counts["cross_video_context"] += 1
            if segment >= current[2]:
                counts["future_or_current_context"] += 1
    return counts


def canonical_context_sha(frame: pd.DataFrame) -> str:
    columns = [
        "split",
        "sample_index",
        "sample_id",
        "video_id",
        "segment_index",
        "context_sample_id_1",
        "context_sample_id_2",
        "context_sample_id_3",
        "context_length",
        "context_mask",
    ]
    records = frame.loc[:, columns].sort_values(
        ["split", "sample_index"], kind="mergesort"
    ).to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def deterministic_hash_choice(values: Sequence[str], key: str) -> str:
    """Choose one value without labels or mutable RNG state."""
    candidates = sorted(map(str, values))
    if not candidates:
        raise ValueError("Cannot choose from an empty sequence.")
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return candidates[int(digest[:16], 16) % len(candidates)]


def train_inner_video_split(video_ids: Iterable[str]) -> Mapping[str, str]:
    """Frozen Stage 15 hash split: buckets 0/1 are inner-valid."""
    result = {}
    for video_id in sorted(set(map(str, video_ids))):
        value = int(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:8], 16)
        result[video_id] = "inner_valid" if value % 10 in {0, 1} else "inner_train"
    return result
