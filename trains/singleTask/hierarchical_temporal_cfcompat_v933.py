"""V9.33 hierarchical temporal information model.

This is the pre-registered feasibility stage for hierarchical temporal CFCompat.
It is deliberately not an output-level expert/router experiment:

* the V9.19 CFCompat backbone is frozen and exposes the representation immediately
  before the original DLF ``proj1`` fusion tail;
* unaligned audio/vision sequences are encoded before the scalar prediction head;
* strictly previous utterances from the same video are encoded causally;
* one copied DLF tail produces the only deployed scalar prediction;
* invalid controls use reversed within-utterance time and deterministic wrong-video
  context. They are diagnostics, never selected using outer labels.

The primary question is whether new temporal information exists. A positive result
is required before training a larger end-to-end model.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _batch_to_device,
    _load_state,
    mode_to_mask,
)
from .oof_group_splits_v92 import canonical_sample_id

VERSION = "hierarchical_temporal_cfcompat_v933_v1"
PRIMARY_VARIANT = "hierarchical_temporal"
VARIANT_NAMES = (
    "current_only",
    "asynchronous_only",
    "ordered_context_only",
    PRIMARY_VARIANT,
)
SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<video_id>.+?)\$_\$(?P<segment_index>[0-9]+)$"
)


@dataclass(frozen=True)
class HierarchicalTemporalConfigV933:
    context_length: int = 3
    temporal_hidden_dim: int = 64
    context_hidden_dim: int = 96
    branch_dropout: float = 0.10
    temporal_kernel_size: int = 5
    batch_size: int = 64
    num_workers: int = 2
    max_epochs: int = 20
    early_stop: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    wrong_context_preservation_weight: float = 0.10
    missing_temporal_weight: float = 0.10
    required_gain_vs_current: float = 0.005
    required_structure_gain: float = 0.002
    required_nondegrading_folds: int = 4
    max_worst_fold_degradation: float = 0.005

    def validate(self) -> None:
        if self.context_length < 1:
            raise ValueError("context_length must be positive")
        if min(self.temporal_hidden_dim, self.context_hidden_dim) < 8:
            raise ValueError("hidden dimensions are too small")
        if not 0 <= self.branch_dropout < 1:
            raise ValueError("branch_dropout must be in [0, 1)")
        if self.temporal_kernel_size < 1 or self.temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd number")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("invalid loader configuration")
        if self.max_epochs < 1 or self.early_stop < 1:
            raise ValueError("invalid epoch configuration")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if min(
            self.wrong_context_preservation_weight,
            self.missing_temporal_weight,
        ) < 0:
            raise ValueError("loss weights must be nonnegative")
        if not 1 <= self.required_nondegrading_folds <= 5:
            raise ValueError("required_nondegrading_folds must be in [1, 5]")


def parse_sample_identity(sample_id: object) -> tuple[str, int]:
    value = canonical_sample_id(sample_id)
    match = SAMPLE_ID_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"unparseable MOSI sample ID: {value!r}")
    return match.group("video_id"), int(match.group("segment_index"))


def deterministic_index(values: Sequence[int], key: str) -> int:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    ordered = sorted(int(value) for value in values)
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return ordered[int(digest[:16], 16) % len(ordered)]


def build_partition_context_bindings(
    sample_ids: Sequence[object],
    partition_indices: Sequence[int],
    context_length: int,
) -> Dict[int, Dict[str, object]]:
    """Build causal and deterministic wrong-video bindings within one partition."""
    partition = sorted(set(int(value) for value in partition_indices))
    identities: Dict[int, tuple[str, int]] = {
        index: parse_sample_identity(sample_ids[index]) for index in partition
    }
    exact_lookup = {identity: index for index, identity in identities.items()}
    result: Dict[int, Dict[str, object]] = {}
    for index in partition:
        video, segment = identities[index]
        ordered = [
            exact_lookup.get((video, segment - offset), -1)
            for offset in range(int(context_length), 0, -1)
        ]
        ordered_mask = [value >= 0 for value in ordered]
        other_video_candidates = [
            candidate
            for candidate in partition
            if identities[candidate][0] != video
        ]
        wrong = []
        wrong_mask = []
        for slot in range(int(context_length)):
            if not other_video_candidates:
                wrong.append(-1)
                wrong_mask.append(False)
            else:
                wrong.append(
                    deterministic_index(
                        other_video_candidates,
                        f"wrong|{index}|{slot}|{video}|{segment}",
                    )
                )
                wrong_mask.append(True)
        result[index] = {
            "ordered_indices": tuple(ordered),
            "ordered_mask": tuple(ordered_mask),
            "wrong_indices": tuple(wrong),
            "wrong_mask": tuple(wrong_mask),
            "video_id": video,
            "segment_index": segment,
        }
    return result


def valid_step_mask(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("temporal tensor must have shape [batch, time, feature]")
    return values.abs().sum(dim=-1) > 1e-8


def reverse_valid_steps(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse only valid steps; keep right padding fixed."""
    result = values.clone()
    for row in range(values.size(0)):
        positions = torch.nonzero(mask[row], as_tuple=False).view(-1)
        if positions.numel() > 1:
            result[row, positions] = values[row, positions.flip(0)]
    return result


def sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(1, dim))
    )
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
    return encoding


class PreFusionCapture:
    """Capture the DLF representation immediately before ``proj1``."""

    def __init__(self, backbone: nn.Module):
        self.value: Optional[torch.Tensor] = None
        self.handle = backbone.proj1.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        del module
        self.value = inputs[0]

    def close(self) -> None:
        self.handle.remove()


@torch.no_grad()
def extract_anchor_representation_cache(
    args,
    dataset: Dataset,
    indices: Sequence[int],
    checkpoint: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    model = MissingModalityWrapper(
        DLF(args),
        int(args.feature_dims[1]),
        int(args.feature_dims[2]),
    ).to(args.device)
    model.load_state_dict(_load_state(checkpoint, args.device), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(
        torch.utils.data.Subset(dataset, [int(value) for value in indices]),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        generator=torch.Generator().manual_seed(int(seed)),
    )
    rows: Dict[int, Dict[str, object]] = {}
    for batch in loader:
        text, audio, vision, labels = _batch_to_device(batch, args.device)
        mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
        capture = PreFusionCapture(model.backbone)
        try:
            output = model(text, audio, vision, mask)
            if capture.value is None:
                raise RuntimeError("V9.33 failed to capture pre-fusion representation")
            feature = capture.value.detach().cpu()
        finally:
            capture.close()
        batch_indices = batch["index"].view(-1).tolist()
        batch_ids = [canonical_sample_id(value) for value in list(batch["id"])]
        for offset, index in enumerate(batch_indices):
            index = int(index)
            if index in rows:
                raise RuntimeError("duplicate representation-cache index")
            rows[index] = {
                "sample_index": index,
                "sample_id": batch_ids[offset],
                "label": float(labels[offset].item()),
                "feature": feature[offset].clone(),
                "anchor_prediction": float(output["output_logit"][offset].item()),
            }
    expected = sorted(set(int(value) for value in indices))
    if sorted(rows) != expected:
        raise RuntimeError("representation cache does not cover requested indices")
    tail_state = {
        "proj1": copy.deepcopy(model.backbone.proj1.state_dict()),
        "proj2": copy.deepcopy(model.backbone.proj2.state_dict()),
        "out_layer": copy.deepcopy(model.backbone.out_layer.state_dict()),
        "output_dropout": float(model.backbone.output_dropout),
        "feature_dim": int(model.backbone.proj1.in_features),
    }
    return {
        "version": VERSION,
        "indices": expected,
        "rows": rows,
        "tail_state": tail_state,
    }


class HierarchicalTemporalDatasetV933(Dataset):
    def __init__(
        self,
        unaligned_dataset,
        indices: Sequence[int],
        cache_rows: Mapping[int, Mapping[str, object]],
        bindings: Mapping[int, Mapping[str, object]],
        context_length: int,
    ):
        self.unaligned = unaligned_dataset
        self.indices = [int(value) for value in indices]
        self.rows = cache_rows
        self.bindings = bindings
        self.context_length = int(context_length)
        if len(self.unaligned) <= max(self.indices, default=-1):
            raise ValueError("unaligned dataset is shorter than requested indices")
        for index in self.indices:
            if index not in self.rows or index not in self.bindings:
                raise ValueError(f"missing cache or context binding for {index}")
            cached_id = canonical_sample_id(self.rows[index]["sample_id"])
            raw_id = canonical_sample_id(self.unaligned.ids[index])
            if cached_id != raw_id:
                raise RuntimeError(
                    f"aligned/unaligned ID mismatch at {index}: {cached_id} != {raw_id}"
                )

    def __len__(self) -> int:
        return len(self.indices)

    def _context_tensor(self, values: Sequence[int], mask: Sequence[bool]):
        feature_dim = int(
            torch.as_tensor(self.rows[self.indices[0]]["feature"]).numel()
        )
        features = torch.zeros(self.context_length, feature_dim, dtype=torch.float32)
        anchor_predictions = torch.zeros(self.context_length, dtype=torch.float32)
        labels = torch.zeros(self.context_length, dtype=torch.float32)
        effective = torch.zeros(self.context_length, dtype=torch.bool)
        for slot, (index, present) in enumerate(zip(values, mask)):
            if not present or int(index) < 0:
                continue
            row = self.rows[int(index)]
            features[slot] = torch.as_tensor(
                row["feature"], dtype=torch.float32
            ).view(-1)
            anchor_predictions[slot] = float(row["anchor_prediction"])
            labels[slot] = float(row["label"])
            effective[slot] = True
        return features, effective, anchor_predictions, labels

    def __getitem__(self, item: int) -> Dict[str, object]:
        index = self.indices[item]
        row = self.rows[index]
        binding = self.bindings[index]
        ordered = self._context_tensor(
            binding["ordered_indices"], binding["ordered_mask"]
        )
        wrong = self._context_tensor(
            binding["wrong_indices"], binding["wrong_mask"]
        )
        return {
            "sample_index": index,
            "sample_id": canonical_sample_id(row["sample_id"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "current_feature": torch.as_tensor(
                row["feature"], dtype=torch.float32
            ).view(-1),
            "anchor_prediction": torch.tensor(
                float(row["anchor_prediction"]), dtype=torch.float32
            ),
            "audio": torch.as_tensor(
                self.unaligned.audio[index], dtype=torch.float32
            ),
            "vision": torch.as_tensor(
                self.unaligned.vision[index], dtype=torch.float32
            ),
            "ordered_context": ordered[0],
            "ordered_context_mask": ordered[1],
            "ordered_context_anchor": ordered[2],
            "ordered_context_labels": ordered[3],
            "wrong_context": wrong[0],
            "wrong_context_mask": wrong[1],
            "video_id": str(binding["video_id"]),
            "segment_index": int(binding["segment_index"]),
        }


class TemporalEventEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        fusion_dim: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.query = nn.Linear(fusion_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 2, fusion_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        values: torch.Tensor,
        current_feature: torch.Tensor,
        reverse_time: bool = False,
        force_missing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = valid_step_mask(values)
        if reverse_time:
            values = reverse_valid_steps(values, mask)
        if force_missing:
            mask = torch.zeros_like(mask)
            values = torch.zeros_like(values)
        encoded = self.conv2(F.gelu(self.conv1(values.transpose(1, 2))))
        encoded = encoded.transpose(1, 2)
        encoded = self.norm(
            encoded
            + sinusoidal_positions(
                encoded.size(1), encoded.size(2), encoded.device, encoded.dtype
            ).unsqueeze(0)
        )
        query = self.query(current_feature)
        keys = self.key(encoded)
        scores = torch.einsum("bth,bh->bt", keys, query) / math.sqrt(keys.size(-1))
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        attention = attention * mask.to(attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("bt,bth->bh", attention, self.value(encoded))
        pooled[empty] = 0.0
        delta = self.output(self.dropout(torch.cat([pooled, query], dim=-1)))
        delta[empty] = 0.0
        return delta, attention


class CausalContextEncoder(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        hidden_dim: int,
        context_length: int,
        dropout: float,
    ):
        super().__init__()
        self.context_length = int(context_length)
        self.input = nn.Linear(fusion_dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.query = nn.Linear(fusion_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 3, fusion_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        current_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.size(0)
        hidden = torch.zeros(
            batch,
            self.cell.hidden_size,
            device=context.device,
            dtype=context.dtype,
        )
        projected = self.input(context)
        for slot in range(context.size(1)):
            candidate = self.cell(projected[:, slot], hidden)
            present = context_mask[:, slot].to(context.dtype).unsqueeze(1)
            hidden = present * candidate + (1.0 - present) * hidden
        query = self.query(current_feature)
        difference = query - hidden
        payload = torch.cat([hidden, query, difference], dim=-1)
        payload = self.norm(payload.view(batch, 3, -1)).view(batch, -1)
        delta = self.output(self.dropout(payload))
        empty = ~context_mask.any(dim=1)
        delta[empty] = 0.0
        return delta, hidden


class HierarchicalTemporalModelV933(nn.Module):
    """Augment the pre-``proj1`` DLF representation and use one copied tail."""

    def __init__(
        self,
        tail_state: Mapping[str, object],
        audio_dim: int,
        vision_dim: int,
        variant: str,
        config: HierarchicalTemporalConfigV933,
    ):
        super().__init__()
        if variant not in VARIANT_NAMES:
            raise ValueError(f"unsupported V9.33 variant: {variant}")
        self.variant = variant
        self.use_async = variant in {"asynchronous_only", PRIMARY_VARIANT}
        self.use_context = variant in {"ordered_context_only", PRIMARY_VARIANT}
        fusion_dim = int(tail_state["feature_dim"])
        self.output_dropout = float(tail_state["output_dropout"])
        self.proj1 = nn.Linear(fusion_dim, fusion_dim)
        self.proj2 = nn.Linear(fusion_dim, fusion_dim)
        self.out_layer = nn.Linear(fusion_dim, 1)
        self.proj1.load_state_dict(tail_state["proj1"], strict=True)
        self.proj2.load_state_dict(tail_state["proj2"], strict=True)
        self.out_layer.load_state_dict(tail_state["out_layer"], strict=True)
        self.audio_encoder = TemporalEventEncoder(
            int(audio_dim),
            int(config.temporal_hidden_dim),
            fusion_dim,
            int(config.temporal_kernel_size),
            float(config.branch_dropout),
        )
        self.vision_encoder = TemporalEventEncoder(
            int(vision_dim),
            int(config.temporal_hidden_dim),
            fusion_dim,
            int(config.temporal_kernel_size),
            float(config.branch_dropout),
        )
        self.context_encoder = CausalContextEncoder(
            fusion_dim,
            int(config.context_hidden_dim),
            int(config.context_length),
            float(config.branch_dropout),
        )

    def tail(self, feature: torch.Tensor) -> torch.Tensor:
        projected = self.proj2(
            F.dropout(
                F.relu(self.proj1(feature), inplace=False),
                p=self.output_dropout,
                training=self.training,
            )
        )
        return self.out_layer(projected + feature).view(-1)

    def forward(
        self,
        current_feature: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        reverse_time: bool = False,
        disable_audio: bool = False,
        disable_vision: bool = False,
        disable_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        audio_delta = torch.zeros_like(current_feature)
        vision_delta = torch.zeros_like(current_feature)
        context_delta = torch.zeros_like(current_feature)
        audio_attention = torch.zeros(
            audio.size(0), audio.size(1), device=audio.device, dtype=audio.dtype
        )
        vision_attention = torch.zeros(
            vision.size(0), vision.size(1), device=vision.device, dtype=vision.dtype
        )
        if self.use_async:
            audio_delta, audio_attention = self.audio_encoder(
                audio, current_feature, reverse_time, disable_audio
            )
            vision_delta, vision_attention = self.vision_encoder(
                vision, current_feature, reverse_time, disable_vision
            )
        if self.use_context and not disable_context:
            context_delta, context_state = self.context_encoder(
                context, context_mask, current_feature
            )
        else:
            context_state = torch.zeros(
                current_feature.size(0),
                self.context_encoder.cell.hidden_size,
                device=current_feature.device,
                dtype=current_feature.dtype,
            )
        augmented = current_feature + audio_delta + vision_delta + context_delta
        prediction = self.tail(augmented)
        current_prediction = self.tail(current_feature)
        return {
            "prediction": prediction,
            "current_prediction": current_prediction,
            "augmented_feature": augmented,
            "audio_delta": audio_delta,
            "vision_delta": vision_delta,
            "context_delta": context_delta,
            "context_state": context_state,
            "audio_attention": audio_attention,
            "vision_attention": vision_attention,
        }


def make_temporal_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        generator=torch.Generator().manual_seed(int(seed)),
        drop_last=False,
    )


def _move_batch(batch: Mapping[str, object], device) -> Dict[str, object]:
    tensor_keys = (
        "label",
        "current_feature",
        "anchor_prediction",
        "audio",
        "vision",
        "ordered_context",
        "ordered_context_mask",
        "wrong_context",
        "wrong_context_mask",
    )
    result = dict(batch)
    for key in tensor_keys:
        result[key] = batch[key].to(device)
    return result


def _checkpoint_reusable(
    path: Path,
    variant: str,
    source_manifest: Mapping[str, object],
    config: HierarchicalTemporalConfigV933,
) -> bool:
    if not Path(path).is_file():
        return False
    payload = torch.load(path, map_location="cpu")
    return (
        payload.get("version") == VERSION
        and payload.get("variant") == variant
        and payload.get("source_manifest") == dict(source_manifest)
        and payload.get("config") == asdict(config)
    )


def train_hierarchical_temporal_model(
    tail_state: Mapping[str, object],
    audio_dim: int,
    vision_dim: int,
    variant: str,
    config: HierarchicalTemporalConfigV933,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device,
    checkpoint: Path,
    source_manifest: Mapping[str, object],
    seed: int,
    resume: bool = True,
) -> Dict[str, object]:
    config.validate()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = HierarchicalTemporalModelV933(
        tail_state, audio_dim, vision_dim, variant, config
    ).to(device)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if resume and _checkpoint_reusable(
        checkpoint, variant, source_manifest, config
    ):
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return {
            "model": model,
            "history": payload.get("history", []),
            "best_epoch": int(payload["best_epoch"]),
            "best_valid_mae": float(payload["best_valid_mae"]),
            "reused": True,
        }
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, int(config.early_stop) // 2),
    )
    amp_enabled = bool(
        config.use_amp
        and torch.cuda.is_available()
        and str(device).startswith("cuda")
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_valid_mae = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        train_values: Dict[str, list[float]] = {
            "total": [],
            "absolute": [],
            "wrong_context_preservation": [],
            "missing_temporal": [],
        }
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            labels = batch["label"].view(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(
                    batch["current_feature"],
                    batch["audio"],
                    batch["vision"],
                    batch["ordered_context"],
                    batch["ordered_context_mask"],
                )
                absolute = F.l1_loss(output["prediction"], labels)
                total = absolute
                wrong_preservation = torch.zeros_like(total)
                if model.use_context and config.wrong_context_preservation_weight > 0:
                    wrong = model(
                        batch["current_feature"],
                        batch["audio"],
                        batch["vision"],
                        batch["wrong_context"],
                        batch["wrong_context_mask"],
                    )
                    no_context = model(
                        batch["current_feature"],
                        batch["audio"],
                        batch["vision"],
                        batch["ordered_context"],
                        batch["ordered_context_mask"],
                        disable_context=True,
                    )
                    wrong_preservation = F.smooth_l1_loss(
                        wrong["prediction"], no_context["prediction"].detach()
                    )
                    total = total + float(
                        config.wrong_context_preservation_weight
                    ) * wrong_preservation
                missing_temporal = torch.zeros_like(total)
                if model.use_async and config.missing_temporal_weight > 0:
                    audio_missing = model(
                        batch["current_feature"],
                        batch["audio"],
                        batch["vision"],
                        batch["ordered_context"],
                        batch["ordered_context_mask"],
                        disable_audio=True,
                    )["prediction"]
                    vision_missing = model(
                        batch["current_feature"],
                        batch["audio"],
                        batch["vision"],
                        batch["ordered_context"],
                        batch["ordered_context_mask"],
                        disable_vision=True,
                    )["prediction"]
                    missing_temporal = 0.5 * (
                        F.l1_loss(audio_missing, labels)
                        + F.l1_loss(vision_missing, labels)
                    )
                    total = total + float(
                        config.missing_temporal_weight
                    ) * missing_temporal
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite V9.33 training loss")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            if float(config.grad_clip_norm) > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.grad_clip_norm)
                )
            scaler.step(optimizer)
            scaler.update()
            train_values["total"].append(float(total.detach().item()))
            train_values["absolute"].append(float(absolute.detach().item()))
            train_values["wrong_context_preservation"].append(
                float(wrong_preservation.detach().item())
            )
            train_values["missing_temporal"].append(
                float(missing_temporal.detach().item())
            )
        valid = predict_hierarchical_temporal_model(
            model, valid_loader, device, diagnostics=False
        )
        valid_mae = float(
            np.mean(np.abs(valid["prediction"] - valid["label"]))
        )
        scheduler.step(valid_mae)
        row = {
            "epoch": int(epoch),
            **{
                f"train_{name}": float(np.mean(values))
                for name, values in train_values.items()
            },
            "valid_mae": valid_mae,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if valid_mae < best_valid_mae - 1e-6:
            best_valid_mae = valid_mae
            best_epoch = epoch
            torch.save(
                {
                    "version": VERSION,
                    "variant": variant,
                    "config": asdict(config),
                    "source_manifest": dict(source_manifest),
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "history": history,
                    "best_epoch": int(best_epoch),
                    "best_valid_mae": float(best_valid_mae),
                    "deployment_output": "single_augmented_dlf_tail",
                    "expert_router_present": False,
                    "scalar_output_residual_present": False,
                },
                checkpoint,
            )
        if epoch - best_epoch >= int(config.early_stop):
            break
    if not checkpoint.is_file():
        raise RuntimeError("V9.33 did not save a checkpoint")
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return {
        "model": model,
        "history": history,
        "best_epoch": int(payload["best_epoch"]),
        "best_valid_mae": float(payload["best_valid_mae"]),
        "reused": False,
    }


@torch.no_grad()
def predict_hierarchical_temporal_model(
    model: HierarchicalTemporalModelV933,
    loader: DataLoader,
    device,
    diagnostics: bool = True,
) -> Dict[str, np.ndarray]:
    model.eval()
    collected: Dict[str, list] = {
        "sample_index": [],
        "sample_id": [],
        "video_id": [],
        "segment_index": [],
        "label": [],
        "anchor_prediction": [],
        "prediction": [],
        "current_prediction": [],
        "wrong_context_prediction": [],
        "reversed_time_prediction": [],
        "both_invalid_prediction": [],
        "audio_delta_norm": [],
        "vision_delta_norm": [],
        "context_delta_norm": [],
        "context_length": [],
    }
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        output = model(
            batch["current_feature"],
            batch["audio"],
            batch["vision"],
            batch["ordered_context"],
            batch["ordered_context_mask"],
        )
        if diagnostics:
            wrong = model(
                batch["current_feature"],
                batch["audio"],
                batch["vision"],
                batch["wrong_context"],
                batch["wrong_context_mask"],
            )["prediction"]
            reversed_time = model(
                batch["current_feature"],
                batch["audio"],
                batch["vision"],
                batch["ordered_context"],
                batch["ordered_context_mask"],
                reverse_time=True,
            )["prediction"]
            both_invalid = model(
                batch["current_feature"],
                batch["audio"],
                batch["vision"],
                batch["wrong_context"],
                batch["wrong_context_mask"],
                reverse_time=True,
            )["prediction"]
        else:
            wrong = output["prediction"]
            reversed_time = output["prediction"]
            both_invalid = output["prediction"]
        batch_size = batch["label"].numel()
        collected["sample_index"].extend(
            int(value) for value in raw_batch["sample_index"].view(-1).tolist()
        )
        collected["sample_id"].extend(
            canonical_sample_id(value) for value in list(raw_batch["sample_id"])
        )
        collected["video_id"].extend(
            str(value) for value in list(raw_batch["video_id"])
        )
        collected["segment_index"].extend(
            int(value) for value in raw_batch["segment_index"].view(-1).tolist()
        )
        for key, tensor in (
            ("label", batch["label"]),
            ("anchor_prediction", batch["anchor_prediction"]),
            ("prediction", output["prediction"]),
            ("current_prediction", output["current_prediction"]),
            ("wrong_context_prediction", wrong),
            ("reversed_time_prediction", reversed_time),
            ("both_invalid_prediction", both_invalid),
            ("audio_delta_norm", output["audio_delta"].norm(dim=1)),
            ("vision_delta_norm", output["vision_delta"].norm(dim=1)),
            ("context_delta_norm", output["context_delta"].norm(dim=1)),
            (
                "context_length",
                batch["ordered_context_mask"].sum(dim=1).to(torch.float32),
            ),
        ):
            values = tensor.detach().cpu().view(-1).tolist()
            if len(values) != batch_size:
                raise RuntimeError(f"prediction diagnostic size mismatch for {key}")
            collected[key].extend(float(value) for value in values)
    result = {}
    for key, values in collected.items():
        if key in {"sample_id", "video_id"}:
            result[key] = np.asarray(values, dtype=object)
        elif key in {"sample_index", "segment_index"}:
            result[key] = np.asarray(values, dtype=np.int64)
        else:
            result[key] = np.asarray(values, dtype=np.float64)
    return result


def paired_metrics(
    prediction: Sequence[float],
    labels: Sequence[float],
    baseline: Sequence[float],
) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if not (prediction.shape == labels.shape == baseline.shape):
        raise ValueError("paired metric shapes differ")
    error = np.abs(prediction - labels)
    baseline_error = np.abs(baseline - labels)
    gain = baseline_error - error
    return {
        "mae": float(error.mean()),
        "gain_vs_baseline": float(gain.mean()),
        "win_rate": float(np.mean(gain > 0)),
        "nondegrade_rate": float(np.mean(gain >= -1e-12)),
        "large_harm_rate_010": float(np.mean(gain < -0.10)),
    }


def group_bootstrap_gain_interval(
    gain: Sequence[float],
    groups: Sequence[object],
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    gain = np.asarray(gain, dtype=np.float64).reshape(-1)
    groups = np.asarray([str(value) for value in groups], dtype=object)
    unique = np.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(repetitions), dtype=np.float64)
    for repeat in range(int(repetitions)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([lookup[group] for group in sampled])
        values[repeat] = float(gain[indices].mean())
    return {
        "gain_ci_low": float(np.quantile(values, 0.025)),
        "gain_ci_high": float(np.quantile(values, 0.975)),
        "bootstrap_positive_probability": float(np.mean(values > 0)),
    }


def success_gate(
    fold_gains: Sequence[float],
    aggregate_gain: float,
    required_gain: float,
    config: HierarchicalTemporalConfigV933,
) -> Dict[str, object]:
    gains = np.asarray(fold_gains, dtype=np.float64).reshape(-1)
    nondegrading = int(np.sum(gains >= -1e-12))
    worst_degradation = float(max(0.0, -float(gains.min())))
    passed = (
        float(aggregate_gain) >= float(required_gain)
        and nondegrading >= int(config.required_nondegrading_folds)
        and worst_degradation <= float(config.max_worst_fold_degradation)
    )
    return {
        "passed": bool(passed),
        "aggregate_gain": float(aggregate_gain),
        "required_gain": float(required_gain),
        "nondegrading_outer_folds": nondegrading,
        "required_nondegrading_outer_folds": int(
            config.required_nondegrading_folds
        ),
        "worst_fold_degradation": worst_degradation,
        "max_worst_fold_degradation": float(
            config.max_worst_fold_degradation
        ),
    }


__all__ = [
    "VERSION",
    "PRIMARY_VARIANT",
    "VARIANT_NAMES",
    "HierarchicalTemporalConfigV933",
    "parse_sample_identity",
    "build_partition_context_bindings",
    "extract_anchor_representation_cache",
    "HierarchicalTemporalDatasetV933",
    "HierarchicalTemporalModelV933",
    "make_temporal_loader",
    "train_hierarchical_temporal_model",
    "predict_hierarchical_temporal_model",
    "paired_metrics",
    "group_bootstrap_gain_interval",
    "success_gate",
    "valid_step_mask",
    "reverse_valid_steps",
]
