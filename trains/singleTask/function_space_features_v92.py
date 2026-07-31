"""Fold-aligned function-space features for V9.2 cross-fitted experts."""

from __future__ import annotations

import torch

from .cfcompat_fold_training_v92 import mode_to_mask
from .oof_group_splits_v92 import canonical_sample_id

FEATURE_SPACE_VERSION = "auxiliary_prediction_logits_v1"
FEATURE_KEYS = (
    "logits_l_hetero",
    "logits_a_hetero",
    "logits_v_hetero",
    "logits_c",
)


def _identifier_from_dataset(dataset, sample_index, fallback):
    base = dataset
    while not hasattr(base, "ids") and hasattr(base, "dataset"):
        base = base.dataset
    if hasattr(base, "ids"):
        return canonical_sample_id(base.ids[int(sample_index)])
    return canonical_sample_id(fallback)


def function_space_feature(output):
    """Return four scalar prediction branches with consistent semantics.

    Hidden representation coordinates from independently trained outer-fold
    models are not guaranteed to align. These four task-space outputs do align:
    every coordinate is a sentiment prediction from the same named DLF branch.
    """
    missing = [key for key in FEATURE_KEYS if key not in output]
    if missing:
        raise KeyError(f"model output is missing function-space keys: {missing}")
    feature = torch.cat([output[key].view(-1, 1) for key in FEATURE_KEYS], dim=1)
    if feature.size(1) != len(FEATURE_KEYS) or not torch.isfinite(feature).all():
        raise FloatingPointError("invalid function-space feature")
    return feature


@torch.no_grad()
def predict_wrapper_function_space(model, loader, device, capture_features=True):
    """Collect LAV prediction and aligned function-space features by sample index."""
    model.eval()
    rows = []
    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["labels"]["M"].to(device).view(-1, 1)
        mask = mode_to_mask("LAV", labels.size(0), device, audio.dtype)
        output = model(text, audio, vision, mask)
        feature = function_space_feature(output).detach().cpu()
        indices = batch["index"].view(-1).cpu().tolist()
        fallback_ids = list(batch.get("id", []))
        identifiers = [
            _identifier_from_dataset(
                loader.dataset,
                index,
                fallback_ids[offset] if offset < len(fallback_ids) else index,
            )
            for offset, index in enumerate(indices)
        ]
        for offset, index in enumerate(indices):
            row = {
                "sample_index": int(index),
                "sample_id": identifiers[offset],
                "label": float(labels[offset].item()),
                "prediction": float(output["output_logit"][offset].item()),
            }
            if capture_features:
                row["feature"] = feature[offset].clone()
            rows.append(row)
    return rows


@torch.no_grad()
def collect_wrapper_function_space(model, dataloader, device):
    """Collect a complete contiguous Validation/Test split."""
    rows = predict_wrapper_function_space(model, dataloader, device, True)
    rows.sort(key=lambda row: row["sample_index"])
    sample_count = len(dataloader.dataset)
    if len(rows) != sample_count:
        raise RuntimeError("function-space collection sample count mismatch")
    if [row["sample_index"] for row in rows] != list(range(sample_count)):
        raise RuntimeError("function-space collection indices are not contiguous")
    return {
        "sample_ids": [row["sample_id"] for row in rows],
        "labels": torch.tensor([row["label"] for row in rows]).view(-1, 1),
        "anchor": torch.tensor([row["prediction"] for row in rows]).view(-1, 1),
        "feature": torch.stack([row["feature"] for row in rows], dim=0),
        "feature_space": FEATURE_SPACE_VERSION,
        "feature_keys": list(FEATURE_KEYS),
    }
