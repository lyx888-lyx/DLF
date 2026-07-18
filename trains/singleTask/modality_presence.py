"""Authoritative mode-metadata presence handling for Stage 17 MCAO."""

from __future__ import annotations

import torch


MODES = ("LAV", "LA", "LV", "L")
MODALITIES = ("L", "A", "V")
MODE_PRESENCE = {
    "LAV": (1.0, 1.0, 1.0),
    "LA": (1.0, 1.0, 0.0),
    "LV": (1.0, 0.0, 1.0),
    "L": (1.0, 0.0, 0.0),
}


def presence_from_mode(mode, batch_size=None, device=None, dtype=torch.float32):
    """Return presence from explicit mode metadata, never from feature values."""
    if mode not in MODE_PRESENCE:
        raise ValueError("Unknown missingness mode: {}".format(mode))
    value = torch.tensor(MODE_PRESENCE[mode], device=device, dtype=dtype)
    if batch_size is not None:
        value = value.unsqueeze(0).expand(int(batch_size), -1)
    return value


def modes_from_presence(presence):
    """Bind every [L,A,V] row to one canonical mode and reject other masks."""
    if presence.ndim != 2 or presence.size(1) != 3:
        raise ValueError("presence must have shape [batch, 3].")
    rows = []
    for row in presence.detach().cpu():
        values = tuple(float(item) for item in row.tolist())
        matches = [mode for mode, expected in MODE_PRESENCE.items() if values == expected]
        if len(matches) != 1:
            raise ValueError("Non-canonical presence row: {}".format(values))
        rows.append(matches[0])
    return rows


def required_modalities_active(required_modalities, presence):
    """Per-sample semantic activation for an explicit dependency set."""
    required = tuple(required_modalities)
    unknown = set(required).difference(MODALITIES)
    if unknown:
        raise ValueError("Unknown modalities: {}".format(sorted(unknown)))
    if presence.ndim != 2 or presence.size(1) != 3:
        raise ValueError("presence must have shape [batch, 3].")
    result = torch.ones(presence.size(0), device=presence.device, dtype=torch.bool)
    for modality in required:
        result &= presence[:, MODALITIES.index(modality)].bool()
    return result


def validate_presence_binding(mode, model_mask):
    """Require exact equality between authoritative metadata and model input mask."""
    expected = presence_from_mode(
        mode,
        batch_size=model_mask.size(0),
        device=model_mask.device,
        dtype=model_mask.dtype,
    )
    mismatch = torch.count_nonzero(model_mask != expected).item()
    return {
        "Mode": mode,
        "BatchSize": int(model_mask.size(0)),
        "MaskShape": list(model_mask.shape),
        "MismatchCount": int(mismatch),
        "Passed": mismatch == 0,
        "Source": "explicit mode enum / sampled missing mask metadata",
        "InferredFromFeatureValues": False,
    }
