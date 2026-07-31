"""Function-space-safe wrapper around nested grouped OOF construction."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import grouped_oof_cfcompat_v92 as _base
from .function_space_features_v92 import (
    FEATURE_KEYS,
    FEATURE_SPACE_VERSION,
    predict_wrapper_function_space,
)

StageLimits = _base.StageLimits


def _audit_resume_caches(output_dir: Path):
    """Reject stale hidden-coordinate fold caches before resume."""
    for path in sorted(Path(output_dir).glob("outer_fold_*/outer_holdout_cache.pth")):
        payload = torch.load(path, map_location="cpu")
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        dimensions = {
            int(row["feature"].numel())
            for row in rows
            if isinstance(row, dict) and torch.is_tensor(row.get("feature"))
        }
        if rows and dimensions != {len(FEATURE_KEYS)}:
            raise RuntimeError(
                f"stale incompatible fold cache {path}: feature dimensions={dimensions}; "
                "remove the V9.2 output directory or rerun the builder with --no-resume"
            )


def run_nested_oof_cfcompat(*args, **kwargs):
    """Run the base OOF protocol with aligned task-space feature extraction."""
    output_dir = Path(kwargs.get("output_dir", args[2] if len(args) > 2 else "."))
    if kwargs.get("resume", True):
        _audit_resume_caches(output_dir)

    original = _base.predict_wrapper_lav
    _base.predict_wrapper_lav = predict_wrapper_function_space
    try:
        payload = _base.run_nested_oof_cfcompat(*args, **kwargs)
    finally:
        _base.predict_wrapper_lav = original

    if payload["oof_feature"].size(1) != len(FEATURE_KEYS):
        raise RuntimeError("OOF cache is not using the registered function-space schema")
    payload["feature_space"] = FEATURE_SPACE_VERSION
    payload["feature_keys"] = list(FEATURE_KEYS)
    cache_path = output_dir / "nested_grouped_oof_cfcompat_cache_v92.pth"
    torch.save(payload, cache_path)

    summary_path = output_dir / "nested_grouped_oof_cfcompat_summary_v92.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["feature_space"] = FEATURE_SPACE_VERSION
    summary["feature_keys"] = list(FEATURE_KEYS)
    summary["feature_dim"] = len(FEATURE_KEYS)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return payload
