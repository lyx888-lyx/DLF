#!/usr/bin/env python3
"""Freeze the core-A1 PCA coordinate schema before any outer evaluation."""

from __future__ import annotations

import json

from stage23d_self_risk_common import OUT, atomic_json, sha256_file, utc_now


def main():
    path = OUT / "protocol" / "pca_input_schema.json"
    if path.exists():
        print(path.read_text())
        return
    schema = {
        "stage": "Stage23D-A train-only PCA input schema",
        "frozen_before_any_phase1_outer_evaluation": True,
        "raw_artifact_layout": (
            "final_fused, shared_fused, replayable A2 fusion/LFA blocks, "
            "then one specific+aligned-shared pair per active modality"
        ),
        "pca_core_layout": (
            "first 450 coordinates (final_fused=300, shared_fused=150) "
            "plus final 100 coordinates per active modality "
            "(specific=50, aligned_shared=50)"
        ),
        "pca_core_dimensions": {"LAV": 750, "LA": 650, "LV": 650, "L": 550},
        "a2_policy": (
            "fusion/LFA raw tensors remain SHA-bound and replayable but do not "
            "enter PCA coordinates; their scalar summaries enter R2 directly"
        ),
        "reason": (
            "avoid padded time-coordinate and sequence-length shortcuts while "
            "retaining all head-driving A1 representations"
        ),
        "pca_fit_role": "self-risk inner_train only",
        "nested_candidate_dimensions": [16, 32, 64],
        "selection_role": "self-risk inner_valid only",
        "outer_fit_or_selection": False,
        "created_at": utc_now(),
    }
    atomic_json(path, schema)
    print(json.dumps({"path": str(path), "sha256": sha256_file(path)}, indent=2))


if __name__ == "__main__":
    main()
