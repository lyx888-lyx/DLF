#!/usr/bin/env python3
"""CPU-only utility smoke test for V9.4."""

from __future__ import annotations

import sys
from pathlib import Path

# Direct execution puts ``scripts/`` on sys.path rather than the repository
# root. Add the root explicitly so project packages such as ``trains`` can be
# imported reliably from the runner and from manual invocations.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from trains.singleTask.oof_positive_residual_system_v94 import (
    _average_precision,
    _grouped_stratified_folds,
    _robust_ridge_fit,
    _robust_ridge_predict,
)


def main():
    sample_ids = [
        f"video_{group}$_$segment_{segment}"
        for group in range(10)
        for segment in range(2)
    ]
    labels = torch.tensor(
        [-2.0, -1.5, -0.7, -0.2, 0.0, 0.0, 0.2, 0.8, 1.5, 2.2] * 2
    ).view(-1, 1)
    folds, group_keys = _grouped_stratified_folds(
        sample_ids,
        labels,
        fold_count=5,
        strong_threshold=1.0,
        neutral_radius=1e-6,
    )
    if sorted(folds.unique().tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Incomplete fold coverage: {folds.tolist()}")
    group_fold = {}
    for key, fold in zip(group_keys, folds.tolist()):
        previous = group_fold.setdefault(key, fold)
        if previous != fold:
            raise RuntimeError(f"Group crossed folds: {key}")

    torch.manual_seed(94)
    features = torch.randn(40, 6)
    target = (
        0.15
        + 0.30 * features[:, :1]
        - 0.10 * features[:, 1:2]
        + 0.02 * torch.randn(40, 1)
    )
    model = _robust_ridge_fit(features[:30], target[:30], l2=1.0)
    prediction = _robust_ridge_predict(model, features[30:])
    if prediction.shape != (10, 1) or not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Robust ridge smoke test failed.")

    ap = _average_precision(
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        torch.tensor([0.1, 0.9, 0.2, 0.8]),
    )
    if abs(ap - 1.0) > 1e-6:
        raise RuntimeError(f"Average precision smoke test failed: {ap}")
    print("V9.4 utility smoke test passed")


if __name__ == "__main__":
    main()
