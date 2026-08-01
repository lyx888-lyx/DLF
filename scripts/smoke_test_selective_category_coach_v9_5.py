"""CPU structural smoke test for V9.5."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from trains.singleTask.model.SelectiveCategoryCoachV95 import (
    SelectiveOrdinalCalibratorV95,
    boundary_distance,
    coach_input_features,
    cumulative_to_region_probs,
    region_index,
    selective_ordinal_loss,
    semantic_region_probabilities,
    specialist_center_progress,
)
from trains.singleTask.oof_group_splits_v92 import build_nested_group_folds


def main():
    sample_ids = [f"video_{i // 5}$_$segment_{i}" for i in range(75)]
    cycle = (-2.5, -1.0, 0.0, 1.0, 2.5)
    labels = torch.tensor([cycle[i % 5] for i in range(75)]).view(-1, 1)
    specs, manifest = build_nested_group_folds(
        sample_ids, labels.view(-1).tolist(), outer_folds=3,
        inner_valid_fraction=0.20, seed=9,
    )
    assert manifest.groupby("sample_index").size().eq(len(specs)).all()
    holdout = manifest.loc[manifest.partition == "outer_holdout"]
    assert holdout.groupby("sample_index").size().eq(1).all()

    torch.manual_seed(9)
    function_space = torch.randn(75, 4)
    anchor = torch.linspace(-2.8, 2.8, 75).view(-1, 1)
    features = coach_input_features(function_space, anchor)
    model = SelectiveOrdinalCalibratorV95(
        features.size(1), hidden_dim=16, dropout=0.0
    )
    output = model(features, anchor)
    assert output["region_probs"].shape == (75, 5)
    torch.testing.assert_close(
        output["region_probs"].sum(dim=1), torch.ones(75), atol=1e-6, rtol=1e-6
    )
    assert bool(
        (output["cumulative_probs"][:, :-1] >= output["cumulative_probs"][:, 1:]).all()
    )
    losses = selective_ordinal_loss(output, labels, anchor)
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    probabilities = semantic_region_probabilities(anchor, 0.45)
    assert probabilities.shape == (75, 5)
    assert region_index(labels).shape == (75,)
    assert boundary_distance(anchor).shape == (75,)
    assert specialist_center_progress(anchor, anchor * 0.5, 2).shape == (75,)
    assert cumulative_to_region_probs(output["cumulative_probs"]).shape == (75, 5)
    print("V9.5 SELECTIVE CATEGORY COACH SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
