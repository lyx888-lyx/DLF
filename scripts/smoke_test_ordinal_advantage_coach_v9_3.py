"""CPU-only structural smoke test for the V9.3 ordinal advantage coach."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from trains.singleTask.model.OrdinalAdvantageCoachV93 import (
    SPECIALIST_TO_REGION,
    AdvantageHeadV93,
    OrdinalRegionCoachV93,
    advantage_input_features,
    advantage_loss,
    coach_input_features,
    ordinal_region_loss,
)
from trains.singleTask.oof_group_splits_v92 import build_nested_group_folds


def main():
    sample_ids = [f"video_{i // 5}$_$segment_{i}" for i in range(75)]
    label_cycle = (-2.5, -1.0, 0.0, 1.0, 2.5)
    labels = [label_cycle[i % 5] for i in range(75)]
    specs, manifest = build_nested_group_folds(
        sample_ids, labels, outer_folds=3, inner_valid_fraction=0.20, seed=7
    )
    assert len(specs) == 3

    # The manifest records every sample once for every outer fold, because its
    # partition changes from inner_train/inner_valid to outer_holdout across
    # folds. Therefore each sample must occur len(specs) times overall, exactly
    # once per outer fold, and exactly once as an outer holdout.
    assert manifest.groupby("sample_index").size().eq(len(specs)).all()
    assert manifest.groupby(["outer_fold", "sample_index"]).size().eq(1).all()
    holdout_counts = (
        manifest.loc[manifest["partition"] == "outer_holdout"]
        .groupby("sample_index")
        .size()
    )
    assert len(holdout_counts) == len(sample_ids)
    assert holdout_counts.eq(1).all()

    for outer_fold, local in manifest.groupby("outer_fold"):
        del outer_fold
        assert local["sample_index"].nunique() == len(sample_ids)
        assert local.groupby("group_id")["partition"].nunique().eq(1).all()

    for spec in specs:
        assert not set(spec.outer_holdout_groups) & set(spec.inner_train_groups)
        assert not set(spec.outer_holdout_groups) & set(spec.inner_valid_groups)
        assert not set(spec.inner_train_groups) & set(spec.inner_valid_groups)

    torch.manual_seed(7)
    function_space = torch.randn(75, 4)
    anchor = torch.linspace(-2.8, 2.8, 75).view(-1, 1)
    target = anchor + 0.4 * torch.tanh(torch.randn_like(anchor))
    features = coach_input_features(function_space, anchor)
    region_model = OrdinalRegionCoachV93(features.size(1), hidden_dim=16, dropout=0.0)
    region_output = region_model(features, anchor)
    assert region_output["region_probs"].shape == (75, 5)
    torch.testing.assert_close(
        region_output["region_probs"].sum(dim=1), torch.ones(75), atol=1e-6, rtol=1e-6
    )
    cumulative = region_output["cumulative_probs"]
    assert bool((cumulative[:, :-1] >= cumulative[:, 1:]).all())
    region_losses = ordinal_region_loss(region_output, target)
    region_losses["total"].backward()
    assert any(parameter.grad is not None for parameter in region_model.parameters())

    expert_prediction = anchor + 0.15 * torch.tanh(torch.randn_like(anchor))
    confidence = torch.sigmoid(torch.randn_like(anchor))
    advantage_features = advantage_input_features(
        features,
        region_output["region_probs"].detach(),
        region_output["score"].detach(),
        anchor,
        expert_prediction,
        confidence,
        SPECIALIST_TO_REGION["strong_positive"],
    )
    advantage_model = AdvantageHeadV93(advantage_features.size(1), hidden_dim=12, dropout=0.0)
    advantage_output = advantage_model(advantage_features)
    realized_gain = torch.abs(anchor - target) - torch.abs(expert_prediction - target)
    losses = advantage_loss(advantage_output, realized_gain)
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in advantage_model.parameters())
    assert torch.isfinite(advantage_output["predicted_gain"]).all()
    assert bool(
        (
            (advantage_output["win_probability"] >= 0)
            & (advantage_output["win_probability"] <= 1)
        ).all()
    )

    print("V9.3 ORDINAL ADVANTAGE COACH SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
