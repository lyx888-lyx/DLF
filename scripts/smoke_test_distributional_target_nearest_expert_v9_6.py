"""CPU smoke test for V9.6 target-distribution and nearest-expert primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.DistributionalTargetCoachV96 import (  # noqa: E402
    DistributionalTargetCoachV96,
    action_risk,
    coach_input_features,
    distributional_target_loss,
    stack_action_predictions,
)
from trains.singleTask.oof_group_splits_v92 import (  # noqa: E402
    build_nested_group_folds,
)


def main():
    sample_ids = [
        f"video_{i // 5}$_$segment_{i}"
        for i in range(75)
    ]
    label_cycle = (-2.5, -1.0, 0.0, 1.0, 2.5)
    fold_labels = [label_cycle[i % 5] for i in range(75)]
    specs, manifest = build_nested_group_folds(
        sample_ids,
        fold_labels,
        outer_folds=3,
        inner_valid_fraction=0.20,
        seed=17,
    )
    assert len(specs) == 3
    assert manifest.groupby("sample_index").size().eq(
        len(specs)
    ).all()
    assert manifest.groupby(
        ["outer_fold", "sample_index"]
    ).size().eq(1).all()
    holdout = manifest.loc[
        manifest["partition"] == "outer_holdout"
    ]
    assert len(holdout) == len(sample_ids)
    assert holdout.groupby("sample_index").size().eq(1).all()
    for _, local in manifest.groupby("outer_fold"):
        assert local["sample_index"].nunique() == len(sample_ids)
        assert local.groupby("group_id")[
            "partition"
        ].nunique().eq(1).all()
    for spec in specs:
        assert not set(spec.outer_holdout_groups) & set(
            spec.inner_train_groups
        )
        assert not set(spec.outer_holdout_groups) & set(
            spec.inner_valid_groups
        )
        assert not set(spec.inner_train_groups) & set(
            spec.inner_valid_groups
        )

    torch.manual_seed(11)
    n = 75
    function_space = torch.randn(n, 4)
    anchor = torch.linspace(-2.8, 2.8, n).view(-1, 1)
    labels = anchor + 0.4 * torch.tanh(torch.randn_like(anchor))
    features = coach_input_features(function_space, anchor)
    model = DistributionalTargetCoachV96(
        features.size(1), hidden_dim=16, dropout=0.0
    )
    output = model(features, anchor)
    assert output["quantiles"].shape == (n, 5)
    assert torch.isfinite(output["quantiles"]).all()
    assert (
        output["quantiles"][:, 1:]
        >= output["quantiles"][:, :-1]
    ).all()
    assert (
        output["interval_width_80"]
        >= output["interval_width_50"]
    ).all()
    losses = distributional_target_loss(output, labels, anchor)
    losses["total"].backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    experts = anchor.unsqueeze(1) + 0.3 * torch.randn(n, 4, 1)
    actions = stack_action_predictions(anchor, experts)
    for mode in ("median_distance", "quantile_risk"):
        risks = action_risk(actions, output, mode)
        assert risks.shape == (n, 5)
        assert torch.isfinite(risks).all()

    print(
        "V9.6 DISTRIBUTIONAL TARGET NEAREST-EXPERT SMOKE TEST PASSED"
    )


if __name__ == "__main__":
    main()
