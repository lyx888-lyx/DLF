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
    torch.manual_seed(11)
    n = 24
    function_space = torch.randn(n, 4)
    anchor = torch.randn(n, 1)
    labels = anchor + 0.4 * torch.randn(n, 1)
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
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    experts = torch.randn(n, 4, 1)
    actions = stack_action_predictions(anchor, experts)
    for mode in ("median_distance", "quantile_risk"):
        risks = action_risk(actions, output, mode)
        assert risks.shape == (n, 5)
        assert torch.isfinite(risks).all()

    sample_ids = [f"video{i // 3}/clip{i}" for i in range(30)]
    fold_labels = [float((i % 7) - 3) for i in range(30)]
    specs, manifest = build_nested_group_folds(
        sample_ids,
        fold_labels,
        outer_folds=3,
        inner_valid_fraction=0.25,
        seed=17,
    )
    assert len(specs) == 3
    assert manifest.groupby(
        ["outer_fold", "sample_index"]
    ).size().eq(1).all()
    holdout = manifest.loc[
        manifest["partition"] == "outer_holdout"
    ]
    assert holdout.groupby("sample_index").size().eq(1).all()
    for _, local in manifest.groupby("outer_fold"):
        assert local.groupby("group_id")[
            "partition"
        ].nunique().eq(1).all()
    print(
        "V9.6 DISTRIBUTIONAL TARGET NEAREST-EXPERT SMOKE TEST PASSED"
    )


if __name__ == "__main__":
    main()
