"""CPU structural smoke test for V9.8 strict nested frontier primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.AttainableFrontierCoachV97 import (  # noqa: E402
    ACTION_NAMES,
    AttainableFrontierCoachV97,
    attainable_frontier_loss,
    attainable_frontier_targets,
    frontier_input_features,
    nearest_action_from_frontier,
    stack_action_predictions,
)
from trains.singleTask.oof_group_splits_v92 import (  # noqa: E402
    build_nested_group_folds,
)
from trains.singleTask.strict_nested_frontier_pool_v98 import (  # noqa: E402
    POOL_VERSION,
    TEACHER_NAMES,
)


def main():
    assert POOL_VERSION == "v98_strict_nested_v93_architecture_frontier_pool"
    assert TEACHER_NAMES == ("clean", "moddrop", "cfcompat")

    sample_ids = [f"video_{i // 4}$_$segment_{i}" for i in range(120)]
    label_cycle = (-2.5, -1.0, 0.0, 1.0, 2.5)
    labels = [label_cycle[i % len(label_cycle)] for i in range(len(sample_ids))]
    specs, manifest = build_nested_group_folds(
        sample_ids,
        labels,
        outer_folds=3,
        inner_valid_fraction=0.20,
        seed=19,
    )
    assert len(specs) == 3
    holdout_groups = []
    for spec in specs:
        assert not set(spec.inner_train_groups) & set(spec.inner_valid_groups)
        assert not set(spec.inner_train_groups) & set(spec.outer_holdout_groups)
        assert not set(spec.inner_valid_groups) & set(spec.outer_holdout_groups)
        holdout_groups.extend(spec.outer_holdout_groups)
    assert len(holdout_groups) == len(set(holdout_groups))
    assert manifest.loc[
        manifest["partition"] == "outer_holdout"
    ].groupby("sample_index").size().eq(1).all()

    torch.manual_seed(19)
    n = 32
    function_space = torch.randn(n, 4)
    anchor = torch.linspace(-2.0, 2.0, n).view(-1, 1)
    experts = anchor.unsqueeze(1) + 0.35 * torch.randn(n, 4, 1)
    confidences = torch.sigmoid(torch.randn(n, 4, 1))
    actions = stack_action_predictions(anchor, experts)
    targets = attainable_frontier_targets(
        actions,
        anchor + 0.25 * torch.randn_like(anchor),
    )
    assert targets["oracle_value"].shape == (n, 1)
    assert targets["oracle_index"].shape == (n,)
    assert set(targets["oracle_index"].tolist()).issubset(
        set(range(len(ACTION_NAMES)))
    )

    features = frontier_input_features(function_space, actions, confidences)
    model = AttainableFrontierCoachV97(
        input_dim=features.size(1),
        hidden_dim=24,
        dropout=0.0,
    )
    output = model(features, actions)
    losses = attainable_frontier_loss(
        output,
        actions,
        anchor + 0.25 * torch.randn_like(anchor),
    )
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    nearest = nearest_action_from_frontier(output, actions)
    assert nearest["selected_index"].shape == (n,)
    assert nearest["selected_value"].shape == (n, 1)
    assert torch.isfinite(nearest["selected_value"]).all()

    print("V9.8 STRICT NESTED FRONTIER COACH SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
