"""CPU structural smoke test for the V9.7 attainable-frontier coach."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.attainable_frontier_coach_system_v97 import (  # noqa: E402
    AttainableFrontierTrainerV97,
    POLICY_PROFILES,
)
from trains.singleTask.model.AttainableFrontierCoachV97 import (  # noqa: E402
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


def main():
    torch.manual_seed(97)
    n = 40
    function_space = torch.randn(n, 4)
    anchor = torch.linspace(-2.5, 2.5, n).view(-1, 1)
    expert_predictions = (
        anchor.view(n, 1, 1) + 0.30 * torch.randn(n, 4, 1)
    )
    confidences = torch.sigmoid(torch.randn(n, 4, 1))
    labels = anchor + 0.25 * torch.randn(n, 1)
    actions = stack_action_predictions(anchor, expert_predictions)
    features = frontier_input_features(
        function_space, actions, confidences
    )
    model = AttainableFrontierCoachV97(
        features.size(1), hidden_dim=24, dropout=0.0
    )
    output = model(features, actions)
    targets = attainable_frontier_targets(actions, labels)
    assert output["frontier_value"].shape == (n, 1)
    assert output["predicted_gain"].shape == (n, 1)
    assert output["action_logits"].shape == (n, 5)
    assert torch.isfinite(output["frontier_value"]).all()
    low = actions.squeeze(-1).amin(dim=1)
    high = actions.squeeze(-1).amax(dim=1)
    assert bool(
        (output["frontier_value"].view(-1) >= low - 1e-6).all()
    )
    assert bool(
        (output["frontier_value"].view(-1) <= high + 1e-6).all()
    )
    assert bool((targets["oracle_gain"] >= -1e-7).all())
    losses = attainable_frontier_loss(output, actions, labels)
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    nearest = nearest_action_from_frontier(output, actions)
    assert nearest["selected_index"].shape == (n,)
    assert nearest["nearest_margin"].shape == (n, 1)
    route = AttainableFrontierTrainerV97._apply_profile(
        actions,
        confidences,
        output,
        POLICY_PROFILES["balanced"],
    )
    assert route["prediction"].shape == (n, 1)
    assert route["deployed_action"].shape == (n,)
    assert bool((route["deployed_action"] >= 0).all())
    assert bool((route["deployed_action"] < 5).all())

    sample_ids = [f"video_{i // 4}$_$segment_{i}" for i in range(60)]
    fold_labels = [float((i % 7) - 3) for i in range(60)]
    specs, manifest = build_nested_group_folds(
        sample_ids,
        fold_labels,
        outer_folds=3,
        inner_valid_fraction=0.20,
        seed=97,
    )
    assert len(specs) == 3
    assert manifest.groupby(
        ["outer_fold", "sample_index"]
    ).size().eq(1).all()
    holdout = manifest[manifest["partition"] == "outer_holdout"]
    assert holdout.groupby("sample_index").size().eq(1).all()
    for _, local in manifest.groupby("outer_fold"):
        assert local.groupby("group_id")[
            "partition"
        ].nunique().eq(1).all()

    print("V9.7 ATTAINABLE FRONTIER COACH SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
