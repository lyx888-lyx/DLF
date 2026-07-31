"""CPU structural smoke test for the V9.4 cost-sensitive OOF coach."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from trains.singleTask.model.CostSensitiveCoachV94 import (
    ACTION_NAMES,
    SPECIALIST_NAMES,
    CachedSpecialistResidualHeadV94,
    CostSensitiveCoachV94,
    coach_input_features,
    cost_sensitive_coach_loss,
    specialist_loss,
    stack_action_predictions,
)
from trains.singleTask.oof_group_splits_v92 import build_nested_group_folds


def main():
    sample_ids = [f"video_{i // 5}$_$segment_{i}" for i in range(75)]
    label_cycle = (-2.5, -1.0, 0.0, 1.0, 2.5)
    labels = [label_cycle[i % 5] for i in range(75)]
    specs, manifest = build_nested_group_folds(
        sample_ids,
        labels,
        outer_folds=3,
        inner_valid_fraction=0.20,
        seed=17,
    )
    assert len(specs) == 3
    assert manifest.groupby("sample_index").size().eq(len(specs)).all()
    assert manifest.groupby(["outer_fold", "sample_index"]).size().eq(1).all()
    holdout = manifest.loc[manifest.partition == "outer_holdout"]
    assert holdout.groupby("sample_index").size().eq(1).all()
    for _, local in manifest.groupby("outer_fold"):
        assert local.groupby("group_id").partition.nunique().eq(1).all()

    torch.manual_seed(17)
    n = 48
    function_space = torch.randn(n, 4)
    anchor = torch.linspace(-2.5, 2.5, n).view(-1, 1)
    target = anchor + 0.4 * torch.tanh(torch.randn_like(anchor))
    predictions = []
    confidences = []
    for role in SPECIALIST_NAMES:
        specialist = CachedSpecialistResidualHeadV94(
            feature_dim=4, hidden_dim=16, dropout=0.0
        )
        output = specialist(function_space, anchor)
        loss = specialist_loss(output, target, role)
        loss["total"].backward()
        assert any(parameter.grad is not None for parameter in specialist.parameters())
        predictions.append(output["prediction"].detach())
        confidences.append(output["applicability_prob"].detach())
    expert_predictions = torch.stack(predictions, dim=1)
    expert_confidences = torch.stack(confidences, dim=1)
    features = coach_input_features(
        function_space, anchor, expert_predictions, expert_confidences
    )
    actions = stack_action_predictions(anchor, expert_predictions)
    assert actions.shape == (n, len(ACTION_NAMES), 1)

    coach = CostSensitiveCoachV94(features.size(1), hidden_dim=20, dropout=0.0)
    output = coach(features, anchor)
    assert output["action_probs"].shape == (n, len(ACTION_NAMES))
    assert output["region_probs"].shape == (n, 5)
    torch.testing.assert_close(
        output["action_probs"].sum(dim=1), torch.ones(n), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        output["region_probs"].sum(dim=1), torch.ones(n), atol=1e-6, rtol=1e-6
    )
    cumulative = output["cumulative_probs"]
    assert bool((cumulative[:, :-1] >= cumulative[:, 1:]).all())
    losses = cost_sensitive_coach_loss(output, actions, target)
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in coach.parameters())
    assert torch.isfinite(losses["total"])

    print("V9.4 COST-SENSITIVE OOF COACH SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
