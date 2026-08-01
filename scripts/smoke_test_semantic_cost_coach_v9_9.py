"""CPU structural smoke test for V9.9 semantic signatures and cost coaching."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.SemanticCostCoachV99 import (  # noqa: E402
    ACTION_NAMES,
    SIGNATURE_DIM,
    SIGNATURE_FIELDS,
    SemanticCostCoachV99,
    cost_soft_mixture,
    global_context_features,
    role_signature,
    select_action_from_cost,
    semantic_cost_loss,
    stack_action_predictions,
    stack_action_signatures,
    tail_signature,
)
from trains.singleTask.oof_group_splits_v92 import (  # noqa: E402
    build_nested_group_folds,
)
from trains.singleTask.strict_semantic_expert_pool_v99 import (  # noqa: E402
    POOL_VERSION,
)


def main():
    assert POOL_VERSION == "v99_strict_nested_semantic_expert_pool"
    assert SIGNATURE_DIM == len(SIGNATURE_FIELDS) == 21

    sample_ids = [f"video_{i // 4}$_$segment_{i}" for i in range(120)]
    label_cycle = (-2.5, -1.0, 0.0, 1.0, 2.5)
    labels = [label_cycle[i % len(label_cycle)] for i in range(len(sample_ids))]
    specs, manifest = build_nested_group_folds(
        sample_ids,
        labels,
        outer_folds=3,
        inner_valid_fraction=0.20,
        seed=29,
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

    torch.manual_seed(29)
    n = 40
    function_space = torch.randn(n, 4)
    anchor = torch.linspace(-2.0, 2.0, n).view(-1, 1)
    expert_predictions = anchor.unsqueeze(1) + 0.35 * torch.randn(n, 4, 1)
    actions = stack_action_predictions(anchor, expert_predictions)

    role_probs_a = torch.softmax(torch.randn(n, 5), dim=1)
    role_probs_b = torch.softmax(torch.randn(n, 5), dim=1)
    mechanism_a = torch.softmax(torch.randn(n, 4), dim=1)
    mechanism_b = torch.softmax(torch.randn(n, 4), dim=1)
    role_a = role_signature(
        expert_predictions[:, 1],
        expert_predictions[:, 1] - anchor,
        role_probs_a[:, 2:3],
        role_probs_a,
        torch.randn(n, 1),
        torch.rand(n, 1),
    )
    role_b = role_signature(
        expert_predictions[:, 2],
        expert_predictions[:, 2] - anchor,
        role_probs_b[:, 3:4],
        role_probs_b,
        torch.randn(n, 1),
        torch.rand(n, 1),
    )
    tail_a = tail_signature(
        expert_predictions[:, 0],
        expert_predictions[:, 0] - anchor,
        1.2 * (expert_predictions[:, 0] - anchor),
        torch.rand(n, 1),
        mechanism_a,
    )
    tail_b = tail_signature(
        expert_predictions[:, 3],
        expert_predictions[:, 3] - anchor,
        1.2 * (expert_predictions[:, 3] - anchor),
        torch.rand(n, 1),
        mechanism_b,
    )
    experts = torch.stack((tail_a, role_a, role_b, tail_b), dim=1)
    signatures = stack_action_signatures(anchor, experts)
    assert signatures.shape == (n, 5, SIGNATURE_DIM)
    assert torch.allclose(signatures[:, 0, 0], anchor.view(-1))
    assert signatures[:, 2, 9].eq(1).all()
    assert signatures[:, 1, 10].eq(1).all()

    context = global_context_features(function_space, actions)
    model = SemanticCostCoachV99(
        context_dim=context.size(1), hidden_dim=32, dropout=0.0
    )
    output = model(context, signatures)
    target = anchor + 0.30 * torch.randn_like(anchor)
    losses = semantic_cost_loss(output, actions, target)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    selected = select_action_from_cost(output, actions, risk_aversion=0.5)
    assert selected["selected_index"].shape == (n,)
    assert selected["predicted_gain"].shape == (n, 1)
    assert selected["cost_margin"].shape == (n, 1)
    soft = cost_soft_mixture(output, actions)
    assert soft["prediction"].shape == (n, 1)
    low = actions.squeeze(-1).amin(dim=1, keepdim=True)
    high = actions.squeeze(-1).amax(dim=1, keepdim=True)
    assert torch.all(soft["prediction"] >= low - 1e-6)
    assert torch.all(soft["prediction"] <= high + 1e-6)

    print("V9.9 SEMANTIC COST COACH SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
