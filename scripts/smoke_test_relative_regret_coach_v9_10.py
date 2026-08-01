"""CPU structural smoke test for V9.10 relative-regret primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.relative_regret_runtime_patch_v910 import (  # noqa: E402
    install_relative_regret_runtime_patch,
)

install_relative_regret_runtime_patch()

from trains.singleTask.model.RelativeRegretCoachV910 import (  # noqa: E402
    REGRET_VERSION,
    RelativeRegretCoachV910,
    regret_soft_mixture,
    relative_regret_loss,
    relative_regret_targets,
    select_action_from_regret,
)
from trains.singleTask.model.SemanticCostCoachV99 import (  # noqa: E402
    SIGNATURE_DIM,
    SPECIALIST_NAMES,
    global_context_features,
    stack_action_predictions,
)
from trains.singleTask.relative_regret_crossfit_v910 import (  # noqa: E402
    pool_tensors,
)


def main():
    assert REGRET_VERSION == "relative_regret_v1"
    torch.manual_seed(23)
    n = 24
    anchor = torch.linspace(-1.5, 1.5, n).view(-1, 1)
    experts = anchor.unsqueeze(1) + 0.25 * torch.randn(n, 4, 1)
    actions = stack_action_predictions(anchor, experts)
    labels = anchor + 0.20 * torch.randn_like(anchor)
    targets = relative_regret_targets(actions, labels)
    manual = (
        torch.abs(actions[:, 1:, 0] - labels)
        - torch.abs(anchor - labels)
    )
    assert torch.allclose(targets["delta"], manual)
    assert targets["oracle_index"].shape == (n,)

    function_space = torch.randn(n, 4)
    expert_signatures = torch.randn(n, 4, SIGNATURE_DIM)

    # The strict Train OOF pool stores dense tensors.
    strict_pool = {
        "anchor": anchor,
        "function_space": function_space,
        "expert_predictions": experts,
        "expert_signatures": expert_signatures,
    }
    strict_context, strict_actions, strict_signatures = pool_tensors(strict_pool)

    # Validation/Test pools store the same values under experts[name].
    frozen_pool = {
        "anchor": anchor,
        "function_space": function_space,
        "experts": {
            name: {
                "prediction": experts[:, index],
                "signature": expert_signatures[:, index],
            }
            for index, name in enumerate(SPECIALIST_NAMES)
        },
    }
    frozen_context, frozen_actions, frozen_signatures = pool_tensors(frozen_pool)

    assert torch.allclose(strict_context, frozen_context)
    assert torch.allclose(strict_actions, frozen_actions)
    assert torch.allclose(strict_signatures, frozen_signatures)
    assert strict_actions.shape == (n, 5, 1)
    assert strict_signatures.shape == (n, 4, SIGNATURE_DIM)

    context = global_context_features(function_space, actions)
    signatures = expert_signatures
    model = RelativeRegretCoachV910(
        context_dim=context.size(1),
        hidden_dim=24,
        action_embedding_dim=6,
        dropout=0.0,
    )
    output = model(context, signatures)
    assert output["predicted_delta"].shape == (n, 4)
    assert output["predicted_scale"].shape == (n, 4)
    assert output["beat_probability"].shape == (n, 4)
    losses = relative_regret_loss(output, actions, labels)
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    manual_output = {
        "predicted_delta": torch.tensor(
            [[0.10, -0.20, 0.05, 0.25], [0.08, 0.03, 0.06, 0.11]]
        ),
        "predicted_scale": torch.full((2, 4), 0.02),
        "beat_probability": torch.tensor(
            [[0.20, 0.90, 0.30, 0.10], [0.20, 0.30, 0.25, 0.10]]
        ),
    }
    manual_actions = actions[:2]
    selected = select_action_from_regret(
        manual_output,
        manual_actions,
        risk_aversion=0.0,
        confidence_z=1.0,
    )
    assert selected["selected_index"].tolist() == [2, 0]
    assert selected["predicted_gain"][0].item() > 0.0
    assert selected["lower_gain"][0].item() > 0.0
    assert selected["predicted_gain"][1].item() == 0.0

    mixture = regret_soft_mixture(
        output,
        actions,
        temperature=0.08,
        risk_aversion=0.25,
    )
    assert mixture["prediction"].shape == (n, 1)
    assert torch.allclose(
        mixture["weights"].sum(dim=1),
        torch.ones(n),
        atol=1e-5,
    )
    print("V9.10 RELATIVE REGRET COACH SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
