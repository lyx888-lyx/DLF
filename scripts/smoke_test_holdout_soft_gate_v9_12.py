"""CPU smoke test for V9.12 convex soft-gating primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.holdout_soft_gate_system_v912 import (  # noqa: E402
    holdout_split_tensors,
)
from trains.singleTask.model.HoldoutSoftGateV912 import (  # noqa: E402
    GATE_VERSION,
    HoldoutSoftGateV912,
    holdout_soft_gate_loss,
    soft_gate_prediction,
)
from trains.singleTask.model.SemanticCostCoachV99 import (  # noqa: E402
    SIGNATURE_DIM,
    SPECIALIST_NAMES,
)


def synthetic_pool(n: int):
    torch.manual_seed(912)
    anchor = torch.linspace(-1.5, 1.5, n).view(-1, 1)
    function_space = torch.randn(n, 4)
    experts = {}
    for index, name in enumerate(SPECIALIST_NAMES):
        correction = 0.10 * (index + 1) * torch.randn(n, 1)
        prediction = anchor + correction
        signature = torch.randn(n, SIGNATURE_DIM)
        signature[:, 0] = prediction.view(-1)
        signature[:, 1] = correction.view(-1)
        signature[:, 2] = correction.abs().view(-1)
        experts[name] = {
            "prediction": prediction,
            "signature": signature,
        }
    labels = anchor + 0.15 * torch.randn_like(anchor)
    return {
        "anchor": anchor,
        "function_space": function_space,
        "labels": labels,
        "experts": experts,
    }


def main():
    assert GATE_VERSION == "holdout_soft_gate_v1"
    n = 32
    pool = synthetic_pool(n)
    values = holdout_split_tensors(pool)
    assert values["actions"].shape == (n, 5, 1)
    assert values["signatures"].shape == (n, 4, SIGNATURE_DIM)
    assert values["context"].shape[0] == n

    model = HoldoutSoftGateV912(
        context_dim=values["context"].size(1),
        hidden_dim=24,
        action_embedding_dim=6,
        dropout=0.0,
    )
    output = model(values["context"], values["signatures"])
    assert output["weights"].shape == (n, 5)
    assert torch.allclose(
        output["weights"].sum(dim=1),
        torch.ones(n),
        atol=1e-6,
    )
    assert bool((output["weights"] >= 0.0).all())

    routed = soft_gate_prediction(
        output,
        values["actions"],
        beta=1.0,
    )
    anchor_only = soft_gate_prediction(
        output,
        values["actions"],
        beta=0.0,
    )
    assert routed["prediction"].shape == (n, 1)
    assert torch.allclose(
        anchor_only["prediction"],
        values["actions"][:, 0],
        atol=1e-7,
    )

    losses = holdout_soft_gate_loss(
        output,
        values["actions"],
        values["labels"],
    )
    losses["total"].backward()
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(parameter.grad is not None for parameter in trainable)
    assert torch.isfinite(losses["total"])

    with torch.no_grad():
        manual = {
            "weights": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ]
            )
        }
        manual_actions = values["actions"][:2]
        manual_result = soft_gate_prediction(
            manual,
            manual_actions,
            beta=1.0,
        )["prediction"]
        assert torch.allclose(
            manual_result[0],
            manual_actions[0, 0],
        )
        assert torch.allclose(
            manual_result[1],
            manual_actions[1, 2],
        )

    print("V9.12 HOLDOUT SOFT GATE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
