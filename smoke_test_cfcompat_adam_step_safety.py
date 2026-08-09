"""Synthetic invariants for CFCompatKD v13 actual-Adam-step safety."""
from __future__ import annotations

import torch

from trains.singleTask.cfcompat_adam_step_safety_utils import (
    clone_tensor_tuple,
    direct_v12_metric_improvement_gate,
    flatten_tensor_tuple,
    project_two_halfspaces,
    unflatten_like,
)


def transfer(all_ntr, beneficial_ntr, nonbeneficial_ntr):
    return {
        "all_missing": {"negative_transfer_rate": float(all_ntr)},
        "teacher_beneficial": {"negative_transfer_rate": float(beneficial_ntr)},
        "teacher_nonbeneficial": {"negative_transfer_rate": float(nonbeneficial_ntr)},
    }


def main():
    # Orthogonal violated halfspaces: closest feasible point is the origin.
    delta = torch.tensor([1.0, 2.0], dtype=torch.float64)
    g1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    projected, diagnostic = project_two_halfspaces(delta, g1, g2)
    assert diagnostic["projected"]
    assert torch.allclose(projected, torch.zeros_like(projected), atol=1e-12, rtol=0.0)
    assert diagnostic["first_dot_after"] <= 1e-12
    assert diagnostic["second_dot_after"] <= 1e-12

    # Already-safe displacement must be bitwise unchanged in fp64 value space.
    safe = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    unchanged, diagnostic = project_two_halfspaces(safe, g1, g2)
    assert not diagnostic["projected"]
    assert torch.equal(unchanged, safe)

    # Collinear constraints must remain numerically stable.
    collinear, diagnostic = project_two_halfspaces(
        torch.tensor([2.0, -1.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([2.0, 0.0], dtype=torch.float64),
    )
    assert float(collinear[0]) <= 1e-12
    assert diagnostic["first_dot_after"] <= 1e-12
    assert diagnostic["second_dot_after"] <= 1e-12

    references = (
        torch.zeros(2, 2, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
    )
    values = (
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        torch.arange(3, dtype=torch.float32),
    )
    clone = clone_tensor_tuple(values)
    vector = flatten_tensor_tuple(clone)
    rebuilt = unflatten_like(vector, references)
    assert all(torch.equal(a, b) for a, b in zip(values, rebuilt))

    v12 = transfer(0.45, 0.31, 0.56)
    improved = transfer(0.40, 0.20, 0.58)
    gate = direct_v12_metric_improvement_gate(0.66, 0.67, improved, v12)
    assert gate["passed"]
    failed = transfer(0.40, 0.32, 0.58)
    gate = direct_v12_metric_improvement_gate(0.66, 0.67, failed, v12)
    assert not gate["passed"]

    print("v13 actual-Adam-step safety smoke passed")


if __name__ == "__main__":
    main()
