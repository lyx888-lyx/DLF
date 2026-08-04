"""Synthetic contract checks for the locked CFCompatKD + SAM screen."""
from __future__ import annotations

import torch
import torch.nn as nn

from trains.singleTask.cfcompat_sam_utils import (
    FORMAL_SEEDS,
    SAMController,
    aggregate_valid_gate,
    capture_rng_state,
    restore_rng_state,
    rng_states_equal,
    select_rho,
)


def sam_rng_test():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sam = SAMController(model.parameters(), rho=0.05)
    inputs = [torch.randn(3, 4), torch.randn(2, 4)]
    targets = [torch.randn(3, 1), torch.randn(2, 1)]
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad()
    rng_before = capture_rng_state()
    first_outputs = []
    for x, y in zip(inputs, targets):
        prediction = model(x)
        first_outputs.append(prediction.detach().clone())
        torch.abs(prediction - y).mean().backward()
    rng_after_first = capture_rng_state()
    first_norm = sam.ascent_step()
    optimizer.zero_grad()

    restore_rng_state(rng_before)
    second_outputs = []
    for x, y in zip(inputs, targets):
        prediction = model(x)
        second_outputs.append(prediction.detach().clone())
        torch.abs(prediction - y).mean().backward()
    rng_after_second = capture_rng_state()
    assert rng_states_equal(rng_after_first, rng_after_second)
    assert first_norm > 0
    assert float(sam.grad_norm()) > 0
    # Outputs differ because parameters are perturbed; dropout masks are replayed.
    assert any(not torch.allclose(a, b) for a, b in zip(first_outputs, second_outputs))
    sam.descent_step(optimizer)
    optimizer.zero_grad()
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(before_parameters, model.parameters())
    )


def gate_test():
    baselines = {
        1111: {
            "J_valid": 0.680,
            "valid_LAV_MAE": 0.680,
            "valid_LA_MAE": 0.680,
            "valid_LV_MAE": 0.680,
            "valid_L_MAE": 0.680,
        },
        1114: {
            "J_valid": 0.670,
            "valid_LAV_MAE": 0.670,
            "valid_LA_MAE": 0.670,
            "valid_LV_MAE": 0.670,
            "valid_L_MAE": 0.670,
        },
    }
    candidates = []
    epochs = []
    for seed in FORMAL_SEEDS:
        base = baselines[seed]["J_valid"]
        for rho, gain in ((0.01, 0.006), (0.05, 0.003)):
            value = base - gain
            candidates.append({
                "Seed": seed,
                "Rho": rho,
                "J_valid": value,
                "valid_LAV_MAE": value,
                "valid_LA_MAE": value,
                "valid_LV_MAE": value,
                "valid_L_MAE": value,
            })
            for epoch in range(3):
                epochs.append({
                    "Seed": seed,
                    "Rho": rho,
                    "J_valid": value + (0.0001 * epoch),
                })
    selected = select_rho(candidates)
    assert selected == 0.01
    gate = aggregate_valid_gate(selected, candidates, baselines, epochs)
    assert gate["passed"]
    assert gate["next_required_stage"] == "train_only_group_stability_screen"


if __name__ == "__main__":
    sam_rng_test()
    gate_test()
    print("CFCompatKD + SAM synthetic smoke test passed")
