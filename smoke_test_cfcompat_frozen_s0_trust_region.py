"""Smoke tests for the frozen-S0 trust-region v6 mechanism utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trains.singleTask.cfcompat_s0_trust_region_utils import (
    DEV_SEED,
    RUN,
    S0_PROTECT_MARGIN,
    development_signal_gate,
    mechanism_transfer_summary,
    s0_trust_decision,
)


def test_s0_one_sided_trust():
    current = torch.tensor([[0.0], [0.9], [0.0], [0.0]], dtype=torch.float32)
    s0 = torch.tensor([[0.6], [0.6], [0.05], [1.5]], dtype=torch.float32)
    baseline = torch.tensor([[0.1], [0.1], [0.1], [0.1]], dtype=torch.float32)
    labels = torch.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.float32)
    d = s0_trust_decision(current, s0, baseline, labels)
    assert bool(d["s0_protected"][0]) and bool(d["s0_trust_active"][0])
    assert abs(float(d["s0_safe_target"][0]) - 0.6) < 1e-6
    assert bool(d["s0_protected"][1]) and not bool(d["s0_trust_active"][1])
    assert abs(float(d["s0_safe_target"][1]) - 0.9) < 1e-6
    assert not bool(d["s0_protected"][2]) and not bool(d["s0_trust_active"][2])
    assert bool(d["s0_protected"][3]) and bool(d["s0_trust_active"][3])
    assert abs(float(d["s0_safe_target"][3]) - 1.0) < 1e-6


def synthetic_events(run, beneficial_ntr, nonbeneficial_ntr):
    rows = []
    rng = np.random.RandomState(7)
    for mode in ("LA", "LV", "L"):
        for index in range(229):
            beneficial = index < 96
            rank = index if beneficial else index - 96
            size = 96 if beneficial else 133
            rate = beneficial_ntr if beneficial else nonbeneficial_ntr
            gain = -0.05 if rank < int(round(size * rate)) else 0.05
            rows.append({
                "Seed": DEV_SEED, "Run": run, "Mode": mode, "sample_index": index,
                "teacher_advantage": 0.05 if beneficial else -0.01,
                "gain_vs_dlf": gain + float(rng.normal(scale=1e-5)),
            })
    return pd.DataFrame(rows)


def test_transfer_and_gate():
    candidate = mechanism_transfer_summary(synthetic_events(RUN, 0.20, 0.52), RUN)
    frozen_v4 = mechanism_transfer_summary(
        synthetic_events("regret_preserve_cfcompat", 0.31, 0.50),
        "regret_preserve_cfcompat",
    )
    signal = development_signal_gate(0.6790, 0.6784, candidate, frozen_v4)
    assert candidate["teacher_beneficial"]["negative_transfer_rate"] < frozen_v4["teacher_beneficial"]["negative_transfer_rate"]
    assert signal["checks"]["valid_J_noninferior_to_v4"]
    assert signal["checks"]["beneficial_teacher_NTR_reduction"]
    assert signal["checks"]["nonbeneficial_teacher_NTR_retained"]
    assert signal["passed"]


def main():
    test_s0_one_sided_trust()
    test_transfer_and_gate()
    print("PASS: Frozen-S0 trust-region v6 smoke tests")
    print("S0 protect margin:", S0_PROTECT_MARGIN)


if __name__ == "__main__":
    main()
