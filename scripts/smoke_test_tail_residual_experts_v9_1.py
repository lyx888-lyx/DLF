"""CPU-only synthetic smoke test for V9.1 tail residual objectives."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from trains.singleTask.tail_residual_experts_v91 import (
    MECHANISM_NAMES,
    TAIL_ROLE_NAMES,
    TailLossWeights,
    exact_tail_mask,
    residual_diagnostic_rows,
    residual_mechanism_index,
    soft_tail_membership,
    tail_residual_loss,
)


def main() -> None:
    torch.manual_seed(17)
    labels = torch.tensor(
        [-2.8, -2.1, -1.7, -1.0, -0.2, 0.4, 1.0, 1.7, 2.2, 2.9]
    ).view(-1, 1)
    anchor = torch.tensor(
        [-1.2, -2.7, 0.3, -0.8, 0.1, 0.2, 0.8, 2.4, 1.2, -0.4]
    ).view(-1, 1)
    residual_target = labels - anchor

    mechanisms = residual_mechanism_index(anchor, labels)
    assert mechanisms.min().item() >= 0
    assert mechanisms.max().item() < len(MECHANISM_NAMES)
    assert set(mechanisms.tolist()) >= {0, 1, 2}

    for role in TAIL_ROLE_NAMES:
        membership = soft_tail_membership(labels, role, temperature=0.25)
        mask = exact_tail_mask(labels, role)
        assert membership.shape == (labels.size(0),)
        assert bool((membership >= 0).all() and (membership <= 1).all())
        assert bool(mask.any())

        raw_correction = torch.zeros_like(labels, requires_grad=True)
        gate_logit = torch.zeros_like(labels, requires_grad=True)
        gate_prob = torch.sigmoid(gate_logit)
        correction = gate_prob.detach() * raw_correction
        prediction = anchor + correction
        mechanism_logits = torch.randn(labels.size(0), 4, requires_grad=True)
        outputs = {
            "prediction": prediction,
            "raw_correction": raw_correction,
            "correction": correction,
            "applicability_logit": gate_logit,
            "mechanism_logits": mechanism_logits,
        }
        losses = tail_residual_loss(
            outputs,
            labels,
            anchor,
            anchor + 0.25 * residual_target,
            role=role,
            membership_temperature=0.25,
            gain_margin=0.08,
            gain_fraction=0.20,
            teacher_scale=1.0,
            weights=TailLossWeights(),
        )
        assert all(torch.isfinite(value) for value in losses.values())
        losses["total"].backward()
        assert raw_correction.grad is not None
        assert gate_logit.grad is not None
        assert mechanism_logits.grad is not None

    rows = residual_diagnostic_rows(
        anchor,
        labels,
        ["synthetic-%02d" % index for index in range(labels.size(0))],
        "synthetic",
    )
    assert rows
    assert {row["role"] for row in rows} == set(TAIL_ROLE_NAMES)
    print("V9.1 TAIL RESIDUAL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
