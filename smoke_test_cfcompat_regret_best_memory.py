"""Synthetic unit smoke for Regret-Aware Best-So-Far Memory CFCompatKD v4.2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trains.singleTask.cfcompat_regret_best_memory_utils import (
    BestSoFarPredictionMemory,
    WEAK_DISTILL_SCALE,
    best_memory_projection_summary,
    regret_best_memory_decision,
    tiered_kd_loss,
)
from trains.singleTask.missing_utils import MISSING_MODES


def assert_close(actual, expected, tolerance=1e-7):
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError("{} != {}".format(actual, expected))


def baseline_fixture():
    rows = []
    for index in range(1284):
        label = 0.0
        row = {"sample_index": index, "sample_id": "s{}".format(index), "label": label}
        for mode in MISSING_MODES:
            row["baseline_{}_pred".format(mode)] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    memory = BestSoFarPredictionMemory(baseline_fixture())

    # First event improves the initial ModDrop anchor from error 1.0 to 0.4.
    update = memory.bind_and_update(
        indices=[0], modes=["LA"],
        current_prediction=torch.tensor([[0.4]]),
        labels=torch.tensor([[0.0]]),
        event_start_ordinal=1,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    if not bool(update["memory_updated"][0]):
        raise AssertionError("Best memory failed to accept an improved Student prediction.")
    assert_close(update["memory_before_prediction"][0], 1.0)
    assert_close(update["memory_after_prediction"][0], 0.4)

    # A later regression must not overwrite memory.
    regression = memory.bind_and_update(
        indices=[0], modes=["LA"],
        current_prediction=torch.tensor([[0.8]]),
        labels=torch.tensor([[0.0]]),
        event_start_ordinal=2,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    if bool(regression["memory_updated"][0]):
        raise AssertionError("Best memory accepted a worse Student prediction.")
    assert_close(regression["memory_after_prediction"][0], 0.4)

    student = torch.tensor([[0.8], [1.0], [1.0], [1.0]], dtype=torch.float32)
    teacher = torch.tensor([[0.1], [0.99], [1.2], [1.0]], dtype=torch.float32)
    memory_pred = torch.tensor([[0.4], [1.0], [0.4], [1.0]], dtype=torch.float32)
    labels = torch.zeros_like(student)
    compatibility = torch.tensor([0.8, 0.6, 0.7, 0.5], dtype=torch.float32)
    decision = regret_best_memory_decision(
        student, teacher, memory_pred, labels, compatibility
    )

    expected_strong = np.asarray([True, False, False, False])
    expected_weak = np.asarray([False, True, False, False])
    expected_preserve = np.asarray([False, False, True, False])
    expected_abstain = np.asarray([False, False, False, True])
    if not np.array_equal(decision["strong_distill"].cpu().numpy(), expected_strong):
        raise AssertionError("Synthetic strong-distill routing changed.")
    if not np.array_equal(decision["weak_distill"].cpu().numpy(), expected_weak):
        raise AssertionError("Synthetic weak-distill routing changed.")
    if not np.array_equal(decision["preserve"].cpu().numpy(), expected_preserve):
        raise AssertionError("Synthetic preserve routing changed.")
    if not np.array_equal(decision["abstain"].cpu().numpy(), expected_abstain):
        raise AssertionError("Synthetic abstain routing changed.")

    # True weak attenuation: with an all-weak one-sample batch the returned KD
    # must be exactly WEAK_DISTILL_SCALE times the unweighted SmoothL1 value.
    weak_student = torch.tensor([1.0])
    weak_target = torch.tensor([0.99])
    weak_gate = torch.tensor([0.9])
    loss, each, eligible, effective = tiered_kd_loss(
        weak_student, weak_target, torch.tensor([0.0]), weak_gate
    )
    assert_close(loss, WEAK_DISTILL_SCALE * each[0], tolerance=1e-9)
    assert_close(eligible[0], 0.9)
    assert_close(effective[0], WEAK_DISTILL_SCALE * 0.9)

    records = []
    projection = decision["teacher_projection"]
    for offset, mode in enumerate(("LA", "LV", "L", "LA")):
        record = {
            key: (
                bool(value[offset].cpu()) if value.dtype == torch.bool
                else float(value[offset].cpu())
            )
            for key, value in projection.items()
        }
        record.update({
            "mode": mode,
            "strong_distill": bool(decision["strong_distill"][offset]),
            "weak_distill": bool(decision["weak_distill"][offset]),
            "preserve": bool(decision["preserve"][offset]),
            "memory_abstain": bool(decision["abstain"][offset]),
            "memory_updated": offset == 0,
            "memory_before_error": 1.0,
            "memory_after_error": 0.4 if offset == 0 else 1.0,
            "memory_improvement": 0.6 if offset == 0 else 0.0,
            "memory_error": float(decision["memory_error"][offset]),
            "teacher_error": float(decision["teacher_error"][offset]),
            "current_error": float(decision["current_error"][offset]),
            "teacher_advantage_vs_memory": float(decision["teacher_advantage_vs_memory"][offset]),
            "current_regret_vs_memory": float(decision["current_regret_vs_memory"][offset]),
            "compatibility": float(compatibility[offset]),
            "mild_compatibility": float(decision["mild_compatibility"][offset]),
            "strong_gate": float(decision["strong_gate"][offset]),
            "weak_gate": float(decision["weak_gate"][offset]),
            "eligible_distill_mass": float(decision["strong_gate"][offset] + decision["weak_gate"][offset]),
            "effective_distill_gate": float(decision["strong_gate"][offset] + WEAK_DISTILL_SCALE * decision["weak_gate"][offset]),
            "preserve_gate": float(decision["preserve_gate"][offset]),
        })
        records.append(record)
    summary = best_memory_projection_summary(records)
    assert_close(summary["strong_distill_fraction"], 0.25)
    assert_close(summary["weak_distill_fraction"], 0.25)
    assert_close(summary["preserve_fraction"], 0.25)
    assert_close(summary["memory_abstain_fraction"], 0.25)
    assert_close(summary["memory_update_fraction"], 0.25)

    final = memory.to_frame()
    selected = final.loc[(final.sample_index == 0) & (final["mode"] == "LA")].iloc[0]
    assert_close(selected.best_prediction, 0.4)
    if int(selected.update_count) != 1:
        raise AssertionError("Best-memory update count changed.")

    print("Regret-aware best-so-far memory v4.2 utility smoke test passed")


if __name__ == "__main__":
    main()
