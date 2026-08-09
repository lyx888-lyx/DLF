"""Synthetic invariants for the v12.2 trajectory audit helpers."""
from __future__ import annotations

import pandas as pd
import torch

from trains.singleTask.cfcompat_trajectory_audit_utils import (
    FIXED_GRADIENT_MILESTONES,
    aggregate_epoch_groups,
    clone_state_dict,
    enrich_transfer_flags,
    epoch_group_summary,
    milestone_roles,
    state_dict_sha256,
    states_exactly_equal,
)


def main():
    state = {
        "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "b": torch.tensor([3.0], dtype=torch.float32),
    }
    clone = clone_state_dict(state)
    assert states_exactly_equal(state, clone)
    assert state_dict_sha256(state) == state_dict_sha256(clone)
    clone["b"][0] = 4.0
    assert not states_exactly_equal(state, clone)
    assert state_dict_sha256(state) != state_dict_sha256(clone)

    roles = milestone_roles(range(0, 20), selected_epoch=14, best_epoch=17)
    assert 0 in FIXED_GRADIENT_MILESTONES
    assert "FIXED_MILESTONE" in roles[0]
    assert "CONSERVATIVE_SELECTED" in roles[14]
    assert "ABSOLUTE_BEST" in roles[17]

    rows = []
    for fold in (0, 1):
        for epoch in (0, 1):
            for i, (teacher_beneficial, s0_beneficial, gain) in enumerate(
                ((True, True, 0.10), (False, False, -0.10))
            ):
                rows.append(
                    {
                        "Fold": fold,
                        "Epoch": epoch,
                        "sample_index": fold * 10 + i,
                        "sample_id": "s{}".format(fold * 10 + i),
                        "video_id": "v{}".format(fold),
                        "mode": "LA",
                        "teacher_beneficial": teacher_beneficial,
                        "s0_beneficial": s0_beneficial,
                        "current_gain_vs_baseline": gain,
                        "current_gain_vs_s0": gain / 2.0,
                        "current_error": 0.2 if gain > 0 else 0.4,
                        "s0_error": 0.3,
                        "residual_abs_delta": 0.05,
                        "residual_crossed_label_from_s0": False,
                        "residual_crossed_baseline_from_s0": False,
                    }
                )
    events = enrich_transfer_flags(pd.DataFrame(rows))
    summary = epoch_group_summary(events)
    aggregate = aggregate_epoch_groups(summary)
    assert len(summary) > 0 and len(aggregate) > 0
    teacher_good = aggregate.loc[
        aggregate.Group.eq("TEACHER_BENEFICIAL") & aggregate.Epoch.eq(0)
    ].iloc[0]
    teacher_bad = aggregate.loc[
        aggregate.Group.eq("TEACHER_NONBENEFICIAL") & aggregate.Epoch.eq(0)
    ].iloc[0]
    assert float(teacher_good.positive_transfer_rate) == 1.0
    assert float(teacher_bad.negative_transfer_rate) == 1.0

    print("v12.2 trajectory audit smoke passed")


if __name__ == "__main__":
    main()
