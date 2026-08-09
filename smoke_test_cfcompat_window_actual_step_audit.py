"""Synthetic invariants for the v12.3 window-level actual-step audit."""
from __future__ import annotations

import pandas as pd
import torch

from trains.singleTask.cfcompat_window_actual_step_audit_utils import (
    AUDIT_EPOCHS,
    aggregate_mechanism,
    assign_parameter_tuple,
    clone_tensor_tuple,
    flatten_tensor_tuple,
    step_mechanism_row,
)


def make_row(surgery, delta, oof, before, after, group="OOF_TEACHER_BENEFICIAL"):
    return step_mechanism_row(
        fold=0,
        epoch=4,
        window=1,
        oof_group=group,
        oof_count=10,
        surgery_gradient=torch.tensor(surgery, dtype=torch.float64),
        actual_delta=torch.tensor(delta, dtype=torch.float64),
        oof_gradient_before=torch.tensor(oof, dtype=torch.float64),
        loss_before=before,
        loss_after=after,
    )


def main():
    assert AUDIT_EPOCHS == tuple(range(4, 13))

    layer = torch.nn.Linear(2, 1)
    params = list(layer.parameters())
    snapshot = clone_tensor_tuple(params)
    vector = flatten_tensor_tuple(snapshot)
    assert vector.numel() == sum(p.numel() for p in params)
    with torch.no_grad():
        for p in params:
            p.add_(1.0)
    assign_parameter_tuple(params, snapshot)
    for p, saved in zip(params, snapshot):
        assert torch.equal(p.detach().cpu(), saved)

    # Raw surgery gradient is harmful: dot(g_update, g_oof) < 0.
    raw_harm = make_row([1.0, 0.0], [-0.1, 0.0], [-1.0, 0.0], 1.0, 0.9)
    assert raw_harm["mechanism_class"] == "RAW_SURGERY_DIRECTION_HARM"

    # Raw descent is safe, but actual Adam displacement points uphill.
    adam_harm = make_row([1.0, 0.0], [0.1, 0.0], [1.0, 0.0], 1.0, 1.1)
    assert adam_harm["mechanism_class"] == "ADAM_TRANSFORM_HARM"

    # Both first-order directions are safe, but finite loss rises.
    nonlinear = make_row([1.0, 0.0], [-0.1, 0.0], [1.0, 0.0], 1.0, 1.1)
    assert nonlinear["mechanism_class"] == "NONLINEAR_FINITE_STEP_HARM"

    safe = make_row([1.0, 0.0], [-0.1, 0.0], [1.0, 0.0], 1.0, 0.9)
    assert safe["mechanism_class"] == "SAFE_OR_IMPROVING"

    frame = pd.DataFrame([raw_harm, adam_harm, nonlinear, safe])
    aggregate = aggregate_mechanism(frame)
    assert len(aggregate) == 1
    row = aggregate.iloc[0]
    assert int(row.WindowCount) == 4
    assert int(row.raw_direction_harm_count) == 1
    assert int(row.adam_transform_harm_count) == 1
    assert int(row.nonlinear_finite_step_harm_count) == 1
    assert int(row.safe_or_improving_count) == 1

    print("v12.3 window actual-step audit smoke passed")


if __name__ == "__main__":
    main()
