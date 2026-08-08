"""Synthetic invariants for CFCompatKD v11 objective audit."""
import pandas as pd
import torch

from trains.singleTask.cfcompat_objective_audit_utils import (
    BASE_TRAIN_COMPONENTS,
    derive_train_component_vectors,
    enrich_oof_event_frame,
    gradient_influence_row,
)


def main():
    frame = pd.DataFrame(
        [
            dict(
                fold=0, sample_index=0, sample_id="v0$_$0", mode="LA", label=1.0,
                baseline_prediction=0.0, s0_prediction=0.5, student_prediction=0.6,
                residual_delta=0.1, teacher_prediction=0.8, teacher_safe_target=0.8,
                preserve_safe_target=0.5, distill=True, preserve=False,
                decision_abstain=False, teacher_beneficial=True, current_regressed=False,
            ),
            dict(
                fold=0, sample_index=1, sample_id="v1$_$0", mode="LV", label=-1.0,
                baseline_prediction=-0.8, s0_prediction=-0.7, student_prediction=-0.5,
                residual_delta=0.2, teacher_prediction=0.1, teacher_safe_target=-0.5,
                preserve_safe_target=-0.8, distill=False, preserve=True,
                decision_abstain=False, teacher_beneficial=False, current_regressed=True,
            ),
            dict(
                fold=0, sample_index=2, sample_id="v2$_$0", mode="L", label=0.2,
                baseline_prediction=0.1, s0_prediction=0.15, student_prediction=0.15,
                residual_delta=0.0, teacher_prediction=0.1, teacher_safe_target=0.15,
                preserve_safe_target=0.15, distill=False, preserve=False,
                decision_abstain=True, teacher_beneficial=False, current_regressed=False,
            ),
        ]
    )
    enriched = enrich_oof_event_frame(frame)
    assert enriched.branch.tolist() == ["DISTILL", "PRESERVE", "ABSTAIN"]
    assert bool(enriched.branch_target_locally_safe.all())
    assert bool(enriched.supervised_missing_active.all())
    assert bool(enriched.loc[enriched.branch.eq("ABSTAIN"), "supervised_missing_active"].all())

    base = {
        name: torch.tensor([float(i + 1), 0.0], dtype=torch.float64)
        for i, name in enumerate(BASE_TRAIN_COMPONENTS)
    }
    derived = derive_train_component_vectors(base)
    expected_supervised = (
        base["SUPERVISED_BRANCH_DISTILL"]
        + base["SUPERVISED_BRANCH_PRESERVE"]
        + base["SUPERVISED_BRANCH_ABSTAIN"]
    )
    assert torch.equal(derived["SUPERVISED_ALL"], expected_supervised)
    assert torch.equal(
        derived["TOTAL_RESIDUAL_OBJECTIVE"],
        derived["SUPERVISED_ALL"] + derived["SELECTIVE_ONLY"],
    )

    improve = gradient_influence_row(
        0, "ALL", "X", "OOF_ALL",
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([2.0, 0.0], dtype=torch.float64),
        3,
    )
    harm = gradient_influence_row(
        0, "ALL", "X", "OOF_ALL",
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([-2.0, 0.0], dtype=torch.float64),
        3,
    )
    assert improve["predicted_effect_of_gradient_descent"] == "IMPROVE_OOF"
    assert improve["predicted_first_order_oof_loss_change_per_unit_step"] < 0
    assert harm["predicted_effect_of_gradient_descent"] == "HARM_OOF"
    assert harm["predicted_first_order_oof_loss_change_per_unit_step"] > 0
    print("v11 objective-audit synthetic smoke passed")


if __name__ == "__main__":
    main()
