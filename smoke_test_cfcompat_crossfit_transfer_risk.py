"""Synthetic checks for Cross-Fitted Transfer-Risk CFCompatKD v5."""
import numpy as np
import pandas as pd
import torch

from trains.singleTask.cfcompat_crossfit_transfer_risk_utils import (
    CROSSFIT_FOLDS,
    MISSING_MODES,
    fit_crossfit_transfer_risk_gate,
    probability_to_distill_weight,
    transfer_risk_kd_loss,
)


def synthetic_frame(n_samples, split, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_samples):
        label = rng.normal(0.0, 1.0)
        baseline_full = label + rng.normal(0.0, 0.45)
        teacher = label + rng.normal(0.0, 0.35)
        for mode_position, mode in enumerate(MISSING_MODES):
            difficulty = 0.20 + 0.12 * mode_position
            baseline_missing = baseline_full + rng.normal(0.0, difficulty)
            initial_student = baseline_missing + rng.normal(0.0, 0.12)
            # Give the label-free gaps some real predictive signal so the
            # synthetic gate must learn rather than rely on row identity.
            if (index + mode_position) % 3 == 0:
                teacher = label + rng.normal(0.0, 0.08)
            rows.append(
                {
                    "sample_index": index,
                    "sample_id": "{}-{}".format(split, index),
                    "mode": mode,
                    "split": split,
                    "label": label,
                    "baseline_missing_prediction": baseline_missing,
                    "baseline_full_prediction": baseline_full,
                    "teacher_full_prediction": teacher,
                    "initial_student_missing_prediction": initial_student,
                }
            )
    return pd.DataFrame(rows)


def main():
    # Utility requires the real Train cardinality so the fold/group invariant is
    # exercised exactly as in MOSI.
    train = synthetic_frame(1284, "train", 11)
    valid = synthetic_frame(229, "valid", 17)
    train_gate, valid_gate, summary = fit_crossfit_transfer_risk_gate(train, valid)

    assert len(train_gate) == 1284 * len(MISSING_MODES)
    assert train_gate.sample_index.nunique() == 1284
    assert set(train_gate.crossfit_fold.astype(int)) == set(range(CROSSFIT_FOLDS))
    per_sample_fold_count = train_gate.groupby("sample_index").crossfit_fold.nunique()
    assert int(per_sample_fold_count.max()) == 1
    assert np.isfinite(train_gate.oof_benefit_probability.to_numpy(float)).all()
    assert np.isfinite(valid_gate.full_train_benefit_probability.to_numpy(float)).all()

    probability = torch.tensor([0.20, 0.50, 0.75, 1.0 - 1e-6], dtype=torch.float32)
    weight = probability_to_distill_weight(probability)
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0 - 2e-6], dtype=torch.float32)
    assert torch.allclose(weight, expected, atol=1e-6, rtol=0.0)

    student = torch.tensor([[0.8], [0.8], [0.8], [0.8]], requires_grad=True)
    target = torch.zeros_like(student)
    active = torch.tensor([True, True, True, True])
    loss, each, effective, eligible, risk_weight = transfer_risk_kd_loss(
        student, target, active, probability
    )
    manual = torch.sum(risk_weight * each) / 4.0
    assert torch.allclose(loss, manual, atol=1e-7, rtol=0.0)
    assert torch.allclose(effective, risk_weight, atol=1e-7, rtol=0.0)
    assert float(eligible.sum()) == 4.0
    loss.backward()
    assert torch.isfinite(student.grad).all()

    print("Cross-fitted transfer-risk v5 utility smoke test passed")
    print("synthetic OOF AUC: {:.6f}".format(summary["train_oof"]["roc_auc"]))
    print("synthetic Valid AUC: {:.6f}".format(summary["valid_full_train_gate"]["roc_auc"]))
    print("group-disjoint folds: True")
    print("true-strength attenuation: True")


if __name__ == "__main__":
    main()
