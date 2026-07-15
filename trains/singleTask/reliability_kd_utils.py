"""Fixed reliability-gated prediction KD helpers for Stage 3A."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .fixed_kd_utils import (
    assert_initial_lav_equivalence,
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)


def reliability_weights(teacher_prediction, labels):
    prediction = teacher_prediction.detach().clone().view(-1)
    target = labels.detach().view(-1)
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Teacher prediction and labels must be finite.")
    weight = torch.exp(-torch.abs(prediction - target))
    if not torch.isfinite(weight).all() or torch.any(weight <= 0) or torch.any(weight > 1):
        raise FloatingPointError("Reliability must be finite and in (0,1].")
    return weight


def reliability_kd_loss(student_prediction, teacher_prediction, weights):
    student = student_prediction.view(-1)
    teacher = teacher_prediction.detach().clone().view(-1)
    weight = weights.detach().view(-1).to(student)
    each = torch.nn.functional.smooth_l1_loss(student, teacher, reduction="none")
    if each.shape != weight.shape:
        raise ValueError("KD and reliability shape mismatch.")
    value = torch.sum(weight * each) / (torch.sum(weight) + 1e-8)
    if not torch.isfinite(value):
        raise FloatingPointError("Reliability KD is non-finite.")
    return value, each


def reliability_checkpoint_path(root, dataset, seed):
    return Path(root) / "missing_baseline" / "reliability_kd_v1" / "DLF_{}_seed{}_best.pth".format(dataset, seed)


def stats(values):
    values = torch.as_tensor(values, dtype=torch.float64).view(-1)
    if values.numel() == 0:
        raise ValueError("Reliability statistics require values.")
    quantiles = torch.quantile(values, torch.tensor([.1, .25, .5, .75, .9, .95], dtype=torch.float64))
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=False)), "min": float(values.min()),
            "p10": float(quantiles[0]), "p25": float(quantiles[1]), "median": float(quantiles[2]),
            "p75": float(quantiles[3]), "p90": float(quantiles[4]), "p95": float(quantiles[5]),
            "max": float(values.max())}


def reliability_static_summary(epoch_rows, provenance="regenerated_from_existing_artifacts_without_training"):
    """Static seed summary, deliberately distinct from the per-epoch metric table."""
    frame = pd.DataFrame(epoch_rows)
    if frame.empty or not {"Seed", "Epoch", "J_valid", "J_test"}.issubset(frame):
        raise ValueError("Epoch metrics must include Seed/Epoch/J_valid/J_test.")
    rows = []
    for seed, local in frame.groupby("Seed", sort=True):
        valid_best = local.loc[local.J_valid.idxmin()]
        test_best = local.loc[local.J_test.idxmin()]
        row = {
            "Seed": int(seed), "EpochCount": int(len(local)),
            "BestValidEpoch": int(valid_best.Epoch), "BestValidJ": float(valid_best.J_valid),
            "J_test_at_valid_best": float(valid_best.J_test),
            "BestObservedTestEpoch": int(test_best.Epoch), "BestObservedTestJ": float(test_best.J_test),
            "provenance": provenance,
        }
        for column in ("KD_loss", "ESS_fraction", "reliability_mean", "teacher_error_mean", "train_teacher_student_abs_gap"):
            if column in local:
                row["mean_{}".format(column)] = float(local[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_reliability_static_summary(epoch_csv, output_csv):
    """Regenerate only the recoverable static Stage 3A summary from epoch CSV."""
    summary = reliability_static_summary(pd.read_csv(epoch_csv))
    summary.to_csv(output_csv, index=False)
    return summary


def reliability_quartile_rows(records, seed, epoch):
    """Future Stage 3A runs emit genuine reliability quartiles from batch records."""
    frame = pd.DataFrame(records)
    required = {"reliability", "teacher_error", "kd", "student_missing_abs_label_error"}
    if frame.empty or not required.issubset(frame):
        raise ValueError("Raw reliability diagnostics are required for quartiles.")
    ordered = frame.sort_values(["reliability", "sample_index"], kind="mergesort").copy()
    ordered["reliability_quartile"] = pd.qcut(np.arange(len(ordered)), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    rows = []
    for group, local in ordered.groupby("reliability_quartile", observed=False):
        rows.append({
            "Seed": int(seed), "Epoch": int(epoch), "quartile": str(group), "count": int(len(local)),
            "mean_reliability": float(local.reliability.mean()), "mean_teacher_error": float(local.teacher_error.mean()),
            "mean_unweighted_kd": float(local.kd.mean()),
            "mean_student_missing_abs_label_error": float(local.student_missing_abs_label_error.mean()),
        })
    return rows
