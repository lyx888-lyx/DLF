"""Shared, Train-only utilities for the Stage23A-v2 protocol audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = ROOT / "result" / "arbiter_audit_v1" / "mosei"
V2_ROOT = ROOT / "result" / "arbiter_audit_v2" / "mosei"
RUNTIME_ROOT = ROOT / "runtime" / "stage23a_v2"

MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
EXPERTS = (
    "uniform_kd_seed1111",
    "moddrop_seed1111",
    "moddrop_seed1114",
    "cfcompat_seed1111",
    "cfcompat_seed1114",
)
COMPONENTS = (
    "clean_seed1111",
    "clean_seed1114",
) + EXPERTS

MODE_AVAILABILITY = {
    "LAV": (1, 1, 1),
    "LA": (1, 1, 0),
    "LV": (1, 0, 1),
    "L": (1, 0, 0),
}
MODE_ACTIVE_HEADS = {
    "LAV": (
        "output_logit",
        "logits_c",
        "logits_l_hetero",
        "logits_a_hetero",
        "logits_v_hetero",
    ),
    "LA": (
        "output_logit",
        "logits_c",
        "logits_l_hetero",
        "logits_a_hetero",
    ),
    "LV": (
        "output_logit",
        "logits_c",
        "logits_l_hetero",
        "logits_v_hetero",
    ),
    "L": (
        "output_logit",
        "logits_c",
        "logits_l_hetero",
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    data = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    display = frame.fillna("NA").replace("", "NA")
    display.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(str(temporary), str(path))


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def stable_bucket(value, salt, modulo):
    digest = hashlib.sha256(
        "{}|{}".format(salt, value).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % int(modulo)


def source_video(sample_id):
    value = str(sample_id)
    if "$_$" not in value:
        raise ValueError("MOSEI sample ID has no source delimiter: {}".format(value))
    return value.rsplit("$_$", 1)[0]


def regression_metrics(prediction, label):
    from sklearn.metrics import f1_score

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    if len(prediction) != len(label) or not len(label):
        raise ValueError("Metric arrays are empty or differ.")
    clipped7_pred = np.clip(prediction, -3, 3)
    clipped7_label = np.clip(label, -3, 3)
    clipped5_pred = np.clip(prediction, -2, 2)
    clipped5_label = np.clip(label, -2, 2)
    nonzero = label != 0
    binary_pred = prediction[nonzero] > 0
    binary_label = label[nonzero] > 0
    corr = (
        float(np.corrcoef(prediction, label)[0, 1])
        if len(label) > 1 and prediction.std() > 0 and label.std() > 0
        else 0.0
    )
    return {
        "MAE": float(np.mean(np.abs(prediction - label))),
        "Corr": corr,
        "Acc7": float(np.mean(np.round(clipped7_pred) == np.round(clipped7_label))),
        "Acc5": float(np.mean(np.round(clipped5_pred) == np.round(clipped5_label))),
        "Acc2": float(np.mean(binary_pred == binary_label)) if nonzero.any() else 0.0,
        "F1": (
            float(
                f1_score(
                    binary_label,
                    binary_pred,
                    average="weighted",
                    zero_division=0,
                )
            )
            if nonzero.any()
            else 0.0
        ),
    }


def overall_j(frame, prediction_column):
    maes = {}
    for mode in MODES:
        local = frame.loc[frame["mode"] == mode]
        maes[mode] = float(
            np.mean(np.abs(local[prediction_column] - local["label"]))
        )
    return 0.5 * maes["LAV"] + 0.5 * np.mean(
        [maes[mode] for mode in MISSING_MODES]
    )


def optimize_simplex(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    count = predictions.shape[1]

    def objective(weights):
        return np.mean(np.abs(predictions.dot(weights) - labels))

    result = minimize(
        objective,
        np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda value: value.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("Simplex optimization failed: {}".format(result.message))
    weights = np.maximum(result.x, 0.0)
    return weights / weights.sum()


def metric_rows(frame, method, prediction_column):
    rows = []
    for mode in MODES:
        local = frame.loc[frame["mode"] == mode]
        values = regression_metrics(local[prediction_column], local["label"])
        rows.append({"method": method, "mode": mode, **values})
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("MAE", "Corr", "Acc7", "Acc5", "Acc2", "F1")
    }
    aggregate["J"] = overall_j(frame, prediction_column)
    rows.append({"method": method, "mode": "Overall", **aggregate})
    for row in rows:
        if row["mode"] != "Overall":
            row["J"] = row["MAE"]
    return rows
