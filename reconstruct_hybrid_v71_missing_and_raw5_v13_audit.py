"""Reconstruct historical MOSI Hybrid V7.1 from the five raw CFCompat best-valid teachers.

This recovery/analysis utility is intentionally separate from ADPEP Stage9A.
The historical Hybrid reference used the five *raw* CFCompat best-valid checkpoints,
whereas ADPEP Stage9A used Stage8 Online replay artifacts.

Inputs are the already-regenerated per-seed valid/test prediction CSVs under
``result/missing_baseline/cfcompat_prediction_ensemble_v1/mosi``.  These CSVs were
produced directly from the five historical raw CFCompat checkpoints by the recovery
utility before its ADPEP Stage9A replay gate stopped.

Protocol:
1. Refit the original V7.1 global/region simplex committee on Valid LAV only.
2. Treat the calibrated student as the anchor (the alpha=0 algebraic reduction) and
   replay the original Valid Hybrid search over committee x beta.
3. Require the recovered Valid policy to match the frozen historical policy and
   Valid objective/MAE.
4. Only then evaluate LAV Test and require all archived LAV metrics to reproduce.
5. Only after that replay passes, apply the *same frozen* region weights, temperature,
   anchor teacher, and beta to LA/LV/L.  No missing-mode refit is allowed.
6. Separately, on Valid only, audit whether Raw5-PE5 and v13 are complementary via
   a fixed small blend grid and error correlations.  No blend is evaluated on Test.

No training is performed and no new sample-level Test artifact is written.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from trains.singleTask.complementarity_v71 import (
    apply_global_committee,
    apply_region_committee,
    fit_committee_cv,
    selection_stats,
)
from trains.singleTask.missing_utils import regression_metrics


SEEDS = (1111, 1112, 1113, 1114, 1115)
MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
EXPECTED_N = {"valid": 229, "test": 686}
REGION_TEMPERATURE = 0.55
COMMITTEE_STEPS = 600
BETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
RAW5_BLEND_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)

# Frozen archive/mosi-anchored-complementarity-0.6995 reference.
HISTORICAL_BASE_TEACHER_INDEX = 3
HISTORICAL_COMMITTEE = "region_simplex"
HISTORICAL_BETA = 0.5
HISTORICAL_VALID_MAE = 0.6593225598335266
HISTORICAL_VALID_OBJECTIVE = 0.6613312934339046
HISTORICAL_TEST = {
    "MAE": 0.6995,
    "Corr": 0.7942,
    "acc_2": 0.8491,
    "F1_score": 0.8484,
    "acc_7": 0.4781,
    "acc_5": 0.5394,
}
HISTORICAL_VALID_TOL = 5e-4
HISTORICAL_TEST_TOL = 5e-4
V13_EXPECTED_VALID_J = 0.6675898631413777
V13_REPLAY_TOL = 2e-6


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid V7.1 exact reconstruction + Raw5/v13 Valid audit")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _raw5_root(result_root: Path, dataset: str) -> Path:
    return result_root / "missing_baseline" / "cfcompat_prediction_ensemble_v1" / dataset


def _raw_prediction_path(root: Path, seed: int, split: str) -> Path:
    return root / "online_seed{}_{}_predictions.csv".format(int(seed), split)


def _load_raw_frame(path: Path, seed: int, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            "Raw5 recovery prediction is missing: {}. Re-run the five-checkpoint "
            "recovery inference first; the historical ADPEP PE5 gate may stop later, "
            "but these per-member CSVs must exist.".format(path)
        )
    frame = pd.read_csv(path)
    required = {
        "sample_index", "sample_id", "label", "Seed", "Method", "Split", "SelectedBy",
        *["{}_pred".format(mode) for mode in MODES],
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("{} lacks {}".format(path, sorted(missing)))
    if set(frame.Seed.astype(int)) != {int(seed)}:
        raise RuntimeError("Seed binding differs in {}".format(path))
    if set(frame.Method.astype(str)) != {"Online"}:
        raise RuntimeError("Method binding differs in {}".format(path))
    if set(frame.Split.astype(str)) != {str(split)}:
        raise RuntimeError("Split binding differs in {}".format(path))
    if set(frame.SelectedBy.astype(str)) != {"validation_J"}:
        raise RuntimeError("Selection binding differs in {}".format(path))
    frame["sample_index"] = frame.sample_index.astype(int)
    frame = frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != EXPECTED_N[split] or frame.sample_index.nunique() != EXPECTED_N[split]:
        raise RuntimeError("Unexpected {} sample binding in {}".format(split, path))
    numeric = frame[["sample_index", "label"] + ["{}_pred".format(m) for m in MODES]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Non-finite raw predictions in {}".format(path))
    return frame


def _bind(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if not np.array_equal(reference.sample_index.to_numpy(np.int64), candidate.sample_index.to_numpy(np.int64)):
        raise RuntimeError("sample_index binding differs")
    if not np.array_equal(reference.sample_id.astype(str).to_numpy(), candidate.sample_id.astype(str).to_numpy()):
        raise RuntimeError("sample_id binding differs")
    if not np.array_equal(reference.label.to_numpy(np.float32), candidate.label.to_numpy(np.float32)):
        raise RuntimeError("label binding differs")


def load_raw5(result_root: Path, dataset: str):
    root = _raw5_root(result_root, dataset)
    frames: Dict[str, Dict[int, pd.DataFrame]] = {split: {} for split in ("valid", "test")}
    for split in ("valid", "test"):
        reference = None
        for seed in SEEDS:
            frame = _load_raw_frame(_raw_prediction_path(root, seed, split), seed, split)
            if reference is None:
                reference = frame
            else:
                _bind(reference, frame)
            frames[split][seed] = frame
    return root, frames


def _teacher_tensor(frames: Dict[int, pd.DataFrame], mode: str) -> torch.Tensor:
    matrix = np.stack(
        [frames[seed]["{}_pred".format(mode)].to_numpy(np.float32) for seed in SEEDS],
        axis=1,
    )
    return torch.from_numpy(matrix).unsqueeze(-1)


def _labels(frame: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(frame.label.to_numpy(np.float32)).view(-1, 1)


def _committees(predictions, anchor, fitted):
    uniform = predictions.mean(dim=1)
    global_simplex = apply_global_committee(predictions, fitted["global_weights"])
    region_simplex = apply_region_committee(
        predictions,
        anchor,
        fitted["region_weights"],
        REGION_TEMPERATURE,
    )
    selected = region_simplex if fitted["selected"] == "region_simplex" else global_simplex
    return {
        "uniform": uniform,
        "global_simplex": global_simplex,
        "region_simplex": region_simplex,
        "selected": selected,
    }


def _fit_historical_hybrid(valid_frames: Dict[int, pd.DataFrame]):
    reference = valid_frames[SEEDS[0]]
    predictions = _teacher_tensor(valid_frames, "LAV")
    labels = _labels(reference)
    maes = torch.abs(predictions - labels.unsqueeze(1)).mean(dim=(0, 2))
    init_index = int(torch.argmin(maes).item())
    anchor = predictions[:, init_index]

    fitted = fit_committee_cv(
        predictions,
        labels,
        anchor,
        reference.sample_id.astype(str).tolist(),
        temperature=REGION_TEMPERATURE,
        steps=COMMITTEE_STEPS,
    )
    committees = _committees(predictions, anchor, fitted)

    rows = []
    best = None
    # Historical alpha=0 algebraic reduction: calibrated student == anchor.
    for committee_name in ("uniform", "global_simplex", "region_simplex", "selected"):
        for beta in BETAS:
            prediction = float(beta) * committees[committee_name] + (1.0 - float(beta)) * anchor
            stats = selection_stats(anchor, prediction, labels)
            objective = float(stats["mae"] + 0.02 * stats["harm_over_010_rate"])
            row = {
                "committee": committee_name,
                "beta": float(beta),
                "objective": objective,
                **{key: float(value) for key, value in stats.items()},
            }
            rows.append(row)
            if best is None or (row["objective"], row["mae"]) < (best["objective"], best["mae"]):
                best = row

    checks = {
        "base_teacher_index": init_index == HISTORICAL_BASE_TEACHER_INDEX,
        "committee_cv_selected": str(fitted["selected"]) == HISTORICAL_COMMITTEE,
        "hybrid_committee": str(best["committee"]) == HISTORICAL_COMMITTEE,
        "hybrid_beta": abs(float(best["beta"]) - HISTORICAL_BETA) <= 1e-12,
        "valid_mae": abs(float(best["mae"]) - HISTORICAL_VALID_MAE) <= HISTORICAL_VALID_TOL,
        "valid_objective": abs(float(best["objective"]) - HISTORICAL_VALID_OBJECTIVE) <= HISTORICAL_VALID_TOL,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Hybrid historical Valid reconstruction failed; do not use missing extension. "
            "checks={} init_index={} fitted_selected={} best={}".format(
                checks, init_index, fitted["selected"], best
            )
        )
    return {
        "init_index": init_index,
        "valid_teacher_mae": [float(value) for value in maes.tolist()],
        "fitted": fitted,
        "best": best,
        "calibration_rows": rows,
        "checks": checks,
    }


def _metric_dict(prediction: torch.Tensor, labels: torch.Tensor) -> dict:
    result = regression_metrics(prediction.view(-1), labels.view(-1))
    return {key: float(value) for key, value in result.items()}


def _mode_hybrid(frames: Dict[int, pd.DataFrame], mode: str, reconstruction: dict):
    predictions = _teacher_tensor(frames, mode)
    anchor = predictions[:, reconstruction["init_index"]]
    fitted = reconstruction["fitted"]
    committees = _committees(predictions, anchor, fitted)
    beta = float(reconstruction["best"]["beta"])
    committee_name = str(reconstruction["best"]["committee"])
    return beta * committees[committee_name] + (1.0 - beta) * anchor


def _hybrid_metrics(frames: Dict[int, pd.DataFrame], reconstruction: dict):
    labels = _labels(frames[SEEDS[0]])
    by_mode = {}
    predictions = {}
    for mode in MODES:
        pred = _mode_hybrid(frames, mode, reconstruction)
        predictions[mode] = pred
        by_mode[mode] = _metric_dict(pred, labels)
    missing_macro = {
        key: float(np.mean([by_mode[mode][key] for mode in MISSING_MODES]))
        for key in by_mode["LAV"]
    }
    j_value = 0.5 * by_mode["LAV"]["MAE"] + 0.5 * missing_macro["MAE"]
    return predictions, by_mode, missing_macro, float(j_value)


def _verify_test_lav(by_mode: dict):
    observed = {key: float(by_mode["LAV"][key]) for key in HISTORICAL_TEST}
    diffs = {key: abs(observed[key] - HISTORICAL_TEST[key]) for key in HISTORICAL_TEST}
    passed = all(value <= HISTORICAL_TEST_TOL for value in diffs.values())
    if not passed:
        raise RuntimeError(
            "Historical Hybrid LAV Test metrics did not replay. Missing extension is forbidden. "
            "observed={} expected={} diffs={}".format(observed, HISTORICAL_TEST, diffs)
        )
    return observed, diffs


def _raw5_mean_frame(frames: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    reference = frames[SEEDS[0]][["sample_index", "sample_id", "label"]].copy()
    for mode in MODES:
        reference["{}_pred".format(mode)] = np.mean(
            np.stack([frames[seed]["{}_pred".format(mode)].to_numpy(float) for seed in SEEDS], axis=0),
            axis=0,
        )
    return reference


def _frame_metrics(frame: pd.DataFrame):
    labels = torch.from_numpy(frame.label.to_numpy(np.float32))
    by_mode = {}
    for mode in MODES:
        pred = torch.from_numpy(frame["{}_pred".format(mode)].to_numpy(np.float32))
        by_mode[mode] = _metric_dict(pred, labels)
    missing_macro = {
        key: float(np.mean([by_mode[m][key] for m in MISSING_MODES]))
        for key in by_mode["LAV"]
    }
    j_value = 0.5 * by_mode["LAV"]["MAE"] + 0.5 * missing_macro["MAE"]
    return by_mode, missing_macro, float(j_value)


def _load_v13_valid(result_root: Path, dataset: str, reference: pd.DataFrame):
    path = (
        result_root / "missing_baseline" / "cfcompat_adam_step_safety_v13" / dataset
        / "valid_screen" / "seed1113_dev" / "adam_step_safety_v13_candidate_raw_valid_events.csv"
    )
    if not path.is_file():
        return None, {"available": False, "reason": "v13_raw_valid_events_missing", "path": str(path)}
    events = pd.read_csv(path)
    required = {"Mode", "sample_index", "sample_id", "label", "candidate_prediction", "Split"}
    if required.difference(events.columns):
        raise ValueError("v13 raw Valid events are incomplete")
    if set(events.Split.astype(str)) != {"valid"} or set(events.Mode.astype(str)) != set(MODES):
        raise RuntimeError("v13 raw event split/mode binding differs")
    frame = reference[["sample_index", "sample_id", "label"]].copy()
    for mode in MODES:
        local = events.loc[events.Mode.astype(str).eq(mode)].sort_values("sample_index", kind="mergesort")
        if len(local) != len(reference):
            raise RuntimeError("v13 mode {} sample count differs".format(mode))
        if not np.array_equal(local.sample_index.to_numpy(np.int64), reference.sample_index.to_numpy(np.int64)):
            raise RuntimeError("v13/raw5 Valid sample_index binding differs for {}".format(mode))
        if not np.array_equal(local.label.to_numpy(np.float32), reference.label.to_numpy(np.float32)):
            raise RuntimeError("v13/raw5 Valid label binding differs for {}".format(mode))
        frame["{}_pred".format(mode)] = local.candidate_prediction.to_numpy(float)
    _, _, j_value = _frame_metrics(frame)
    if abs(j_value - V13_EXPECTED_VALID_J) > V13_REPLAY_TOL:
        raise RuntimeError("v13 Valid J replay failed: actual={} expected={}".format(j_value, V13_EXPECTED_VALID_J))
    return frame, {"available": True, "path": str(path), "ValidJ": j_value}


def _blend_audit(raw5: pd.DataFrame, v13: pd.DataFrame):
    labels = raw5.label.to_numpy(float)
    rows = []
    for weight in RAW5_BLEND_WEIGHTS:
        frame = raw5[["sample_index", "sample_id", "label"]].copy()
        for mode in MODES:
            frame["{}_pred".format(mode)] = (
                float(weight) * raw5["{}_pred".format(mode)].to_numpy(float)
                + (1.0 - float(weight)) * v13["{}_pred".format(mode)].to_numpy(float)
            )
        by_mode, missing_macro, j_value = _frame_metrics(frame)
        rows.append({
            "Raw5Weight": float(weight),
            "V13Weight": 1.0 - float(weight),
            "ValidJ": float(j_value),
            "LAV_MAE": float(by_mode["LAV"]["MAE"]),
            "MissingMacroMAE": float(missing_macro["MAE"]),
            "LA_MAE": float(by_mode["LA"]["MAE"]),
            "LV_MAE": float(by_mode["LV"]["MAE"]),
            "L_MAE": float(by_mode["L"]["MAE"]),
        })
    grid = pd.DataFrame(rows).sort_values(["ValidJ", "Raw5Weight"], kind="mergesort").reset_index(drop=True)

    correlations = []
    for mode in MODES:
        raw_err = raw5["{}_pred".format(mode)].to_numpy(float) - labels
        v13_err = v13["{}_pred".format(mode)].to_numpy(float) - labels
        signed_corr = float(np.corrcoef(raw_err, v13_err)[0, 1])
        abs_corr = float(np.corrcoef(np.abs(raw_err), np.abs(v13_err))[0, 1])
        correlations.append({
            "Mode": mode,
            "SignedErrorCorrelation": signed_corr,
            "AbsoluteErrorCorrelation": abs_corr,
            "MeanAbsPredictionDisagreement": float(np.mean(np.abs(
                raw5["{}_pred".format(mode)].to_numpy(float)
                - v13["{}_pred".format(mode)].to_numpy(float)
            ))),
        })
    return grid, pd.DataFrame(correlations)


def _metric_rows(method: str, split: str, by_mode: dict, missing_macro: dict, j_value: float):
    rows = []
    for mode in MODES:
        rows.append({"Method": method, "Split": split, "Mode": mode, "J": j_value, **by_mode[mode]})
    rows.append({"Method": method, "Split": split, "Mode": "MissingMacro", "J": j_value, **missing_macro})
    return rows


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main():
    cli = parse_args()
    result_root = Path(cli.result_root)
    output = result_root / "missing_baseline" / "hybrid_v71_missing_reconstruction_v1" / cli.dataset
    if output.exists():
        if not cli.overwrite:
            raise FileExistsError("Output exists; inspect it or use --overwrite: {}".format(output))
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    raw_root, raw = load_raw5(result_root, cli.dataset)
    reconstruction = _fit_historical_hybrid(raw["valid"])
    pd.DataFrame(reconstruction["calibration_rows"]).to_csv(output / "hybrid_v71_valid_calibration_replay.csv", index=False)

    all_metric_rows = []
    split_summary = {}
    for split in ("valid", "test"):
        _, by_mode, missing_macro, j_value = _hybrid_metrics(raw[split], reconstruction)
        all_metric_rows.extend(_metric_rows("hybrid_v71_missing_extension", split, by_mode, missing_macro, j_value))
        split_summary[split] = {
            "J": j_value,
            "LAV": by_mode["LAV"],
            "MissingMacro": missing_macro,
            "LA": by_mode["LA"],
            "LV": by_mode["LV"],
            "L": by_mode["L"],
        }

    test_observed, test_diffs = _verify_test_lav({
        mode: split_summary["test"][mode] for mode in MODES
    })

    raw5_valid = _raw5_mean_frame(raw["valid"])
    raw5_by_mode, raw5_missing, raw5_j = _frame_metrics(raw5_valid)
    all_metric_rows.extend(_metric_rows("raw5_pe5", "valid", raw5_by_mode, raw5_missing, raw5_j))

    v13_valid, v13_status = _load_v13_valid(result_root, cli.dataset, raw5_valid)
    blend_grid = None
    corr = None
    if v13_valid is not None:
        v13_by_mode, v13_missing, v13_j = _frame_metrics(v13_valid)
        all_metric_rows.extend(_metric_rows("v13", "valid", v13_by_mode, v13_missing, v13_j))
        blend_grid, corr = _blend_audit(raw5_valid, v13_valid)
        blend_grid.to_csv(output / "raw5_v13_valid_blend_grid.csv", index=False)
        corr.to_csv(output / "raw5_v13_valid_error_correlation.csv", index=False)

    pd.DataFrame(all_metric_rows).to_csv(output / "hybrid_raw5_v13_valid_metrics.csv", index=False)

    fitted = reconstruction["fitted"]
    summary = {
        "protocol": {
            "hybrid_source": "five raw historical CFCompat best-valid teachers",
            "adpep_stage9a_used": False,
            "training": False,
            "committee_fit_split": "valid_LAV_only",
            "missing_mode_refit": False,
            "test_used_for_selection": False,
            "test_lav_used_only_as_historical_replay_verification": True,
            "sample_level_test_output_written": False,
            "raw5_v13_blend_audit_split": "valid_only",
            "raw5_v13_blend_test_evaluated": False,
        },
        "raw5_prediction_root": str(raw_root.resolve()),
        "historical_reconstruction": {
            "base_teacher_index": reconstruction["init_index"],
            "base_teacher_seed": SEEDS[reconstruction["init_index"]],
            "valid_teacher_lav_mae": reconstruction["valid_teacher_mae"],
            "committee_selected": fitted["selected"],
            "committee_selected_regularization": fitted["selected_regularization"],
            "global_cv_score": fitted["global_cv_score"],
            "region_cv_score": fitted["region_cv_score"],
            "global_weights": fitted["global_weights"],
            "region_weights": fitted["region_weights"],
            "region_temperature": REGION_TEMPERATURE,
            "hybrid_valid_selected": reconstruction["best"],
            "valid_checks": reconstruction["checks"],
            "test_lav_observed": test_observed,
            "test_lav_expected": HISTORICAL_TEST,
            "test_lav_absolute_differences": test_diffs,
            "test_lav_replay_passed": True,
        },
        "hybrid_missing_metrics": split_summary,
        "raw5_valid": {
            "J": raw5_j,
            "LAV_MAE": raw5_by_mode["LAV"]["MAE"],
            "MissingMacroMAE": raw5_missing["MAE"],
        },
        "v13_valid_status": v13_status,
        "blend_valid": None if blend_grid is None else {
            "best_grid_row": blend_grid.iloc[0].to_dict(),
            "fixed_50_50_row": blend_grid.loc[np.isclose(blend_grid.Raw5Weight, 0.5)].iloc[0].to_dict(),
            "improves_best_single": bool(
                float(blend_grid.iloc[0].ValidJ) < min(raw5_j, float(v13_status["ValidJ"]))
            ),
        },
    }
    (output / "hybrid_v71_missing_reconstruction_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Hybrid V7.1 historical reconstruction: PASS")
    print("  base teacher: seed{}".format(SEEDS[reconstruction["init_index"]]))
    print("  committee: {} beta={:.2f}".format(reconstruction["best"]["committee"], reconstruction["best"]["beta"]))
    print("  historical Test LAV MAE replay: {:.9f}".format(split_summary["test"]["LAV"]["MAE"]))
    print("Hybrid missing extension:")
    print("  Valid J: {:.9f}".format(split_summary["valid"]["J"]))
    print("  Test  J: {:.9f}".format(split_summary["test"]["J"]))
    print("  Test MissingMacro MAE: {:.9f}".format(split_summary["test"]["MissingMacro"]["MAE"]))
    print("Raw5 PE5 Valid J: {:.9f}".format(raw5_j))
    if blend_grid is not None:
        fixed = blend_grid.loc[np.isclose(blend_grid.Raw5Weight, 0.5)].iloc[0]
        best = blend_grid.iloc[0]
        print("Raw5/v13 fixed 50/50 Valid J: {:.9f}".format(float(fixed.ValidJ)))
        print("Raw5/v13 best fixed-grid Valid J: {:.9f} at Raw5Weight={:.2f}".format(
            float(best.ValidJ), float(best.Raw5Weight)
        ))
        print("Blend improves best single on Valid:", bool(summary["blend_valid"]["improves_best_single"]))
    else:
        print("Raw5/v13 blend audit unavailable:", v13_status.get("reason"))
    print("output:", output)


if __name__ == "__main__":
    main()
