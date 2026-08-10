"""Decision-preserving ordinal projection for the frozen Raw5/v13 FixedBlend.

Goal
----
Use an anchor prediction only for Acc7/Acc5 decisions while keeping the latest
FixedBlend prediction as the continuous target.  For every sample/mode we project
FixedBlend to the closest float32 value that preserves the anchor's exact Acc7
and Acc5 evaluator decisions.  This is the ADPEP-57 geometry with a stronger
continuous target.

Important protocol points
-------------------------
* The projection API receives no labels.
* Anchor selection uses only validation J over the five fixed Raw5 members.
* No blend weight is searched; Raw5/v13 weights are frozen at 0.5/0.5.
* MOSI Test is deliberately blocked in this script because it has already been
  repeatedly accessed during exploratory development.
* MOSEI Valid/Test are supported once the corresponding frozen Raw5 member and
  v13 prediction files exist.
* Only aggregate metrics/diagnostics are written; projected sample predictions
  remain in memory.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
    select_anchor_seed,
)
from trains.singleTask.missing_utils import MISSING_MODES, regression_metrics


SEEDS = (1111, 1112, 1113, 1114, 1115)
MODES = ("LAV",) + MISSING_MODES
METRICS = ("MAE", "Corr", "acc_2", "F1_score", "acc_7", "acc_5")
RAW5_WEIGHT = 0.5
V13_WEIGHT = 0.5
METHOD = "FixedBlend-DP57-v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project frozen 50/50 Raw5-v13 FixedBlend into anchor Acc7/Acc5 decision cells"
    )
    parser.add_argument("--dataset", choices=("mosi", "mosei"), default="mosi")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--raw5-root")
    parser.add_argument(
        "--v13-predictions",
        help=(
            "Either a wide prediction CSV with LAV_pred/LA_pred/LV_pred/L_pred, "
            "or an event CSV with Mode and candidate_prediction."
        ),
    )
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset == "mosi" and args.split == "test":
        parser.error(
            "MOSI Test is intentionally blocked for this new method. Develop on Valid only; "
            "use MOSEI for the next untouched/generalization evaluation."
        )

    if args.raw5_root is None:
        args.raw5_root = str(
            Path(args.result_root)
            / "missing_baseline"
            / "cfcompat_prediction_ensemble_v1"
            / args.dataset
        )
    if args.v13_predictions is None:
        if args.dataset == "mosi" and args.split == "valid":
            args.v13_predictions = str(
                Path(args.result_root)
                / "missing_baseline"
                / "cfcompat_adam_step_safety_v13"
                / "mosi"
                / "valid_screen"
                / "seed1113_dev"
                / "adam_step_safety_v13_candidate_raw_valid_events.csv"
            )
        else:
            parser.error(
                "--v13-predictions is required for this dataset/split. "
                "Provide a frozen wide prediction CSV or v13 event CSV."
            )
    if args.output_root is None:
        args.output_root = str(
            Path(args.result_root)
            / "missing_baseline"
            / "fixedblend_dp57_v1"
            / args.dataset
            / args.split
        )
    return args


def _load_member(path: Path, seed: int, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    required = {"sample_index", "label", *[f"{mode}_pred" for mode in MODES]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    if "Seed" in frame.columns and set(frame.Seed.astype(int)) != {int(seed)}:
        raise RuntimeError(f"Seed binding differs for {path}")
    if "Split" in frame.columns and set(frame.Split.astype(str)) != {str(split)}:
        raise RuntimeError(f"Split binding differs for {path}")
    if frame.sample_index.duplicated().any():
        raise RuntimeError(f"Duplicate sample_index in {path}")
    numeric = frame[["sample_index", "label"] + [f"{m}_pred" for m in MODES]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise FloatingPointError(f"Non-finite prediction in {path}")
    return frame


def load_raw5(root: Path, split: str):
    frames = {}
    reference = None
    for seed in SEEDS:
        path = root / f"online_seed{seed}_{split}_predictions.csv"
        frame = _load_member(path, seed, split)
        if reference is None:
            reference = frame
        else:
            if not np.array_equal(
                reference.sample_index.to_numpy(np.int64),
                frame.sample_index.to_numpy(np.int64),
            ):
                raise RuntimeError("Raw5 sample_index differs across seeds")
            if "sample_id" in reference.columns and "sample_id" in frame.columns:
                if not np.array_equal(
                    reference.sample_id.astype(str).to_numpy(),
                    frame.sample_id.astype(str).to_numpy(),
                ):
                    raise RuntimeError("Raw5 sample_id differs across seeds")
            if np.max(np.abs(reference.label.to_numpy(float) - frame.label.to_numpy(float))) > 1e-7:
                raise RuntimeError("Raw5 labels differ across seeds")
        frames[seed] = frame
    return frames


def _wide_v13(frame: pd.DataFrame, reference: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {"sample_index", "label", *[f"{mode}_pred" for mode in MODES]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"v13 wide frame lacks {sorted(missing)}")
    local = frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if "Split" in local.columns and set(local.Split.astype(str)) != {str(split)}:
        raise RuntimeError("v13 wide frame split differs")
    return local


def _event_v13(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_index", "label", "Mode", "candidate_prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"v13 event frame lacks {sorted(missing)}")
    base_cols = ["sample_index", "label"]
    if "sample_id" in reference.columns:
        base_cols.insert(1, "sample_id")
    result = reference[base_cols].copy().sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    for mode in MODES:
        local = (
            frame.loc[frame.Mode.astype(str).eq(mode)]
            .sort_values("sample_index", kind="mergesort")
            .reset_index(drop=True)
        )
        if len(local) != len(result):
            raise RuntimeError(f"v13 event count differs for mode={mode}: {len(local)} != {len(result)}")
        if not np.array_equal(
            local.sample_index.to_numpy(np.int64), result.sample_index.to_numpy(np.int64)
        ):
            raise RuntimeError(f"v13 sample_index binding differs for mode={mode}")
        if np.max(np.abs(local.label.to_numpy(float) - result.label.to_numpy(float))) > 1e-7:
            raise RuntimeError(f"v13 labels differ for mode={mode}")
        result[f"{mode}_pred"] = local.candidate_prediction.to_numpy(float)
    return result


def load_v13(path: Path, reference: pd.DataFrame, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if {"Mode", "candidate_prediction"}.issubset(frame.columns):
        result = _event_v13(frame, reference)
    else:
        result = _wide_v13(frame, reference, split)
    result = result.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    ref = reference.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(result.sample_index.to_numpy(np.int64), ref.sample_index.to_numpy(np.int64)):
        raise RuntimeError("Raw5/v13 sample_index binding failed")
    if np.max(np.abs(result.label.to_numpy(float) - ref.label.to_numpy(float))) > 1e-7:
        raise RuntimeError("Raw5/v13 label binding failed")
    return result


def frame_metrics(frame: pd.DataFrame):
    labels = torch.tensor(frame.label.to_numpy(np.float32), dtype=torch.float32)
    by_mode = {}
    for mode in MODES:
        pred = torch.tensor(frame[f"{mode}_pred"].to_numpy(np.float32), dtype=torch.float32)
        by_mode[mode] = regression_metrics(pred, labels)
    by_mode["MissingMacro"] = {
        metric: float(np.mean([by_mode[m][metric] for m in MISSING_MODES]))
        for metric in METRICS
    }
    j = 0.5 * by_mode["LAV"]["MAE"] + 0.5 * by_mode["MissingMacro"]["MAE"]
    return by_mode, float(j)


def raw5_mean(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    ref = frames[SEEDS[0]]
    cols = ["sample_index", "label"]
    if "sample_id" in ref.columns:
        cols.insert(1, "sample_id")
    result = ref[cols].copy()
    for mode in MODES:
        result[f"{mode}_pred"] = np.mean(
            np.stack([frames[s][f"{mode}_pred"].to_numpy(float) for s in SEEDS], axis=0),
            axis=0,
        )
    return result


def fixed_blend(raw5: pd.DataFrame, v13: pd.DataFrame) -> pd.DataFrame:
    cols = ["sample_index", "label"]
    if "sample_id" in raw5.columns:
        cols.insert(1, "sample_id")
    result = raw5[cols].copy()
    for mode in MODES:
        result[f"{mode}_pred"] = (
            RAW5_WEIGHT * raw5[f"{mode}_pred"].to_numpy(float)
            + V13_WEIGHT * v13[f"{mode}_pred"].to_numpy(float)
        )
    return result


def select_anchor(frames: dict[int, pd.DataFrame]):
    rows = []
    by_seed = {}
    for seed in SEEDS:
        metrics, j = frame_metrics(frames[seed])
        rows.append({"Seed": int(seed), "J": float(j)})
        by_seed[seed] = {"J": float(j), "LAV_MAE": float(metrics["LAV"]["MAE"])}
    seed = select_anchor_seed(rows, SEEDS)
    return seed, by_seed


def project_dp57(anchor: pd.DataFrame, target: pd.DataFrame, dataset: str):
    cols = ["sample_index", "label"]
    if "sample_id" in target.columns:
        cols.insert(1, "sample_id")
    result = target[cols].copy()
    diagnostics = []
    for mode in MODES:
        anchor_values = anchor[f"{mode}_pred"].to_numpy(np.float32)
        target_values = target[f"{mode}_pred"].to_numpy(np.float32)
        projected, details = project_array(anchor_values, target_values, dataset, "adpep57")
        result[f"{mode}_pred"] = projected

        anchor_decisions = evaluator_decisions(anchor_values, dataset)
        projected_decisions = evaluator_decisions(projected, dataset)
        target_decisions = evaluator_decisions(target_values, dataset)
        acc7_mismatch = int(np.count_nonzero(anchor_decisions[0] != projected_decisions[0]))
        acc5_mismatch = int(np.count_nonzero(anchor_decisions[1] != projected_decisions[1]))
        if acc7_mismatch or acc5_mismatch:
            raise RuntimeError(f"DP57 decision preservation failed for {mode}")

        changed = projected != target_values
        sign_changed_from_target = projected_decisions[2] != target_decisions[2]
        diagnostics.append(
            {
                "Mode": mode,
                "N": int(len(projected)),
                "ProjectedCount": int(changed.sum()),
                "ProjectedRate": float(changed.mean()),
                "TargetAlreadyFeasibleRate": float(np.mean([d.pe5_already_feasible for d in details])),
                "BoundaryAdjustedCount": int(sum(d.boundary_adjusted for d in details)),
                "FallbackCount": int(sum(d.fallback_to_anchor for d in details)),
                "Acc7MismatchVsAnchor": acc7_mismatch,
                "Acc5MismatchVsAnchor": acc5_mismatch,
                "Acc2DecisionChangedVsFixedBlendCount": int(sign_changed_from_target.sum()),
                "Acc2DecisionChangedVsFixedBlendRate": float(sign_changed_from_target.mean()),
                "MeanAbsChangeVsFixedBlend": float(np.mean(np.abs(projected.astype(np.float64) - target_values.astype(np.float64)))),
            }
        )
    return result, pd.DataFrame(diagnostics)


def metric_rows(method: str, by_mode: dict, j: float):
    rows = []
    for mode in MODES + ("MissingMacro",):
        rows.append({"Method": method, "Mode": mode, "J": float(j), **by_mode[mode]})
    return rows


def main():
    args = parse_args()
    output = Path(args.output_root)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; inspect it or use --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    raw_frames = load_raw5(Path(args.raw5_root), args.split)
    anchor_seed, anchor_valid = select_anchor(raw_frames)
    anchor = raw_frames[anchor_seed]
    raw5 = raw5_mean(raw_frames)
    v13 = load_v13(Path(args.v13_predictions), raw_frames[SEEDS[0]], args.split)
    blend = fixed_blend(raw5, v13)
    projected, diagnostics = project_dp57(anchor, blend, args.dataset)

    methods = {
        "anchor_seed{}".format(anchor_seed): anchor,
        "raw5_pe5": raw5,
        "v13": v13,
        "raw5_0p5_v13_0p5_fixed_blend": blend,
        METHOD: projected,
    }
    rows = []
    summary_metrics = {}
    for method, frame in methods.items():
        by_mode, j = frame_metrics(frame)
        rows.extend(metric_rows(method, by_mode, j))
        summary_metrics[method] = {"J": j, **by_mode}

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "fixedblend_dp57_metrics.csv", index=False)
    diagnostics.to_csv(output / "fixedblend_dp57_projection_diagnostics.csv", index=False)

    anchor_name = "anchor_seed{}".format(anchor_seed)
    anchor_metrics = summary_metrics[anchor_name]
    dp_metrics = summary_metrics[METHOD]
    blend_metrics = summary_metrics["raw5_0p5_v13_0p5_fixed_blend"]

    exact_inheritance = {}
    for mode in MODES + ("MissingMacro",):
        exact_inheritance[mode] = {
            "acc7_equal_anchor": bool(abs(dp_metrics[mode]["acc_7"] - anchor_metrics[mode]["acc_7"]) <= 1e-12),
            "acc5_equal_anchor": bool(abs(dp_metrics[mode]["acc_5"] - anchor_metrics[mode]["acc_5"]) <= 1e-12),
        }

    summary = {
        "method": METHOD,
        "dataset": args.dataset,
        "split": args.split,
        "protocol": {
            "raw5_weight": RAW5_WEIGHT,
            "v13_weight": V13_WEIGHT,
            "anchor_selection": "argmin five fixed seeds on validation J, tie -> lower seed",
            "projection": "closest FixedBlend prediction inside anchor Acc7/Acc5 decision interval",
            "projection_reads_labels": False,
            "test_weight_search": False,
            "mosi_test_blocked": True,
            "sample_level_projection_written": False,
        },
        "anchor_seed": int(anchor_seed),
        "anchor_validation_summary": anchor_valid,
        "exact_acc7_acc5_inheritance": exact_inheritance,
        "fixedblend_J": float(blend_metrics["J"]),
        "dp57_J": float(dp_metrics["J"]),
        "delta_J_dp57_minus_fixedblend": float(dp_metrics["J"] - blend_metrics["J"]),
        "fixedblend_LAV": blend_metrics["LAV"],
        "dp57_LAV": dp_metrics["LAV"],
        "fixedblend_MissingMacro": blend_metrics["MissingMacro"],
        "dp57_MissingMacro": dp_metrics["MissingMacro"],
        "projection_diagnostics": diagnostics.to_dict("records"),
    }
    (output / "fixedblend_dp57_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("FixedBlend-DP57 audit complete")
    print("dataset/split:", args.dataset, args.split)
    print("anchor seed:", anchor_seed)
    print("FixedBlend J: {:.9f}".format(blend_metrics["J"]))
    print("DP57 J:       {:.9f}".format(dp_metrics["J"]))
    print("delta J:      {:+.9f}".format(dp_metrics["J"] - blend_metrics["J"]))
    print()
    print("LAV metrics: FixedBlend -> DP57 -> Anchor")
    for key in METRICS:
        print(
            "  {:8s} {:.9f} -> {:.9f} -> {:.9f}".format(
                key,
                float(blend_metrics["LAV"][key]),
                float(dp_metrics["LAV"][key]),
                float(anchor_metrics["LAV"][key]),
            )
        )
    print()
    print("MissingMacro: FixedBlend -> DP57 -> Anchor")
    for key in METRICS:
        print(
            "  {:8s} {:.9f} -> {:.9f} -> {:.9f}".format(
                key,
                float(blend_metrics["MissingMacro"][key]),
                float(dp_metrics["MissingMacro"][key]),
                float(anchor_metrics["MissingMacro"][key]),
            )
        )
    print()
    print("Exact Acc7/Acc5 inheritance:", all(
        item["acc7_equal_anchor"] and item["acc5_equal_anchor"]
        for item in exact_inheritance.values()
    ))
    print("sample-level projected predictions written: False")
    print("output:", output)


if __name__ == "__main__":
    main()
