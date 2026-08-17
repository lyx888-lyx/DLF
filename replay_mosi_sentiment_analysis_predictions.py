"""Reconstruct frozen MOSI sample-level predictions for post-hoc paper analysis.

This script exists only because several historical MOSI Test probes intentionally
persisted aggregate metrics but not sample-level predictions. It does not train,
select checkpoints, search weights, or calibrate anything. It replays already
frozen sources and writes analysis-only sample-level prediction CSVs so that the
sentiment-region / missing-modality robustness analysis can be performed.

IMPORTANT
---------
MOSI Test has already been historically accessed in this repository. Therefore
these outputs are *post-hoc analysis artifacts*, not a pristine final Test.
Running this script requires an explicit acknowledgement flag.

Outputs include:
  * clean DLF teacher under direct masking (LAV/LA/LV/L),
  * frozen Stage-1 ModDrop evaluator (LAV/LA/LV/L),
  * Raw5 equal-weight CFCompat ensemble (LAV/LA/LV/L),
  * frozen v13 consensus predictor (LAV/LA/LV/L),
  * frozen Raw5/v13 FixedBlend (LAV/LA/LV/L),
  * an aggregate metric / main-table identity audit.

No result from this script may be used to retune a model, choose a checkpoint,
or search a blend weight.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch

from data_loader import MMDataLoader
from evaluate_cfcompat_exploratory_test_viability_v13 import (
    DEV_SEED,
    EXPECTED_TEST_N,
    collect_model_once,
    collect_reference,
    load_consensus_model,
    load_frozen_paths,
    load_v8_model,
)
from evaluate_cfcompat_pe5_v13_fixedblend_test_v1 import (
    blend_frames,
    frame_metrics,
    historical_v13_row,
    load_raw5_test,
    verify_v13_replay,
)
from train_cf_compat_kd import build_config
from trains.singleTask.cf_compat_kd_utils import build_frozen_evaluator
from trains.singleTask.cfcompat_crossfit_residual_consensus_utils import module_state_sha256
from trains.singleTask.fixed_kd_utils import build_frozen_teacher, checkpoint_sha256
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    apply_direct_mask,
    mode_to_mask,
    regression_metrics,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


VERSION = "sentiment_region_missing_robustness_replay_v1"
MODES: Tuple[str, ...] = ("LAV",) + MISSING_MODES

# These values are used only as a descriptive identity audit. They never gate,
# select, tune, or modify a prediction. Acc values are stored in [0, 1].
PAPER_TABLE_SIGNATURES = {
    "DLF_table_row": {
        "acc_7": 0.4708,
        "acc_5": 0.5233,
        "acc_2": 0.8506,
        "F1_score": 0.8504,
        "Corr": 0.781,
        "MAE": 0.731,
    },
    "Ours_table_row": {
        "acc_7": 0.4947,
        "acc_5": 0.5510,
        "acc_2": 0.8506,
        "F1_score": 0.8501,
        "Corr": 0.802,
        "MAE": 0.693,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay frozen MOSI predictions for post-hoc sentiment-region analysis"
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--acknowledge-posthoc-test-analysis",
        action="store_true",
        help="Required: MOSI Test is historically accessed; outputs are post-hoc only.",
    )
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("This replay fixes --num-workers=1 to match the historical v13 Test path.")
    if not args.acknowledge_posthoc_test_analysis:
        parser.error(
            "Pass --acknowledge-posthoc-test-analysis. This writes post-hoc MOSI Test "
            "sample predictions from already-frozen models and must not be used for tuning."
        )
    return args


def output_root(cli: argparse.Namespace) -> Path:
    root = (
        Path(cli.result_root)
        / "posthoc_analysis"
        / VERSION
        / cli.dataset
        / "test"
    )
    if root.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "Post-hoc replay output already exists. Inspect it instead of rerunning, "
                "or pass --overwrite for a conscious technical replay: {}".format(root)
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"sample_index", "label", *[f"{mode}_pred" for mode in MODES]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("{} lacks columns {}".format(name, missing))
    local = frame.copy().sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(local) != EXPECTED_TEST_N or local.sample_index.nunique() != EXPECTED_TEST_N:
        raise RuntimeError(
            "{} expected {} unique MOSI Test samples, got {} / {}".format(
                name, EXPECTED_TEST_N, len(local), local.sample_index.nunique()
            )
        )
    numeric = local[["label", *[f"{mode}_pred" for mode in MODES]]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("{} contains NaN/Inf predictions.".format(name))
    return local


def _assert_same_samples(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    left = reference.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    right = other.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(
        left.sample_index.to_numpy(np.int64), right.sample_index.to_numpy(np.int64)
    ):
        raise RuntimeError("sample_index binding differs for {}.".format(name))
    if np.max(np.abs(left.label.to_numpy(float) - right.label.to_numpy(float))) > 1e-7:
        raise RuntimeError("label binding differs for {}.".format(name))


def collect_clean_dlf_directmask(teacher, loader, args) -> pd.DataFrame:
    """Evaluate the frozen clean DLF under test-time direct modality zeroing.

    LAV is the ordinary clean-teacher prediction. LA/LV/L are deliberately
    *not* missing-aware retraining: audio/vision are zeroed according to the
    requested mode and the same frozen complete-modality DLF is evaluated.
    This is useful for visualizing cliff-like degradation of a conventional
    complete-modality model when inputs disappear.
    """
    teacher.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels = batch["labels"]["M"].view(-1).cpu().numpy().astype(np.float64)
            indices = batch["index"].view(-1).cpu().numpy().astype(np.int64)
            by_mode = {}
            for mode in MODES:
                mask = mode_to_mask(
                    mode,
                    batch_size=text.size(0),
                    device=args.device,
                    dtype=audio.dtype,
                )
                input_audio, input_vision = apply_direct_mask(audio, vision, mask)
                pred = teacher(text, input_audio, input_vision)["output_logit"]
                by_mode[mode] = pred.detach().view(-1).cpu().numpy()
            for offset, index in enumerate(indices):
                rows.append(
                    {
                        "sample_index": int(index),
                        "label": float(labels[offset]),
                        **{
                            f"{mode}_pred": float(by_mode[mode][offset])
                            for mode in MODES
                        },
                    }
                )
    return _validate_frame(pd.DataFrame(rows), "clean_dlf_directmask")


def moddrop_frame_from_reference(reference: pd.DataFrame) -> pd.DataFrame:
    result = reference[["sample_index", "label"]].copy()
    for mode in MODES:
        result[f"{mode}_pred"] = reference[f"baseline_{mode}_pred"].to_numpy(float)
    return _validate_frame(result, "moddrop_evaluator")


def teacher_lav_from_reference(reference: pd.DataFrame) -> pd.DataFrame:
    """A LAV-only convenience artifact for checking the exact clean teacher."""
    return reference[["sample_index", "label", "teacher_prediction"]].rename(
        columns={"teacher_prediction": "LAV_pred"}
    )


def metrics_for_frame(frame: pd.DataFrame) -> Dict[str, object]:
    frame = _validate_frame(frame, "metrics_frame")
    labels = torch.as_tensor(frame.label.to_numpy(np.float32)).view(-1, 1)
    by_mode = {}
    for mode in MODES:
        pred = torch.as_tensor(frame[f"{mode}_pred"].to_numpy(np.float32)).view(-1, 1)
        by_mode[mode] = regression_metrics(pred, labels)
    return {
        "J": float(validation_objective(by_mode)),
        "MissingMacroMAE": float(np.mean([by_mode[m]["MAE"] for m in MISSING_MODES])),
        "by_mode": by_mode,
    }


def lav_signature(metrics: Dict[str, object]) -> Dict[str, float]:
    lav = metrics["by_mode"]["LAV"]
    return {
        "acc_7": float(lav["acc_7"]),
        "acc_5": float(lav["acc_5"]),
        "acc_2": float(lav["acc_2"]),
        "F1_score": float(lav["F1_score"]),
        "Corr": float(lav["Corr"]),
        "MAE": float(lav["MAE"]),
    }


def display_signature(signature: Dict[str, float]) -> Dict[str, float]:
    return {
        "Acc7": round(100.0 * signature["acc_7"], 2),
        "Acc5": round(100.0 * signature["acc_5"], 2),
        "Acc2": round(100.0 * signature["acc_2"], 2),
        "F1": round(100.0 * signature["F1_score"], 2),
        "Corr": round(signature["Corr"], 3),
        "MAE": round(signature["MAE"], 3),
    }


def signature_audit(signature: Dict[str, float]) -> Dict[str, object]:
    result = {}
    for label, expected in PAPER_TABLE_SIGNATURES.items():
        deltas = {key: float(signature[key] - expected[key]) for key in expected}
        result[label] = {
            "max_abs_native_delta": float(max(abs(value) for value in deltas.values())),
            "deltas": deltas,
            "expected_display": display_signature(expected),
        }
    return result


def _write_frame(root: Path, name: str, frame: pd.DataFrame) -> Path:
    path = root / name
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    cli = parse_args()
    root = output_root(cli)

    print("POSTHOC_MOSI_TEST_ANALYSIS_REPLAY")
    print("TEST_ALREADY_HISTORICALLY_ACCESSED")
    print("NO_TRAINING_NO_SELECTION_NO_WEIGHT_SEARCH")
    print("SAMPLE_LEVEL_OUTPUTS_ARE_ANALYSIS_ONLY")

    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    args.mode = "test"
    args.is_training = False
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"test"}:
        raise RuntimeError("Replay must construct exactly the MOSI Test split.")
    test_loader = loaders["test"]
    if len(test_loader.dataset) != EXPECTED_TEST_N:
        raise RuntimeError(
            "Unexpected MOSI Test size {} (expected {}).".format(
                len(test_loader.dataset), EXPECTED_TEST_N
            )
        )

    bindings = load_frozen_paths(cli)

    # 1) Frozen clean DLF teacher and Stage-1 ModDrop evaluator.
    teacher = build_frozen_teacher(
        DLF, args, bindings["original_cfcompat_v1"]["teacher_checkpoint"]
    )
    evaluator = build_frozen_evaluator(
        DLF, args, bindings["original_cfcompat_v1"]["evaluator_checkpoint"]
    )
    reference = collect_reference(evaluator, teacher, test_loader, args)
    clean_dlf = collect_clean_dlf_directmask(teacher, test_loader, args)
    moddrop = moddrop_frame_from_reference(reference)
    teacher_lav = teacher_lav_from_reference(reference)

    # Direct-mask LAV must be exactly the same clean Teacher forward (within
    # ordinary floating-point replay tolerance).
    joined = clean_dlf[["sample_index", "LAV_pred"]].merge(
        teacher_lav[["sample_index", "LAV_pred"]],
        on="sample_index",
        suffixes=("_direct", "_teacher"),
        validate="one_to_one",
    )
    max_teacher_lav_diff = float(
        np.max(np.abs(joined.LAV_pred_direct.to_numpy(float) - joined.LAV_pred_teacher.to_numpy(float)))
    )
    if max_teacher_lav_diff > 2e-6:
        raise RuntimeError(
            "Clean DLF direct-mask LAV does not replay frozen Teacher: max diff={}".format(
                max_teacher_lav_diff
            )
        )
    del teacher, evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) Raw5 is reconstructed offline from the five frozen member CSVs.
    raw5, raw5_manifest = load_raw5_test(cli)
    raw5 = _validate_frame(raw5, "raw5")
    _assert_same_samples(clean_dlf, raw5, "Raw5")

    # 3) Replay the already-frozen v13 consensus exactly once.
    historical_path, historical_v13 = historical_v13_row(cli)
    v8_model = load_v8_model(args, bindings["sample_residual_v8"]["checkpoint"])
    expected_s0_sha = str(
        bindings["v13_summary"].get("parameter_isolation", {}).get(
            "s0_state_sha256_before", ""
        )
    )
    actual_s0_sha = module_state_sha256(v8_model.s0)
    if expected_s0_sha and actual_s0_sha != expected_s0_sha:
        raise RuntimeError("v8-carried S0 hash differs from frozen v13 S0.")
    s0_state = {
        key: value.detach().cpu().clone()
        for key, value in v8_model.s0.state_dict().items()
    }
    del v8_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    v13_model = load_consensus_model(
        args, s0_state, bindings["adam_step_safety_v13"]["checkpoints"]
    )
    v13, v13_flat_metrics = collect_model_once(v13_model, test_loader, args)
    del v13_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    v13 = _validate_frame(v13, "v13")
    _assert_same_samples(clean_dlf, v13, "v13")
    replay_diffs = verify_v13_replay(v13_flat_metrics, historical_v13)

    # 4) Frozen FixedBlend composition. No weight search is performed here.
    fixedblend = _validate_frame(blend_frames(raw5, v13), "fixedblend")
    _assert_same_samples(clean_dlf, fixedblend, "FixedBlend")

    frames = {
        "clean_dlf_directmask": clean_dlf,
        "moddrop_evaluator": moddrop,
        "raw5_pe5": raw5,
        "v13_auxiliary": v13,
        "fixedblend_raw5_v13": fixedblend,
    }
    metrics = {name: metrics_for_frame(frame) for name, frame in frames.items()}

    paths = {
        "clean_dlf_directmask": _write_frame(
            root, "clean_dlf_directmask_test_predictions.csv", clean_dlf
        ),
        "clean_dlf_teacher_lav": _write_frame(
            root, "clean_dlf_teacher_lav_test_predictions.csv", teacher_lav
        ),
        "moddrop_evaluator": _write_frame(
            root, "moddrop_evaluator_test_predictions.csv", moddrop
        ),
        "raw5_pe5": _write_frame(root, "raw5_pe5_test_predictions.csv", raw5),
        "v13_auxiliary": _write_frame(root, "v13_auxiliary_test_predictions.csv", v13),
        "fixedblend_raw5_v13": _write_frame(
            root, "fixedblend_raw5_v13_test_predictions.csv", fixedblend
        ),
    }

    metric_rows = []
    identity_audit = {}
    for name, values in metrics.items():
        signature = lav_signature(values)
        identity_audit[name] = {
            "lav_native": signature,
            "lav_display": display_signature(signature),
            "paper_table_audit": signature_audit(signature),
        }
        row = {
            "Method": name,
            "J": float(values["J"]),
            "MissingMacroMAE": float(values["MissingMacroMAE"]),
        }
        for mode in MODES:
            for key, value in values["by_mode"][mode].items():
                row[f"{mode}_{key}"] = float(value)
        metric_rows.append(row)
    metrics_path = root / "replayed_method_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    summary = {
        "version": VERSION,
        "dataset": cli.dataset,
        "split": "test",
        "status": "POSTHOC_ANALYSIS_ONLY_TEST_ALREADY_HISTORICALLY_ACCESSED",
        "protocol": {
            "training": False,
            "checkpoint_selection": False,
            "calibration": False,
            "blend_weight_search": False,
            "raw5_new_model_forward": False,
            "v13_new_frozen_model_forward_count": 1,
            "clean_teacher_forward_for_analysis": True,
            "moddrop_evaluator_forward_for_analysis": True,
            "sample_level_predictions_written": True,
            "sample_level_predictions_must_not_be_used_for_tuning": True,
        },
        "teacher_lav_replay_max_abs_diff": max_teacher_lav_diff,
        "v13_historical_replay_abs_differences": replay_diffs,
        "sources": {
            "teacher_checkpoint": str(
                Path(bindings["original_cfcompat_v1"]["teacher_checkpoint"]).resolve()
            ),
            "teacher_checkpoint_sha256": checkpoint_sha256(
                bindings["original_cfcompat_v1"]["teacher_checkpoint"]
            ),
            "moddrop_evaluator_checkpoint": str(
                Path(bindings["original_cfcompat_v1"]["evaluator_checkpoint"]).resolve()
            ),
            "moddrop_evaluator_checkpoint_sha256": checkpoint_sha256(
                bindings["original_cfcompat_v1"]["evaluator_checkpoint"]
            ),
            "raw5_members": raw5_manifest,
            "v13_historical_aggregate_source": str(historical_path.resolve()),
            "v13_fold_manifest": str(
                bindings["adam_step_safety_v13"]["manifest"].resolve()
            ),
        },
        "prediction_csvs": {name: str(path.resolve()) for name, path in paths.items()},
        "metrics_csv": str(metrics_path.resolve()),
        "main_table_identity_audit": identity_audit,
    }
    summary_path = root / "replay_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nLAV identity audit (descriptive only; no selection/tuning):")
    for name in frames:
        display = identity_audit[name]["lav_display"]
        dlf_delta = identity_audit[name]["paper_table_audit"]["DLF_table_row"][
            "max_abs_native_delta"
        ]
        ours_delta = identity_audit[name]["paper_table_audit"]["Ours_table_row"][
            "max_abs_native_delta"
        ]
        print(
            "  {:24s} {}  maxDelta(DLF)={:.6f} maxDelta(Ours)={:.6f}".format(
                name, display, dlf_delta, ours_delta
            )
        )
    print("\nWritten analysis-only predictions:")
    for name, path in paths.items():
        print("  {}: {}".format(name, path))
    print("metrics:", metrics_path)
    print("summary:", summary_path)
    print("REMINDER: Do not use these post-hoc Test rows for model/weight/checkpoint tuning.")

    del test_loader, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
