"""Execution hardening for the one-shot MOSEI final-Test evaluator.

This wrapper changes no frozen method decision.  It only:

1. moves each completed model back to CPU before CUDA cache cleanup so sequential
   8 GiB evaluation cannot transiently retain the previous GPU model through a
   caller reference;
2. reproduces the already-audited Raw5 arithmetic path exactly as on Valid:
   float32 member predictions -> float64 mean -> float32 ensemble prediction;
3. before Official Test can be constructed, dry-runs the exact final inference
   and composition path on Valid, requiring every checkpoint prediction to match
   its frozen Valid artifact and every frozen Valid J to replay.

The runner invokes this wrapper rather than the implementation module directly.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_mosei_fixedblend_dp57_final_test as impl
from trains.singleTask.anchor_decision_projection import evaluator_decisions, project_array


VALID_PRED_TOL = 1e-6
VALID_LABEL_TOL = 1e-7


def _release_model_to_cpu(model) -> None:
    # Mutating the caller-owned module onto CPU releases CUDA tensors even while
    # the caller still holds a Python reference.  This avoids RHS-before-LHS
    # assignment overlap when the next large DLF model is constructed.
    model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compose_frozen_method_valid_arithmetic(member_predictions, v13_predictions):
    raw5 = {}
    for mode in impl.MODES:
        stacked = np.stack(
            [
                np.asarray(member_predictions[seed][mode], dtype=np.float32).astype(
                    np.float64
                )
                for seed in impl.SEEDS
            ],
            axis=0,
        )
        raw5[mode] = np.mean(stacked, axis=0, dtype=np.float64).astype(np.float32)

    fixedblend = {
        mode: (
            impl.BLEND_WEIGHT_RAW5 * raw5[mode]
            + impl.BLEND_WEIGHT_V13
            * np.asarray(v13_predictions[mode], dtype=np.float32)
        ).astype(np.float32)
        for mode in impl.MODES
    }
    anchor = {
        mode: np.asarray(
            member_predictions[impl.FROZEN_ANCHOR_SEED][mode], dtype=np.float32
        )
        for mode in impl.MODES
    }

    dp57 = {}
    projection_summary = {}
    for mode in impl.MODES:
        projected, results = project_array(
            anchor[mode], fixedblend[mode], impl.DATASET, impl.DP_VARIANT
        )
        dp57[mode] = projected.astype(np.float32)
        anchor7, anchor5, _ = evaluator_decisions(anchor[mode], impl.DATASET)
        projected7, projected5, _ = evaluator_decisions(dp57[mode], impl.DATASET)
        if not np.array_equal(anchor7, projected7):
            raise RuntimeError(
                f"Final Test DP57 Acc7 inheritance failed mode={mode}."
            )
        if not np.array_equal(anchor5, projected5):
            raise RuntimeError(
                f"Final Test DP57 Acc5 inheritance failed mode={mode}."
            )
        _, _, blend2 = evaluator_decisions(fixedblend[mode], impl.DATASET)
        _, _, dp2 = evaluator_decisions(dp57[mode], impl.DATASET)
        feasible = np.asarray(
            [result.pe5_already_feasible for result in results], dtype=bool
        )
        boundary = np.asarray(
            [result.boundary_adjusted for result in results], dtype=bool
        )
        fallback = np.asarray(
            [result.fallback_to_anchor for result in results], dtype=bool
        )
        projection_summary[mode] = {
            "N": int(len(results)),
            "projected_fraction": float((~feasible).mean()),
            "target_feasible_fraction": float(feasible.mean()),
            "boundary_adjusted_count": int(boundary.sum()),
            "fallback_count": int(fallback.sum()),
            "acc2_changed_vs_fixedblend_count": int(np.sum(blend2 != dp2)),
        }
    return anchor, raw5, fixedblend, dp57, projection_summary


def _load_frozen_valid_frame(path: Path, expected_n: int = 1871):
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    required = {"sample_index", "label"} | {
        f"{mode}_pred" for mode in impl.MODES
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Frozen Valid prediction lacks columns {sorted(missing)}: {path}")
    if len(frame) != expected_n or frame.sample_index.nunique() != expected_n:
        raise RuntimeError(f"Frozen Valid prediction row count drifted: {path}")
    if not np.array_equal(
        frame.sample_index.to_numpy(np.int64), np.arange(expected_n, dtype=np.int64)
    ):
        raise RuntimeError(f"Frozen Valid sample_index drifted: {path}")
    if "Split" in frame.columns and set(frame.Split.astype(str).str.lower()) != {"valid"}:
        raise RuntimeError(f"Frozen Valid artifact contains non-Valid rows: {path}")
    return frame


def _assert_valid_prediction_replay(predictions, labels, frame, tag: str) -> None:
    recorded_labels = frame.label.to_numpy(np.float32)
    label_diff = float(
        np.max(
            np.abs(
                np.asarray(labels, dtype=np.float32).astype(np.float64)
                - recorded_labels.astype(np.float64)
            )
        )
    )
    if label_diff > VALID_LABEL_TOL:
        raise RuntimeError(f"{tag} Valid label replay drifted: max_abs={label_diff}")
    for mode in impl.MODES:
        recorded = frame[f"{mode}_pred"].to_numpy(np.float32)
        current = np.asarray(predictions[mode], dtype=np.float32)
        if current.shape != recorded.shape:
            raise RuntimeError(f"{tag} Valid prediction shape drifted mode={mode}")
        max_diff = float(
            np.max(np.abs(current.astype(np.float64) - recorded.astype(np.float64)))
        )
        if max_diff > VALID_PRED_TOL:
            raise RuntimeError(
                f"{tag} Valid prediction replay drifted mode={mode}: max_abs={max_diff}"
            )


def _valid_pipeline_replay(cli, args, sources) -> None:
    # Explicitly Valid-only.  Official Test is not constructed by this audit.
    valid_loader = impl.build_single_split_loader(args, "valid", cli.num_workers)
    batches, labels, _ = impl.cache_official_test_once(valid_loader)
    if len(labels) != 1871:
        raise RuntimeError(f"MOSEI Valid count drifted during final preflight: {len(labels)}")
    del valid_loader

    raw_root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_prediction_ensemble_v1"
        / impl.DATASET
    )
    member_predictions = {}
    for seed in impl.SEEDS:
        model = impl.build_cfcompat_model(
            args, Path(sources["cfcompat"][seed]["checkpoint"])
        )
        current = impl.infer_cached_batches(model, batches, args.device)
        impl.release_model(model)
        frame = _load_frozen_valid_frame(
            raw_root / f"online_seed{seed}_valid_predictions.csv"
        )
        _assert_valid_prediction_replay(current, labels, frame, f"CFCompat seed{seed}")
        member_predictions[seed] = current

    model = impl.build_v13_model(args, Path(sources["v13"]["checkpoint"]))
    v13_predictions = impl.infer_cached_batches(model, batches, args.device)
    impl.release_model(model)
    v13_frame = _load_frozen_valid_frame(
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_adam_step_safety_v13"
        / impl.DATASET
        / "valid_screen"
        / "seed1113_dev"
        / "v13_valid_predictions.csv"
    )
    _assert_valid_prediction_replay(v13_predictions, labels, v13_frame, "formal v13")

    anchor, raw5, fixedblend, dp57, _ = impl.compose_frozen_method(
        member_predictions, v13_predictions
    )
    replay_sets = {
        "Anchor": anchor,
        "Raw5": raw5,
        "v13": v13_predictions,
        "FixedBlend": fixedblend,
        "DP57": dp57,
    }
    for name, predictions in replay_sets.items():
        _, observed_j = impl.evaluate_prediction_set(predictions, labels)
        expected_j = impl.FROZEN_VALID_J[name]
        if abs(float(observed_j) - float(expected_j)) > impl.VALID_J_TOL:
            raise RuntimeError(
                f"Final pipeline Valid replay drifted {name}: {observed_j} != {expected_j}"
            )

    del batches, labels, member_predictions, v13_predictions
    del anchor, raw5, fixedblend, dp57
    gc.collect()


_original_verify_sources = impl.verify_sources


def _verify_sources_with_full_valid_replay(cli, args, reconstruct_models: bool):
    # The Valid pipeline replay itself strict-loads every model, so avoid doing
    # a second redundant reconstruction pass in the underlying verifier.
    sources = _original_verify_sources(cli, args, reconstruct_models=False)
    if reconstruct_models:
        _valid_pipeline_replay(cli, args, sources)
    return sources


impl.release_model = _release_model_to_cpu
impl.compose_frozen_method = _compose_frozen_method_valid_arithmetic
impl.verify_sources = _verify_sources_with_full_valid_replay


if __name__ == "__main__":
    impl.main()
