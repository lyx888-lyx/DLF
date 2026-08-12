"""Narrow execution hardening for the one-shot MOSEI final-Test evaluator.

This wrapper changes no frozen method decision.  It only:

1. moves each completed model back to CPU before CUDA cache cleanup so sequential
   8 GiB evaluation cannot transiently retain the previous GPU model through a
   caller reference; and
2. reproduces the already-audited Raw5 arithmetic path exactly as on Valid:
   float32 member predictions -> float64 mean -> float32 ensemble prediction.

The runner invokes this wrapper rather than the implementation module directly.
"""
from __future__ import annotations

import gc

import numpy as np
import torch

import evaluate_mosei_fixedblend_dp57_final_test as impl
from trains.singleTask.anchor_decision_projection import evaluator_decisions, project_array


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


impl.release_model = _release_model_to_cpu
impl.compose_frozen_method = _compose_frozen_method_valid_arithmetic


if __name__ == "__main__":
    impl.main()
