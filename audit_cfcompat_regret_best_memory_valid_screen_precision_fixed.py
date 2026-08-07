"""Precision-corrected independent audit for best-memory CFCompatKD v4.2.

The original v4.2 audit replays memory state in Python/NumPy float64 while the
training route receives memory/Student/Teacher/label tensors in float32.  A
1e-7 comparison can therefore reject a scientifically identical route.  It
also initialized final-memory validity from the route flag, so one float route
mismatch forced the final-memory check to fail without independently checking
final state.

This erratum audit preserves every non-memory check from the original audit,
then independently replays:
  1. persistent best-so-far state using the exact float32 values that were
     promoted to Python float64 by the training memory implementation;
  2. routing using float32 arithmetic, matching the tensors used in training;
  3. final memory independently from route-check success.
No training, validation selection, or Test access occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_regret_best_memory_utils import (
    MEMORY_UPDATE_EPS,
    MILD_CFCOMPAT_BASE_V4P2,
    MILD_CFCOMPAT_SCALE_V4P2,
    PRESERVE_MARGIN_V4P2,
    STRONG_DISTILL_MARGIN,
    WEAK_DISTILL_SCALE,
)
from trains.singleTask.missing_utils import MISSING_MODES


PREFIX = "regret_best_memory_v4p2"
FLOAT32_TOLERANCE = 1e-6
STATE_TOLERANCE = 1e-7
PROJECTION_TOLERANCE = 1e-12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def f32(value):
    return np.float32(value)


def close32(recorded, expected):
    return abs(float(recorded) - float(f32(expected))) <= FLOAT32_TOLERANCE


def clip32(value, lower, upper):
    return f32(np.maximum(np.minimum(f32(value), f32(upper)), f32(lower)))


def as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError("Cannot parse boolean value: {!r}".format(value))


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    original_audit_path = root / f"{PREFIX}_audit_check.json"
    baseline_path = root / f"{PREFIX}_train_baseline_cache.csv"
    decisions_path = root / f"{PREFIX}_train_decisions.csv"
    final_memory_path = root / f"{PREFIX}_final_memory.csv"
    required = [original_audit_path, baseline_path, decisions_path, final_memory_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing v4.2 precision-audit artifacts:\n" + "\n".join(missing))

    original = json.loads(original_audit_path.read_text(encoding="utf-8"))
    baseline = pd.read_csv(baseline_path)
    decisions = pd.read_csv(decisions_path)
    final_memory = pd.read_csv(final_memory_path)

    exempt = {"best_memory_and_routes_recomputed", "final_memory_recomputed"}
    inherited_checks = {
        key: bool(value)
        for key, value in original.get("checks", {}).items()
        if key not in exempt
    }
    inherited_ok = bool(inherited_checks) and all(inherited_checks.values())

    # Persistent state exactly mirrors BestSoFarPredictionMemory: values come
    # from float32 tensors, are promoted to Python/NumPy float64, then retained
    # as Python floats in the CPU dictionary.
    state = {}
    for row in baseline.itertuples(index=False):
        label64 = float(f32(row.label))
        for mode in MISSING_MODES:
            pred64 = float(f32(getattr(row, "baseline_{}_pred".format(mode))))
            error64 = abs(pred64 - label64)
            state[(int(row.sample_index), str(mode))] = [pred64, error64, 0, 0]

    ordered = decisions.sort_values("event_ordinal", kind="mergesort").reset_index(drop=True)
    memory_state_ok = True
    route_ok = True
    first_state_mismatch = None
    first_route_mismatch = None

    expected_ordinals = np.arange(1, len(ordered) + 1, dtype=np.int64)
    if not np.array_equal(ordered.event_ordinal.to_numpy(dtype=np.int64), expected_ordinals):
        memory_state_ok = False
        first_state_mismatch = {"reason": "event_ordinal_not_contiguous"}

    if memory_state_ok:
        for row in ordered.itertuples(index=False):
            key = (int(row.sample_index), str(row.mode))
            if key not in state:
                memory_state_ok = False
                first_state_mismatch = {"event": int(row.event_ordinal), "reason": "unknown_key", "key": list(key)}
                break

            before_pred, before_err, update_count, last_event = state[key]
            current64 = float(f32(row.student_prediction))
            label64 = float(f32(row.label))
            current_err64 = abs(current64 - label64)
            do_update = bool(current_err64 + MEMORY_UPDATE_EPS < before_err)
            after_pred = current64 if do_update else before_pred
            after_err = current_err64 if do_update else before_err
            after_count = update_count + (1 if do_update else 0)
            after_last = int(row.event_ordinal) if do_update else last_event

            state_comparisons = {
                "memory_before_prediction": close32(row.memory_before_prediction, before_pred),
                "memory_before_error": close32(row.memory_before_error, before_err),
                "memory_updated": as_bool(row.memory_updated) == do_update,
                "memory_after_prediction": close32(row.memory_after_prediction, after_pred),
                "memory_after_error": close32(row.memory_after_error, after_err),
                "memory_improvement": close32(row.memory_improvement, before_err - after_err),
                "memory_update_count_after": int(row.memory_update_count_after) == int(after_count),
            }
            if not all(state_comparisons.values()):
                memory_state_ok = False
                first_state_mismatch = {
                    "event": int(row.event_ordinal),
                    "key": list(key),
                    "comparisons": state_comparisons,
                    "expected": {
                        "before_prediction_f32": float(f32(before_pred)),
                        "before_error_f32": float(f32(before_err)),
                        "updated": do_update,
                        "after_prediction_f32": float(f32(after_pred)),
                        "after_error_f32": float(f32(after_err)),
                        "update_count_after": int(after_count),
                    },
                }
                break

            # Routing is performed by torch float32 tensors, not by the Python
            # float64 dictionary values above. Recompute with float32 semantics.
            student32 = f32(row.student_prediction)
            teacher32 = f32(row.teacher_prediction)
            label32 = f32(row.label)
            memory32 = f32(after_pred)
            compat32 = f32(row.compatibility)

            lower32 = f32(np.minimum(student32, label32))
            upper32 = f32(np.maximum(student32, label32))
            teacher_safe32 = clip32(teacher32, lower32, upper32)
            preserve_safe32 = clip32(memory32, lower32, upper32)
            active = not np.isclose(
                teacher_safe32, student32,
                atol=PROJECTION_TOLERANCE, rtol=0.0,
            )

            memory_error32 = f32(np.abs(f32(memory32 - label32)))
            teacher_error32 = f32(np.abs(f32(teacher32 - label32)))
            current_error32 = f32(np.abs(f32(student32 - label32)))
            advantage32 = f32(memory_error32 - teacher_error32)
            regret32 = f32(current_error32 - memory_error32)
            direction_product32 = f32(f32(teacher32 - memory32) * f32(label32 - memory32))
            direction_correct = bool(direction_product32 > f32(0.0))
            teacher_better = bool(advantage32 > f32(0.0))
            strong_candidate = bool(advantage32 >= f32(STRONG_DISTILL_MARGIN))
            weak_candidate = bool(
                teacher_better
                and advantage32 < f32(STRONG_DISTILL_MARGIN)
                and direction_correct
            )
            strong = bool(strong_candidate and active)
            weak = bool(weak_candidate and active)
            current_regressed = bool(regret32 >= f32(PRESERVE_MARGIN_V4P2))
            preserve = bool((not teacher_better) and current_regressed)
            abstain = bool(not (strong or weak or preserve))
            mild32 = f32(
                f32(MILD_CFCOMPAT_BASE_V4P2)
                + f32(MILD_CFCOMPAT_SCALE_V4P2) * compat32
            )
            strong_gate32 = mild32 if strong else f32(0.0)
            weak_gate32 = mild32 if weak else f32(0.0)
            eligible32 = f32(strong_gate32 + weak_gate32)
            effective32 = f32(strong_gate32 + f32(WEAK_DISTILL_SCALE) * weak_gate32)
            preserve_gate32 = f32(1.0 if preserve else 0.0)

            route_comparisons = {
                "memory_error": close32(row.memory_error, memory_error32),
                "teacher_error": close32(row.teacher_error, teacher_error32),
                "current_error": close32(row.current_error, current_error32),
                "teacher_advantage_vs_memory": close32(row.teacher_advantage_vs_memory, advantage32),
                "current_regret_vs_memory": close32(row.current_regret_vs_memory, regret32),
                "strong_distill": as_bool(row.strong_distill) == strong,
                "weak_distill": as_bool(row.weak_distill) == weak,
                "preserve": as_bool(row.preserve) == preserve,
                "memory_abstain": as_bool(row.memory_abstain) == abstain,
                "teacher_safe_target": close32(row.teacher_safe_target, teacher_safe32),
                "preserve_safe_target": close32(row.preserve_safe_target, preserve_safe32),
                "mild_compatibility": close32(row.mild_compatibility, mild32),
                "strong_gate": close32(row.strong_gate, strong_gate32),
                "weak_gate": close32(row.weak_gate, weak_gate32),
                "eligible_distill_mass": close32(row.eligible_distill_mass, eligible32),
                "effective_distill_gate": close32(row.effective_distill_gate, effective32),
                "preserve_gate": close32(row.preserve_gate, preserve_gate32),
            }
            if not all(route_comparisons.values()) and first_route_mismatch is None:
                route_ok = False
                first_route_mismatch = {
                    "event": int(row.event_ordinal),
                    "key": list(key),
                    "comparisons": route_comparisons,
                    "expected": {
                        "memory_error_f32": float(memory_error32),
                        "teacher_error_f32": float(teacher_error32),
                        "current_error_f32": float(current_error32),
                        "advantage_f32": float(advantage32),
                        "regret_f32": float(regret32),
                        "strong": strong,
                        "weak": weak,
                        "preserve": preserve,
                        "abstain": abstain,
                    },
                }

            state[key] = [after_pred, after_err, after_count, after_last]

    # Final-memory validity is intentionally independent from route_ok. The
    # original auditor coupled these checks and therefore reported two failures
    # from one precision mismatch.
    final_ok = memory_state_ok
    first_final_mismatch = None
    if final_ok:
        final_indexed = final_memory.set_index(["sample_index", "mode"])
        for key, (prediction, error, count, last_event) in state.items():
            if key not in final_indexed.index:
                final_ok = False
                first_final_mismatch = {"key": list(key), "reason": "missing_final_key"}
                break
            row = final_indexed.loc[key]
            comparisons = {
                "best_prediction": abs(float(row.best_prediction) - float(prediction)) <= STATE_TOLERANCE,
                "best_error": abs(float(row.best_error) - float(error)) <= STATE_TOLERANCE,
                "update_count": int(row.update_count) == int(count),
                "last_update_event": int(row.last_update_event) == int(last_event),
                "nonincreasing_vs_initial": float(row.best_error) <= float(row.initial_error) + STATE_TOLERANCE,
            }
            if not all(comparisons.values()):
                final_ok = False
                first_final_mismatch = {
                    "key": list(key),
                    "comparisons": comparisons,
                    "expected": {
                        "best_prediction": float(prediction),
                        "best_error": float(error),
                        "update_count": int(count),
                        "last_update_event": int(last_event),
                    },
                }
                break

    corrected_checks = {
        "all_original_non_memory_checks_passed": inherited_ok,
        "best_memory_state_recomputed": bool(memory_state_ok),
        "best_memory_routes_recomputed_float32": bool(memory_state_ok and route_ok),
        "final_memory_recomputed_independently": bool(final_ok),
    }
    passed = bool(all(corrected_checks.values()))
    payload = {
        "audit_erratum": "v4p2_float32_route_and_independent_final_memory",
        "passed": passed,
        "scientific_verdict": original.get("verdict"),
        "original_audit_passed": bool(original.get("passed", False)),
        "original_failed_checks": [
            key for key, value in original.get("checks", {}).items() if not bool(value)
        ],
        "float32_tolerance": FLOAT32_TOLERANCE,
        "state_tolerance": STATE_TOLERANCE,
        "checks": corrected_checks,
        "first_state_mismatch": first_state_mismatch,
        "first_route_mismatch": first_route_mismatch,
        "first_final_mismatch": first_final_mismatch,
    }
    output = root / f"{PREFIX}_audit_check_precision_fixed.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Best-So-Far Memory v4.2 precision-corrected independent audit")
    for key, value in corrected_checks.items():
        print("{}: {}".format(key, value))
    print("scientific verdict:", original.get("verdict"))
    print("output:", output)
    if not passed:
        raise RuntimeError(
            "Precision-corrected v4.2 audit failed: {}".format(
                [key for key, value in corrected_checks.items() if not value]
            )
        )


if __name__ == "__main__":
    main()
