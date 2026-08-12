"""MOSEI Valid-only final composition for frozen FixedBlend-DP57-v1.

Protocol:
  1. Rebuild Raw5-PE5 from the five formal CFCompat-v1 Valid prediction files
     with equal 0.2 membership and no seed deletion/calibration.
  2. Load the formal frozen v13 Valid prediction file (seed1113 expert).
  3. Apply the MOSI-frozen blend exactly once: 0.5 * Raw5 + 0.5 * v13.
     No MOSEI blend-weight search is implemented.
  4. Select the anchor solely from the five CFCompat validation J values and
     require the already-frozen MOSEI selection seed1114.
  5. Apply label-free DP57 (Acc7+Acc5 decision preservation) to the FixedBlend
     target, separately for LAV/LA/LV/L.

Official Test is never constructed or read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
)
from trains.singleTask.missing_utils import MISSING_MODES, regression_metrics

DATASET = "mosei"
SEEDS = (1111, 1112, 1113, 1114, 1115)
MODES = ("LAV",) + MISSING_MODES
EXPECTED_VALID_N = 1871
FROZEN_ANCHOR_SEED = 1114
RAW5_WEIGHT = np.float32(0.2)
BLEND_WEIGHT_RAW5 = np.float32(0.5)
BLEND_WEIGHT_V13 = np.float32(0.5)
DP_VARIANT = "adpep57"


def parse_args():
    parser = argparse.ArgumentParser(description="MOSEI FixedBlend-DP57-v1 Valid-only audit")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--model-root", default="pt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction(path: Path, expected_seed=None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    required = {"sample_index", "label"} | {f"{mode}_pred" for mode in MODES}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_VALID_N or frame.sample_index.nunique() != EXPECTED_VALID_N:
        raise RuntimeError(f"{path}: expected {EXPECTED_VALID_N} unique Valid rows, got {len(frame)}")
    if not np.array_equal(frame.sample_index.to_numpy(np.int64), np.arange(EXPECTED_VALID_N)):
        raise RuntimeError(f"{path}: sample_index must be exactly 0..{EXPECTED_VALID_N - 1}")
    if frame[["label"] + [f"{mode}_pred" for mode in MODES]].isna().any().any():
        raise FloatingPointError(f"{path}: NaN in label/prediction columns")
    if not np.isfinite(frame[["label"] + [f"{mode}_pred" for mode in MODES]].to_numpy(np.float64)).all():
        raise FloatingPointError(f"{path}: NaN/Inf in label/prediction columns")
    if "Split" in frame.columns and set(frame.Split.astype(str).str.lower()) != {"valid"}:
        raise RuntimeError(f"{path}: non-Valid rows detected")
    if expected_seed is not None and "Seed" in frame.columns:
        observed = set(frame.Seed.astype(int))
        if observed != {int(expected_seed)}:
            raise RuntimeError(f"{path}: Seed metadata mismatch: {observed}")
    return frame


def assert_binding(reference: pd.DataFrame, current: pd.DataFrame, tag: str) -> None:
    if not np.array_equal(reference.sample_index.to_numpy(np.int64), current.sample_index.to_numpy(np.int64)):
        raise RuntimeError(f"{tag}: sample_index binding differs")
    if "sample_id" in reference.columns and "sample_id" in current.columns:
        if not np.array_equal(reference.sample_id.astype(str).to_numpy(), current.sample_id.astype(str).to_numpy()):
            raise RuntimeError(f"{tag}: sample_id binding differs")
    max_label_diff = float(np.max(np.abs(reference.label.to_numpy(np.float64) - current.label.to_numpy(np.float64))))
    if max_label_diff > 1e-7:
        raise RuntimeError(f"{tag}: label binding differs, max abs diff={max_label_diff}")


def evaluate(predictions: dict[str, np.ndarray], labels: np.ndarray):
    label_tensor = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
    metrics = {}
    for mode in MODES:
        pred_tensor = torch.tensor(np.asarray(predictions[mode], dtype=np.float32), dtype=torch.float32)
        metrics[mode] = regression_metrics(pred_tensor, label_tensor)
    metrics["MissingMacro"] = {
        key: float(np.mean([metrics[mode][key] for mode in MISSING_MODES]))
        for key in metrics["LAV"]
    }
    j = 0.5 * float(metrics["LAV"]["MAE"]) + 0.5 * float(metrics["MissingMacro"]["MAE"])
    return metrics, float(j)


def metric_line(name: str, metrics: dict, j: float) -> str:
    lav = metrics["LAV"]
    mm = metrics["MissingMacro"]
    return (
        f"{name:12s} J={j:.9f} "
        f"LAV_MAE={lav['MAE']:.9f} MissingMacro_MAE={mm['MAE']:.9f} "
        f"LAV_Corr={lav['Corr']:.9f} LAV_Acc2={lav['acc_2']:.9f} "
        f"LAV_F1={lav['F1_score']:.9f} LAV_Acc7={lav['acc_7']:.9f} LAV_Acc5={lav['acc_5']:.9f}"
    )


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    model_root = Path(args.model_root)

    raw_root = result_root / "missing_baseline" / "cfcompat_prediction_ensemble_v1" / DATASET
    member_paths = {seed: raw_root / f"online_seed{seed}_valid_predictions.csv" for seed in SEEDS}
    members = {seed: load_prediction(path, expected_seed=seed) for seed, path in member_paths.items()}
    reference = members[SEEDS[0]]
    for seed in SEEDS[1:]:
        assert_binding(reference, members[seed], f"Raw5 seed{seed}")

    # Reproduce the previously audited Raw5 numerical path: decimal CSV ->
    # float64 mean -> evaluator/storage float32.
    raw5 = {}
    for mode in MODES:
        stacked = np.stack([members[seed][f"{mode}_pred"].to_numpy(np.float64) for seed in SEEDS], axis=0)
        raw5[mode] = np.mean(stacked, axis=0).astype(np.float32)

    v13_root = result_root / "missing_baseline" / "cfcompat_adam_step_safety_v13" / DATASET / "valid_screen" / "seed1113_dev"
    v13_path = v13_root / "v13_valid_predictions.csv"
    v13_summary_path = v13_root / "adam_step_safety_v13_mosei_summary.json"
    v13_checkpoint = model_root / "missing_baseline" / "cfcompat_adam_step_safety_v13" / DATASET / "valid_screen" / "seed1113_dev" / "frozen_consensus_valid_ready.pth"
    if not v13_summary_path.is_file() or not v13_checkpoint.is_file():
        raise FileNotFoundError(f"Missing formal v13 summary/checkpoint: {v13_summary_path} / {v13_checkpoint}")
    v13_summary = json.loads(v13_summary_path.read_text(encoding="utf-8"))
    protocol = v13_summary.get("protocol", {})
    if bool(protocol.get("official_test_constructed", True)) or bool(protocol.get("official_test_accessed", True)):
        raise RuntimeError("Formal v13 summary does not certify Test isolation")
    if str(v13_summary.get("verdict", "")).upper().startswith("SMOKE"):
        raise RuntimeError("Refusing smoke v13 artifact for frozen Valid composition")
    v13_frame = load_prediction(v13_path, expected_seed=1113)
    assert_binding(reference, v13_frame, "formal v13")
    v13 = {mode: v13_frame[f"{mode}_pred"].to_numpy(np.float32) for mode in MODES}

    # Frozen 0.5/0.5 blend only. There is intentionally no weight CLI.
    fixedblend = {
        mode: (BLEND_WEIGHT_RAW5 * raw5[mode] + BLEND_WEIGHT_V13 * v13[mode]).astype(np.float32)
        for mode in MODES
    }

    per_seed_path = result_root / "missing_baseline" / "cf_compat_kd_v1" / "mosei_per_seed.csv"
    if not per_seed_path.is_file():
        raise FileNotFoundError(per_seed_path)
    per_seed = pd.read_csv(per_seed_path)
    if "Seed" not in per_seed.columns or "J_valid" not in per_seed.columns:
        raise RuntimeError("Canonical CFCompat per-seed CSV lacks Seed/J_valid")
    local = per_seed.loc[per_seed.Seed.astype(int).isin(SEEDS), ["Seed", "J_valid"]].copy()
    if len(local) != 5 or local.Seed.astype(int).nunique() != 5:
        raise RuntimeError("Canonical CFCompat Valid J rows are incomplete")
    anchor_seed = int(local.sort_values(["J_valid", "Seed"], kind="mergesort").iloc[0].Seed)
    if anchor_seed != FROZEN_ANCHOR_SEED:
        raise RuntimeError(f"Frozen MOSEI anchor drifted: expected {FROZEN_ANCHOR_SEED}, observed {anchor_seed}")
    anchor_frame = members[anchor_seed]
    anchor = {mode: anchor_frame[f"{mode}_pred"].to_numpy(np.float32) for mode in MODES}

    dp57 = {}
    projection_rows = []
    for mode in MODES:
        projected, results = project_array(anchor[mode], fixedblend[mode], DATASET, DP_VARIANT)
        dp57[mode] = projected.astype(np.float32)
        anchor7, anchor5, _ = evaluator_decisions(anchor[mode], DATASET)
        projected7, projected5, _ = evaluator_decisions(dp57[mode], DATASET)
        if not np.array_equal(anchor7, projected7):
            raise RuntimeError(f"DP57 Acc7 inheritance failed for {mode}")
        if not np.array_equal(anchor5, projected5):
            raise RuntimeError(f"DP57 Acc5 inheritance failed for {mode}")
        _, _, blend2 = evaluator_decisions(fixedblend[mode], DATASET)
        _, _, dp2 = evaluator_decisions(dp57[mode], DATASET)
        for index, result in enumerate(results):
            projection_rows.append({
                "sample_index": int(index),
                "Mode": mode,
                "anchor_prediction": float(anchor[mode][index]),
                "fixedblend_prediction": float(fixedblend[mode][index]),
                "dp57_prediction": float(dp57[mode][index]),
                "target_already_feasible": bool(result.pe5_already_feasible),
                "boundary_adjusted": bool(result.boundary_adjusted),
                "fallback_to_anchor": bool(result.fallback_to_anchor),
                "fallback_reason": str(result.fallback_reason),
                "acc2_changed_vs_fixedblend": bool(blend2[index] != dp2[index]),
            })

    labels = reference.label.to_numpy(np.float32)
    raw5_metrics, raw5_j = evaluate(raw5, labels)
    v13_metrics, v13_j = evaluate(v13, labels)
    blend_metrics, blend_j = evaluate(fixedblend, labels)
    dp57_metrics, dp57_j = evaluate(dp57, labels)
    anchor_metrics, anchor_j = evaluate(anchor, labels)

    # Bind against already-audited numbers so accidental numerical drift is loud.
    if abs(raw5_j - 0.513192346) > 5e-7:
        raise RuntimeError(f"Raw5 replay drifted: {raw5_j:.12f} vs frozen 0.513192346")
    if abs(v13_j - 0.531753172) > 5e-7:
        raise RuntimeError(f"formal v13 replay drifted: {v13_j:.12f} vs frozen 0.531753172")

    out = result_root / "missing_baseline" / "fixedblend_dp57_v1" / DATASET / "valid_screen"
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists; inspect it or pass --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    base_cols = {
        "sample_index": reference.sample_index.to_numpy(np.int64),
        "label": labels,
    }
    if "sample_id" in reference.columns:
        base_cols["sample_id"] = reference.sample_id.astype(str).to_numpy()

    raw5_out = pd.DataFrame(base_cols)
    blend_out = pd.DataFrame(base_cols)
    dp57_out = pd.DataFrame(base_cols)
    for mode in MODES:
        raw5_out[f"{mode}_pred"] = raw5[mode]
        blend_out[f"{mode}_pred"] = fixedblend[mode]
        dp57_out[f"{mode}_pred"] = dp57[mode]
        dp57_out[f"anchor_{mode}_pred"] = anchor[mode]
    raw5_out["Method"] = "Raw5-PE5"
    raw5_out["Split"] = "valid"
    blend_out["Method"] = "FixedBlend-0p5-Raw5-0p5-v13"
    blend_out["Split"] = "valid"
    dp57_out["Method"] = "FixedBlend-DP57-v1"
    dp57_out["AnchorSeed"] = anchor_seed
    dp57_out["Split"] = "valid"

    raw5_out.to_csv(out / "raw5_valid_predictions.csv", index=False)
    blend_out.to_csv(out / "fixedblend_valid_predictions.csv", index=False)
    dp57_out.to_csv(out / "fixedblend_dp57_valid_predictions.csv", index=False)
    projection = pd.DataFrame(projection_rows)
    projection.to_csv(out / "dp57_projection_audit.csv", index=False)

    projection_summary = {
        mode: {
            "N": int(len(projection.loc[projection.Mode.eq(mode)])),
            "projected_fraction": float((~projection.loc[projection.Mode.eq(mode), "target_already_feasible"]).mean()),
            "target_feasible_fraction": float(projection.loc[projection.Mode.eq(mode), "target_already_feasible"].mean()),
            "boundary_adjusted_count": int(projection.loc[projection.Mode.eq(mode), "boundary_adjusted"].sum()),
            "fallback_count": int(projection.loc[projection.Mode.eq(mode), "fallback_to_anchor"].sum()),
            "acc2_changed_vs_fixedblend_count": int(projection.loc[projection.Mode.eq(mode), "acc2_changed_vs_fixedblend"].sum()),
        }
        for mode in MODES
    }

    summary = {
        "dataset": DATASET,
        "method": "FixedBlend-DP57-v1",
        "protocol": {
            "raw5_members": list(SEEDS),
            "raw5_equal_weight": float(RAW5_WEIGHT),
            "v13_seed": 1113,
            "blend_weights": {"Raw5": float(BLEND_WEIGHT_RAW5), "v13": float(BLEND_WEIGHT_V13)},
            "blend_weight_search": False,
            "anchor_selected_by": "minimum_CFCompat_validation_J",
            "anchor_seed": anchor_seed,
            "dp_variant": DP_VARIANT,
            "dp_preserves": ["Acc7", "Acc5"],
            "labels_used_by_dp_projection": False,
            "official_test_constructed": False,
            "official_test_accessed": False,
        },
        "sources": {
            "raw5_members": {str(seed): str(member_paths[seed]) for seed in SEEDS},
            "formal_v13_predictions": str(v13_path),
            "formal_v13_checkpoint": str(v13_checkpoint),
            "formal_v13_checkpoint_sha256": sha256(v13_checkpoint),
            "cfcompat_per_seed_valid": str(per_seed_path),
        },
        "metrics": {
            "Anchor": {"J": anchor_j, **anchor_metrics},
            "Raw5": {"J": raw5_j, **raw5_metrics},
            "v13": {"J": v13_j, **v13_metrics},
            "FixedBlend": {"J": blend_j, **blend_metrics},
            "DP57": {"J": dp57_j, **dp57_metrics},
        },
        "deltas": {
            "FixedBlend_minus_Raw5_J": blend_j - raw5_j,
            "DP57_minus_FixedBlend_J": dp57_j - blend_j,
            "DP57_minus_Raw5_J": dp57_j - raw5_j,
        },
        "projection_summary": projection_summary,
    }
    (out / "valid_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("================ MOSEI FixedBlend-DP57 Valid =====================")
    print("anchor seed:", anchor_seed)
    print(metric_line("Anchor", anchor_metrics, anchor_j))
    print(metric_line("Raw5", raw5_metrics, raw5_j))
    print(metric_line("v13", v13_metrics, v13_j))
    print(metric_line("FixedBlend", blend_metrics, blend_j))
    print(metric_line("DP57", dp57_metrics, dp57_j))
    print("FixedBlend delta J vs Raw5:", f"{blend_j - raw5_j:+.9f}")
    print("DP57 delta J vs FixedBlend:", f"{dp57_j - blend_j:+.9f}")
    print("DP57 delta J vs Raw5:", f"{dp57_j - raw5_j:+.9f}")
    for mode in MODES:
        item = projection_summary[mode]
        print(
            f"{mode} projection: projected={item['projected_fraction']:.6f} "
            f"feasible={item['target_feasible_fraction']:.6f} "
            f"boundary_adjusted={item['boundary_adjusted_count']} "
            f"fallback={item['fallback_count']} "
            f"acc2_changed_vs_blend={item['acc2_changed_vs_fixedblend_count']}"
        )
    print("blend weight search:", False)
    print("TEST CONSTRUCTED:", False)
    print("summary:", out / "valid_summary.json")


if __name__ == "__main__":
    main()
