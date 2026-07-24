"""MOSI transfer audit using the frozen MOSEI Stage23A analysis machinery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import stage23a_analyze as frozen
from stage23a_mosi_common import (
    EXPERTS,
    MODES,
    MISSING_MODES,
    N_FOLDS,
    RESULT_ROOT,
    atomic_csv,
    atomic_json,
    git_head,
    judge_fold,
    load_preregistered,
    load_split_manifest,
    overall_j,
    regression_metrics,
    sha256_file,
    stable_bucket,
)


def load_ledger():
    frames, manifests = [], []
    for fold in range(N_FOLDS):
        directory = RESULT_ROOT / "expert_oof" / "outer_fold{}".format(fold)
        path = directory / "oof_predictions.csv"
        manifest_path = directory / "fold_manifest.json"
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("MOSI fold artifact missing: {}".format(directory))
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("locked_test_access_count") != 0:
            raise RuntimeError("MOSI fold Test lock is not zero.")
        if manifest.get("prediction_sha256") != sha256_file(path):
            raise RuntimeError("MOSI OOF prediction SHA mismatch.")
        frames.append(pd.read_csv(path, dtype={"sample_id": str, "video_id": str}))
        manifests.append(manifest)
    frame = pd.concat(frames, ignore_index=True)
    keys = ["sample_id", "mode", "expert_id"]
    if frame.duplicated(keys).any():
        raise RuntimeError("MOSI OOF ledger duplicates sample/mode/expert.")
    if set(frame.expert_id) != set(EXPERTS) or set(frame["mode"]) != set(MODES):
        raise RuntimeError("MOSI OOF candidate or mode set differs.")
    frame["absolute_error"] = np.abs(frame.prediction - frame.label)
    frame["signed_residual"] = frame.prediction - frame.label
    frame["regret"] = (
        frame.absolute_error
        - frame.groupby(["sample_id", "mode"]).absolute_error.transform("min")
    )
    frame["judge_fold"] = frame.video_id.map(judge_fold).astype(int)
    return frame, manifests


def fold_robustness(ledger, wide, retained):
    fold_map = ledger[["sample_id", "outer_fold"]].drop_duplicates()
    wide = wide.merge(fold_map, on="sample_id", how="left", validate="many_to_one")
    errors = np.abs(
        wide[list(EXPERTS)].to_numpy() - wide.label.to_numpy()[:, None]
    )
    near_rows = []
    robust_experts = []
    for position, expert in enumerate(EXPERTS):
        fold_count = 0
        for fold in range(N_FOLDS):
            local = wide.outer_fold.to_numpy() == fold
            fraction = float(
                np.mean(
                    errors[local, position]
                    <= errors[local].min(axis=1) + 0.05
                )
            )
            near_rows.append(
                {
                    "expert_id": expert,
                    "outer_fold": fold,
                    "near_best_0p05_fraction": fraction,
                }
            )
            fold_count += int(fraction >= 0.05)
        if expert in retained and fold_count >= 2:
            robust_experts.append(expert)
    fold_headroom = []
    for fold in range(N_FOLDS):
        local = wide.loc[wide.outer_fold == fold].copy()
        fixed = overall_j(local, "per_mode_fixed_stacking")
        oracle = overall_j(local, "oracle_expert_selection")
        fold_headroom.append(
            {
                "outer_fold": fold,
                "per_mode_fixed_J": fixed,
                "oracle_J": oracle,
                "oracle_delta_J": oracle - fixed,
            }
        )
    headroom_frame = pd.DataFrame(fold_headroom)
    return (
        wide,
        robust_experts,
        pd.DataFrame(near_rows),
        headroom_frame,
        int((headroom_frame.oracle_delta_J < 0).sum()),
    )


def oof_component_report(ledger):
    rows = []
    component_rows = []
    for fold in range(N_FOLDS):
        fold_ledger = ledger.loc[ledger.outer_fold == fold]
        for expert in EXPERTS:
            local = fold_ledger.loc[fold_ledger.expert_id == expert]
            metrics_by_mode = {}
            for mode in MODES:
                values = local.loc[local["mode"] == mode]
                metrics_by_mode[mode] = regression_metrics(
                    values.prediction, values.label
                )
                rows.append(
                    {
                        "outer_fold": fold,
                        "expert_id": expert,
                        "mode": mode,
                        **metrics_by_mode[mode],
                    }
                )
            missing = {
                key: float(
                    np.mean([metrics_by_mode[mode][key] for mode in MISSING_MODES])
                )
                for key in ("J", "MAE", "Corr", "Acc7", "Acc5", "Acc2", "F1")
            }
            rows.append(
                {
                    "outer_fold": fold,
                    "expert_id": expert,
                    "mode": "MissingMacro",
                    **missing,
                }
            )
        fold_root = RESULT_ROOT / "expert_oof" / "outer_fold{}".format(fold)
        for component in (
            "clean_seed1111",
            "clean_seed1114",
            *EXPERTS,
        ):
            group = "components" if component.startswith("clean") else "experts"
            path = fold_root / group / component / "run_manifest.json"
            manifest = json.loads(path.read_text())
            component_rows.append(
                {
                    "outer_fold": fold,
                    "component": component,
                    "best_inner_J": manifest["best_inner_train_only_J"],
                    "best_epoch": manifest["best_epoch"],
                    "wall_seconds": manifest.get("wall_seconds"),
                    "peak_gpu_memory_bytes": manifest.get("peak_gpu_memory_bytes"),
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(component_rows)


def disagreement_report(ledger):
    wide = ledger.pivot_table(
        index=["sample_id", "mode"],
        columns="expert_id",
        values="prediction",
        aggfunc="first",
    )
    rows = []
    for mode in list(MODES) + ["Overall"]:
        local = wide if mode == "Overall" else wide.loc[wide.index.get_level_values("mode") == mode]
        for left_index, left in enumerate(EXPERTS):
            for right in EXPERTS[left_index + 1 :]:
                difference = np.abs(local[left] - local[right])
                rows.append(
                    {
                        "mode": mode,
                        "expert_left": left,
                        "expert_right": right,
                        "mean_absolute_disagreement": float(difference.mean()),
                        "p90_absolute_disagreement": float(difference.quantile(0.9)),
                    }
                )
    return pd.DataFrame(rows)


def write_mosi_report(
    output,
    status,
    complementarity,
    retained,
    robust_experts,
    method_seed,
    improving_folds,
    judge_gate,
    judges,
):
    speed_path = RESULT_ROOT / "speed_probe" / "speed_probe_manifest.json"
    speed = json.loads(speed_path.read_text()) if speed_path.is_file() else {}
    monitor_path = RESULT_ROOT / "parallel_safety_monitor.json"
    monitor = json.loads(monitor_path.read_text()) if monitor_path.is_file() else {}
    waiver_path = RESULT_ROOT / "parallel_resource_user_waiver.json"
    waiver = json.loads(waiver_path.read_text()) if waiver_path.is_file() else {}
    runtime_path = RESULT_ROOT / "runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {}
    speed_ratio_by_fold = {
        fold: values["baseline_seconds_per_epoch"]
        / speed.get("observed_seconds_per_epoch", float("nan"))
        for fold, values in monitor.get("baseline", {}).items()
    }
    slowdown_by_fold = {
        str(values["fold"]): values["slowdown_fraction"]
        for values in monitor.get("cross_fold_slowdown", [])
    }
    j2 = (
        judges.loc[judges.judge == "J2_predictions_disagreement"]
        if judges is not None
        else None
    )
    lines = [
        "# Stage 23A-MOSI — Frozen-Protocol Parallel Transfer Audit",
        "",
        "Final status: `{}`".format(status),
        "",
        "## Required answers",
        "",
        "1. MOSI observed seconds/epoch: {:.3f}; it was {:.1f}x–{:.1f}x faster per epoch than the two pre-parallel MOSEI baselines.".format(
            speed.get("observed_seconds_per_epoch", float("nan")),
            min(speed_ratio_by_fold.values(), default=float("nan")),
            max(speed_ratio_by_fold.values(), default=float("nan")),
        ),
        "2. MOSEI throughput impact: the initial monitor measured fold0 +{:.1%} and fold1 +{:.1%} seconds/epoch. The automatic stop verdict was `{}`; the user explicitly waived only that runtime stop and requested continued parallel execution. MOSEI was never stopped or modified.".format(
            slowdown_by_fold.get("0", float("nan")),
            slowdown_by_fold.get("1", float("nan")),
            monitor.get("verdict", "not yet available"),
        ),
        "3. Five-fold OOF expert quality: see `oof_expert_metrics_by_fold.csv`; all predictions are held-out-source OOF.",
        "4. Larger complementarity source: `{}`.".format(
            method_seed["larger_complementarity_source"]
        ),
        "5. Oracle ΔJ vs per-mode fixed stacking: {:.6f}; improving outer folds: {}/5.".format(
            complementarity["oracle_delta_J"], improving_folds
        ),
        "6. Retained experts with cross-fold contribution: {}.".format(
            ", ".join(robust_experts) if robust_experts else "none"
        ),
        "7. Judge error/regret prediction: {}.".format(
            "not trained because the Expert gate failed"
            if j2 is None
            else "mean Spearman {:.4f}, AUROC {:.4f}, ranking accuracy {:.4f}".format(
                j2.error_spearman.mean(),
                j2.top20_high_error_auroc.mean(),
                j2.regret_ranking_accuracy.mean(),
            )
        ),
        "8. Joint-risk vs fixed stacking: {}.".format(
            "not evaluated"
            if judge_gate is None
            else "mean ΔJ {:.6f}".format(
                judge_gate["mean_delta_J_vs_per_mode_fixed"]
            )
        ),
        "9. MOSI vs MOSEI trend: MOSEI is still running; no final cross-dataset claim is made.",
        "10. Worth waiting for MOSEI: Yes; MOSI is only a secondary fast falsification check.",
        "11. Recommend Student training: {}.".format(
            "No"
            if status
            != "STAGE23A_MOSI_FROZEN_PROTOCOL_CONFIRMED"
            else "Yes, subject to the primary MOSEI conclusion"
        ),
        "12. Official Valid accessed: No.",
        "13. Locked Test access count: 0.",
        "",
        "## Runtime",
        "",
        "- Five-fold wall time: {:.3f} seconds ({:.2f} hours)".format(
            runtime.get("wall_seconds", float("nan")),
            runtime.get("wall_seconds", float("nan")) / 3600,
        ),
        "- GPU: {}".format(runtime.get("gpu_id", "unknown")),
        "- Maximum simultaneous MOSI training workers: {}".format(
            runtime.get("maximum_simultaneous_training_workers", "unknown")
        ),
        "- Parallel slowdown waiver recorded: {}".format(bool(waiver)),
        "",
        "## Locks",
        "",
        "- Student trained: No",
        "- Official Valid access count: 0",
        "- Locked Test access count: 0",
        "- Existing MOSI/MOSEI worktrees modified: No",
        "- Dependencies upgraded: No",
        "",
    ]
    (output / "stage23a_mosi_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    protocol, protocol_sha = load_preregistered()
    _, split_sha = load_split_manifest()
    output = RESULT_ROOT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    ledger, fold_manifests = load_ledger()
    oof_metrics, component_metrics = oof_component_report(ledger)
    atomic_csv(oof_metrics, output / "oof_expert_metrics_by_fold.csv")
    atomic_csv(component_metrics, output / "component_training_metrics.csv")
    atomic_csv(disagreement_report(ledger), output / "expert_disagreement.csv")

    # Bind the frozen analysis implementation to MOSI paths/functions.
    frozen.RESULT_ROOT = RESULT_ROOT
    frozen.EXPERTS = EXPERTS
    frozen.load_preregistered = load_preregistered
    frozen.load_split_manifest = load_split_manifest
    frozen.load_ledger = load_ledger
    frozen.overall_j = overall_j
    frozen.judge_fold = judge_fold
    frozen.stable_bucket = stable_bucket
    wide, retained, complementarity, expert_metrics, contributions = (
        frozen.complementarity_audit(ledger, list(EXPERTS), output)
    )
    (
        wide,
        robust_experts,
        near_by_fold,
        headroom_by_fold,
        improving_folds,
    ) = fold_robustness(ledger, wide, retained)
    atomic_csv(near_by_fold, output / "expert_near_best_by_fold.csv")
    atomic_csv(headroom_by_fold, output / "oracle_headroom_by_fold.csv")
    fold_gate = improving_folds >= 3 and set(retained) == set(robust_experts)
    complementarity["fold_robustness_passed"] = fold_gate
    complementarity["improving_outer_folds"] = improving_folds
    complementarity["robust_retained_experts"] = robust_experts
    complementarity["gate_passed"] = bool(
        complementarity["gate_passed"] and fold_gate
    )
    atomic_json(output / "complementarity_gate.json", complementarity)
    method_seed = frozen.method_vs_seed_complementarity(ledger)
    atomic_json(output / "method_vs_seed_complementarity.json", method_seed)

    if not complementarity["gate_passed"]:
        status = "STAGE23A_MOSI_NO_ACTIONABLE_EXPERT_COMPLEMENTARITY"
        judge_gate = judges = None
    else:
        retained_wide = wide[
            ["sample_id", "video_id", "mode", "label", "judge_fold"]
            + robust_experts
        ].copy()
        teacher_metrics, judges, judge_gate, split_summary = (
            frozen.teacher_comparison(
                retained_wide, robust_experts, output
            )
        )
        status = (
            "STAGE23A_MOSI_TRAIN_ONLY_PASSED"
            if judge_gate["passed"]
            else "STAGE23A_MOSI_JUDGE_NOT_ACTIONABLE"
        )
    write_mosi_report(
        output,
        status,
        complementarity,
        retained,
        robust_experts,
        method_seed,
        improving_folds,
        judge_gate,
        judges,
    )
    manifest = {
        "stage": "Stage 23A-MOSI",
        "status": status,
        "protocol_sha256": protocol_sha,
        "source_split_sha256": split_sha,
        "fold_prediction_sha256": [
            value["prediction_sha256"] for value in fold_manifests
        ],
        "retained_experts": retained,
        "robust_retained_experts": robust_experts,
        "complementarity_gate": complementarity,
        "judge_teacher_gate": judge_gate,
        "method_vs_seed": method_seed,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
        "dependencies_upgraded": False,
        "code_commit": git_head(),
    }
    atomic_json(output / "stage23a_mosi_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
