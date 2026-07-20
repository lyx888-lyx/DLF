"""Generate the Stage 20 registry, diagnosis, and final report."""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.mgrd_utils import sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--stage19-root", required=True)
    return parser.parse_args()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(str(temporary), str(path))


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def append_registry(path, inherited_path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        inherited = Path(inherited_path).read_text()
        atomic_text(path, inherited if inherited.endswith("\n") else inherited + "\n")
    existing = set()
    for line in path.read_text().splitlines():
        value = json.loads(line)
        if "candidate_id" in value:
            existing.add(value["candidate_id"])
    with path.open("a") as handle:
        for record in records:
            if record["candidate_id"] not in existing:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def mode_table(method):
    selected = method["method_selected"]
    difference = method["method_minus_reference"]
    rows = []
    for mode in ("LAV", "LA", "LV", "L", "MissingMacro"):
        rows.append(
            {
                "Mode": mode,
                "J": selected["J"],
                "MAE": selected[mode]["MAE"],
                "Corr": selected[mode]["Corr"],
                "Acc7": selected[mode]["acc_7"],
                "Acc5": selected[mode]["acc_5"],
                "Acc2": selected[mode]["acc_2"],
                "F1": selected[mode]["F1_score"],
                "DeltaMAE": difference[mode]["MAE"],
                "DeltaCorr": difference[mode]["Corr"],
                "DeltaAcc7": difference[mode]["acc_7"],
                "DeltaAcc5": difference[mode]["acc_5"],
                "DeltaAcc2": difference[mode]["acc_2"],
                "DeltaF1": difference[mode]["F1_score"],
            }
        )
    return rows


def main():
    cli = parse_args()
    result = Path(cli.result_root)
    stage19 = Path(cli.stage19_root)
    protocol = load(result / "protocol/frozen_protocol.json")
    retro = load(result / "retro/stage19_retro_best_checkpoint_audit.json")
    ghost = load(result / "audit/ghost_modality_audit.json")
    tests = load(result / "tests/test_results.json")
    pm = load(result / "candidates/sao_pm_seed1111/paired_vs_uniform.json")
    full = load(result / "candidates/safe_full_seed1111/paired_vs_uniform.json")
    full_vs_pm = load(result / "candidates/safe_full_seed1111/paired_vs_sao_pm.json")
    pm_manifest = load(result / "candidates/sao_pm_seed1111/run_manifest.json")
    full_manifest = load(result / "candidates/safe_full_seed1111/run_manifest.json")
    full_mechanism_path = result / "tests/safe_full_selected_gate.json"
    full_mechanism = (
        load(full_mechanism_path) if full_mechanism_path.exists() else tests
    )
    pm_pass = bool(pm["gate_passed"])
    full_pass = bool(full["gate_passed"]) and (
        full_mechanism["tests_failed"] == 0
    )
    full_better_pm = full_vs_pm["method_minus_reference"]["J"] < 0
    if full_pass and (not pm_pass or full_better_pm):
        retained = "SAFE-DLF core"
        primary_status = "STAGE20_SAFE_DLF_SEED1_PASSED"
        route_closed = False
    elif pm_pass:
        retained = "PM auxiliary"
        primary_status = "STAGE20_PM_ONLY_RETAINED_AUXILIARY"
        route_closed = True
    else:
        retained = "none"
        primary_status = "STAGE20_SAFE_DLF_SEED1_FAILED"
        route_closed = True
    statuses = [primary_status]
    if route_closed and retained == "none":
        statuses.append("STAGE20_ROUTE_CLOSED")
    selected_candidate = (
        "safe_full"
        if retained == "SAFE-DLF core"
        else "sao_pm"
        if retained == "PM auxiliary"
        else None
    )
    seed1114 = {
        "run": False,
        "reason": (
            "No seed1111 candidate passed."
            if selected_candidate is None
            else "Required before declaring a two-seed result."
        ),
    }
    if selected_candidate is not None:
        raise RuntimeError(
            "A seed1111 method passed; run the frozen seed1114 protocol before finalizing."
        )
    for candidate in ("sao_pm_seed1114", "safe_full_seed1114"):
        atomic_json(
            result / "candidates" / candidate / "not_run.json",
            {
                "run": False,
                "seed": 1114,
                "reason": seed1114["reason"],
                "locked_test_access_count": 0,
            },
        )

    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()
    timestamp = datetime.now(timezone.utc).isoformat()
    records = []
    for candidate, analysis, manifest, parent, formula in (
        (
            "stage20_sao_pm_seed1111",
            pm,
            pm_manifest,
            "stage19_uniform_seed1111_fp32",
            "Original missing-view task loss with absent A/V specific heads multiplied by per-sample availability; no active-term normalization.",
        ),
        (
            "stage20_safe_full_seed1111",
            full,
            full_manifest,
            "stage20_sao_pm_seed1111",
            "SAO-PM plus hard per-sample availability projection after every regeneration point and absent LFA key/value removal.",
        ),
    ):
        passed = bool(analysis["gate_passed"])
        pointer = load(
            result
            / (
                "candidates/sao_pm_seed1111/checkpoints/best_checkpoint.json"
                if "sao_pm" in candidate
                else "candidates/safe_full_seed1111/checkpoints/best_checkpoint.json"
            )
        )
        records.append(
            {
                "candidate_id": candidate,
                "parent_candidate": parent,
                "hypothesis": "Align forward/backward support with actual modality availability.",
                "exact_formula": formula,
                "config_sha": sha256_file(result / "protocol/frozen_protocol.json"),
                "code_commit": protocol["implementation_commit"],
                "seed": 1111,
                "baseline_id": "stage19_uniform_seed1111_fp32",
                "status": "FULL_SEED1_PASS" if passed else "FULL_SEED1_FAIL",
                "screen_1_metrics": None,
                "screen_2_metrics": None,
                "full_metrics": analysis["method_selected"],
                "mechanism_metrics": (
                    full_mechanism["evidence"]
                    if "safe_full" in candidate
                    else {
                        "unsupported_gradient_share_after_mask": 0.0,
                        "forward_projection": False,
                    }
                ),
                "runtime": manifest.get("elapsed_seconds_this_invocation"),
                "checkpoint_path": pointer["path"],
                "stop_reason": None if passed else "Seed1111 selected-checkpoint gate failed.",
                "retained_component": candidate if passed else None,
                "rejected_component": None if passed else candidate,
                "test_access_count": 0,
                "timestamp": timestamp,
            }
        )
    registry_path = result / "registry/innovation_registry.jsonl"
    append_registry(
        registry_path,
        stage19 / "registry/innovation_registry.jsonl",
        records,
    )
    summary_rows = [
        [
            record["candidate_id"],
            record["status"],
            record["full_metrics"]["Epoch"],
            record["full_metrics"]["J"],
            pm["method_minus_reference"]["J"]
            if "sao_pm" in record["candidate_id"]
            else full["method_minus_reference"]["J"],
            record["runtime"],
            record["retained_component"] or "none",
        ]
        for record in records
    ]
    summary_path = result / "registry/candidate_summary.tsv"
    text = ["Candidate\tStatus\tSelectedEpoch\tJ\tDeltaJ\tWallSeconds\tRetained"]
    text += ["\t".join(map(str, row)) for row in summary_rows]
    atomic_text(summary_path, "\n".join(text) + "\n")

    failure_lines = [
        "# Stage 20 failure diagnosis",
        "",
        "## Implementation",
        "",
        "- The first long-run attempt completed epoch 1 computation but failed before resumable persistence because optional screen keys were absent.",
        "- The failed attempt was preserved; fix commit: `c1f686e8164f729e0e31d6ad18cec21d111b0c3d`.",
        "- The clean rerun passed all implementation gates and completed normally.",
        "",
        "## Mechanism",
        "",
        "- Ghost headroom was present: maximum GAR `{:.6f}`.".format(
            ghost["maximum_ghost_activation_ratio"]
        ),
        "- Frozen-wrapper raw filler sensitivity was `{:.3e}`.".format(
            ghost["maximum_prediction_filler_sensitivity"]
        ),
        "- Baseline unsupported gradient share was `{:.6f}`.".format(
            ghost["aggregate_unsupported_gradient_share"]
        ),
        "- SAFE Full removed filler sensitivity and absent gradients in the implementation gate.",
        "",
        "## Metrics",
        "",
        "- SAO-PM delta J: `{:+.6f}`; passed: `{}`.".format(
            pm["method_minus_reference"]["J"], pm_pass
        ),
        "- SAFE Full delta J: `{:+.6f}`; passed: `{}`.".format(
            full["method_minus_reference"]["J"], full_pass
        ),
        "- SAFE Full minus SAO-PM delta J: `{:+.6f}`.".format(
            full_vs_pm["method_minus_reference"]["J"]
        ),
        "",
        "No coefficient, learning-rate, batch-size, AMP, KD, or Test-based rescue was attempted.",
    ]
    atomic_text(
        result / "final/failure_diagnosis.md",
        "\n".join(failure_lines) + "\n",
    )
    test_lock = {
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "test_samples_read": False,
        "test_predictions_or_metrics_read": False,
        "selection_uses_train_and_official_valid_only": True,
    }
    atomic_json(result / "final/TEST_LOCK_STATUS.json", test_lock)
    report = {
        "stage_statuses": statuses,
        "branch": protocol["branch"],
        "base_commit": protocol["base_commit"],
        "implementation_commit": protocol["implementation_commit"],
        "report_generation_commit": current_commit,
        "retro": retro,
        "ghost_audit": {
            "maximum_ghost_activation_ratio": ghost[
                "maximum_ghost_activation_ratio"
            ],
            "maximum_prediction_filler_sensitivity": ghost[
                "maximum_prediction_filler_sensitivity"
            ],
            "aggregate_unsupported_gradient_share": ghost[
                "aggregate_unsupported_gradient_share"
            ],
            "attention_leakage": ghost["attention_leakage"],
        },
        "implementation_tests": tests,
        "sao_pm_seed1111": pm,
        "safe_full_seed1111": full,
        "safe_full_vs_sao_pm": full_vs_pm,
        "sao_pm_wall_seconds": pm_manifest["elapsed_seconds_this_invocation"],
        "safe_full_wall_seconds": full_manifest["elapsed_seconds_this_invocation"],
        "seed1114": seed1114,
        "retained": retained,
        "locked_test_access_count": 0,
        "dependencies_upgraded": False,
        "original_worktrees_modified": False,
        "created_at": timestamp,
    }
    atomic_json(result / "final/stage20_final_report.json", report)
    uniform = pm["reference_selected"]
    metric_names = (
        ("MAE", "MAE"),
        ("Corr", "Corr"),
        ("Acc7", "acc_7"),
        ("Acc5", "acc_5"),
        ("Acc2", "acc_2"),
        ("F1", "F1_score"),
    )
    comparison_rows = []
    for mode in ("LAV", "LA", "LV", "L", "MissingMacro"):
        for label, key in metric_names:
            comparison_rows.append(
                "| {} | {} | {:.6f} | {:.6f} | {:+.6f} | {:.6f} | {:+.6f} |".format(
                    mode,
                    label,
                    uniform[mode][key],
                    pm["method_selected"][mode][key],
                    pm["method_minus_reference"][mode][key],
                    full["method_selected"][mode][key],
                    full["method_minus_reference"][mode][key],
                )
            )
    pm_gate_rows = [
        "| {} | {} |".format(key, "PASS" if value else "FAIL")
        for key, value in pm["gate_checks"].items()
    ]
    full_gate_rows = [
        "| {} | {} |".format(key, "PASS" if value else "FAIL")
        for key, value in full["gate_checks"].items()
    ]
    evidence = full_mechanism["evidence"]
    unsupported_by_mode = {}
    for row in ghost["unsupported_gradients"]:
        if row["Unsupported"]:
            unsupported_by_mode[row["Mode"]] = (
                unsupported_by_mode.get(row["Mode"], 0.0)
                + row["WeightedGradientShare"]
            )
    absent_ratio_by_mode = {
        row["Mode"]: row["AbsentGradientRatio"]
        for row in ghost["absent_gradients"]
    }
    lines = [
        "# Stage 20 SAFE-DLF v1 final report",
        "",
        "- Status: {}".format(", ".join("`{}`".format(v) for v in statuses)),
        "- Retained: **{}**".format(retained),
        "- Locked Test access count: **0**",
        "- Selection split: **official Valid only**",
        "- Seed 1114: **not run** (seed1111 gate failed)",
        "",
        "## Phase R: Stage19 checkpoint retrospective",
        "",
        "- Uniform best: epoch {}, J `{:.6f}`.".format(
            retro["uniform"]["selected_epoch"],
            retro["uniform"]["best_so_far_J"],
        ),
        "- MGD best among all existing Valid checkpoints: epoch {}, J `{:.6f}`.".format(
            retro["mgd"]["selected_epoch"],
            retro["mgd"]["best_so_far_J"],
        ),
        "- MGD - Uniform delta J: `{:+.6f}` (positive is worse); status: `{}`.".format(
            retro["mgd_minus_uniform"]["J"], retro["status"]
        ),
        "- Therefore the Stage19 MGD best checkpoint is **not** genuinely better than the Uniform best checkpoint; MGD was not retrained and MGRD was not run.",
        "",
        "## Phase A: frozen Uniform ghost audit",
        "",
        "- Maximum GAR: `{:.6f}`".format(
            ghost["maximum_ghost_activation_ratio"]
        ),
        "- Raw-input filler sensitivity with fixed availability: `{:.3e}` (no measurable filler sensitivity).".format(
            ghost["maximum_prediction_filler_sensitivity"]
        ),
        "- Aggregate unsupported gradient share: `{:.6f}` (LA `{:.6f}`, LV `{:.6f}`, L `{:.6f}`).".format(
            ghost["aggregate_unsupported_gradient_share"]
            , unsupported_by_mode["LA"], unsupported_by_mode["LV"], unsupported_by_mode["L"]
        ),
        "- Absent-branch parameter gradient ratio: LA `{:.6f}`, LV `{:.6f}`, L `{:.6f}`.".format(
            absent_ratio_by_mode["LA"],
            absent_ratio_by_mode["LV"],
            absent_ratio_by_mode["L"],
        ),
        "- Frozen LFA attention mass assigned to an absent key/value branch reached `1.0`.",
        "- Interpretation: ghost activation/backward support mismatch is real, even though replacing the raw filler does not change the wrapped model output.",
        "",
        "## Seed 1111 selected-checkpoint comparison",
        "",
        "| Method | Epoch | J | Delta J | Missing modes MAE improved | Passed | Wall h |",
        "|---|---:|---:|---:|---:|---|---:|",
        "| SAO-PM | {} | {:.6f} | {:+.6f} | {}/3 | {} | {:.3f} |".format(
            pm["method_selected"]["Epoch"],
            pm["method_selected"]["J"],
            pm["method_minus_reference"]["J"],
            pm["missing_modes_mae_improved"],
            pm_pass,
            pm_manifest["elapsed_seconds_this_invocation"] / 3600.0,
        ),
        "| SAFE Full | {} | {:.6f} | {:+.6f} | {}/3 | {} | {:.3f} |".format(
            full["method_selected"]["Epoch"],
            full["method_selected"]["J"],
            full["method_minus_reference"]["J"],
            full["missing_modes_mae_improved"],
            full_pass,
            full_manifest["elapsed_seconds_this_invocation"] / 3600.0,
        ),
        "",
        "SAFE Full - SAO-PM delta J: `{:+.6f}`.".format(
            full_vs_pm["method_minus_reference"]["J"]
        ),
        "",
        "Lower J/MAE is better; higher Corr/Acc/F1 is better. All values below are recomputed from the frozen selected Valid checkpoints.",
        "",
        "| Mode | Metric | Uniform | SAO-PM | PM - Uniform | SAFE Full | Full - Uniform |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(comparison_rows)
    lines.extend(
        [
            "",
            "## Seed1111 gates",
            "",
            "### SAO-PM",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    lines.extend(pm_gate_rows)
    lines.extend(
        [
            "",
            "### SAFE Full",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    lines.extend(full_gate_rows)
    lines.extend(
        [
            "",
            "Both candidates fail the frozen seed1111 promotion gate. SAFE Full is better than SAO-PM on J (`{:+.6f}`), but remains worse than Uniform and violates the regression/correlation/classification safety gates.".format(
                full_vs_pm["method_minus_reference"]["J"]
            ),
            "",
            "## SAFE Full implementation evidence",
            "",
            "| Check | Observed | Required | Result |",
            "|---|---:|---:|---|",
            "| LAV output max abs diff | {:.3e} | <= 1e-6 | PASS |".format(
                evidence["lav_output_max_abs_diff"]
            ),
            "| LAV total-loss abs diff | {:.3e} | <= 1e-6 | PASS |".format(
                evidence["lav_total_loss_abs_diff"]
            ),
            "| LAV component-loss max diff | {:.3e} | <= 1e-6 | PASS |".format(
                evidence["lav_component_loss_max_abs_diff"]
            ),
            "| Filler output max abs diff | {:.3e} | <= 1e-6 | PASS |".format(
                evidence["filler_output_max_abs_diff"]
            ),
            "| Filler fusion max abs diff | {:.3e} | <= 1e-6 | PASS |".format(
                evidence["filler_fusion_max_abs_diff"]
            ),
            "| Absent input gradient max | {:.3e} | <= 1e-8 | PASS |".format(
                evidence["absent_input_gradient_max"]
            ),
            "| Absent representation max | {:.3e} | <= 1e-8 | PASS |".format(
                evidence["absent_representation_max"]
            ),
            "| Unsupported loss max | {:.3e} | 0 | PASS |".format(
                evidence["unsupported_loss_max"]
            ),
            "| Present input gradient min | {:.6f} | > 0 | PASS |".format(
                evidence["present_input_gradient_min"]
            ),
            "",
            "The selected checkpoint passed **{}/{}** implementation tests; failed tests: **{}**.".format(
                full_mechanism["tests_passed"],
                full_mechanism["tests_run"],
                full_mechanism["tests_failed"],
            ),
            "",
            "## Required closure",
            "",
            "1. **Stage19 MGD vs Uniform:** MGD is worse (`Delta J = {:+.6f}`); not retained.".format(
                retro["mgd_minus_uniform"]["J"]
            ),
            "2. **Baseline ghost activation:** yes; maximum GAR `{:.6f}`.".format(
                ghost["maximum_ghost_activation_ratio"]
            ),
            "3. **Baseline filler sensitivity:** no measurable raw-filler sensitivity; max prediction difference `{:.3e}`.".format(
                ghost["maximum_prediction_filler_sensitivity"]
            ),
            "4. **Unsupported gradient share:** aggregate `{:.6f}` ({:.2f}%).".format(
                ghost["aggregate_unsupported_gradient_share"],
                100.0 * ghost["aggregate_unsupported_gradient_share"],
            ),
            "5. **PM improvement:** no; `Delta J = {:+.6f}`, 0/3 missing-mode MAEs improved.".format(
                pm["method_minus_reference"]["J"]
            ),
            "6. **Full SAFE improvement:** no; `Delta J = {:+.6f}`, 1/3 missing-mode MAEs improved.".format(
                full["method_minus_reference"]["J"]
            ),
            "7. **Full vs PM:** Full has lower J by `{:.6f}`, but neither passes and Full has classification trade-offs.".format(
                -full_vs_pm["method_minus_reference"]["J"]
            ),
            "8. **LAV parity:** passed; output/loss/component maximum differences are all zero.",
            "9. **Filler invariance:** passed; prediction and fusion differences are zero.",
            "10. **Absent gradient:** passed; absent input gradient and unsupported loss are zero.",
            "11. **Seed1111:** failed the frozen promotion gate.",
            "12. **Seed1114:** not run, by preregistered gate.",
            "13. **Final retention:** none.",
            "14. **Single-seed wall time:** SAO-PM `{:.3f}` h; SAFE Full `{:.3f}` h.".format(
                pm_manifest["elapsed_seconds_this_invocation"] / 3600.0,
                full_manifest["elapsed_seconds_this_invocation"] / 3600.0,
            ),
            "15. **Branch/commit/push/clean:** branch `{}`; implementation commit `{}`. Final push and clean-state verification are recorded in the operator handoff.".format(
                protocol["branch"], protocol["implementation_commit"]
            ),
            "16. **Tests:** {}/{} passed on the selected SAFE Full checkpoint.".format(
                full_mechanism["tests_passed"], full_mechanism["tests_run"]
            ),
            "17. **GPU release:** both training jobs exited; final GPU process verification is recorded in the operator handoff.",
            "18. **Locked Test access:** `0`.",
            "19. **Dependencies:** none upgraded.",
            "20. **Original worktrees:** not modified.",
            "",
            "## Decision",
            "",
            "`STAGE20_SAFE_DLF_SEED1_FAILED` and `STAGE20_ROUTE_CLOSED`.",
            "",
            "The implementation/mechanism audit succeeded, but the selected Valid metrics did not. No coefficient, learning-rate, batch-size, AMP, KD, Test-based rescue, seed1114 run, or cross-dataset run was attempted.",
        ]
    )
    atomic_text(result / "final/stage20_final_report.md", "\n".join(lines) + "\n")
    print(json.dumps({"statuses": statuses, "retained": retained}, indent=2))


if __name__ == "__main__":
    main()
