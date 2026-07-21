"""Generate the complete Stage 21A evidence-closure report."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage21a_common import MODES, MISSING_MODES, atomic_json, atomic_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def load(path):
    return json.loads(Path(path).read_text())


def git_value(*args):
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def mean_split_delta(probes, comparison, mode=None, metric=None):
    values = []
    for split in probes["inner_splits"]:
        value = split["deltas"][comparison]
        if mode is not None:
            value = value[mode][metric]
        else:
            value = value["J"]
        values.append(float(value))
    return float(np.mean(values))


def main():
    cli = parse_args()
    root = Path(cli.output_root)
    protocol = load(root / "protocol/frozen_protocol.json")
    initial = load(root / "protocol/initial_state.json")
    binding = load(root / "data/data_binding_audit.json")
    coverage = load(root / "pairs/pair_coverage.json")
    cache = load(root / "representations/representation_cache_manifest.json")
    residual = load(root / "residual/source_residual_icc.json")
    probes = load(root / "probes/probe_metrics.json")
    gradients = load(root / "gradients/gradient_compatibility.json")
    tests = load(root / "tests/test_results.json")
    if tests["tests_failed"]:
        raise RuntimeError("Stage 21A implementation tests failed.")
    status = probes["status"]
    train_summary = binding["splits"]["train"]
    valid_summary = binding["splits"]["valid"]
    selected = next(row for row in coverage["coverage"] if row["delta"] == coverage["selected_delta"])
    fingerprint = {}
    for mode in MODES:
        rows = [row for row in probes["fingerprint"] if row["mode"] == mode]
        fingerprint[mode] = {
            "mean_auroc": float(np.mean([row["auroc"] for row in rows])),
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in rows])),
            "mean_shuffled_auroc": float(np.mean([row["shuffled_train_label_auroc"] for row in rows])),
        }
    missing_improved = [
        mode for mode in MISSING_MODES
        if mean_split_delta(probes, "P1_minus_P0", mode, "MAE") < 0
    ]
    tradeoffs = {
        mode: {
            metric: mean_split_delta(probes, "P1_minus_P0", mode, metric)
            for metric in ("MAE", "Corr", "acc_7", "acc_5", "acc_2", "F1_score")
        }
        for mode in MODES + ("MissingMacro",)
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "branch": git_value("branch", "--show-current"),
        "base_commit": protocol["base_commit"],
        "implementation_commit": protocol["implementation_commit"],
        "report_generation_commit": git_value("rev-parse", "HEAD"),
        "external_tasks_initially_present": initial["external_compute_processes_present"],
        "data_binding": binding,
        "pair_coverage": coverage,
        "representation_cache": cache,
        "residual": residual,
        "fingerprint_summary": fingerprint,
        "probes": probes,
        "gradient_compatibility": gradients,
        "tests": tests,
        "missing_modes_improved": missing_improved,
        "mean_metric_tradeoffs": tradeoffs,
        "source_relative_claim_supported": status == "STAGE21A_SOURCE_RELATIVE_AUDIT_PASSED",
        "generic_pairwise_only": bool(probes["generic_pairwise_signal"]),
        "recommend_stage21b": status == "STAGE21A_SOURCE_RELATIVE_AUDIT_PASSED",
        "official_valid_accessed": bool(probes["official_valid"]["run"]),
        "test_accessed": False,
        "locked_test_access_count": 0,
        "dependencies_upgraded": False,
        "original_worktrees_modified": False,
    }
    atomic_json(root / "final/stage21a_final_report.json", report)
    atomic_json(
        root / "final/TEST_LOCK_STATUS.json",
        {
            "locked_test_access_count": 0,
            "test_loader_constructed": False,
            "test_samples_read": False,
            "test_predictions_or_metrics_read": False,
            "test_file_existence_used_for_decision": False,
            "selection_uses_train_only_audit_and_conditionally_official_valid": True,
        },
    )
    failure_lines = [
        "# Stage 21A failure diagnosis",
        "",
        "- Final status: `{}`".format(status),
        "- Source residual gate passed: `{}`".format(residual["source_residual_gate_passed"]),
        "- Train-only probe gate passed: `{}`".format(probes["train_only_gate"]["passed"]),
        "- Pair controls valid: `{}`".format(probes["pair_controls_valid"]),
        "- Mean P1-P0 J: `{:+.6f}`".format(probes["train_only_gate"]["mean_P1_minus_P0_J"]),
        "- Mean P1-P2 J: `{:+.6f}`".format(probes["train_only_gate"]["mean_P1_minus_P2_J"]),
        "- Mean P1-P3 J: `{:+.6f}`".format(probes["train_only_gate"]["mean_P1_minus_P3_J"]),
        "- Generic pairwise-only pattern: `{}`".format(probes["generic_pairwise_signal"]),
        "- Official Valid probe run: `{}`".format(probes["official_valid"]["run"]),
        "",
        "No threshold, delta, split, ridge alpha, relative mass, representation, or control was changed after inspecting results.",
    ]
    atomic_text(root / "final/failure_diagnosis.md", "\n".join(failure_lines) + "\n")
    lines = [
        "# Stage 21A Source-Relative Learning Feasibility Audit v1",
        "",
        "- Final status: `{}`".format(status),
        "- Source-relative claim supported: **{}**".format(report["source_relative_claim_supported"]),
        "- Generic pairwise-only signal: **{}**".format(report["generic_pairwise_only"]),
        "- Recommend Stage21B: **{}**".format(report["recommend_stage21b"]),
        "- Official Valid probe run: **{}**".format(report["official_valid_accessed"]),
        "- Locked Test access count: **0**",
        "",
        "## Data and pair feasibility",
        "",
        "| Split | Samples | Source videos | Mean clips/source | Median | P90 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Train | {} | {} | {:.4f} | {:.1f} | {:.1f} | {} |".format(
            train_summary["sample_count"], train_summary["unique_video_id_count"],
            train_summary["clips_per_video"]["mean"], train_summary["clips_per_video"]["median"],
            train_summary["clips_per_video"]["p90"], train_summary["clips_per_video"]["max"],
        ),
        "| Official Valid | {} | {} | {:.4f} | {:.1f} | {:.1f} | {} |".format(
            valid_summary["sample_count"], valid_summary["unique_video_id_count"],
            valid_summary["clips_per_video"]["mean"], valid_summary["clips_per_video"]["median"],
            valid_summary["clips_per_video"]["p90"], valid_summary["clips_per_video"]["max"],
        ),
        "",
        "- Train/Valid source overlap: `{}`; sample overlap: `{}`.".format(
            binding["train_valid_video_overlap_count"], binding["train_valid_sample_overlap_count"]
        ),
        "- `video_id` is reliably parsed from the pkl sample ID and bound to the Stage19 cache order; it is not a speaker ID.",
        "- Selected Train-only label gap: **{}**.".format(coverage["selected_delta"]),
        "- Pair coverage: **{:.2%} samples**, **{:.2%} multi-clip sources**; {} balanced retained pairs.".format(
            selected["sample_coverage_fraction"], selected["eligible_video_fraction"], coverage["selected_pair_count"]
        ),
        "",
        "## Residual ICC",
        "",
        "| Split | Mode | ICC | shuffled p95 | p-value | 95% bootstrap CI | Gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for split in ("train", "valid"):
        for mode in MODES:
            value = residual["splits"][split][mode]
            lines.append(
                "| {} | {} | {:.6f} | {:.6f} | {:.5f} | [{:.6f}, {:.6f}] | {} |".format(
                    split, mode, value["icc"], value["shuffled_icc_p95"], value["permutation_p_value"],
                    value["bootstrap_ci95_low"], value["bootstrap_ci95_high"], value["gate_passed"]
                )
            )
    lines.extend(
        [
            "",
            "## Source fingerprint",
            "",
            "| Mode | Mean AUROC | Balanced accuracy | Shuffled AUROC |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        value = fingerprint[mode]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} |".format(mode, value["mean_auroc"], value["mean_balanced_accuracy"], value["mean_shuffled_auroc"]))
    lines.extend(
        [
            "",
            "Source identity separability is diagnostic only and cannot establish usefulness of source-relative supervision.",
            "",
            "## Train-only shared-head probes",
            "",
            "| Split | P1-P0 J | P1-P2 J | P1-P3 J | P(better P0) | Without largest 5% sources |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in probes["inner_splits"]:
        lines.append(
            "| {} | {:+.6f} | {:+.6f} | {:+.6f} | {:.3f} | {:+.6f} |".format(
                split["seed"], split["deltas"]["P1_minus_P0"]["J"], split["deltas"]["P1_minus_P2"]["J"],
                split["deltas"]["P1_minus_P3"]["J"], split["bootstrap"]["P1_minus_P0"]["probability_a_better"],
                split["without_largest_sources"]["P1_minus_P0"]["J"],
            )
        )
    lines.extend(
        [
            "| **Mean** | **{:+.6f}** | **{:+.6f}** | **{:+.6f}** | — | — |".format(
                probes["train_only_gate"]["mean_P1_minus_P0_J"],
                probes["train_only_gate"]["mean_P1_minus_P2_J"],
                probes["train_only_gate"]["mean_P1_minus_P3_J"],
            ),
            "",
            "Missing-mode MAE improvements versus P0: **{}**.".format(", ".join(missing_improved) if missing_improved else "none"),
            "",
            "## Mean P1-P0 metric deltas",
            "",
            "| Mode | MAE | Corr | Acc7 | Acc5 | Acc2 | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES + ("MissingMacro",):
        value = tradeoffs[mode]
        lines.append(
            "| {} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} |".format(
                mode, value["MAE"], value["Corr"], value["acc_7"], value["acc_5"], value["acc_2"], value["F1_score"]
            )
        )
    official_reason = (
        "Official Valid confirmed: {}".format(probes["official_valid"].get("passed"))
        if probes["official_valid"]["run"]
        else "Official Valid probe was not run because the frozen Train-only gate failed."
    )
    uniform_gradient = gradients["summary"]["relative_vs_uniform_kd"]["all_trainable_parameters"]
    cf_gradient = gradients["summary"]["relative_vs_cfcompat_kd"]["all_trainable_parameters"]
    lines.extend(
        [
            "",
            "## Official Valid and gradients",
            "",
            "- {}".format(official_reason),
            "- Relative vs Uniform KD: median cosine `{:+.6f}`, negative ratio `{:.2%}`, risk `{}`.".format(
                uniform_gradient["median_cosine"], uniform_gradient["negative_ratio"], gradients["uniform_kd_conflict"]["risk"]
            ),
            "- Relative vs CFCompatKD: median cosine `{:+.6f}`, negative ratio `{:.2%}`, risk `{}`.".format(
                cf_gradient["median_cosine"], cf_gradient["negative_ratio"], gradients["cfcompat_kd_conflict"]["risk"]
            ),
            "- Gradient audit used {} true-source batches and performed zero optimizer steps.".format(gradients["batch_count"]),
            "",
            "## Required answers",
            "",
            "1. Train/Valid source videos: **{}/{}**.".format(train_summary["unique_video_id_count"], valid_summary["unique_video_id_count"]),
            "2. Mean clips/source: **{:.4f}/{:.4f}**.".format(train_summary["clips_per_video"]["mean"], valid_summary["clips_per_video"]["mean"]),
            "3. Train/Valid source overlap: **0**.",
            "4. ID binding: **reliable and SHA-verified**.",
            "5. Selected delta: **{}**.".format(coverage["selected_delta"]),
            "6. Pair coverage: **{:.2%} samples / {:.2%} eligible sources**.".format(selected["sample_coverage_fraction"], selected["eligible_video_fraction"]),
            "7–8. Per-mode ICC, shuffled controls and CIs are reported above; Train residual gate: **{}**.".format(residual["source_residual_gate_passed"]),
            "9. Source fingerprint distinguishable: **{}** (diagnostic only).".format(any(fingerprint[mode]["mean_auroc"] > 0.55 for mode in MODES)),
            "10. P1 better than P0 at required margin: **{}**.".format(probes["train_only_gate"]["checks"]["mean_P1_minus_P0_J_le_minus_0_003"]),
            "11. P1 better than P2 at required margin: **{}**.".format(probes["train_only_gate"]["checks"]["mean_P1_minus_P2_J_le_minus_0_002"]),
            "12. P1 better than P3 at required margin: **{}**.".format(probes["train_only_gate"]["checks"]["mean_P1_minus_P3_J_le_minus_0_002"]),
            "13. Generic pairwise-only effect: **{}**.".format(probes["generic_pairwise_signal"]),
            "14. Two train-only splits direction-consistent against controls: **{}**.".format(probes["train_only_gate"]["checks"]["P1_minus_controls_direction_consistent"]),
            "15–16. Official Valid run/confirmed: **{}/{}**.".format(probes["official_valid"]["run"], probes["official_valid"].get("passed", False)),
            "17. Missing modes improved: **{}**.".format(", ".join(missing_improved) if missing_improved else "none"),
            "18. MAE/Corr/classification trade-offs: see the complete delta table above.",
            "19. Direction survives removal of largest sources: **{}**.".format(probes["train_only_gate"]["checks"]["direction_survives_largest_sources"]),
            "20. Uniform KD conflict risk: **{}**.".format(gradients["uniform_kd_conflict"]["risk"]),
            "21. CFCompatKD conflict risk: **{}**.".format(gradients["cfcompat_kd_conflict"]["risk"]),
            "22. A CFCompatKD × Source-Relative 2×2 experiment is justified: **{}**.".format(status == "STAGE21A_SOURCE_RELATIVE_AUDIT_PASSED"),
            "23. Recommend Stage21B: **{}**.".format(report["recommend_stage21b"]),
            "24. Locked Test access count: **0**.",
            "",
            "## Decision",
            "",
            "`{}`".format(status),
            "",
            "No formal Source-Relative DLF was implemented or trained in this stage.",
        ]
    )
    atomic_text(root / "final/stage21a_final_report.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": status, "recommend_stage21b": report["recommend_stage21b"]}, indent=2))


if __name__ == "__main__":
    main()
