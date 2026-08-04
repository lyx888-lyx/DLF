"""Frozen MOSI long-tail and semantic-risk coupling analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trains.singleTask.dlf_role_specialization_utils import long_tail_status
from trains.singleTask.dlf_tail_risk_utils import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CATASTROPHIC_ABS_ERROR,
    FORMAL_SEEDS,
    HEAD_BIN_COUNT,
    METHOD,
    MODES,
    OUTPUT_TAG,
    SOURCE_OUTPUT_TAG,
    TAIL_BIN_COUNT,
    VERSION,
    add_risk_events,
    compute_bin_and_run_metrics,
    coupling_gate,
    joint_video_bootstrap,
    sha256_file,
    tail_head_definition,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit whether MOSI rarity is coupled to baseline error and semantic risk."
    )
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--source-dir")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.bootstrap_replicates = 100
    elif int(args.bootstrap_replicates) != BOOTSTRAP_REPLICATES:
        parser.error(
            "Formal audit fixes --bootstrap-replicates to {}.".format(
                BOOTSTRAP_REPLICATES
            )
        )
    return args


def output_root(cli) -> Path:
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / "mosi"
        / "valid_only"
    )
    if cli.smoke_test:
        root = root / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_root(cli) -> Path:
    if cli.source_dir:
        return Path(cli.source_dir)
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / SOURCE_OUTPUT_TAG
        / "mosi"
        / "valid_train_audit"
    )


def load_source(cli):
    root = source_root(cli)
    required = {
        "predictions": root / "baseline_predictions_train_valid.csv",
        "distribution": root / "label_distribution.csv",
        "samples": root / "label_samples_train_valid.csv",
        "role_summary": root / "role_specialization_summary.json",
        "role_source": root / "role_specialization_source_manifest.json",
        "role_audit_v2": root / "role_specialization_audit_check_v2.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing independently audited role-study artifacts:\n"
            + "\n".join(missing)
        )
    role_summary = json.loads(
        required["role_summary"].read_text(encoding="utf-8")
    )
    role_source = json.loads(
        required["role_source"].read_text(encoding="utf-8")
    )
    role_audit = json.loads(
        required["role_audit_v2"].read_text(encoding="utf-8")
    )
    if not role_audit.get("passed", False):
        raise RuntimeError("Source role-specialization v2 audit did not pass.")
    if role_source.get("official_test_constructed", True):
        raise RuntimeError("Source artifact reports official Test construction.")
    if role_source.get("test_loader_construction_count", -1) != 0:
        raise RuntimeError("Source artifact reports a Test loader construction.")
    if role_summary["protocol"].get("official_test_constructed", True):
        raise RuntimeError("Source summary reports official Test construction.")
    predictions = pd.read_csv(required["predictions"])
    distribution = pd.read_csv(required["distribution"])
    samples = pd.read_csv(required["samples"])
    if "test" in set(predictions.Split.astype(str)) or "test" in set(samples.Split.astype(str)):
        raise RuntimeError("Source tables unexpectedly contain official Test rows.")
    binding = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in required.items()
    }
    return root, predictions, distribution, samples, role_summary, role_source, binding


def render_report(summary):
    gate = summary["coupling_gate"]
    definition = summary["frequency_definition"]
    lines = [
        "# DLF MOSI tail-risk coupling audit v1",
        "",
        "## Decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Tail-weight coupling supported: `{}`".format(
            gate["mae_coupling_supported"]
        ),
        "- Semantic-risk coupling supported: `{}`".format(
            gate["semantic_risk_coupling_supported"]
        ),
        "- Official Test constructed: `False`",
        "- Model training performed: `False`",
        "",
        "## Frequency groups",
        "",
        "- Tail bins: `{}`".format(definition["tail_bins"]),
        "- Head bins: `{}`".format(definition["head_bins"]),
        "- Middle bin: `{}`".format(definition["middle_bins"]),
        "",
        "## MAE coupling",
        "",
        "- Mean tail-head macro-MAE gap: `{:+.6f}`".format(
            gate["mean_tail_head_macro_mae_gap"]
        ),
        "- Mean count-versus-MAE Spearman: `{:+.6f}`".format(
            gate["mean_count_vs_mae_spearman"]
        ),
        "- Positive gap coverage: `{:.3f}`".format(
            gate["mae_positive_gap_coverage"]
        ),
        "- Joint video-bootstrap 95% CI: `[{:+.6f}, {:+.6f}]`".format(
            gate["mae_bootstrap"]["ci95_low"],
            gate["mae_bootstrap"]["ci95_high"],
        ),
        "",
        "## High-cost semantic-risk coupling",
        "",
        "A high-cost event is a direct positive-negative flip or absolute error "
        "of at least 2.0. Neutralization and neutral escape are reported separately.",
        "",
        "- Mean tail-head high-cost-rate gap: `{:+.6f}`".format(
            gate["mean_tail_head_high_cost_rate_gap"]
        ),
        "- Mean count-versus-high-cost Spearman: `{:+.6f}`".format(
            gate["mean_count_vs_high_cost_spearman"]
        ),
        "- Positive gap coverage: `{:.3f}`".format(
            gate["risk_positive_gap_coverage"]
        ),
        "- Joint video-bootstrap 95% CI: `[{:+.6f}, {:+.6f}]`".format(
            gate["risk_bootstrap"]["ci95_low"],
            gate["risk_bootstrap"]["ci95_high"],
        ),
        "",
        "## Interpretation",
        "",
    ]
    verdict = summary["verdict"]
    if verdict == "PROMOTE_SEPARATE_FINAL_TAIL_AND_RISK_SCREENS":
        lines.append(
            "Both mechanisms are supported, but they must be trained in separate "
            "frozen screens before any combination."
        )
    elif verdict == "PROMOTE_FINAL_TAIL_WEIGHT_SCREEN_ONLY":
        lines.append(
            "Rarity is coupled to MAE, but semantic-risk concentration is not. "
            "Only a bounded final-layer tail-weight screen is authorized."
        )
    elif verdict == "PROMOTE_FINAL_SEMANTIC_RISK_SCREEN_ONLY":
        lines.append(
            "High-cost errors concentrate in rare bins without stable MAE coupling. "
            "Only a final-layer soft semantic-risk screen is authorized."
        )
    elif verdict == "PARTIAL_TAIL_RISK_COUPLING_DO_NOT_TRAIN":
        lines.append(
            "Some average signal exists, but the preregistered seed/view and "
            "video-bootstrap stability requirements are incomplete."
        )
    elif verdict == "STOP_LONG_TAIL_PREMISE_NOT_SUPPORTED":
        lines.append("The source train distribution does not meet the long-tail premise.")
    else:
        lines.append(
            "MOSI is imbalanced, but rarity is not consistently coupled to the "
            "measured errors across seeds and missing-modality views."
        )
    return "\n".join(lines) + "\n"


def main():
    cli = parse_args()
    root = output_root(cli)
    (
        source,
        predictions,
        distribution,
        samples,
        role_summary,
        role_source,
        source_binding,
    ) = load_source(cli)
    definition = tail_head_definition(distribution)
    events = add_risk_events(predictions)
    bins, runs = compute_bin_and_run_metrics(events, definition)
    bootstrap = joint_video_bootstrap(
        events,
        definition,
        replicates=cli.bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    long_tail = long_tail_status(distribution, "mosi")
    gate = coupling_gate(runs, bootstrap, long_tail["tail_present"])

    events.to_csv(root / "tail_risk_sample_events.csv", index=False)
    bins.to_csv(root / "tail_risk_bin_metrics.csv", index=False)
    runs.to_csv(root / "tail_risk_run_summary.csv", index=False)
    bootstrap.to_csv(root / "tail_risk_video_bootstrap.csv", index=False)

    manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_branch": "analysis/dlf-role-specialization-audit-v1",
        "source_directory": str(source.resolve()),
        "source_binding": source_binding,
        "source_role_verdict": role_summary["verdict"],
        "formal_seeds": list(FORMAL_SEEDS),
        "modes": list(MODES),
        "tail_bin_count": TAIL_BIN_COUNT,
        "head_bin_count": HEAD_BIN_COUNT,
        "catastrophic_absolute_error_threshold": CATASTROPHIC_ABS_ERROR,
        "bootstrap_replicates": int(cli.bootstrap_replicates),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "model_loaded": False,
        "optimizer_constructed": False,
        "backward_called": False,
        "model_parameters_updated": False,
        "artifacts": {},
    }
    artifact_names = [
        "tail_risk_sample_events.csv",
        "tail_risk_bin_metrics.csv",
        "tail_risk_run_summary.csv",
        "tail_risk_video_bootstrap.csv",
    ]
    for name in artifact_names:
        path = root / name
        manifest["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    (root / "tail_risk_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": gate["verdict"],
        "long_tail": long_tail,
        "frequency_definition": definition,
        "coupling_gate": gate,
        "protocol": {
            "source_split": "official_train_frequency_and_official_valid_error_only",
            "official_test_constructed": False,
            "official_test_authorized": False,
            "model_training_performed": False,
            "tail_and_risk_are_separate_decisions": True,
            "high_cost_event_definition": (
                "direct_positive_negative_flip_or_absolute_error_ge_2"
            ),
            "video_cluster_bootstrap": True,
            "bootstrap_replicates": int(cli.bootstrap_replicates),
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    (root / "tail_risk_coupling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "tail_risk_coupling_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print("DLF tail-risk coupling audit complete")
    print("verdict:", gate["verdict"])
    print("tail MAE coupling supported:", gate["mae_coupling_supported"])
    print(
        "semantic-risk coupling supported:",
        gate["semantic_risk_coupling_supported"],
    )
    print(
        "mean tail-head macro MAE gap:",
        "{:+.6f}".format(gate["mean_tail_head_macro_mae_gap"]),
    )
    print(
        "mean tail-head high-cost gap:",
        "{:+.6f}".format(gate["mean_tail_head_high_cost_rate_gap"]),
    )
    print("official Test was not constructed")
    print("report:", root / "tail_risk_coupling_report.md")


if __name__ == "__main__":
    main()
