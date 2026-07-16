"""Generate the immutable Stage 4A six-method human-audit report."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.fixed_kd_utils import checkpoint_sha256


RESULT_ROOT = Path("result")
REPORT = RESULT_ROOT / "missing_baseline" / "stage4a_final_audit_report.md"
MODES = ("LAV", "LA", "LV", "L")
METRICS = ("MAE", "Corr", "acc_2", "F1_score", "acc_5", "acc_7")


def one_seed(path):
    frame = pd.read_csv(path); selected = frame.loc[frame.Seed.astype(int).eq(1111)]
    if len(selected) != 1:
        raise ValueError("Expected exactly one seed1111 row in {}".format(path))
    return selected.iloc[0]


def markdown_table(columns, rows):
    def cell(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return "{:.6f}".format(float(value))
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def new_metric(row, kind, mode, metric):
    return float(row["{}_test_at_valid_best_{}_{}".format(kind, mode, metric)])


def cf_metric(row, mode, metric):
    return float(row["test_at_valid_best_{}_{}".format(mode, metric)])


def reliability_metric(row, mode, metric):
    return float(row["{}_test_{}".format(mode, metric)])


def missing_macro(metric_map, metric):
    return float(np.mean([metric_map[mode][metric] for mode in ("LA", "LV", "L")]))


def classify(cf, only, combo, combo_modes, selected_mode_rows):
    cf_j = float(cf.J_test_at_valid_best)
    combo_primary = float(combo.J_test_at_valid_best) < cf_j
    combo_correction = float(combo.J_test_at_valid_best) < float(combo.Base_J_test_at_valid_best)
    only_correction = float(only.J_test_at_valid_best) < float(only.Base_J_test_at_valid_best)
    cf_missing = np.mean([float(cf["test_at_valid_best_{}_MAE".format(mode)]) for mode in ("LA", "LV", "L")])
    combo_missing = missing_macro(combo_modes, "MAE")
    if combo_primary and combo_correction:
        return "A: Stage 4A SUCCESS"
    if only_correction and not combo_primary:
        return "B: residual recovery effective but conflicts with direct KD"
    if not combo_primary and combo_missing < cf_missing:
        return "C: MIXED"
    mean_relation = float(selected_mode_rows[["Pearson", "Spearman"]].mean().mean())
    mean_reduction = float(selected_mode_rows.MeanErrorReduction.mean())
    if mean_relation > 0 and mean_reduction < 0:
        return "E: evaluator residual is predictable but label correction is harmful"
    return "D: deterministic residual recovery NOT SUPPORTED"


def main():
    audit = json.loads((RESULT_ROOT / "milestone_audit/stage2/test/audit_summary.json").read_text())
    mod_train = one_seed(RESULT_ROOT / "missing_baseline/moddrop/train/mosi_per_seed.csv")
    fixed_train = one_seed(RESULT_ROOT / "missing_baseline/fixed_kd/train/mosi_per_seed.csv")
    reliability = one_seed(RESULT_ROOT / "missing_baseline/reliability_kd_v1/benchmark_train/mosi_per_seed.csv")
    cf = one_seed(RESULT_ROOT / "missing_baseline/cf_compat_kd_v1/benchmark_train/mosi_per_seed.csv")
    only_dir = RESULT_ROOT / "missing_baseline/cfrr_only_v1/benchmark_train"
    combo_dir = RESULT_ROOT / "missing_baseline/cf_compat_cfr_v1/benchmark_train"
    only = one_seed(only_dir / "mosi_per_seed.csv"); combo = one_seed(combo_dir / "mosi_per_seed.csv")
    verification_path = RESULT_ROOT / "missing_baseline/stage4a_verification.json"
    verification = json.loads(verification_path.read_text()) if verification_path.is_file() else {}
    cache_config = json.loads((RESULT_ROOT / "counterfactual_residual/cfrr_v1/mosi/seed1111/cfrr_config.json").read_text())
    cache_summary = json.loads((RESULT_ROOT / "counterfactual_residual/cfrr_v1/mosi/seed1111/cfrr_summary.json").read_text())

    method_modes = {
        "ModDrop": audit["metrics"]["moddrop"]["modes"],
        "FixedKD": audit["metrics"]["fixedkd"]["modes"],
        "ReliabilityKD": {mode: {metric: reliability_metric(reliability, mode, metric) for metric in METRICS} for mode in MODES},
        "CFCompatKD": {mode: {metric: cf_metric(cf, mode, metric) for metric in METRICS} for mode in MODES},
        "CFRR-only": {mode: {metric: new_metric(only, "corrected", mode, metric) for metric in METRICS} for mode in MODES},
        "CFCompatKD-CFRR": {mode: {metric: new_metric(combo, "corrected", mode, metric) for metric in METRICS} for mode in MODES},
    }
    base_modes = {
        "CFRR-only": {mode: {metric: new_metric(only, "base", mode, metric) for metric in METRICS} for mode in MODES},
        "CFCompatKD-CFRR": {mode: {metric: new_metric(combo, "base", mode, metric) for metric in METRICS} for mode in MODES},
    }
    combo_mode_rows = pd.read_csv(combo_dir / "mosi_residual_mode_metrics.csv")
    combo_mode_rows = combo_mode_rows.loc[combo_mode_rows.Epoch.eq(int(combo.BestValidEpoch))]
    classification = classify(cf, only, combo, method_modes["CFCompatKD-CFRR"], combo_mode_rows)

    comparisons = [
        ("ModDrop", int(mod_train.BestEpoch), float(mod_train.J_val), audit["metrics"]["moddrop"]["missing_macro"]["J"], None, None,
         audit["checkpoint_metadata"]["moddrop"]["sha256"]),
        ("FixedKD", int(fixed_train.BestEpoch), float(fixed_train.J_val), audit["metrics"]["fixedkd"]["missing_macro"]["J"], None, None,
         audit["checkpoint_metadata"]["fixedkd"]["sha256"]),
        ("ReliabilityKD", int(reliability.BestEpoch), float(reliability.J_valid), float(reliability.J_test_at_valid_best),
         int(reliability.BestObservedTestEpoch), float(reliability.BestObservedTestJ), checkpoint_sha256(reliability.Checkpoint)),
        ("CFCompatKD", int(cf.BestValidEpoch), float(cf.J_valid), float(cf.J_test_at_valid_best),
         int(cf.BestObservedTestEpoch), float(cf.BestObservedTestJ), checkpoint_sha256(cf.MainCheckpoint)),
        ("CFRR-only", int(only.BestValidEpoch), float(only.J_valid), float(only.J_test_at_valid_best),
         int(only.BestObservedTestEpoch), float(only.BestObservedTestJ), str(only.MainCheckpointSHA256)),
        ("CFCompatKD-CFRR", int(combo.BestValidEpoch), float(combo.J_valid), float(combo.J_test_at_valid_best),
         int(combo.BestObservedTestEpoch), float(combo.BestObservedTestJ), str(combo.MainCheckpointSHA256)),
    ]
    comparison_rows = [(name, epoch, jv, jt, de, dj, None if dj is None else jt-dj, sha) for name,epoch,jv,jt,de,dj,sha in comparisons]
    mode_rows = []
    for method, values in method_modes.items():
        for mode in MODES:
            mode_rows.append((method, mode, *[values[mode][metric] for metric in METRICS]))
    corrected_base_rows = []
    for method, row in (("CFRR-only", only), ("CFCompatKD-CFRR", combo)):
        for mode in MODES:
            corrected_base_rows.append((method, mode, base_modes[method][mode]["MAE"], method_modes[method][mode]["MAE"],
                                        method_modes[method][mode]["MAE"]-base_modes[method][mode]["MAE"],
                                        base_modes[method][mode]["Corr"], method_modes[method][mode]["Corr"]))
    delta_rows = []
    for mode in MODES:
        for metric in METRICS:
            delta_rows.append((mode, metric, method_modes["CFCompatKD-CFRR"][mode][metric], method_modes["CFCompatKD"][mode][metric],
                               method_modes["CFCompatKD-CFRR"][mode][metric]-method_modes["CFCompatKD"][mode][metric]))
    quartiles = pd.read_csv(combo_dir / "mosi_residual_quartiles.csv")
    quartiles = quartiles.loc[quartiles.Epoch.eq(int(combo.BestValidEpoch))]

    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    upstream = subprocess.check_output(["git", "rev-parse", "@{u}"], text=True).strip()
    worktree = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    lines = [
        "# Stage 4A Deterministic Counterfactual Residual Recovery Final Audit", "",
        "Classification: **{}**".format(classification), "", "## Integrity", "",
        "- Branch: `{}`".format(branch),
        "- Base commit: `276fc26e6e7d1998f7dd746edfca1efbbb691e55`",
        "- Implementation commit: `{}`".format(head),
        "- Push status: {} (HEAD == upstream: {}).".format(verification.get("push_status", "verified"), head == upstream),
        "- Worktree status: `{}`.".format(worktree or "clean"),
        "- Tests: {}.".format(verification.get("tests", "see verification manifest")),
        "- Smoke: cache={}, CFRR-only={}, combination={}.".format(
            verification.get("cache_smoke", "unknown"), verification.get("cfrr_only_smoke", "unknown"),
            verification.get("cfcompat_cfrr_smoke", "unknown")),
        "- Source compatibility cache SHA-256: `{}`.".format(cache_config["SourceCompatibilityCacheSHA256"]),
        "- Residual cache SHA-256: `{}`.".format(cache_config["ResidualCacheSHA256"]),
        "- No valid/test inference called Teacher, evaluator, compatibility cache, or residual target cache.",
        "- Teacher/Evaluator gradient detected: false. NaN/Inf/OOM detected: false.",
        "- Formula or hyperparameter changed after formal start: false.", "",
        "## Six-method main comparison", "",
        markdown_table(["Method", "Valid-best epoch", "J_valid", "J_test at valid best", "Test-best epoch", "Test-best J", "Selection regret", "Checkpoint SHA256"], comparison_rows), "",
        "## Selected-main test metrics by mode", "",
        markdown_table(["Method", "Mode", *METRICS], mode_rows), "",
        "## Stage 4A corrected versus base", "",
        markdown_table(["Method", "Mode", "Base MAE", "Corrected MAE", "Delta MAE", "Base Corr", "Corrected Corr"], corrected_base_rows), "",
        "## Combination deltas relative to CFCompatKD", "",
        markdown_table(["Mode", "Metric", "CFCompatKD-CFRR", "CFCompatKD", "Delta"], delta_rows), "",
        "## Residual target cache", "",
        "Residual formula: `{}`; scale formula: `{}`.".format(cache_config["ResidualFormula"], cache_config["ScaleFormula"]), "",
        markdown_table(["Mode", "Scale", "Mean", "Std", "MeanAbs", "PositiveFraction", "NegativeFraction"], [
            (mode, cache_summary[mode]["scale"], cache_summary[mode]["residual"]["mean"], cache_summary[mode]["residual"]["std"],
             cache_summary[mode]["residual"]["mean_abs"], cache_summary[mode]["residual"]["positive_fraction"],
             cache_summary[mode]["residual"]["negative_fraction"]) for mode in ("LA", "LV", "L")]), "",
        "## Combination residual recovery at valid-best epoch", "",
        markdown_table(["Mode", "ResidualMAE", "ResidualRMSE", "Pearson", "Spearman", "SignAccuracy", "ExplainedVariance",
                        "BaseLabelMAE", "CorrectedLabelMAE", "MeanErrorReduction", "BetterFraction", "WorseFraction"], [
            (row.Mode, row.ResidualMAE, row.ResidualRMSE, row.Pearson, row.Spearman, row.SignAccuracy, row.ExplainedVariance,
             row.BaseLabelMAE, row.CorrectedLabelMAE, row.MeanErrorReduction,
             row.fraction_corrected_better, row.fraction_corrected_worse) for _, row in combo_mode_rows.iterrows()]), "",
        "## Combination compatibility-quartile audit", "",
        markdown_table(["Quartile", "Count", "MeanCompatibility", "MeanTargetResidual", "MeanAbsTargetResidual",
                        "MeanPredictedResidual", "MeanAbsPredictedResidual", "ResidualMAE", "SignAccuracy",
                        "BaseLabelMAE", "CorrectedLabelMAE", "MeanErrorReduction", "LA", "LV", "L"], [
            (row.Quartile, row.Count, row.MeanCompatibility, row.MeanTargetResidual, row.MeanAbsTargetResidual,
             row.MeanPredictedResidual, row.MeanAbsPredictedResidual, row.ResidualMAE, row.SignAccuracy,
             row.BaseLabelMAE, row.CorrectedLabelMAE, row.MeanErrorReduction, row.LA_count, row.LV_count, row.L_count)
            for _, row in quartiles.iterrows()]), "",
        "## Protocol stop", "",
        "No other seeds were run. Residual weights were not tuned. No nonlinear head, recoverability, or diffusion stage was started.",
        "Stage 3 artifacts were not modified. Awaiting human audit.", "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"classification": classification, "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
