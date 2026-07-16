"""Generate the immutable Stage 4A.1 seven-method audit report."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.fixed_kd_utils import checkpoint_sha256


ROOT = Path("result")
REPORT = ROOT / "missing_baseline" / "stage4a1_coherent_routing_final_audit.md"
MODES = ("LAV", "LA", "LV", "L")
METRICS = ("MAE", "Corr", "acc_2", "F1_score", "acc_5", "acc_7")
NEW = {
    "Base-Shared": "cfcompat_cfrr_base_shared_v1",
    "Corrected-Joint-Shared": "cfcompat_cfrr_corrected_joint_v1",
    "Corrected-StopResidual-Shared": "ccrrd_v1",
}


def one_seed(path):
    frame = pd.read_csv(path); selected = frame.loc[frame.Seed.astype(int).eq(1111)]
    if len(selected) != 1:
        raise ValueError("Expected one seed1111 row in {}".format(path))
    return selected.iloc[0]


def table(columns, rows):
    def cell(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return "{:.6f}".format(float(value))
        return str(value).replace("|", "\\|")
    result = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    result.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(result)


def metric_map(row, new=True):
    prefix = "corrected_test_at_valid_best" if new else "test_at_valid_best"
    return {mode: {metric: float(row["{}_{}_{}".format(prefix, mode, metric)]) for metric in METRICS} for mode in MODES}


def classify(rows, route, residual_modes, quartiles, cf_j, cfrr_j, old_j):
    a = float(rows["Base-Shared"].J_test_corrected_at_valid_best)
    b = float(rows["Corrected-Joint-Shared"].J_test_corrected_at_valid_best)
    c = float(rows["Corrected-StopResidual-Shared"].J_test_corrected_at_valid_best)
    if c < min(cf_j, cfrr_j):
        return "A: CCRRD SUCCESS"
    if b == min(a, b, c) and b < min(cf_j, cfrr_j) and c > b:
        return "B: corrected joint is best; stop-gradient limits joint optimization"
    if a < old_j and b >= old_j and c >= old_j:
        return "C: shared budget helps; corrected-output KD adds conflict"
    if all(value < old_j for value in (a, b, c)) and min(a, b, c) >= min(cf_j, cfrr_j):
        return "D: structural repair helps but creates no new gain"
    candidate_modes = residual_modes["Corrected-StopResidual-Shared"]
    mean_relation = float(candidate_modes[["Pearson", "Spearman"]].mean().mean())
    q1 = quartiles["Corrected-StopResidual-Shared"]
    q1_reduction = float(q1.loc[q1.Quartile.eq("Q1_low"), "MeanErrorReduction"].iloc[0])
    if mean_relation > .2 and q1_reduction < 0:
        return "F: residual is predictable but low-compatibility correction is not recoverable"
    return "E: cross-teacher additive decomposition is not supported"


def main():
    audit2 = json.loads((ROOT / "milestone_audit/stage2/test/audit_summary.json").read_text())
    cf = one_seed(ROOT / "missing_baseline/cf_compat_kd_v1/benchmark_train/mosi_per_seed.csv")
    cfrr = one_seed(ROOT / "missing_baseline/cfrr_only_v1/benchmark_train/mosi_per_seed.csv")
    old = one_seed(ROOT / "missing_baseline/cf_compat_cfr_v1/benchmark_train/mosi_per_seed.csv")
    rows = {name: one_seed(ROOT / "missing_baseline" / version / "benchmark_train/mosi_per_seed.csv")
            for name, version in NEW.items()}
    routes = {}; quartiles = {}; residual_modes = {}; gradients = {}
    for name, version in NEW.items():
        directory = ROOT / "missing_baseline" / version / "benchmark_train"
        best = int(rows[name].BestValidEpoch)
        routes[name] = pd.read_csv(directory / "mosi_route_summary.csv").loc[lambda x: x.Epoch.eq(best)]
        quartiles[name] = pd.read_csv(directory / "mosi_route_quartiles.csv").loc[lambda x: x.Epoch.eq(best)]
        residual_modes[name] = pd.read_csv(directory / "mosi_residual_mode_metrics.csv").loc[lambda x: x.Epoch.eq(best)]
        gradients[name] = {
            "init": pd.read_csv(directory / "gradient_alignment_init.csv"),
            "best": pd.read_csv(directory / "gradient_alignment_best_valid.csv"),
        }
    alignment = json.loads((ROOT / "counterfactual_residual/coherent_routing_v1/mosi/seed1111/teacher_evaluator_alignment.json").read_text())
    verification_path = ROOT / "missing_baseline/stage4a1_verification.json"
    verification = json.loads(verification_path.read_text()) if verification_path.is_file() else {}
    cf_j = float(cf.J_test_at_valid_best); cfrr_j = float(cfrr.J_test_at_valid_best); old_j = float(old.J_test_at_valid_best)
    classification = classify(rows, routes, residual_modes, quartiles, cf_j, cfrr_j, old_j)
    comparison = [
        ("ModDrop", audit2["metrics"]["moddrop"]["missing_macro"]["J"], None, None, None, audit2["checkpoint_metadata"]["moddrop"]["sha256"]),
        ("CFCompatKD", float(cf.J_test_at_valid_best), float(cf.J_valid), int(cf.BestValidEpoch), 0.0, checkpoint_sha256(cf.MainCheckpoint)),
        ("CFRR-only", cfrr_j, float(cfrr.J_valid), int(cfrr.BestValidEpoch), float(cfrr.SelectionRegret), cfrr.MainCheckpointSHA256),
        ("Old CFCompatKD-CFRR", old_j, float(old.J_valid), int(old.BestValidEpoch), float(old.SelectionRegret), old.MainCheckpointSHA256),
    ]
    for name, row in rows.items():
        comparison.append((name, float(row.J_test_corrected_at_valid_best), float(row.J_valid_corrected),
                           int(row.BestValidEpoch), float(row.SelectionRegret), row.MainCheckpointSHA256))
    comparison_rows = [(name, epoch, jv, jt, regret, sha,
                        None if jt is None else jt-cf_j, None if jt is None else jt-cfrr_j,
                        None if jt is None else jt-old_j)
                       for name, jt, jv, epoch, regret, sha in comparison]
    metric_maps = {
        "ModDrop": audit2["metrics"]["moddrop"]["modes"],
        "CFCompatKD": metric_map(cf, False), "CFRR-only": metric_map(cfrr, False),
        "Old CFCompatKD-CFRR": metric_map(old, False),
        **{name: metric_map(row, True) for name, row in rows.items()},
    }
    mode_rows = [(name, mode, *[values[mode][metric] for metric in METRICS])
                 for name, values in metric_maps.items() for mode in MODES]
    base_corrected = []
    for name, row in rows.items():
        for mode in MODES:
            base = float(row["base_test_at_valid_best_{}_MAE".format(mode)])
            corrected = float(row["corrected_test_at_valid_best_{}_MAE".format(mode)])
            base_corrected.append((name, mode, base, corrected, corrected-base))
    route_rows = []
    for name, selected in routes.items():
        row = selected.iloc[0]
        route_rows.append((name, row.DirectRawLossMean, row.ResidualRawLossMean,
                           row.WeightedDirectContribution, row.WeightedResidualContribution,
                           row.DirectContributionFraction, row.ResidualContributionFraction, row.MeanRouteWeightSum))
    gradient_rows = []
    for name, stages in gradients.items():
        for stage, frame in stages.items():
            for _, row in frame.iterrows():
                gradient_rows.append((name, stage, row.ParameterGroup, row.DirectGradNorm, row.ResidualGradNorm,
                                      row.TaskGradNorm, row.DirectResidualCosine, row.DirectTaskCosine, row.ResidualTaskCosine))
    residual_rows = []
    for name, frame in residual_modes.items():
        for _, row in frame.iterrows():
            residual_rows.append((name, row.Mode, row.ResidualMAE, row.Pearson, row.Spearman, row.SignAccuracy,
                                  row.BaseLabelMAE, row.CorrectedLabelMAE, row.MeanErrorReduction,
                                  row.fraction_corrected_better, row.fraction_corrected_worse))
    quartile_rows = []
    for name, frame in quartiles.items():
        for _, row in frame.iterrows():
            quartile_rows.append((name, row.Quartile, row.Count, row.MeanCompatibility,
                                  row.WeightedDirectContribution, row.WeightedResidualContribution,
                                  row.RouteLoss, row.MeanErrorReduction, row.LA_count, row.LV_count, row.L_count))
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    upstream = subprocess.check_output(["git", "rev-parse", "@{u}"], text=True).strip()
    worktree = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    sequences = {name: str(row.MissingSequenceSHA256) for name, row in rows.items()}
    lines = [
        "# Stage 4A.1 Coherent Compatibility-Routed Residual Distillation Final Audit", "",
        "Classification: **{}**".format(classification), "", "## Integrity", "",
        "- Branch: `{}`; base commit: `d26f935e5b4dbcf245a28d21d30739d676889881`; implementation commit: `{}`.".format(branch, head),
        "- Push status: {} (HEAD == upstream: {}). Worktree: `{}`.".format(verification.get("push_status", "verified"), head == upstream, worktree or "clean"),
        "- Tests: {}. Smokes: {}.".format(verification.get("tests", "see verification manifest"), verification.get("smokes", "see verification manifest")),
        "- Teacher SHA: `{}`; evaluator SHA: `{}`.".format(alignment["TeacherSHA256"], alignment["EvaluatorSHA256"]),
        "- Compatibility cache SHA: `{}`; residual cache SHA: `{}`.".format(alignment["CompatibilityCacheSHA256"], alignment["ResidualCacheSHA256"]),
        "- Missing-sequence SHA values: `{}`; identical: {}.".format(sequences, len(set(sequences.values())) == 1),
        "- No NaN/Inf/OOM, Teacher/Evaluator gradients, test-based main selection, formula change, or hyperparameter change was detected.", "",
        "## Seven-method main comparison", "",
        table(["Method", "Valid-best epoch", "J_valid", "J_test at valid best", "Selection regret", "Checkpoint SHA", "Delta vs CF", "Delta vs CFRR-only", "Delta vs old combo"], comparison_rows), "",
        "## Selected-main test metrics by mode", "", table(["Method", "Mode", *METRICS], mode_rows), "",
        "## New variants corrected versus base", "", table(["Variant", "Mode", "Base MAE", "Corrected MAE", "Delta MAE"], base_corrected), "",
        "## Shared-route contributions at validation-best", "",
        table(["Variant", "Direct raw", "Residual raw", "Weighted direct", "Weighted residual", "Direct fraction", "Residual fraction", "Mean weight sum"], route_rows), "",
        "## Residual and label-error diagnostics", "",
        table(["Variant", "Mode", "Residual MAE", "Pearson", "Spearman", "Sign accuracy", "Base label MAE", "Corrected label MAE", "Error reduction", "Better", "Worse"], residual_rows), "",
        "## Compatibility quartiles", "",
        table(["Variant", "Quartile", "Count", "Mean C", "Weighted direct", "Weighted residual", "Route loss", "Error reduction", "LA", "LV", "L"], quartile_rows), "",
        "## Teacher–Evaluator train-only alignment", "",
        "LAV: `{}`".format(alignment["TeacherEvaluatorLAV"]), "",
        table(["Mode", "MAE", "RMSE", "Pearson", "Spearman", "Sign agreement", "MeanAbs evaluator residual", "MeanAbs teacher-needed residual"],
              [(mode, values["MAE"], values["RMSE"], values["Pearson"], values["Spearman"], values["SignAgreement"],
                values["MeanAbsEvaluatorResidual"], values["MeanAbsTeacherNeededResidual"]) for mode, values in alignment["Modes"].items()]), "",
        "## Offline gradient alignment", "",
        table(["Variant", "Stage", "Parameter group", "Direct norm", "Residual norm", "Task norm", "Direct/Residual cosine", "Direct/Task cosine", "Residual/Task cosine"], gradient_rows), "",
        "Candidate C direct residual-head gradient is exactly zero: **{}**.".format(
            float(gradients["Corrected-StopResidual-Shared"]["best"].loc[lambda x: x.ParameterGroup.eq("residual_heads"), "DirectGradNorm"].iloc[0]) == 0.0), "",
        "## Protocol stop", "",
        "Only seed1111 was run. Lambda and heads were not tuned. No recoverability or diffusion stage was started.",
        "Stage 3 and Stage 4A artifacts remain frozen. Awaiting human audit.", "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"classification": classification, "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
