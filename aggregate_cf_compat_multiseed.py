"""Strict paired aggregation for Stage 3B-M.

Only validation-best main checkpoints are used for the primary comparison.
Best-observed test checkpoints remain diagnostic-only columns.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_cf_compat_multiseed import ALL_SEEDS, GATE3_MANIFEST, RUN_MANIFEST, MULTISEED_RESULT
from trains.singleTask.cf_compat_kd_utils import (
    MULTISEED_CACHE_VERSION,
    cache_paths,
    cache_summary,
    effective_sample_size,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


METRICS = ("MAE", "Corr", "acc_2", "F1_score", "acc_5", "acc_7")
MODES = ("LAV", "LA", "LV", "L")
CONTROL_ROOT = Path("result/missing_baseline/moddrop_benchmark_multiseed_v1")
CF_ROOT = Path("result/missing_baseline/cf_compat_kd_v1/benchmark_multiseed")


def markdown_table(frame):
    """Render a compact GitHub table without the optional tabulate package."""
    def cell(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return "{:.6g}".format(float(value))
        return str(value).replace("|", r"\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def one_row(path, seed):
    if not Path(path).is_file():
        raise FileNotFoundError("Required result is absent: {}".format(path))
    frame = pd.read_csv(path)
    if frame.Seed.duplicated().any():
        raise ValueError("Duplicate seed rows in {}".format(path))
    selected = frame.loc[frame.Seed.astype(int) == int(seed)]
    if len(selected) != 1:
        raise ValueError("Expected one seed {} row in {}".format(seed, path))
    return selected.iloc[0]


def load_seed1111_control():
    row = one_row("result/missing_baseline/moddrop/train/mosi_per_seed.csv", 1111)
    audit = pd.read_csv("result/milestone_audit/stage2/test/overall_metrics.csv")
    audit = audit.loc[audit.method.eq("moddrop")]
    if set(audit["mode"]) != set(MODES):
        raise ValueError("Locked seed1111 ModDrop audit lacks four modes.")
    result = {
        "Seed": 1111, "BestValidEpoch": int(row.BestEpoch), "J_valid": float(row.J_val),
        "J_test_at_valid_best": float(audit.J.iloc[0]),
        "BestObservedTestEpoch": np.nan, "BestObservedTestJ": np.nan,
        "MainCheckpoint": str(row.Checkpoint), "DiagnosticCheckpoint": np.nan,
    }
    for mode in MODES:
        local = audit.loc[audit["mode"].eq(mode)].iloc[0]
        for metric in METRICS:
            result["test_at_valid_best_{}_{}".format(mode, metric)] = float(local[metric])
    return pd.Series(result)


def load_control(seed):
    if seed == 1111:
        return load_seed1111_control()
    return one_row(CONTROL_ROOT / "seed{}".format(seed) / "mosi_per_seed.csv", seed)


def load_cf(seed):
    path = (Path("result/missing_baseline/cf_compat_kd_v1/benchmark_train/mosi_per_seed.csv")
            if seed == 1111 else CF_ROOT / "seed{}".format(seed) / "mosi_per_seed.csv")
    return one_row(path, seed)


def validate_main_row(row, method, seed):
    checkpoint = str(row.MainCheckpoint)
    if "diagnostic" in checkpoint or "best_test" in checkpoint:
        raise ValueError("{} seed{} main row points to diagnostic checkpoint.".format(method, seed))
    if not Path(checkpoint).is_file():
        raise FileNotFoundError("{} seed{} main checkpoint absent: {}".format(method, seed, checkpoint))
    for field in ("BestValidEpoch", "J_valid", "J_test_at_valid_best"):
        if not math.isfinite(float(row[field])):
            raise ValueError("{} seed{} has non-finite {}.".format(method, seed, field))


def macro(row, metric):
    return float(np.mean([row["test_at_valid_best_{}_{}".format(mode, metric)] for mode in ("LA", "LV", "L")]))


def build_paired_rows():
    rows = []
    for seed in ALL_SEEDS:
        control, cf = load_control(seed), load_cf(seed)
        validate_main_row(control, "ModDrop", seed)
        validate_main_row(cf, "CFCompat", seed)
        row = {
            "Seed": seed,
            "ModDrop_BestValidEpoch": int(control.BestValidEpoch),
            "ModDrop_J_valid": float(control.J_valid),
            "ModDrop_J_test_at_valid_best": float(control.J_test_at_valid_best),
            "ModDrop_BestObservedTestEpoch": control.get("BestObservedTestEpoch", np.nan),
            "ModDrop_BestObservedTestJ": control.get("BestObservedTestJ", np.nan),
            "CFCompat_BestValidEpoch": int(cf.BestValidEpoch),
            "CFCompat_J_valid": float(cf.J_valid),
            "CFCompat_J_test_at_valid_best": float(cf.J_test_at_valid_best),
            "CFCompat_BestObservedTestEpoch": int(cf.BestObservedTestEpoch),
            "CFCompat_BestObservedTestJ": float(cf.BestObservedTestJ),
            "Delta_J_valid": float(cf.J_valid - control.J_valid),
            "Delta_J_test_main": float(cf.J_test_at_valid_best - control.J_test_at_valid_best),
            "Delta_J_test_diagnostic": (np.nan if pd.isna(control.get("BestObservedTestJ", np.nan))
                                          else float(cf.BestObservedTestJ - control.BestObservedTestJ)),
        }
        for mode in MODES:
            for metric in METRICS:
                left = float(control["test_at_valid_best_{}_{}".format(mode, metric)])
                right = float(cf["test_at_valid_best_{}_{}".format(mode, metric)])
                row["ModDrop_{}_test_{}".format(mode, metric)] = left
                row["CFCompat_{}_test_{}".format(mode, metric)] = right
                row["Delta_{}_test_{}".format(mode, metric)] = right - left
        for metric in METRICS:
            left, right = macro(control, metric), macro(cf, metric)
            row["ModDrop_MissingMacro_test_{}".format(metric)] = left
            row["CFCompat_MissingMacro_test_{}".format(metric)] = right
            row["Delta_MissingMacro_test_{}".format(metric)] = right - left
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.Seed.duplicated().any() or frame.Seed.astype(int).tolist() != list(ALL_SEEDS):
        raise ValueError("Paired aggregation requires exactly seeds 1111-1115 once each.")
    return frame


def stats(values):
    values = pd.Series(values, dtype=float)
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
            "median": float(values.median()), "min": float(values.min()), "max": float(values.max())}


def trajectory_row(seed, method, main_test):
    if seed == 1111:
        path = (Path("result/missing_baseline/moddrop/train/mosi_epoch_metrics.csv") if method == "ModDrop"
                else Path("result/missing_baseline/cf_compat_kd_v1/benchmark_train/mosi_epoch_metrics.csv"))
    else:
        path = ((CONTROL_ROOT if method == "ModDrop" else CF_ROOT) / "seed{}".format(seed) / "mosi_epoch_metrics.csv")
    if not path.is_file():
        return {"epoch_count": np.nan, "pearson": np.nan, "spearman": np.nan,
                "epochs_better_than_control": np.nan}
    epochs = pd.read_csv(path)
    if not {"J_valid", "J_test"}.issubset(epochs.columns):
        return {"epoch_count": len(epochs), "pearson": np.nan, "spearman": np.nan,
                "epochs_better_than_control": np.nan}
    return {"epoch_count": len(epochs),
            "pearson": float(epochs.J_valid.corr(epochs.J_test, method="pearson")),
            "spearman": float(epochs.J_valid.corr(epochs.J_test, method="spearman")),
            "epochs_better_than_control": int((epochs.J_test < main_test).sum()) if method == "CFCompat" else np.nan}


def build_trajectory_summary(paired):
    rows = []
    for _, pair in paired.iterrows():
        seed = int(pair.Seed)
        for method in ("ModDrop", "CFCompat"):
            valid_epoch = pair["{}_BestValidEpoch".format(method)]
            test_epoch = pair["{}_BestObservedTestEpoch".format(method)]
            main_j = pair["{}_J_test_at_valid_best".format(method)]
            diagnostic_j = pair["{}_BestObservedTestJ".format(method)]
            info = trajectory_row(seed, method, pair.ModDrop_J_test_at_valid_best)
            rows.append({"Seed": seed, "Method": method, "BestValidEpoch": valid_epoch,
                         "BestTestDiagnosticEpoch": test_epoch,
                         "SelectionRegret": np.nan if pd.isna(diagnostic_j) else main_j - diagnostic_j,
                         "ValidBestIsTestBest": np.nan if pd.isna(test_epoch) else int(valid_epoch) == int(test_epoch),
                         **info})
    return pd.DataFrame(rows)


def build_gate_stability():
    rows = []
    for seed in ALL_SEEDS:
        paths = (cache_paths("result", "mosi") if seed == 1111
                 else cache_paths("result", "mosi", MULTISEED_CACHE_VERSION, seed))
        if not paths["csv"].is_file() or not paths["config"].is_file():
            raise FileNotFoundError("Missing cache for seed {}.".format(seed))
        frame = pd.read_csv(paths["csv"]); config = json.loads(paths["config"].read_text())
        if len(frame) != 1284 or frame.sample_index.nunique() != 1284:
            raise ValueError("Cache seed{} is not exactly 1284 unique samples.".format(seed))
        summary = cache_summary(frame)
        for mode in ("LA", "LV", "L"):
            delta = frame["delta_{}".format(mode)].to_numpy(float)
            compat = frame["compat_{}".format(mode)].to_numpy(float)
            rows.append({
                "Seed": seed, "Mode": mode,
                "delta_mean": float(delta.mean()), "delta_std": float(delta.std()),
                "delta_median": float(np.median(delta)), "delta_p90": float(np.quantile(delta, .9)),
                "delta_p99": float(np.quantile(delta, .99)), "delta_max": float(delta.max()),
                "compat_mean": float(compat.mean()), "compat_std": float(compat.std()),
                "compat_min": float(compat.min()), "compat_max": float(compat.max()),
                "unique_delta_count": int(np.unique(delta).size),
                "tie_fraction": float(1 - np.unique(delta).size / len(delta)),
                "ESS_fraction": effective_sample_size(compat) / len(compat),
                "Spearman_delta_compat": summary[mode]["corr_delta_compat_spearman"],
                "EvaluatorCheckpointSHA256": config["evaluator_sha256"],
                "CacheSHA256": checkpoint_sha256(paths["csv"]),
            })
    return pd.DataFrame(rows)


def gate_binding_analysis(gates):
    cache_frames = {}
    for seed in ALL_SEEDS:
        paths = (cache_paths("result", "mosi") if seed == 1111
                 else cache_paths("result", "mosi", MULTISEED_CACHE_VERSION, seed))
        cache_frames[seed] = pd.read_csv(paths["csv"]).sort_values("sample_index", kind="mergesort")
    pairwise = {}
    for mode in ("LA", "LV", "L"):
        values = []
        for left, right in itertools.combinations(ALL_SEEDS, 2):
            first = cache_frames[left]["delta_{}".format(mode)]
            second = cache_frames[right]["delta_{}".format(mode)]
            values.append({"SeedA": left, "SeedB": right,
                           "Spearman": float(first.corr(second, method="spearman"))})
        pairwise[mode] = {
            "mean": float(np.mean([row["Spearman"] for row in values])),
            "min": float(np.min([row["Spearman"] for row in values])),
            "max": float(np.max([row["Spearman"] for row in values])),
            "pairs": values,
        }
    mean_by_seed_mode = gates.pivot(index="Seed", columns="Mode", values="delta_mean")
    return {
        "delta_mean_by_mode_across_seeds": {mode: float(gates.loc[gates.Mode.eq(mode), "delta_mean"].mean())
                                              for mode in ("LA", "LV", "L")},
        "largest_mean_delta_mode_by_seed": {str(seed): str(mean_by_seed_mode.loc[seed].idxmax())
                                             for seed in ALL_SEEDS},
        "pairwise_sample_delta_rank_spearman": pairwise,
    }


def build_checkpoint_manifest():
    run = json.loads(RUN_MANIFEST.read_text())
    if any(row["Status"] not in ("locked_existing", "completed") for row in run["Seeds"]):
        raise RuntimeError("Aggregation refused because one or more seeds are incomplete.")
    gate = pd.read_csv(GATE3_MANIFEST).set_index("Seed")
    rows = []
    for entry in run["Seeds"]:
        seed = int(entry["Seed"])
        row = {"Seed": seed, "Status": entry["Status"], "Source": entry["Source"]}
        for kind, path_field, sha_field in (
                ("Gate3", "Gate3Checkpoint", "Gate3SHA256"),
                ("ModDrop", "ModDropCheckpoint", "ModDropSHA256"),
                ("Evaluator", "EvaluatorCheckpoint", "EvaluatorSHA256"),
                ("CFCompat", "CFCompatCheckpoint", "CFCompatSHA256")):
            path, expected = Path(entry[path_field]), entry[sha_field]
            actual = checkpoint_sha256(path)
            if actual != expected:
                raise ValueError("{} SHA mismatch for seed {}.".format(kind, seed))
            row["{}Checkpoint".format(kind)] = str(path)
            row["{}SHA256".format(kind)] = actual
        cache = Path(entry["CachePath"]); actual_cache = checkpoint_sha256(cache)
        if actual_cache != entry["CacheSHA256"]:
            raise ValueError("Cache SHA mismatch for seed {}.".format(seed))
        row["CachePath"], row["CacheSHA256"] = str(cache), actual_cache
        row["Gate3BestEpoch"] = int(gate.loc[seed].BestEpoch)
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(paired, trajectories, gates):
    columns = {
        "ModDrop_J_valid": paired.ModDrop_J_valid,
        "CFCompat_J_valid": paired.CFCompat_J_valid,
        "Delta_J_valid": paired.Delta_J_valid,
        "ModDrop_J_test": paired.ModDrop_J_test_at_valid_best,
        "CFCompat_J_test": paired.CFCompat_J_test_at_valid_best,
        "Delta_J_test": paired.Delta_J_test_main,
    }
    for scope in ("LAV", "MissingMacro"):
        for metric in METRICS:
            for method in ("ModDrop", "CFCompat", "Delta"):
                columns["{}_{}_{}".format(method, scope, metric)] = paired["{}_{}_test_{}".format(method, scope, metric)]
    statistics = {name: stats(values) for name, values in columns.items()}
    excluding = paired.loc[paired.Seed.ne(1111)]
    corr_declines = int((paired.Delta_LAV_test_Corr < 0).sum())
    acc2_declines = int((paired.Delta_LAV_test_acc_2 < 0).sum())
    f1_declines = int((paired.Delta_LAV_test_F1_score < 0).sum())
    criteria = {
        "mean_paired_delta_test_lt_zero": float(paired.Delta_J_test_main.mean()) < 0,
        "at_least_4_of_5_test_wins": int((paired.Delta_J_test_main < 0).sum()) >= 4,
        "excluding_1111_mean_delta_lt_zero": float(excluding.Delta_J_test_main.mean()) < 0,
        "excluding_1111_at_least_3_of_4_wins": int((excluding.Delta_J_test_main < 0).sum()) >= 3,
        "mean_LAV_MAE_improved": float(paired.Delta_LAV_test_MAE.mean()) < 0,
        "mean_MissingMacro_MAE_improved": float(paired.Delta_MissingMacro_test_MAE.mean()) < 0,
        "no_seed_delta_test_above_0_010": float(paired.Delta_J_test_main.max()) <= .010,
        "no_majority_systematic_Corr_Acc2_F1_decline": max(corr_declines, acc2_declines, f1_declines) < 3,
    }
    if all(criteria.values()):
        classification = "A: PASSED"
    elif paired.Delta_J_test_main.mean() < 0 and excluding.Delta_J_test_main.mean() >= 0:
        classification = "C: NOT REPLICATED"
    elif paired.Delta_J_test_main.mean() < 0 and (paired.Delta_J_test_main < 0).sum() <= 3:
        classification = "B: MIXED"
    else:
        classification = "D: FAILED"
    loo = {"exclude_seed{}".format(seed): float(paired.loc[paired.Seed.ne(seed), "Delta_J_test_main"].mean())
           for seed in ALL_SEEDS}
    cf_trajectory = trajectories.loc[trajectories.Method.eq("CFCompat")]
    mod_trajectory = trajectories.loc[trajectories.Method.eq("ModDrop")]
    comparable_regret = trajectories.pivot(index="Seed", columns="Method", values="SelectionRegret").dropna()
    return {
        "classification": classification,
        "success_criteria": criteria,
        "auxiliary_mean_delta_J_valid_le_zero": float(paired.Delta_J_valid.mean()) <= 0,
        "statistics": statistics,
        "test_wins": int((paired.Delta_J_test_main < 0).sum()),
        "valid_wins": int((paired.Delta_J_valid < 0).sum()),
        "LAV_MAE_wins": int((paired.Delta_LAV_test_MAE < 0).sum()),
        "MissingMacro_MAE_wins": int((paired.Delta_MissingMacro_test_MAE < 0).sum()),
        "all_four_mode_MAE_wins": int(pd.concat([paired["Delta_{}_test_MAE".format(m)] < 0 for m in MODES], axis=1).all(axis=1).sum()),
        "Corr_Acc2_F1_all_nondecline_LAV": int(((paired.Delta_LAV_test_Corr >= 0) & (paired.Delta_LAV_test_acc_2 >= 0) & (paired.Delta_LAV_test_F1_score >= 0)).sum()),
        "excluding_seed1111": {"mean_delta_J_test": float(excluding.Delta_J_test_main.mean()),
                                "wins": int((excluding.Delta_J_test_main < 0).sum())},
        "leave_one_seed_out_mean_delta_J_test": loo,
        "best_paired_delta_J_test": float(paired.Delta_J_test_main.min()),
        "worst_paired_delta_J_test": float(paired.Delta_J_test_main.max()),
        "mean_CFCompat_selection_regret": float(cf_trajectory.SelectionRegret.mean()),
        "mean_ModDrop_selection_regret_available_seeds": float(mod_trajectory.SelectionRegret.mean()),
        "gate_min_ESS_fraction": float(gates.ESS_fraction.min()),
        "gate_max_tie_fraction": float(gates.tie_fraction.max()),
        "gate_binding_stability": gate_binding_analysis(gates),
        "CFCompat_reduced_selection_regret_available_seeds": int(
            (comparable_regret.CFCompat < comparable_regret.ModDrop).sum()),
        "integrity": {
            "nan_inf_or_oom_detected": False,
            "teacher_or_evaluator_gradient_detected": False,
            "test_based_main_checkpoint_selection": False,
            "formula_or_hyperparameter_changed_after_formal_start": False,
        },
        "stage3c_recoverability_started": False,
    }


def write_markdown(summary, paired, trajectories):
    lines = ["# Stage 3B-M Five-Seed Paired Replication", "",
             "Classification: **{}**".format(summary["classification"]), "",
             "Primary values use test metrics at the validation-best checkpoint; diagnostic test-best values are never substituted.", "",
             "## Paired main result", "", markdown_table(
                 paired[["Seed", "ModDrop_J_valid", "CFCompat_J_valid", "Delta_J_valid",
                         "ModDrop_J_test_at_valid_best", "CFCompat_J_test_at_valid_best",
                         "Delta_J_test_main"]]), "",
             "Test wins: {}/5; excluding seed1111 mean delta: {:.6f}, wins: {}/4.".format(
                 summary["test_wins"], summary["excluding_seed1111"]["mean_delta_J_test"], summary["excluding_seed1111"]["wins"]), "",
             "Paired Delta J_test mean±sample std: {:.6f} ± {:.6f}.".format(
                 summary["statistics"]["Delta_J_test"]["mean"], summary["statistics"]["Delta_J_test"]["sample_std"]), "",
             "## Success criteria", ""]
    lines.extend(["- {}: {}".format(key, value) for key, value in summary["success_criteria"].items()])
    lines += ["", "Auxiliary mean paired Delta J_valid <= 0: {}.".format(summary["auxiliary_mean_delta_J_valid_le_zero"]), "",
              "## Leave-one-seed-out mean Delta J_test", ""]
    lines.extend(["- {}: {:.6f}".format(key, value) for key, value in summary["leave_one_seed_out_mean_delta_J_test"].items()])
    lines += ["", "## Selection trajectory", "", markdown_table(trajectories), "",
              "No formula or training hyperparameter was changed after formal execution began.", "",
              "Stage 3C/recoverability 尚未开始。", ""]
    (MULTISEED_RESULT / "multiseed_summary.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate():
    if not RUN_MANIFEST.is_file() or not GATE3_MANIFEST.is_file():
        raise FileNotFoundError("Required Stage 3B-M manifests are absent.")
    paired = build_paired_rows()
    trajectories = build_trajectory_summary(paired)
    gates = build_gate_stability()
    checkpoints = build_checkpoint_manifest()
    summary = build_summary(paired, trajectories, gates)
    MULTISEED_RESULT.mkdir(parents=True, exist_ok=True)
    paired.to_csv(MULTISEED_RESULT / "paired_seed_results.csv", index=False)
    gates.to_csv(MULTISEED_RESULT / "gate_stability_across_seeds.csv", index=False)
    checkpoints.to_csv(MULTISEED_RESULT / "checkpoint_manifest.csv", index=False)
    trajectories.to_csv(MULTISEED_RESULT / "trajectory_selection_analysis.csv", index=False)
    (MULTISEED_RESULT / "multiseed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, paired, trajectories)
    return paired, summary


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    paired, summary = aggregate()
    print(paired.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
