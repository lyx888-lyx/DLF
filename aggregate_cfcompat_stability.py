"""Five-seed aggregation and validation-only strategy selection for Stage 8."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_1samp

from eval_cfcompat_stability import seed_directory, verify_seed
from trains.singleTask.cfcompat_stability_utils import (
    STAGE8_SEEDS,
    markdown_table,
    select_main_strategy,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    regression_metrics,
)


METHODS = ("Online", "EMA", "Soup-3", "Soup-5")
STRATEGIES = ("EMA", "Soup-3", "Soup-5")
MODES = ("LAV", "LA", "LV", "L")
METRICS = ("MAE", "Corr", "acc_2", "F1_score", "acc_5", "acc_7", "Loss")


def result_directory(result_root, dataset):
    return (
        Path(result_root) / "missing_baseline" / "cfcompat_stability_v1"
        / dataset
    )


def bootstrap_mean_ci(values, seed=8082026, samples=20000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def load_all_seed_artifacts(root, dataset):
    rows, ema_epochs, soup_sources, manifests = [], [], [], []
    for seed in STAGE8_SEEDS:
        directory = seed_directory(root, dataset, seed)
        verify_seed(directory, require_replay=True)
        rows.append(pd.read_csv(directory / "per_seed_all_methods.csv"))
        ema_epochs.append(pd.read_csv(directory / "ema_epoch_metrics.csv"))
        soup_sources.append(pd.read_csv(directory / "soup_source_checkpoints.csv"))
        manifests.append(json.loads((directory / "seed_manifest.json").read_text()))
    per_seed = pd.concat(rows, ignore_index=True)
    if (
        per_seed.duplicated(["Seed", "Method"]).any()
        or per_seed.Seed.astype(int).tolist()
        != [seed for seed in STAGE8_SEEDS for _ in METHODS]
        or per_seed.Method.tolist() != list(METHODS) * len(STAGE8_SEEDS)
    ):
        raise RuntimeError("Five-seed method table is not in the locked order.")
    return (
        per_seed,
        pd.concat(ema_epochs, ignore_index=True),
        pd.concat(soup_sources, ignore_index=True),
        manifests,
    )


def build_paired_deltas(per_seed):
    online = per_seed.loc[per_seed.Method.eq("Online")].set_index("Seed")
    numeric_metrics = [
        column
        for column in per_seed.select_dtypes(include=[np.number]).columns
        if column not in ("Seed", "BestValidEpoch")
    ]
    rows = []
    for strategy in STRATEGIES:
        candidate = per_seed.loc[per_seed.Method.eq(strategy)].set_index("Seed")
        for seed in STAGE8_SEEDS:
            row = {"Seed": seed, "Strategy": strategy}
            for column in numeric_metrics:
                row["Delta_{}".format(column)] = (
                    float(candidate.loc[seed, column])
                    - float(online.loc[seed, column])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_method_metrics(per_seed):
    rows = []
    excluded = {
        "Seed",
        "BestValidEpoch",
        "BestObservedTestEpoch",
        "SourceCount",
        "EMAUpdateCount",
    }
    numeric = [
        column
        for column in per_seed.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    for method in METHODS:
        local = per_seed.loc[per_seed.Method.eq(method)]
        for metric in numeric:
            values = local[metric].astype(float)
            if values.isna().all():
                continue
            rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "Mean": float(values.mean()),
                    "SampleStd": float(values.std(ddof=1)),
                    "Median": float(values.median()),
                    "Min": float(values.min()),
                    "Max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def paired_statistics(paired):
    rows = []
    for strategy in STRATEGIES:
        local = paired.loc[paired.Strategy.eq(strategy)]
        for metric in (
            "Delta_J_valid",
            "Delta_J_test_at_valid_best",
            "Delta_test_at_valid_best_LAV_MAE",
            "Delta_test_at_valid_best_MissingMacro_MAE",
            "Delta_test_at_valid_best_LAV_Corr",
            "Delta_test_at_valid_best_LAV_acc_2",
            "Delta_test_at_valid_best_LAV_F1_score",
        ):
            values = local[metric].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_ci(
                values, seed=8082026 + sum(ord(char) for char in strategy + metric)
            )
            test = ttest_1samp(values, popmean=0.0)
            rows.append(
                {
                    "Strategy": strategy,
                    "DeltaMetric": metric,
                    "Mean": float(values.mean()),
                    "SampleStd": float(values.std(ddof=1)),
                    "Median": float(np.median(values)),
                    "Min": float(values.min()),
                    "Max": float(values.max()),
                    "ImprovedSeedCount": int((values < 0).sum())
                    if metric not in (
                        "Delta_test_at_valid_best_LAV_Corr",
                        "Delta_test_at_valid_best_LAV_acc_2",
                        "Delta_test_at_valid_best_LAV_F1_score",
                    )
                    else int((values > 0).sum()),
                    "Bootstrap95CILow": ci_low,
                    "Bootstrap95CIHigh": ci_high,
                    "PairedTStatisticDescriptive": float(test.statistic),
                    "PairedTPValueDescriptive": float(test.pvalue),
                    "NoSignificanceClaimAtN5": True,
                }
            )
    return pd.DataFrame(rows)


def ensemble_split(root, dataset, split):
    frames = []
    for seed in STAGE8_SEEDS:
        path = (
            seed_directory(root, dataset, seed)
            / "online_{}_predictions.csv".format(split)
        )
        frame = pd.read_csv(path).sort_values("sample_index", kind="mergesort")
        frames.append(frame)
    reference = frames[0][["sample_index", "sample_id", "label"]].reset_index(drop=True)
    for frame in frames[1:]:
        local = frame[["sample_index", "sample_id", "label"]].reset_index(drop=True)
        if not reference.equals(local):
            raise RuntimeError("Cross-seed ensemble sample/label binding differs.")
    result = reference.copy()
    for mode in MODES:
        values = np.stack(
            [frame["{}_pred".format(mode)].to_numpy() for frame in frames],
            axis=0,
        )
        result["{}_pred".format(mode)] = values.mean(axis=0)
    result["Split"] = split
    result["Method"] = "Online-5Seed-EqualPredictionEnsemble"
    result["DiagnosticOnly"] = True
    result["MultiModelInferenceUpperBound"] = True
    metric_rows = []
    by_mode = {}
    labels = torch.tensor(result.label.to_numpy(), dtype=torch.float32)
    for mode in MODES:
        predictions = torch.tensor(
            result["{}_pred".format(mode)].to_numpy(), dtype=torch.float32
        )
        metrics = regression_metrics(predictions, labels)
        by_mode[mode] = metrics
    j_value = .5 * by_mode["LAV"]["MAE"] + .5 * np.mean(
        [by_mode[mode]["MAE"] for mode in MISSING_MODES]
    )
    for mode in MODES:
        metric_rows.append(
            {
                "Split": split,
                "Method": "Online-5Seed-EqualPredictionEnsemble",
                "Mode": mode,
                "J": j_value,
                **by_mode[mode],
                "DiagnosticOnly": True,
                "MultiModelInferenceUpperBound": True,
                "ParticipatesInMainStrategySelection": False,
            }
        )
    return result, metric_rows


def success_audit(per_seed, paired, strategy):
    local = paired.loc[paired.Strategy.eq(strategy)].copy()
    delta_test = local.Delta_J_test_at_valid_best
    delta_valid = local.Delta_J_valid
    lav = local.Delta_test_at_valid_best_LAV_MAE
    missing = local.Delta_test_at_valid_best_MissingMacro_MAE
    corr = local.Delta_test_at_valid_best_LAV_Corr
    classification_metrics = (
        "Delta_test_at_valid_best_LAV_acc_2",
        "Delta_test_at_valid_best_LAV_F1_score",
        "Delta_test_at_valid_best_LAV_acc_5",
        "Delta_test_at_valid_best_LAV_acc_7",
    )
    no_systematic_classification_decline = all(
        float(local[column].mean()) >= 0 for column in classification_metrics
    )
    basic = {
        "MeanDeltaJTestLEMinus0.003": float(delta_test.mean()) <= -.003,
        "AtLeast4of5TestImproved": int((delta_test < 0).sum()) >= 4,
        "NoSeedDegradedAbovePlus0.005": float(delta_test.max()) <= .005,
        "LAVMAENonDegradedAtLeast4of5": int((lav <= 0).sum()) >= 4,
        "MissingMacroMAENonDegradedAtLeast4of5": int((missing <= 0).sum()) >= 4,
        "MeanDeltaJValidLEZero": float(delta_valid.mean()) <= 0,
        "NotDrivenBySingleExtremeSeed": int((delta_test < 0).sum()) >= 4,
    }
    strong = {
        "MeanDeltaJTestLEMinus0.005": float(delta_test.mean()) <= -.005,
        "AtLeast4of5TestImproved": int((delta_test < 0).sum()) >= 4,
        "LAVMAEImprovedAtLeast4of5": int((lav < 0).sum()) >= 4,
        "MissingMacroMAEImprovedAtLeast4of5": int((missing < 0).sum()) >= 4,
        "MeanLAVCorrNonDecline": float(corr.mean()) >= 0,
        "NoSystematicClassificationDecline": no_systematic_classification_decline,
    }
    basic_pass = all(basic.values())
    strong_pass = basic_pass and all(strong.values())
    classification = (
        "STRONG SUCCESS"
        if strong_pass
        else "BASIC SUCCESS"
        if basic_pass
        else "STABILITY STRATEGY NOT SUPPORTED"
    )
    return {
        "MainStrategy": strategy,
        "Classification": classification,
        "BasicSuccess": basic_pass,
        "StrongSuccess": strong_pass,
        "BasicCriteria": basic,
        "StrongCriteria": strong,
        "MeanDeltaJTest": float(delta_test.mean()),
        "MeanDeltaJValid": float(delta_valid.mean()),
        "TestImprovedSeedCount": int((delta_test < 0).sum()),
        "WorstSeedDeltaJTest": float(delta_test.max()),
        "LAVMAENonDegradedSeedCount": int((lav <= 0).sum()),
        "MissingMacroMAENonDegradedSeedCount": int((missing <= 0).sum()),
    }


def write_report(root, per_seed, paired, selection, audit, paired_stats, ensemble):
    strategy = selection["MainStrategy"]
    performance = per_seed[
        ["Seed", "Method", "BestValidEpoch", "J_valid", "J_test_at_valid_best"]
    ]
    deltas = paired.loc[paired.Strategy.eq(strategy)][
        [
            "Seed",
            "Strategy",
            "Delta_J_valid",
            "Delta_J_test_at_valid_best",
            "Delta_test_at_valid_best_LAV_MAE",
            "Delta_test_at_valid_best_MissingMacro_MAE",
        ]
    ]
    stats = paired_stats.loc[
        paired_stats.Strategy.eq(strategy)
        & paired_stats.DeltaMetric.isin(
            ("Delta_J_valid", "Delta_J_test_at_valid_best")
        )
    ]
    lines = [
        "# Stage 8 Stability-First CFCompatKD Final Audit",
        "",
        "## Outcome",
        "",
        "**{} — MainStrategy: {}**".format(audit["Classification"], strategy),
        "",
        "The strategy was selected only by the lowest five-seed mean validation J. "
        "Online CFCompatKD was a baseline and test metrics did not participate.",
        "",
        "## Five-seed validation-selected performance",
        "",
        markdown_table(performance),
        "",
        "## MainStrategy paired deltas (strategy - online)",
        "",
        markdown_table(deltas),
        "",
        "## Paired descriptive statistics",
        "",
        markdown_table(stats),
        "",
        "No statistical-significance claim is made at n=5; the paired t-test is "
        "reported descriptively only.",
        "",
        "## Basic success criteria",
        "",
    ]
    lines.extend(
        "- {}: {}".format(key, value)
        for key, value in audit["BasicCriteria"].items()
    )
    lines += ["", "## Strong success criteria", ""]
    lines.extend(
        "- {}: {}".format(key, value)
        for key, value in audit["StrongCriteria"].items()
    )
    lines += [
        "",
        "## Equal-weight five-online-model prediction ensemble upper bound",
        "",
        markdown_table(ensemble),
        "",
        "This is diagnostic multi-model inference only. It is not a single-model "
        "EMA/Soup result and did not participate in MainStrategy selection.",
        "",
        "## Integrity declarations",
        "",
        "- All five Online trajectories passed their locked Stage3 replay gates.",
        "- EMA decay was fixed at 0.999 and EMA never entered backward/optimizer.",
        "- Soup sources were same-seed online checkpoints ranked by validation J only.",
        "- Soup averaging used CPU FP64 and no fine-tuning.",
        "- No cross-seed weight soup, greedy selection, test weighting, or extra seed was used.",
        "- Main checkpoints and the global strategy were validation-selected.",
        "",
    ]
    (root / "stage8_cfcompat_stability_final_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def aggregate(result_root="result", dataset="mosi"):
    root = result_directory(result_root, dataset)
    per_seed, ema_epochs, soup_sources, manifests = load_all_seed_artifacts(
        result_root, dataset
    )
    paired = build_paired_deltas(per_seed)
    aggregate_metrics = aggregate_method_metrics(per_seed)
    paired_stats = paired_statistics(paired)
    strategy, validation_means = select_main_strategy(per_seed)
    selection = {
        "MainStrategy": strategy,
        "MeanValidationJ": validation_means,
        "SelectionRule": "argmin mean_seed(J_valid)",
        "TiePriority": ["EMA", "Soup-3", "Soup-5"],
        "TestUsedForSelection": False,
        "PerSeedStrategySelection": False,
        "OnlineParticipates": False,
    }
    audit = success_audit(per_seed, paired, strategy)

    ensemble_rows = []
    for split in ("valid", "test"):
        predictions, rows = ensemble_split(result_root, dataset, split)
        predictions.to_csv(
            root / "ensemble_upper_bound_predictions_{}.csv".format(split),
            index=False,
        )
        ensemble_rows.extend(rows)
    ensemble = pd.DataFrame(ensemble_rows)

    checkpoint_manifest = {
        "Methods": [
            {
                "Seed": int(row.Seed),
                "Method": row.Method,
                "Checkpoint": row.Checkpoint,
                "CheckpointSHA256": row.CheckpointSHA256,
                "BestValidEpoch": int(row.BestValidEpoch),
                "J_valid": float(row.J_valid),
                "J_test_at_valid_best": float(row.J_test_at_valid_best),
            }
            for _, row in per_seed.iterrows()
        ],
        "SeedManifests": manifests,
        "AllOnlineReplaysPassed": True,
        "NoTestSelectedCheckpoint": True,
        "NoCrossSeedWeightAveraging": True,
    }
    for item in checkpoint_manifest["Methods"]:
        if checkpoint_sha256(item["Checkpoint"]) != item["CheckpointSHA256"]:
            raise RuntimeError("Checkpoint changed during aggregation.")

    per_seed.to_csv(root / "per_seed_all_methods.csv", index=False)
    aggregate_metrics.to_csv(root / "aggregate_all_methods.csv", index=False)
    paired.to_csv(root / "paired_deltas.csv", index=False)
    ema_epochs.to_csv(root / "ema_epoch_metrics.csv", index=False)
    soup_sources.to_csv(root / "soup_source_checkpoints.csv", index=False)
    ensemble.to_csv(root / "ensemble_upper_bound.csv", index=False)
    paired_stats.to_csv(root / "paired_statistics.csv", index=False)
    (root / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n"
    )
    (root / "main_strategy_selection.json").write_text(
        json.dumps({**selection, "SuccessAudit": audit}, indent=2, sort_keys=True)
        + "\n"
    )
    write_report(
        root, per_seed, paired, selection, audit, paired_stats, ensemble
    )
    return selection, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--result-root", default="result")
    args = parser.parse_args()
    selection, audit = aggregate(args.result_root, args.dataset)
    print(json.dumps({**selection, "SuccessAudit": audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
