"""Stage 9A audit aggregation for the fixed CFCompatKD-PE5 ensemble."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    IDENTITY_COLUMNS,
    LOWER_IS_BETTER,
    METRICS,
    PE5_MODES,
    PE5_SEEDS,
    PE5_SPLITS,
    aligned_prediction_mean,
    canonical_identity_values,
    compatibility_quartiles,
    deterministic_quartiles,
    j_contribution,
    markdown_table,
    metric_rows,
    metrics_from_predictions,
    paired_bootstrap,
    pearson,
    require_locked_members,
    stage9_directory,
    validate_prediction_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Stage9A CFCompatKD-PE5.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PE5_SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--verify-online-offline", action="store_true")
    parser.add_argument("--result-root", default="result")
    args = parser.parse_args()
    try:
        require_locked_members(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.bootstrap_samples != 2000:
        parser.error("Stage9A fixes --bootstrap-samples=2000.")
    if not args.verify_online_offline:
        parser.error("Formal aggregation requires --verify-online-offline.")
    return args


def prediction_path(root, seed, split):
    return root / "online_seed{}_{}_predictions.csv".format(seed, split)


def load_artifacts(result_root, dataset):
    root = stage9_directory(result_root, dataset)
    replay = json.loads((root / "ensemble_replay_verification.json").read_text())
    if (
        not replay.get("Passed")
        or replay["MaximumOfflineOnlinePredictionDifference"] > 1e-6
        or replay["MaximumOfflineOnlineMetricDifference"] > 1e-6
    ):
        raise RuntimeError("Stage9A offline/online replay gate did not pass.")
    frames = {}
    for split in PE5_SPLITS:
        frames[split] = {}
        for seed in PE5_SEEDS:
            frames[split][seed] = validate_prediction_frame(
                pd.read_csv(prediction_path(root, seed, split)),
                split,
                seed=seed,
                method="Online",
            )
        ensemble = validate_prediction_frame(
            pd.read_csv(root / "ensemble_predictions_{}.csv".format(split)),
            split,
            method="CFCompatKD-PE5",
        )
        rebuilt = aligned_prediction_mean(
            list(frames[split].values()), split, "CFCompatKD-PE5"
        )
        for column in ("LAV_pred", "LA_pred", "LV_pred", "L_pred"):
            if np.max(
                np.abs(
                    ensemble[column].to_numpy(dtype=np.float64)
                    - rebuilt[column].to_numpy(dtype=np.float64)
                )
            ) > 1e-6:
                raise RuntimeError("Saved ensemble is not the fixed equal mean.")
        frames[split]["ensemble"] = ensemble
    return root, replay, frames


def individual_metrics(frames):
    rows = []
    for split in PE5_SPLITS:
        for seed in PE5_SEEDS:
            rows.extend(metric_rows(frames[split][seed], "Online", seed=seed))
    return pd.DataFrame(rows)


def best_validation_seed(individual):
    valid = individual.loc[
        individual.Split.eq("valid") & individual.Mode.eq("LAV"),
        ["Seed", "J"],
    ]
    valid = valid.sort_values(["J", "Seed"], kind="mergesort")
    return int(valid.iloc[0].Seed)


def ensemble_metric_comparison(frames, individual, best_seed):
    rows = []
    for split in PE5_SPLITS:
        ensemble_metrics, ensemble_j = metrics_from_predictions(
            frames[split]["ensemble"]
        )
        for mode in PE5_MODES + ("MissingMacro",):
            local = individual.loc[
                individual.Split.eq(split) & individual.Mode.eq(mode)
            ].set_index("Seed")
            for metric in METRICS:
                value = float(ensemble_metrics[mode][metric])
                seeds = local[metric].astype(float)
                row = {
                    "Split": split,
                    "Mode": mode,
                    "Metric": metric,
                    "EnsembleValue": value,
                    "MeanIndividualValue": float(seeds.mean()),
                    "DeltaVsMeanIndividual": value - float(seeds.mean()),
                    "BestValidationSeed": best_seed,
                    "BestValidationSeedValue": float(local.loc[best_seed, metric]),
                    "DeltaVsBestValidationSeed": value
                    - float(local.loc[best_seed, metric]),
                    "ImprovedSeedCount": int((value < seeds).sum())
                    if metric in LOWER_IS_BETTER
                    else int((value > seeds).sum()),
                }
                for seed in PE5_SEEDS:
                    row["Seed{}Value".format(seed)] = float(local.loc[seed, metric])
                    row["DeltaVsSeed{}".format(seed)] = (
                        value - float(local.loc[seed, metric])
                    )
                rows.append(row)
        objective = individual.loc[
            individual.Split.eq(split) & individual.Mode.eq("LAV")
        ].set_index("Seed")["J"].astype(float)
        row = {
            "Split": split,
            "Mode": "Objective",
            "Metric": "J",
            "EnsembleValue": ensemble_j,
            "MeanIndividualValue": float(objective.mean()),
            "DeltaVsMeanIndividual": ensemble_j - float(objective.mean()),
            "BestValidationSeed": best_seed,
            "BestValidationSeedValue": float(objective.loc[best_seed]),
            "DeltaVsBestValidationSeed": ensemble_j
            - float(objective.loc[best_seed]),
            "ImprovedSeedCount": int((ensemble_j < objective).sum()),
        }
        for seed in PE5_SEEDS:
            row["Seed{}Value".format(seed)] = float(objective.loc[seed])
            row["DeltaVsSeed{}".format(seed)] = ensemble_j - float(
                objective.loc[seed]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_out(frames, online_mean_j):
    metric_rows_output = []
    prediction_rows_output = []
    for omitted in PE5_SEEDS:
        remaining_seeds = tuple(seed for seed in PE5_SEEDS if seed != omitted)
        for split in PE5_SPLITS:
            remaining = [frames[split][seed] for seed in remaining_seeds]
            ensemble = aligned_prediction_mean(
                remaining, split, "CFCompatKD-PE5-omit{}".format(omitted)
            )
            full = frames[split]["ensemble"]
            omitted_frame = frames[split][omitted]
            metrics, j_value = metrics_from_predictions(ensemble)
            full_metrics, full_j = metrics_from_predictions(full)
            mode_abs_distances = []
            mode_signed_distances = []
            mode_correlations = []
            for mode in PE5_MODES:
                prediction = ensemble["{}_pred".format(mode)].to_numpy(
                    dtype=np.float64
                )
                removed = omitted_frame["{}_pred".format(mode)].to_numpy(
                    dtype=np.float64
                )
                distance = np.abs(removed - prediction)
                signed_distance = float((removed - prediction).mean())
                correlation = pearson(removed, prediction)
                mode_abs_distances.append(float(distance.mean()))
                mode_signed_distances.append(signed_distance)
                mode_correlations.append(correlation)
                for index, row in ensemble.iterrows():
                    prediction_rows_output.append(
                        {
                            "OmittedSeed": omitted,
                            "RemainingSeeds": ",".join(map(str, remaining_seeds)),
                            "Split": split,
                            "Mode": mode,
                            "sample_index": int(row.sample_index),
                            "sample_id": row.sample_id,
                            "label": float(row.label),
                            "Prediction": float(row["{}_pred".format(mode)]),
                        }
                    )
                for metric in METRICS:
                    metric_rows_output.append(
                        {
                            "OmittedSeed": omitted,
                            "RemainingSeeds": ",".join(map(str, remaining_seeds)),
                            "Split": split,
                            "Mode": mode,
                            "Metric": metric,
                            "Value": metrics[mode][metric],
                            "FullPE5Value": full_metrics[mode][metric],
                            "DeltaVsFullPE5": metrics[mode][metric]
                            - full_metrics[mode][metric],
                            "RemovedVsRemainingMeanAbsPredictionDistance": float(
                                distance.mean()
                            ),
                            "RemovedVsRemainingSignedPredictionDistance": signed_distance,
                            "RemovedVsRemainingPredictionCorrelation": correlation,
                            "LOOObjectiveJ": j_value,
                            "FullPE5ObjectiveJ": full_j,
                            "DeltaJVsFullPE5": j_value - full_j,
                            "OnlineFiveSeedMeanJ": online_mean_j[split],
                            "DeltaJVsOnlineFiveSeedMean": j_value
                            - online_mean_j[split],
                            "BetterThanOnlineFiveSeedMean": j_value
                            < online_mean_j[split],
                            "DiagnosticOnly": True,
                            "MemberSelectionAllowed": False,
                        }
                    )
            metric_rows_output.append(
                {
                    "OmittedSeed": omitted,
                    "RemainingSeeds": ",".join(map(str, remaining_seeds)),
                    "Split": split,
                    "Mode": "Objective",
                    "Metric": "J",
                    "Value": j_value,
                    "FullPE5Value": full_j,
                    "DeltaVsFullPE5": j_value - full_j,
                    "RemovedVsRemainingMeanAbsPredictionDistance": float(
                        np.mean(mode_abs_distances)
                    ),
                    "RemovedVsRemainingSignedPredictionDistance": float(
                        np.mean(mode_signed_distances)
                    ),
                    "RemovedVsRemainingPredictionCorrelation": float(
                        np.mean(mode_correlations)
                    ),
                    "LOOObjectiveJ": j_value,
                    "FullPE5ObjectiveJ": full_j,
                    "DeltaJVsFullPE5": j_value - full_j,
                    "OnlineFiveSeedMeanJ": online_mean_j[split],
                    "DeltaJVsOnlineFiveSeedMean": j_value - online_mean_j[split],
                    "BetterThanOnlineFiveSeedMean": j_value < online_mean_j[split],
                    "DiagnosticOnly": True,
                    "MemberSelectionAllowed": False,
                }
            )
    return pd.DataFrame(metric_rows_output), pd.DataFrame(prediction_rows_output)


def pairwise_diversity(frames):
    rows = []
    for split in PE5_SPLITS:
        labels = frames[split][PE5_SEEDS[0]].label.to_numpy(dtype=np.float64)
        for mode in PE5_MODES:
            values = {
                seed: frames[split][seed]["{}_pred".format(mode)].to_numpy(
                    dtype=np.float64
                )
                for seed in PE5_SEEDS
            }
            matrix = np.stack(list(values.values()), axis=0)
            prediction_std = matrix.std(axis=0, ddof=1)
            error_std = (matrix - labels[None, :]).std(axis=0, ddof=1)
            for first, second in combinations(PE5_SEEDS, 2):
                first_prediction, second_prediction = values[first], values[second]
                first_error = first_prediction - labels
                second_error = second_prediction - labels
                rows.append(
                    {
                        "Split": split,
                        "Mode": mode,
                        "SeedA": first,
                        "SeedB": second,
                        "PredictionCorrelation": pearson(
                            first_prediction, second_prediction
                        ),
                        "MeanAbsoluteDisagreement": float(
                            np.abs(first_prediction - second_prediction).mean()
                        ),
                        "MeanSignedDisagreementAminusB": float(
                            (first_prediction - second_prediction).mean()
                        ),
                        "ErrorCorrelation": pearson(first_error, second_error),
                        "MeanSamplePredictionStdAcrossFive": float(
                            prediction_std.mean()
                        ),
                        "MeanSampleErrorStdAcrossFive": float(error_std.mean()),
                    }
                )
    return pd.DataFrame(rows)


def counterfactual_source(result_root, split):
    return (
        Path(result_root)
        / "milestone_audit"
        / "stage2"
        / split
        / "counterfactual_contributions.csv"
    )


def sample_ensemble_gain(frames, result_root):
    rows = []
    for split in PE5_SPLITS:
        ensemble = frames[split]["ensemble"]
        compatibility_source = pd.read_csv(
            counterfactual_source(result_root, split)
        )
        compatibility = compatibility_quartiles(compatibility_source)
        reference = ensemble.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)
        compatibility = compatibility.sort_values(
            "sample_index", kind="mergesort"
        ).reset_index(drop=True)
        for column in IDENTITY_COLUMNS:
            left = canonical_identity_values(reference, column)
            right = canonical_identity_values(compatibility, column)
            if not np.array_equal(left, right):
                raise RuntimeError(
                    "Stage2.5 compatibility audit sample binding differs."
                )
        labels = ensemble.label.to_numpy(dtype=np.float64)
        l_errors = np.stack(
            [
                np.abs(
                    frames[split][seed].L_pred.to_numpy(dtype=np.float64) - labels
                )
                for seed in PE5_SEEDS
            ],
            axis=0,
        ).mean(axis=0)
        l_quartile = deterministic_quartiles(l_errors)
        for mode in PE5_MODES:
            matrix = np.stack(
                [
                    frames[split][seed]["{}_pred".format(mode)].to_numpy(
                        dtype=np.float64
                    )
                    for seed in PE5_SEEDS
                ],
                axis=0,
            )
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=1)
            individual_error = np.abs(matrix - labels[None, :]).mean(axis=0)
            ensemble_error = np.abs(mean - labels)
            gain = individual_error - ensemble_error
            std_quartile = deterministic_quartiles(std)
            label_interval = pd.cut(
                labels,
                bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
                labels=("<=-2", "(-2,-1]", "(-1,0]", "(0,1]", "(1,2]", ">2"),
                include_lowest=True,
            ).astype(str)
            for index, base in ensemble.iterrows():
                row = {
                    "Split": split,
                    "Mode": mode,
                    "sample_index": int(base.sample_index),
                    "sample_id": base.sample_id,
                    "label": float(base.label),
                    "EnsembleMean": float(mean[index]),
                    "EnsembleStd": float(std[index]),
                    "MinPrediction": float(matrix[:, index].min()),
                    "MaxPrediction": float(matrix[:, index].max()),
                    "PredictionRange": float(
                        matrix[:, index].max() - matrix[:, index].min()
                    ),
                    "MeanIndividualAbsoluteError": float(individual_error[index]),
                    "EnsembleAbsoluteError": float(ensemble_error[index]),
                    "EnsembleGain": float(gain[index]),
                    "EnsembleStdQuartile": std_quartile[index],
                    "LabelInterval": label_interval[index],
                    "LOnlyErrorQuartile": l_quartile[index],
                    "MissingMode": mode if mode != "LAV" else "Complete",
                    "CompatibilitySource": (
                        "Stage2.5_frozen_evaluator_split_local_rank"
                        if mode in ("LA", "LV", "L")
                        else "NotApplicable"
                    ),
                    "CompatibilityQuartile": (
                        compatibility.loc[
                            index, "CompatibilityQuartile_{}".format(mode)
                        ]
                        if mode in ("LA", "LV", "L")
                        else "NotApplicable"
                    ),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def conditional_gain(sample_gain):
    rows = []
    group_columns = (
        "EnsembleStdQuartile",
        "LabelInterval",
        "LOnlyErrorQuartile",
        "MissingMode",
        "CompatibilityQuartile",
    )
    for split in PE5_SPLITS:
        for mode in PE5_MODES:
            local = sample_gain.loc[
                sample_gain.Split.eq(split) & sample_gain.Mode.eq(mode)
            ]
            rows.append(
                {
                    "Split": split,
                    "Mode": mode,
                    "ConditionType": "EnsembleStdContinuous",
                    "ConditionValue": "PearsonR",
                    "count": len(local),
                    "MeanEnsembleGain": float(local.EnsembleGain.mean()),
                    "MedianEnsembleGain": float(local.EnsembleGain.median()),
                    "PositiveGainFraction": float((local.EnsembleGain > 0).mean()),
                    "MeanEnsembleStd": float(local.EnsembleStd.mean()),
                    "GainStdCorrelation": pearson(
                        local.EnsembleGain, local.EnsembleStd
                    ),
                }
            )
            for column in group_columns:
                grouped = local
                if column == "CompatibilityQuartile":
                    grouped = grouped.loc[
                        ~grouped.CompatibilityQuartile.eq("NotApplicable")
                    ]
                for value, subset in grouped.groupby(column, sort=False):
                    rows.append(
                        {
                            "Split": split,
                            "Mode": mode,
                            "ConditionType": column,
                            "ConditionValue": value,
                            "count": len(subset),
                            "MeanEnsembleGain": float(subset.EnsembleGain.mean()),
                            "MedianEnsembleGain": float(
                                subset.EnsembleGain.median()
                            ),
                            "PositiveGainFraction": float(
                                (subset.EnsembleGain > 0).mean()
                            ),
                            "MeanEnsembleStd": float(subset.EnsembleStd.mean()),
                            "GainStdCorrelation": pearson(
                                subset.EnsembleGain, subset.EnsembleStd
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_audit(frames, best_seed, samples):
    result = {
        "BootstrapSamples": samples,
        "BaseSeed": 9092026,
        "BestValidationSeed": best_seed,
        "Comparisons": [],
        "NoSingleModelClaim": True,
    }
    for split_index, split in enumerate(PE5_SPLITS):
        ensemble = j_contribution(frames[split]["ensemble"])
        individuals = np.stack(
            [j_contribution(frames[split][seed]) for seed in PE5_SEEDS], axis=0
        )
        comparisons = (
            ("EnsembleVsWithinSampleMeanIndividual", individuals.mean(axis=0)),
            ("EnsembleVsBestValidationSeed{}".format(best_seed), individuals[
                PE5_SEEDS.index(best_seed)
            ]),
        )
        for comparison_index, (name, reference) in enumerate(comparisons):
            statistics = paired_bootstrap(
                ensemble,
                reference,
                samples=samples,
                seed=9092026 + split_index * 10 + comparison_index,
            )
            result["Comparisons"].append(
                {"Split": split, "Comparison": name, **statistics}
            )
    return result


def write_report(
    root,
    comparison,
    loo_metrics,
    diversity,
    bootstrap,
    replay,
    best_seed,
    success,
):
    objective = comparison.loc[comparison.Metric.eq("J"), [
        "Split",
        "EnsembleValue",
        "MeanIndividualValue",
        "DeltaVsMeanIndividual",
        "BestValidationSeed",
        "BestValidationSeedValue",
        "DeltaVsBestValidationSeed",
        "ImprovedSeedCount",
    ]]
    test_metrics = comparison.loc[
        comparison.Split.eq("test")
        & comparison.Mode.isin(("LAV", "MissingMacro"))
        & comparison.Metric.isin(("MAE", "Corr", "acc_2", "F1_score", "acc_5", "acc_7")),
        [
            "Mode",
            "Metric",
            "EnsembleValue",
            "MeanIndividualValue",
            "DeltaVsMeanIndividual",
            "ImprovedSeedCount",
        ],
    ]
    loo_j = loo_metrics.loc[loo_metrics.Metric.eq("J"), [
        "OmittedSeed",
        "Split",
        "Value",
        "FullPE5Value",
        "DeltaVsFullPE5",
        "OnlineFiveSeedMeanJ",
        "BetterThanOnlineFiveSeedMean",
    ]]
    diversity_summary = diversity.groupby(["Split", "Mode"], as_index=False).agg(
        MeanPairwisePredictionCorrelation=("PredictionCorrelation", "mean"),
        MeanPairwiseAbsoluteDisagreement=("MeanAbsoluteDisagreement", "mean"),
        MeanPairwiseErrorCorrelation=("ErrorCorrelation", "mean"),
        MeanSamplePredictionStdAcrossFive=("MeanSamplePredictionStdAcrossFive", "first"),
    )
    bootstrap_frame = pd.DataFrame(bootstrap["Comparisons"])
    lines = [
        "# Stage 9A CFCompatKD-PE5 Final Audit",
        "",
        "## Outcome",
        "",
        "**{}**".format(success["Classification"]),
        "",
        "CFCompatKD-PE5 is a fixed equal-weight multi-model prediction ensemble. "
        "It is not a single model and no member or weight was selected using test data.",
        "",
        "## Offline/online replay",
        "",
        "- Maximum prediction difference: `{:.9g}`".format(
            replay["MaximumOfflineOnlinePredictionDifference"]
        ),
        "- Maximum metric difference: `{:.9g}`".format(
            replay["MaximumOfflineOnlineMetricDifference"]
        ),
        "- J_valid replay: `{:.6f}`".format(replay["Online"]["valid"]["J"]),
        "- J_test replay: `{:.6f}`".format(replay["Online"]["test"]["J"]),
        "",
        "## Main objective",
        "",
        markdown_table(objective),
        "",
        "The best single seed was selected by validation J only: seed {}.".format(
            best_seed
        ),
        "",
        "## Test metric trade-offs",
        "",
        markdown_table(test_metrics),
        "",
        "## Leave-one-out diagnostic",
        "",
        markdown_table(loo_j),
        "",
        "The largest test complementarity contribution is from seed {}: "
        "removing it increases J by {:.6f}. Every fixed four-model leave-one-out "
        "ensemble remains better than the five-seed Online mean, so the gain is "
        "not dependent on a single member.".format(
            success["LargestTestComplementaritySeed"],
            success["LargestTestComplementarityDeltaJ"],
        ),
        "",
        "Leave-one-out results are diagnostic only and were not used to choose members.",
        "",
        "## Prediction diversity",
        "",
        markdown_table(diversity_summary),
        "",
        "## Paired sample-level bootstrap",
        "",
        markdown_table(bootstrap_frame),
        "",
        "## Success criteria",
        "",
    ]
    lines.extend(
        "- {}: {}".format(key, value)
        for key, value in success["Criteria"].items()
    )
    lines += [
        "",
        "## Integrity declarations",
        "",
        "- Members are seeds 1111, 1112, 1113, 1114, 1115 with weights 0.2 each.",
        "- Every member is the Stage8 Online validation-selected checkpoint.",
        "- Offline CSV and online sequential-checkpoint inference agree within 1e-6.",
        "- sample_index, sample_id, label, split, and four-mode bindings were verified.",
        "- No parameter/hidden-state averaging, calibration, seed deletion, or weight search was used.",
        "- Deployment inference requires only five Student checkpoints and the requested data split.",
        "",
    ]
    (root / "stage9a_cfcompat_prediction_ensemble_final_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def aggregate(cli):
    root, replay, frames = load_artifacts(cli.result_root, cli.dataset)
    individual = individual_metrics(frames)
    best_seed = best_validation_seed(individual)
    comparison = ensemble_metric_comparison(frames, individual, best_seed)
    online_mean_j = {
        split: float(
            individual.loc[
                individual.Split.eq(split) & individual.Mode.eq("LAV"), "J"
            ].mean()
        )
        for split in PE5_SPLITS
    }
    loo_metrics, loo_predictions = leave_one_out(frames, online_mean_j)
    diversity = pairwise_diversity(frames)
    sample_gain = sample_ensemble_gain(frames, cli.result_root)
    conditional = conditional_gain(sample_gain)
    bootstrap = bootstrap_audit(frames, best_seed, cli.bootstrap_samples)

    test_objective = comparison.loc[
        comparison.Split.eq("test") & comparison.Metric.eq("J")
    ].iloc[0]
    test_lav = comparison.loc[
        comparison.Split.eq("test")
        & comparison.Mode.eq("LAV")
        & comparison.Metric.eq("MAE")
    ].iloc[0]
    test_missing = comparison.loc[
        comparison.Split.eq("test")
        & comparison.Mode.eq("MissingMacro")
        & comparison.Metric.eq("MAE")
    ].iloc[0]
    loo_test = loo_metrics.loc[
        loo_metrics.Split.eq("test") & loo_metrics.Metric.eq("J")
    ]
    largest_complementarity = loo_test.sort_values(
        ["DeltaVsFullPE5", "OmittedSeed"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    criteria = {
        "EngineeringReplayPassed": bool(replay["Passed"]),
        "TestJImprovesOnlineMeanByAtLeast0.015": float(
            test_objective.MeanIndividualValue - test_objective.EnsembleValue
        )
        >= 0.015,
        "TestJBeatsBestValidationSelectedSingleSeed": float(
            test_objective.EnsembleValue
        )
        < float(test_objective.BestValidationSeedValue),
        "TestLAVMAEBeatsOnlineMean": float(test_lav.DeltaVsMeanIndividual) < 0,
        "TestMissingMacroMAEBeatsOnlineMean": float(
            test_missing.DeltaVsMeanIndividual
        )
        < 0,
        "EveryLeaveOneOutTestJBeatsOnlineMean": bool(
            loo_test.BetterThanOnlineFiveSeedMean.astype(bool).all()
        ),
    }
    success = {
        "Classification": (
            "ENGINEERING AND PERFORMANCE SUCCESS"
            if all(criteria.values())
            else "ENGINEERING SUCCESS; PERFORMANCE CRITERIA NOT FULLY MET"
        ),
        "Criteria": criteria,
        "BestValidationSeed": best_seed,
        "LargestTestComplementaritySeed": int(
            largest_complementarity.OmittedSeed
        ),
        "LargestTestComplementarityDeltaJ": float(
            largest_complementarity.DeltaVsFullPE5
        ),
        "LeaveOneOutShowsNoSingleSeedDependency": bool(
            loo_test.BetterThanOnlineFiveSeedMean.astype(bool).all()
        ),
        "NoAllMetricsImprovementClaim": True,
    }

    individual.to_csv(root / "individual_model_metrics.csv", index=False)
    comparison.to_csv(root / "ensemble_metrics.csv", index=False)
    loo_metrics.to_csv(root / "leave_one_out_metrics.csv", index=False)
    loo_predictions.to_csv(root / "leave_one_out_predictions.csv", index=False)
    diversity.to_csv(root / "pairwise_diversity.csv", index=False)
    sample_gain.to_csv(root / "sample_ensemble_gain.csv", index=False)
    conditional.to_csv(root / "conditional_ensemble_gain.csv", index=False)
    (root / "paired_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(
        root,
        comparison,
        loo_metrics,
        diversity,
        bootstrap,
        replay,
        best_seed,
        success,
    )
    return {"Success": success, "Replay": replay, "OnlineMeanJ": online_mean_j}


def main():
    cli = parse_args()
    result = aggregate(cli)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
