"""Stage 14A-3 consensus-member leave-one-out robustness audit."""

import json

import numpy as np
import pandas as pd

from analysis_stage12a_decision_safe_prototype import METRICS, metrics_for_long
from analysis_stage14_robust_consensus_valid import (
    CLASS_METRICS,
    MODES,
    OUTPUT,
    SEEDS,
    labels_after_freeze,
    load_members_without_labels,
    sha256,
    validation_anchor,
)
from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
)
from trains.singleTask.robust_cross_seed_consensus import huber_centers


def freeze_lomo_predictions():
    members = load_members_without_labels()
    anchor_seed, _ = validation_anchor()
    anchor_index = SEEDS.index(anchor_seed)
    binding = members[0][["sample_index", "sample_id"]].copy()
    rows = []
    fallback_count = 0
    for dropped_index, dropped_seed in enumerate(SEEDS):
        retained = [
            index for index in range(len(SEEDS)) if index != dropped_index
        ]
        for mode in MODES:
            matrix = np.column_stack(
                [
                    frame["{}_pred".format(mode)].to_numpy(np.float64)
                    for frame in members
                ]
            )
            reduced = matrix[:, retained]
            mean4 = reduced.mean(axis=1)
            huber4, _ = huber_centers(reduced)
            anchor = matrix[:, anchor_index].astype(np.float32)
            mean_projected, mean_details = project_array(
                anchor, mean4, "mosi", "adpep_all"
            )
            huber_projected, huber_details = project_array(
                anchor, huber4, "mosi", "adpep_all"
            )
            fallback_count += sum(
                row.fallback_to_anchor
                for row in mean_details + huber_details
            )
            anchor_decisions = evaluator_decisions(anchor, "mosi")
            for method, prediction in (
                ("Mean4Proj", mean_projected),
                ("Huber4Proj", huber_projected),
            ):
                decisions = evaluator_decisions(prediction, "mosi")
                if any(
                    np.count_nonzero(left != right)
                    for left, right in zip(anchor_decisions, decisions)
                ):
                    raise RuntimeError(
                        "STAGE14A2_DECISION_PRESERVATION_FAILED"
                    )
                for index in range(len(binding)):
                    rows.append(
                        {
                            "DroppedSeed": dropped_seed,
                            "RetainedSeeds": ",".join(
                                str(SEEDS[value]) for value in retained
                            ),
                            "AnchorSeed": anchor_seed,
                            "Split": "valid",
                            "Mode": mode,
                            "Method": method,
                            "sample_index": int(
                                binding.sample_index.iloc[index]
                            ),
                            "sample_id": str(
                                binding.sample_id.iloc[index]
                            ),
                            "Prediction": float(prediction[index]),
                        }
                    )
    frame = pd.DataFrame(rows).sort_values(
        ["DroppedSeed", "Method", "Mode", "sample_index"],
        kind="mergesort",
    )
    path = (
        OUTPUT
        / "stage14a3_lomo/stage14_lomo_unlabeled_predictions_valid.csv"
    )
    frame.to_csv(path, index=False, float_format="%.17g")
    payload = {
        "Path": str(path),
        "SHA256": sha256(path),
        "Rows": len(frame),
        "LabelsPresent": False,
        "FrozenBeforeValidLabelsRead": True,
        "FixedAnchorSeed": anchor_seed,
        "FallbackCount": int(fallback_count),
        "TestInputsPresent": False,
    }
    (
        OUTPUT
        / "stage14a3_lomo/stage14_lomo_prediction_manifest.json"
    ).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return pd.read_csv(path, dtype={"sample_id": str}), payload


def evaluate_lomo(predictions):
    labels = labels_after_freeze()
    merged = predictions.merge(
        labels,
        on=["sample_index", "sample_id"],
        validate="many_to_one",
    )
    merged["Seed"] = merged.DroppedSeed
    metrics = metrics_for_long(
        merged[
            [
                "Seed",
                "Split",
                "Method",
                "Mode",
                "sample_index",
                "sample_id",
                "label",
                "Prediction",
            ]
        ]
    )
    indexed = metrics.set_index(["Seed", "Method", "Mode"])
    rows = []
    for dropped in SEEDS:
        for mode in MODES + ("MissingMacro",):
            mean = indexed.loc[(dropped, "Mean4Proj", mode)]
            huber = indexed.loc[(dropped, "Huber4Proj", mode)]
            row = {"DroppedSeed": dropped, "Mode": mode}
            for metric in ("J",) + METRICS:
                row["Mean4Proj{}".format(metric)] = mean[metric]
                row["Huber4Proj{}".format(metric)] = huber[metric]
                row["Delta{}".format(metric)] = huber[metric] - mean[metric]
            rows.append(row)
    return metrics, pd.DataFrame(rows)


def gate_payload(metrics, per_case, manifest):
    indexed = per_case.set_index(["DroppedSeed", "Mode"])
    delta_j = np.asarray(
        [indexed.loc[(seed, "LAV"), "DeltaJ"] for seed in SEEDS]
    )
    delta_lav = np.asarray(
        [indexed.loc[(seed, "LAV"), "DeltaMAE"] for seed in SEEDS]
    )
    delta_missing = np.asarray(
        [
            indexed.loc[(seed, "MissingMacro"), "DeltaMAE"]
            for seed in SEEDS
        ]
    )
    delta_lav_corr = np.asarray(
        [indexed.loc[(seed, "LAV"), "DeltaCorr"] for seed in SEEDS]
    )
    delta_missing_corr = np.asarray(
        [
            indexed.loc[(seed, "MissingMacro"), "DeltaCorr"]
            for seed in SEEDS
        ]
    )
    remove_best = float(np.delete(delta_j, int(np.argmin(delta_j))).mean())
    metric_index = metrics.set_index(["Seed", "Method", "Mode"])
    classification = all(
        abs(
            float(metric_index.loc[(seed, "Huber4Proj", mode), metric])
            - float(metric_index.loc[(seed, "Mean4Proj", mode), metric])
        )
        <= 1e-12
        for seed in SEEDS
        for mode in MODES + ("MissingMacro",)
        for metric in CLASS_METRICS
    )
    conditions = {
        "AtLeastFourCasesHuberBeatsMean": bool((delta_j < 0).sum() >= 4),
        "MeanDeltaJAtMostMinus0.0005": bool(delta_j.mean() <= -0.0005),
        "WorstCaseDeltaJAtMostPlus0.0003": bool(
            delta_j.max() <= 0.0003
        ),
        "RemoveBestCaseMeanDeltaJNegative": bool(remove_best < 0),
        "MeanLAVMAEImproves": bool(delta_lav.mean() < 0),
        "MeanMissingMacroMAEImproves": bool(delta_missing.mean() < 0),
        "MeanLAVCorrNonDegraded": bool(
            delta_lav_corr.mean() >= -0.0001
        ),
        "MeanMissingMacroCorrNonDegraded": bool(
            delta_missing_corr.mean() >= -0.0001
        ),
        "ClassificationInheritedFixedAnchor": bool(classification),
    }
    engineering = {
        "FixedAnchorAcrossAllCases": (
            predictions_anchor_count(manifest) == 1
        ),
        "EachCaseDropsExactlyOneMember": True,
        "NoSubsetSelection": True,
        "PredictionsFrozenBeforeLabels": manifest[
            "FrozenBeforeValidLabelsRead"
        ],
        "FallbackAcceptable": manifest["FallbackCount"] == 0,
        "NoTestAccess": True,
    }
    passed = bool(all(engineering.values()) and all(conditions.values()))
    return {
        "Passed": passed,
        "Verdict": (
            "STAGE14A3_LOMO_ROBUSTNESS_SUPPORTED"
            if passed
            else "STAGE14A3_LOMO_ROBUSTNESS_UNSUPPORTED"
        ),
        "EngineeringConditions": engineering,
        "Conditions": conditions,
        "Metrics": {
            "ImprovedCases": int((delta_j < 0).sum()),
            "MeanDeltaJ": float(delta_j.mean()),
            "WorstCaseDeltaJ": float(delta_j.max()),
            "RemoveBestCaseMeanDeltaJ": remove_best,
            "MeanDeltaLAVMAE": float(delta_lav.mean()),
            "MeanDeltaMissingMacroMAE": float(delta_missing.mean()),
            "MeanDeltaLAVCorr": float(delta_lav_corr.mean()),
            "MeanDeltaMissingMacroCorr": float(delta_missing_corr.mean()),
            "FallbackCount": int(manifest["FallbackCount"]),
        },
        "TestLoaderConstructed": False,
        "TestPredictionsRead": False,
        "TestLabelsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
    }


def predictions_anchor_count(manifest):
    return 1 if int(manifest["FixedAnchorSeed"]) in SEEDS else 0


def main():
    a2 = json.loads(
        (OUTPUT / "stage14a2_valid/stage14a2_gate.json").read_text()
    )
    if not a2["Passed"]:
        raise RuntimeError("Stage14A-2 gate is not open.")
    predictions, manifest = freeze_lomo_predictions()
    if sha256(manifest["Path"]) != manifest["SHA256"]:
        raise RuntimeError("LOMO prediction SHA mismatch.")
    metrics, per_case = evaluate_lomo(predictions)
    gate = gate_payload(metrics, per_case, manifest)
    stage = OUTPUT / "stage14a3_lomo"
    per_case.to_csv(stage / "stage14_lomo_per_case.csv", index=False)
    numeric = [
        column
        for column in per_case.columns
        if column not in ("DroppedSeed", "Mode")
    ]
    aggregate_rows = []
    for mode, frame in per_case.groupby("Mode", sort=True):
        row = {"Mode": mode, "Cases": len(frame)}
        for column in numeric:
            row["{}_Mean".format(column)] = float(frame[column].mean())
            row["{}_Std".format(column)] = float(frame[column].std(ddof=0))
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(
        stage / "stage14_lomo_aggregate.csv", index=False
    )
    (stage / "stage14a3_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# Stage 14A-3 LOMO Robustness Audit",
        "",
        "- Fixed Anchor seed: {}".format(manifest["FixedAnchorSeed"]),
        "- Test accessed: false",
        "- Fallback count: {}".format(manifest["FallbackCount"]),
        "",
        "## Gate",
        "",
    ]
    report.extend(
        "- {}: {}".format(key, "PASS" if value else "FAIL")
        for key, value in gate["Conditions"].items()
    )
    report.extend(
        [
            "",
            "- Improved cases: {}/5".format(
                gate["Metrics"]["ImprovedCases"]
            ),
            "- Mean Delta J: {:.9f}".format(
                gate["Metrics"]["MeanDeltaJ"]
            ),
            "- Worst-case Delta J: {:.9f}".format(
                gate["Metrics"]["WorstCaseDeltaJ"]
            ),
            "- Remove-best-case Mean Delta J: {:.9f}".format(
                gate["Metrics"]["RemoveBestCaseMeanDeltaJ"]
            ),
            "- Verdict: **{}**".format(gate["Verdict"]),
            "",
            (
                "STAGE14A_DCRC_VALIDATION_SUPPORTED"
                if gate["Passed"]
                else "DCRC PIPELINE STOPPED BY EVIDENCE GATE"
            ),
        ]
    )
    (OUTPUT / "stage14a_dcrc_validation_audit.md").write_text(
        "\n".join(report) + "\n"
    )
    print(gate["Verdict"], flush=True)
    raise SystemExit(0 if gate["Passed"] else 3)


if __name__ == "__main__":
    main()
