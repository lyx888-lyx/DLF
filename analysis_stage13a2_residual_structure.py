"""Stage 13A-2: train-only cross-seed safe-residual structure audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stage13a1_safe_oracle import (
    MODES,
    OUTPUT_ROOT,
    SEEDS,
    sha256,
    stage11_path,
)
from trains.singleTask.cross_seed_median_residual import (
    is_strong_group,
    median_absolute_deviation,
    minimum_group_count,
    residual_sign,
    sign_agreement,
)
from trains.singleTask.decision_feasible_interval import (
    decision_cell,
    normalized_half,
    safe_oracle_predictions,
)


GROUP_COLUMNS = ["Mode", "Cell", "Half"]


def cell_name(cell):
    return "c7={}|c5={}|c2={}".format(
        int(cell[0]), int(cell[1]), int(bool(cell[2]))
    )


def attach_groups(frame, prediction_column):
    frame = frame.copy()
    cells, halves = [], []
    for prediction in frame[prediction_column].to_numpy(np.float32):
        cells.append(cell_name(decision_cell(prediction)))
        halves.append(normalized_half(prediction))
    frame["Cell"] = cells
    frame["Half"] = halves
    return frame


def train_residual_frame():
    rows = []
    expected_samples = None
    for seed in SEEDS:
        frame = pd.read_csv(
            stage11_path(seed, "train"), dtype={"sample_id": str}
        ).sort_values("sample_index", kind="mergesort")
        if frame.sample_index.duplicated().any():
            raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
        if expected_samples is None:
            expected_samples = len(frame)
        elif len(frame) != expected_samples:
            raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
        for mode in MODES:
            prediction = frame["{}_pred".format(mode)].to_numpy(np.float32)
            labels = frame.label.to_numpy(np.float32)
            target, _ = safe_oracle_predictions(prediction, labels)
            local = frame[
                ["sample_index", "sample_id", "label"]
            ].copy()
            local["Seed"] = seed
            local["Split"] = "train"
            local["Mode"] = mode
            local["Prediction"] = prediction
            local["SafeTarget"] = target
            local["SafeResidual"] = target - prediction
            local["SafeOracleMAEGain"] = (
                np.abs(prediction - labels) - np.abs(target - labels)
            )
            rows.append(attach_groups(local, "Prediction"))
    return pd.concat(rows, ignore_index=True), int(expected_samples)


def per_seed_statistics(train, n_min):
    rows = []
    for keys, frame in train.groupby(
        ["Seed"] + GROUP_COLUMNS, sort=True
    ):
        values = frame.SafeResidual.to_numpy(np.float64)
        q1, q3 = np.percentile(values, [25, 75])
        median = float(np.median(values))
        rows.append(
            {
                "Seed": int(keys[0]),
                "Mode": keys[1],
                "Cell": keys[2],
                "Half": int(keys[3]),
                "Count": len(frame),
                "NMin": n_min,
                "Supported": len(frame) >= n_min,
                "MedianResidual": median,
                "MeanResidual": float(values.mean()),
                "ResidualSign": residual_sign(median),
                "IQR": float(q3 - q1),
                "MAD": median_absolute_deviation(values),
                "PositiveResidualRatio": float((values > 0).mean()),
                "NegativeResidualRatio": float((values < 0).mean()),
                "SafeOracleMAEGain": float(
                    frame.SafeOracleMAEGain.mean()
                ),
                "SafeOracleMAEGainSum": float(
                    frame.SafeOracleMAEGain.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_seed_consistency(per_seed, n_min):
    rows = []
    for keys, frame in per_seed.groupby(GROUP_COLUMNS, sort=True):
        by_seed = frame.set_index("Seed")
        medians, counts = [], []
        for seed in SEEDS:
            if seed in by_seed.index:
                medians.append(float(by_seed.loc[seed, "MedianResidual"]))
                counts.append(int(by_seed.loc[seed, "Count"]))
            else:
                medians.append(np.nan)
                counts.append(0)
        supported = [
            median
            for median, count in zip(medians, counts)
            if count >= n_min and np.isfinite(median)
        ]
        agreement, direction = (
            sign_agreement(supported) if supported else (0, 0)
        )
        loso = {}
        for removed in SEEDS:
            values = [
                median
                for seed, median, count in zip(SEEDS, medians, counts)
                if seed != removed and count >= n_min and np.isfinite(median)
            ]
            loso[str(removed)] = (
                float(np.median(values)) if values else None
            )
        rows.append(
            {
                "Mode": keys[0],
                "Cell": keys[1],
                "Half": int(keys[2]),
                "SupportedSeedCount": len(supported),
                "SignAgreementCount": agreement,
                "AgreedSign": direction,
                "MedianResidualStdAcrossSeeds": (
                    float(np.std(supported, ddof=0))
                    if supported
                    else np.nan
                ),
                "MedianOfSeedMedians": (
                    float(np.median(supported)) if supported else np.nan
                ),
                "LeaveOneSeedOutMedians": json.dumps(
                    loso, sort_keys=True
                ),
                "SupportSampleCount": int(
                    sum(
                        count
                        for count in counts
                        if count >= n_min
                    )
                ),
                "StrongConsistent": is_strong_group(
                    medians, counts, n_min, required=4
                ),
            }
        )
    return pd.DataFrame(rows)


def valid_group_frame():
    rows = []
    for seed in SEEDS:
        # The authoritative Online valid file is read without a label column.
        source = (
            Path("/code/DLF/result/missing_baseline/")
            / "cfcompat_prediction_ensemble_v1/mosi"
            / "online_seed{}_valid_predictions.csv".format(seed)
        )
        frame = pd.read_csv(
            source,
            usecols=[
                "sample_index",
                "sample_id",
                *["{}_pred".format(mode) for mode in MODES],
            ],
            dtype={"sample_id": str},
        )
        for mode in MODES:
            local = frame[["sample_index", "sample_id"]].copy()
            local["Seed"] = seed
            local["Split"] = "valid"
            local["Mode"] = mode
            local["Prediction"] = frame[
                "{}_pred".format(mode)
            ].to_numpy(np.float32)
            rows.append(attach_groups(local, "Prediction"))
    return pd.concat(rows, ignore_index=True)


def coverage_outputs(valid, consistency):
    flags = consistency[
        GROUP_COLUMNS
        + [
            "StrongConsistent",
            "SupportedSeedCount",
            "SignAgreementCount",
        ]
    ]
    mapped = valid.merge(flags, on=GROUP_COLUMNS, how="left", validate="many_to_one")
    mapped["StrongConsistent"] = mapped.StrongConsistent.fillna(False)
    mapped["RobustFourOrFiveSign"] = (
        mapped.SignAgreementCount.fillna(0).ge(4)
    )
    rows = []
    for mode, frame in list(mapped.groupby("Mode", sort=True)) + [
        ("Overall", mapped)
    ]:
        strong = frame.StrongConsistent
        rows.append(
            {
                "Scope": mode,
                "Samples": len(frame),
                "StrongCoverageCount": int(strong.sum()),
                "StrongCoverageRatio": float(strong.mean()),
                "FourOrFiveSignWithinStrongRatio": (
                    float(
                        frame.loc[
                            strong, "RobustFourOrFiveSign"
                        ].mean()
                    )
                    if strong.any()
                    else 0.0
                ),
            }
        )
    return mapped, pd.DataFrame(rows)


def loso_coverage(valid, per_seed, full_coverage, n_min):
    rows = []
    for removed in SEEDS:
        subset = per_seed.loc[per_seed.Seed.ne(removed)]
        flags = []
        for keys, frame in subset.groupby(GROUP_COLUMNS, sort=True):
            by_seed = frame.set_index("Seed")
            remaining = [seed for seed in SEEDS if seed != removed]
            medians, counts = [], []
            for seed in remaining:
                if seed in by_seed.index:
                    medians.append(
                        float(by_seed.loc[seed, "MedianResidual"])
                    )
                    counts.append(int(by_seed.loc[seed, "Count"]))
                else:
                    medians.append(np.nan)
                    counts.append(0)
            flags.append(
                {
                    "Mode": keys[0],
                    "Cell": keys[1],
                    "Half": keys[2],
                    "StrongLOSO": is_strong_group(
                        medians, counts, n_min, required=3
                    ),
                }
            )
        mapped = valid.merge(
            pd.DataFrame(flags), on=GROUP_COLUMNS, how="left"
        )
        coverage = float(mapped.StrongLOSO.fillna(False).mean())
        rows.append(
            {
                "RemovedSeed": removed,
                "StrongCoverageRatio": coverage,
                "CoverageDrop": full_coverage - coverage,
            }
        )
    return pd.DataFrame(rows)


def extreme_coherence(consistency, mapped):
    strong = consistency.loc[consistency.StrongConsistent].copy()
    strong["C7"] = strong.Cell.str.extract(r"c7=(-?\d+)")[0].astype(int)
    present = set(
        mapped.Cell.str.extract(r"c7=(-?\d+)")[0].astype(int).to_list()
    )
    negative_needed = any(value <= -2 for value in present)
    positive_needed = any(value >= 2 for value in present)
    negative_supported = bool((strong.C7 <= -2).any())
    positive_supported = bool((strong.C7 >= 2).any())
    # The calibrator is cell-conditional, so opposite correction directions
    # across negative and positive extreme cells are allowed.  Coherence means
    # that each occupied extreme side has at least one cross-seed strong rule.
    return (
        (not negative_needed or negative_supported)
        and (not positive_needed or positive_supported)
    )


def main():
    output = OUTPUT_ROOT
    stage = output / "stage13a2_residual_structure"
    a1_gate = json.loads(
        (
            output
            / "stage13a1_safe_oracle/stage13a1_gate.json"
        ).read_text()
    )
    if not a1_gate["Passed"]:
        raise RuntimeError("Stage13A-1 gate is not open.")
    train, train_samples = train_residual_frame()
    n_min = minimum_group_count(train_samples)
    per_seed = per_seed_statistics(train, n_min)
    consistency = cross_seed_consistency(per_seed, n_min)
    valid = valid_group_frame()
    mapped, coverage = coverage_outputs(valid, consistency)
    overall = float(
        coverage.loc[coverage.Scope.eq("Overall"), "StrongCoverageRatio"].iloc[0]
    )
    loso = loso_coverage(valid, per_seed, overall, n_min)
    cell_gain = (
        train.groupby("Cell", sort=True)
        .SafeOracleMAEGain.sum()
        .sort_values(ascending=False)
    )
    max_cell_ratio = float(cell_gain.iloc[0] / cell_gain.sum())
    mode_count = int(
        coverage.loc[
            coverage.Scope.isin(MODES)
            & coverage.StrongCoverageRatio.ge(0.60)
        ].shape[0]
    )
    robust_ratio = float(
        coverage.loc[
            coverage.Scope.eq("Overall"),
            "FourOrFiveSignWithinStrongRatio",
        ].iloc[0]
    )
    conditions = {
        "StrongGroupsCoverAtLeast70PercentValid": overall >= 0.70,
        "AtLeast70PercentStrongSamplesHaveFourOrFiveSignAgreement": (
            robust_ratio >= 0.70
        ),
        "AtLeastThreeModesCoverAtLeast60Percent": mode_count >= 3,
        "ExtremeDirectionsCoherent": extreme_coherence(consistency, mapped),
        "AnySeedRemovalDropsCoverageAtMost15Points": bool(
            loso.CoverageDrop.max() <= 0.15
        ),
        "LargestCellGainContributionAtMost60Percent": max_cell_ratio <= 0.60,
    }
    engineering = {
        "Stage13A1Passed": True,
        "TrainLabelsOnly": True,
        "NoValidLabelsRead": True,
        "NoTestLoaderConstructed": True,
        "NoTestLabelsRead": True,
        "NoTestPredictionsRead": True,
        "NoTestEvaluationPerformed": True,
        "OfficialDecisionCellsUsed": True,
        "NoNaNInf": bool(
            np.isfinite(
                per_seed.select_dtypes(include=[np.number]).to_numpy()
            ).all()
        ),
    }
    passed = bool(all(engineering.values()) and all(conditions.values()))
    gate = {
        "Passed": passed,
        "Verdict": (
            "STAGE13A2_CROSS_SEED_RESIDUAL_STRUCTURE_SUPPORTED"
            if passed
            else "STAGE13A2_NO_STABLE_RESIDUAL_STRUCTURE"
        ),
        "EngineeringConditions": engineering,
        "Conditions": conditions,
        "Metrics": {
            "TrainSamplesPerSeed": train_samples,
            "NMin": n_min,
            "StrongGroupValidCoverage": overall,
            "FourOrFiveSignWithinStrongRatio": robust_ratio,
            "ModesAtLeast60PercentCoverage": mode_count,
            "WorstLOSOValidCoverageDrop": float(
                loso.CoverageDrop.max()
            ),
            "LargestCellGainContribution": max_cell_ratio,
            "StrongGroups": int(consistency.StrongConsistent.sum()),
        },
    }
    per_seed.to_csv(stage / "stage13a2_residual_per_seed.csv", index=False)
    consistency.to_csv(
        stage / "stage13a2_cross_seed_consistency.csv", index=False
    )
    coverage.to_csv(stage / "stage13a2_group_coverage.csv", index=False)
    loso.to_csv(stage / "stage13a2_loso_coverage.csv", index=False)
    pd.DataFrame(
        {
            "Cell": cell_gain.index,
            "Gain": cell_gain.values,
            "ContributionRatio": cell_gain.values / cell_gain.sum(),
        }
    ).to_csv(stage / "stage13a2_cell_gain_contribution.csv", index=False)
    (stage / "stage13a2_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# Stage 13A-2 Cross-Seed Residual Structure Audit",
        "",
        "- Train labels used: true",
        "- Valid labels used: false",
        "- Test accessed: false",
        "- n_min: {}".format(n_min),
        "",
        "## Gate",
        "",
    ]
    report.extend(
        "- {}: {}".format(key, "PASS" if value else "FAIL")
        for key, value in conditions.items()
    )
    report.extend(
        [
            "",
            "- Strong-group valid coverage: {:.6f}".format(overall),
            "- Modes with >=60% coverage: {}/4".format(mode_count),
            "- Worst LOSO coverage drop: {:.6f}".format(
                loso.CoverageDrop.max()
            ),
            "- Largest Cell gain contribution: {:.6f}".format(
                max_cell_ratio
            ),
            "- Verdict: **{}**".format(gate["Verdict"]),
            "",
            (
                "Proceed to Stage13A-3."
                if passed
                else "CS-DFMRC PIPELINE STOPPED BY EVIDENCE GATE"
            ),
        ]
    )
    report_path = stage / "stage13a2_residual_structure_audit.md"
    report_path.write_text("\n".join(report) + "\n")
    manifest = {
        "Stage": "Stage13A-2",
        "Verdict": gate["Verdict"],
        "TrainLabelsUsed": True,
        "ValidLabelsRead": False,
        "TestLoaderConstructed": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "ResidualTableSHA256": sha256(
            stage / "stage13a2_residual_per_seed.csv"
        ),
        "ConsistencyTableSHA256": sha256(
            stage / "stage13a2_cross_seed_consistency.csv"
        ),
    }
    (stage / "stage13a2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(gate["Verdict"], flush=True)
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
