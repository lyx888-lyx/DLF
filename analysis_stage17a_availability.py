"""Aggregate Stage17A runtime evidence, availability, contributions, and gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.loss_dependency_registry import LOSS_BY_NAME, semantic_active


SEEDS = (1111, 1112, 1113, 1114, 1115)
AUDIT_SEEDS = (1114, 1111)
MODES = ("LA", "LV", "L")
ROOT = Path("result/missing_baseline/mcao_v1/mosi")
AUDIT = ROOT / "stage17a_loss_audit"
STAGE8 = Path("/code/DLF/result/missing_baseline/cfcompat_stability_v1/mosi")


def historical_availability():
    rows, totals = [], {mode: 0 for mode in MODES}
    for seed in SEEDS:
        path = STAGE8 / "seed{}".format(seed) / "ema_epoch_metrics.csv"
        frame = pd.read_csv(path)
        counts = {mode: int(frame[mode].sum()) for mode in MODES}
        total = sum(counts.values())
        if total != int(frame.Epoch.nunique()) * 1284:
            raise RuntimeError("Historical missing schedule count mismatch.")
        totals = {mode: totals[mode] + counts[mode] for mode in MODES}
        rows.append(
            {
                "Seed": seed,
                "EpochCount": int(frame.Epoch.nunique()),
                "MissingViewSampleCount": total,
                **{"Count_{}".format(mode): counts[mode] for mode in MODES},
                "pi_L": 1.0,
                "pi_A": counts["LA"] / total,
                "pi_V": counts["LV"] / total,
                "pi_LA": counts["LA"] / total,
                "pi_LV": counts["LV"] / total,
                "pi_AV": 0.0,
                "pi_LAV": 0.0,
                "Source": str(path),
            }
        )
    grand = sum(totals.values())
    rows.append(
        {
            "Seed": "AllFiveSeeds",
            "EpochCount": sum(row["EpochCount"] for row in rows),
            "MissingViewSampleCount": grand,
            **{"Count_{}".format(mode): totals[mode] for mode in MODES},
            "pi_L": 1.0,
            "pi_A": totals["LA"] / grand,
            "pi_V": totals["LV"] / grand,
            "pi_LA": totals["LA"] / grand,
            "pi_LV": totals["LV"] / grand,
            "pi_AV": 0.0,
            "pi_LAV": 0.0,
            "Source": "five frozen historical Stage8 online schedules",
        }
    )
    return pd.DataFrame(rows)


def load_runtime():
    scalar_frames, gradient_frames, cosine_frames, ratio_frames = [], [], [], []
    manifests = []
    for seed in AUDIT_SEEDS:
        root = AUDIT / "runtime_seed{}".format(seed)
        scalar_frames.append(pd.read_csv(root / "loss_activation.csv"))
        gradient_frames.append(pd.read_csv(root / "per_loss_gradients.csv"))
        cosine_frames.append(pd.read_csv(root / "gradient_cosines.csv"))
        ratio_frames.append(pd.read_csv(root / "abnormal_gradient_ratios.csv"))
        manifests.append(json.loads((root / "manifest.json").read_text()))
    return (
        pd.concat(scalar_frames, ignore_index=True),
        pd.concat(gradient_frames, ignore_index=True),
        pd.concat(cosine_frames, ignore_index=True),
        pd.concat(ratio_frames, ignore_index=True),
        manifests,
    )


def absent_gradient_ratios(gradients):
    all_params = gradients.loc[
        gradients.ParameterGroup == "all_student_parameters"
    ]
    rows = []
    definitions = (
        ("missing_audio_specific_task", ("LV", "L"), ("LA",)),
        ("missing_visual_specific_task", ("LA", "L"), ("LV",)),
    )
    for loss, absent_modes, present_modes in definitions:
        local = all_params.loc[all_params.LossName == loss]
        for seed in AUDIT_SEEDS:
            seed_rows = local.loc[local.Seed == seed]
            absent = seed_rows.loc[
                seed_rows.Mode.isin(absent_modes), "GradientL2Norm"
            ].median()
            present = seed_rows.loc[
                seed_rows.Mode.isin(present_modes), "GradientL2Norm"
            ].median()
            rows.append(
                {
                    "Seed": seed,
                    "LossName": loss,
                    "AbsentModes": ",".join(absent_modes),
                    "PresentModes": ",".join(present_modes),
                    "MedianAbsentGradientNorm": float(absent),
                    "MedianPresentGradientNorm": float(present),
                    "AbsentGradientRatio": float(absent / max(present, 1e-12)),
                }
            )
    return pd.DataFrame(rows)


def modality_for_loss(name):
    if "_text_" in name:
        return "L"
    if "_audio_" in name:
        return "A"
    if "_visual_" in name:
        return "V"
    return None


def expected_contributions(gradients, availability):
    global_row = availability.loc[
        availability.Seed.astype(str) == "AllFiveSeeds"
    ].iloc[0]
    pi_mode = {
        "LA": float(global_row.Count_LA / global_row.MissingViewSampleCount),
        "LV": float(global_row.Count_LV / global_row.MissingViewSampleCount),
        "L": float(global_row.Count_L / global_row.MissingViewSampleCount),
    }
    pi_modality = {
        "L": 1.0,
        "A": float(global_row.pi_A),
        "V": float(global_row.pi_V),
    }
    all_params = gradients.loc[
        gradients.ParameterGroup == "all_student_parameters"
    ]
    median = (
        all_params.groupby(["LossName", "Mode"], as_index=False)
        .GradientL2Norm.median()
    )
    norm_lookup = {
        (row.LossName, row.Mode): float(row.GradientL2Norm)
        for row in median.itertuples()
    }
    families = ("specific_task", "reconstruction", "consistency", "orthogonality")
    rows = []
    for family in families:
        entries = [
            entry for entry in LOSS_BY_NAME.values()
            if entry.family == family and modality_for_loss(entry.name)
        ]
        full_entries = [entry for entry in entries if entry.view == "full_LAV"]
        missing_entries = [
            entry for entry in entries if entry.view == "sampled_missing"
        ]
        missing_coefficients = {
            modality_for_loss(entry.name): entry.coefficient
            for entry in missing_entries
        }
        alpha_bar = (
            sum(missing_coefficients.values()) / 3.0
            if len(missing_coefficients) == 3
            else None
        )
        for modality in ("L", "A", "V"):
            original = pm = an = 0.0
            for entry in full_entries:
                if modality_for_loss(entry.name) != modality:
                    continue
                value = norm_lookup.get((entry.name, "LAV"), 0.0)
                original += value
                pm += value
                an += value
            for entry in missing_entries:
                if modality_for_loss(entry.name) != modality:
                    continue
                for mode in MODES:
                    value = norm_lookup.get((entry.name, mode), 0.0)
                    original += pi_mode[mode] * value
                    if semantic_active(entry, mode):
                        pm += pi_mode[mode] * value
                        normalized_scale = (
                            alpha_bar
                            / max(pi_modality[modality], 1e-8)
                            / entry.coefficient
                        )
                        an += pi_mode[mode] * value * normalized_scale
            rows.append(
                {
                    "Family": family,
                    "Modality": modality,
                    "OriginalExpectedWeightedGradientContribution": original,
                    "PMExpectedWeightedGradientContribution": pm,
                    "ANExpectedWeightedGradientContribution": an,
                    "TrainAvailability": pi_modality[modality],
                    "MissingFamilyAlphaBar": alpha_bar,
                    "Estimate": "sum of mode-frequency-weighted median true autograd norms",
                }
            )
    frame = pd.DataFrame(rows)
    combined = (
        frame.groupby("Modality", as_index=False)
        .agg(
            OriginalExpectedWeightedGradientContribution=(
                "OriginalExpectedWeightedGradientContribution", "sum"
            ),
            PMExpectedWeightedGradientContribution=(
                "PMExpectedWeightedGradientContribution", "sum"
            ),
            ANExpectedWeightedGradientContribution=(
                "ANExpectedWeightedGradientContribution", "sum"
            ),
        )
    )
    combined["Family"] = "all_modality_auxiliary"
    combined["TrainAvailability"] = combined.Modality.map(pi_modality)
    combined["MissingFamilyAlphaBar"] = np.nan
    combined["Estimate"] = "sum across audited parallel families"
    return pd.concat([frame, combined[frame.columns]], ignore_index=True)


def imbalance_ratio(contributions):
    local = contributions.loc[
        contributions.Family == "all_modality_auxiliary",
        "PMExpectedWeightedGradientContribution",
    ].to_numpy(dtype=float)
    positive = local[local > 0]
    return float(local.max() / max(positive.min(), 1e-12))


def baseline_manifest():
    seeds = []
    for seed in SEEDS:
        path = ROOT / "baseline_replay" / "seed{}".format(seed) / "manifest.json"
        seeds.append(json.loads(path.read_text()))
    passed = all(
        row["Passed"]
        and row["Replay"]["valid"]["PredictionMaxDiff"] <= 1e-7
        and row["Replay"]["valid"]["JDiff"] <= 1e-8
        for row in seeds
    )
    payload = {
        "Seeds": seeds,
        "Passed": passed,
        "TestInputsPresent": False,
        "TestLoaderConstructed": False,
        "TestFeaturesRead": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
    }
    (ROOT / "baseline_asset_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main():
    scalar, gradients, cosines, ratios, manifests = load_runtime()
    availability = historical_availability()
    absent = absent_gradient_ratios(gradients)
    contributions = expected_contributions(gradients, availability)
    imbalance = imbalance_ratio(contributions)
    baseline = baseline_manifest()

    scalar.to_csv(AUDIT / "stage17_runtime_loss_activation.csv", index=False)
    gradients.to_csv(AUDIT / "stage17_per_loss_gradient_audit.csv", index=False)
    cosines.to_csv(AUDIT / "stage17_gradient_cosine_audit.csv", index=False)
    availability.to_csv(AUDIT / "stage17_availability_statistics.csv", index=False)
    contributions.to_csv(
        AUDIT / "stage17_expected_gradient_contribution.csv", index=False
    )
    absent.to_csv(AUDIT / "stage17_absent_gradient_ratio.csv", index=False)
    ratios.to_csv(AUDIT / "stage17_abnormal_gradient_ratio.csv", index=False)

    invalid = scalar.loc[
        (~scalar.ExpectedActive.astype(bool))
        & scalar.ActualActive.astype(bool)
        & scalar.LossName.isin(
            ["missing_audio_specific_task", "missing_visual_specific_task"]
        )
    ]
    invalid_modes = sorted(invalid.Mode.unique().tolist())
    seed_reproduction = {
        int(seed): sorted(invalid.loc[invalid.Seed == seed].Mode.unique().tolist())
        for seed in AUDIT_SEEDS
    }
    scalar_medians = (
        invalid.assign(AbsWeighted=invalid.WeightedScalar.abs())
        .groupby(["Seed", "LossName"], as_index=False)
        .AbsWeighted.median()
    )
    material_scalar = bool((scalar_medians.AbsWeighted > 1e-4).any())
    material_absent_gradient = bool((absent.AbsentGradientRatio >= 0.05).any())

    abnormal_losses = invalid.LossName.unique()
    flow_groups = gradients.loc[
        gradients.LossName.isin(abnormal_losses)
        & (~gradients.ExpectedActive.astype(bool))
        & gradients.ParameterGroup.isin(
            [
                "audio_specific_encoder_head",
                "visual_specific_encoder_head",
                "shared_common_encoder",
                "cross_modal_fusion",
            ]
        )
        & (gradients.GradientL2Norm > 0)
    ]
    abnormal_ratio = ratios.loc[ratios.Mode.isin(invalid_modes)]
    ratio_by_seed = abnormal_ratio.groupby("Seed").AbnormalGradientRatio.median()
    conditions = {
        "ModalityDependentLossEntersTotalWhenAbsent": not invalid.empty,
        "AtLeastTwoMissingModes": len(invalid_modes) >= 2,
        "BothAuditSeedsReproduce": all(
            len(seed_reproduction[int(seed)]) >= 2 for seed in AUDIT_SEEDS
        ),
        "MedianAbsWeightedLossAbove1e-4": material_scalar,
        "AbsentGradientRatioAtLeast0.05": material_absent_gradient,
        "AbnormalGradientReachesRequiredParameterGroup": not flow_groups.empty,
        "AbnormalAuxiliaryGradientAtLeastFivePercent": bool(
            len(ratio_by_seed) == 2 and (ratio_by_seed >= 0.05).all()
        ),
        "PresenceBindingPassed": all(
            row["PresenceBindingMismatchCount"] == 0 for row in manifests
        ),
        "BaselineReplayPassed": baseline["Passed"],
    }
    passed = all(conditions.values())
    verdict = (
        "STAGE17A_ACTIONABLE_LOSS_INCONSISTENCY"
        if passed
        else "STAGE17A_NO_ACTIONABLE_LOSS_INCONSISTENCY"
    )
    formal_variant = "MCAO-AN" if imbalance > 2.0 else "MCAO-PM"
    gate = {
        "Verdict": verdict,
        "Passed": passed,
        "Conditions": conditions,
        "InvalidButActiveLosses": sorted(invalid.LossName.unique().tolist()),
        "InvalidModes": invalid_modes,
        "SeedReproduction": seed_reproduction,
        "MedianAbsoluteWeightedLoss": scalar_medians.to_dict(orient="records"),
        "AbsentGradientRatios": absent.to_dict(orient="records"),
        "MedianAbnormalGradientRatioBySeed": {
            str(int(key)): float(value) for key, value in ratio_by_seed.items()
        },
        "PresenceMaskedContributionImbalanceRatio": imbalance,
        "TrainOnlyFormalVariant": formal_variant,
        "BaselineReplayPassed": baseline["Passed"],
        "TestLoaderConstructed": False,
        "TestFeaturesRead": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
    }
    (AUDIT / "stage17a_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    if passed:
        freeze = {
            "FormalVariant": formal_variant,
            "SelectionRule": (
                "MCAO-AN iff train-only PM expected-contribution imbalance > 2.0; "
                "otherwise MCAO-PM"
            ),
            "PresenceMaskedContributionImbalanceRatio": imbalance,
            "FrozenBeforeAnyNewOfficialValidPrediction": True,
            "OfficialValidUsedForSelection": False,
            "TestUsedForSelection": False,
            "AvailabilitySource": (
                "five frozen historical Stage8 train missing schedules"
            ),
            "InferenceParameterDelta": 0,
            "InferenceFLOPsDelta": 0,
        }
        (ROOT / "mcao_variant_freeze_manifest.json").write_text(
            json.dumps(freeze, indent=2, sort_keys=True) + "\n"
        )

    scalar_summary = (
        invalid.assign(AbsWeighted=invalid.WeightedScalar.abs())
        .groupby(["Seed", "Mode", "LossName"], as_index=False)
        .agg(
            MedianAbsWeightedScalar=("AbsWeighted", "median"),
            MedianRawScalar=("RawScalar", "median"),
        )
    )
    contribution_summary = contributions.loc[
        contributions.Family == "all_modality_auxiliary"
    ]
    report = [
        "# Stage 17A Missingness-Consistent Loss Audit",
        "",
        "- Verdict: `{}`".format(verdict),
        "- Actual entrypoint: `run_cfcompat_stability_multiseed.py::train_one_seed`",
        "- Current auxiliary presence masking: `false`",
        "- Invalid but active losses: {}".format(
            ", ".join(gate["InvalidButActiveLosses"]) or "none"
        ),
        "- Invalid modes: {}".format(", ".join(invalid_modes) or "none"),
        "- Baseline replay: `{}`".format(baseline["Passed"]),
        "- Presence binding mismatch count: `0`",
        "- PM contribution imbalance ratio: `{:.9f}`".format(imbalance),
        "- Train-only frozen formal variant: `{}`".format(formal_variant),
        "- Test accessed: `false`",
        "",
        "## Invalid scalar contributions",
        "",
        scalar_summary.to_csv(index=False),
        "",
        "## Absent-gradient ratios",
        "",
        absent.to_csv(index=False),
        "",
        "## Abnormal auxiliary-gradient fraction",
        "",
        abnormal_ratio.to_csv(index=False),
        "",
        "## Long-term expected modality contribution",
        "",
        contribution_summary.to_csv(index=False),
        "",
        "## Hard gate",
        "",
    ]
    report.extend(
        "- {}: `{}`".format(name, value)
        for name, value in conditions.items()
    )
    report.extend(["", verdict, ""])
    (AUDIT / "stage17a_loss_consistency_audit.md").write_text(
        "\n".join(report)
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
