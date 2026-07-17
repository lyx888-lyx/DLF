"""Test-once coordinator, PE5 builder, and label-free ADPEP-All projector."""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from scripts.mosei.stage10_common import (
    SEEDS,
    STAGES,
    append_jsonl,
    atomic_json,
    git_head,
    load_json,
    sha256,
    stage_directory,
    stage_manifest_path,
    utc_now,
    validate_sentinel,
    write_state,
)
from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
    retention_ratio,
    select_anchor_seed,
)
from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    METRICS,
    aligned_prediction_mean,
    metric_rows,
    metrics_from_predictions,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    build_single_split_loader,
    mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed

MODES = ("LAV", "LA", "LV", "L")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def _model_args(config, seed):
    args = get_config_regression("DLF", "mosei", config["ConfigFile"])
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False
    args.train_mode = "regression"
    args.seed = args.cur_seed = int(seed)
    args.device = assign_gpu([0])
    return args


def _load_model(config, method, seed):
    args = _model_args(config, seed)
    manifest = load_json(
        stage_manifest_path(config["ResultRoot"], method, seed)
    )
    checkpoint = Path(manifest["Checkpoint"])
    if sha256(checkpoint) != manifest["CheckpointSHA256"]:
        raise RuntimeError("Locked-test checkpoint SHA mismatch.")
    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    model.eval()
    return args, model, manifest


def _predict(model, loader, device, split, method, seed):
    rows = []
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["labels"]["M"].view(-1).cpu().numpy()
            predictions = {}
            for mode in MODES:
                mask = mode_to_mask(
                    mode, len(labels), device=device, dtype=audio.dtype
                )
                predictions[mode] = (
                    model(text, audio, vision, mask)["output_logit"]
                    .view(-1)
                    .cpu()
                    .numpy()
                )
            indices = batch["index"].view(-1).cpu().numpy()
            ids = list(batch["id"])
            for offset, index in enumerate(indices):
                rows.append(
                    {
                        "sample_index": int(index),
                        "sample_id": str(ids[offset]),
                        "label": float(labels[offset]),
                        **{
                            "{}_pred".format(mode): float(
                                predictions[mode][offset]
                            )
                            for mode in MODES
                        },
                        "Split": split,
                        "Method": method,
                        "Seed": int(seed),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "sample_index", kind="mergesort"
    ).reset_index(drop=True)


def _metric_frame(predictions, method):
    rows = []
    for (seed, split), frame in predictions.items():
        rows.extend(metric_rows(frame, method, seed))
    return pd.DataFrame(rows)


def _paired_rows(moddrop_metrics, cfcompat_metrics):
    keys = ["Seed", "Split", "Mode"]
    merged = moddrop_metrics.merge(
        cfcompat_metrics, on=keys, suffixes=("_ModDrop", "_CFCompat")
    )
    for metric in ("J",) + METRICS:
        merged["Delta_{}".format(metric)] = (
            merged["{}_CFCompat".format(metric)]
            - merged["{}_ModDrop".format(metric)]
        )
    return merged.sort_values(keys)


def _aggregate_paired(paired):
    rows = []
    for (split, mode), group in paired.groupby(["Split", "Mode"], sort=False):
        row = {"Split": split, "Mode": mode, "Seeds": len(group)}
        for metric in ("J",) + METRICS:
            values = group["Delta_{}".format(metric)]
            row["MeanDelta_{}".format(metric)] = float(values.mean())
            row["ImprovedSeeds_{}".format(metric)] = int(
                (values < 0).sum()
                if metric in ("J", "MAE", "Loss")
                else (values > 0).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _projection_frame(anchor, pe5, split, anchor_seed):
    for identity in ("sample_index", "sample_id", "label"):
        if not np.array_equal(
            anchor[identity].astype(str).to_numpy(),
            pe5[identity].astype(str).to_numpy(),
        ):
            raise RuntimeError("Anchor/PE5 identity binding differs.")
    output = pe5[
        ["sample_index", "sample_id", "label"]
    ].copy()
    fallback_rows = []
    verification = []
    for mode in MODES:
        anchor_values = anchor["{}_pred".format(mode)].to_numpy(np.float32)
        pe5_values = pe5["{}_pred".format(mode)].to_numpy(np.float32)
        projected, details = project_array(
            anchor_values, pe5_values, "mosei", "adpep_all"
        )
        output["{}_pred".format(mode)] = projected
        anchor_decisions = evaluator_decisions(anchor_values, "mosei")
        projected_decisions = evaluator_decisions(projected, "mosei")
        mismatches = [
            int(np.count_nonzero(left != right))
            for left, right in zip(anchor_decisions, projected_decisions)
        ]
        verification.append(
            {
                "Split": split,
                "Mode": mode,
                "Acc7MismatchCount": mismatches[0],
                "Acc5MismatchCount": mismatches[1],
                "Acc2MismatchCount": mismatches[2],
                "Passed": not any(mismatches),
            }
        )
        for index, detail in enumerate(details):
            if detail.fallback_to_anchor:
                fallback_rows.append(
                    {
                        "Split": split,
                        "Mode": mode,
                        "sample_index": int(output.sample_index.iloc[index]),
                        "sample_id": str(output.sample_id.iloc[index]),
                        "Reason": detail.fallback_reason,
                    }
                )
    output["Split"] = split
    output["Method"] = "ADPEP-All"
    output["AnchorSeed"] = int(anchor_seed)
    return output, fallback_rows, verification


def _retention(anchor_rows, pe5_rows, adpep_rows):
    anchor = anchor_rows.set_index(["Split", "Mode"])
    pe5 = pe5_rows.set_index(["Split", "Mode"])
    adpep = adpep_rows.set_index(["Split", "Mode"])
    rows = []
    quantities = (
        ("J", "LAV", "J", True),
        ("LAV_MAE", "LAV", "MAE", True),
        ("MissingMacro_MAE", "MissingMacro", "MAE", True),
        ("LAV_Corr", "LAV", "Corr", False),
        ("MissingMacro_Corr", "MissingMacro", "Corr", False),
    )
    for split in ("valid", "test"):
        for quantity, mode, metric, lower in quantities:
            gain, retained, ratio = retention_ratio(
                anchor.loc[(split, mode), metric],
                pe5.loc[(split, mode), metric],
                adpep.loc[(split, mode), metric],
                lower,
            )
            rows.append(
                {
                    "Split": split,
                    "Quantity": quantity,
                    "PE5Gain": gain,
                    "ADPEPGain": retained,
                    "RetentionRatio": ratio,
                }
            )
    return pd.DataFrame(rows)


def main():
    cli = parse_args()
    run = Path(cli.run_dir).resolve()
    config = load_json(run / "config.json")
    result = Path(config["ResultRoot"])
    final = result / "final"
    locked = result / "locked_test"
    pe5_dir = result / "pe5"
    adpep_dir = result / "adpep"
    for path in (final, locked, pe5_dir, adpep_dir):
        path.mkdir(parents=True, exist_ok=True)
    if git_head(ROOT) != config["LaunchCommit"]:
        raise RuntimeError("Coordinator checkout moved after launch.")
    manifests = []
    for seed in SEEDS:
        for stage in STAGES:
            sentinel = validate_sentinel(
                run / "sentinels" / "seed_{}".format(seed) / "{}.done".format(stage),
                result,
                seed,
                stage,
                config["LaunchCommit"],
            )
            if sentinel is None:
                raise RuntimeError(
                    "Missing completion sentinel for seed={} stage={}.".format(
                        seed, stage
                    )
                )
            manifest = load_json(stage_manifest_path(result, stage, seed))
            if not manifest["NoTestAccess"] or manifest["TestLoaderConstructed"]:
                raise RuntimeError("Worker recorded forbidden test access.")
            manifests.append(manifest)
    if not config["TestsPassed"]["Passed"]:
        raise RuntimeError("Launch tests gate is not passed.")
    unlock = {
        "UnlockedAt": utc_now(),
        "LaunchCommit": config["LaunchCommit"],
        "Seeds": list(SEEDS),
        "PE5MembersFrozen": list(SEEDS),
        "AnchorRule": "argmin validation J, lower seed on ties",
        "AllValidationBest": all(row["ValidationSelected"] for row in manifests),
        "NoWorkerTestAccess": True,
        "TestsPassed": True,
        "TestOnce": True,
        "ValidatedStageManifestSHA256": [
            sha256(stage_manifest_path(result, row["Stage"], row["Seed"]))
            for row in manifests
        ],
    }
    unlock_path = result / "TEST_UNLOCK_MANIFEST.json"
    atomic_json(unlock_path, unlock)
    write_state(run, "RUNNING_LOCKED_TEST")

    all_predictions = {"moddrop": {}, "cfcompat": {}}
    prediction_records = []
    for method in ("moddrop", "cfcompat"):
        display = "ModDrop" if method == "moddrop" else "CFCompatKD"
        for seed in SEEDS:
            setup_seed(seed)
            args, model, manifest = _load_model(config, method, seed)
            for split in ("valid", "test"):
                # The first construction of a test loader occurs only after unlock.
                loader = build_single_split_loader(args, split, 1)
                frame = _predict(
                    model, loader, args.device, split, display, seed
                )
                path = locked / "{}_seed{}_{}.csv".format(method, seed, split)
                frame.to_csv(path, index=False, float_format="%.9g")
                all_predictions[method][(seed, split)] = frame
                prediction_records.append(
                    {
                        "Method": display,
                        "Seed": seed,
                        "Split": split,
                        "Path": str(path),
                        "SHA256": sha256(path),
                        "Rows": len(frame),
                        "CheckpointSHA256": manifest["CheckpointSHA256"],
                    }
                )
            del model
            torch.cuda.empty_cache()
    atomic_json(
        locked / "locked_test_prediction_manifest.json",
        {
            "UnlockManifest": str(unlock_path),
            "UnlockManifestSHA256": sha256(unlock_path),
            "Predictions": prediction_records,
            "NoCheckpointOrMethodSelectionFromTest": True,
        },
    )
    moddrop_metrics = _metric_frame(all_predictions["moddrop"], "ModDrop")
    cfcompat_metrics = _metric_frame(all_predictions["cfcompat"], "CFCompatKD")
    moddrop_metrics.to_csv(
        final / "mosei_moddrop_per_seed.csv", index=False
    )
    cfcompat_metrics.to_csv(
        final / "mosei_cfcompat_per_seed.csv", index=False
    )
    paired = _paired_rows(moddrop_metrics, cfcompat_metrics)
    paired.to_csv(
        final / "mosei_paired_cfcompat_vs_moddrop.csv", index=False
    )
    aggregate = _aggregate_paired(paired)
    aggregate.to_csv(
        final / "mosei_cfcompat_aggregate.csv", index=False
    )
    atomic_json(
        final / "mosei_clean_dlf_manifest.json",
        {
            "Seeds": list(SEEDS),
            "Members": [
                load_json(stage_manifest_path(result, "clean", seed))
                for seed in SEEDS
            ],
        },
    )

    write_state(run, "BUILDING_PE5")
    pe5_frames = {}
    pe5_metric_rows = []
    for split in ("valid", "test"):
        member_frames = [
            all_predictions["cfcompat"][(seed, split)] for seed in SEEDS
        ]
        frame = aligned_prediction_mean(
            member_frames, split, "CFCompatKD-PE5"
        )
        frame.to_csv(
            pe5_dir / "mosei_pe5_predictions_{}.csv".format(split),
            index=False,
            float_format="%.9g",
        )
        pe5_frames[split] = frame
        pe5_metric_rows.extend(metric_rows(frame, "PE5"))
    pe5_metrics = pd.DataFrame(pe5_metric_rows)
    pe5_metrics.to_csv(final / "mosei_pe5_metrics.csv", index=False)

    valid_j = (
        cfcompat_metrics.loc[
            (cfcompat_metrics.Split == "valid")
            & (cfcompat_metrics.Mode == "LAV"),
            ["Seed", "J"],
        ]
        .sort_values("Seed")
        .to_dict("records")
    )
    anchor_seed = select_anchor_seed(valid_j, SEEDS)
    anchor_selection = {
        "Rule": "argmin_fixed_seeds_validation_J_then_lower_seed",
        "FixedSeeds": list(SEEDS),
        "AnchorSeed": anchor_seed,
        "ValidationJBySeed": {
            str(int(row["Seed"])): float(row["J"]) for row in valid_j
        },
        "SelectedWithoutTest": True,
        "HardCodedAnchorSeed": False,
    }
    atomic_json(final / "mosei_anchor_selection.json", anchor_selection)

    write_state(run, "BUILDING_ADPEP")
    adpep_frames, fallback_rows, verification_rows = {}, [], []
    adpep_metric_rows, anchor_metric_rows = [], []
    for split in ("valid", "test"):
        anchor = all_predictions["cfcompat"][(anchor_seed, split)]
        frame, fallbacks, verification = _projection_frame(
            anchor, pe5_frames[split], split, anchor_seed
        )
        label_free = frame.drop(columns=["label"])
        label_free_path = adpep_dir / "mosei_adpep_all_predictions_{}.csv".format(split)
        label_free.to_csv(label_free_path, index=False, float_format="%.9g")
        frozen_sha = sha256(label_free_path)
        # Labels are joined only after the projected predictions are frozen.
        evaluated = label_free.merge(
            anchor[["sample_index", "sample_id", "label"]],
            on=["sample_index", "sample_id"],
            validate="one_to_one",
        )
        adpep_frames[split] = evaluated
        adpep_metric_rows.extend(metric_rows(evaluated, "ADPEP-All"))
        anchor_metric_rows.extend(metric_rows(anchor, "Anchor", anchor_seed))
        for row in verification:
            row["FrozenPredictionSHA256"] = frozen_sha
        fallback_rows.extend(fallbacks)
        verification_rows.extend(verification)
    adpep_metrics = pd.DataFrame(adpep_metric_rows)
    anchor_metrics = pd.DataFrame(anchor_metric_rows)
    adpep_metrics.to_csv(final / "mosei_adpep_metrics.csv", index=False)
    verification = pd.DataFrame(verification_rows)
    # Classification metrics must also be exactly inherited from Anchor.
    for split in ("valid", "test"):
        for mode in MODES:
            left = anchor_metrics.loc[
                (anchor_metrics.Split == split) & (anchor_metrics.Mode == mode)
            ].iloc[0]
            right = adpep_metrics.loc[
                (adpep_metrics.Split == split) & (adpep_metrics.Mode == mode)
            ].iloc[0]
            if any(
                abs(float(left[metric]) - float(right[metric])) > 1e-12
                for metric in ("acc_7", "acc_5", "acc_2", "F1_score")
            ):
                raise RuntimeError("ADPEP classification inheritance failed.")
    if not verification.Passed.all():
        raise RuntimeError("ADPEP decision preservation failed.")
    verification.to_csv(
        final / "mosei_adpep_decision_verification.csv", index=False
    )
    pd.DataFrame(
        fallback_rows,
        columns=["Split", "Mode", "sample_index", "sample_id", "Reason"],
    ).to_csv(final / "mosei_projection_fallback_samples.csv", index=False)
    retention = _retention(anchor_metrics, pe5_metrics, adpep_metrics)
    retention.to_csv(final / "mosei_adpep_retention.csv", index=False)

    write_state(run, "AGGREGATING")
    test_paired = aggregate.loc[aggregate.Split == "test"]
    mean_delta_j = float(
        test_paired.loc[test_paired.Mode == "LAV", "MeanDelta_J"].iloc[0]
    )
    improved_j = int(
        test_paired.loc[
            test_paired.Mode == "LAV", "ImprovedSeeds_J"
        ].iloc[0]
    )
    test_lav = test_paired.loc[test_paired.Mode == "LAV"].iloc[0]
    test_missing = test_paired.loc[
        test_paired.Mode == "MissingMacro"
    ].iloc[0]
    cf_supported = (
        mean_delta_j < 0
        and improved_j >= 4
        and float(test_lav.MeanDelta_MAE) < 0
        and float(test_missing.MeanDelta_MAE) < 0
        and float(test_lav.MeanDelta_Corr) >= 0
        and float(test_missing.MeanDelta_Corr) >= 0
    )
    cf_class = (
        "SUPPORTED"
        if cf_supported
        else "PARTIAL"
        if mean_delta_j < 0
        else "UNSUPPORTED"
    )
    indexed_anchor = anchor_metrics.set_index(["Split", "Mode"])
    indexed_adpep = adpep_metrics.set_index(["Split", "Mode"])
    required_retention = retention.loc[
        (retention.Split == "test")
        & retention.Quantity.isin(
            ["J", "LAV_MAE", "MissingMacro_MAE"]
        )
    ].RetentionRatio
    regression_better = all(
        float(indexed_adpep.loc[("test", mode), metric])
        < float(indexed_anchor.loc[("test", mode), metric])
        for mode, metric in (
            ("LAV", "J"),
            ("LAV", "MAE"),
            ("MissingMacro", "MAE"),
        )
    )
    corr_ok = all(
        float(indexed_adpep.loc[("test", mode), "Corr"])
        >= float(indexed_anchor.loc[("test", mode), "Corr"])
        for mode in ("LAV", "MissingMacro")
    )
    retention_ok = (
        len(required_retention) == 3
        and required_retention.notna().all()
        and (required_retention >= 0.50).all()
    )
    if regression_better and corr_ok and retention_ok:
        adpep_class = "FULL GENERALIZATION"
    elif regression_better:
        adpep_class = "PARTIAL GENERALIZATION"
    elif all(
        abs(
            float(indexed_adpep.loc[("test", mode), metric])
            - float(indexed_anchor.loc[("test", mode), metric])
        )
        <= 1e-8
        for mode, metric in (
            ("LAV", "J"),
            ("LAV", "MAE"),
            ("MissingMacro", "MAE"),
        )
    ):
        adpep_class = "CLASSIFICATION-ONLY"
    else:
        adpep_class = "UNSUPPORTED"
    audit = [
        "# Stage 10 MOSEI Generalization Final Audit",
        "",
        "- Frozen method commit: `{}`".format(config["FrozenMethodCommit"]),
        "- Launch commit: `{}`".format(config["LaunchCommit"]),
        "- Official seeds: {}".format(", ".join(map(str, SEEDS))),
        "- Test unlock passed before any test loader: yes",
        "- Worker test access: none",
        "- Anchor selected from validation J only: seed {}".format(anchor_seed),
        "- ADPEP-All Acc7/Acc5/Acc2/F1 inheritance: exact",
        "- CFCompatKD generalization classification: **{}**".format(cf_class),
        "- ADPEP-All generalization classification: **{}**".format(
            adpep_class
        ),
        "- Mean paired test delta J: {:.9f}".format(mean_delta_j),
        "- Test seeds improving J: {}/5".format(improved_j),
        "",
        "All metrics were computed from frozen per-sample predictions. No test "
        "result changed a checkpoint, member, anchor, projection rule, or seed.",
    ]
    (final / "stage10_mosei_generalization_final_audit.md").write_text(
        "\n".join(audit) + "\n"
    )
    write_state(
        run,
        "COMPLETED",
        Error=None,
        FinalAudit=str(
            final / "stage10_mosei_generalization_final_audit.md"
        ),
    )


if __name__ == "__main__":
    arguments = parse_args()
    try:
        main()
    except Exception as error:
        run_path = Path(arguments.run_dir)
        append_jsonl(
            run_path / "failures.jsonl",
            {
                "Component": "coordinator",
                "Error": repr(error),
                "Traceback": traceback.format_exc(),
                "At": utc_now(),
            },
        )
        write_state(run_path, "FAILED", Error=repr(error))
        raise
