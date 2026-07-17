"""Stage 7B Conditional Modality Utility Gating training entrypoint."""
import argparse
import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    compatibility_for_modes, gated_kd_loss, locate_stage1_evaluator,
    modes_from_masks,
)
from trains.singleTask.conditional_modality_gate_utils import (
    MODALITIES, MODES, VARIANTS, ConditionalModalityGateWrapper,
    build_utility_groups, checkpoint_paths, first_epoch_counts_ok,
    flatten_mode_metrics, load_stage7a_derangements, matched_shuffle_loss,
    load_locked_compatibility, missing_macro_mae, objective, qualification_report,
    reliable_target_table,
    result_directory, summarize_gate_samples, targets_for_indices,
    utility_bce, utility_group_summary,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer, build_frozen_teacher, capture_rng_state,
    checkpoint_sha256, restore_rng_state, teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper, build_single_split_loader, clean_checkpoint_path,
    compute_full_dlf_loss, compute_task_loss, count_missing_modes,
    evaluate_all_modes, mode_to_mask, sample_missing_masks,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


BASE_COMMIT = "1e5ab2430307576127ad0845fde32b433a0da744"
REFERENCE = {
    "BestValidEpoch": 9, "J_valid": 0.6779637237389882,
    "J_test": 0.7178811430931091,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage7B CMUG.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--cmug-variant", choices=VARIANTS)
    parser.add_argument("--build-utility-groups-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[2])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    if args.seed != 1111:
        parser.error("Stage7B is locked to seed1111.")
    if args.num_workers != 0:
        parser.error("Stage7B fixes num_workers=0.")
    if not args.build_utility_groups_only and args.cmug_variant is None:
        parser.error("--cmug-variant is required for training.")
    if args.build_utility_groups_only and args.cmug_variant is not None:
        parser.error("Preflight does not accept a training variant.")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if args.smoke_test:
        args.max_epochs = min(args.max_epochs or 2, 2)
    return args


def build_config(cli):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = cli.seed
    args.device = assign_gpu(list(cli.gpu_ids))
    return args


def stage7a_directory(cli):
    return (
        Path(cli.result_root) / "analysis" / "modality_utility_v1"
        / cli.dataset / "seed{}".format(cli.seed)
    )


def create_logger(cli):
    Path(cli.log_dir).mkdir(parents=True, exist_ok=True)
    tag = "preflight" if cli.build_utility_groups_only else cli.cmug_variant
    kind = "smoke" if cli.smoke_test else "train"
    path = Path(cli.log_dir) / "DLF-{}-cmug-{}-{}-{}.log".format(
        cli.dataset, tag, kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cmug")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def write_preflight(cli, groups, report):
    directory = (
        Path(cli.result_root) / "missing_baseline" / "cmug_v1" / "preflight"
    )
    directory.mkdir(parents=True, exist_ok=True)
    groups.to_csv(directory / "{}_utility_groups.csv".format(cli.dataset), index=False)
    summary = utility_group_summary(groups)
    summary.to_csv(directory / "{}_utility_group_summary.csv".format(cli.dataset), index=False)
    payload = {
        "Dataset": cli.dataset, "Seed": cli.seed,
        "Stage7AArtifact": str(stage7a_directory(cli)),
        "TrainTargetsOnly": True, "ValidTargetsDiagnosticOnly": True,
        "TestAccessed": False, "Qualification": report,
        "QualifiedModalities": [modality for modality in MODALITIES if report[modality]["qualified"]],
    }
    (directory / "utility_preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Stage7B CMUG Utility Subgroup Preflight", "",
        "Only Stage7A train/valid raw artifacts were read; test was not accessed.", "",
    ]
    for modality in MODALITIES:
        values = report[modality]
        lines.extend([
            "## {}".format("Audio" if modality == "A" else "Vision"), "",
            "- train positive/negative/ambiguous: {}/{}/{}".format(
                values["train_positive"], values["train_negative"], values["train_ambiguous"]
            ),
            "- valid positive/negative/ambiguous: {}/{}/{}".format(
                values["valid_positive"], values["valid_negative"], values["valid_ambiguous"]
            ),
            "- qualified: **{}**".format(values["qualified"]), "",
        ])
    (directory / "stage7b_utility_preflight.md").write_text("\n".join(lines))
    return directory


def initialize_models(args, cli, qualified, loaders):
    checkpoint = clean_checkpoint_path(cli.model_save_dir, args.dataset_name, cli.seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    teacher = build_frozen_teacher(DLF, args, checkpoint)
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    student = ConditionalModalityGateWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2],
        qualified_modalities=qualified,
    ).to(args.device)
    # A temporary Stage3 wrapper provides an exact all-mode equivalence oracle.
    rng = capture_rng_state()
    try:
        reference_backbone = DLF(args).to(args.device)
        reference_backbone.load_state_dict(
            torch.load(checkpoint, map_location=args.device), strict=True
        )
        reference = MissingModalityWrapper(
            reference_backbone, args.feature_dims[1], args.feature_dims[2]
        ).to(args.device)
        reference.eval()
        student.eval()
        first = next(iter(loaders["valid"]))
        text, audio, vision, _ = batch_to_device(first, args.device)
        max_difference = 0.0
        with torch.no_grad():
            for mode in MODES:
                mask = mode_to_mask(mode, text.size(0), args.device, audio.dtype)
                expected = reference(text, audio, vision, mask)["output_logit"]
                actual = student(text, audio, vision, mask)["output_logit"]
                max_difference = max(
                    max_difference, float(torch.max(torch.abs(expected - actual)).cpu())
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del reference
    finally:
        restore_rng_state(rng)
    return teacher, student, checkpoint, checkpoint_sha256(checkpoint), max_difference


def batch_to_device(batch, device):
    return (
        batch["text"].to(device), batch["audio"].to(device),
        batch["vision"].to(device),
        batch["labels"]["M"].to(device).view(-1, 1),
    )


def group_lookup(groups):
    return {
        (str(row.Split), str(row.Modality), int(row.SampleIndex)): {
            "UtilityGroup": str(row.UtilityGroup),
            "UtilityTarget": (
                np.nan if row.UtilityGroup == "ambiguous" else float(row.UtilityTarget)
            ),
        }
        for row in groups.itertuples()
    }


def label_bin(value):
    if value < -1:
        return "[-3,-1)"
    if value < 0:
        return "[-1,0)"
    if value < 1:
        return "[0,1)"
    return "[1,3]"


def append_gate_records(rows, output, batch, labels, epoch, split, lookup):
    indices = batch["index"].view(-1).cpu().numpy().astype(int)
    identifiers = list(batch["id"])
    for modality in MODALITIES:
        q = output["utility_q_{}".format(modality)].detach().cpu().numpy()
        g = output["utility_g_{}".format(modality)].detach().cpu().numpy()
        for offset, index in enumerate(indices):
            binding = lookup[(split, modality, int(index))]
            value = float(labels[offset].detach().cpu())
            rows.append({
                "Epoch": epoch, "Split": split, "Modality": modality,
                "SampleIndex": int(index), "SampleID": str(identifiers[offset]),
                "Label": value, "LabelBin": label_bin(value),
                "UtilityGroup": binding["UtilityGroup"],
                "UtilityTarget": binding["UtilityTarget"],
                "q": float(q[offset]), "g": float(g[offset]),
                "SelectedBestValid": False,
            })


def collect_valid_gate_samples(model, loader, device, epoch, lookup):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            mask = mode_to_mask("LAV", labels.size(0), device, audio.dtype)
            output = model(text, audio, vision, mask)
            append_gate_records(rows, output, batch, labels, epoch, "valid", lookup)
    return rows


def prediction_rows(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            values = {}
            for mode in MODES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                values[mode] = model(text, audio, vision, mask)["output_logit"].view(-1).cpu().numpy()
            for offset, index in enumerate(batch["index"].view(-1).tolist()):
                rows.append({
                    "sample_index": int(index), "sample_id": str(batch["id"][offset]),
                    "label": float(labels[offset].detach().cpu()),
                    **{"{}_pred".format(mode): float(values[mode][offset]) for mode in MODES},
                })
    return pd.DataFrame(rows).sort_values("sample_index", kind="mergesort")


def dataset_modality(dataset, indices, modality, device):
    source = dataset.audio if modality == "A" else dataset.vision
    return torch.from_numpy(np.asarray(source[np.asarray(indices, dtype=np.int64)])).float().to(device)


def finalize_selected_gate_samples(model, loaders, device, epoch, lookup):
    rows = []
    model.eval()
    for split in ("train", "valid"):
        with torch.no_grad():
            for batch in loaders[split]:
                text, audio, vision, labels = batch_to_device(batch, device)
                lav = model(
                    text, audio, vision,
                    mode_to_mask("LAV", labels.size(0), device, audio.dtype),
                )
                l_only = model(
                    text, audio, vision,
                    mode_to_mask("L", labels.size(0), device, audio.dtype),
                )["output_logit"].view(-1)
                before = len(rows)
                append_gate_records(rows, lav, batch, labels, epoch, split, lookup)
                errors = torch.abs(l_only - labels.view(-1)).cpu().numpy()
                for position in range(before, len(rows)):
                    sample_offset = (position - before) % labels.size(0)
                    rows[position]["TextOnlyError"] = float(errors[sample_offset])
                    rows[position]["SelectedBestValid"] = True
    frame = pd.DataFrame(rows)
    edges = {}
    for modality in MODALITIES:
        train = frame[
            frame.Split.eq("train") & frame.Modality.eq(modality)
        ].drop_duplicates("SampleIndex")
        edge = np.quantile(train.TextOnlyError, [.25, .5, .75])
        edges[modality] = edge.tolist()
        local_mask = frame.Modality.eq(modality)
        frame.loc[local_mask, "TextErrorQuartile"] = [
            ("Q1_low", "Q2", "Q3", "Q4_high")[index]
            for index in np.searchsorted(edge, frame.loc[local_mask, "TextOnlyError"], side="right")
        ]
    return frame, edges


def conditional_gate_summary(selected):
    rows = []
    for group_type, column in (
        ("utility_group", "UtilityGroup"),
        ("label_bin", "LabelBin"),
        ("text_error_quartile", "TextErrorQuartile"),
    ):
        for keys, local in selected.groupby(
            ["Split", "Modality", column], sort=True
        ):
            rows.append({
                "GroupType": group_type, "Split": keys[0],
                "Modality": keys[1], "Group": keys[2],
                "Count": len(local), "MeanQ": float(local.q.mean()),
                "StdQ": float(local.q.std(ddof=0)),
                "MeanG": float(local.g.mean()),
            })
    return pd.DataFrame(rows)


def train_variant(cli, logger):
    setup_seed(cli.seed)
    args = build_config(cli)
    stage7a = stage7a_directory(cli)
    groups = build_utility_groups(stage7a)
    qualification = qualification_report(groups)
    qualified = [
        modality for modality in MODALITIES if qualification[modality]["qualified"]
    ]
    if not qualified:
        raise RuntimeError("Stable conditional utility subgroup not supported.")
    if cli.cmug_variant == "identity_replay":
        trainable_modalities = []
    else:
        trainable_modalities = qualified
    derangements = load_stage7a_derangements(stage7a)
    target_table = reliable_target_table(groups, "train")
    lookup = group_lookup(groups)
    evaluator, _, _ = locate_stage1_evaluator(
        cli.result_root, cli.dataset, cli.seed
    )
    cache_frame, cache_by_index = load_locked_compatibility(
        cli.result_root, cli.dataset, checkpoint_sha256(evaluator),
    )
    if len(cache_frame) != 1284:
        raise RuntimeError("Locked CFCompat train cache changed.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Training must construct train/valid loaders.")
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    teacher, student, init_checkpoint, init_sha, initial_max_diff = initialize_models(
        args, cli, trainable_modalities, loaders
    )
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=args.patience
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(cli.seed + 104729)
    version, main_checkpoint, diagnostic_checkpoint = checkpoint_paths(
        cli.model_save_dir, cli.cmug_variant, cli.dataset, cli.seed, cli.smoke_test
    )
    result_dir = result_directory(
        cli.result_root, cli.cmug_variant, cli.smoke_test
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    main_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(result_dir / "{}_utility_groups.csv".format(cli.dataset), index=False)
    best_valid_j = best_test_j = float("inf")
    best_valid_epoch = best_test_epoch = 0
    epoch_rows, gate_rows, matched_rows = [], [], []
    logger.info(
        "variant=%s qualified=%s trainable=%s initial_max_diff=%.12g",
        cli.cmug_variant, qualified, trainable_modalities, initial_max_diff,
    )
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        losses = {"full": [], "missing": [], "kd": [], "utility": [], "match": []}
        epoch_train_gate_rows = []
        match_data = {modality: {"matched": [], "shuffled": [], "loss": []} for modality in MODALITIES}
        mapping = derangements["train"][(epoch - 1) % 10]
        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            matched_rng = (
                capture_rng_state()
                if cli.cmug_variant == "utility_gate_matched" else None
            )
            full_output = student(text, audio, vision, full_mask)
            full_loss, _ = compute_full_dlf_loss(
                full_output, labels, criterion, cosine, hinge
            )
            missing_mask = sample_missing_masks(
                labels.size(0), missing_generator, args.device, audio.dtype
            )
            modes = modes_from_masks(missing_mask)
            counts.update(count_missing_modes(missing_mask))
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
            compatibility = compatibility_for_modes(
                cache_by_index, indices, modes, args.device, labels.dtype
            )
            kd_loss, _ = gated_kd_loss(
                missing_output["output_logit"], teacher_prediction, compatibility
            )
            utility_terms = []
            targets = {}
            for modality in MODALITIES:
                targets[modality] = targets_for_indices(
                    target_table, indices, modality, args.device, labels.dtype
                )
                if modality in trainable_modalities:
                    probability = full_output["utility_q_{}".format(modality)]
                    term, _ = utility_bce(
                        probability, targets[modality],
                        torch.ones_like(targets[modality], dtype=torch.bool),
                    )
                    utility_terms.append(term)
            utility_loss = (
                torch.stack(utility_terms).mean()
                if utility_terms else full_output["output_logit"].sum() * 0.0
            )
            matched_terms = []
            if cli.cmug_variant == "utility_gate_matched":
                mapped = mapping[np.asarray(indices, dtype=np.int64)]
                for modality in trainable_modalities:
                    positive = targets[modality].eq(1)
                    if not positive.any():
                        continue
                    shuffled_audio, shuffled_vision = audio, vision
                    if modality == "A":
                        shuffled_audio = dataset_modality(
                            loaders["train"].dataset, mapped, "A", args.device
                        )
                    else:
                        shuffled_vision = dataset_modality(
                            loaders["train"].dataset, mapped, "V", args.device
                        )
                    resume_rng = capture_rng_state()
                    try:
                        # Reuse the matched forward's dropout masks and do not
                        # perturb Stage3's training RNG stream.
                        restore_rng_state(matched_rng)
                        shuffled_output = student(
                            text, shuffled_audio, shuffled_vision, full_mask
                        )
                    finally:
                        restore_rng_state(resume_rng)
                    term, matched_error, shuffled_error, _ = matched_shuffle_loss(
                        full_output["output_logit"], shuffled_output["output_logit"],
                        labels, positive,
                    )
                    matched_terms.append(term)
                    match_data[modality]["matched"].extend(
                        matched_error[positive].detach().cpu().numpy().tolist()
                    )
                    match_data[modality]["shuffled"].extend(
                        shuffled_error[positive].detach().cpu().numpy().tolist()
                    )
                    match_data[modality]["loss"].append(float(term.detach()))
            match_loss = (
                torch.stack(matched_terms).mean()
                if matched_terms else full_output["output_logit"].sum() * 0.0
            )
            total_loss = full_loss + missing_loss + kd_loss + utility_loss + match_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN/Inf in CMUG loss.")
            total_loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Full Teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()
            append_gate_records(
                epoch_train_gate_rows, full_output, batch, labels,
                epoch, "train", lookup,
            )
            for key, value in (
                ("full", full_loss), ("missing", missing_loss), ("kd", kd_loss),
                ("utility", utility_loss), ("match", match_loss),
            ):
                losses[key].append(float(value.detach()))
        if epoch == 1 and not first_epoch_counts_ok(counts):
            raise RuntimeError("Epoch1 missing counts must be LA=435 LV=430 L=419.")
        gate_rows.extend(epoch_train_gate_rows)
        gate_rows.extend(
            collect_valid_gate_samples(
                student, loaders["valid"], args.device, epoch, lookup
            )
        )
        valid = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        test = evaluate_all_modes(
            student, test_loader, args.device, "moddrop", criterion
        )
        j_valid, j_test = objective(valid), objective(test)
        if not math.isfinite(j_valid) or not math.isfinite(j_test):
            raise FloatingPointError("Non-finite CMUG metric.")
        scheduler.step(j_valid)
        is_best_valid = j_valid <= best_valid_j - 1e-6
        is_best_test = j_test <= best_test_j - 1e-6
        if is_best_valid:
            best_valid_j, best_valid_epoch = j_valid, epoch
            torch.save(student.state_dict(), main_checkpoint)
        if is_best_test:
            best_test_j, best_test_epoch = j_test, epoch
            torch.save(student.state_dict(), diagnostic_checkpoint)
        for modality in MODALITIES:
            matched = np.asarray(match_data[modality]["matched"], dtype=float)
            shuffled = np.asarray(match_data[modality]["shuffled"], dtype=float)
            matched_rows.append({
                "Epoch": epoch, "Modality": modality,
                "PositiveCount": len(matched),
                "MatchedErrorMean": float(matched.mean()) if len(matched) else np.nan,
                "ShuffledErrorMean": float(shuffled.mean()) if len(shuffled) else np.nan,
                "FractionMatchedBetter": float((matched < shuffled).mean()) if len(matched) else np.nan,
                "L_match": float(np.mean(match_data[modality]["loss"]))
                if match_data[modality]["loss"] else 0.0,
                "ShuffleID": (epoch - 1) % 10,
            })
        epoch_rows.append({
            "Seed": cli.seed, "Epoch": epoch, "Variant": cli.cmug_variant,
            "J_valid": j_valid, "J_test": j_test,
            "IsBestValid": is_best_valid, "IsBestTestDiagnostic": is_best_test,
            "LA_count": counts["LA"], "LV_count": counts["LV"], "L_count": counts["L"],
            **{"{}_loss".format(key): float(np.mean(value)) for key, value in losses.items()},
            "valid_MissingMacro_MAE": missing_macro_mae(valid),
            "test_MissingMacro_MAE": missing_macro_mae(test),
            **flatten_mode_metrics(valid, "valid"),
            **flatten_mode_metrics(test, "test"),
        })
        logger.info(
            "epoch=%d LA=%d LV=%d L=%d J_valid=%.6f J_test=%.6f utility=%.6f match=%.6f",
            epoch, counts["LA"], counts["LV"], counts["L"], j_valid, j_test,
            epoch_rows[-1]["utility_loss"], epoch_rows[-1]["match_loss"],
        )
        if epoch - best_valid_epoch >= args.early_stop:
            break
    if not main_checkpoint.is_file() or not diagnostic_checkpoint.is_file():
        raise RuntimeError("CMUG checkpoints are incomplete.")
    student.load_state_dict(
        torch.load(main_checkpoint, map_location=args.device), strict=True
    )
    final_valid = evaluate_all_modes(
        student, loaders["valid"], args.device, "moddrop", criterion
    )
    final_test = evaluate_all_modes(
        student, test_loader, args.device, "moddrop", criterion
    )
    selected_gate, text_edges = finalize_selected_gate_samples(
        student, loaders, args.device, best_valid_epoch, lookup
    )
    selected_gate.to_csv(
        result_dir / "{}_gate_samples_best_valid.csv".format(cli.dataset), index=False
    )
    conditional_gate_summary(selected_gate).to_csv(
        result_dir / "{}_gate_conditional_summary.csv".format(cli.dataset), index=False
    )
    gate_frame = pd.DataFrame(gate_rows)
    gate_frame["SelectedBestValid"] = gate_frame.Epoch.eq(best_valid_epoch)
    gate_frame[gate_frame.Split.eq("train")].to_csv(
        result_dir / "{}_gate_samples_train.csv".format(cli.dataset), index=False
    )
    gate_frame[gate_frame.Split.eq("valid")].to_csv(
        result_dir / "{}_gate_samples_valid.csv".format(cli.dataset), index=False
    )
    summarize_gate_samples(gate_frame).to_csv(
        result_dir / "{}_gate_summary.csv".format(cli.dataset), index=False
    )
    pd.DataFrame(matched_rows).to_csv(
        result_dir / "{}_matched_shuffle_summary.csv".format(cli.dataset), index=False
    )
    pd.DataFrame(epoch_rows).to_csv(
        result_dir / "{}_epoch_metrics.csv".format(cli.dataset), index=False
    )
    predictions = prediction_rows(student, loaders["valid"], args.device)
    predictions["selected_by"] = "validation_J"
    predictions["student_only"] = True
    predictions.to_csv(
        result_dir / "{}_best_valid_predictions.csv".format(cli.dataset), index=False
    )
    row = {
        "Seed": cli.seed, "Variant": cli.cmug_variant,
        "BestValidEpoch": best_valid_epoch,
        "J_valid": objective(final_valid),
        "J_test_at_valid_best": objective(final_test),
        "BestObservedTestEpoch": best_test_epoch,
        "BestObservedTestJ": best_test_j,
        "MainCheckpoint": str(main_checkpoint),
        "DiagnosticCheckpoint": str(diagnostic_checkpoint),
        "SelectedBy": "validation_J", "StudentOnlyEval": True,
        "TeacherCheckpoint": str(init_checkpoint),
        "TeacherSHA256": init_sha, "StudentInitSHA256": init_sha,
        "InitialMaxAbsDifference": initial_max_diff,
        "QualifiedModalities": ",".join(qualified),
        "TrainableGateModalities": ",".join(trainable_modalities),
        "UtilityTargetsTrainOnly": True, "TestUsedForUtilityTargets": False,
        "TextErrorTrainQuartileEdges": json.dumps(text_edges, sort_keys=True),
        "valid_MissingMacro_MAE": missing_macro_mae(final_valid),
        "test_at_valid_best_MissingMacro_MAE": missing_macro_mae(final_test),
        **flatten_mode_metrics(final_valid, "valid"),
        **flatten_mode_metrics(final_test, "test_at_valid_best"),
    }
    pd.DataFrame([row]).to_csv(
        result_dir / "{}_per_seed.csv".format(cli.dataset), index=False
    )
    config = {
        "variant": cli.cmug_variant, "version": version,
        "base_commit": BASE_COMMIT, "seed": cli.seed,
        "qualified_modalities": qualified,
        "trainable_gate_modalities": trainable_modalities,
        "loss_weights": {
            "full": 1.0, "missing": 1.0, "CFKD": 1.0,
            "utility": 0.0 if cli.cmug_variant == "identity_replay" else 1.0,
            "matched": 1.0 if cli.cmug_variant == "utility_gate_matched" else 0.0,
        },
        "utility_targets": "train reliable positive/negative only",
        "valid_targets": "diagnostic only", "test_targets": False,
        "shuffle_rule": "(epoch-1) mod 10",
        "checkpoint_selection": "validation J only",
        "initial_max_abs_difference": initial_max_diff,
    }
    (result_dir / "audit_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return row


def main():
    cli = parse_args()
    logger, log_path = create_logger(cli)
    groups = build_utility_groups(stage7a_directory(cli))
    qualification = qualification_report(groups)
    preflight = write_preflight(cli, groups, qualification)
    logger.info("preflight=%s qualification=%s", preflight, qualification)
    if not any(values["qualified"] for values in qualification.values()):
        logger.error("Stable conditional utility subgroup not supported.")
        return
    if cli.build_utility_groups_only:
        logger.info("utility subgroup preflight complete; no model or test loader created.")
        return
    result = train_variant(cli, logger)
    logger.info("complete result=%s log=%s", result, log_path)


if __name__ == "__main__":
    main()
