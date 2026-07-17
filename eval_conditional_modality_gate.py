"""Student-only Stage7B CMUG validation utility audit and final report."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.conditional_modality_gate_utils import (
    MODALITIES, MODES, ConditionalModalityGateWrapper, build_utility_groups,
    checkpoint_paths, load_stage7a_derangements, qualification_report,
    result_directory, safe_auroc,
)
from trains.singleTask.missing_utils import mode_to_mask
from trains.singleTask.model.DLF import DLF
from trains.singleTask.modality_utility_utils import (
    bootstrap_summary, classify_utility, effective_rank, infer_padding_mask,
    masked_input_sensitivity, stable_seed,
)
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Stage7B student-only evaluation.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--cmug-variant",
        choices=("identity_replay", "utility_gate", "utility_gate_matched"),
    )
    parser.add_argument("--final-report", action="store_true")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[2])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    if args.seed != 1111 or args.num_workers != 0:
        parser.error("Stage7B eval fixes seed1111 and num_workers=0.")
    if args.final_report == (args.cmug_variant is not None):
        parser.error("Choose exactly one of --cmug-variant or --final-report.")
    return args


def build_config(cli):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False
    args.train_mode = "regression"
    args.seed = args.cur_seed = cli.seed
    args.device = assign_gpu(list(cli.gpu_ids))
    return args


def stage7a_directory(cli):
    return (
        Path(cli.result_root) / "analysis" / "modality_utility_v1"
        / cli.dataset / "seed{}".format(cli.seed)
    )


def initialize_student(cli, args):
    groups = build_utility_groups(stage7a_directory(cli))
    qualification = qualification_report(groups)
    qualified = [
        modality for modality in MODALITIES if qualification[modality]["qualified"]
    ]
    gates = [] if cli.cmug_variant == "identity_replay" else qualified
    backbone = DLF(args).to(args.device)
    checkpoint = Path(cli.model_save_dir) / "DLF_{}_seed{}_best.pth".format(
        cli.dataset, cli.seed
    )
    backbone.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    model = ConditionalModalityGateWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2], gates
    ).to(args.device)
    _, selected, _ = checkpoint_paths(
        cli.model_save_dir, cli.cmug_variant, cli.dataset, cli.seed, False
    )
    model.load_state_dict(torch.load(selected, map_location=args.device), strict=True)
    model.eval()
    return model


def valid_audit(cli):
    setup_seed(cli.seed)
    args = build_config(cli)
    dataset = MMDataset(args, mode="valid")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=0,
        shuffle=False, drop_last=False,
    )
    model = initialize_student(cli, args)
    mappings = load_stage7a_derangements(stage7a_directory(cli))["valid"]
    predictions = {mode: [] for mode in MODES}
    representations = {mode: [] for mode in MODES}
    labels_all, sensitivity_a, sensitivity_v = [], [], []
    shuffle_a, shuffle_v = [], []
    captured = []
    capture_enabled = [True]

    def hook(module, inputs):
        if capture_enabled[0]:
            captured.append(inputs[0].detach().cpu().numpy())

    handle = model.backbone.proj1.register_forward_pre_hook(hook)
    for batch in loader:
        text = batch["text"].to(args.device)
        audio = batch["audio"].to(args.device)
        vision = batch["vision"].to(args.device)
        labels = batch["labels"]["M"].to(args.device).view(-1)
        indices = batch["index"].view(-1).cpu().numpy().astype(int)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        for mode in MODES:
            before = len(captured)
            mask = mode_to_mask(mode, labels.size(0), args.device, audio.dtype)
            if mode == "LAV":
                audio_grad = audio.detach().clone().requires_grad_(True)
                vision_grad = vision.detach().clone().requires_grad_(True)
                output = model(text, audio_grad, vision_grad, mask)
                grad_a, grad_v = torch.autograd.grad(
                    output["output_logit"].sum(), (audio_grad, vision_grad)
                )
                _, local_a = masked_input_sensitivity(
                    audio_grad, grad_a, infer_padding_mask(audio_grad)
                )
                _, local_v = masked_input_sensitivity(
                    vision_grad, grad_v, infer_padding_mask(vision_grad)
                )
                sensitivity_a.extend(local_a.tolist())
                sensitivity_v.extend(local_v.tolist())
            else:
                with torch.no_grad():
                    output = model(text, audio, vision, mask)
            if len(captured) != before + 1:
                raise RuntimeError("Shared representation hook mismatch.")
            predictions[mode].extend(
                output["output_logit"].detach().view(-1).cpu().numpy().tolist()
            )
            representations[mode].append(captured[-1])
        capture_enabled[0] = False
        local_a, local_v = [], []
        for repeat in range(10):
            mapped = mappings[repeat][indices]
            shuffled_audio = torch.from_numpy(
                np.asarray(dataset.audio[mapped])
            ).float().to(args.device)
            shuffled_vision = torch.from_numpy(
                np.asarray(dataset.vision[mapped])
            ).float().to(args.device)
            mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            with torch.no_grad():
                local_a.append(
                    model(text, shuffled_audio, vision, mask)["output_logit"]
                    .view(-1).cpu().numpy()
                )
                local_v.append(
                    model(text, audio, shuffled_vision, mask)["output_logit"]
                    .view(-1).cpu().numpy()
                )
        shuffle_a.extend(np.mean(np.stack(local_a), axis=0).tolist())
        shuffle_v.extend(np.mean(np.stack(local_v), axis=0).tolist())
        capture_enabled[0] = True
    handle.remove()
    labels = np.asarray(labels_all)
    pred = {key: np.asarray(value) for key, value in predictions.items()}
    reps = {key: np.concatenate(value, axis=0) for key, value in representations.items()}
    rows = []
    for modality, with_mode, shuffled, sensitivity in (
        ("A", "LA", np.asarray(shuffle_a), np.asarray(sensitivity_a)),
        ("V", "LV", np.asarray(shuffle_v), np.asarray(sensitivity_v)),
    ):
        gain = np.abs(pred["L"] - labels) - np.abs(pred[with_mode] - labels)
        damage = np.abs(shuffled - labels) - np.abs(pred["LAV"] - labels)
        shift = np.abs(pred["LAV"] - shuffled)
        delta = np.linalg.norm(reps[with_mode] - reps["L"], axis=1)
        relative = delta / (np.linalg.norm(reps["L"], axis=1) + 1e-12)
        gain_stats = bootstrap_summary(
            gain, 2000, stable_seed(270700, cli.cmug_variant, modality, "gain")
        )
        damage_stats = bootstrap_summary(
            damage, 2000, stable_seed(270700, cli.cmug_variant, modality, "damage")
        )
        classification = classify_utility(
            gain_stats, damage_stats, shift.mean(), pred["LAV"].std(), relative.mean()
        )
        rows.append({
            "Variant": cli.cmug_variant, "Split": "valid", "Modality": modality,
            "SampleCount": len(labels),
            "MeanGain": gain_stats["mean"],
            "MeanGainCILow": gain_stats["ci_low"],
            "MeanGainCIHigh": gain_stats["ci_high"],
            "ShuffleDamage": damage_stats["mean"],
            "ShuffleDamageCILow": damage_stats["ci_low"],
            "ShuffleDamageCIHigh": damage_stats["ci_high"],
            "PredictionShiftMean": float(shift.mean()),
            "RelativeShiftMean": float(relative.mean()),
            "InputSensitivityMean": float(sensitivity.mean()),
            "EffectiveRank": effective_rank(reps[with_mode]),
            "UtilityClassification": classification,
            "TestAccessedForUtilityAudit": False,
            "StudentOnly": True,
        })
    result = result_directory(cli.result_root, cli.cmug_variant, False)
    output = result / "{}_valid_modality_utility.csv".format(cli.dataset)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return output


def value_at_best(result, filename, modality, column):
    seed = pd.read_csv(result / "mosi_per_seed.csv").iloc[0]
    epoch = int(seed.BestValidEpoch)
    frame = pd.read_csv(result / filename)
    local = frame[frame.Epoch.eq(epoch) & frame.Modality.eq(modality)]
    if "Split" in local:
        local = local[local.Split.eq("valid")]
    return float(local.iloc[0][column])


def write_final_report(cli):
    roots = {
        variant: result_directory(cli.result_root, variant, False)
        for variant in ("identity_replay", "utility_gate", "utility_gate_matched")
    }
    results = {
        variant: pd.read_csv(root / "mosi_per_seed.csv").iloc[0]
        for variant, root in roots.items()
    }
    replay = results["identity_replay"]
    replay_pass = (
        int(replay.BestValidEpoch) == 9
        and abs(float(replay.J_valid) - 0.6779637237389882) <= 1e-4
        and abs(float(replay.J_test_at_valid_best) - 0.7178811430931091) <= 1e-4
    )
    cmug = results["utility_gate_matched"]
    utility = results["utility_gate"]
    aurocs = {
        modality: value_at_best(
            roots["utility_gate_matched"], "mosi_gate_summary.csv",
            modality, "UtilityAUROC",
        )
        for modality in MODALITIES
    }
    gaps = {
        modality: value_at_best(
            roots["utility_gate_matched"], "mosi_gate_summary.csv",
            modality, "PositiveNegativeGateGap",
        )
        for modality in MODALITIES
    }
    matched = pd.read_csv(
        roots["utility_gate_matched"] / "mosi_matched_shuffle_summary.csv"
    )
    matched = matched[matched.Epoch.eq(int(cmug.BestValidEpoch))]
    recognizable = any(
        aurocs[modality] > .55 and gaps[modality] > 0 for modality in MODALITIES
    )
    improved = float(cmug.J_test_at_valid_best) < 0.7178811430931091
    if not replay_pass:
        classification = "F. Identity Replay failed; implementation failure."
    elif improved and float(cmug.J_test_at_valid_best) < float(utility.J_test_at_valid_best):
        audits = pd.read_csv(
            roots["utility_gate_matched"] / "mosi_valid_modality_utility.csv"
        )
        baseline = pd.read_csv(
            stage7a_directory(cli) / "shuffle_summary.csv"
        )
        positive_audit = bool((audits.MeanGain > 0).any() or (audits.ShuffleDamage > 0).any())
        matched_better = bool((matched.FractionMatchedBetter > .5).any())
        if (
            float(cmug.J_test_at_valid_best) < 0.716961
            and positive_audit and matched_better and recognizable
        ):
            classification = "A. CMUG SUCCESS; freeze before any five-seed replication."
        elif recognizable:
            classification = "A. CMUG exceeds CFCompatKD and Utility-Gate (basic success)."
        else:
            classification = "C. Metric improvement without supported utility recognition."
    elif float(utility.J_test_at_valid_best) < min(
        float(cmug.J_test_at_valid_best), 0.7178811430931091
    ):
        classification = "B. Utility-Gate is best; matched-shuffle constraint is harmful."
    elif recognizable:
        classification = "D. Gate recognizes utility but J does not improve; stop this route."
    else:
        classification = "E. CMUG not supported; stop this route."
    table = pd.DataFrame([{
        "Variant": variant,
        "BestValidEpoch": int(row.BestValidEpoch),
        "J_valid": float(row.J_valid),
        "J_test_at_valid_best": float(row.J_test_at_valid_best),
        "LAV_MAE": float(row.test_at_valid_best_LAV_MAE),
        "MissingMacro_MAE": float(row.test_at_valid_best_MissingMacro_MAE),
    } for variant, row in results.items()])
    lines = [
        "# Stage 7B Conditional Modality Utility Gating Final Audit", "",
        "## Result classification", "", "**{}**".format(classification), "",
        "## Validation-selected performance", "", table.to_markdown(index=False), "",
        "## Identity replay gate", "",
        "- Passed: **{}**".format(replay_pass),
        "- Reference: epoch 9, J_valid 0.677963724, J_test 0.717881143.", "",
        "## CMUG gate diagnostics at validation-best", "",
        "- Audio AUROC/gap: {:.6f} / {:.6f}".format(aurocs["A"], gaps["A"]),
        "- Vision AUROC/gap: {:.6f} / {:.6f}".format(aurocs["V"], gaps["V"]), "",
        "## Matched-shuffle diagnostics", "",
        matched.to_markdown(index=False), "",
        "## Protocol declaration", "",
        "- Main checkpoints were selected by validation J only.",
        "- Utility labels came from Stage7A train artifacts; valid labels were diagnostic only.",
        "- Test never entered utility targets, gate supervision, or modality-utility audit.",
        "- Student-only valid/test inference was used.",
        "- No new Teacher, residual, gradient surgery, adapter, mode head, or representation loss was added.",
        "- Stage7B is complete. No other seed or five-seed replication was started.", "",
    ]
    output = roots["utility_gate_matched"] / "stage7b_cmug_final_audit.md"
    output.write_text("\n".join(lines))
    payload = {
        "Classification": classification,
        "IdentityReplayPassed": replay_pass,
        "CMUGRecognizableUtilityGate": recognizable,
        "CMUGImprovedOverCFCompat": improved,
        "AudioValidAUROC": aurocs["A"], "VisionValidAUROC": aurocs["V"],
        "AudioPositiveNegativeGap": gaps["A"], "VisionPositiveNegativeGap": gaps["V"],
        "NoFiveSeedStarted": True,
    }
    (roots["utility_gate_matched"] / "stage7b_cmug_final_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def main():
    cli = parse_args()
    if cli.final_report:
        write_final_report(cli)
    else:
        valid_audit(cli)


if __name__ == "__main__":
    main()
