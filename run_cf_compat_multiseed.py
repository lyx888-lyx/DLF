"""Stage 3B-M five-seed paired CFCompatKD replication runner.

The runner freezes the Stage 3B mathematics and exposes four ordered actions:
Gate 3 audit, paired ModDrop control, train-only cache, and CFCompatKD training.
Seed 1111 is referenced as a locked historical result and is never trained here.
"""
import argparse
import json
import logging
import math
import os
import platform
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import train_cf_compat_kd
import train_missing
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    MULTISEED_CACHE_VERSION,
    MULTISEED_SMOKE_CACHE_VERSION,
    cache_paths,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    flatten_mode_metrics,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


ALL_SEEDS = (1111, 1112, 1113, 1114, 1115)
NEW_SEEDS = (1112, 1113, 1114, 1115)
VALID_STATUSES = {"locked_existing", "pending", "running", "completed", "invalid", "failed"}
GATE3_LOG = Path("log/clean/DLF-mosi-gate3-train-20260714-072021.log")
GATE3_SOURCE_RESULT = Path("result/clean/train/mosi.csv")
MULTISEED_RESULT = Path("result/missing_baseline/cf_compat_kd_v1/benchmark_multiseed")
GATE3_MANIFEST = MULTISEED_RESULT / "gate3_checkpoint_manifest.csv"
RUN_MANIFEST = MULTISEED_RESULT / "RUN_MANIFEST.json"
CACHE_MANIFEST_NAME = "cache_manifest.csv"
REQUIRED_SEED_OUTPUTS = (
    "mosi_per_seed.csv", "mosi_epoch_metrics.csv",
    "mosi_best_valid_predictions.csv", "mosi_best_test_diagnostic_predictions.csv",
    "mosi_gate_summary.csv", "mosi_gate_quartiles.csv",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_value(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Frozen Stage 3B-M paired replication runner.")
    parser.add_argument("--action", required=True, choices=("audit-gate3", "moddrop", "cache", "cfcompat"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--gate-mode", default="compat")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    cli = parser.parse_args(argv)
    if cli.gate_mode != "compat":
        parser.error("Stage 3B-M permits only compatibility gating.")
    if cli.eta != 1.0 or cli.lambda_kd != 1.0:
        parser.error("Stage 3B-M fixes eta=lambda_kd=1.0.")
    if cli.action == "audit-gate3":
        if cli.seed is not None:
            parser.error("Gate 3 audit always covers all five fixed seeds.")
    elif cli.seed not in NEW_SEEDS:
        parser.error("Only new seeds 1112-1115 may be executed; seed1111 is locked.")
    if cli.max_epochs is not None and cli.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if cli.smoke_test:
        cli.max_epochs = 2 if cli.max_epochs is None else min(2, cli.max_epochs)
    return cli


def parse_gate3_best_epochs(log_path=GATE3_LOG):
    if not Path(log_path).is_file():
        raise FileNotFoundError("Gate 3 audit log is absent: {}".format(log_path))
    current = None
    losses = {index: [] for index in range(1, 6)}
    for line in Path(log_path).read_text(errors="replace").splitlines():
        match = re.search(r"Epoch:\s*(\d+) TRAIN .*\[(\d+)/(\d+)/(\d+)\]", line)
        if match:
            current = (int(match.group(4)), int(match.group(1)))
        elif current and "VAL-(DLF)" in line:
            loss = re.search(r"Loss:\s*([0-9.]+)", line)
            if loss:
                losses[current[0]].append((current[1], float(loss.group(1))))
    if any(not rows for rows in losses.values()):
        raise RuntimeError("Gate 3 log does not contain five complete validation trajectories.")
    return {seed: min(losses[index], key=lambda item: item[1])[0]
            for index, seed in enumerate(ALL_SEEDS, 1)}


def audit_gate3_checkpoints(model_save_dir="pt"):
    train_source = Path("train.py").read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", "", train_source)
    if "seeds=[1111,1112,1113,1114,1115]" not in normalized:
        raise RuntimeError("Gate 3 source does not bind the fixed five-seed sequence.")
    if not GATE3_SOURCE_RESULT.is_file():
        raise FileNotFoundError("Gate 3 aggregate result is absent: {}".format(GATE3_SOURCE_RESULT))
    best_epochs = parse_gate3_best_epochs()
    rows, hashes = [], set()
    for seed in ALL_SEEDS:
        checkpoint = Path(model_save_dir) / "DLF_mosi_seed{}_best.pth".format(seed)
        if not checkpoint.is_file():
            raise FileNotFoundError("Gate 3 checkpoint is absent for seed {}: {}".format(seed, checkpoint))
        state = torch.load(checkpoint, map_location="cpu")
        sha = checkpoint_sha256(checkpoint)
        if sha in hashes:
            raise RuntimeError("Different seeds unexpectedly share one Gate 3 checkpoint SHA.")
        hashes.add(sha)
        rows.append({
            "Seed": seed,
            "Checkpoint": str(checkpoint),
            "SHA256": sha,
            "SizeBytes": checkpoint.stat().st_size,
            "StateDictKeyCount": len(state),
            "BestEpoch": best_epochs[seed],
            "SourceResult": "{};{};tag=gate3-mosi-clean-5seed".format(GATE3_LOG, GATE3_SOURCE_RESULT),
            "Protocol": "Gate3 validation-best clean test-once five-seed",
            "Verified": True,
        })
        del state
    GATE3_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(GATE3_MANIFEST, index=False)
    initialize_run_manifest(rows)
    return pd.DataFrame(rows)


def _locked_seed1111_entry(gate_row):
    moddrop = Path("pt/missing_baseline/moddrop/DLF_mosi_seed1111_best.pth")
    cfcompat = Path("pt/missing_baseline/cf_compat_kd_v1/DLF_mosi_seed1111_best_valid.pth")
    old_cache = cache_paths("result", "mosi")
    return {
        "Seed": 1111,
        "Status": "locked_existing",
        "Source": "reused_locked_existing_result",
        "Gate3Checkpoint": gate_row["Checkpoint"],
        "Gate3SHA256": gate_row["SHA256"],
        "ModDropCheckpoint": str(moddrop),
        "ModDropSHA256": checkpoint_sha256(moddrop),
        "EvaluatorCheckpoint": str(moddrop),
        "EvaluatorSHA256": checkpoint_sha256(moddrop),
        "CachePath": str(old_cache["csv"]),
        "CacheSHA256": checkpoint_sha256(old_cache["csv"]),
        "CFCompatCheckpoint": str(cfcompat),
        "CFCompatSHA256": checkpoint_sha256(cfcompat),
        "ModDropBestValidEpoch": 21,
        "CFCompatBestValidEpoch": 9,
        "StartTime": None,
        "EndTime": None,
        "ExitCode": 0,
        "CodeCommit": "d6adc7b170c8b7cc13df72dea93b86458c9ae36e",
        "Processes": [],
    }


def initialize_run_manifest(gate_rows):
    mapped = {int(row["Seed"]): row for row in gate_rows}
    manifest = {
        "Protocol": "Stage 3B-M CFCompatKD-v1 Five-Seed Paired Replication",
        "AllowedStatuses": sorted(VALID_STATUSES),
        "BaseCommit": "d6adc7b170c8b7cc13df72dea93b86458c9ae36e",
        "Seeds": [_locked_seed1111_entry(mapped[1111])],
    }
    for seed in NEW_SEEDS:
        manifest["Seeds"].append({
            "Seed": seed, "Status": "pending", "Source": "new_formal_replication",
            "Gate3Checkpoint": mapped[seed]["Checkpoint"], "Gate3SHA256": mapped[seed]["SHA256"],
            "ModDropCheckpoint": None, "ModDropSHA256": None,
            "EvaluatorCheckpoint": None, "EvaluatorSHA256": None,
            "CachePath": None, "CacheSHA256": None,
            "CFCompatCheckpoint": None, "CFCompatSHA256": None,
            "ModDropBestValidEpoch": None, "CFCompatBestValidEpoch": None,
            "StartTime": None, "EndTime": None, "ExitCode": None,
            "CodeCommit": None, "Processes": [],
        })
    RUN_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_run_manifest():
    if not RUN_MANIFEST.is_file():
        raise FileNotFoundError("RUN_MANIFEST is absent; run --action audit-gate3 first.")
    manifest = json.loads(RUN_MANIFEST.read_text())
    if [int(row["Seed"]) for row in manifest["Seeds"]] != list(ALL_SEEDS):
        raise ValueError("RUN_MANIFEST seed order is invalid.")
    return manifest


def update_run_manifest(seed, **updates):
    manifest = load_run_manifest()
    selected = [row for row in manifest["Seeds"] if int(row["Seed"]) == int(seed)]
    if len(selected) != 1:
        raise ValueError("RUN_MANIFEST does not contain one unique seed row.")
    if "Status" in updates and updates["Status"] not in VALID_STATUSES:
        raise ValueError("Invalid run status: {}".format(updates["Status"]))
    selected[0].update(updates)
    RUN_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def record_process(seed, action, start, end, exit_code, log_path, extra=None):
    manifest = load_run_manifest()
    row = next(item for item in manifest["Seeds"] if int(item["Seed"]) == int(seed))
    row.setdefault("Processes", []).append({
        "Action": action, "PID": os.getpid(), "LogPath": str(log_path),
        "StartTime": start, "EndTime": end, "ExitCode": int(exit_code),
        "CodeCommit": git_value("rev-parse", "HEAD"), "CUDADevice": "cuda:{}".format(0),
        "Python": platform.python_version(), "Torch": torch.__version__,
        "Transformers": __import__("transformers").__version__, **(extra or {}),
    })
    row["StartTime"] = row["StartTime"] or start
    row["EndTime"], row["ExitCode"], row["CodeCommit"] = end, int(exit_code), git_value("rev-parse", "HEAD")
    RUN_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def assert_gate3_binding(seed):
    if not GATE3_MANIFEST.is_file():
        raise FileNotFoundError("Gate 3 checkpoint manifest is absent.")
    rows = pd.read_csv(GATE3_MANIFEST)
    selected = rows.loc[rows.Seed.astype(int) == int(seed)]
    if len(selected) != 1 or not bool(selected.iloc[0].Verified):
        raise ValueError("Gate 3 checkpoint is not uniquely verified for seed {}.".format(seed))
    checkpoint = Path(str(selected.iloc[0].Checkpoint))
    if checkpoint_sha256(checkpoint) != str(selected.iloc[0].SHA256):
        raise ValueError("Gate 3 checkpoint SHA changed for seed {}.".format(seed))
    return checkpoint, str(selected.iloc[0].SHA256)


def moddrop_paths(cli, seed):
    result = Path(cli.result_root) / "missing_baseline" / "moddrop_benchmark_multiseed_v1"
    checkpoint_root = Path(cli.model_save_dir) / "missing_baseline" / "moddrop_benchmark_multiseed_v1"
    if cli.smoke_test:
        result, checkpoint_root = result / "smoke", checkpoint_root / "smoke"
    result = result / "seed{}".format(seed)
    main = checkpoint_root / "seed{}".format(seed) / "DLF_mosi_seed{}_best_valid.pth".format(seed)
    diagnostic = main.parent / "diagnostic" / "DLF_mosi_seed{}_best_test_diagnostic.pth".format(seed)
    return result, main, diagnostic


def create_logger(cli, tag):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "train"
    if tag == "moddrop-benchmark":
        name = "DLF-mosi-moddrop-benchmark-seed{}-{}-{}.log".format(cli.seed, kind, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        name = "DLF-mosi-cfcompatkd-multiseed-seed{}-{}-{}.log".format(cli.seed, kind, datetime.now().strftime("%Y%m%d-%H%M%S"))
    path = directory / name
    logger = logging.getLogger("stage3bm-{}".format(tag))
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("pid=%s commit=%s device_request=%s python=%s torch=%s transformers=%s",
                os.getpid(), git_value("rev-parse", "HEAD"), cli.gpu_ids,
                platform.python_version(), torch.__version__, __import__("transformers").__version__)
    return logger, path


def _finite(metrics):
    return all(math.isfinite(float(value)) for row in metrics.values() for value in row.values())


def _prefix(metrics, prefix):
    return {"{}_{}".format(prefix, key): value for key, value in flatten_mode_metrics(metrics).items()}


def train_moddrop_benchmark(cli, logger):
    seed = int(cli.seed)
    gate3_checkpoint, gate3_sha = assert_gate3_binding(seed)
    result_dir, main_checkpoint, diagnostic_checkpoint = moddrop_paths(cli, seed)
    if not cli.smoke_test and (result_dir.exists() or main_checkpoint.exists() or diagnostic_checkpoint.exists()):
        raise FileExistsError("Formal ModDrop output already exists for seed {}; selective reruns are forbidden.".format(seed))
    setup_seed(seed)
    args = train_missing.build_config(cli, seed)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("ModDrop benchmark training must expose only train/valid through MMDataLoader.")
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(gate3_checkpoint, map_location=args.device), strict=True)
    model = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=args.patience)
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    generator = torch.Generator().manual_seed(seed + 104729)
    main_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_valid_j = best_test_j = float("inf")
    best_valid_epoch = best_test_epoch = 0
    epoch_rows, first_counts = [], None
    logger.info("seed=%s Gate3=%s Gate3SHA=%s benchmark valid/test every epoch; main selection=J_valid only",
                seed, gate3_checkpoint, gate3_sha)
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        model.train(); optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        totals = {"full": 0., "missing": 0., "total": 0.}
        grad_audio = grad_vision = 0.
        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = train_cf_compat_kd.batch_to_device(batch, args.device)
            full = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            full_loss, _ = compute_full_dlf_loss(model(text, audio, vision, full), labels, criterion, cosine, hinge)
            mask = sample_missing_masks(labels.size(0), generator, args.device, audio.dtype)
            counts.update(count_missing_modes(mask))
            missing_loss, _ = compute_task_loss(model(text, audio, vision, mask), labels, criterion)
            loss = full_loss + missing_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("NaN/Inf in paired ModDrop loss.")
            loss.backward()
            if model.missing_audio_token.grad is not None:
                grad_audio = max(grad_audio, float(model.missing_audio_token.grad.norm()))
            if model.missing_vision_token.grad is not None:
                grad_vision = max(grad_vision, float(model.missing_vision_token.grad.norm()))
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                if args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step(); optimizer.zero_grad()
            totals["full"] += float(full_loss); totals["missing"] += float(missing_loss); totals["total"] += float(loss)
        if epoch == 1:
            first_counts = dict(counts)
        valid = evaluate_all_modes(model, loaders["valid"], args.device, "moddrop", criterion)
        test = evaluate_all_modes(model, test_loader, args.device, "moddrop", criterion)
        if not _finite(valid) or not _finite(test) or grad_audio <= 0 or grad_vision <= 0:
            raise FloatingPointError("Non-finite metric or missing-token zero gradient in paired ModDrop.")
        j_valid, j_test = validation_objective(valid), validation_objective(test)
        scheduler.step(j_valid)
        is_best_valid = j_valid <= best_valid_j - 1e-6
        is_best_test = j_test <= best_test_j - 1e-6
        if is_best_valid:
            best_valid_j, best_valid_epoch = j_valid, epoch
            torch.save(model.state_dict(), main_checkpoint)
        if is_best_test:
            best_test_j, best_test_epoch = j_test, epoch
            torch.save(model.state_dict(), diagnostic_checkpoint)
        batches = len(loaders["train"])
        epoch_rows.append({
            "Seed": seed, "Epoch": epoch, "Method": "DLF-ModDrop-Paired-v1",
            "J_valid": j_valid, "J_test": j_test,
            "IsBestValid": is_best_valid, "IsBestTestDiagnostic": is_best_test,
            "LA_count": counts["LA"], "LV_count": counts["LV"], "L_count": counts["L"],
            "full_loss": totals["full"] / batches, "missing_loss": totals["missing"] / batches,
            "total_loss": totals["total"] / batches, "token_grad_audio": grad_audio,
            "token_grad_vision": grad_vision, **_prefix(valid, "valid"), **_prefix(test, "test"),
        })
        logger.info("seed=%s epoch=%s LA=%s LV=%s L=%s J_valid=%.6f J_test=%.6f",
                    seed, epoch, counts["LA"], counts["LV"], counts["L"], j_valid, j_test)
        if epoch - best_valid_epoch >= args.early_stop:
            break
    if not main_checkpoint.is_file() or not diagnostic_checkpoint.is_file():
        raise RuntimeError("Both paired ModDrop checkpoints must exist.")
    model.load_state_dict(torch.load(main_checkpoint, map_location=args.device), strict=True)
    final_valid = evaluate_all_modes(model, loaders["valid"], args.device, "moddrop", criterion)
    final_test = evaluate_all_modes(model, test_loader, args.device, "moddrop", criterion)
    main_predictions = train_cf_compat_kd.prediction_rows(model, loaders["valid"], args.device)
    main_predictions["selected_by"] = "validation"; main_predictions["diagnostic_only"] = False
    model.load_state_dict(torch.load(diagnostic_checkpoint, map_location=args.device), strict=True)
    diagnostic_test = evaluate_all_modes(model, test_loader, args.device, "moddrop", criterion)
    diagnostic_predictions = train_cf_compat_kd.prediction_rows(model, test_loader, args.device)
    diagnostic_predictions["selected_by"] = "test"; diagnostic_predictions["diagnostic_only"] = True
    diagnostic_predictions["not_main_result"] = True
    row = {
        "Seed": seed, "Method": "DLF-ModDrop-Paired-v1", "BestValidEpoch": best_valid_epoch,
        "J_valid": validation_objective(final_valid), "J_test_at_valid_best": validation_objective(final_test),
        "BestObservedTestEpoch": best_test_epoch, "BestObservedTestJ": validation_objective(diagnostic_test),
        "MainCheckpoint": str(main_checkpoint), "DiagnosticCheckpoint": str(diagnostic_checkpoint),
        "Gate3Checkpoint": str(gate3_checkpoint), "Gate3SHA256": gate3_sha,
        "FirstEpochCount_LA": first_counts["LA"], "FirstEpochCount_LV": first_counts["LV"],
        "FirstEpochCount_L": first_counts["L"], "EpochCount": len(epoch_rows),
        **_prefix(final_valid, "valid"), **_prefix(final_test, "test_at_valid_best"),
        **_prefix(diagnostic_test, "test_diagnostic"),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    write_result_csvs([row], result_dir, cli.dataset)
    pd.DataFrame(epoch_rows).to_csv(result_dir / "mosi_epoch_metrics.csv", index=False)
    main_predictions.to_csv(result_dir / "mosi_best_valid_predictions.csv", index=False)
    diagnostic_predictions.to_csv(result_dir / "mosi_best_test_diagnostic_predictions.csv", index=False)
    return row


def cf_cli(cli):
    return argparse.Namespace(
        dataset=cli.dataset, seeds=[cli.seed], gate_mode="compat", eta=1.0, lambda_kd=1.0,
        build_gate_cache_only=False, smoke_test=cli.smoke_test, max_epochs=cli.max_epochs,
        num_workers=cli.num_workers, gpu_ids=cli.gpu_ids, model_save_dir=cli.model_save_dir,
        result_root=cli.result_root, log_dir=cli.log_dir, config_file=cli.config_file,
        multiseed_replication=True,
    )


def build_seed_cache(cli, logger):
    assert_gate3_binding(cli.seed)
    return train_cf_compat_kd.build_gate_cache_only(cf_cli(cli), cli.seed, logger)


def refresh_cache_manifest(root, dataset):
    """Write an auditable manifest for the locked cache and available new caches."""
    rows = []
    for seed in ALL_SEEDS:
        paths = (cache_paths(root, dataset) if seed == 1111 else
                 cache_paths(root, dataset, MULTISEED_CACHE_VERSION, seed))
        if not paths["csv"].is_file() or not paths["config"].is_file():
            continue
        config = json.loads(paths["config"].read_text())
        actual_cache_sha = checkpoint_sha256(paths["csv"])
        if actual_cache_sha != config.get("cache_sha256", actual_cache_sha):
            raise ValueError("Cache SHA mismatch for seed {}.".format(seed))
        if int(config["train_sample_count"]) != 1284 or config.get("source") != "train_only":
            raise ValueError("Cache seed{} is not an exact train-only cache.".format(seed))
        rows.append({
            "Seed": seed,
            "Status": "reused_locked_existing_result" if seed == 1111 else "generated_same_seed",
            "EvaluatorCheckpoint": config["evaluator_checkpoint"],
            "EvaluatorSHA256": config["evaluator_sha256"],
            "TrainSampleCount": int(config["train_sample_count"]),
            "ConfigSHA256": config["config_sha256"],
            "CachePath": str(paths["csv"]),
            "CacheSHA256": actual_cache_sha,
            "CreatedFromTrainOnly": config.get("created_from_train_only", config["source"] == "train_only"),
            "RNGStatePreserved": config.get("rng_state_preserved", pd.NA),
        })
    destination = Path(root) / "counterfactual_compatibility" / MULTISEED_CACHE_VERSION / dataset
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / CACHE_MANIFEST_NAME, index=False)
    return rows


def train_seed_cfcompat(cli, logger):
    assert_gate3_binding(cli.seed)
    local = cf_cli(cli)
    result_dir, main_template, diagnostic_template = train_cf_compat_kd.method_paths(local, cli.dataset, cli.seed)
    if not cli.smoke_test and (result_dir.exists() or Path(str(main_template).format(cli.seed)).exists()):
        raise FileExistsError("Formal CFCompat output already exists for seed {}; selective reruns are forbidden.".format(cli.seed))
    row, epochs, summaries, quartiles, valid, diagnostic = train_cf_compat_kd.train_one_seed(local, cli.seed, logger)
    write_result_csvs([row], result_dir, cli.dataset)
    pd.DataFrame(epochs).to_csv(result_dir / "mosi_epoch_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(result_dir / "mosi_gate_summary.csv", index=False)
    pd.DataFrame(quartiles).to_csv(result_dir / "mosi_gate_quartiles.csv", index=False)
    valid.to_csv(result_dir / "mosi_best_valid_predictions.csv", index=False)
    diagnostic.to_csv(result_dir / "mosi_best_test_diagnostic_predictions.csv", index=False)
    return row


def _formal_phase_guard(action, seed):
    manifest = load_run_manifest()
    rows = {int(row["Seed"]): row for row in manifest["Seeds"]}
    if action == "moddrop":
        for prior in NEW_SEEDS:
            if prior >= seed:
                break
            if not rows[prior].get("ModDropCheckpoint"):
                raise RuntimeError("Fixed Phase B order requires ModDrop seed{} first.".format(prior))
    elif action == "cache":
        if any(not rows[s].get("ModDropCheckpoint") for s in NEW_SEEDS):
            raise RuntimeError("Phase C cannot start until all four ModDrop controls complete.")
        for prior in NEW_SEEDS:
            if prior >= seed:
                break
            if not rows[prior].get("CachePath"):
                raise RuntimeError("Fixed Phase C cache order was violated.")
    elif action == "cfcompat":
        if any(not rows[s].get("CachePath") for s in NEW_SEEDS):
            raise RuntimeError("Phase D cannot start until all four caches complete.")
        for prior in NEW_SEEDS:
            if prior >= seed:
                break
            if rows[prior].get("Status") != "completed":
                raise RuntimeError("Fixed Phase D CFCompat order was violated.")


def main(argv=None):
    cli = parse_args(argv)
    if cli.action == "audit-gate3":
        frame = audit_gate3_checkpoints(cli.model_save_dir)
        print(frame.to_string(index=False))
        return
    if not cli.smoke_test:
        _formal_phase_guard(cli.action, cli.seed)
        update_run_manifest(cli.seed, Status="running")
    logger, log_path = create_logger(cli, "moddrop-benchmark" if cli.action == "moddrop" else "cfcompatkd")
    start = utc_now()
    try:
        if cli.action == "moddrop":
            row = train_moddrop_benchmark(cli, logger)
            extra = {"ModDropCheckpoint": row["MainCheckpoint"], "ModDropSHA256": checkpoint_sha256(row["MainCheckpoint"]),
                     "EvaluatorCheckpoint": row["MainCheckpoint"], "EvaluatorSHA256": checkpoint_sha256(row["MainCheckpoint"]),
                     "ModDropBestValidEpoch": int(row["BestValidEpoch"]), "Status": "pending"}
        elif cli.action == "cache":
            paths = build_seed_cache(cli, logger)
            version = MULTISEED_SMOKE_CACHE_VERSION if cli.smoke_test else MULTISEED_CACHE_VERSION
            config = json.loads(cache_paths(cli.result_root, cli.dataset, version, cli.seed)["config"].read_text())
            if not cli.smoke_test:
                refresh_cache_manifest(cli.result_root, cli.dataset)
            extra = {"CachePath": str(paths["csv"]), "CacheSHA256": config["cache_sha256"],
                     "Status": "pending"}
        else:
            row = train_seed_cfcompat(cli, logger)
            extra = {"CFCompatCheckpoint": row["MainCheckpoint"], "CFCompatSHA256": checkpoint_sha256(row["MainCheckpoint"]),
                     "CFCompatBestValidEpoch": int(row["BestValidEpoch"]), "Status": "completed"}
        end = utc_now()
        logger.info("action=%s seed=%s completed exit_code=0 log=%s", cli.action, cli.seed, log_path)
        if not cli.smoke_test:
            update_run_manifest(cli.seed, **extra)
            record_process(cli.seed, cli.action, start, end, 0, log_path,
                           {"CUDADevice": "cuda:{}".format(cli.gpu_ids[0])})
    except Exception:
        end = utc_now()
        logger.exception("action=%s seed=%s failed", cli.action, cli.seed)
        if not cli.smoke_test:
            update_run_manifest(cli.seed, Status="failed")
            record_process(cli.seed, cli.action, start, end, 1, log_path,
                           {"CUDADevice": "cuda:{}".format(cli.gpu_ids[0])})
        raise


if __name__ == "__main__":
    main()
