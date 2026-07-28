"""Train safe independent text/audio/vision experts for V8."""
from __future__ import annotations

import argparse, json, logging
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model.UnimodalExpertV8 import UnimodalExpertV8
from trains.singleTask.unimodal_expert_system_v8 import dataset_diagnostics, fit_train_normalizer
from trains.singleTask.unimodal_expert_system_v8_safe import SafeUnimodalExpertTrainerV8
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed

LOGGER = logging.getLogger("MMSA")
MODALITIES = ("text", "audio", "vision")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", default="./config/config.json")
    p.add_argument("--save-root", default="./result/unimodal_experts_v8")
    p.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    p.add_argument("--seeds", nargs="+", type=int, default=[1111])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--text-hidden-dim", type=int, default=256)
    p.add_argument("--audio-hidden-dim", type=int, default=96)
    p.add_argument("--vision-hidden-dim", type=int, default=96)
    p.add_argument("--text-layers", type=int, default=0)
    p.add_argument("--audio-layers", type=int, default=2)
    p.add_argument("--vision-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ffn-multiplier", type=int, default=4)
    p.add_argument("--text-dropout", type=float, default=.20)
    p.add_argument("--audio-dropout", type=float, default=.35)
    p.add_argument("--vision-dropout", type=float, default=.35)
    p.add_argument("--text-pooling", choices=["cls", "mean"], default="cls")
    p.add_argument("--layer-fusion", choices=["final", "mid_last"], default="final")
    p.add_argument("--freeze-text-encoder", action="store_true")
    p.add_argument("--prediction-epochs-text", type=int, default=20)
    p.add_argument("--prediction-epochs-av", type=int, default=20)
    p.add_argument("--uncertainty-epochs", type=int, default=8)
    p.add_argument("--joint-epochs", type=int, default=0)
    p.add_argument("--prediction-patience-text", type=int, default=8)
    p.add_argument("--prediction-patience-av", type=int, default=8)
    p.add_argument("--joint-patience", type=int, default=3)
    p.add_argument("--joint-mode", choices=["disabled", "detached", "shared"], default="disabled")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--text-learning-rate", type=float, default=1e-5)
    p.add_argument("--uncertainty-learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--uncertainty-weight", type=float, default=.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--prediction-loss-text", choices=["mae", "mse", "huber"], default="mae")
    p.add_argument("--prediction-loss-av", choices=["mae", "mse", "huber"], default="mse")
    p.add_argument("--max-prediction-degradation", type=float, default=.005)
    p.add_argument("--max-corr-degradation", type=float, default=.005)
    return p.parse_args()


def profile(c, modality):
    if modality == "text":
        return dict(hidden=c.text_hidden_dim, layers=c.text_layers, dropout=c.text_dropout,
                    pooling=c.text_pooling, loss=c.prediction_loss_text,
                    epochs=c.prediction_epochs_text, patience=c.prediction_patience_text)
    prefix = "audio" if modality == "audio" else "vision"
    return dict(hidden=getattr(c, f"{prefix}_hidden_dim"), layers=getattr(c, f"{prefix}_layers"),
                dropout=getattr(c, f"{prefix}_dropout"), pooling="mean", loss=c.prediction_loss_av,
                epochs=c.prediction_epochs_av, patience=c.prediction_patience_av)


def deterministic_loaders(loaders, batch_size, workers):
    result = dict(loaders)
    for split in ("valid", "test"):
        result[split] = DataLoader(loaders[split].dataset, batch_size=batch_size,
                                   shuffle=False, drop_last=False, num_workers=workers)
    result["train_eval"] = DataLoader(loaders["train"].dataset, batch_size=batch_size,
                                      shuffle=False, drop_last=False, num_workers=workers)
    return result


def aggregate(frame):
    metrics = ["MAE", "Corr", "acc_7", "acc_5", "acc_2", "F1_score",
               "error_spearman", "high_error_auroc", "q4_q1_ratio"]
    rows = []
    for modality, group in frame.groupby("modality", sort=False):
        row = {"modality": modality, "seed_count": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"], row[f"{metric}_std"] = float(values.mean()), float(values.std(ddof=0))
        row["direction_pass_rate"] = float(group["uncertainty_direction_pass"].astype(float).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    c = parse_args(); logging.basicConfig(level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    device = assign_gpu([c.gpu]); root = Path(c.save_root)/c.dataset; root.mkdir(parents=True, exist_ok=True)
    rows, wrote_diagnostics = [], False
    for seed in c.seeds:
        setup_seed(seed); args = get_config_regression("DLF", c.dataset, Path(c.config))
        args.update(dict(device=device, train_mode="regression", feature_T="", feature_A="", feature_V="",
                         seed=seed, cur_seed=seed, batch_size=c.batch_size,
                         use_finetune=not c.freeze_text_encoder))
        loaders = deterministic_loaders(MMDataLoader(args, c.num_workers), c.batch_size, c.num_workers)
        if not wrote_diagnostics:
            (root/"v8_dataset_diagnostics.json").write_text(
                json.dumps(dataset_diagnostics(loaders, c.modalities), indent=2), encoding="utf-8")
            wrote_diagnostics = True
        for modality in c.modalities:
            setup_seed(seed); cfg = profile(c, modality); run = root/f"seed_{seed}"/modality; run.mkdir(parents=True, exist_ok=True)
            LOGGER.info("V8 start modality=%s seed=%d profile=%s joint=%s", modality, seed, cfg, c.joint_mode)
            model = UnimodalExpertV8(args, modality, cfg["hidden"], cfg["layers"], c.num_heads,
                                    c.ffn_multiplier, cfg["dropout"], 512, c.layer_fusion,
                                    cfg["pooling"], not c.freeze_text_encoder).to(device)
            mean, std, norm = fit_train_normalizer(loaders["train"].dataset, modality)
            if mean is not None: model.set_normalizer(mean.to(device), std.to(device))
            trainer = SafeUnimodalExpertTrainerV8(
                args=args, metrics_fn=MetricsTop("regression").getMetics(c.dataset), modality=modality,
                save_dir=run, prediction_epochs=cfg["epochs"], uncertainty_epochs=c.uncertainty_epochs,
                joint_epochs=c.joint_epochs, prediction_patience=cfg["patience"], joint_patience=c.joint_patience,
                learning_rate=c.learning_rate, text_learning_rate=c.text_learning_rate,
                uncertainty_learning_rate=c.uncertainty_learning_rate, weight_decay=c.weight_decay,
                uncertainty_weight=c.uncertainty_weight, grad_clip=c.grad_clip, prediction_loss=cfg["loss"],
                joint_mode=c.joint_mode, max_prediction_degradation=c.max_prediction_degradation,
                max_corr_degradation=c.max_corr_degradation)
            summary = trainer.train_and_evaluate(model, loaders, norm); summary["seed"] = seed
            (run/"unimodal_expert_v8_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
            m, u = summary["test"]["metrics"], summary["test"]["uncertainty"]
            rows.append(dict(seed=seed, modality=modality, selected_stage=summary["selected_stage"],
                             score_orientation=summary["score_orientation"], **m,
                             error_spearman=u["error_spearman"], high_error_auroc=u["high_error_auroc"],
                             q4_q1_ratio=u["q4_q1_ratio"], quartile_monotonic=u["quartile_monotonic"],
                             uncertainty_direction_pass=summary["acceptance"]["checks"]["uncertainty_direction"]))
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    seed_frame = pd.DataFrame(rows); seed_frame.to_csv(root/"v8_unimodal_seed_results.csv", index=False)
    agg = aggregate(seed_frame); agg.to_csv(root/"v8_unimodal_aggregate.csv", index=False)
    best = None if agg.empty else str(agg.sort_values("MAE_mean").iloc[0]["modality"])
    (root/"unimodal_experts_v8_summary.json").write_text(json.dumps({
        "method":"safe_independent_unimodal_experts_v8","dataset":c.dataset,"seeds":c.seeds,
        "modalities":c.modalities,"joint_mode":c.joint_mode,"best_modality_by_mean_test_mae":best,
        "seed_results":rows,"aggregate":agg.to_dict(orient="records")}, indent=2, allow_nan=True), encoding="utf-8")
    LOGGER.info("V8 results:\n%s", seed_frame.to_string(index=False))


if __name__ == "__main__": main()
