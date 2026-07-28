"""Train independent text, audio and vision experts for MOSI/MOSEI V8."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model.UnimodalExpertV8 import UnimodalExpertV8
from trains.singleTask.unimodal_expert_system_v8 import (
    UnimodalExpertTrainerV8,
    dataset_diagnostics,
    fit_train_normalizer,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


LOGGER = logging.getLogger("MMSA")
VALID_MODALITIES = ("text", "audio", "vision")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train stable single-modality experts with prediction and explicit "
            "sample-error heads. Existing DLF/V7 code and checkpoints are not modified."
        )
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument(
        "--save-root",
        type=str,
        default="./result/unimodal_experts_v8",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(VALID_MODALITIES),
        choices=list(VALID_MODALITIES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--text-hidden-dim", type=int, default=128)
    parser.add_argument("--audio-hidden-dim", type=int, default=96)
    parser.add_argument("--vision-hidden-dim", type=int, default=96)
    parser.add_argument("--text-layers", type=int, default=3)
    parser.add_argument("--audio-layers", type=int, default=2)
    parser.add_argument("--vision-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--text-dropout", type=float, default=0.22)
    parser.add_argument("--audio-dropout", type=float, default=0.35)
    parser.add_argument("--vision-dropout", type=float, default=0.35)
    parser.add_argument(
        "--layer-fusion",
        choices=["final", "mid_last"],
        default="final",
        help="E0=final is the default; E1=mid_last is a separate ablation.",
    )
    parser.add_argument(
        "--freeze-text-encoder",
        action="store_true",
        help="Keep BERT frozen. By default BERT is fine-tuned with a smaller LR.",
    )

    parser.add_argument("--prediction-epochs-text", type=int, default=15)
    parser.add_argument("--prediction-epochs-av", type=int, default=20)
    parser.add_argument("--uncertainty-epochs", type=int, default=6)
    parser.add_argument("--joint-epochs", type=int, default=6)
    parser.add_argument("--prediction-patience-text", type=int, default=6)
    parser.add_argument("--prediction-patience-av", type=int, default=8)
    parser.add_argument("--joint-patience", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--text-learning-rate", type=float, default=2e-5)
    parser.add_argument("--uncertainty-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--uncertainty-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-prediction-degradation", type=float, default=0.005)
    parser.add_argument("--max-corr-degradation", type=float, default=0.005)
    return parser.parse_args()


def _profile(cli, modality):
    if modality == "text":
        return {
            "hidden_dim": cli.text_hidden_dim,
            "num_layers": cli.text_layers,
            "dropout": cli.text_dropout,
            "prediction_epochs": cli.prediction_epochs_text,
            "prediction_patience": cli.prediction_patience_text,
        }
    if modality == "audio":
        return {
            "hidden_dim": cli.audio_hidden_dim,
            "num_layers": cli.audio_layers,
            "dropout": cli.audio_dropout,
            "prediction_epochs": cli.prediction_epochs_av,
            "prediction_patience": cli.prediction_patience_av,
        }
    return {
        "hidden_dim": cli.vision_hidden_dim,
        "num_layers": cli.vision_layers,
        "dropout": cli.vision_dropout,
        "prediction_epochs": cli.prediction_epochs_av,
        "prediction_patience": cli.prediction_patience_av,
    }


def _aggregate(seed_frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "MAE",
        "Corr",
        "acc_7",
        "acc_5",
        "acc_2",
        "F1_score",
        "error_spearman",
        "high_error_auroc",
        "q4_q1_ratio",
    ]
    rows = []
    for modality, frame in seed_frame.groupby("modality", sort=False):
        row = {"modality": modality, "seed_count": int(len(frame))}
        for column in metric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        row["all_acceptance_checks_pass_rate"] = float(
            frame["all_acceptance_checks_pass"].astype(float).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    device = assign_gpu([cli.gpu])
    save_root = Path(cli.save_root) / cli.dataset
    save_root.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    diagnostics_written = False

    for seed in cli.seeds:
        setup_seed(seed)
        args = get_config_regression("DLF", cli.dataset, Path(cli.config))
        args["device"] = device
        args["train_mode"] = "regression"
        args["feature_T"] = ""
        args["feature_A"] = ""
        args["feature_V"] = ""
        args["seed"] = seed
        args["cur_seed"] = seed
        args["batch_size"] = cli.batch_size
        args["use_finetune"] = bool(not cli.freeze_text_encoder)
        dataloaders = MMDataLoader(args, cli.num_workers)

        if not diagnostics_written:
            diagnostics = dataset_diagnostics(dataloaders, cli.modalities)
            (save_root / "v8_dataset_diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2), encoding="utf-8"
            )
            diagnostics_written = True

        for modality in cli.modalities:
            setup_seed(seed)
            profile = _profile(cli, modality)
            run_dir = save_root / f"seed_{seed}" / modality
            run_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(
                "Starting V8 expert modality=%s seed=%d profile=%s",
                modality,
                seed,
                profile,
            )

            model = UnimodalExpertV8(
                args=args,
                modality=modality,
                hidden_dim=profile["hidden_dim"],
                num_layers=profile["num_layers"],
                num_heads=cli.num_heads,
                ffn_multiplier=cli.ffn_multiplier,
                dropout=profile["dropout"],
                max_length=512,
                layer_fusion=cli.layer_fusion,
                finetune_text_encoder=not cli.freeze_text_encoder,
            ).to(device)
            mean, std, normalizer_info = fit_train_normalizer(
                dataloaders["train"].dataset, modality
            )
            if mean is not None and std is not None:
                model.set_normalizer(mean.to(device), std.to(device))

            trainer = UnimodalExpertTrainerV8(
                args=args,
                metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
                modality=modality,
                save_dir=run_dir,
                prediction_epochs=profile["prediction_epochs"],
                uncertainty_epochs=cli.uncertainty_epochs,
                joint_epochs=cli.joint_epochs,
                prediction_patience=profile["prediction_patience"],
                joint_patience=cli.joint_patience,
                learning_rate=cli.learning_rate,
                text_learning_rate=cli.text_learning_rate,
                uncertainty_learning_rate=cli.uncertainty_learning_rate,
                weight_decay=cli.weight_decay,
                uncertainty_weight=cli.uncertainty_weight,
                grad_clip=cli.grad_clip,
                max_prediction_degradation=cli.max_prediction_degradation,
                max_corr_degradation=cli.max_corr_degradation,
            )
            summary = trainer.train_and_evaluate(
                model, dataloaders, normalizer_info=normalizer_info
            )
            summary["seed"] = seed
            summary["architecture"] = {
                **profile,
                "num_heads": cli.num_heads,
                "ffn_multiplier": cli.ffn_multiplier,
                "layer_fusion": cli.layer_fusion,
                "finetune_text_encoder": bool(not cli.freeze_text_encoder),
            }
            (run_dir / "unimodal_expert_v8_summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
            )

            metrics = summary["test"]["metrics"]
            uncertainty = summary["test"]["uncertainty"]
            seed_rows.append({
                "seed": seed,
                "modality": modality,
                "selected_stage": summary["selected_stage"],
                **metrics,
                "error_spearman": uncertainty["error_spearman"],
                "high_error_auroc": uncertainty["high_error_auroc"],
                "q4_q1_ratio": uncertainty["q4_q1_ratio"],
                "quartile_monotonic": uncertainty["quartile_monotonic"],
                "all_acceptance_checks_pass": summary["acceptance"]["all_pass"],
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(save_root / "v8_unimodal_seed_results.csv", index=False)
    aggregate = _aggregate(seed_frame)
    aggregate.to_csv(save_root / "v8_unimodal_aggregate.csv", index=False)

    best_modality = None
    if not aggregate.empty:
        best_modality = str(
            aggregate.sort_values("MAE_mean", ascending=True).iloc[0]["modality"]
        )
    overall = {
        "method": "independent_unimodal_experts_v8",
        "dataset": cli.dataset,
        "seeds": list(cli.seeds),
        "modalities": list(cli.modalities),
        "best_modality_by_mean_test_mae": best_modality,
        "seed_results": seed_rows,
        "aggregate": aggregate.to_dict(orient="records"),
        "selection_protocol": (
            "All checkpoints use validation metrics only. Test predictions are "
            "generated after each expert and its error scale are frozen."
        ),
    }
    (save_root / "unimodal_experts_v8_summary.json").write_text(
        json.dumps(overall, indent=2, allow_nan=True), encoding="utf-8"
    )

    LOGGER.info("V8 seed results:\n%s", seed_frame.to_string(index=False))
    LOGGER.info("V8 aggregate:\n%s", aggregate.to_string(index=False))
    LOGGER.info("Best modality by mean Test MAE: %s", best_modality)


if __name__ == "__main__":
    main()
