"""Windows-native, validation-only clean DLF training support.

This module preserves the original DLF objective and its gradient-accumulation
semantics while avoiding the legacy PyTorch scheduler ``verbose`` argument.
It never constructs or traverses an official Test loader.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import MetricsTop
from .HingeLoss import HingeLoss
from .missing_utils import compute_full_dlf_loss


class CleanDLFValidOnlyTrainer:
    """Train one clean DLF seed using Train and Valid only."""

    def __init__(self, args, logger: Optional[logging.Logger] = None):
        self.args = args
        self.logger = logger or logging.getLogger("windows_valid_only_clean_dlf")
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.hinge = HingeLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    @staticmethod
    def _learning_rate(optimizer: optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def _batch_to_device(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        return (
            batch["text"].to(self.args.device),
            batch["audio"].to(self.args.device),
            batch["vision"].to(self.args.device),
            batch["labels"]["M"].to(self.args.device).view(-1, 1),
        )

    def evaluate(self, model: nn.Module, loader) -> Dict[str, float]:
        """Evaluate the original clean DLF output head on one non-Test split."""
        model.eval()
        losses: List[float] = []
        predictions: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                text, audio, vision, target = self._batch_to_device(batch)
                output = model(text, audio, vision)
                loss = self.criterion(output["output_logit"], target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite clean DLF validation loss.")
                losses.append(float(loss.detach().cpu()))
                predictions.append(output["output_logit"].detach().cpu())
                labels.append(target.detach().cpu())

        if not losses:
            raise RuntimeError("Validation loader is empty.")
        result = {
            str(key): float(value)
            for key, value in self.metrics(
                torch.cat(predictions, dim=0), torch.cat(labels, dim=0)
            ).items()
        }
        # Preserve the original trainer's four-decimal checkpoint criterion.
        result["Loss"] = float(round(float(np.mean(losses)), 4))
        return result

    def train(
        self,
        model: nn.Module,
        loaders: Dict[str, Any],
        checkpoint: Path,
        max_epochs: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Train and return the validation-best result and epoch trajectory."""
        if set(loaders) != {"train", "valid"}:
            raise RuntimeError(
                "Clean DLF Valid-only training requires exactly train/valid loaders."
            )
        if max_epochs is not None and int(max_epochs) < 1:
            raise ValueError("max_epochs must be positive.")

        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        # PyTorch 2.7 removed the legacy ``verbose`` constructor argument.
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.args.patience,
        )

        best_valid_loss = float("inf")
        best_valid_epoch = 0
        best_valid_metrics: Optional[Dict[str, float]] = None
        epoch_rows: List[Dict[str, Any]] = []
        epoch = 0

        while True:
            epoch += 1
            model.train()
            optimizer.zero_grad()
            accumulated_batches = 0
            train_losses: List[float] = []
            train_predictions: List[torch.Tensor] = []
            train_labels: List[torch.Tensor] = []

            for batch in loaders["train"]:
                text, audio, vision, target = self._batch_to_device(batch)
                output = model(text, audio, vision)
                loss, _ = compute_full_dlf_loss(
                    output,
                    target,
                    self.criterion,
                    self.cosine,
                    self.hinge,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite clean DLF training loss.")

                loss.backward()
                if self.args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(
                        model.parameters(), float(self.args.grad_clip)
                    )

                accumulated_batches += 1
                if accumulated_batches == int(self.args.update_epochs):
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated_batches = 0

                train_losses.append(float(loss.detach().cpu()))
                train_predictions.append(output["output_logit"].detach().cpu())
                train_labels.append(target.detach().cpu())

            # The original DLF trainer intentionally does not step a partial
            # final accumulation window. Clear it here before the next epoch.
            dropped_tail_batches = int(accumulated_batches)
            optimizer.zero_grad()

            train_metrics = {
                str(key): float(value)
                for key, value in self.metrics(
                    torch.cat(train_predictions, dim=0),
                    torch.cat(train_labels, dim=0),
                ).items()
            }
            train_metrics["Loss"] = float(np.mean(train_losses))

            valid_metrics = self.evaluate(model, loaders["valid"])
            current_valid_loss = float(valid_metrics["Loss"])
            scheduler.step(current_valid_loss)

            is_best = current_valid_loss <= best_valid_loss - 1e-6
            if is_best:
                best_valid_loss = current_valid_loss
                best_valid_epoch = epoch
                best_valid_metrics = dict(valid_metrics)
                torch.save(model.state_dict(), checkpoint)

            row: Dict[str, Any] = {
                "Epoch": int(epoch),
                "IsBestValid": bool(is_best),
                "LearningRate": self._learning_rate(optimizer),
                "DroppedTailAccumulationBatches": dropped_tail_batches,
            }
            row.update(
                {"train_{}".format(key): value for key, value in train_metrics.items()}
            )
            row.update(
                {"valid_{}".format(key): value for key, value in valid_metrics.items()}
            )
            epoch_rows.append(row)

            self.logger.info(
                "seed=%s epoch=%s train_loss=%.6f valid_loss=%.4f "
                "valid_mae=%.4f best_epoch=%s lr=%.8g dropped_tail=%s",
                self.args.seed,
                epoch,
                train_metrics["Loss"],
                current_valid_loss,
                valid_metrics.get("MAE", float("nan")),
                best_valid_epoch,
                self._learning_rate(optimizer),
                dropped_tail_batches,
            )

            reached_smoke_limit = max_epochs is not None and epoch >= int(max_epochs)
            reached_early_stop = (
                best_valid_epoch > 0
                and epoch - best_valid_epoch >= int(self.args.early_stop)
            )
            if reached_smoke_limit or reached_early_stop:
                break

        if best_valid_metrics is None or not checkpoint.is_file():
            raise RuntimeError("No validation-best clean DLF checkpoint was saved.")

        state = torch.load(checkpoint, map_location=self.args.device)
        model.load_state_dict(state, strict=True)
        final_valid = self.evaluate(model, loaders["valid"])
        if abs(float(final_valid["Loss"]) - float(best_valid_loss)) > 1e-8:
            raise RuntimeError(
                "Reloaded clean DLF checkpoint does not reproduce best Valid loss."
            )

        result: Dict[str, Any] = {
            "BestValidEpoch": int(best_valid_epoch),
            "BestValidLoss": float(best_valid_loss),
            "EpochCount": int(epoch),
            "Checkpoint": str(checkpoint),
            "TestConstructed": False,
            "TestLoaderConstructionCount": 0,
            "PreservedOriginalPartialAccumulationDrop": True,
        }
        result.update(
            {"valid_{}".format(key): value for key, value in final_valid.items()}
        )
        return result, epoch_rows
