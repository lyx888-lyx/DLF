"""Group-cross-fitted ordinal sentiment-strength coach for V9.3."""

from __future__ import annotations

from pathlib import Path
from statistics import median

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .coach_utils_v93 import cpu_state, region_metrics, safe_group_folds
from .model.OrdinalAdvantageCoachV93 import (
    REGION_NAMES,
    OrdinalRegionCoachV93,
    coach_input_features,
    ordinal_region_loss,
    region_index,
)


class OrdinalRegionCoachCrossFitterV93:
    def __init__(
        self,
        args,
        save_dir,
        oof_cache_path,
        hidden_dim=64,
        dropout=0.15,
        residual_max=2.0,
        ordinal_temperature=0.45,
        folds=5,
        max_epochs=50,
        early_stop=8,
        learning_rate=3e-4,
        weight_decay=1e-3,
        batch_size=64,
    ):
        self.args = args
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.oof = torch.load(oof_cache_path, map_location="cpu")
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual_max = float(residual_max)
        self.ordinal_temperature = float(ordinal_temperature)
        self.folds = int(folds)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        required = {
            "sample_ids",
            "group_ids",
            "labels",
            "oof_prediction",
            "oof_feature",
        }
        if not required.issubset(self.oof):
            raise ValueError(
                f"OOF cache missing keys: {sorted(required - set(self.oof))}"
            )
        if self.oof.get("feature_space") != "auxiliary_prediction_logits_v1":
            raise ValueError(
                "V9.3 requires the V9.2 aligned function-space OOF cache"
            )
        self.anchor = self.oof["oof_prediction"].float().view(-1, 1)
        self.labels = self.oof["labels"].float().view(-1, 1)
        self.features = coach_input_features(
            self.oof["oof_feature"].float(), self.anchor
        )

    def new_model(self):
        return OrdinalRegionCoachV93(
            input_dim=self.features.size(1),
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            residual_max=self.residual_max,
            ordinal_temperature=self.ordinal_temperature,
        ).to(self.args.device)

    def loader(self, indices, shuffle, seed):
        index = torch.as_tensor(indices, dtype=torch.long)
        dataset = TensorDataset(
            self.features[index], self.anchor[index], self.labels[index]
        )
        generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            generator=generator if shuffle else None,
            drop_last=False,
        )

    @torch.no_grad()
    def collect(self, model, features, anchor):
        model.eval()
        buffers = {
            key: []
            for key in ("score", "region_probs", "region_confidence")
        }
        for start in range(0, len(features), self.batch_size):
            output = model(
                features[start : start + self.batch_size].to(
                    self.args.device
                ),
                anchor[start : start + self.batch_size].to(
                    self.args.device
                ),
            )
            for key in buffers:
                buffers[key].append(output[key].detach().cpu())
        return {
            key: torch.cat(values, dim=0)
            for key, values in buffers.items()
        }

    def train_fold(self, train_indices, valid_indices, seed):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_loader = self.loader(train_indices, True, seed)
        valid_index = torch.as_tensor(valid_indices, dtype=torch.long)
        best = None
        history = []
        stale = 0
        boundaries = self.labels.new_tensor(
            (-1.5, -0.5, 0.5, 1.5)
        ).view(1, -1)
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            totals = {}
            for features, anchor, labels in train_loader:
                output = model(
                    features.to(self.args.device),
                    anchor.to(self.args.device),
                )
                losses = ordinal_region_loss(
                    output, labels.to(self.args.device)
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().item()
                    )
            valid = self.collect(
                model,
                self.features[valid_index],
                self.anchor[valid_index],
            )
            valid_payload = {
                "score": valid["score"],
                "ordinal_logits": (
                    valid["score"] - boundaries
                ) / max(self.ordinal_temperature, 1e-6),
                "region_probs": valid["region_probs"],
                "region_confidence": valid["region_confidence"],
            }
            valid_losses = ordinal_region_loss(
                valid_payload, self.labels[valid_index]
            )
            objective = float(valid_losses["total"].item())
            row = {"epoch": epoch, "valid_objective": objective}
            row.update(
                {
                    "train_" + key: value / max(1, len(train_loader))
                    for key, value in totals.items()
                }
            )
            row.update(
                {
                    "valid_" + key: float(value.item())
                    for key, value in valid_losses.items()
                }
            )
            row.update(
                {
                    "valid_" + key: value
                    for key, value in region_metrics(
                        valid["region_probs"], self.labels[valid_index]
                    ).items()
                }
            )
            history.append(row)
            if best is None or objective < best["objective"] - 1e-6:
                best = {
                    "objective": objective,
                    "epoch": epoch,
                    "state": cpu_state(model),
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.early_stop:
                break
        if best is None:
            raise RuntimeError(
                "region fold failed to select a checkpoint"
            )
        model.load_state_dict(best["state"])
        return model, best, history

    def fit(self):
        sample_ids = self.oof["sample_ids"]
        specs, manifest = safe_group_folds(
            sample_ids,
            self.labels.view(-1).tolist(),
            self.folds,
            int(self.args.seed) + 17011,
        )
        manifest.to_csv(
            self.save_dir / "v93_region_group_manifest.csv",
            index=False,
        )
        probabilities = torch.full(
            (len(sample_ids), len(REGION_NAMES)), float("nan")
        )
        scores = torch.full((len(sample_ids), 1), float("nan"))
        confidences = torch.full((len(sample_ids), 1), float("nan"))
        best_epochs = []
        history_rows = []
        for spec in specs:
            seed = int(self.args.seed) + 3109 * (spec.outer_fold + 1)
            model, best, history = self.train_fold(
                spec.inner_train_indices,
                spec.inner_valid_indices,
                seed,
            )
            holdout = torch.as_tensor(
                spec.outer_holdout_indices, dtype=torch.long
            )
            output = self.collect(
                model, self.features[holdout], self.anchor[holdout]
            )
            probabilities[holdout] = output["region_probs"]
            scores[holdout] = output["score"]
            confidences[holdout] = output["region_confidence"]
            best_epochs.append(int(best["epoch"]))
            for row in history:
                history_rows.append(
                    {"fold": spec.outer_fold, **row}
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not torch.isfinite(probabilities).all():
            raise RuntimeError(
                "region OOF probabilities are incomplete"
            )
        pd.DataFrame(history_rows).to_csv(
            self.save_dir / "v93_region_fold_history.csv",
            index=False,
        )
        frame = {
            "sample_id": sample_ids,
            "group_id": self.oof["group_ids"],
            "label": self.labels.view(-1).tolist(),
            "oof_anchor": self.anchor.view(-1).tolist(),
            "ordinal_score": scores.view(-1).tolist(),
            "region_confidence": confidences.view(-1).tolist(),
            "true_region": region_index(self.labels).tolist(),
            "predicted_region": probabilities.argmax(dim=1).tolist(),
        }
        for index, name in enumerate(REGION_NAMES):
            frame["p_" + name] = probabilities[:, index].tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "v93_region_oof_predictions.csv",
            index=False,
        )
        oof_metrics = region_metrics(probabilities, self.labels)
        anchor_probabilities = torch.nn.functional.one_hot(
            region_index(self.anchor), num_classes=len(REGION_NAMES)
        ).float()
        anchor_metrics = region_metrics(
            anchor_probabilities, self.labels
        )
        full_epochs = max(1, int(round(median(best_epochs))))
        torch.manual_seed(int(self.args.seed))
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loader = self.loader(
            range(len(sample_ids)), True, int(self.args.seed)
        )
        full_history = []
        for epoch in range(1, full_epochs + 1):
            model.train()
            totals = {}
            for features, anchor, labels in loader:
                output = model(
                    features.to(self.args.device),
                    anchor.to(self.args.device),
                )
                losses = ordinal_region_loss(
                    output, labels.to(self.args.device)
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().item()
                    )
            full_history.append(
                {
                    "epoch": epoch,
                    **{
                        key: value / max(1, len(loader))
                        for key, value in totals.items()
                    },
                }
            )
        pd.DataFrame(full_history).to_csv(
            self.save_dir / "v93_region_full_history.csv",
            index=False,
        )
        checkpoint = self.save_dir / "ordinal_region_coach_v93.pth"
        torch.save(
            {
                "method": "ordinal_region_coach_v9_3",
                "feature_dim": int(self.features.size(1)),
                "hidden_dim": self.hidden_dim,
                "residual_max": self.residual_max,
                "ordinal_temperature": self.ordinal_temperature,
                "crossfit_best_epochs": best_epochs,
                "full_train_epochs": full_epochs,
                "oof_metrics": oof_metrics,
                "anchor_threshold_oof_metrics": anchor_metrics,
                "state_dict": cpu_state(model),
            },
            checkpoint,
        )
        return model, {
            "probabilities": probabilities,
            "scores": scores,
            "confidences": confidences,
            "metrics": oof_metrics,
            "anchor_threshold_metrics": anchor_metrics,
            "checkpoint": checkpoint,
        }
