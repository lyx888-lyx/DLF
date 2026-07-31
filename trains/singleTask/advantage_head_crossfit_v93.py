"""Validation-group-cross-fitted specialist advantage heads for V9.3."""

from __future__ import annotations

from pathlib import Path
from statistics import median

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .coach_utils_v93 import (
    cpu_state,
    gain_diagnostics,
    safe_group_folds,
)
from .model.OrdinalAdvantageCoachV93 import (
    REGION_NAMES,
    SPECIALIST_NAMES,
    SPECIALIST_TO_REGION,
    AdvantageHeadV93,
    advantage_input_features,
    advantage_loss,
    coach_input_features,
)
from .oof_group_splits_v92 import conversation_group_id


class AdvantageHeadCrossFitterV93:
    def __init__(
        self,
        args,
        save_dir,
        valid_pool,
        test_pool,
        region_collector,
        region_model,
        hidden_dim=48,
        dropout=0.10,
        gain_max=1.5,
        folds=5,
        max_epochs=60,
        early_stop=8,
        learning_rate=3e-4,
        weight_decay=1e-3,
        batch_size=64,
        win_margin=0.02,
    ):
        self.args = args
        self.save_dir = Path(save_dir)
        self.valid = valid_pool
        self.test = test_pool
        self.region_collector = region_collector
        self.region_model = region_model
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.gain_max = float(gain_max)
        self.folds = int(folds)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.win_margin = float(win_margin)

    def new_model(self, input_dim):
        return AdvantageHeadV93(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            gain_max=self.gain_max,
        ).to(self.args.device)

    def loader(self, features, gains, indices, shuffle, seed):
        index = torch.as_tensor(indices, dtype=torch.long)
        dataset = TensorDataset(features[index], gains[index])
        generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            generator=generator if shuffle else None,
            drop_last=False,
        )

    @torch.no_grad()
    def collect(self, model, features):
        model.eval()
        buffers = {"predicted_gain": [], "win_probability": []}
        for start in range(0, len(features), self.batch_size):
            output = model(
                features[start : start + self.batch_size].to(
                    self.args.device
                )
            )
            for key in buffers:
                buffers[key].append(output[key].detach().cpu())
        return {
            key: torch.cat(values, dim=0)
            for key, values in buffers.items()
        }

    def train_fold(
        self,
        features,
        gains,
        train_indices,
        valid_indices,
        seed,
    ):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        model = self.new_model(features.size(1))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loader = self.loader(
            features, gains, train_indices, True, seed
        )
        valid_index = torch.as_tensor(
            valid_indices, dtype=torch.long
        )
        best = None
        history = []
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            totals = {}
            for batch_features, batch_gains in loader:
                output = model(
                    batch_features.to(self.args.device)
                )
                losses = advantage_loss(
                    output,
                    batch_gains.to(self.args.device),
                    self.win_margin,
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().item()
                    )
            valid = self.collect(model, features[valid_index])
            valid_losses = advantage_loss(
                {
                    "predicted_gain": valid["predicted_gain"],
                    "win_logit": torch.logit(
                        valid["win_probability"].clamp(
                            1e-6, 1.0 - 1e-6
                        )
                    ),
                },
                gains[valid_index],
                self.win_margin,
            )
            objective = float(valid_losses["total"].item())
            row = {"epoch": epoch, "valid_objective": objective}
            row.update(
                {
                    "train_" + key: value / max(1, len(loader))
                    for key, value in totals.items()
                }
            )
            row.update(
                {
                    "valid_" + key: float(value.item())
                    for key, value in valid_losses.items()
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
                "advantage fold failed to select a checkpoint"
            )
        model.load_state_dict(best["state"])
        return model, best, history

    def fit(self):
        valid_coach_features = coach_input_features(
            self.valid["function_space"].float(),
            self.valid["anchor"].float(),
        )
        test_coach_features = coach_input_features(
            self.test["function_space"].float(),
            self.test["anchor"].float(),
        )
        valid_region = self.region_collector.collect(
            self.region_model,
            valid_coach_features,
            self.valid["anchor"],
        )
        test_region = self.region_collector.collect(
            self.region_model,
            test_coach_features,
            self.test["anchor"],
        )
        sample_ids = self.valid["sample_ids"]
        labels = self.valid["labels"].view(-1)
        specs, manifest = safe_group_folds(
            sample_ids,
            labels.tolist(),
            self.folds,
            int(self.args.seed) + 29021,
        )
        manifest.to_csv(
            self.save_dir / "v93_advantage_group_manifest.csv",
            index=False,
        )
        results = {}
        history_rows = []
        prediction_frame = {
            "sample_id": sample_ids,
            "group_id": [
                conversation_group_id(value)
                for value in sample_ids
            ],
            "label": labels.tolist(),
            "anchor": self.valid["anchor"].view(-1).tolist(),
        }
        for name in SPECIALIST_NAMES:
            expert = self.valid["experts"][name]
            features = advantage_input_features(
                valid_coach_features,
                valid_region["region_probs"],
                valid_region["score"],
                self.valid["anchor"],
                expert["prediction"],
                expert["confidence"],
                SPECIALIST_TO_REGION[name],
            )
            gains = (
                torch.abs(
                    self.valid["anchor"] - self.valid["labels"]
                )
                - torch.abs(
                    expert["prediction"] - self.valid["labels"]
                )
            )
            oof_gain = torch.full_like(gains, float("nan"))
            oof_win = torch.full_like(gains, float("nan"))
            best_epochs = []
            for spec in specs:
                seed = (
                    int(self.args.seed)
                    + 5003 * (spec.outer_fold + 1)
                    + 97 * SPECIALIST_NAMES.index(name)
                )
                model, best, history = self.train_fold(
                    features,
                    gains,
                    spec.inner_train_indices,
                    spec.inner_valid_indices,
                    seed,
                )
                holdout = torch.as_tensor(
                    spec.outer_holdout_indices,
                    dtype=torch.long,
                )
                output = self.collect(model, features[holdout])
                oof_gain[holdout] = output["predicted_gain"]
                oof_win[holdout] = output["win_probability"]
                best_epochs.append(int(best["epoch"]))
                for row in history:
                    history_rows.append(
                        {
                            "expert": name,
                            "fold": spec.outer_fold,
                            **row,
                        }
                    )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if (
                not torch.isfinite(oof_gain).all()
                or not torch.isfinite(oof_win).all()
            ):
                raise RuntimeError(
                    f"incomplete OOF advantage predictions for {name}"
                )
            full_epochs = max(
                1, int(round(median(best_epochs)))
            )
            torch.manual_seed(
                int(self.args.seed)
                + 97 * SPECIALIST_NAMES.index(name)
            )
            model = self.new_model(features.size(1))
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            loader = self.loader(
                features,
                gains,
                range(len(features)),
                True,
                int(self.args.seed),
            )
            for _ in range(full_epochs):
                model.train()
                for batch_features, batch_gains in loader:
                    output = model(
                        batch_features.to(self.args.device)
                    )
                    losses = advantage_loss(
                        output,
                        batch_gains.to(self.args.device),
                        self.win_margin,
                    )
                    optimizer.zero_grad()
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), 2.0
                    )
                    optimizer.step()
            test_expert = self.test["experts"][name]
            test_features = advantage_input_features(
                test_coach_features,
                test_region["region_probs"],
                test_region["score"],
                self.test["anchor"],
                test_expert["prediction"],
                test_expert["confidence"],
                SPECIALIST_TO_REGION[name],
            )
            test_output = self.collect(model, test_features)
            diagnostics = gain_diagnostics(
                oof_gain,
                oof_win,
                gains,
                self.win_margin,
            )
            checkpoint = (
                self.save_dir
                / f"advantage_head_{name}_v93.pth"
            )
            torch.save(
                {
                    "method": (
                        "group_crossfit_advantage_head_v9_3"
                    ),
                    "expert": name,
                    "input_dim": int(features.size(1)),
                    "hidden_dim": self.hidden_dim,
                    "gain_max": self.gain_max,
                    "win_margin": self.win_margin,
                    "crossfit_best_epochs": best_epochs,
                    "full_train_epochs": full_epochs,
                    "oof_diagnostics": diagnostics,
                    "state_dict": cpu_state(model),
                },
                checkpoint,
            )
            prediction_frame[name + "_prediction"] = (
                expert["prediction"].view(-1).tolist()
            )
            prediction_frame[name + "_realized_gain"] = (
                gains.view(-1).tolist()
            )
            prediction_frame[name + "_predicted_gain"] = (
                oof_gain.view(-1).tolist()
            )
            prediction_frame[name + "_win_probability"] = (
                oof_win.view(-1).tolist()
            )
            results[name] = {
                "oof_predicted_gain": oof_gain,
                "oof_win_probability": oof_win,
                "realized_gain": gains,
                "test_predicted_gain": (
                    test_output["predicted_gain"]
                ),
                "test_win_probability": (
                    test_output["win_probability"]
                ),
                "diagnostics": diagnostics,
                "checkpoint": checkpoint,
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pd.DataFrame(history_rows).to_csv(
            self.save_dir / "v93_advantage_fold_history.csv",
            index=False,
        )
        for index, region_name in enumerate(REGION_NAMES):
            prediction_frame["p_" + region_name] = (
                valid_region["region_probs"][:, index].tolist()
            )
        pd.DataFrame(prediction_frame).to_csv(
            self.save_dir / "v93_advantage_oof_predictions.csv",
            index=False,
        )
        return results, valid_region, test_region
