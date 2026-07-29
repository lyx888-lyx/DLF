"""Staged checkpoint selection for decoupled region mixture V9.1.

This class keeps the V9.1 model/loss/policy definitions, but selects the expert
checkpoint only by standalone direct-mixture Validation performance. The compact
committee/beta/gamma attribution search is run once after the expert checkpoint
is frozen.
"""

from __future__ import annotations

import json

import pandas as pd
import torch

from .decoupled_region_mixture_system_v91 import (
    LOGGER,
    DecoupledRegionMixtureTrainerV91,
    _cpu_state_dict,
)


class StagedDecoupledRegionMixtureTrainerV91(
    DecoupledRegionMixtureTrainerV91
):
    """Separate expert checkpoint selection from final policy selection."""

    def train(self, model, dataloaders):
        model.freeze_legacy()
        model.mixture_blend_logit.requires_grad_(False)
        optimizer = self._optimizer(model)

        history = []
        valid_committees = self._committee_predictions("valid")
        best_expert = None
        last_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_row = self._train_epoch(
                model, dataloaders["train"], optimizer
            )
            valid = self.collect(model, dataloaders["valid"], "valid")
            direct_metrics = self._policy_metrics(
                valid["mixture_value"],
                valid["labels"],
                valid["sample_ids"],
            )
            direct_objective = self._selection_objective(direct_metrics)
            row = {
                "epoch": epoch,
                **train_row,
                "direct_mixture_objective": direct_objective,
                "direct_mixture_mae": direct_metrics["mae"],
                "direct_mixture_worst_region_mae": direct_metrics[
                    "worst_region_mae"
                ],
                "direct_mixture_ordinary_positive_mae": direct_metrics[
                    "ordinary_positive_mae"
                ],
                "direct_mixture_fold_mae_std": direct_metrics["fold_mae_std"],
            }
            history.append(row)
            LOGGER.info(
                "V9.1 expert epoch=%d direct Valid(MAE=%.4f worst=%.4f "
                "op=%.4f fold_std=%.4f)",
                epoch,
                direct_metrics["mae"],
                direct_metrics["worst_region_mae"],
                direct_metrics["ordinary_positive_mae"],
                direct_metrics["fold_mae_std"],
            )

            improved = (
                best_expert is None
                or direct_objective < best_expert["objective"] - 1e-6
                or (
                    abs(
                        direct_objective - best_expert["objective"]
                    )
                    <= 1e-6
                    and direct_metrics["mae"]
                    < best_expert["direct_metrics"]["mae"] - 1e-6
                )
            )
            if improved:
                best_expert = {
                    "epoch": epoch,
                    "objective": float(direct_objective),
                    "direct_metrics": {
                        key: value
                        for key, value in direct_metrics.items()
                        if key != "region_rows"
                    },
                    "state": _cpu_state_dict(model),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.early_stop:
                break

        if best_expert is None:
            raise RuntimeError("V9.1 completed no expert-training epochs.")

        model.load_state_dict(best_expert["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        selected, policy_rows, _, _ = self._select_policy(
            valid, valid_committees, allow_trained=True
        )

        # Add the frozen final policy to each history row so the existing audit
        # remains backward-compatible while checkpoint selection stays direct.
        for row in history:
            row.update(
                {
                    "selected_source": selected["source"],
                    "selected_mae": selected["mae"],
                    "selected_beta": selected["beta"],
                    "selected_gamma": selected["gamma"],
                }
            )
        pd.DataFrame(history).to_csv(
            self.save_dir / "v91_training_history.csv", index=False
        )

        serializable_rows = []
        for row in policy_rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            serializable_rows.append(value)
        pd.DataFrame(serializable_rows).to_csv(
            self.save_dir / "v91_valid_policy_search.csv", index=False
        )

        best = {
            "epoch": int(best_expert["epoch"]),
            "expert_objective": float(best_expert["objective"]),
            "expert_direct_metrics": dict(best_expert["direct_metrics"]),
            "objective": float(selected["objective"]),
            "state": best_expert["state"],
            "policy": dict(selected),
        }
        torch.save(
            best,
            self.save_dir / "decoupled_region_mixture_v91_best.pth",
        )
        LOGGER.info(
            "V9.1 frozen expert epoch=%d; selected source=%s committee=%s "
            "beta=%.2f gamma=%.2f Valid(MAE=%.4f worst=%.4f op=%.4f)",
            best["epoch"],
            selected["source"],
            selected["committee"],
            selected["beta"],
            selected["gamma"],
            selected["mae"],
            selected["worst_region_mae"],
            selected["ordinary_positive_mae"],
        )
        return best
