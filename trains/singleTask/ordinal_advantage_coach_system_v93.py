"""End-to-end V9.3 coach orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from .advantage_head_crossfit_v93 import AdvantageHeadCrossFitterV93
from .coach_routing_v93 import CoachRouterV93
from .model.OrdinalAdvantageCoachV93 import SPECIALIST_NAMES
from .ordinal_region_coach_system_v93 import (
    OrdinalRegionCoachCrossFitterV93,
)


logger = logging.getLogger("MMSA")


class OrdinalAdvantageCoachTrainerV93:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        oof_cache_path,
        valid_pool,
        test_pool,
        region_hidden_dim=64,
        region_dropout=0.15,
        region_residual_max=2.0,
        ordinal_temperature=0.45,
        region_folds=5,
        region_max_epochs=50,
        advantage_hidden_dim=48,
        advantage_dropout=0.10,
        advantage_gain_max=1.5,
        advantage_folds=5,
        advantage_max_epochs=60,
        early_stop=8,
        learning_rate=3e-4,
        weight_decay=1e-3,
        batch_size=64,
        win_margin=0.02,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.valid = valid_pool
        self.test = test_pool
        for split_name, pool in (
            ("valid", self.valid),
            ("test", self.test),
        ):
            count = len(pool["sample_ids"])
            if (
                pool["labels"].shape != (count, 1)
                or pool["anchor"].shape != (count, 1)
            ):
                raise ValueError(
                    f"invalid {split_name} expert-pool shapes"
                )
            if set(pool["experts"]) != set(SPECIALIST_NAMES):
                raise ValueError(
                    f"{split_name} expert pool is incomplete"
                )
        if self.valid["feature_space"] != self.test["feature_space"]:
            raise ValueError(
                "validation and test feature spaces differ"
            )
        self.region = OrdinalRegionCoachCrossFitterV93(
            args=args,
            save_dir=self.save_dir,
            oof_cache_path=oof_cache_path,
            hidden_dim=region_hidden_dim,
            dropout=region_dropout,
            residual_max=region_residual_max,
            ordinal_temperature=ordinal_temperature,
            folds=region_folds,
            max_epochs=region_max_epochs,
            early_stop=early_stop,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )
        self.advantage_options = {
            "hidden_dim": advantage_hidden_dim,
            "dropout": advantage_dropout,
            "gain_max": advantage_gain_max,
            "folds": advantage_folds,
            "max_epochs": advantage_max_epochs,
            "early_stop": early_stop,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "win_margin": win_margin,
        }

    def train_all(self):
        region_model, region_meta = self.region.fit()
        advantage = AdvantageHeadCrossFitterV93(
            args=self.args,
            save_dir=self.save_dir,
            valid_pool=self.valid,
            test_pool=self.test,
            region_collector=self.region,
            region_model=region_model,
            **self.advantage_options,
        )
        advantage_results, valid_region, test_region = (
            advantage.fit()
        )
        router = CoachRouterV93(
            args=self.args,
            metrics_fn=self.metrics_fn,
            save_dir=self.save_dir,
            valid_pool=self.valid,
            test_pool=self.test,
        )
        route_policy = router.calibrate(
            advantage_results, valid_region
        )
        summary = router.evaluate(
            region_meta,
            advantage_results,
            valid_region,
            test_region,
            route_policy,
        )
        logger.info(
            "V9.3 TEST anchor=%.6f anchor-score=%.6f "
            "hard-ordinal=%.6f advantage=%.6f true-region=%.6f",
            summary["test_results"]["anchor"]["MAE"],
            summary["test_results"][
                "anchor_score_region_valid_selected"
            ]["MAE"],
            summary["test_results"][
                "hard_ordinal_valid_selected"
            ]["MAE"],
            summary["test_results"][
                "ordinal_advantage_coach_valid_selected"
            ]["MAE"],
            summary["test_results"][
                "true_region_expert_policy"
            ]["MAE"],
        )
        return summary
