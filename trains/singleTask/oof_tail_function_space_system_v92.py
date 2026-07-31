"""Function-space-safe V9.2 tail trainer."""

from __future__ import annotations

import logging

import torch

from .function_space_features_v92 import (
    FEATURE_KEYS,
    FEATURE_SPACE_VERSION,
    collect_wrapper_function_space,
)
from .oof_tail_residual_system_v92 import (
    OOFTailResidualTrainerV92,
    load_cfcompat_checkpoint,
)

logger = logging.getLogger("MMSA")


class OOFTailFunctionSpaceTrainerV92(OOFTailResidualTrainerV92):
    """Use aligned auxiliary prediction logits for OOF and full-data features."""

    def _select_and_collect_anchor(self, valid_loader, test_loader):
        candidates = []
        for index, path in enumerate(self.teacher_paths):
            model = self._new_wrapper()
            _, mode = load_cfcompat_checkpoint(model, path, self.args.device)
            valid = collect_wrapper_function_space(model, valid_loader, self.args.device)
            mae = float(torch.abs(valid["anchor"] - valid["labels"]).mean().item())
            candidates.append((mae, index, valid, mode))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _, anchor_index, valid, load_mode = min(candidates, key=lambda row: row[0])
        model = self._new_wrapper()
        load_cfcompat_checkpoint(model, self.teacher_paths[anchor_index], self.args.device)
        test = collect_wrapper_function_space(model, test_loader, self.args.device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "V9.2 function-space anchor index=%d valid_mae=%.6f path=%s mode=%s",
            anchor_index,
            float(torch.abs(valid["anchor"] - valid["labels"]).mean().item()),
            self.teacher_paths[anchor_index],
            load_mode,
        )
        return anchor_index, valid, test, load_mode

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_space = self.oof.get("feature_space")
        cache_keys = tuple(self.oof.get("feature_keys", ()))
        if cache_space != FEATURE_SPACE_VERSION or cache_keys != FEATURE_KEYS:
            raise RuntimeError(
                "OOF cache feature schema mismatch: "
                f"space={cache_space!r}, keys={cache_keys!r}"
            )
        if self.feature_dim != len(FEATURE_KEYS):
            raise RuntimeError("unexpected V9.2 function-space feature dimension")
