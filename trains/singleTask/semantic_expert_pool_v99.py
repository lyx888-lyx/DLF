"""Frozen V9.3 expert collection with fixed semantic signatures for V9.9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

from .cfcompat_fold_training_v92 import MissingModalityWrapper
from .function_space_features_v92 import (
    _identifier_from_dataset,
    collect_wrapper_function_space,
)
from .model.CachedTailResidualHeadV92 import CachedTailResidualHeadV92
from .model.DLF import DLF
from .model.RoleConditionedDLF import RoleConditionedDLF
from .model.SemanticCostCoachV99 import (
    SIGNATURE_FIELDS,
    SIGNATURE_VERSION,
    role_signature,
    tail_signature,
)
from .oof_group_splits_v92 import canonical_sample_id
from .oof_tail_residual_system_v92 import load_cfcompat_checkpoint

ROLE_INDEX = {"boundary": 2, "positive": 3}
TAIL_NAMES = ("strong_negative", "strong_positive")


def _checkpoint_payload(path: Path):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"), dict
    ):
        raise ValueError(f"checkpoint has no state_dict: {path}")
    return payload


def anchor_checkpoint_from_summary(v92_root: Path) -> Path:
    path = Path(v92_root) / "oof_tail_residual_experts_v92_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"V9.2 summary missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(str(data.get("anchor_checkpoint", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"V9.2 anchor checkpoint missing: {checkpoint}"
        )
    return checkpoint


def default_expert_paths(v9_root: Path, v92_root: Path) -> Dict[str, Path]:
    return {
        "boundary": Path(v9_root)
        / "boundary"
        / "role_conditioned_expert_v9_best.pth",
        "positive": Path(v9_root)
        / "positive"
        / "role_conditioned_expert_v9_best.pth",
        "strong_negative": Path(v92_root)
        / "strong_negative"
        / "oof_tail_residual_expert_v92_best.pth",
        "strong_positive": Path(v92_root)
        / "strong_positive"
        / "oof_tail_residual_expert_v92_best.pth",
    }


def _infer_hidden_dim(state: Mapping[str, torch.Tensor]) -> int:
    key = "adapter.1.weight"
    if key not in state or state[key].dim() != 2:
        raise ValueError(f"cannot infer hidden dimension from state key {key}")
    return int(state[key].size(0))


@torch.no_grad()
def collect_role_semantic_rows(
    model,
    loader,
    device,
    role_index: int,
    top_id_map: Optional[Mapping[str, int]] = None,
):
    model.eval()
    rows = []
    for batch in loader:
        output = model(
            batch["text"].to(device),
            batch["audio"].to(device),
            batch["vision"].to(device),
        )
        indices = batch["index"].view(-1).cpu().tolist()
        fallback_ids = list(batch.get("id", []))
        identifiers = [
            _identifier_from_dataset(
                loader.dataset,
                index,
                fallback_ids[offset] if offset < len(fallback_ids) else index,
            )
            for offset, index in enumerate(indices)
        ]
        prediction = output["prediction"].detach().cpu().view(-1, 1)
        correction = output["correction"].detach().cpu().view(-1, 1)
        region_probs = output["region_probs"].detach().cpu()
        confidence = region_probs[:, int(role_index)].view(-1, 1)
        region_expected = output["region_expected"].detach().cpu().view(-1, 1)
        predicted_error = (
            output["predicted_abs_error"].detach().cpu().view(-1, 1)
        )
        signature = role_signature(
            prediction,
            correction,
            confidence,
            region_probs,
            region_expected,
            predicted_error,
        )
        for offset, index in enumerate(indices):
            sample_id = canonical_sample_id(identifiers[offset])
            row = {
                "sample_index": int(index),
                "sample_id": sample_id,
                "prediction": prediction[offset].clone(),
                "confidence": confidence[offset].clone(),
                "correction": correction[offset].clone(),
                "region_probs": region_probs[offset].clone(),
                "region_expected": region_expected[offset].clone(),
                "predicted_abs_error": predicted_error[offset].clone(),
                "signature": signature[offset].clone(),
            }
            if top_id_map is not None:
                if sample_id not in top_id_map:
                    raise KeyError(
                        f"semantic role sample missing from top OOF: {sample_id}"
                    )
                row["top_index"] = int(top_id_map[sample_id])
            rows.append(row)
    key = "top_index" if top_id_map is not None else "sample_index"
    rows.sort(key=lambda row: row[key])
    return rows


@torch.no_grad()
def collect_tail_semantics(
    model,
    feature: torch.Tensor,
    anchor: torch.Tensor,
    device,
    batch_size: int,
):
    buffers = {
        key: []
        for key in (
            "prediction",
            "correction",
            "raw_correction",
            "applicability_prob",
            "mechanism_probs",
        )
    }
    model.eval()
    for start in range(0, len(feature), int(batch_size)):
        output = model(
            feature[start : start + batch_size].float().to(device),
            anchor[start : start + batch_size].float().to(device),
        )
        for key in buffers:
            buffers[key].append(output[key].detach().cpu())
    result = {key: torch.cat(value, dim=0) for key, value in buffers.items()}
    result["confidence"] = result["applicability_prob"]
    result["signature"] = tail_signature(
        result["prediction"],
        result["correction"],
        result["raw_correction"],
        result["applicability_prob"],
        result["mechanism_probs"],
    )
    return result


class FrozenSemanticExpertPoolV99:
    """Original frozen V9.3 deployment experts with semantic signatures."""

    def __init__(
        self,
        args,
        anchor_checkpoint: Path,
        expert_paths: Mapping[str, Path],
        role_residual_max: float = 0.45,
        tail_residual_max: float = 1.50,
    ) -> None:
        self.args = args
        self.device = args.device
        self.anchor_checkpoint = Path(anchor_checkpoint)
        self.expert_paths = {
            name: Path(path) for name, path in expert_paths.items()
        }
        self.role_residual_max = float(role_residual_max)
        self.tail_residual_max = float(tail_residual_max)
        for path in [self.anchor_checkpoint, *self.expert_paths.values()]:
            if not path.is_file():
                raise FileNotFoundError(path)

    def _new_anchor(self):
        backbone = DLF(self.args).to(self.device)
        wrapper = MissingModalityWrapper(
            backbone,
            self.args.feature_dims[1],
            self.args.feature_dims[2],
        ).to(self.device)
        load_cfcompat_checkpoint(
            wrapper, self.anchor_checkpoint, self.device
        )
        wrapper.eval()
        return wrapper

    def _collect_anchor(self, loader):
        model = self._new_anchor()
        result = collect_wrapper_function_space(model, loader, self.device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def _collect_role(self, name: str, loader):
        payload = _checkpoint_payload(self.expert_paths[name])
        state = payload["state_dict"]
        hidden_dim = _infer_hidden_dim(state)
        model = RoleConditionedDLF(
            self.args,
            hidden_dim=hidden_dim,
            dropout=0.0,
            residual_max=self.role_residual_max,
        ).to(self.device)
        model.load_state_dict(state, strict=True)
        rows = collect_role_semantic_rows(
            model, loader, self.device, ROLE_INDEX[name]
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows

    def _collect_tail(self, name: str, anchor_data):
        payload = _checkpoint_payload(self.expert_paths[name])
        state = payload["state_dict"]
        hidden_dim = _infer_hidden_dim(state)
        feature_dim = int(state["adapter.0.weight"].numel()) - 1
        if feature_dim != int(anchor_data["feature"].size(1)):
            raise RuntimeError(
                f"tail feature dimension mismatch for {name}: "
                f"{feature_dim} vs {anchor_data['feature'].size(1)}"
            )
        model = CachedTailResidualHeadV92(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=0.0,
            residual_max=self.tail_residual_max,
        ).to(self.device)
        model.load_state_dict(state, strict=True)
        result = collect_tail_semantics(
            model,
            anchor_data["feature"],
            anchor_data["anchor"],
            self.device,
            max(1, len(anchor_data["anchor"])),
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def collect(self, loader) -> Dict[str, object]:
        anchor_data = self._collect_anchor(loader)
        result: Dict[str, object] = {
            "sample_ids": list(anchor_data["sample_ids"]),
            "labels": anchor_data["labels"].float(),
            "anchor": anchor_data["anchor"].float(),
            "function_space": anchor_data["feature"].float(),
            "feature_space": anchor_data["feature_space"],
            "feature_keys": list(anchor_data["feature_keys"]),
            "signature_version": SIGNATURE_VERSION,
            "signature_fields": list(SIGNATURE_FIELDS),
            "experts": {},
        }
        for name in ("boundary", "positive"):
            rows = self._collect_role(name, loader)
            ids = [row["sample_id"] for row in rows]
            if ids != result["sample_ids"]:
                raise RuntimeError(
                    f"sample ID mismatch while loading semantic {name}"
                )
            result["experts"][name] = {
                "prediction": torch.cat(
                    [row["prediction"] for row in rows], dim=0
                ).view(-1, 1),
                "confidence": torch.cat(
                    [row["confidence"] for row in rows], dim=0
                ).view(-1, 1),
                "correction": torch.cat(
                    [row["correction"] for row in rows], dim=0
                ).view(-1, 1),
                "signature": torch.stack(
                    [row["signature"] for row in rows], dim=0
                ),
                "checkpoint": str(self.expert_paths[name]),
            }
        for name in TAIL_NAMES:
            values = self._collect_tail(name, anchor_data)
            values["checkpoint"] = str(self.expert_paths[name])
            result["experts"][name] = values
        for name, values in result["experts"].items():
            if values["prediction"].shape != result["anchor"].shape:
                raise RuntimeError(f"prediction shape mismatch for {name}")
            if not torch.isfinite(values["signature"]).all():
                raise FloatingPointError(
                    f"non-finite semantic signature for {name}"
                )
        return result
