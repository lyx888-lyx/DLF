"""Deterministic signed counterfactual residual recovery utilities for Stage 4A."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cf_compat_kd_utils import compatibility_for_modes, stable_average_ranks
from .fixed_kd_utils import capture_rng_state, checkpoint_sha256, restore_rng_state
from .missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    mode_to_mask,
    regression_metrics,
    validation_objective,
)
from .model.DLF import DLF


CFRR_CACHE_VERSION = "cfrr_v1"
MODE_ORDER = ("LA", "LV", "L")
MODE_TO_INDEX = {mode: index for index, mode in enumerate(MODE_ORDER)}
RESIDUAL_FORMULA = "residual_m=evaluator_LAV_pred-evaluator_m_pred"
SCALE_FORMULA = "population_std(residual_m,ddof=0)"
SOURCE_COLUMNS = (
    "sample_index", "sample_id", "label",
    "evaluator_LAV_pred", "evaluator_LA_pred", "evaluator_LV_pred", "evaluator_L_pred",
    "delta_LA", "delta_LV", "delta_L",
    "compat_LA", "compat_LV", "compat_L",
)
RESIDUAL_COLUMNS = SOURCE_COLUMNS + ("residual_LA", "residual_LV", "residual_L")


def residual_cache_paths(root, dataset, seed=1111):
    directory = Path(root) / "counterfactual_residual" / CFRR_CACHE_VERSION / dataset / "seed{}".format(seed)
    return {
        "directory": directory,
        "csv": directory / "train_counterfactual_residual.csv",
        "config": directory / "cfrr_config.json",
        "summary": directory / "cfrr_summary.json",
        "bins": directory / "cfrr_mode_bins.csv",
        "manifest": directory / "cfrr_cache_manifest.json",
    }


def rng_states_equal(first, second):
    if not torch.equal(first["torch"], second["torch"]):
        return False
    if first["python"] != second["python"]:
        return False
    a, b = first["numpy"], second["numpy"]
    if a[0] != b[0] or not np.array_equal(a[1], b[1]) or a[2:] != b[2:]:
        return False
    if "cuda" in first or "cuda" in second:
        if len(first.get("cuda", [])) != len(second.get("cuda", [])):
            return False
        if not all(torch.equal(x, y) for x, y in zip(first.get("cuda", []), second.get("cuda", []))):
            return False
    return True


def zero_initialized_residual_heads(fusion_dim):
    """Create the three fixed Linear heads without advancing any RNG."""
    before = capture_rng_state()
    try:
        heads = nn.ModuleDict({mode: nn.Linear(int(fusion_dim), 1, bias=True) for mode in MODE_ORDER})
        for head in heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        return heads
    finally:
        restore_rng_state(before)


def modes_from_any_mask(mask):
    mapping = {
        (1, 1, 1): "LAV", (1, 1, 0): "LA",
        (1, 0, 1): "LV", (1, 0, 0): "L",
    }
    try:
        return [mapping[tuple(row)] for row in mask.detach().cpu().to(torch.int64).tolist()]
    except KeyError as error:
        raise ValueError("Unsupported residual modality mask: {}".format(error))


class CounterfactualResidualStudent(nn.Module):
    """Backward-compatible MissingModalityWrapper plus deterministic signed residual heads."""

    def __init__(self, base_student, mode_scales):
        super().__init__()
        if not isinstance(base_student, MissingModalityWrapper):
            raise TypeError("CFRR must wrap MissingModalityWrapper(DLF).")
        fusion_dim = int(base_student.backbone.out_layer.in_features)
        scales = torch.as_tensor([float(mode_scales[mode]) for mode in MODE_ORDER], dtype=torch.float32)
        if not torch.isfinite(scales).all() or torch.any(scales <= 1e-8):
            raise ValueError("Every mode scale must be finite and greater than 1e-8.")
        self.base_student = base_student
        self.residual_heads = zero_initialized_residual_heads(fusion_dim)
        self.register_buffer("mode_scales", scales)

    def scale_for(self, mode):
        if mode not in MODE_TO_INDEX:
            raise KeyError("LAV has no residual scale.")
        return self.mode_scales[MODE_TO_INDEX[mode]]

    def forward(self, text, audio, vision, modality_mask):
        output = self.base_student(text, audio, vision, modality_mask)
        if "fusion_feature" not in output:
            raise KeyError("DLF must expose the exact final fusion_feature.")
        base = output["output_logit"]
        feature = output["fusion_feature"]
        modes = modes_from_any_mask(modality_mask)
        predicted_z = torch.zeros_like(base)
        predicted_residual = torch.zeros_like(base)
        for mode in MODE_ORDER:
            selector = torch.as_tensor([item == mode for item in modes], device=base.device).view(-1, 1)
            mode_z = self.residual_heads[mode](feature)
            predicted_z = predicted_z + mode_z * selector.to(mode_z)
            predicted_residual = predicted_residual + mode_z * self.scale_for(mode).to(mode_z) * selector.to(mode_z)
        corrected = base + predicted_residual
        result = dict(output)
        result.update({
            "base_output_logit": base,
            "predicted_residual_z": predicted_z,
            "predicted_residual": predicted_residual,
            "corrected_output_logit": corrected,
            "residual_mode": tuple(modes),
            "output_logit": corrected,
        })
        return result


def build_residual_student(args, gate3_checkpoint, mode_scales):
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(gate3_checkpoint, map_location=args.device), strict=True)
    base = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    return CounterfactualResidualStudent(base, mode_scales).to(args.device)


def load_residual_student_for_eval(args, checkpoint):
    """Load a deployable checkpoint without Teacher, evaluator, or any train cache."""
    state = torch.load(checkpoint, map_location=args.device)
    if any(key.startswith(("teacher.", "evaluator.")) for key in state):
        raise ValueError("Student checkpoint contains forbidden frozen-model parameters.")
    backbone = DLF(args).to(args.device)
    base = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    model = CounterfactualResidualStudent(base, {mode: 1.0 for mode in MODE_ORDER}).to(args.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _stage3_manifest_path(result_root):
    return Path(result_root) / "missing_baseline" / "cf_compat_kd_v1" / "benchmark_multiseed" / "checkpoint_manifest.csv"


def locate_stage3_seed1111_cache(result_root):
    """Resolve source cache and evaluator from the locked Stage 3 manifest, never by guessing."""
    manifest_path = _stage3_manifest_path(result_root)
    if not manifest_path.is_file():
        raise FileNotFoundError("Stage 3 checkpoint manifest is absent: {}".format(manifest_path))
    manifest = pd.read_csv(manifest_path)
    selected = manifest.loc[manifest.Seed.astype(int).eq(1111)]
    if len(selected) != 1 or str(selected.iloc[0].Status) != "locked_existing":
        raise ValueError("Stage 3 seed1111 must be a unique locked manifest row.")
    row = selected.iloc[0]
    cache = Path(str(row.CachePath)); evaluator = Path(str(row.EvaluatorCheckpoint))
    if not cache.is_file() or checkpoint_sha256(cache) != str(row.CacheSHA256):
        raise ValueError("Locked Stage 3 compatibility cache SHA mismatch.")
    if not evaluator.is_file() or checkpoint_sha256(evaluator) != str(row.EvaluatorSHA256):
        raise ValueError("Locked Stage 3 evaluator SHA mismatch.")
    if "diagnostic" in str(evaluator) or "best_test" in str(evaluator):
        raise ValueError("CFRR source evaluator must be validation-best ModDrop.")
    config_path = cache.parent / "cf_compat_config.json"
    config = json.loads(config_path.read_text())
    if config.get("source") != "train_only" or int(config.get("train_sample_count", -1)) != 1284:
        raise ValueError("Stage 3 source cache is not the audited train-only cache.")
    if config.get("evaluator_sha256") != str(row.EvaluatorSHA256):
        raise ValueError("Stage 3 cache/evaluator metadata binding mismatch.")
    return {
        "manifest": manifest_path, "cache": cache, "cache_sha256": str(row.CacheSHA256),
        "evaluator": evaluator, "evaluator_sha256": str(row.EvaluatorSHA256),
        "compat_config": config_path,
    }


def signed_distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Residual statistics require finite non-empty values.")
    q = np.quantile(values, [.1, .25, .5, .75, .9, .95, .99])
    return {
        "mean": float(values.mean()), "std": float(values.std(ddof=0)),
        "mean_abs": float(np.abs(values).mean()), "min": float(values.min()),
        "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "p90": float(q[4]), "p95": float(q[5]),
        "p99": float(q[6]), "max": float(values.max()),
        "positive_fraction": float((values > 0).mean()),
        "negative_fraction": float((values < 0).mean()),
        "zero_fraction": float((values == 0).mean()),
    }


def derive_residual_frame(source_frame):
    missing = [column for column in SOURCE_COLUMNS if column not in source_frame]
    if missing:
        raise ValueError("Stage 3 cache misses required columns: {}".format(missing))
    frame = source_frame.loc[:, SOURCE_COLUMNS].copy().sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != 1284 or frame.sample_index.nunique() != 1284:
        raise ValueError("CFRR requires exactly 1284 unique train samples.")
    if not np.array_equal(frame.sample_index.to_numpy(dtype=int), np.arange(1284)):
        raise ValueError("Residual cache indices are not the complete train index set.")
    for mode in MODE_ORDER:
        residual = frame.evaluator_LAV_pred.to_numpy(np.float64) - frame["evaluator_{}_pred".format(mode)].to_numpy(np.float64)
        delta = frame["delta_{}".format(mode)].to_numpy(np.float64)
        if not np.allclose(np.abs(residual), delta, rtol=0.0, atol=1e-7):
            raise ValueError("abs(signed residual) != Stage 3 delta for {}.".format(mode))
        frame["residual_{}".format(mode)] = residual
    return frame.loc[:, RESIDUAL_COLUMNS]


def _cache_bin_rows(frame):
    rows = []
    for mode in MODE_ORDER:
        local = frame[["compat_{}".format(mode), "residual_{}".format(mode)]].copy()
        local.columns = ["compatibility", "residual"]
        local = local.sort_values("compatibility", kind="mergesort").reset_index(drop=True)
        local["quartile"] = pd.qcut(np.arange(len(local)), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        for quartile, part in local.groupby("quartile", observed=False):
            rows.append({
                "Mode": mode, "CompatibilityQuartile": str(quartile), "Count": int(len(part)),
                "MeanCompatibility": float(part.compatibility.mean()),
                "MeanResidual": float(part.residual.mean()),
                "MeanAbsResidual": float(part.residual.abs().mean()),
            })
    return rows


def build_residual_cache(result_root, dataset="mosi", seed=1111):
    if int(seed) != 1111:
        raise ValueError("Stage 4A is locked to seed1111.")
    before = capture_rng_state()
    source = locate_stage3_seed1111_cache(result_root)
    frame = derive_residual_frame(pd.read_csv(source["cache"]))
    summaries = {}; scales = {}
    for mode in MODE_ORDER:
        stats = signed_distribution(frame["residual_{}".format(mode)])
        scale = float(stats["std"])
        if not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError("Residual scale for {} is non-finite or too small.".format(mode))
        scales[mode] = scale
        summaries[mode] = {"residual": stats, "scale": scale, "scale_formula": SCALE_FORMULA}
    paths = residual_cache_paths(result_root, dataset, seed)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    config = {
        "Method": "Stage 4A Deterministic Counterfactual Residual Recovery",
        "Version": CFRR_CACHE_VERSION, "Dataset": dataset, "Seed": int(seed),
        "SourceCompatibilityCache": str(source["cache"]),
        "SourceCompatibilityCacheSHA256": source["cache_sha256"],
        "EvaluatorCheckpoint": str(source["evaluator"]), "EvaluatorSHA256": source["evaluator_sha256"],
        "TrainSampleCount": int(len(frame)), "ResidualFormula": RESIDUAL_FORMULA,
        "ScaleFormula": SCALE_FORMULA, "ModeScales": scales, "CreatedFromTrainOnly": True,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["ConfigSHA256"] = hashlib.sha256(canonical.encode()).hexdigest()
    frame.to_csv(paths["csv"], index=False)
    config["ResidualCacheSHA256"] = checkpoint_sha256(paths["csv"])
    paths["config"].write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    paths["summary"].write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(_cache_bin_rows(frame)).to_csv(paths["bins"], index=False)
    paths["manifest"].write_text(json.dumps({
        "Seed": int(seed), "ResidualCache": str(paths["csv"]),
        "ResidualCacheSHA256": config["ResidualCacheSHA256"],
        "Config": str(paths["config"]), "ConfigSHA256": config["ConfigSHA256"],
        "SourceCompatibilityCache": str(source["cache"]),
        "SourceCompatibilityCacheSHA256": source["cache_sha256"],
        "EvaluatorCheckpoint": str(source["evaluator"]), "EvaluatorSHA256": source["evaluator_sha256"],
        "TrainSampleCount": 1284, "CreatedFromTrainOnly": True,
    }, indent=2, sort_keys=True) + "\n")
    after = capture_rng_state()
    if not rng_states_equal(before, after):
        raise RuntimeError("Residual cache derivation changed an RNG state.")
    return paths, config, frame


def load_residual_cache(result_root, dataset="mosi", seed=1111):
    paths = residual_cache_paths(result_root, dataset, seed)
    for name in ("csv", "config", "manifest"):
        if not paths[name].is_file():
            raise FileNotFoundError("CFRR cache artifact missing: {}".format(paths[name]))
    config = json.loads(paths["config"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    if not config.get("CreatedFromTrainOnly") or int(config.get("TrainSampleCount", -1)) != 1284:
        raise ValueError("CFRR cache is not the audited train-only cache.")
    if checkpoint_sha256(paths["csv"]) != config.get("ResidualCacheSHA256"):
        raise ValueError("CFRR residual cache SHA mismatch.")
    if checkpoint_sha256(config["SourceCompatibilityCache"]) != config["SourceCompatibilityCacheSHA256"]:
        raise ValueError("Stage 3 source cache changed after CFRR derivation.")
    if checkpoint_sha256(config["EvaluatorCheckpoint"]) != config["EvaluatorSHA256"]:
        raise ValueError("Stage 3 evaluator changed after CFRR derivation.")
    if manifest.get("ResidualCacheSHA256") != config["ResidualCacheSHA256"]:
        raise ValueError("CFRR config/manifest binding mismatch.")
    frame = derive_residual_frame(pd.read_csv(paths["csv"]))
    scales = {mode: float(config["ModeScales"][mode]) for mode in MODE_ORDER}
    for mode, scale in scales.items():
        actual = float(frame["residual_{}".format(mode)].std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-8 or not np.isclose(scale, actual, rtol=0, atol=1e-12):
            raise ValueError("CFRR mode scale is not the train population std for {}.".format(mode))
    return frame, {int(row.sample_index): row._asdict() for row in frame.itertuples(index=False)}, scales, config


def residual_targets_for_modes(cache_by_index, indices, modes, scales, device, dtype):
    compatibility = compatibility_for_modes(cache_by_index, indices, modes, device, dtype)
    residual = []; target_z = []
    for index, mode in zip(indices, modes):
        if mode not in MODE_TO_INDEX or int(index) not in cache_by_index:
            raise KeyError("Invalid CFRR target binding index={} mode={}".format(index, mode))
        raw = float(cache_by_index[int(index)]["residual_{}".format(mode)])
        residual.append(raw); target_z.append(raw / float(scales[mode]))
    residual = torch.as_tensor(residual, device=device, dtype=dtype)
    target_z = torch.as_tensor(target_z, device=device, dtype=dtype)
    if not torch.isfinite(residual).all() or not torch.isfinite(target_z).all():
        raise FloatingPointError("CFRR signed targets must be finite.")
    return compatibility.detach(), residual.detach(), target_z.detach()


def complementary_residual_loss(predicted_z, target_z, compatibility):
    prediction = predicted_z.view(-1)
    target = target_z.detach().view(-1).to(prediction)
    weight = (1.0 - compatibility.detach().view(-1)).to(prediction)
    each = F.smooth_l1_loss(prediction, target, reduction="none")
    if each.shape != weight.shape:
        raise ValueError("Residual loss and complementary weights differ in shape.")
    total = torch.sum(weight * each) / (torch.sum(weight) + 1e-8)
    if not torch.isfinite(total):
        raise FloatingPointError("Complementary residual loss is non-finite.")
    return total, each, weight


def _pearson(first, second):
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if len(first) < 2 or first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _spearman(first, second):
    return _pearson(stable_average_ranks(first), stable_average_ranks(second))


def residual_diagnostic_rows(records, seed, epoch, method):
    data = pd.DataFrame(records)
    if data.empty or not np.isfinite(data.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Residual diagnostics require finite records.")
    predicted_stats = signed_distribution(data.predicted_residual)
    target_stats = signed_distribution(data.target_residual)
    summary = {
        "Seed": seed, "Epoch": epoch, "Method": method, "SampleCount": int(len(data)),
        **{"target_residual_{}".format(k): v for k, v in target_stats.items()},
        **{"predicted_residual_{}".format(k): v for k, v in predicted_stats.items()},
    }
    mode_rows = []
    for mode in MODE_ORDER:
        local = data.loc[data["mode"].eq(mode)]
        target = local.target_residual.to_numpy(float); prediction = local.predicted_residual.to_numpy(float)
        base_error = np.abs(local.base_prediction - local.label)
        corrected_error = np.abs(local.corrected_prediction - local.label)
        reduction = base_error - corrected_error
        target_variance = float(np.var(target))
        mode_rows.append({
            "Seed": seed, "Epoch": epoch, "Method": method, "Mode": mode, "Count": int(len(local)),
            **{"Target_{}".format(k): v for k, v in signed_distribution(target).items()},
            **{"Predicted_{}".format(k): v for k, v in signed_distribution(prediction).items()},
            "ResidualMAE": float(np.mean(np.abs(prediction - target))),
            "ResidualRMSE": float(np.sqrt(np.mean(np.square(prediction - target)))),
            "Pearson": _pearson(prediction, target), "Spearman": _spearman(prediction, target),
            "SignAccuracy": float(np.mean(np.sign(prediction) == np.sign(target))),
            "ExplainedVariance": float(1.0 - np.var(target - prediction) / target_variance) if target_variance > 0 else 0.0,
            "BaseLabelMAE": float(base_error.mean()), "CorrectedLabelMAE": float(corrected_error.mean()),
            "MeanErrorReduction": float(reduction.mean()),
            "fraction_corrected_better": float((reduction > 0).mean()),
            "fraction_corrected_worse": float((reduction < 0).mean()),
            "fraction_equal": float((reduction == 0).mean()),
        })
    ordered = data.sort_values(["compatibility", "sample_index"], kind="mergesort").copy()
    ordered["quartile"] = pd.qcut(np.arange(len(ordered)), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    quartiles = []
    for quartile, local in ordered.groupby("quartile", observed=False):
        target = local.target_residual.to_numpy(float); prediction = local.predicted_residual.to_numpy(float)
        base_error = np.abs(local.base_prediction - local.label)
        corrected_error = np.abs(local.corrected_prediction - local.label)
        row = {
            "Seed": seed, "Epoch": epoch, "Method": method, "Quartile": str(quartile), "Count": int(len(local)),
            "MeanCompatibility": float(local.compatibility.mean()), "MeanTargetResidual": float(target.mean()),
            "MeanAbsTargetResidual": float(np.abs(target).mean()), "MeanPredictedResidual": float(prediction.mean()),
            "MeanAbsPredictedResidual": float(np.abs(prediction).mean()),
            "ResidualMAE": float(np.mean(np.abs(prediction - target))),
            "SignAccuracy": float(np.mean(np.sign(prediction) == np.sign(target))),
            "BaseLabelMAE": float(base_error.mean()), "CorrectedLabelMAE": float(corrected_error.mean()),
            "MeanErrorReduction": float((base_error - corrected_error).mean()),
        }
        row.update({"{}_count".format(mode): int(local["mode"].eq(mode).sum()) for mode in MODE_ORDER})
        quartiles.append(row)
    return summary, mode_rows, quartiles


def evaluate_residual_modes(model, dataloader, device, criterion):
    """Student-only corrected/base evaluation for LAV/LA/LV/L."""
    model.eval()
    collected = {kind: {mode: {"prediction": [], "label": [], "loss": []} for mode in ("LAV",) + MODE_ORDER}
                 for kind in ("corrected", "base")}
    with torch.no_grad():
        for batch in dataloader:
            text, audio, vision = batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device)
            labels = batch["labels"]["M"].to(device).view(-1, 1)
            for mode in ("LAV",) + MODE_ORDER:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                output = model(text, audio, vision, mask)
                for kind, key in (("corrected", "output_logit"), ("base", "base_output_logit")):
                    prediction = output[key]
                    collected[kind][mode]["prediction"].append(prediction.detach().cpu())
                    collected[kind][mode]["label"].append(labels.detach().cpu())
                    collected[kind][mode]["loss"].append(float(criterion(prediction, labels)))
    result = {"corrected": {}, "base": {}}
    for kind in result:
        for mode, values in collected[kind].items():
            metrics = regression_metrics(torch.cat(values["prediction"]), torch.cat(values["label"]))
            metrics["Loss"] = float(np.mean(values["loss"]))
            result[kind][mode] = metrics
    return result


def correction_gaps(model, dataloader, device):
    model.eval(); totals = {mode: 0.0 for mode in MODE_ORDER}; count = 0
    with torch.no_grad():
        for batch in dataloader:
            text, audio, vision = batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device)
            for mode in MODE_ORDER:
                mask = mode_to_mask(mode, text.size(0), device, audio.dtype)
                output = model(text, audio, vision, mask)
                totals[mode] += float(torch.abs(output["output_logit"] - output["base_output_logit"]).sum())
            count += int(text.size(0))
    if not count:
        raise RuntimeError("Correction-gap loader is empty.")
    return {"Gap_{}".format(mode): totals[mode] / count for mode in MODE_ORDER}


def prediction_rows(model, loader, device):
    model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision = batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device)
            labels = batch["labels"]["M"].to(device).view(-1)
            values = {}
            for mode in ("LAV",) + MODE_ORDER:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                output = model(text, audio, vision, mask)
                values[mode] = {
                    "base": output["base_output_logit"].view(-1).cpu().numpy(),
                    "corrected": output["output_logit"].view(-1).cpu().numpy(),
                    "residual": output["predicted_residual"].view(-1).cpu().numpy(),
                }
            identifiers = list(batch["id"])
            for offset, index in enumerate(batch["index"].view(-1).cpu().numpy().astype(int)):
                row = {"sample_index": int(index), "sample_id": str(identifiers[offset]),
                       "label": float(labels[offset]), "mode": "all_modes"}
                for mode in ("LAV",) + MODE_ORDER:
                    row["{}_base_pred".format(mode)] = float(values[mode]["base"][offset])
                    row["{}_corrected_pred".format(mode)] = float(values[mode]["corrected"][offset])
                    row["{}_predicted_residual".format(mode)] = float(values[mode]["residual"][offset])
                rows.append(row)
    return pd.DataFrame(rows).sort_values("sample_index", kind="mergesort")


def flatten_evaluation(evaluation, prefix):
    values = {}
    for kind in ("corrected", "base"):
        for mode, metrics in evaluation[kind].items():
            for metric, value in metrics.items():
                values["{}_{}_{}_{}".format(kind, prefix, mode, metric)] = float(value)
    return values


def evaluation_objectives(evaluation):
    return validation_objective(evaluation["corrected"]), validation_objective(evaluation["base"])
