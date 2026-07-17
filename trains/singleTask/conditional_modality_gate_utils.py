"""Stage 7B Conditional Modality Utility Gating helpers."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .cf_compat_kd_utils import CACHE_COLUMNS, cache_paths
from .missing_utils import apply_moddrop_tokens


VARIANTS = ("identity_replay", "utility_gate", "utility_gate_matched")
MODALITIES = ("A", "V")
MODES = ("LAV", "LA", "LV", "L")
QUALIFICATION_MINIMUMS = {
    "train_positive": 32, "train_negative": 32,
    "valid_positive": 8, "valid_negative": 8,
}


def load_locked_compatibility(root, dataset, expected_evaluator_sha):
    """Read the historical seed1111 Stage3 cache without requiring newer optional flags."""
    paths = cache_paths(root, dataset)
    frame = pd.read_csv(paths["csv"])
    config = __import__("json").loads(paths["config"].read_text())
    if (
        config.get("version") != "cf_compat_v1"
        or config.get("source") != "train_only"
        or int(config.get("train_sample_count", -1)) != 1284
        or config.get("evaluator_sha256") != expected_evaluator_sha
        or list(frame.columns) != list(CACHE_COLUMNS)
        or len(frame) != 1284
        or frame.sample_index.nunique() != 1284
    ):
        raise ValueError("Locked Stage3 compatibility cache binding failed.")
    return frame, {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }


def build_utility_groups(stage7a_directory):
    """Build stable four-way-sign utility labels from Stage7A raw CSVs."""
    directory = Path(stage7a_directory)
    gains = pd.read_csv(directory / "sample_modality_gains.csv")
    shuffle = pd.read_csv(directory / "shuffle_sample_metrics.csv")
    if set(gains.Split) != {"train", "valid"} or set(shuffle.Split) != {"train", "valid"}:
        raise ValueError("Stage7A utility sources must contain train/valid only.")
    shuffled = shuffle.groupby(
        ["State", "Split", "SampleIndex", "SampleID", "Label"],
        as_index=False, sort=True,
    ).mean(numeric_only=True)
    rows = []
    for modality in MODALITIES:
        state_frames = []
        for state, prefix in (
            ("gate3_init", "Gate3"), ("cfcompat_best_valid", "CFCompat")
        ):
            gain = gains[
                gains.State.eq(state) & gains.Modality.eq(modality)
            ][["Split", "SampleIndex", "SampleID", "Label", "Gain"]].rename(
                columns={"Gain": "{}_Gain".format(prefix)}
            )
            damage = shuffled[shuffled.State.eq(state)][[
                "Split", "SampleIndex", "SampleID", "Label",
                "ShuffleDamage_{}".format(modality),
            ]].rename(columns={
                "ShuffleDamage_{}".format(modality):
                "{}_ShuffleDamage".format(prefix)
            })
            state_frames.append(gain.merge(
                damage, on=["Split", "SampleIndex", "SampleID", "Label"],
                validate="one_to_one",
            ))
        local = state_frames[0].merge(
            state_frames[1],
            on=["Split", "SampleIndex", "SampleID", "Label"],
            validate="one_to_one",
        )
        evidence = [
            "Gate3_Gain", "Gate3_ShuffleDamage",
            "CFCompat_Gain", "CFCompat_ShuffleDamage",
        ]
        positive = (local[evidence] > 0).all(axis=1)
        negative = (local[evidence] <= 0).all(axis=1)
        local["Modality"] = modality
        local["UtilityGroup"] = np.where(
            positive, "positive", np.where(negative, "negative", "ambiguous")
        )
        local["UtilityTarget"] = np.where(
            positive, 1.0, np.where(negative, 0.0, np.nan)
        )
        local["UsedForTraining"] = (
            local.Split.eq("train") & local.UtilityGroup.ne("ambiguous")
        )
        rows.append(local)
    frame = pd.concat(rows, ignore_index=True).sort_values(
        ["Modality", "Split", "SampleIndex"], kind="mergesort"
    ).reset_index(drop=True)
    if frame.Split.eq("test").any():
        raise RuntimeError("Utility artifact contains test.")
    return frame


def utility_group_summary(frame):
    rows = []
    evidence = [
        "Gate3_Gain", "Gate3_ShuffleDamage",
        "CFCompat_Gain", "CFCompat_ShuffleDamage",
    ]
    for (modality, split, group), local in frame.groupby(
        ["Modality", "Split", "UtilityGroup"], sort=True
    ):
        rows.append({
            "Modality": modality, "Split": split, "UtilityGroup": group,
            "Count": len(local),
            "Fraction": len(local) / float((frame.Modality.eq(modality) & frame.Split.eq(split)).sum()),
            **{"Mean_{}".format(column): float(local[column].mean()) for column in evidence},
        })
    return pd.DataFrame(rows)


def qualification_report(frame):
    report = {}
    for modality in MODALITIES:
        local = frame[frame.Modality.eq(modality)]
        counts = {
            "{}_{}".format(split, group): int(
                (local.Split.eq(split) & local.UtilityGroup.eq(group)).sum()
            )
            for split in ("train", "valid")
            for group in ("positive", "negative", "ambiguous")
        }
        passed = all(counts[key] >= value for key, value in QUALIFICATION_MINIMUMS.items())
        report[modality] = {**counts, "qualified": bool(passed)}
    return report


def reliable_target_table(groups, split="train"):
    if split != "train":
        raise ValueError("Only train utility targets may enter optimization.")
    table = {}
    for modality in MODALITIES:
        local = groups[
            groups.Modality.eq(modality) & groups.Split.eq(split)
        ]
        table[modality] = {
            int(row.SampleIndex): (
                None if row.UtilityGroup == "ambiguous" else int(row.UtilityTarget)
            )
            for row in local.itertuples()
        }
    return table


def targets_for_indices(table, indices, modality, device, dtype):
    values = []
    for index in indices:
        if int(index) not in table[modality]:
            raise KeyError("Missing train utility binding.")
        value = table[modality][int(index)]
        values.append(-1.0 if value is None else float(value))
    return torch.tensor(values, device=device, dtype=dtype)


def utility_bce(probability, target, present):
    reliable = target.ge(0) & present.bool()
    if not reliable.any():
        return probability.sum() * 0.0, 0
    return F.binary_cross_entropy(probability[reliable], target[reliable]), int(reliable.sum())


def matched_shuffle_loss(matched_prediction, shuffled_prediction, labels, positive):
    matched_error = F.smooth_l1_loss(
        matched_prediction.view(-1), labels.view(-1), reduction="none"
    )
    shuffled_error = F.smooth_l1_loss(
        shuffled_prediction.view(-1), labels.view(-1), reduction="none"
    )
    positive = positive.bool()
    if not positive.any():
        return matched_prediction.sum() * 0.0, matched_error, shuffled_error, 0
    loss = F.relu(matched_error[positive] - shuffled_error[positive]).mean()
    return loss, matched_error, shuffled_error, int(positive.sum())


def load_stage7a_derangements(stage7a_directory):
    frame = pd.read_csv(Path(stage7a_directory) / "shuffle_manifest.csv")
    if set(frame.Split) != {"train", "valid"} or frame.CrossSplit.astype(bool).any():
        raise ValueError("Stage7A derangement split binding is invalid.")
    result = {}
    for split in ("train", "valid"):
        result[split] = {}
        local_split = frame[frame.Split.eq(split)]
        if set(local_split.Repeat.astype(int)) != set(range(10)):
            raise ValueError("Stage7A must contain exactly ten derangements.")
        for repeat, local in local_split.groupby("Repeat"):
            local = local.sort_values("SourceIndex", kind="mergesort")
            mapping = local.TargetIndex.to_numpy(dtype=np.int64)
            if (
                not np.array_equal(local.SourceIndex.to_numpy(), np.arange(len(local)))
                or not np.array_equal(np.sort(mapping), np.arange(len(mapping)))
                or np.any(mapping == np.arange(len(mapping)))
            ):
                raise ValueError("Stage7A derangement is not a fixed-point-free bijection.")
            if not np.array_equal(mapping, local.AudioTargetIndex.to_numpy(dtype=np.int64)):
                raise ValueError("Audio derangement binding differs.")
            if not np.array_equal(mapping, local.VisionTargetIndex.to_numpy(dtype=np.int64)):
                raise ValueError("Vision derangement binding differs.")
            result[split][int(repeat)] = mapping
    return result


class ConditionalModalityGateWrapper(nn.Module):
    """Stage3 missing wrapper plus zero-initialized pre-fusion A/V scalar gates."""

    def __init__(self, backbone, audio_dim, vision_dim, qualified_modalities=()):
        super().__init__()
        if backbone.orig_d_l == backbone.d_l:
            raise ValueError("Text projection hook is unavailable.")
        self.backbone = backbone
        self.missing_audio_token = nn.Parameter(torch.zeros(1, 1, int(audio_dim)))
        self.missing_vision_token = nn.Parameter(torch.zeros(1, 1, int(vision_dim)))
        self.mask_adapter = nn.Linear(3, backbone.out_layer.in_features, bias=False)
        nn.init.zeros_(self.mask_adapter.weight)
        qualified = set(qualified_modalities)
        unknown = qualified - set(MODALITIES)
        if unknown:
            raise ValueError("Unknown qualified modalities: {}".format(unknown))
        gate_dim = 2 * int(backbone.d_l)
        # New zero-initialized gates must not shift Stage3's dropout RNG stream.
        cpu_rng = torch.get_rng_state().clone()
        try:
            self.audio_utility_gate = nn.Linear(gate_dim, 1) if "A" in qualified else None
            self.vision_utility_gate = nn.Linear(gate_dim, 1) if "V" in qualified else None
            for gate in (self.audio_utility_gate, self.vision_utility_gate):
                if gate is not None:
                    nn.init.zeros_(gate.weight)
                    nn.init.zeros_(gate.bias)
        finally:
            torch.set_rng_state(cpu_rng)
        self._current_mask = None
        self._text_projection = None
        self._last_probability = {}
        self._last_scale = {}
        self._hook_handles = [
            self.backbone.proj_l.register_forward_hook(self._capture_text),
            self.backbone.proj_a.register_forward_hook(
                lambda module, inputs, output: self._gate_projection("A", output)
            ),
            self.backbone.proj_v.register_forward_hook(
                lambda module, inputs, output: self._gate_projection("V", output)
            ),
        ]

    def _capture_text(self, module, inputs, output):
        self._text_projection = output
        return output

    def _gate_projection(self, modality, output):
        gate = self.audio_utility_gate if modality == "A" else self.vision_utility_gate
        batch = output.size(0)
        if gate is None:
            q = output.new_full((batch,), 0.5)
            g = output.new_ones(batch)
        else:
            if self._text_projection is None or self._current_mask is None:
                raise RuntimeError("Utility gate context was not initialized.")
            text_pool = self._text_projection.mean(dim=2)
            modality_pool = output.mean(dim=2)
            q = torch.sigmoid(gate(torch.cat([text_pool, modality_pool], dim=1))).view(-1)
            g = 2.0 * q
        column = 1 if modality == "A" else 2
        present = self._current_mask[:, column].to(output).view(-1)
        applied = 1.0 + (g - 1.0) * present
        self._last_probability[modality] = q
        self._last_scale[modality] = applied
        return output * applied.view(-1, 1, 1)

    def forward(self, text, audio, vision, modality_mask):
        modality_mask = modality_mask.to(device=audio.device, dtype=audio.dtype)
        masked_audio, masked_vision = apply_moddrop_tokens(
            audio, vision, modality_mask,
            self.missing_audio_token, self.missing_vision_token,
        )
        self._current_mask = modality_mask
        self._text_projection = None
        self._last_probability = {}
        self._last_scale = {}
        residual = self.mask_adapter(1.0 - modality_mask)
        output = self.backbone(
            text, masked_audio, masked_vision, fusion_residual=residual
        )
        for modality in MODALITIES:
            if modality not in self._last_probability:
                self._last_probability[modality] = audio.new_full((audio.size(0),), 0.5)
                self._last_scale[modality] = audio.new_ones(audio.size(0))
            output["utility_q_{}".format(modality)] = self._last_probability[modality]
            output["utility_g_{}".format(modality)] = self._last_scale[modality]
        self._current_mask = None
        self._text_projection = None
        return output


def gate_distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return {key: float("nan") for key in (
            "mean", "std", "p05", "p25", "median", "p75", "p95",
            "low_saturation_fraction", "high_saturation_fraction",
        )}
    return {
        "mean": float(x.mean()), "std": float(x.std(ddof=0)),
        "p05": float(np.quantile(x, .05)), "p25": float(np.quantile(x, .25)),
        "median": float(np.median(x)), "p75": float(np.quantile(x, .75)),
        "p95": float(np.quantile(x, .95)),
        "low_saturation_fraction": float((x < .05).mean()),
        "high_saturation_fraction": float((x > .95).mean()),
    }


def safe_auroc(target, probability):
    target = np.asarray(target)
    probability = np.asarray(probability)
    keep = np.isfinite(target) & np.isfinite(probability)
    if keep.sum() == 0 or len(np.unique(target[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(target[keep], probability[keep]))


def summarize_gate_samples(samples):
    rows = []
    for (epoch, split, modality), local in samples.groupby(
        ["Epoch", "Split", "Modality"], sort=True
    ):
        reliable = local[local.UtilityGroup.isin(["positive", "negative"])]
        positive = reliable[reliable.UtilityGroup.eq("positive")]
        negative = reliable[reliable.UtilityGroup.eq("negative")]
        q_stats = gate_distribution(local.q)
        g_stats = gate_distribution(local.g)
        rows.append({
            "Epoch": int(epoch), "Split": split, "Modality": modality,
            "SampleCount": len(local),
            **{"q_{}".format(key): value for key, value in q_stats.items()},
            **{"g_{}".format(key): value for key, value in g_stats.items()},
            "ReliablePositiveCount": len(positive),
            "ReliableNegativeCount": len(negative),
            "ReliablePositiveMeanQ": float(positive.q.mean()) if len(positive) else np.nan,
            "ReliableNegativeMeanQ": float(negative.q.mean()) if len(negative) else np.nan,
            "PositiveNegativeGateGap": (
                float(positive.q.mean() - negative.q.mean())
                if len(positive) and len(negative) else np.nan
            ),
            "UtilityAUROC": safe_auroc(reliable.UtilityTarget, reliable.q),
        })
    return pd.DataFrame(rows)


def checkpoint_paths(root, variant, dataset, seed, smoke=False):
    versions = {
        "identity_replay": "cmug_identity_replay_v1",
        "utility_gate": "cmug_utility_gate_v1",
        "utility_gate_matched": "cmug_v1",
    }
    version = versions[variant]
    main = Path(root) / "missing_baseline" / version
    if smoke:
        main = main / "smoke"
    main = main / "DLF_{}_seed{}_best_valid.pth".format(dataset, seed)
    diagnostic = main.parent / "diagnostic" / main.name.replace(
        "_best_valid.pth", "_best_test_diagnostic.pth"
    )
    return version, main, diagnostic


def result_directory(root, variant, smoke=False):
    version = {
        "identity_replay": "cmug_identity_replay_v1",
        "utility_gate": "cmug_utility_gate_v1",
        "utility_gate_matched": "cmug_v1",
    }[variant]
    path = Path(root) / "missing_baseline" / version
    return path / "smoke" if smoke else path


def flatten_mode_metrics(metrics, prefix):
    result = {}
    for mode, values in metrics.items():
        for key, value in values.items():
            result["{}_{}_{}".format(prefix, mode, key)] = float(value)
    return result


def missing_macro_mae(metrics):
    return float(np.mean([metrics[mode]["MAE"] for mode in ("LA", "LV", "L")]))


def objective(metrics):
    return 0.5 * metrics["LAV"]["MAE"] + 0.5 * missing_macro_mae(metrics)


def first_epoch_counts_ok(counts):
    return dict(counts) == {"LA": 435, "LV": 430, "L": 419}
