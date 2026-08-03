"""Zero-training audit of audio, vision, and audio-vision synergy contributions.

The audit consumes the five validation-selected Stage-8 CFCompatKD Online
prediction CSVs.  It never loads a model, Teacher, evaluator, cache, or label
outside the already materialized prediction artifacts.

For each sample and seed:
    b = T_L
    a = T_LA - T_L
    v = T_LV - T_L
    s = T_LAV - T_LA - T_LV + T_L

The no-synergy reconstruction is T_LA + T_LV - T_L.  Synergy gain is the
absolute-error reduction from adding s to that reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "av_synergy_contribution_audit_v1"
DEFAULT_SEEDS = (1111, 1112, 1113, 1114, 1115)
DEFAULT_SPLITS = ("valid", "test")
PREDICTION_COLUMNS = (
    "sample_id",
    "sample_index",
    "label",
    "LAV_pred",
    "LA_pred",
    "LV_pred",
    "L_pred",
    "Seed",
    "Method",
    "Split",
    "SelectedBy",
)
EFFECTS = ("audio", "vision", "synergy")
GAINS = ("audio", "vision", "synergy", "total_av")


@dataclass(frozen=True)
class AuditConfig:
    result_root: str = "result"
    dataset: str = "mosi"
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    splits: Tuple[str, ...] = DEFAULT_SPLITS
    decision_split: str = "valid"
    bootstrap_repetitions: int = 5000
    bootstrap_seed: int = 93431
    magnitude_threshold: float = 0.10
    required_synergy_gain: float = 0.003
    required_marginal_gain: float = 0.002
    required_positive_seeds: int = 4
    required_sign_agreement: float = 0.70
    required_high_magnitude_coverage: float = 0.05

    def validate(self) -> None:
        if self.dataset != "mosi":
            raise ValueError("This frozen audit currently supports MOSI only.")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("At least two unique seeds are required.")
        if self.decision_split not in self.splits:
            raise ValueError("decision_split must be included in splits.")
        if any(split not in {"valid", "test"} for split in self.splits):
            raise ValueError("Only valid/test Stage8 predictions are supported.")
        if self.bootstrap_repetitions < 100:
            raise ValueError("bootstrap_repetitions must be at least 100.")
        if self.magnitude_threshold <= 0:
            raise ValueError("magnitude_threshold must be positive.")
        if not 1 <= self.required_positive_seeds <= len(self.seeds):
            raise ValueError("required_positive_seeds is invalid.")
        if not 0 <= self.required_sign_agreement <= 1:
            raise ValueError("required_sign_agreement must lie in [0,1].")
        if not 0 <= self.required_high_magnitude_coverage <= 1:
            raise ValueError("required_high_magnitude_coverage must lie in [0,1].")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=list(DEFAULT_SPLITS))
    parser.add_argument("--decision-split", choices=("valid", "test"), default="valid")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=93431)
    parser.add_argument("--magnitude-threshold", type=float, default=0.10)
    parser.add_argument("--required-synergy-gain", type=float, default=0.003)
    parser.add_argument("--required-marginal-gain", type=float, default=0.002)
    parser.add_argument("--required-positive-seeds", type=int, default=4)
    parser.add_argument("--required-sign-agreement", type=float, default=0.70)
    parser.add_argument("--required-high-magnitude-coverage", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default="result/missing_baseline/av_synergy_contribution_audit_v1/mosi",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_prediction_path(config: AuditConfig, seed: int, split: str) -> Path:
    return (
        Path(config.result_root)
        / "missing_baseline"
        / "cfcompat_stability_v1"
        / config.dataset
        / f"seed{int(seed)}"
        / f"online_{split}_predictions.csv"
    )


def video_id_from_sample_id(value: object) -> str:
    text = str(value)
    for separator in ("$_$", "[SEP]", "::"):
        if separator in text:
            return text.split(separator, 1)[0]
    if "_" in text:
        head, tail = text.rsplit("_", 1)
        if tail.isdigit():
            return head
    return text


def _require_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")


def load_sources(config: AuditConfig) -> Tuple[Dict[str, Dict[int, pd.DataFrame]], pd.DataFrame]:
    config.validate()
    sources: Dict[str, Dict[int, pd.DataFrame]] = {}
    manifest_rows: List[dict] = []
    for split in config.splits:
        sources[split] = {}
        reference: pd.DataFrame | None = None
        for seed in config.seeds:
            path = source_prediction_path(config, seed, split)
            if not path.is_file():
                raise FileNotFoundError(f"Required Stage8 prediction is absent: {path}")
            frame = pd.read_csv(path)
            _require_columns(frame, path)
            if frame["sample_index"].duplicated().any() or frame["sample_id"].astype(str).duplicated().any():
                raise ValueError(f"Duplicate sample binding in {path}")
            if set(frame["Seed"].astype(int)) != {int(seed)}:
                raise ValueError(f"Seed column mismatch in {path}")
            if set(frame["Method"].astype(str)) != {"Online"}:
                raise ValueError(f"Only Online Stage8 predictions are permitted: {path}")
            if set(frame["Split"].astype(str)) != {split}:
                raise ValueError(f"Split column mismatch in {path}")
            if set(frame["SelectedBy"].astype(str)) != {"validation_J"}:
                raise ValueError(f"Prediction was not validation-selected: {path}")
            numeric = frame[["label", "LAV_pred", "LA_pred", "LV_pred", "L_pred"]].to_numpy(float)
            if not np.isfinite(numeric).all():
                raise FloatingPointError(f"Non-finite values in {path}")
            frame = frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
            if reference is None:
                reference = frame[["sample_index", "sample_id", "label"]].copy()
            else:
                if frame["sample_index"].astype(int).tolist() != reference["sample_index"].astype(int).tolist():
                    raise RuntimeError(f"sample_index mismatch across seeds for {split}")
                if frame["sample_id"].astype(str).tolist() != reference["sample_id"].astype(str).tolist():
                    raise RuntimeError(f"sample_id mismatch across seeds for {split}")
                if not np.array_equal(
                    frame["label"].to_numpy(np.float32),
                    reference["label"].to_numpy(np.float32),
                ):
                    raise RuntimeError(f"label mismatch across seeds for {split}")
            sources[split][int(seed)] = frame
            manifest_rows.append(
                {
                    "split": split,
                    "seed": int(seed),
                    "path": str(path),
                    "sha256": sha256(path),
                    "sample_count": int(len(frame)),
                    "selected_by": "validation_J",
                    "method": "Online",
                }
            )
    return sources, pd.DataFrame(manifest_rows).sort_values(["split", "seed"])


def contribution_frame(frame: pd.DataFrame, seed: int, split: str) -> pd.DataFrame:
    result = frame[["sample_index", "sample_id", "label"]].copy()
    result["video_id"] = result["sample_id"].map(video_id_from_sample_id)
    result["seed"] = int(seed)
    result["split"] = split
    for column in ("L_pred", "LA_pred", "LV_pred", "LAV_pred"):
        result[column] = frame[column].to_numpy(float)
    result["no_synergy_pred"] = result["LA_pred"] + result["LV_pred"] - result["L_pred"]
    result["audio_effect"] = result["LA_pred"] - result["L_pred"]
    result["vision_effect"] = result["LV_pred"] - result["L_pred"]
    result["synergy_effect"] = (
        result["LAV_pred"] - result["LA_pred"] - result["LV_pred"] + result["L_pred"]
    )
    result["total_av_effect"] = result["LAV_pred"] - result["L_pred"]
    label = result["label"].to_numpy(float)
    errors = {
        "L": np.abs(result["L_pred"].to_numpy(float) - label),
        "LA": np.abs(result["LA_pred"].to_numpy(float) - label),
        "LV": np.abs(result["LV_pred"].to_numpy(float) - label),
        "no_synergy": np.abs(result["no_synergy_pred"].to_numpy(float) - label),
        "LAV": np.abs(result["LAV_pred"].to_numpy(float) - label),
    }
    for name, values in errors.items():
        result[f"abs_error_{name}"] = values
    result["audio_gain"] = errors["L"] - errors["LA"]
    result["vision_gain"] = errors["L"] - errors["LV"]
    result["synergy_gain"] = errors["no_synergy"] - errors["LAV"]
    result["total_av_gain"] = errors["L"] - errors["LAV"]
    result["synergy_helpful"] = result["synergy_gain"] > 0
    return result


def effect_distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    absolute = np.abs(values)
    return {
        "mean": float(values.mean()),
        "mean_abs": float(absolute.mean()),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "fraction_abs_gt_0p10": float(np.mean(absolute > 0.10)),
        "fraction_abs_gt_0p25": float(np.mean(absolute > 0.25)),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
    }


def per_seed_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (split, seed), local in samples.groupby(["split", "seed"], sort=True):
        base = {
            "split": split,
            "seed": int(seed),
            "sample_count": int(len(local)),
            "video_count": int(local["video_id"].nunique()),
            "mae_L": float(local["abs_error_L"].mean()),
            "mae_LA": float(local["abs_error_LA"].mean()),
            "mae_LV": float(local["abs_error_LV"].mean()),
            "mae_no_synergy": float(local["abs_error_no_synergy"].mean()),
            "mae_LAV": float(local["abs_error_LAV"].mean()),
        }
        for name in EFFECTS:
            stats = effect_distribution(local[f"{name}_effect"].to_numpy(float))
            for key, value in stats.items():
                base[f"{name}_effect_{key}"] = value
        for name in GAINS:
            values = local[f"{name}_gain"].to_numpy(float)
            base[f"{name}_gain_mean"] = float(values.mean())
            base[f"{name}_gain_median"] = float(np.median(values))
            base[f"{name}_gain_positive_fraction"] = float(np.mean(values > 0))
            base[f"{name}_gain_large_harm_fraction"] = float(np.mean(values < -0.10))
        oracle_error = np.minimum(local["abs_error_no_synergy"], local["abs_error_LAV"])
        base["synergy_oracle_mae"] = float(oracle_error.mean())
        base["synergy_oracle_gain_vs_no_synergy"] = float(
            local["abs_error_no_synergy"].mean() - oracle_error.mean()
        )
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["split", "seed"])


def ensemble_samples(samples: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for split, local in samples.groupby("split", sort=True):
        pivoted: Dict[str, pd.DataFrame] = {}
        for column in ("L_pred", "LA_pred", "LV_pred", "LAV_pred"):
            table = local.pivot(index="sample_index", columns="seed", values=column)
            table = table.loc[:, list(seeds)]
            pivoted[column] = table
        first = (
            local.sort_values(["sample_index", "seed"], kind="mergesort")
            .drop_duplicates("sample_index")
            .set_index("sample_index")
        )
        frame = pd.DataFrame(index=pivoted["L_pred"].index)
        frame["sample_index"] = frame.index.astype(int)
        frame["sample_id"] = first.loc[frame.index, "sample_id"].astype(str).to_numpy()
        frame["video_id"] = first.loc[frame.index, "video_id"].astype(str).to_numpy()
        frame["label"] = first.loc[frame.index, "label"].to_numpy(float)
        frame["split"] = split
        for column, table in pivoted.items():
            frame[column] = table.mean(axis=1).to_numpy(float)
        frame = contribution_frame(
            frame.assign(Seed=0, Method="Online", Split=split, SelectedBy="validation_J"),
            seed=0,
            split=split,
        )
        frame = frame.drop(columns=["seed"])
        for name in EFFECTS:
            table = local.pivot(index="sample_index", columns="seed", values=f"{name}_effect").loc[:, list(seeds)]
            values = table.to_numpy(float)
            positive = (values > 0).sum(axis=1)
            negative = (values < 0).sum(axis=1)
            frame[f"{name}_effect_mean_abs_across_seeds"] = np.abs(values).mean(axis=1)
            frame[f"{name}_effect_sign_agreement"] = np.maximum(positive, negative) / float(len(seeds))
            frame[f"{name}_effect_all_same_nonzero_sign"] = (
                (positive == len(seeds)) | (negative == len(seeds))
            )
        for name in GAINS:
            table = local.pivot(index="sample_index", columns="seed", values=f"{name}_gain").loc[:, list(seeds)]
            values = table.to_numpy(float)
            frame[f"{name}_gain_mean_across_seeds"] = values.mean(axis=1)
            frame[f"{name}_gain_positive_seed_count"] = (values > 0).sum(axis=1)
        rows.append(frame.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def group_bootstrap(values: np.ndarray, groups: Sequence[str], repetitions: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups).astype(str)
    if len(values) != len(groups) or len(values) == 0:
        raise ValueError("Invalid group bootstrap inputs.")
    unique = np.asarray(sorted(np.unique(groups)))
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([indices[group] for group in sampled])
        draws[index] = float(values[positions].mean())
    return {
        "mean_gain": float(values.mean()),
        "gain_ci_low": float(np.quantile(draws, 0.025)),
        "gain_ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_positive_probability": float(np.mean(draws > 0)),
        "video_count": int(len(unique)),
    }


def bootstrap_table(ensemble: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    rows: List[dict] = []
    for split, local in ensemble.groupby("split", sort=True):
        for offset, name in enumerate(GAINS):
            rows.append(
                {
                    "split": split,
                    "gain": name,
                    **group_bootstrap(
                        local[f"{name}_gain"].to_numpy(float),
                        local["video_id"].astype(str).tolist(),
                        config.bootstrap_repetitions,
                        config.bootstrap_seed + 7919 * (offset + 1) + (0 if split == "valid" else 104729),
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "gain"])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy(float)
    right_rank = pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy(float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def cross_seed_consistency(samples: pd.DataFrame, ensemble: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    rows: List[dict] = []
    for split, local in samples.groupby("split", sort=True):
        ensemble_local = ensemble.loc[ensemble["split"] == split].set_index("sample_index")
        for name in EFFECTS:
            table = local.pivot(index="sample_index", columns="seed", values=f"{name}_effect").loc[:, list(config.seeds)]
            pairwise = [
                spearman(table[left].to_numpy(float), table[right].to_numpy(float))
                for left, right in itertools.combinations(config.seeds, 2)
            ]
            mean_abs = ensemble_local.loc[table.index, f"{name}_effect_mean_abs_across_seeds"].to_numpy(float)
            agreement = ensemble_local.loc[table.index, f"{name}_effect_sign_agreement"].to_numpy(float)
            high = mean_abs >= config.magnitude_threshold
            rows.append(
                {
                    "split": split,
                    "quantity": f"{name}_effect",
                    "mean_pairwise_spearman": float(np.nanmean(pairwise)),
                    "min_pairwise_spearman": float(np.nanmin(pairwise)),
                    "max_pairwise_spearman": float(np.nanmax(pairwise)),
                    "mean_sign_agreement_all": float(agreement.mean()),
                    "high_magnitude_threshold": config.magnitude_threshold,
                    "high_magnitude_count": int(high.sum()),
                    "high_magnitude_coverage": float(high.mean()),
                    "mean_sign_agreement_high_magnitude": float(agreement[high].mean()) if high.any() else float("nan"),
                    "all_same_nonzero_sign_high_magnitude_fraction": float(
                        ensemble_local.loc[table.index, f"{name}_effect_all_same_nonzero_sign"].to_numpy(bool)[high].mean()
                    ) if high.any() else float("nan"),
                }
            )
        for name in GAINS:
            seed_means = local.groupby("seed")[f"{name}_gain"].mean().reindex(config.seeds)
            rows.append(
                {
                    "split": split,
                    "quantity": f"{name}_gain",
                    "mean_pairwise_spearman": float("nan"),
                    "min_pairwise_spearman": float("nan"),
                    "max_pairwise_spearman": float("nan"),
                    "mean_sign_agreement_all": float("nan"),
                    "high_magnitude_threshold": float("nan"),
                    "high_magnitude_count": 0,
                    "high_magnitude_coverage": float("nan"),
                    "mean_sign_agreement_high_magnitude": float("nan"),
                    "all_same_nonzero_sign_high_magnitude_fraction": float("nan"),
                    "positive_seed_count": int((seed_means > 0).sum()),
                    "seed_gain_mean": float(seed_means.mean()),
                    "seed_gain_min": float(seed_means.min()),
                    "seed_gain_max": float(seed_means.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "quantity"])


def _region_masks(local: pd.DataFrame, threshold: float) -> Mapping[str, np.ndarray]:
    audio = local["audio_effect"].to_numpy(float)
    vision = local["vision_effect"].to_numpy(float)
    text = local["L_pred"].to_numpy(float)
    text_error = local["abs_error_L"].to_numpy(float)
    high_text_error = text_error >= np.quantile(text_error, 0.75)
    return {
        "all": np.ones(len(local), dtype=bool),
        "synergy_large": np.abs(local["synergy_effect"].to_numpy(float)) >= threshold,
        "audio_vision_same_direction": (np.sign(audio) == np.sign(vision)) & (audio != 0) & (vision != 0),
        "audio_vision_opposite_direction": (np.sign(audio) != np.sign(vision)) & (audio != 0) & (vision != 0),
        "audio_opposes_text": (audio * text) < 0,
        "vision_opposes_text": (vision * text) < 0,
        "text_high_error_q4": high_text_error,
    }


def conditional_regions(ensemble: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    rows: List[dict] = []
    for split, local in ensemble.groupby("split", sort=True):
        local = local.reset_index(drop=True)
        for offset, (region, mask) in enumerate(_region_masks(local, config.magnitude_threshold).items()):
            if not mask.any():
                continue
            values = local.loc[mask, "synergy_gain"].to_numpy(float)
            groups = local.loc[mask, "video_id"].astype(str).tolist()
            boot = group_bootstrap(
                values,
                groups,
                config.bootstrap_repetitions,
                config.bootstrap_seed + 200003 + 3571 * (offset + 1) + (0 if split == "valid" else 104729),
            )
            rows.append(
                {
                    "split": split,
                    "region": region,
                    "sample_count": int(mask.sum()),
                    "sample_fraction": float(mask.mean()),
                    "synergy_effect_mean_abs": float(np.abs(local.loc[mask, "synergy_effect"]).mean()),
                    "synergy_gain_positive_fraction": float((values > 0).mean()),
                    **boot,
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "region"])


def video_concentration(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for split, local in ensemble.groupby("split", sort=True):
        per_video = local.groupby("video_id", sort=True)["synergy_gain"].agg(["sum", "mean", "count"]).reset_index()
        positive = np.maximum(per_video["sum"].to_numpy(float), 0)
        total_positive = float(positive.sum())
        order = np.argsort(-positive)
        top_count = max(1, int(math.ceil(0.10 * len(per_video))))
        rows.append(
            {
                "split": split,
                "video_count": int(len(per_video)),
                "videos_with_positive_mean_gain": int((per_video["mean"] > 0).sum()),
                "max_single_video_positive_gain_share": float(positive[order[0]] / total_positive) if total_positive > 0 else float("nan"),
                "top_decile_video_positive_gain_share": float(positive[order[:top_count]].sum() / total_positive) if total_positive > 0 else float("nan"),
                "leave_one_video_out_gain_min": float(
                    min(
                        local.loc[local["video_id"] != video, "synergy_gain"].mean()
                        for video in per_video["video_id"]
                    )
                ),
                "leave_one_video_out_gain_max": float(
                    max(
                        local.loc[local["video_id"] != video, "synergy_gain"].mean()
                        for video in per_video["video_id"]
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("split")


def aggregate_metrics(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for split, local in ensemble.groupby("split", sort=True):
        for name in EFFECTS:
            stats = effect_distribution(local[f"{name}_effect"].to_numpy(float))
            rows.append({"split": split, "quantity": f"{name}_effect", **stats})
        for name in GAINS:
            values = local[f"{name}_gain"].to_numpy(float)
            rows.append(
                {
                    "split": split,
                    "quantity": f"{name}_gain",
                    "mean": float(values.mean()),
                    "mean_abs": float(np.abs(values).mean()),
                    "median_abs": float(np.median(np.abs(values))),
                    "p90_abs": float(np.quantile(np.abs(values), 0.90)),
                    "p95_abs": float(np.quantile(np.abs(values), 0.95)),
                    "fraction_abs_gt_0p10": float(np.mean(np.abs(values) > 0.10)),
                    "fraction_abs_gt_0p25": float(np.mean(np.abs(values) > 0.25)),
                    "positive_fraction": float(np.mean(values > 0)),
                    "negative_fraction": float(np.mean(values < 0)),
                }
            )
        rows.extend(
            [
                {"split": split, "quantity": "mae_L", "mean": float(local["abs_error_L"].mean())},
                {"split": split, "quantity": "mae_LA", "mean": float(local["abs_error_LA"].mean())},
                {"split": split, "quantity": "mae_LV", "mean": float(local["abs_error_LV"].mean())},
                {"split": split, "quantity": "mae_no_synergy", "mean": float(local["abs_error_no_synergy"].mean())},
                {"split": split, "quantity": "mae_LAV", "mean": float(local["abs_error_LAV"].mean())},
                {
                    "split": split,
                    "quantity": "synergy_oracle_gain_vs_no_synergy",
                    "mean": float(
                        local["abs_error_no_synergy"].mean()
                        - np.minimum(local["abs_error_no_synergy"], local["abs_error_LAV"]).mean()
                    ),
                },
            ]
        )
    return pd.DataFrame(rows).sort_values(["split", "quantity"])


def _lookup(frame: pd.DataFrame, split: str, key_column: str, key: str, value: str) -> float:
    selected = frame[(frame["split"] == split) & (frame[key_column] == key)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {split}/{key}")
    return float(selected.iloc[0][value])


def decision_summary(
    config: AuditConfig,
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    consistency: pd.DataFrame,
) -> dict:
    del per_seed
    split = config.decision_split
    synergy_gain = _lookup(aggregate, split, "quantity", "synergy_gain", "mean")
    synergy_ci_low = _lookup(bootstrap, split, "gain", "synergy", "gain_ci_low")
    synergy_positive_seeds = int(
        consistency[(consistency["split"] == split) & (consistency["quantity"] == "synergy_gain")]
        .iloc[0]
        .get("positive_seed_count", 0)
    )
    synergy_effect_row = consistency[
        (consistency["split"] == split) & (consistency["quantity"] == "synergy_effect")
    ].iloc[0]
    high_coverage = float(synergy_effect_row["high_magnitude_coverage"])
    high_agreement = float(synergy_effect_row["mean_sign_agreement_high_magnitude"])
    synergy_gate = {
        "mean_gain": synergy_gain,
        "required_mean_gain": config.required_synergy_gain,
        "ci_low": synergy_ci_low,
        "required_ci_low_gt_zero": True,
        "positive_seed_count": synergy_positive_seeds,
        "required_positive_seeds": config.required_positive_seeds,
        "high_magnitude_coverage": high_coverage,
        "required_high_magnitude_coverage": config.required_high_magnitude_coverage,
        "high_magnitude_sign_agreement": high_agreement,
        "required_sign_agreement": config.required_sign_agreement,
    }
    synergy_gate["passed"] = bool(
        synergy_gain >= config.required_synergy_gain
        and synergy_ci_low > 0
        and synergy_positive_seeds >= config.required_positive_seeds
        and high_coverage >= config.required_high_magnitude_coverage
        and np.isfinite(high_agreement)
        and high_agreement >= config.required_sign_agreement
    )

    marginal_details = {}
    for name in ("audio", "vision"):
        gain = _lookup(aggregate, split, "quantity", f"{name}_gain", "mean")
        ci_low = _lookup(bootstrap, split, "gain", name, "gain_ci_low")
        positive_seeds = int(
            consistency[
                (consistency["split"] == split)
                & (consistency["quantity"] == f"{name}_gain")
            ].iloc[0].get("positive_seed_count", 0)
        )
        marginal_details[name] = {
            "mean_gain": gain,
            "required_mean_gain": config.required_marginal_gain,
            "ci_low": ci_low,
            "positive_seed_count": positive_seeds,
            "required_positive_seeds": config.required_positive_seeds,
            "passed": bool(
                gain >= config.required_marginal_gain
                and ci_low > 0
                and positive_seeds >= config.required_positive_seeds
            ),
        }
    marginal_gate = {
        "audio": marginal_details["audio"],
        "vision": marginal_details["vision"],
        "passed": bool(marginal_details["audio"]["passed"] or marginal_details["vision"]["passed"]),
    }
    if synergy_gate["passed"] and marginal_gate["passed"]:
        verdict = "SUPPORTED_BUILD_CONTRIBUTION_FACTORIZATION_DISTILLATION"
    elif synergy_gate["passed"] or marginal_gate["passed"]:
        verdict = "PARTIAL_SIGNAL_EXPAND_ONLY_SUPPORTED_COMPONENT"
    else:
        verdict = "NOT_SUPPORTED_DO_NOT_TRAIN_CONTRIBUTION_FACTORIZATION"
    return {
        "decision_split": split,
        "synergy_gate": synergy_gate,
        "marginal_gate": marginal_gate,
        "verdict": verdict,
    }


def build_artifacts(config: AuditConfig) -> dict:
    sources, source_manifest = load_sources(config)
    sample_frames = [
        contribution_frame(frame, seed, split)
        for split in config.splits
        for seed, frame in sources[split].items()
    ]
    samples = pd.concat(sample_frames, ignore_index=True)
    seed_metrics = per_seed_metrics(samples)
    ensemble = ensemble_samples(samples, config.seeds)
    aggregate = aggregate_metrics(ensemble)
    bootstrap = bootstrap_table(ensemble, config)
    consistency = cross_seed_consistency(samples, ensemble, config)
    regions = conditional_regions(ensemble, config)
    concentration = video_concentration(ensemble)
    decision = decision_summary(config, seed_metrics, aggregate, bootstrap, consistency)
    summary = {
        "version": VERSION,
        "method": "zero_training_counterfactual_av_contribution_factorization_audit",
        "config": asdict(config),
        "sample_count_by_split": {
            split: int((ensemble["split"] == split).sum()) for split in config.splits
        },
        "source_file_count": int(len(source_manifest)),
        "outer_or_test_labels_used_for_fitting": False,
        "models_loaded": False,
        "training_performed": False,
        "teacher_or_evaluator_loaded": False,
        "primary_decision_uses_test": config.decision_split == "test",
        **decision,
    }
    return {
        "source_manifest": source_manifest,
        "per_seed_samples": samples,
        "per_seed_metrics": seed_metrics,
        "ensemble_samples": ensemble,
        "aggregate_metrics": aggregate,
        "group_bootstrap": bootstrap,
        "cross_seed_consistency": consistency,
        "conditional_regions": regions,
        "video_concentration": concentration,
        "summary": summary,
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    local = frame[[column for column in columns if column in frame.columns]].copy()
    headers = list(local.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in local.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_artifacts(artifacts: Mapping[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    file_map = {
        "source_manifest": "avsc_source_manifest.csv",
        "per_seed_samples": "avsc_per_seed_samples.csv",
        "per_seed_metrics": "avsc_per_seed_metrics.csv",
        "ensemble_samples": "avsc_ensemble_samples.csv",
        "aggregate_metrics": "avsc_aggregate_metrics.csv",
        "group_bootstrap": "avsc_group_bootstrap.csv",
        "cross_seed_consistency": "avsc_cross_seed_consistency.csv",
        "conditional_regions": "avsc_conditional_regions.csv",
        "video_concentration": "avsc_video_concentration.csv",
    }
    for key, name in file_map.items():
        assert isinstance(artifacts[key], pd.DataFrame)
        artifacts[key].to_csv(output / name, index=False)
    summary = artifacts["summary"]
    (output / "avsc_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = artifacts["aggregate_metrics"]
    bootstrap = artifacts["group_bootstrap"]
    consistency = artifacts["cross_seed_consistency"]
    regions = artifacts["conditional_regions"]
    per_seed = artifacts["per_seed_metrics"]
    lines = [
        "# AV Synergy Contribution Audit v1",
        "",
        "This is a zero-training audit of validation-selected Stage8 CFCompatKD Online predictions.",
        "The formal decision uses only the configured decision split; Test is diagnostic unless explicitly selected.",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Decision split: `{summary['decision_split']}`",
        f"- Synergy gate: `{summary['synergy_gate']['passed']}`",
        f"- Marginal gate: `{summary['marginal_gate']['passed']}`",
        "",
        "## Per-seed gain means",
        "",
        markdown_table(
            per_seed,
            ["split", "seed", "audio_gain_mean", "vision_gain_mean", "synergy_gain_mean", "total_av_gain_mean", "mae_L", "mae_no_synergy", "mae_LAV"],
        ),
        "",
        "## Ensemble aggregate",
        "",
        markdown_table(aggregate, ["split", "quantity", "mean", "mean_abs", "p90_abs", "positive_fraction"]),
        "",
        "## Video-group bootstrap",
        "",
        markdown_table(bootstrap, ["split", "gain", "mean_gain", "gain_ci_low", "gain_ci_high", "bootstrap_positive_probability"]),
        "",
        "## Cross-seed consistency",
        "",
        markdown_table(
            consistency,
            ["split", "quantity", "mean_pairwise_spearman", "positive_seed_count", "high_magnitude_coverage", "mean_sign_agreement_high_magnitude"],
        ),
        "",
        "## Conditional synergy regions",
        "",
        markdown_table(regions, ["split", "region", "sample_count", "sample_fraction", "mean_gain", "gain_ci_low", "gain_ci_high"]),
        "",
        "A positive effect magnitude is not treated as usefulness. Promotion requires label-error gain, grouped uncertainty, cross-seed support, and high-magnitude sign consistency.",
    ]
    (output / "avsc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    config = AuditConfig(
        result_root=cli.result_root,
        dataset=cli.dataset,
        seeds=tuple(cli.seeds),
        splits=tuple(dict.fromkeys(cli.splits)),
        decision_split=cli.decision_split,
        bootstrap_repetitions=cli.bootstrap_repetitions,
        bootstrap_seed=cli.bootstrap_seed,
        magnitude_threshold=cli.magnitude_threshold,
        required_synergy_gain=cli.required_synergy_gain,
        required_marginal_gain=cli.required_marginal_gain,
        required_positive_seeds=cli.required_positive_seeds,
        required_sign_agreement=cli.required_sign_agreement,
        required_high_magnitude_coverage=cli.required_high_magnitude_coverage,
    )
    artifacts = build_artifacts(config)
    output = Path(cli.output_dir)
    write_artifacts(artifacts, output)
    summary = artifacts["summary"]
    print("AV SYNERGY CONTRIBUTION AUDIT COMPLETE")
    print("decision split:", summary["decision_split"])
    print("synergy gate:", summary["synergy_gate"]["passed"])
    print("marginal gate:", summary["marginal_gate"]["passed"])
    print("verdict:", summary["verdict"])
    print("report:", output / "avsc_report.md")


if __name__ == "__main__":
    main()
