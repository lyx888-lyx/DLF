"""Hardened formal entry point for the MOSI CFCompat audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_mosi_cfcompat_dataset_mechanism as base
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_COLUMNS,
    CACHE_VERSION,
    MISSING_MODES,
    cache_paths,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.mosi_cfcompat_audit_v2_utils import (
    joint_video_bootstrap,
    opportunity_ranking,
)


_ORIGINAL_BUILD_CONFIG = base.build_config
_ORIGINAL_EVALUATE_SEED = base.evaluate_seed
_AUDIT_SEQ_LENS = None


def patched_build_config(cli, seed):
    args = _ORIGINAL_BUILD_CONFIG(cli, seed)
    if _AUDIT_SEQ_LENS is not None:
        args.seq_lens = tuple(int(value) for value in _AUDIT_SEQ_LENS)
    return args


def patched_evaluate_seed(cli, seed, valid_dataset, metadata):
    global _AUDIT_SEQ_LENS
    _AUDIT_SEQ_LENS = valid_dataset.get_seq_len()
    return _ORIGINAL_EVALUATE_SEED(cli, seed, valid_dataset, metadata)


def legacy_safe_load_counterfactual_cache(
    root,
    dataset,
    version=CACHE_VERSION,
    seed=None,
    expected_evaluator_sha=None,
):
    """Load the frozen cache while accepting the historical seed-1111 config."""
    paths = cache_paths(root, dataset, version=version, seed=seed)
    if not paths["csv"].is_file() or not paths["config"].is_file():
        raise FileNotFoundError("Counterfactual cache has not been built: {}".format(paths["directory"]))
    frame = pd.read_csv(paths["csv"])
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    expected_seed = None if seed is None else int(seed)
    if config.get("version") != version or config.get("seed") != expected_seed:
        raise ValueError("Counterfactual cache version/seed binding is invalid.")
    if (
        expected_evaluator_sha is not None
        and config.get("evaluator_sha256") != expected_evaluator_sha
    ):
        raise ValueError("Counterfactual cache evaluator SHA binding is invalid.")
    if config.get("source") != "train_only":
        raise ValueError("Counterfactual cache is not train-only.")
    historical_seed1111 = (
        version == CACHE_VERSION
        and expected_seed is None
        and "created_from_train_only" not in config
    )
    if config.get("created_from_train_only") is not True and not historical_seed1111:
        raise ValueError("Counterfactual cache lacks the train-only creation binding.")
    if list(frame.columns) != list(CACHE_COLUMNS):
        raise ValueError("Counterfactual cache schema is malformed.")
    if frame.sample_index.duplicated().any() or len(frame) != 1284:
        raise ValueError("Counterfactual cache sample binding is malformed.")
    if not np.array_equal(
        frame.sample_index.to_numpy(dtype=int), np.arange(len(frame), dtype=int)
    ):
        raise ValueError("Counterfactual cache indices are not contiguous.")
    if config.get("cache_sha256") not in (None, checkpoint_sha256(paths["csv"])):
        raise ValueError("Counterfactual cache SHA binding is invalid.")
    for mode in MISSING_MODES:
        values = frame["compat_{}".format(mode)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.all((values > 0) & (values < 1)):
            raise ValueError("Counterfactual compatibility is outside (0,1).")
    by_index = {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }
    return frame, by_index


def main():
    base.build_config = patched_build_config
    base.evaluate_seed = patched_evaluate_seed
    base.load_counterfactual_cache = legacy_safe_load_counterfactual_cache
    base.joint_video_bootstrap = joint_video_bootstrap
    base.opportunity_ranking = opportunity_ranking
    base.main()


if __name__ == "__main__":
    main()
