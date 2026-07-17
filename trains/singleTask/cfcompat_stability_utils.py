"""Frozen utilities for Stage 8 Stability-First CFCompatKD.

This module only maintains evaluation-time state derived from one unchanged
online CFCompatKD trajectory.  It does not define a new loss, teacher, network,
optimizer, scheduler, or data transformation.
"""
import copy
import hashlib
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EMA_DECAY = 0.999
SOUP_TOP_K = (3, 5)
STAGE8_SEEDS = (1111, 1112, 1113, 1114, 1115)
METHOD_PRIORITY = {"EMA": 0, "Soup-3": 1, "Soup-5": 2}
MISSING_MODE_INDEX = {"LA": 0, "LV": 1, "L": 2}


def capture_rng_state():
    """Capture every RNG used by the frozen Stage 3 training process."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_states_equal(left, right):
    if left["python"] != right["python"]:
        return False
    if left["numpy"][0] != right["numpy"][0]:
        return False
    if not np.array_equal(left["numpy"][1], right["numpy"][1]):
        return False
    if left["numpy"][2:] != right["numpy"][2:]:
        return False
    if not torch.equal(left["torch"], right["torch"]):
        return False
    return len(left["cuda"]) == len(right["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])
    )


@contextmanager
def preserve_rng_state():
    """Make auxiliary EMA/evidence work observationally invisible to RNG."""
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def initialize_ema(online):
    """Copy the initial Student without consuming RNG and freeze the copy."""
    before = capture_rng_state()
    with preserve_rng_state():
        ema = copy.deepcopy(online)
    after = capture_rng_state()
    if not rng_states_equal(before, after):
        raise RuntimeError("EMA initialization changed RNG state.")
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, online, decay=EMA_DECAY):
    """Average model parameters and exactly copy every model buffer."""
    if float(decay) != EMA_DECAY:
        raise ValueError("Stage 8 fixes EMA decay at 0.999.")

    ema_parameters = dict(ema.named_parameters())
    online_parameters = dict(online.named_parameters())
    if tuple(ema_parameters) != tuple(online_parameters):
        raise RuntimeError("EMA and online parameter keys differ.")
    for key, target in ema_parameters.items():
        source = online_parameters[key].detach()
        if target.shape != source.shape or target.dtype != source.dtype:
            raise RuntimeError("EMA parameter mismatch at {}.".format(key))
        if target.is_floating_point():
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            target.copy_(source)

    # DLF has no BatchNorm/running-stat buffers. Some transformer modules do
    # register floating ``_float_tensor`` device/dtype anchors whose scalar
    # values are intentionally irrelevant and may be uninitialized. They are
    # not parameters and must not be numerically averaged.
    ema_buffers = dict(ema.named_buffers())
    online_buffers = dict(online.named_buffers())
    if tuple(ema_buffers) != tuple(online_buffers):
        raise RuntimeError("EMA and online buffer keys differ.")
    for key, target in ema_buffers.items():
        source = online_buffers[key].detach()
        if target.shape != source.shape or target.dtype != source.dtype:
            raise RuntimeError("EMA buffer mismatch at {}.".format(key))
        target.copy_(source)


def optimizer_step_and_update_ema(optimizer, ema, online):
    """The only Stage8 optimizer-step wrapper: online first, exactly one EMA update."""
    optimizer.step()
    update_ema(ema, online, decay=EMA_DECAY)


def clone_state_cpu(model):
    """Clone one Student state to CPU without changing RNG."""
    with preserve_rng_state():
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }


def parameter_distance(left_model, right_model):
    """Root-mean-square distance over floating named parameters only."""
    left = dict(left_model.named_parameters())
    right = dict(right_model.named_parameters())
    return state_distance(left, right, parameter_keys=tuple(left))


def state_distance(left, right, parameter_keys=None):
    """RMS distance, optionally restricted to explicit parameter keys."""
    if tuple(left) != tuple(right):
        raise RuntimeError("State keys differ while computing distance.")
    keys = tuple(left) if parameter_keys is None else tuple(parameter_keys)
    missing = [key for key in keys if key not in left or key not in right]
    if missing:
        raise RuntimeError(
            "Distance parameter keys are absent: {}.".format(missing)
        )
    squared, count = 0.0, 0
    for key in keys:
        a, b = left[key].detach().cpu(), right[key].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("State mismatch at {}.".format(key))
        if a.is_floating_point():
            delta = a.to(torch.float64) - b.to(torch.float64)
            squared += float(delta.pow(2).sum())
            count += delta.numel()
        elif not torch.equal(a, b):
            raise RuntimeError("Non-floating state differs at {}.".format(key))
    return float(np.sqrt(squared / max(count, 1)))


def rank_trajectory(entries):
    """Rank online checkpoints by validation J, then earlier epoch."""
    return sorted(entries, key=lambda entry: (float(entry["J_valid"]), int(entry["Epoch"])))


def consider_trajectory_checkpoint(entries, model, seed, epoch, j_valid):
    """Keep only the five best online states from one trajectory."""
    ranked = rank_trajectory(entries)
    candidate_key = (float(j_valid), int(epoch))
    if len(ranked) >= 5:
        worst_key = (float(ranked[-1]["J_valid"]), int(ranked[-1]["Epoch"]))
        if candidate_key >= worst_key:
            return ranked
    metadata = {
        "Seed": int(seed),
        "Epoch": int(epoch),
        "J_valid": float(j_valid),
        "SourceMethod": "Online",
    }
    metadata["state"] = clone_state_cpu(model)
    return rank_trajectory(entries + [metadata])[:5]


def validate_soup_sources(entries, expected_seed):
    if len(entries) < max(SOUP_TOP_K):
        raise RuntimeError("Trajectory Soup requires five online checkpoints.")
    for entry in entries:
        if int(entry["Seed"]) != int(expected_seed):
            raise RuntimeError("Cross-seed soup is forbidden.")
        if entry.get("SourceMethod") != "Online":
            raise RuntimeError("Trajectory Soup may only use online checkpoints.")
        if "state" not in entry:
            raise RuntimeError("Soup source state is missing.")


def uniform_soup(entries, top_k, expected_seed):
    """Uniformly average same-seed online states in CPU FP64."""
    if int(top_k) not in SOUP_TOP_K:
        raise ValueError("Stage 8 only permits Top-3 and Top-5 soup.")
    ranked = rank_trajectory(entries)
    validate_soup_sources(ranked, expected_seed)
    selected = ranked[: int(top_k)]
    reference = selected[0]["state"]
    keys = tuple(reference)
    result = {}
    for entry in selected[1:]:
        state = entry["state"]
        if tuple(state) != keys:
            raise RuntimeError("Soup source parameter keys differ.")
        for key in keys:
            if state[key].shape != reference[key].shape:
                raise RuntimeError("Soup source shape mismatch at {}.".format(key))
            if state[key].dtype != reference[key].dtype:
                raise RuntimeError("Soup source dtype mismatch at {}.".format(key))
    for key in keys:
        tensors = [entry["state"][key].detach().cpu() for entry in selected]
        if reference[key].is_floating_point():
            averaged = torch.stack(
                [tensor.to(torch.float64) for tensor in tensors], dim=0
            ).mean(dim=0)
            result[key] = averaged.to(reference[key].dtype)
        else:
            if not all(torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise RuntimeError(
                    "Non-floating soup source state differs at {}.".format(key)
                )
            result[key] = tensors[0].clone()
    return result, selected


class MissingSequenceHasher:
    """Hash the exact Stage 3 LA/LV/L draw sequence."""

    def __init__(self):
        self._hash = hashlib.sha256()
        self.count = 0

    def update(self, modes):
        values = bytes(MISSING_MODE_INDEX[mode] for mode in modes)
        self._hash.update(values)
        self.count += len(values)

    def hexdigest(self):
        return self._hash.hexdigest()


def expected_missing_sequence_sha(seed, epochs, batch_sizes):
    generator = torch.Generator().manual_seed(int(seed) + 104729)
    hasher = hashlib.sha256()
    count = 0
    for _ in range(int(epochs)):
        for batch_size in batch_sizes:
            values = torch.randint(
                3, (int(batch_size),), generator=generator, dtype=torch.int64
            )
            hasher.update(bytes(int(value) for value in values.tolist()))
            count += int(batch_size)
    return hasher.hexdigest(), count


def stage3_result_path(result_root, seed):
    base = Path(result_root) / "missing_baseline" / "cf_compat_kd_v1"
    if int(seed) == 1111:
        return base / "benchmark_train" / "mosi_per_seed.csv"
    return (
        base / "benchmark_multiseed" / "seed{}".format(int(seed))
        / "mosi_per_seed.csv"
    )


def load_stage3_reference(result_root, seed):
    path = stage3_result_path(result_root, seed)
    if not path.is_file():
        raise FileNotFoundError("Stage 3 reference is absent: {}".format(path))
    frame = pd.read_csv(path)
    selected = frame.loc[frame.Seed.astype(int).eq(int(seed))]
    if len(selected) != 1:
        raise RuntimeError("Stage 3 seed{} reference is not unique.".format(seed))
    return selected.iloc[0], path


def verify_online_replay(current, reference, tolerance=1e-4):
    """Apply the per-seed Stage 8 online replay gate."""
    differences = {
        "J_valid": abs(float(current["J_valid"]) - float(reference["J_valid"])),
        "J_test_at_valid_best": abs(
            float(current["J_test_at_valid_best"])
            - float(reference["J_test_at_valid_best"])
        ),
    }
    for mode in ("LAV", "LA", "LV", "L"):
        key = "test_at_valid_best_{}_MAE".format(mode)
        differences[key] = abs(float(current[key]) - float(reference[key]))
    epoch_match = (
        int(current["BestValidEpoch"]) == int(reference["BestValidEpoch"])
    )
    passed = epoch_match and all(value <= tolerance for value in differences.values())
    return {
        "Passed": bool(passed),
        "EpochMatch": bool(epoch_match),
        "Tolerance": float(tolerance),
        "Differences": differences,
    }


def select_main_strategy(per_seed):
    """Select one global strategy using mean validation J only."""
    candidates = per_seed.loc[
        per_seed.Method.isin(("EMA", "Soup-3", "Soup-5"))
    ]
    if set(candidates.Method) != {"EMA", "Soup-3", "Soup-5"}:
        raise RuntimeError("MainStrategy candidates are incomplete.")
    means = candidates.groupby("Method").J_valid.mean().to_dict()
    selected = min(
        means, key=lambda method: (float(means[method]), METHOD_PRIORITY[method])
    )
    return selected, means


def markdown_table(frame):
    def cell(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return "{:.6g}".format(float(value))
        return str(value).replace("|", r"\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)
