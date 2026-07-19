"""Preregistered Stage 18 KD controls with deterministic sample bindings."""

import hashlib

import numpy as np

from trains.singleTask.missing_utils import MISSING_MODES


CONTROL_METHODS = (
    "moddrop",
    "uniform",
    "equal_mass",
    "mode_mean",
    "shuffled_gate",
    "shuffled_teacher",
    "oracle",
    "cfcompat",
)


def deterministic_permutation(indices, seed, epoch, mode, tag):
    indices = [int(index) for index in indices]
    keys = []
    for position, index in enumerate(indices):
        payload = "{}|{}|{}|{}|{}".format(tag, seed, epoch, mode, index)
        keys.append(
            (
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                index,
                position,
            )
        )
    permutation = [entry[2] for entry in sorted(keys)]
    if len(permutation) > 1 and permutation == list(range(len(permutation))):
        permutation = permutation[1:] + permutation[:1]
    return permutation


def oracle_direction(teacher, evaluator, label):
    return float((float(teacher) - float(evaluator)) * (float(label) - float(evaluator)) > 0)


def build_epoch_bindings(
    method,
    seed,
    epoch,
    indices,
    modes,
    cache_by_index,
    teacher_by_index,
    label_by_index,
):
    if method not in CONTROL_METHODS:
        raise ValueError("Unknown Stage18 control: {}".format(method))
    if len(indices) != len(modes) or len(set(indices)) != len(indices):
        raise ValueError("Epoch schedule must contain one mode for every unique sample.")
    base_gate = {
        int(index): float(
            cache_by_index[int(index)]["compat_{}".format(mode)]
        )
        for index, mode in zip(indices, modes)
    }
    gate = {}
    teacher_source = {int(index): int(index) for index in indices}
    if method == "moddrop":
        gate = {int(index): 0.0 for index in indices}
    elif method == "uniform":
        gate = {int(index): 1.0 for index in indices}
    elif method == "equal_mass":
        mean = float(np.mean([base_gate[int(index)] for index in indices]))
        gate = {int(index): mean for index in indices}
    elif method in ("cfcompat", "shuffled_teacher"):
        gate = dict(base_gate)
    elif method == "oracle":
        for index, mode in zip(indices, modes):
            index = int(index)
            evaluator = cache_by_index[index][
                "evaluator_{}_pred".format(mode)
            ]
            gate[index] = oracle_direction(
                teacher_by_index[index],
                evaluator,
                label_by_index[index],
            )
    else:
        for mode in MISSING_MODES:
            local = [
                int(index)
                for index, observed in zip(indices, modes)
                if observed == mode
            ]
            values = [base_gate[index] for index in local]
            if method == "mode_mean":
                mean = float(np.mean(values))
                for index in local:
                    gate[index] = mean
            else:
                permutation = deterministic_permutation(
                    local, seed, epoch, mode, "shuffled_gate"
                )
                for index, donor_position in zip(local, permutation):
                    gate[index] = values[donor_position]
    if method == "shuffled_teacher":
        for mode in MISSING_MODES:
            local = [
                int(index)
                for index, observed in zip(indices, modes)
                if observed == mode
            ]
            permutation = deterministic_permutation(
                local, seed, epoch, mode, "shuffled_teacher"
            )
            for index, donor_position in zip(local, permutation):
                teacher_source[index] = local[donor_position]

    mass = {"total": float(sum(gate.values()))}
    reference = {"total": float(sum(base_gate.values()))}
    for mode in MISSING_MODES:
        local = [
            int(index)
            for index, observed in zip(indices, modes)
            if observed == mode
        ]
        mass[mode] = float(sum(gate[index] for index in local))
        reference[mode] = float(sum(base_gate[index] for index in local))
    if method == "equal_mass" and abs(mass["total"] - reference["total"]) > 1e-8:
        raise AssertionError("Equal-Mass global KD mass mismatch.")
    if method in ("mode_mean", "shuffled_gate"):
        for mode in MISSING_MODES:
            if abs(mass[mode] - reference[mode]) > 1e-8:
                raise AssertionError("{} {} KD mass mismatch.".format(method, mode))
    return {
        "gate": gate,
        "teacher_source": teacher_source,
        "base_gate": base_gate,
        "mass": mass,
        "reference_mass": reference,
        "gate_sha256": _mapping_sha(gate),
        "teacher_binding_sha256": _mapping_sha(teacher_source),
    }


def _mapping_sha(mapping):
    digest = hashlib.sha256()
    for key in sorted(mapping):
        digest.update("{}={:.17g}\n".format(int(key), float(mapping[key])).encode())
    return digest.hexdigest()
