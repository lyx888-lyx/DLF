"""Narrow runtime hotfix for the MOSEI CFCompat-v1 diagnostic gate summary.

The frozen CFCompat-v1 port uses gate == compatibility.  The main trainer's
per-sample diagnostic records store the value under ``compat`` while the epoch
summary helper expects a ``gate`` field.  This wrapper normalizes that diagnostic
field before delegating to the original helper.  Training losses, gradients,
optimizer steps, checkpoint selection, RNG, cache semantics, and data splits are
unchanged.
"""
from __future__ import annotations

import train_mosei_cfcompat_v1 as base


_original_gate_epoch_summary = base.gate_epoch_summary


def _gate_epoch_summary_hotfix(records, seed, epoch):
    normalized = []
    for record in records:
        if "gate" in record:
            normalized.append(record)
            continue
        if "compat" not in record:
            raise RuntimeError("CFCompat diagnostic record has neither gate nor compat field.")
        fixed = dict(record)
        fixed["gate"] = fixed["compat"]
        normalized.append(fixed)
    return _original_gate_epoch_summary(normalized, seed, epoch)


base.gate_epoch_summary = _gate_epoch_summary_hotfix


if __name__ == "__main__":
    base.main()
