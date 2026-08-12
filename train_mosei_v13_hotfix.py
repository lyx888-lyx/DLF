"""Narrow execution hotfixes for the MOSEI v13 frozen port.

Two port-only audit bugs are corrected here without changing the frozen v13
training algorithm:

1. Initial clean LAV equivalence must compare teacher/student in eval mode;
   otherwise student-side DLF dropout makes a deterministic-equivalence audit
   fail before training starts.
2. The historical mechanism-transfer helper hard-coded MOSI Valid as
   229 clips x 3 missing modes = 687 events.  MOSEI has 1871 Valid clips, so
   the corresponding exact audit is 1871 x 3 = 5613 events.

No loss, optimizer, RNG stream, fold assignment, checkpoint selector, residual
architecture, gradient surgery, Adam-step safety projection, consensus rule,
blend rule, or data-split policy is changed.
"""
from __future__ import annotations

import train_mosei_v13 as impl
import trains.singleTask.cfcompat_adapter_isolation_utils as adapter_utils
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence as _original_assert_initial_lav_equivalence,
)
from trains.singleTask.missing_utils import MISSING_MODES


def _eval_mode_initial_lav_equivalence(
    teacher,
    student,
    text,
    audio,
    vision,
    rtol=1e-5,
    atol=1e-6,
):
    teacher_was_training = bool(teacher.training)
    student_was_training = bool(student.training)
    try:
        teacher.eval()
        student.eval()
        return _original_assert_initial_lav_equivalence(
            teacher,
            student,
            text,
            audio,
            vision,
            rtol=rtol,
            atol=atol,
        )
    finally:
        teacher.train(teacher_was_training)
        student.train(student_was_training)


def _mosei_mechanism_transfer_summary(events, run):
    """MOSEI-sized version of the frozen v7/v13 transfer diagnostic.

    The statistic itself is unchanged.  Only the dataset-size integrity audit
    is generalized from the MOSI constant 229 to the frozen MOSEI Valid count.
    """
    required = {"Seed", "Run", "Mode", "sample_index", "teacher_advantage", "gain_vs_dlf"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(
            "Valid events lack mechanism columns: {}".format(sorted(missing))
        )

    local = events.loc[
        events.Seed.astype(int).eq(impl.DEV_SEED)
        & events.Run.astype(str).eq(str(run))
        & events.Mode.astype(str).isin(MISSING_MODES)
    ].copy()

    expected_per_mode = int(impl.EXPECTED_VALID_N)
    expected_total = expected_per_mode * len(MISSING_MODES)
    if len(local) != expected_total:
        raise RuntimeError(
            "Expected exactly {} Seed{} MOSEI missing-mode Valid events, got {}.".format(
                expected_total,
                impl.DEV_SEED,
                len(local),
            )
        )

    duplicate = local.duplicated(["sample_index", "Mode"])
    if bool(duplicate.any()):
        raise RuntimeError("MOSEI v13 Valid mechanism events contain duplicate sample/mode rows.")

    counts = local.groupby("Mode", sort=False).size().to_dict()
    for mode in MISSING_MODES:
        observed = int(counts.get(mode, 0))
        if observed != expected_per_mode:
            raise RuntimeError(
                "MOSEI v13 Valid mechanism count mismatch mode={}: {} != {}.".format(
                    mode,
                    observed,
                    expected_per_mode,
                )
            )

    beneficial = local.teacher_advantage >= adapter_utils.DISTILL_MARGIN
    if not bool(beneficial.any()) or not bool((~beneficial).any()):
        raise RuntimeError(
            "MOSEI v13 transfer diagnostic requires both Teacher-beneficial and nonbeneficial events."
        )

    return {
        "teacher_beneficial_prevalence": float(beneficial.mean()),
        "all_missing": adapter_utils._subset_transfer(local),
        "teacher_beneficial": adapter_utils._subset_transfer(local.loc[beneficial]),
        "teacher_nonbeneficial": adapter_utils._subset_transfer(local.loc[~beneficial]),
    }


# train_mosei_v13 imported these symbols directly, so patch its module-local bindings.
impl.assert_initial_lav_equivalence = _eval_mode_initial_lav_equivalence
impl.mechanism_transfer_summary = _mosei_mechanism_transfer_summary


if __name__ == "__main__":
    impl.main()
