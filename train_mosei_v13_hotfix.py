"""Narrow execution hotfix for the MOSEI v13 initialization audit.

The frozen teacher is evaluated in eval mode, while a newly constructed
MissingModalityWrapper defaults to train mode.  The LAV-equivalence assertion
must compare the same deterministic function, so run that assertion with both
modules in eval mode.  No training objective, optimizer, RNG stream, fold
assignment, checkpoint selector, safety projection, or data split is changed.
"""
from __future__ import annotations

import train_mosei_v13 as impl
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence as _original_assert_initial_lav_equivalence,
)


def _eval_mode_initial_lav_equivalence(teacher, student, text, audio, vision, rtol=1e-5, atol=1e-6):
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


# train_mosei_v13 imported this symbol directly, so patch its module-local binding.
impl.assert_initial_lav_equivalence = _eval_mode_initial_lav_equivalence


if __name__ == "__main__":
    impl.main()
