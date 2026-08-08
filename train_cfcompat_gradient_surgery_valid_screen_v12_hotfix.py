"""Numerical-hotfix entrypoint for the frozen CFCompatKD v12 experiment.

The experiment definition is unchanged.  This wrapper replaces only the v12
projection implementation with a float64/relative-tolerance equivalent that
avoids false float32 post-projection assertion failures.
"""
from __future__ import annotations

import train_cfcompat_gradient_surgery_valid_screen_v12 as v12
from trains.singleTask.cfcompat_gradient_surgery_numerical_hotfix import (
    asymmetric_project_supervised_stable,
)


# train_one_fold_surgery resolves this function from the v12 module globals at
# runtime, so replacing it here changes only numerical realization of the same
# projection formula.  No loss, gate, fold, selector, optimizer, or inference
# rule is changed.
v12.asymmetric_project_supervised = asymmetric_project_supervised_stable
v12.VERSION = "cfcompat_gradient_surgery_valid_screen_v12_numeric_hotfix"


if __name__ == "__main__":
    v12.main()
