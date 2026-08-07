"""Fast synthetic checks for CFCompatKD v10 conservative selection."""
from trains.singleTask.cfcompat_conservative_crossfit_residual_utils import (
    NEAR_OPTIMAL_REL_TOL,
    select_earliest_near_optimal,
)


def main():
    rows = [
        {"Epoch": 1, "HoldoutJ": 0.1900},
        {"Epoch": 2, "HoldoutJ": 0.1810},
        {"Epoch": 3, "HoldoutJ": 0.1790},
        {"Epoch": 4, "HoldoutJ": 0.1780},
        {"Epoch": 5, "HoldoutJ": 0.1779},
    ]
    selected = select_earliest_near_optimal(rows)
    # best=0.1779, 1% cutoff=0.179679, so epoch 3 is the first acceptable epoch.
    assert selected["absolute_best_epoch"] == 5
    assert selected["selected_epoch"] == 3
    assert selected["selected_holdout_J"] == 0.1790
    assert selected["epoch_reduction_vs_absolute_best"] == 2
    assert selected["selected_relative_J_degradation"] <= NEAR_OPTIMAL_REL_TOL + 1e-12

    strict = select_earliest_near_optimal(rows, rel_tol=0.0)
    assert strict["selected_epoch"] == 5

    flat = [
        {"Epoch": 1, "HoldoutJ": 0.1005},
        {"Epoch": 2, "HoldoutJ": 0.1000},
        {"Epoch": 3, "HoldoutJ": 0.0999},
    ]
    flat_selected = select_earliest_near_optimal(flat)
    assert flat_selected["selected_epoch"] == 1
    assert flat_selected["absolute_best_epoch"] == 3

    print("v10 conservative selector smoke passed")
    print("relative tolerance:", NEAR_OPTIMAL_REL_TOL)
    print("example absolute best epoch:", selected["absolute_best_epoch"])
    print("example conservative epoch:", selected["selected_epoch"])


if __name__ == "__main__":
    main()
