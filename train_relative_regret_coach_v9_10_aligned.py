"""Run V9.10 with deployment/OOF scale semantics aligned before imports."""

from trains.singleTask.relative_regret_runtime_patch_v910 import (
    install_relative_regret_runtime_patch,
)

install_relative_regret_runtime_patch()

from train_relative_regret_coach_v9_10 import main  # noqa: E402


if __name__ == "__main__":
    main()
