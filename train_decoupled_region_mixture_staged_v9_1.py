"""Run V9.1 with staged expert checkpoint and policy selection."""

from trains.singleTask.decoupled_region_mixture_staged_system_v91 import (
    StagedDecoupledRegionMixtureTrainerV91,
)

import train_decoupled_region_mixture_v9_1 as entry


if __name__ == "__main__":
    # The base entrypoint resolves this module global when constructing the
    # trainer, so replace it before calling main without duplicating CLI logic.
    entry.DecoupledRegionMixtureTrainerV91 = (
        StagedDecoupledRegionMixtureTrainerV91
    )
    entry.main()
