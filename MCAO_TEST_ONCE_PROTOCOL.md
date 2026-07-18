# MCAO Test-Once Protocol

Stage17A through Stage17D may construct only train and valid data surfaces.
Every manifest records all test-access flags as false and locked test count zero.

The test surface is unlocked only after all evidence gates pass, the PM/AN
variant is frozen from train-only evidence, all five validation runs and tests
pass, the branch is pushed and clean, and `TEST_UNLOCK_MANIFEST.json` is
generated. The locked test then runs once and cannot change the objective.
