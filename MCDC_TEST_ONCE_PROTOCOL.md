# MCDC Test-Once Protocol

Before a valid `TEST_UNLOCK_MANIFEST.json` exists, Stage 15 code must not
construct a test loader or read test IDs, features, labels, or predictions.
Every pre-unlock audit records all test-access flags as false and the locked test
access count as zero.

Unlock requires Stage15A, Stage15B, Stage15C, and Stage15D to pass; the five
matched methods, checkpoints, selected epochs, parser, representation extractor,
safe projection, evaluator, and hashes must be frozen; automated tests must pass;
the branch must be pushed and clean.

After unlock, the test context index is built without labels, predictions and
decision signatures are frozen and hashed, and labels are read only for the one
final evaluation. No later method modification or repeat test is permitted.
