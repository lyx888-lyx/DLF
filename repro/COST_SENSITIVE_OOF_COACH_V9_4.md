# V9.4 Fully Nested OOF Cost-Sensitive Coach

## Goal

V9.3 showed that the five-region expert pool has a strong true-region upper
bound, but a region classifier and four Validation-only advantage heads do not
route reliably. V9.4 changes the supervision target from region membership or
predicted gain to the final action cost itself.

The five actions are:

```text
anchor
strong_negative
boundary
positive
strong_positive
```

For Train sample `i` and action `k`, the target cost is:

```text
cost[i,k] = abs(OOF_prediction[i,k] - label[i])
```

Every action prediction used in this cost table is out of fold with respect to
the complete source video/conversation group.

## Fully nested leakage boundary

V9.4 reuses the audited V9.2 top-level grouped OOF cache. For each top-level
outer holdout:

1. remove all top-holdout videos from the development data;
2. run a new grouped inner OOF CFCompatKD pipeline only on the development data;
3. train four low-capacity residual specialists by cross-fitting over the inner
   OOF cache;
4. use those inner OOF specialist predictions to build development action costs;
5. train and select a fold-local cost-sensitive coach only on development data;
6. refit the four specialists on all development inner-OOF samples;
7. predict the untouched top-level holdout with the refit specialists and coach.

Therefore the top holdout is excluded from:

- clean DLF training;
- ModDrop evaluator training;
- compatibility fitting;
- CFCompatKD training and checkpoint selection;
- specialist training;
- coach training and epoch selection.

After all top folds finish, every Train sample has a fully nested OOF Anchor,
four specialist predictions, and five action costs.

## Unified V9.4 specialists

V9.4 uses one common low-capacity residual architecture for all four specialist
actions. This replaces the architecture mismatch between V9 Boundary/Positive
and V9.2 tail heads for the coach-training protocol.

The specialist regions are:

```text
strong_negative: y < -1.5
boundary:        abs(y) <= 0.5
positive:        0.5 < y <= 1.5
strong_positive: y > 1.5
```

The ordinary-negative action remains the immutable Anchor.

For final deployment, each specialist is refit on the complete V9.2 Train OOF
cache for a fixed pre-registered epoch count. A specialist is enabled only if
its fully nested OOF designated-region MAE beats the OOF Anchor.

## Cost-sensitive coach

The coach receives only label-free quantities:

- four aligned DLF function-space logits;
- Anchor prediction and magnitude;
- branch mean, standard deviation, and range;
- four specialist predictions;
- four specialist corrections;
- four specialist applicability probabilities.

It outputs five action probabilities. Its main objective minimizes expected OOF
action cost directly:

```text
sum_k p(action=k | x) * cost[k]
```

Additional losses distill the soft minimum-cost action distribution, penalize
routing away from Anchor when no specialist has a clear gain, and preserve an
ordinal sentiment-strength auxiliary task.

The ordinal task helps the coach understand emotional intensity, but it no
longer hard-codes the final action.

## Validation-only deployment calibration

The final specialists and coach are refit on Train OOF data. Validation selects:

- action-logit temperature;
- minimum top-action probability;
- minimum top-vs-second probability margin;
- Anchor-to-specialist blend coefficient `beta`.

If the proposed specialist fails confidence or margin thresholds, the deployed
action remains Anchor. The calibration grid explicitly contains the all-Anchor
policy, so a harmful coach is not forced into deployment.

Test labels are used only for final metrics and named diagnostic upper bounds.

## Cost

This version is substantially more expensive than V9.3. With three top folds
and three inner folds, it trains nine additional inner CFCompatKD pipelines.
Each pipeline contains Clean DLF, ModDrop, and CFCompatKD stages, for 27 large
model-training stages in total.

The large checkpoints can require roughly 12–15 GB of additional disk space.
Each top fold and every inner OOF fold is resumable. Re-running the same command
reuses completed caches.

## Required previous output

V9.2 must already exist:

```text
result/oof_tail_residual_experts_v92/mosi/seed_1111/
```

In particular:

```text
oof_cfcompat/nested_grouped_oof_cfcompat_cache_v92.pth
oof_tail_residual_experts_v92_summary.json
```

V9.3 outputs are not required.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/cost-sensitive-oof-coach-v9-4
git pull origin innovation/cost-sensitive-oof-coach-v9-4

python3 scripts/smoke_test_cost_sensitive_oof_coach_v9_4.py

GPU=0 \
SEED=1111 \
INNER_FOLDS=3 \
bash scripts/run_mosi_cost_sensitive_oof_coach_v9_4.sh \
2>&1 | tee mosi_cost_sensitive_oof_coach_v94_seed1111.log
```

## Output root

```text
result/cost_sensitive_oof_coach_v94/mosi/seed_1111/
```

Important files:

```text
oof_expert_pool/fully_nested_oof_expert_pool_v94.pth
oof_expert_pool/fully_nested_oof_expert_pool_v94.csv
oof_expert_pool/fully_nested_oof_expert_pool_v94_summary.json
oof_expert_pool/v94_top_outer_manifest.csv
oof_expert_pool/outer_fold_*/development_inner_oof/
oof_expert_pool/outer_fold_*/v94_inner_specialist_crossfit_history.csv
oof_expert_pool/outer_fold_*/v94_inner_coach_history.csv

v94_nested_oof_specialist_capability.csv
v94_final_coach_history.csv
v94_route_calibration.csv
v94_test_summary.csv
cost_sensitive_oof_coach_v94_predictions.csv
cost_sensitive_oof_coach_v94_summary.json
```

## Interpretation order

1. Engineering audit passes.
2. Inspect nested OOF specialist designated-region gains. A specialist with no
   positive OOF gain is automatically disabled.
3. Compare nested OOF Anchor and sample-oracle action costs.
4. Inspect the Validation-selected route activation rate. A near-100% activation
   rate is suspicious unless the Validation gain is large and stable.
5. Compare Test Anchor, category baselines, and the deployable cost-sensitive
   coach.
6. True-region and sample-oracle results remain diagnostics only.

The default audit reports a warning rather than raising merely because Test gain
is negative. Use `--require-deployable-gain 0` when a strict scientific gate is
intended after the design is frozen.
