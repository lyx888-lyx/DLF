# Stage 21A failure diagnosis

- Final status: `STAGE21A_FAILED_TRUE_SOURCE_NOT_BETTER_THAN_CONTROLS`
- Source residual gate passed: `True`
- Train-only probe gate passed: `False`
- Pair controls valid: `True`
- Mean P1-P0 J: `+0.003820`
- Mean P1-P2 J: `-0.014595`
- Mean P1-P3 J: `-0.009332`
- Generic pairwise-only pattern: `False`
- Official Valid probe run: `False`

No threshold, delta, split, ridge alpha, relative mass, representation, or control was changed after inspecting results.
The true-source probe beat both relative controls but was worse than the pointwise P0 control on both splits; that pointwise failure closes the route.
