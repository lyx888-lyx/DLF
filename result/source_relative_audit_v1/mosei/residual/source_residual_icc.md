# Stage 21A source residual ICC audit

- Train-only residual gate: **True**
- Passing Train modes: LAV, LA, LV, L
- 2,000 group-size-preserving permutations and 2,000 source-cluster bootstraps per split/mode.

| Split | Mode | ICC | shuffled p95 | percentile | p-value | bootstrap 95% CI | without largest 5% | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| train | LAV | 0.072280 | 0.008707 | 1.000 | 0.00050 | [0.056597, 0.086581] | 0.063212 | True |
| train | LA | 0.095665 | 0.008701 | 1.000 | 0.00050 | [0.079057, 0.112799] | 0.085038 | True |
| train | LV | 0.090304 | 0.009220 | 1.000 | 0.00050 | [0.074047, 0.106800] | 0.079126 | True |
| train | L | 0.114087 | 0.008233 | 1.000 | 0.00050 | [0.096332, 0.130943] | 0.101383 | True |
| valid | LAV | 0.118963 | 0.024596 | 1.000 | 0.00050 | [0.072505, 0.162514] | 0.127888 | True |
| valid | LA | 0.130201 | 0.024427 | 1.000 | 0.00050 | [0.083689, 0.174729] | 0.140841 | True |
| valid | LV | 0.126327 | 0.025521 | 1.000 | 0.00050 | [0.079634, 0.169861] | 0.134260 | True |
| valid | L | 0.137694 | 0.025909 | 1.000 | 0.00050 | [0.092469, 0.178649] | 0.147306 | True |
