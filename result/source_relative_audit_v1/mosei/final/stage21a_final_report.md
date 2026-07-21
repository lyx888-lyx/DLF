# Stage 21A Source-Relative Learning Feasibility Audit v1

- Final status: `STAGE21A_FAILED_TRUE_SOURCE_NOT_BETTER_THAN_CONTROLS`
- Source-relative claim supported: **False**
- Generic pairwise-only signal: **False**
- Recommend Stage21B: **False**
- Official Valid probe run: **False**
- Locked Test access count: **0**
- Uniform epoch-2 metric parity: **True** (max metric difference `0.000e+00`, J difference `0.000e+00`)

## Data and pair feasibility

| Split | Samples | Source videos | Mean clips/source | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Train | 16326 | 2249 | 7.2592 | 5.0 | 15.0 | 98 |
| Official Valid | 1871 | 300 | 6.2367 | 4.5 | 13.0 | 39 |

- Train/Valid source overlap: `0`; sample overlap: `0`.
- `video_id` is reliably parsed from the pkl sample ID and bound to the Stage19 cache order; it is not a speaker ID.
- Selected Train-only label gap: **1.0**.
- Pair coverage: **84.99% samples**, **76.88% multi-clip sources**; 29615 balanced retained pairs.

## Residual ICC

| Split | Mode | ICC | shuffled p95 | p-value | 95% bootstrap CI | Gate |
|---|---|---:|---:|---:|---:|---|
| train | LAV | 0.072280 | 0.008707 | 0.00050 | [0.056597, 0.086581] | True |
| train | LA | 0.095665 | 0.008701 | 0.00050 | [0.079057, 0.112799] | True |
| train | LV | 0.090304 | 0.009220 | 0.00050 | [0.074047, 0.106800] | True |
| train | L | 0.114087 | 0.008233 | 0.00050 | [0.096332, 0.130943] | True |
| valid | LAV | 0.118963 | 0.024596 | 0.00050 | [0.072505, 0.162514] | True |
| valid | LA | 0.130201 | 0.024427 | 0.00050 | [0.083689, 0.174729] | True |
| valid | LV | 0.126327 | 0.025521 | 0.00050 | [0.079634, 0.169861] | True |
| valid | L | 0.137694 | 0.025909 | 0.00050 | [0.092469, 0.178649] | True |

## Source fingerprint

| Mode | Mean AUROC | Balanced accuracy | Shuffled AUROC |
|---|---:|---:|---:|
| LAV | 0.567529 | 0.536041 | 0.465101 |
| LA | 0.537329 | 0.521716 | 0.535505 |
| LV | 0.565701 | 0.531756 | 0.488187 |
| L | 0.524237 | 0.514160 | 0.484985 |

Source identity separability is diagnostic only and cannot establish usefulness of source-relative supervision.

## Train-only shared-head probes

| Split | P1-P0 J | P1-P2 J | P1-P3 J | P(better P0) | Without largest 5% sources |
|---:|---:|---:|---:|---:|---:|
| 2101 | +0.003964 | -0.021282 | -0.003069 | 0.000 | +0.003008 |
| 2102 | +0.003677 | -0.007908 | -0.015595 | 0.000 | +0.002916 |
| **Mean** | **+0.003820** | **-0.014595** | **-0.009332** | — | — |

Missing-mode MAE improvements versus P0: **none**.

## Mean P1-P0 metric deltas

| Mode | MAE | Corr | Acc7 | Acc5 | Acc2 | F1 |
|---|---:|---:|---:|---:|---:|---:|
| LAV | +0.004363 | -0.000693 | -0.004500 | -0.004794 | +0.000354 | +0.000392 |
| LA | +0.002993 | -0.000310 | -0.003370 | -0.003202 | -0.000411 | -0.000353 |
| LV | +0.003603 | -0.000671 | -0.001387 | -0.000945 | -0.001700 | -0.001672 |
| L | +0.003239 | -0.000353 | -0.004842 | -0.003784 | -0.002352 | -0.002308 |
| MissingMacro | +0.003278 | -0.000445 | -0.003200 | -0.002644 | -0.001488 | -0.001445 |

## Official Valid and gradients

- Official Valid probe was not run because the frozen Train-only gate failed.
- Relative vs Uniform KD: median cosine `-0.050547`, negative ratio `55.00%`, risk `NOT_HIGH`.
- Relative vs CFCompatKD: median cosine `-0.015477`, negative ratio `50.00%`, risk `NOT_HIGH`.
- Gradient audit used 20 true-source batches and performed zero optimizer steps.

## Required answers

1. Train/Valid source videos: **2249/300**.
2. Mean clips/source: **7.2592/6.2367**.
3. Train/Valid source overlap: **0**.
4. ID binding: **reliable and SHA-verified**.
5. Selected delta: **1.0**.
6. Pair coverage: **84.99% samples / 76.88% eligible sources**.
7–8. Per-mode ICC, shuffled controls and CIs are reported above; Train residual gate: **True**.
9. Source fingerprint distinguishable: **True** (diagnostic only).
10. P1 better than P0 at required margin: **False**.
11. P1 better than P2 at required margin: **True**.
12. P1 better than P3 at required margin: **True**.
13. Generic pairwise-only effect: **False**.
14. Two train-only splits direction-consistent against controls: **True**.
15–16. Official Valid run/confirmed: **False/False**.
17. Missing modes improved: **none**.
18. MAE/Corr/classification trade-offs: see the complete delta table above.
19. Direction survives removal of largest sources: **False**.
20. Uniform KD conflict risk: **NOT_HIGH**.
21. CFCompatKD conflict risk: **NOT_HIGH**.
22. A CFCompatKD × Source-Relative 2×2 experiment is justified: **False**.
23. Recommend Stage21B: **False**.
24. Locked Test access count: **0**.

## Decision

`STAGE21A_FAILED_TRUE_SOURCE_NOT_BETTER_THAN_CONTROLS`

No formal Source-Relative DLF was implemented or trained in this stage.
