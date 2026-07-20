# Stage 19 retrospective best-checkpoint audit

- Status: `STAGE19_RETRO_MGD_CONFIRMED_CLOSED`
- Uniform selected epoch: 2; J=0.515493413
- MGD selected epoch: 8; J=0.532165666
- Delta best (MGD - Uniform): `+0.016672254`
- Missing modes with lower MAE: `0/3`
- Locked Test access count: `0`

| Mode | Delta MAE | Delta Corr | Delta Acc7 | Delta Acc5 | Delta Acc2 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| LAV | +0.018990 | -0.008989 | -0.018707 | -0.018172 | -0.006259 | -0.003722 |
| LA | +0.012487 | -0.005913 | -0.021913 | -0.021913 | -0.004172 | -0.000913 |
| LV | +0.018283 | -0.009012 | -0.026724 | -0.025655 | -0.007650 | -0.006038 |
| L | +0.012294 | -0.006140 | -0.025120 | -0.024051 | -0.004172 | -0.001276 |
| MissingMacro | +0.014355 | -0.007022 | -0.024586 | -0.023873 | -0.005331 | -0.002742 |

MGD is not retrained, MGRD is not run, and no tau or loss weight is changed.
