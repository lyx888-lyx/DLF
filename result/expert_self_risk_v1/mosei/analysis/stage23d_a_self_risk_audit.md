# Stage23D-A Expert Self-Risk Signal Audit

当前结论：**READY_WAITING_FOR_FREE_GPU**。Phase-1 为 **SELF_RISK_SIGNAL_WEAK**（通过 8/10 个冻结门槛），仅授权有限 A5 pilot；GPU 0/1/2 为 Stage23C 保留，GPU 3 正被外部任务占用，因此未启动 A5。

## 易懂结论

1. Expert 能否从内部状态预测自身误差？平均 Spearman=0.2404；按冻结门槛判定为 WEAK。
2. 内部状态是否明显优于只看 prediction/head disagreement？R2−R1 Spearman=+0.0344。
3. 哪些内部信号最有效？单 proxy 中 `hidden_centroid_distance` 的平均 Spearman 最高（0.1614）。
4. hidden-space 异常是否对应更高错误？centroid/knn/nearest-source proxy 的独立结果已写入 `individual_proxy_metrics.tsv`，不以单一汇总掩盖方向差异。
5. submode/perturbation instability 是否有价值？submode instability 已纳入 A4；A5 perturbation 仅在 Phase-1 为 WEAK/PASS 时授权。
6. 能否识别 worst-20% error？R2 mean AUROC=0.6240。
7. 能否识别 confident-wrong？R2 mean AUROC=0.8857。
8. 哪个 Expert 最有自知、哪个最差？按两折四模式 Spearman，最佳 `moddrop_seed1114`，最弱 `cfcompat_seed1111`。
9. 哪个 missing mode 信号最强？四模式汇总中 `L` 最高；完整 missing-mode 表保留在 per-mode 指标中。
10. 两个 checkpoint replication 是否一致？fold0=0.2476，fold1=0.2333。
11. 风险输出能否校准到统一 MAE 单位？R2 直接输出 expected absolute error，量化覆盖率、pinball 和 q90−q50 宽度均以同一绝对误差单位保存。
12. 是否值得进入 Stage23D-B？尚不能；需先完成冻结授权的 A5，再做人工决策。
13. 若失败，原因是无信号还是样本不足？每折 OOF 外层仍有约千级样本；应优先依据 R2 对 R1/N0/N2/N5 的差异判断信号增量，而不是把失败归因于样本数。
14. 是否有理由继续动态专家选择主线？只读跨专家诊断 top-1=0.2516、top-2=0.4341；这不是 Arbiter 性能声明。
15. checkpoint 指纹风险如何？scalar summary identity AUROC=0.9999（随机为 0.5）；raw PCA 坐标未用于该结论。
16. 是否访问 Official Valid/Test 或训练新模型？Official Valid=0、Test=0；未训练 Arbiter、Student 或 Expert，Risk probe 是本阶段授权的小模型。
17. 下一步是什么？等待 GPU 3 真正空闲后，仅运行冻结的两位 Expert × 两折 A5 pilot；绝不抢占 GPU 0/1/2，也不自动进入 Stage23D-B。

## 冻结边界

- Stage23A-v2a = SIGNAL_AUDIT_FAIL，未修改。
- Stage23A-v2b = LOCAL_COMPETENCE_WEAK，未修改。
- Stage23C 主工作树、进程和 GPU0/1/2 未被本阶段修改或占用。
- 五个 frozen Expert、两个 checkpoint fold，共 10 个独立 self-risk audits。
- 每个 audit 的 outer labels 仅在冻结预测之后打开一次。

## Expert 汇总

| expert_id | Spearman | bad20_AUROC | confident_wrong_AUROC |
| --- | --- | --- | --- |
| moddrop_seed1114 | 0.2648 | 0.6322 | 0.8936 |
| moddrop_seed1111 | 0.2515 | 0.6101 | 0.8753 |
| cfcompat_seed1114 | 0.2513 | 0.6342 | 0.8932 |
| uniform_kd_seed1111 | 0.2201 | 0.6259 | 0.8863 |
| cfcompat_seed1111 | 0.2146 | 0.6174 | 0.8800 |

## Mode 汇总

| mode | Spearman | bad20_AUROC | confident_wrong_AUROC |
| --- | --- | --- | --- |
| L | 0.2553 | 0.6311 | 0.8841 |
| LAV | 0.2394 | 0.6232 | 0.8881 |
| LA | 0.2392 | 0.6188 | 0.8883 |
| LV | 0.2280 | 0.6227 | 0.8824 |

完整表、负对照、risk–coverage、quantile calibration、敏感性和 SHA 清单位于同目录。


## A5 readiness seal

- Phase-1 gate: `WEAK`; mean Spearman=0.2404,
  bad20 AUROC=0.6240, confident-wrong AUROC=
  0.8857.
- Authorized pilot only: `moddrop_seed1111` and `cfcompat_seed1111`, both
  checkpoint folds, four preregistered stochastic passes.
- A5 extraction/probe code passed 10 CPU tests and was committed before use.
- Eligible GPU 3 was occupied at the readiness snapshot; A5 jobs started: 0.
- Official Valid access: 0; Test access: 0; Stage23C modifications: 0.
