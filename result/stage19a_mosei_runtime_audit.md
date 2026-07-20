# Stage 19A — MOSEI Training Runtime Forensic Audit v1

最终状态：`STAGE19A_RUNTIME_AUDIT_COMPLETE`

## 执行摘要

“单 seed 约 30 小时”的前提不成立。实际 Stage 10 历史记录显示，**五个 seed、四个 worker stage 串行跑完共 27.913 h**；加上 worker 结束后的 locked coordinator 阶段约 0.326 h，总 wall time 约 28.24 h。单 seed 完整四阶段为 **4.495–6.434 h，均值 5.583 h**。

若一个新候选仍使用相同 dataset/split、clean 初始化、frozen teacher、ModDrop evaluator、compatibility 定义和 missing schedule，则最小必须重跑的是候选 Student（当前等价于 CFCompat stage），历史耗时 **1.408–3.431 h/seed，均值 2.473 h/seed**。clean、ModDrop 和 compatibility cache 可在严格 cache-key 校验后复用；不能因为目录存在就盲目复用。

train-only 动态 profiling 在物理 GPU 2 上完成。最终权威 run 为 20 warmup + 100 timed microbatches，未构造 Valid/Test loader、未保存 checkpoint。实测 3.245 microbatch/s、51.92 sample/s，peak allocated 3.477 GiB、reserved 3.647 GiB。每 microbatch 是 2 次 Student forward（LAV + 一个随机 missing view）、2 次可训练 Student BERT、1 次 frozen Teacher/BERT 在线 forward、0 次 evaluator forward、1 次 backward。瓶颈不是 DataLoader：DataLoader wait 仅 1.70 ms，而 backward 135.19 ms、两个 Student forward 合计 89.72 ms、Teacher forward 40.21 ms、full loss 32.01 ms，其中 Hinge 30.27 ms。

最优先的工作不是直接把 batch 从 16 放大，而是：

1. 按强 cache key 复用 clean/ModDrop/compatibility artifacts，只训练 proposed Student；
2. 将 frozen Teacher 的 train LAV `output_logit` 离线 FP32 缓存；
3. 严格等价地向量化 Hinge/去除逐样本 Python loop，并做数值等价门禁；
4. 取消每 microbatch 的 `float(tensor)` 同步，epoch 末统一汇总；
5. 在同协议重训全部 baseline 的前提下评估 AMP 与 batch/accumulation 新协议。

## 1. 审计范围、安全与环境

### 1.1 仓库状态

| 项目 | 值 |
|---|---|
| 被审计 MOSEI 工作树 | `/code/DLF-mosei-generalization-v1` |
| 被审计 branch | `experiment/mosei-cfcompat-adpep-generalization-v1` |
| 被审计 HEAD | `dc8536f338c7e310a73447e4185c38be52800f48` |
| 被审计工作树 | clean，未修改 |
| 独立审计工作树 | `/code/DLF-mosei-runtime-audit-v1` |
| 审计 branch | `audit/mosei-runtime-forensics-v1` |
| audit 基线 HEAD | `dc8536f338c7e310a73447e4185c38be52800f48` |

Phase 0 未发现 DLF/MOSEI/Stage10/Stage18 worker。Stage 10 runtime state 为 `COMPLETED`；四张 GPU 均无 compute process，因此允许受限 profiling。没有停止、暂停、renice 或 attach 任何现有进程。

### 1.2 实际环境

| 类别 | 审计值 |
|---|---|
| Python | 3.9.13 |
| PyTorch | 1.13.0+cu117 |
| CUDA runtime / cuDNN | 11.7 / 8500 |
| transformers | 4.33.1 |
| GPU | 4 × NVIDIA GeForce RTX 2080 Ti |
| compute capability | 7.5 |
| 每卡显存 | 22528 MiB |
| CPU | Intel Core i9-9900K，8 cores / 16 threads |
| RAM | 约 60 GiB |
| 文件系统 | `/` 可用约 410 GB；`/data4t` 可用约 2.2 TB |
| 数据文件 | `/data4t/lyx/datasets/MOSEI/Processed/aligned_50.pkl` |
| 数据 SHA256 | `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf7991` |

运行时 backend：AMP/autocast/GradScaler 未使用；matmul TF32=false，cuDNN TF32=true，cuDNN benchmark=false，deterministic algorithms=false，anomaly detection=false；未发现 `CUDA_LAUNCH_BLOCKING`、`CUBLAS_WORKSPACE_CONFIG`、OMP/MKL 强制变量。代码中 `.cpu()`、`float(tensor)` 和显式 profiling synchronize 会同步；正式代码没有 profiling synchronize。

Locked Test access count = **0**。只静态查看 coordinator 的门禁和调用行，没有读取 Test prediction、Test metric JSON，也没有构造或迭代 Test DataLoader。

## 2. 实际 Pipeline 与 artifact 依赖

完整 DAG 见 `stage19a_pipeline_dag.md`。真实顺序是：

```text
clean checkpoint
  -> ModDrop validation-best evaluator
  -> train-only LAV/LA/LV/L compatibility cache
  -> CFCompat student (clean init + frozen clean teacher + compatibility)
  -> all five seeds complete
  -> locked coordinator
```

一个新方法不天然需要重训前三阶段。只有当它改变了相应依赖语义时才需要：

| Stage | 单 candidate/seed 是否必跑 | 可复用条件 | 主要失效条件 |
|---|---:|---|---|
| clean | 否 | 相同数据、split、seed、DLF 和训练协议 | teacher/student init 或 baseline protocol 改变 |
| ModDrop | 否 | 相同 clean SHA、missing schedule、ModDrop objective | evaluator 定义、checkpoint 或 schedule 改变 |
| compatibility | 否 | 相同 evaluator、样本身份/顺序、mode、公式、dtype | 任一 cache key 改变 |
| proposed/CFCompat Student | 是 | 不适用 | 候选方法本身 |

现有 cache 的强项：train-only、16326 行、唯一且连续 `sample_index`、evaluator SHA 和 CSV SHA 可校验。缺口：没有独立 `split_sha`、`sample_order_sha`、`missing_schedule_sha`。建议 key 至少包括 dataset/split/sample-order/teacher/evaluator SHA、seed、missing schedule SHA、feature layer、dtype、code commit 和公式版本。

分类：

- dataset-specific：clean/ModDrop/checkpoints、compatibility cache；
- seed-specific：所有当前 checkpoints/cache 和 missing RNG；
- teacher-checkpoint-specific：Teacher logits、使用该 teacher 的 KD artifact；
- evaluator-checkpoint-specific：compatibility cache；
- missing-schedule-specific：ModDrop trajectory 和 compatibility cache；
- method-specific：proposed/CFCompat Student checkpoint、epoch metrics。

## 3. 每 microbatch 计算图

实际 batch size 16，train samples 16326，故每 epoch `ceil(16326/16)=1021` microbatches。`update_epochs=10`，epoch 尾部会补 step，因此每 epoch `ceil(1021/10)=103` optimizer steps。

| 组件 | 每 microbatch 调用数 | requires_grad | 复杂度/shape | 可缓存或合并 |
|---|---:|---:|---|---|
| Student LAV DLF | 1 | 是 | text BERT + 三模态 DLF | 不能跨 step 缓存；Student 正在微调 |
| Student missing DLF | 1（每样本随机 LA/LV/L） | 是 | 同一 text 再做完整 BERT/DLF | 可研究单次 text encoding 复用，但会改变执行图/数值轨迹，需新协议 |
| Frozen Teacher LAV | 1 | 否 | 完整 DLF/BERT，最终只使用 `[B,1] output_logit` | 可完全离线缓存 train logits |
| Frozen evaluator | 0 在线 | 否 | 已在 compatibility stage 对 LAV/LA/LV/L 离线运行 | 已缓存 gate；需加强 sample-order binding |
| backward | 1 | — | 三项总 loss 的单次 backward | 不可缓存 |
| optimizer step | 每 10 批一次，epoch 残余一次 | — | clip → Adam step → zero_grad | 当前语义不可擅改 |

动态 hooks 在 100 个 timed batches 中观测到 Student BERT 200 次、Teacher BERT 100 次，与源代码计数完全一致。同一文本每批串行进入 BERT 三次。Student BERT 参与梯度；Teacher `eval()`、全部 `requires_grad=False`、`inference_mode()`。

### 3.1 Loss

LAV full objective：

- 5 个 L1 task heads：final、shared、language-specific（权重 3）、vision-specific、audio-specific；
- 3 个 reconstruction MSE；
- 3 个 specific-consistency MSE；
- 3 个 specific/shared cosine orthogonality；
- 1 个 shared-feature Hinge similarity。

missing view 只计算相同的 5-head task objective；再加 compatibility-weighted SmoothL1 prediction KD。没有额外 triplet 项；这里代码所称 similarity 是 `HingeLoss`。

Hinge 输入先把三模态 shared feature 拼成 `N=3B` 行、feature dim 50。B=16 时 N=48，pair tensor 为 `2304×50`；B=32 时 N=96，pair 元素数为 4 倍；B=64 时 N=192，为 16 倍。实现还对 N 逐行 Python loop。其 pair construction 是 O(B²F)，内部正/负组合最坏也呈二次增长。因此增大 microbatch 会提高这部分占比，并改变 batch-dependent pair 集合，不是原协议的严格等价替换。

实测 Hinge mean 30.27 ms，而整个 full-loss construction mean 32.01 ms；这是显著瓶颈。可进行向量化，但 reduction 顺序可能产生浮点差异，需逐项 loss/gradient tolerance gate 后才可归入“近似严格等价”。

### 3.2 Gradient accumulation 的法证结论

- 每 microbatch 一次 `loss.backward()`；
- loss **没有**除以 accumulation steps=10，因此不是标准平均梯度累积，而是梯度求和；
- epoch 开始 `zero_grad()`，每 10 批 clip → step → zero_grad；
- epoch 末不足 10 批的残余梯度会 clip/step，不丢失；
- grad clipping 只在 optimizer step 前；
- 无 scaler/unscale；
- scheduler 每 epoch在 Valid J 后调用；
- early stopping 监控 validation J，`early_stop=10`。

任何“改成标准平均累积”、microbatch/accumulation 变化都会改变有效梯度和 Hinge pair 集合，必须重训同协议 baseline。

## 4. Train、Valid、checkpoint、logging 与 DataLoader

### 4.1 Evaluation

- worker training 只构造 train + Official Valid；不构造 Test；
- clean 每 epoch 评估一次 LAV；
- ModDrop/CFCompat 每 epoch对 LAV/LA/LV/L 串行做四次完整 Student forward；
- 使用 `torch.no_grad()`，没有 autocast；
- 只计算 final prediction L1/回归指标，不计算 full auxiliary objective；
- 每 batch 将 prediction/label 搬到 CPU，并把 criterion 转 Python float；
- CFCompat 结束后重载 best checkpoint，又对 Valid 四模式完整推理一次以生成 CSV；
- 没有每 epoch重复 Test；locked coordinator 只在全部 worker 完成和 unlock 后运行。

Valid 的 epoch 内精确耗时无法从旧日志拆出，因为没有 train/valid 子阶段时间戳。不能伪造比例。静态上它每 valid batch有四次完整 BERT/DLF forward，必然是非零开销。

### 4.2 Checkpoint 与 logging

- 只保存 validation-best `state_dict`，同一路径覆盖；不保存 last 或 epoch-N；
- clean checkpoint 约 449,892,193 bytes；ModDrop/CFCompat 约 449,909,341 bytes；
- 每当 validation score 改善时写一次，写次数可从 epoch CSV 的 best 更新轨迹重建，但旧日志没有 save start/end；
- 因此 checkpoint I/O **无法精确分解占总时间多少**；
- 不存在多 worker 写同一 seed/stage 目录，当前调度无写冲突；
- 正式 train loop 每批有 `float(loss.detach())`，CFCompat 另有 `float(kd_loss.detach())`，会触发 GPU 同步；不逐 batch 写磁盘、不使用 tqdm、不保存 train predictions。

### 4.3 DataLoader

| 参数 | 实际值 |
|---|---|
| batch_size | 16 |
| shuffle | train=true，valid=false |
| drop_last | false |
| num_workers | 1 |
| pin_memory | 未设置（false） |
| persistent_workers | 未设置（false） |
| prefetch_factor | 未显式设置（PyTorch worker 默认） |
| worker_init_fn | 无 |
| collate_fn | default |
| H2D non_blocking | false |

每个 split 的 `MMDataset` 都会重新反序列化整个 pickle；train/valid 两个实例因此加载两次。数组整体驻留 RAM、astype float32；`__getitem__` 每样本重新从 numpy slice 构造三个 `torch.Tensor`。aligned 输入固定为 50 帧，feature dims `[768,74,35]`。动态 profile 的 DataLoader wait mean 1.70 ms、H2D 0.263 ms，只占每批约 308 ms 的很小部分，故当前不是 CPU starvation。

## 5. 历史时间法证

旧日志只有 stage start/end 和 epoch-level metric rows，没有 train、Valid、save 子计时。下表的总时间可靠；子阶段必须标记“无法精确分解”。

| Seed | Stage | Start UTC | End UTC | Epochs | Best epoch | Train/Valid/Save breakdown | 总时(h) |
|---:|---|---|---|---:|---:|---|---:|
| 1111 | clean | 07-17 15:00:42 | 07-17 15:42:26 | 12 | 2 | 无法精确分解 | 0.696 |
| 1111 | moddrop | 07-17 15:42:27 | 07-17 17:28:35 | 19 | 9 | 无法精确分解 | 1.769 |
| 1111 | compatibility | 07-17 17:28:36 | 07-17 17:31:26 | 0 | 0 | 全部为 train-only cache stage | 0.047 |
| 1111 | cfcompat | 07-17 17:31:26 | 07-17 20:19:56 | 30 | 20 | 无法精确分解 | 2.808 |
| 1112 | clean | 07-17 20:19:57 | 07-17 21:08:39 | 17 | 7 | 无法精确分解 | 0.812 |
| 1112 | moddrop | 07-17 21:08:40 | 07-17 22:26:47 | 15 | 5 | 无法精确分解 | 1.302 |
| 1112 | compatibility | 07-17 22:26:48 | 07-17 22:29:44 | 0 | 0 | 全部为 train-only cache stage | 0.049 |
| 1112 | cfcompat | 07-17 22:29:44 | 07-18 00:49:41 | 25 | 15 | 无法精确分解 | 2.333 |
| 1113 | clean | 07-18 00:49:41 | 07-18 01:56:02 | 23 | 13 | 无法精确分解 | 1.106 |
| 1113 | moddrop | 07-18 01:56:02 | 07-18 03:46:49 | 22 | 12 | 无法精确分解 | 1.846 |
| 1113 | compatibility | 07-18 03:46:50 | 07-18 03:49:53 | 0 | 0 | 全部为 train-only cache stage | 0.051 |
| 1113 | cfcompat | 07-18 03:49:53 | 07-18 07:15:45 | 30 | 20 | 无法精确分解 | 3.431 |
| 1114 | clean | 07-18 07:15:46 | 07-18 08:33:10 | 22 | 12 | 无法精确分解 | 1.290 |
| 1114 | moddrop | 07-18 08:33:11 | 07-18 11:15:11 | 26 | 16 | 无法精确分解 | 2.700 |
| 1114 | compatibility | 07-18 11:15:12 | 07-18 11:18:30 | 0 | 0 | 全部为 train-only cache stage | 0.055 |
| 1114 | cfcompat | 07-18 11:18:30 | 07-18 12:42:58 | 12 | 2 | 无法精确分解 | 1.408 |
| 1115 | clean | 07-18 12:42:59 | 07-18 14:27:05 | 28 | 18 | 无法精确分解 | 1.735 |
| 1115 | moddrop | 07-18 14:27:06 | 07-18 16:27:50 | 22 | 12 | 无法精确分解 | 2.012 |
| 1115 | compatibility | 07-18 16:27:51 | 07-18 16:32:40 | 0 | 0 | 全部为 train-only cache stage | 0.080 |
| 1115 | cfcompat | 07-18 16:32:40 | 07-18 18:55:38 | 20 | 10 | 无法精确分解 | 2.383 |

Stage 汇总：

| Stage | 五 seed 总时(h) | 均值(h/seed) | worker-stage 占比 |
|---|---:|---:|---:|
| clean | 5.638 | 1.128 | 20.2% |
| moddrop | 9.630 | 1.926 | 34.5% |
| compatibility | 0.282 | 0.056 | 1.0% |
| cfcompat | 12.363 | 2.473 | 44.3% |
| 合计 | 27.913 | 5.583 | 100% |

每 seed 完整四阶段：1111=5.320h、1112=4.495h、1113=6.434h、1114=5.453h、1115=6.210h。复用 clean + ModDrop + compatibility 可避免历史 worker 时间的 **55.7%**，但节省量不是固定 wall time，仍受候选 early stopping epoch 影响。

worker 在 2026-07-18 18:55:38 UTC 结束，runtime 在 19:15:16 UTC 完成，coordinator 约 0.326h。该时间含 locked inference、PE5、ADPEP 和 aggregation；本审计没有读取其中的 Test 指标。

## 6. Train-only 动态 profiling

安全条件全部满足后，在物理 GPU 2（进程内部 GPU0）运行独立审计脚本。profiling 脚本有 `--max-train-batches <= 300` 硬上限、GPU 空闲/受保护进程检查，只能构造 train loader。为加入独立 Hinge 计时，脚本经过一次 instrumentation 校验后重跑；两次合计 240 个 microbatches，仍低于 300 上限。下表取最终 20 warmup + 100 timed run。

| 组件 | mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| DataLoader wait | 1.704 | 1.686 | 1.919 |
| H2D | 0.263 | 0.259 | 0.294 |
| Student LAV forward | 45.823 | 44.484 | 48.979 |
| full loss（含 Hinge） | 32.009 | 32.107 | 33.221 |
| Hinge（full loss 子集） | 30.273 | 30.384 | 31.360 |
| Student missing forward | 43.893 | 43.117 | 46.364 |
| missing loss | 0.270 | 0.268 | 0.307 |
| Teacher LAV forward | 40.207 | 36.894 | 60.221 |
| compatibility + KD | 0.470 | 0.391 | 0.438 |
| backward | 135.190 | 134.322 | 139.342 |
| optimizer step（按所有 batch 摊销） | 4.287 | 0 | 42.695 |

总 elapsed 30.819s/100 batches。组件计时均有 `torch.cuda.synchronize()`，会扰动流水并略高估正式异步 wall；它用于归因，不作为正式性能承诺。未运行 `torch.profiler`，因为组件级同步计时已覆盖要求，且避免额外 profiler 扰动。

主要瓶颈排序：

1. backward（约 43.9% component wall）；
2. 三次串行 DLF/BERT forward（两个 Student + Teacher，合计约 42%）；
3. full-LAV loss，尤其逐 N Python-loop 的 O(B²) Hinge（约 9.8%）。

DataLoader/H2D 合计不足 1%，checkpoint 与 Valid 不在该 train-only microbatch profile 中。

## 7. Frozen Teacher 与 compatibility cache

当前 KD 只使用 frozen Teacher LAV 的最终 `output_logit [B,1]`。对 train 16326 个样本完整缓存：

- FP32：`16326 × 1 × 4 = 65,304 bytes`，约 63.8 KiB（不含索引/容器元数据）；
- FP16：32,652 bytes，约 31.9 KiB。

FP32 tensor cache 在相同 checkpoint、eval/inference mode、数据预处理、样本顺序和软件栈下可保持目标语义；应保存 sample ID/index 与 SHA，避免 CSV 小数截断。FP16 会改变 KD target 数值，属于新数值协议。若未来需要 hidden features，则大小按 `N×feature_dim×dtype_bytes` 重新计算，不能沿用上述估算。

compatibility 已离线缓存，在线 evaluator forward=0；现有 CSV约 5.1 MB/seed，含每样本 evaluator LAV/LA/LV/L prediction、delta/rank/q/compat。它能验证唯一连续 index 和文件/evaluator SHA，但缺独立 sample-order SHA，故目前只能在原 dataset/split/order 未变且由 manifest 链验证时复用。

## 8. 优化分类

### A. 近似严格保持当前训练语义

| 优化 | 证据/修改点 | 预计收益 | 风险与 baseline |
|---|---|---|---|
| 复用 clean/ModDrop/compat artifacts | DAG 与五 seed stage 时间 | 新 candidate 只剩约 1.4–3.4h，而非 4.5–6.4h | cache key 不完整会误绑定；不需重跑 baseline，需先补强验证 |
| FP32 离线 Teacher logits | Teacher 每批 40.2ms且只用 `[B,1]` | train microbatch 理论上约 10–13% | 必须保持顺序/精度/checkpoint；做逐样本 bit/tolerance gate |
| 向量化 Hinge | Hinge 30.27ms、O(B²)、Python loop | 理论约 5–10% train wall | reduction 顺序浮点差异；需 loss+gradient equivalence gate |
| epoch 末汇总 loss | 每批两次 `float(tensor)` | 约 1–3%，减少同步 | 仅影响日志；无需重跑 baseline |
| 避免重复 pickle/tensor 构造，pin+non_blocking | Data wait仅1.70ms | 当前约 0–2% | RNG/order必须不变；无需重跑 baseline |
| 去掉训练结束后的重复 Valid prediction（开发协议） | CFCompat best 后又四模式 Valid | 只节省末次 Valid | 若 artifact 消费者需要 CSV则不能删；正式报告需保留 |

收益不可直接相加，必须实现后重新 profile。

### B. 改变数值轨迹；同协议重训 baseline 后可公平使用

| 方案 | 为什么改变协议 | 等价/公平门禁 | 粗略加速区间（未实测） |
|---|---|---|---|
| AMP FP16 | matmul/reduction/optimizer 数值变化 | FP32 单批 loss/grad 对照、溢出检查；五 seed baseline/候选同协议重训 | train 约 1.3–1.8× |
| 改 microbatch 16→32/64 | Hinge pair 集合、BN无但 reduction/梯度与显存变化 | 保持或重新定义 effective batch；baseline 全重训 | 可能 1.1–1.5×，O(B²) 会抵消 |
| 标准平均 gradient accumulation | 当前是求和，除以10改变有效步长 | baseline 全重训、学习率/梯度 norm gate | 主要是规范化，不保证提速 |
| 合并/复用 Student text encoding | 改执行图与 dropout/RNG/反传路径 | logits/loss/grad tolerance、同协议重训 | 可能 1.15–1.4× |
| TF32 | Turing 2080 Ti 对 TF32无 Ampere tensor-core收益 | 无建议价值 | 近零 |
| 改 Valid/checkpoint frequency | 改 early stop/selection trajectory | 固定新规则并全 baseline 重训 | 依 Valid 占比，旧日志无法精确估计 |

建议 AMP：**建议作为 Stage 19B 新协议实验，不允许直接与旧 FP32 结果混比**。建议增大 batch：**不建议直接改正式协议；只建议先 profile B=32 并重建 baseline**，B=64 的 Hinge pair tensor 是 B16 的16倍。

### C. 仅 fast screen

- 冻结 Student BERT；
- train subset；
- 大幅降低 max epoch/early-stop；
- 缩短序列；
- 去掉 auxiliary/Hinge/reconstruction loss；
- 只训练上层或使用低精度代理模型。

这些可把筛选压到约 0.5–1.5h/seed，但不能作为论文最终协议或与正式 baseline 直接比较。

## 9. 十五个必须回答的问题

1. **30 小时是什么？** 不是单模型或单 seed；是五 seed 四 stage 串行 worker（27.913h）加 coordinator（约0.326h）的整套 run。
2. **新候选理论最少重跑什么？** 在 artifact 依赖完全相同且强校验通过时，只重跑 proposed Student；当前对应 CFCompat 类 stage。
3. **每 train microbatch 次数？** Student=2；BERT=3（Student 2 + Teacher 1）；Teacher=1；evaluator=0；backward=1。
4. **同一文本是否重复 BERT？** 是，串行三次；动态 hook 100 批实测 200 Student +100 Teacher。
5. **Frozen Teacher 能否完全缓存？** 能，当前只需 train `[16326,1] output_logit`；FP32约63.8KiB，FP16约31.9KiB。FP32强绑定 cache 可保持语义，FP16改变数值协议。
6. **compatibility 是否缓存且顺序可靠？** 已缓存；唯一连续 index、sample ID、CSV/evaluator SHA存在，但没有独立 sample-order SHA，可靠性不完整。
7. **增大 microbatch 会否被 O(B²) 抵消？** 可能。Hinge pair 元素 B32/B64 相对 B16为4×/16×，且 pair 集合改变；必须先 profile。
8. **gradient accumulation 是否标准平均？** 否，loss 未除以10，是梯度求和。
9. **epoch 尾梯度是否未 step？** 否，代码有 residual step，不丢失。
10. **训练每 epoch 是否访问 Test/重复评估？** 不访问 Test。每 epoch一次 Valid；missing 方法对四模式串行评估；CFCompat 训练结束额外重复一次 best Valid prediction。
11. **checkpoint I/O 占比？** 旧日志无 save 子时间戳，无法精确分解。best 改善时写约450MB state_dict，不保存 last/epoch-N。
12. **低 GPU 利用主因？** 多次串行小批 forward + backward + 复杂小 kernel/Python-loop Hinge；不是 CPU starvation。Valid 四模式和每批同步也贡献；磁盘只在 best save 时突发。
13. **现实单次时间？** 复用 artifact 的 proposed Student 当前已是1.4–3.4h（均值2.47h）；安全优化预计1.8–2.4h均值；AMP新协议约1.2–1.9h；fast screen约0.5–1.5h。完整四 stage 当前4.5–6.4h，已经低于10–15h。
14. **同一22GB GPU并发几任务？** 峰值 reserved约3.65GiB但算力竞争先于显存；建议最大 **2 个**独立任务，并先做并发 profile。未验证前不建议4–5个。
15. **最高优先级五项？** artifact 强复用；Teacher FP32 logits cache；Hinge 等价向量化；移除逐批同步日志；AMP/B32 作为同协议重训实验。

## 10. 最终建议与时间预算

| 使用场景 | 建议协议 | 预计单 seed |
|---|---|---:|
| 当前完整四-stage复现 | 不改协议 | 4.5–6.4h |
| 新候选正式开发 | 复用前三 stage，只训 Student | 1.4–3.4h，均值2.47h |
| 安全优化后候选 | + Teacher FP32 cache、Hinge向量化、日志同步优化 | 约1.8–2.4h均值 |
| AMP/新 batch 协议 | 全 baseline/候选同协议重训 | 约1.2–1.9h（候选 Student） |
| fast screen | 冻结/子集/少 epoch，仅筛选 | 约0.5–1.5h |

这些是基于实际 stage 时间和单协议 microbatch profile 的工程区间，不是正式 benchmark；early stopping seed 方差很大。建议先实现不改目标的 Teacher cache 与日志同步优化，做单批逐样本等价 gate，再做 100-batch A/B profile。Hinge 向量化需 loss/gradient gate。AMP、batch 和 Student encoder 合并必须定义 Stage 19B 新协议并重训 baseline。

安全确认：

- Locked Test access count = 0；
- 未停止或干扰任何现有进程；
- 未修改原 MOSEI 工作树；
- 未升级或安装依赖；
- 未运行完整训练；
- profiling 仅 train、总 microbatch 240、无 checkpoint 写入；
- 未实现正式加速方案。
