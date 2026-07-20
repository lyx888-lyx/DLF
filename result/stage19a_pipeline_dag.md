# Stage 19A — MOSEI Stage 10 实际调用链与流水线 DAG

审计对象：`/code/DLF-mosei-generalization-v1`，commit `dc8536f338c7e310a73447e4185c38be52800f48`。本文件仅描述静态调用关系；未构造 Test DataLoader。

## 调用链

| 层级 | 文件与入口 | 调用者 | 输入 | 输出 |
|---|---|---|---|---|
| 启动门禁 | `scripts/mosei/launch_stage10_async.py:102-201`, `main` | 人工启动命令 | 数据/配置/测试门禁、GPU 3、五 seeds | run config、GPU allocation、detached supervisor |
| supervisor | `scripts/mosei/stage10_supervisor.py:28-107`, `main` | launcher | run config | worker；所有 worker 完成后才启动 coordinator |
| seed/stage 调度 | `scripts/mosei/stage10_worker.py:38-131`, `main` | supervisor | seeds 1111–1115、stage 顺序、物理 GPU | 每阶段 sentinel、manifest；同一 worker 串行执行 |
| stage 路由 | `scripts/mosei/stage10_train.py:620-635`, `main` | worker | `--stage`、seed、train/valid config | 对应阶段 artifact |
| clean | `stage10_train.py:186-252`, `train_clean` | stage router | MOSEI train/valid、随机初始化 DLF | validation-best clean checkpoint、epoch CSV、manifest |
| ModDrop | `stage10_train.py:255-353`, `train_moddrop` | stage router | 同 seed clean checkpoint、train/valid | validation-best ModDrop checkpoint、epoch CSV、manifest |
| compatibility | `stage10_train.py:356-411`, `build_compatibility` | stage router | 同 seed ModDrop checkpoint、train-only loader、固定 missing RNG | train-only counterfactual compatibility CSV、summary、manifest |
| CFCompat | `stage10_train.py:447-617`, `train_cfcompat` | stage router | 同 seed clean checkpoint、ModDrop identity、compatibility CSV、train/valid | validation-best CFCompat checkpoint、valid predictions、epoch CSV、manifest |
| locked coordinator | `scripts/mosei/stage10_coordinator.py:254-345`, `main` | supervisor（所有 worker 成功后） | 全部 sentinels/manifests、unlock gate | locked evaluation/PE5/ADPEP/final aggregation；本审计只读源代码，未读取其数据或指标 |

## Artifact DAG

```text
MOSEI aligned_50.pkl + config + seed
                  |
                  v
        Clean DLF training (train + valid)
                  |
                  +-----------------------------+
                  |                             |
                  v                             v
       Clean validation-best ckpt     Frozen LAV teacher / Student init
                  |                             |
                  v                             |
       ModDrop training (train + valid)         |
                  |                             |
                  v                             |
       ModDrop validation-best ckpt             |
                  |                             |
                  v                             |
  Frozen evaluator, one train-only pass         |
        LAV/LA/LV/L per sample                  |
                  |                             |
                  v                             |
       compatibility CSV -----------------------+
                  |
                  v
      CFCompat student training (train + valid)
                  |
                  v
       validation-best CFCompat checkpoint
                  |
       [all five seeds/stages complete]
                  |
                  v
       locked coordinator gate (not accessed)
```

## 复用边界

| Stage | 输入 | 输出 | 新候选是否必须重跑 | 可跨候选复用 | 失效条件 |
|---|---|---|---|---|---|
| Clean | dataset、split、seed、config、code | clean checkpoint | 否，若候选仍使用完全相同的初始化 teacher/student | 是 | dataset/split、seed、DLF/训练协议或 checkpoint 改变 |
| ModDrop | clean checkpoint、missing schedule、train/valid | evaluator checkpoint | 否，若 compatibility 定义及 evaluator 固定 | 是 | clean SHA、missing schedule、ModDrop 目标/协议或 code 改变 |
| Compatibility | evaluator、train sample identity/order、missing schedule | train cache | 否，若 gate 定义不变 | 是 | evaluator SHA、样本集合/顺序、schedule、mode 或公式改变 |
| CFCompat | clean init/teacher、cache、候选方法、train/valid | candidate checkpoint | 是 | 否 | 本身就是候选方法训练 |
| Locked coordinator | 五 seed validation-best checkpoint | locked outputs | 开发期不运行 | 不作为开发选择信号 | 只能在正式门禁后运行 |

推荐 cache key：

```text
dataset_sha / split_sha / sample_order_sha / teacher_checkpoint_sha /
evaluator_checkpoint_sha / seed / missing_schedule_sha / feature_layer /
dtype / code_commit_sha / compatibility_formula_version
```

当前 compatibility CSV 已绑定 evaluator SHA、seed、train split、唯一 `sample_index` 和 CSV SHA；但没有独立的 split SHA、sample-order SHA、missing-schedule SHA，因此“文件自身完整”不等于“可在协议变化后安全复用”。
