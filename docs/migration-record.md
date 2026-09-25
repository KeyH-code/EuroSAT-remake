# 迁移记录（EuroSAT → EuroSAT-remake）

本文件是新仓库自带的**历史档案**：记录迁移了什么、旧仓库当时长什么样、旧配置与新配置差在哪、历史指标是多少。
它取代原来的「运行时依赖旧快照」做法——**代码不再读取任何历史产物，历史实验一律重新训练**。

- 记录日期：2026-09-24（迁移当日核对）+ 2026-09-25（补齐本记录时的复核）
- 源教学仓库：`E:\deeplearning`，迁移时 HEAD `807fec304449f250ec19036048cc6217d81432f9`
- 源缓存路径：`E:\deeplearning\.cache\datasets\eurosat`
- 本仓库快照位置：`artifacts/legacy_eurosat/`（**原样冻结，不删不改，也不再被任何代码引用**）

术语约定：本文中「**复核**」= 本次由本记录作者实际读取文件得出；「**转载**」= 迁移当时写下的核对结论，本次未重新逐文件验证。

---

## 1. 迁移内容与完整性

| 项目 | 目标位置 | 规模 | 状态 |
| --- | --- | --- | --- |
| 完整运行记录快照 | `artifacts/legacy_eurosat/` | 28,094 文件 / 1,239,999,442 B | 字节数与迁移记录一致（复核） |
| 完整 RGB 图像 | `data/EuroSAT_RGB/2750/` | 27,000 张 64×64 | 文件数一致（复核） |
| 原样划分清单 | `data/manifests/my_split.csv` | 2,873,312 B | SHA-256 与快照内副本一致（复核） |
| ResNet18 预训练权重 | `artifacts/resnet18-f37072fd.pth` | 46,830,571 B | SHA-256 `f37072fd…`（转载） |

**清单哈希**（三方一致：源缓存 / 快照 / `data/manifests/`）：

```
my_split.csv  SHA-256 = 66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf
```

划分固定为 train 18,900 / val 4,050 / test 4,050；CSV 内记录的旧绝对路径 `E:\deeplearning\…` **按字节保留未重写**，运行时由 `eurosat_paths.map_manifest_images` 在内存中按「类别 + 文件名」映射到 `data/EuroSAT_RGB/2750/`。

**迁移当时的逐文件核对结论（转载）**：源缓存与目标快照 28,094 对文件、双方各 1,239,999,442 字节，缺失 0、SHA-256 不同 0；快照内 RGB 目录 27,000 对文件，缺失 0、SHA-256 不同 0。未修改源仓库或其缓存。

---

## 2. 快照内容清单

### 2.1 顶层

| 路径 | 内容 |
| --- | --- |
| `checkpoints/` | 教学期 CNN 的 `best.pt`(ep5)、`last.pt`(ep6)、`resume_source_epoch3.pt` |
| `EuroSAT_RGB/2750/` | 快照内的图像副本（与 `data/` 同一批） |
| `evaluations/` | 教学期 CNN 的 `summary.json`、`per_class_metrics.csv`、`error_samples.csv`、`predictions_val_32.csv` |
| `experiments/` | 18 个 CNN 正式实验目录 + `TRAINING-JOURNAL.md`（274 行训练轨迹） |
| `predict_inputs/val_32_epoch5/` | 10 类共 32 张历史预测输入 |
| `resnet18_finetune/` | 12 个 ResNet18 微调运行目录 + 并行日志 |
| `runs/run_20260921T203800_936952+0800/` | 教学期单次会话记录 |
| `my_split.csv` | 清单副本 |

### 2.2 逐实验目录规模（复核）

| 实验目录 | 文件数 | MiB |
| --- | ---: | ---: |
| `experiments/formal_training` | 23 | 2.6 |
| `experiments/learning_example_work` | 34 | 6.9 |
| `experiments/warmup_lr_0.002` | 26 | 2.8 |
| `experiments/lr_warmup_sweep_15ep/warmup5_to_002` | 29 | 2.8 |
| `experiments/lr_warmup_sweep_15ep/warmup1_to_002` | 29 | 2.8 |
| `experiments/lr_warmup_sweep_15ep/warmup1_to_003` | 30 | 2.8 |
| `experiments/lr_004_exploration_15ep/warmup1_to_004` | 29 | 2.8 |
| `experiments/lr_004_exploration_15ep/stepup_epoch7_to_004` | 29 | 2.8 |
| `experiments/batch_size_64_15ep` | 32 | 3.2 |
| `experiments/optimizer_regularization_15ep/weight_decay_0` | 30 | 2.8 |
| `experiments/optimizer_regularization_15ep/weight_decay_005` | 30 | 2.8 |
| `experiments/optimizer_regularization_15ep/momentum_sgd_09` | 30 | 2.1 |
| `experiments/optimizer_regularization_15ep/adamw_beta1_095` | 30 | 2.8 |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0001` | 30 | 2.1 |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0002` | 30 | 2.1 |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0006` | 30 | 2.1 |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0008` | 31 | 2.1 |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0006_extend20` | 37 | 2.2 |
| `experiments/adamw_003_continue_30ep` | 47 | 3.1 |
| `resnet18_finetune/*`（12 个运行目录） | 33–48 | 各 86 |

每个正式实验目录内通常含：`config.json`、`metrics.jsonl`、`per_class_metrics.jsonl`、`learning_rate_history.jsonl`、`sessions.jsonl`、`epoch_metrics.csv`、`learning_curves.png`、`learning_rate_curve.png`、`per_class_recall.png`、`confusion_matrices/`、`checkpoints/{best,last}.pt`、`final_evaluation/`。

### 2.3 checkpoint 形态（复核，66 个 `.pt`）

| 来源 | 数量 | 形态 |
| --- | ---: | --- |
| `experiments/**`（CNN 正式实验） | 44 | 含 `config_path` + `config_sha256` + 内嵌 `config` |
| `checkpoints/`、`learning_example_work/`（教学期） | 6 | 无 `config_path`/`config_sha256`，仅含 `args`/`run_dir`/`run_id` |
| `resnet18_finetune/**` | 24 | 含内嵌 `config`，无 `config_sha256` |

> 历史 checkpoint **不参与**新仓库训练、评价或预测。它们只是证据，不再被任何入口读取。

---

## 3. 历史结果

### 3.1 小 CNN（`EurosatCNN`，98,970 参数，FP32、无增强）

best val accuracy（val 固定 4,050 张；`final_evaluation/summary.json`，复核）：

| 实验 | 优化器 / lr | 轮数 | Best epoch | Val acc | Val loss | 错误 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `formal_training` | AdamW 0.001 / wd 0.01 | 10 | 10 | 0.899506 | 0.285885 | 407 |
| `warmup_lr_0.002` | AdamW 0.002 / warmup 1ep | 10 | 8 | 0.915062 | 0.256198 | 344 |
| `lr_warmup_sweep_15ep/warmup5_to_002` | AdamW 0.002 / warmup 5ep | 15 | 15 | 0.923704 | 0.233935 | 309 |
| `lr_warmup_sweep_15ep/warmup1_to_002` | AdamW 0.002 / warmup 1ep | 15 | 15 | 0.926420 | 0.227310 | 298 |
| **`lr_warmup_sweep_15ep/warmup1_to_003`（原 C 参照）** | AdamW 0.003 / warmup 1ep | 15 | 15 | **0.936543** | 0.199390 | 257 |
| `lr_004_exploration_15ep/warmup1_to_004` | AdamW 0.004 / warmup 1ep | 15 | 12 | 0.927407 | 0.213182 | 294 |
| `lr_004_exploration_15ep/stepup_epoch7_to_004` | AdamW 0.004 阶跃 | 15 | 15 | 0.909383 | 0.265081 | 367 |
| `batch_size_64_15ep` | AdamW 0.003 / batch 64 | 15 | 11 | 0.937778 | 0.185248 | 252 |
| `optimizer_regularization_15ep/weight_decay_0` | AdamW 0.003 / wd 0 | 15 | 11 | 0.914568 | 0.240756 | 346 |
| `optimizer_regularization_15ep/weight_decay_005` | AdamW 0.003 / wd 0.05 | 15 | 11 | 0.924691 | 0.210865 | 305 |
| `optimizer_regularization_15ep/momentum_sgd_09` | SGD 0.003 / m 0.9 | 15 | 11 | 0.886914 | 0.362414 | 458 |
| `optimizer_regularization_15ep/adamw_beta1_095` | AdamW β1=0.95 | 15 | 15 | 0.925926 | 0.210340 | 300 |
| `sgd_learning_rate_15ep/sgd_lr_0001` | SGD 0.001 | 15 | 13 | 0.880741 | 0.382869 | 483 |
| `sgd_learning_rate_15ep/sgd_lr_0002` | SGD 0.002 | 15 | 15 | 0.881481 | 0.352214 | 480 |
| `sgd_learning_rate_15ep/sgd_lr_0006` | SGD 0.006 | 15 | 15 | 0.899753 | 0.352117 | 406 |
| `sgd_learning_rate_15ep/sgd_lr_0006_extend20` | SGD 0.006，续训 20 轮 | 20 | 15 | 0.899753 | 0.352117 | 406 |
| `sgd_learning_rate_15ep/sgd_lr_0008` | SGD 0.008 | 15 | 10 | 0.861975 | 0.416334 | 559 |
| **`adamw_003_continue_30ep`（历史最佳）** | AdamW 0.003，续训 30 轮 | 30 | 28 | **0.940247** | 0.171498 | 242 |

另有一条教学期 baseline：`checkpoints/best.pt`（epoch 5）在完整 val 上的 loss `0.3172211845697444`、accuracy `0.8911111111111111`、错误 441 张（`evaluations/summary.json`，复核）。

**历史最好成绩 0.9402（best@28）距目标 98% 仍有明显差距**；所有实验 `test_split_used` 均为 `false`。

### 3.2 预训练 ResNet18（冻结 backbone 与 BN，只训分类头）

`final_evaluation/summary.json`（复核，Best epoch 列为对应 checkpoint 轮次）：

| 运行目录 | 配置 | 优化器 / lr | 轮数 | Best epoch | Val acc | Val loss | 错误 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18_head_imagenet_5ep_01` | config01–05 | AdamW 0.001 / wd 0.01 | 5→35 | 29 | 0.953827 | 0.150087 | 187 |
| `resnet18_head_lr0p0015_20ep_01` | config07 | AdamW 0.0015 / wd 0.01 | 20 | 19 | 0.951605 | 0.153141 | 196 |
| `resnet18_head_lr0p002_20ep_01` | config06 | AdamW 0.002 / wd 0.01 | 20 | 19 | 0.952099 | 0.151327 | 194 |
| `resnet18_head_lr0p003_20ep_01` | config08 | AdamW 0.003 / wd 0.01 | 20 | 15 | 0.952840 | 0.150419 | 191 |
| `resnet18_head_lr0p003_wd0_20ep_01` | config12 | AdamW 0.003 / wd 0 | 20 | 15 | 0.952840 | 0.150933 | 191 |
| `resnet18_head_lr0p004_20ep_01` | config09 | AdamW 0.004 / wd 0.01 | 20 | 15 | 0.951852 | 0.153397 | 195 |
| `resnet18_head_lr0p005_20ep_01` | config10 | AdamW 0.005 / wd 0.01 | 20 | 11 | 0.949136 | 0.160802 | 206 |
| `resnet18_head_lr0p006_20ep_01` | config11 | AdamW 0.006 / wd 0.01 | 20 | 11 | 0.951852 | 0.155322 | 195 |
| `resnet18_head_lr0p01_wd0p01_20ep_01` | config13 | AdamW 0.01 / wd 0.01 | 20 | 15 | 0.948395 | 0.177915 | 209 |
| `resnet18_head_lr0p01_wd0_20ep_01` | config14 | AdamW 0.01 / wd 0 | 20 | 12 | 0.948148 | 0.171800 | 210 |
| `resnet18_head_sgd_lr0p01_m0p9_20ep_01` | config15 | SGD 0.01 / m 0.9 | 20 | 17 | 0.946914 | 0.162531 | 215 |
| `resnet18_head_sgd_lr0p05_m0p9_20ep_01` | config16 | SGD 0.05 / m 0.9 | 20 | 11 | 0.950864 | 0.155821 | 199 |

其中 config01 单独 5 轮时的 best 为 0.9373（3796/4050，epoch 5）；续训到 15 轮得 0.950617（epoch 15）、25 轮未刷新（epoch 24 与 15 精确相同 3850/4050）、35 轮最终 0.953827。全程用 val 选模，test 未使用；40 个 BN 运行统计量缓冲区与 ImageNet 预训练权重一致。

**ResNet18 历史最好成绩 0.9538，同样未达 98%。** 详细过程见 `resnet18/reports/01`–`08`。

---

## 4. 旧配置与迁移差异（关键差异清单）

旧配置一律保留旧仓库的绝对路径，**不能直接在新仓库运行**。差异只有两类：

### 4.1 CNN 配置（18 份 `experiments/**/config.json`）

| 字段 | 旧配置值 | 新仓库取值 |
| --- | --- | --- |
| `data.manifest_path` | `E:\deeplearning\.cache\datasets\eurosat\my_split.csv` | `data/manifests/my_split.csv`（或同义绝对路径） |
| 其余字段（种子、batch、优化器、scheduler、precision、augmentation、evaluation、outputs…） | — | **原样沿用，无需改动** |

`cnn/configs/baseline.json` 即由 `experiments/formal_training/config.json` 改写 `manifest_path` 一项得到，其余字段逐项一致（复核）。**新 CNN 实验从这里复制，不要再复制快照里的旧配置。**

两份带 `lineage` 的续训配置另有指向旧目录的字段，不使用：
`adamw_003_continue_30ep.config.lineage.source_directory`、`sgd_learning_rate_15ep/sgd_lr_0006_extend20.config.lineage.source_run`，两者都是 `E:\deeplearning\.cache\datasets\eurosat\experiments\…`。

### 4.2 ResNet18 配置（`resnet18/configs/config01`–`16`）

| 项 | 旧值 | 新仓库 |
| --- | --- | --- |
| 配置内绝对路径 | **无**（配置里只有 loader/transform/optimizer） | 无需改动 |
| 旧运行目录 | `E:\deeplearning\.cache\datasets\eurosat\resnet18_finetune\…` | `runs/resnet18/<experiment_id>/` |
| `optimizer.name` | config01 等早期配置**缺该字段** | `resnet18/src/optimizer.py` 缺省按 `adamw` 处理 |
| `epoch` | 配置 02–05 是 15/25/30/35 的累进目标值 | 只用于「本实验目标总轮数」；续训只能增大 |

按 `experiment_id` 建立新运行目录；**`config01`–`16` 的 `experiment_id` 已被历史运行占用**，若在新仓库复用同名配置，请先改 `experiment_id`，否则 `resnet18/src/train.py` 会因目录已存在而拒绝启动。

### 4.3 旧配置全文的查阅位置

旧配置**不复制进 Git**（快照本身已冻结在同一台机器上，复制会产生两份可能各自漂移的副本；核对以下哈希即可确认未被改动）：

- CNN：`artifacts/legacy_eurosat/experiments/<实验名>/config.json`
- ResNet18：`artifacts/legacy_eurosat/resnet18_finetune/<运行名>/config.json`

目录索引与逐文件 SHA-256 见 `docs/legacy-configs/README.md`。

---

## 5. 迁移当日的独立运行核验

以下结论来自迁移当天的运行记录（转载，并注明本次复核结果）：

| 核验项 | 结论 | 本次复核 |
| --- | --- | --- |
| 单元测试 | 5 项通过 | **已失效**：其中 1 项依赖快照，本次解耦时删除；现有测试见第 6 节 |
| 真实图片单批次参数更新 | CNN、ResNet18 各 2 张图完成 1 次 optimizer step，记录在 `runs/migration_smoke/` | 记录文件仍在 |
| 32 张历史预测输入推理 | `runs/cnn_predictions/`、`runs/resnet18/predictions/` | CSV 仍在（含 `artifacts\legacy_eurosat\predict_inputs\…` 路径，属当时事实） |
| 历史 CNN best(ep5) 完整 val | loss `0.3172211845697444`、accuracy `0.8911111111111111`、错误 441，与快照 `evaluations/summary.json` 完全一致 | 一致（复核） |
| 历史 ResNet18 best(ep11) 完整 val | loss `0.15582130051321452`、accuracy `0.9508641975308642`、错误 199，与快照 `final_evaluation/summary.json` 完全一致 | 一致（复核） |
| 三入口 `--help` 可启动 | 通过 | 解耦后重新验证，见第 6 节 |

这些都是**当时**的核验。它们证明迁移搬运无误，**不构成新仓库对旧产物的持续依赖**。

---

## 6. 解耦决定与验收边界

**决定（2026-09-25）**：新仓库不再复用任何历史产物。历史实验如需重跑，用第 3 节的配置参数在 `runs/` 下**重新训练**。

已移除的耦合：

| 位置 | 原行为 | 现在 |
| --- | --- | --- |
| `eurosat_paths.py` | 导出 `LEGACY_ROOT`、`archived_config_path()`（把旧路径重定位到快照） | 已删除；模块只管数据、清单、`runs/` |
| `cnn/src/experiment_runner.py` | 放行旧清单绝对路径；从快照取旧配置校验续训身份 | 已删除；`manifest_path` 必须指向本仓库清单，续训只认 checkpoint 内嵌配置 |
| `cnn/evaluate.py` | `--checkpoint` 默认快照 `checkpoints/best.pt` | `--checkpoint` 必填 |
| `cnn/predict.py` | `--checkpoint` 默认快照 `checkpoints/best.pt` | `--checkpoint` 必填 |
| `resnet18/evaluate.py` | 允许 checkpoint 位于快照 `resnet18_finetune/` | 只允许 `runs/resnet18/` |
| `resnet18/src/02_compare_lr_curves.py` | 从快照读运行目录 | 从 `runs/resnet18/` 读 |
| `cnn/tests/test_legacy_resume_relocation.py` | 依赖快照配置 | 已删除 |
| `cnn/tests/test_independent_paths.py` | 断言清单与快照副本逐字节相同 | 改为校验哈希与划分数量，不再触碰快照 |

仍然有效、且**不受解耦影响**的不变量：清单 SHA-256 固定、train/val/test = 18900/4050/4050、test 不参与调参、新实验只写 `runs/`、快照与 `data/` 原样保留。

**验收边界**：`specs/migration-plan.md` 第 5 条（分阶段 Git 提交）随本记录一并闭环；M2 的「自定义 CNN ≥98%」条件**没有通过，也没有调整**——历史最佳仅 0.9402。

---

## 7. 解耦后的首次短训练（新仓库自有基线）

2026-09-25 在解耦完成后实跑，目的是验证「重训」这条路本身可用，**不是**复现历史指标，也**不构成** M2 证据。

| 实验 | 配置 | 轮数 | 耗时 | Best epoch | Val acc | Val loss | 错误 | test 使用 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 小 CNN `baseline_ep2` | `runs/cnn/baseline_ep2/config.json`（= `formal_training` 参数，仅 2 轮） | 2 | 94.4 s | 2 | 0.838519 | 0.488516 | 654 | `false` |
| ResNet18 `resnet18_head_probe_1ep_01` | `resnet18/configs/config17_probe_1ep.json`（AdamW 0.001 / wd 0.01，1 轮） | 1 | — | 1 | 0.905926 | 0.387340 | 381 | `false` |

产物位置：`runs/cnn/baseline_ep2/`（checkpoints、逐轮指标、混淆矩阵、图表、`final_evaluation/` 齐全）、`runs/resnet18/resnet18_head_probe_1ep_01/`、`runs/resnet18/evaluations/resnet18_head_probe_1ep_01_20260925T123302_024837/`。这些是新仓库**第一批属于自己的产物**，与第 3 节的历史档案没有任何数据血缘。

同一轮验证的其余结论（全部实测）：

- 6 个入口 `--help` 均可启动；`cnn/evaluate.py`、`cnn/predict.py` 的 `--checkpoint` 为必填。
- `cnn/predict.py` 用新 checkpoint 对 `data/EuroSAT_RGB/2750/AnnualCrop/` 的 3,000 张图推理成功（输出 3,000 行，`runs/cnn_predictions/20260925T123101_686033.csv`）。
- `resnet18/evaluate.py` 只接受 `runs/resnet18/` 下的 checkpoint；对本次 1 轮 checkpoint 复评得到与训练记录一致的 `0.905926`。
- 续训契约 4 项：新 checkpoint 延长轮数可恢复；未延长轮数被拒；历史快照 checkpoint 被拒（`checkpoint 缺少 config，不是本仓库训练的产物`）；旧仓库清单绝对路径被拒（`config manifest_path 不是当前正式 my_split.csv`）。
- 测试：`conda run -n dl-reboot python -m pytest -q cnn/tests resnet18/tests` → **8 passed**，其中一项为「源码不得引用历史快照」的守卫。

**新增的未跟踪产物说明**：`runs/cnn_predictions/20260925T123101_686033.csv` 与 `runs/resnet18/predictions/20260924T174719_508966.csv`、`20260924T175604_675683.csv` 三份 CSV 中的 `filepath` 仍是 `artifacts\legacy_eurosat\predict_inputs\…`——迁移当日那次推理确实读的是历史输入目录。这是当时的事实记录，不代表现在的代码仍会这样做。
