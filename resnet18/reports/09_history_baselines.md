# 09 历史训练基线（跨模型汇总）

本文件汇总**迁移之前**在两个模型上做过的全部实验及其 val 指标，供选择新实验参数时对照。
原始产物是迁移时的运行记录快照 `artifacts/legacy_eurosat/`（只读、不被任何代码读取）；
本表是它的可读摘要。各组配置参数与详细过程分别见 `01`–`08` 号报告。

- 划分：train 18900 / val 4050 / test 4050，清单 SHA-256 `66730fa4…e54bf`
- 所有实验 `test_split_used` 均为 `false`；选模只看 val
- **两个模型的历史最好成绩（小 CNN 0.9402、ResNet18 0.9538）都未达到 98% 门槛**
- 表内 ResNet18 结果全部是**冻结 backbone、只训分类头**设定下取得的；2026-09-26 起
  该设定已被解冻 layer4 取代，因此这些数字不与新设定直接可比

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

> 本节全部结果均在**冻结 backbone、只训练分类头**的设定下取得。2026-09-26 起该设定已被取代（解冻 layer4），见第 8 节；本节数字保留为历史对照，不与新设定直接可比。

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

## 参数索引：各组用的是什么配置

| 模型 | 组 | 优化器 / lr / wd | 轮数 | batch |
| --- | --- | --- | ---: | ---: |
| 小 CNN | `formal_training` | AdamW 0.001 / 0.01 | 10 | 128 |
| 小 CNN | `warmup_lr_0.002` | AdamW 0.002，warmup 1ep | 10 | 128 |
| 小 CNN | `lr_warmup_sweep_15ep/warmup5_to_002` | AdamW 0.002，warmup 5ep | 15 | 128 |
| 小 CNN | `lr_warmup_sweep_15ep/warmup1_to_002` | AdamW 0.002，warmup 1ep | 15 | 128 |
| 小 CNN | `lr_warmup_sweep_15ep/warmup1_to_003`（原 C 参照） | AdamW 0.003，warmup 1ep | 15 | 128 |
| 小 CNN | `lr_004_exploration_15ep/warmup1_to_004` | AdamW 0.004，warmup 1ep | 15 | 128 |
| 小 CNN | `lr_004_exploration_15ep/stepup_epoch7_to_004` | AdamW 0.004，epoch7 阶跃 | 15 | 128 |
| 小 CNN | `batch_size_64_15ep` | AdamW 0.003，warmup 1ep | 15 | 64 |
| 小 CNN | `optimizer_regularization_15ep/weight_decay_0` | AdamW 0.003 / wd 0 | 15 | 128 |
| 小 CNN | `optimizer_regularization_15ep/weight_decay_005` | AdamW 0.003 / wd 0.05 | 15 | 128 |
| 小 CNN | `optimizer_regularization_15ep/momentum_sgd_09` | SGD 0.003 / m 0.9 / wd 0.01 | 15 | 128 |
| 小 CNN | `optimizer_regularization_15ep/adamw_beta1_095` | AdamW β1=0.95 | 15 | 128 |
| 小 CNN | `sgd_learning_rate_15ep/sgd_lr_0001` | SGD 0.001 / m 0.9 / wd 0.01 | 15 | 128 |
| 小 CNN | `sgd_learning_rate_15ep/sgd_lr_0002` | SGD 0.002 / m 0.9 / wd 0.01 | 15 | 128 |
| 小 CNN | `sgd_learning_rate_15ep/sgd_lr_0006` | SGD 0.006 / m 0.9 / wd 0.01 | 15 | 128 |
| 小 CNN | `sgd_learning_rate_15ep/sgd_lr_0006_extend20` | SGD 0.006，续训 | 20 | 128 |
| 小 CNN | `sgd_learning_rate_15ep/sgd_lr_0008` | SGD 0.008 / m 0.9 / wd 0.01 | 15 | 128 |
| 小 CNN | `adamw_003_continue_30ep`（历史最佳） | AdamW 0.003，续训 | 30 | 128 |
| ResNet18 | config01–05 | AdamW 0.001 / 0.01 | 5→35 | 128 |
| ResNet18 | config06 | AdamW 0.002 / 0.01 | 20 | 128 |
| ResNet18 | config07 | AdamW 0.0015 / 0.01 | 20 | 128 |
| ResNet18 | config08 | AdamW 0.003 / 0.01 | 20 | 128 |
| ResNet18 | config09 | AdamW 0.004 / 0.01 | 20 | 128 |
| ResNet18 | config10 | AdamW 0.005 / 0.01 | 20 | 128 |
| ResNet18 | config11 | AdamW 0.006 / 0.01 | 20 | 128 |
| ResNet18 | config12 | AdamW 0.003 / wd 0 | 20 | 128 |
| ResNet18 | config13 | AdamW 0.01 / 0.01 | 20 | 128 |
| ResNet18 | config14 | AdamW 0.01 / wd 0 | 20 | 128 |
| ResNet18 | config15 | SGD 0.01 / m 0.9 / wd 0 | 20 | 128 |
| ResNet18 | config16 | SGD 0.05 / m 0.9 / wd 0 | 20 | 128 |

统一设定（两组模型相同）：seed 20260920、无随机增强、`num_workers=0`。
小 CNN 为纯 FP32；ResNet18 为 FP16 autocast + GradScaler，输入 224×224、ImageNet mean/std，
冻结 BN 使用预训练运行统计量。

## 重跑这些实验

配置参数照抄本表即可，但**不要**直接把上表的 lr 用到解冻 layer4 的设定上：实测 0.001 与 3e-4
会在第一个 batch 之后退化，稳定上限约为 5e-5–1e-4。
