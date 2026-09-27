# 15：layer4+fc 学习率探索 baseline

整理日期：2026-09-26；基线补录：2026-09-27

## 数据边界与比较口径

本 baseline 汇总迁移后解冻 `layer4+fc` 的学习率实验（报告 10–14）及本轮 batch-size 对照。统一使用同一份固定 EuroSAT split（train 18,900 / val 4,050，SHA-256 `66730fa4…e54bf`）；所有最佳值依据 val，不读取 test。报告 09 的冻结分类头结果保留作历史背景，但未和 layer4+fc 数字并列排名，因为可训练范围不同。

准确率以 4,050 张 val 样本计数；best epoch 按最高 val accuracy 选择（并列时沿用训练记录先出现的 epoch）。表中 `val loss @ best acc` 与 `minimum val loss` 分开列出，因为两者可能来自不同 epoch。

## 首次突破 98% 的 batch64 参考基线

后续 label smoothing、weight decay 与训练集增强实验以该组作共同参考，不重新训练基线。

| 项目 | 数值或路径 |
| --- | --- |
| 实验 | `resnet18_layer4_lr0p0001_warmup_cosine_bs64_20ep_01` |
| 配置 | `resnet18/configs/config32_layer4_lr0p0001_warmup_cosine_bs64_20ep.json` |
| 训练设定 | `layer4+fc`，AdamW，lr=1e-4，weight decay=0.01，batch64，1 epoch warmup + 20 epoch cosine |
| 首次超过98% | epoch 12：val acc 0.980741（3972/4050），val loss 0.087566 |
| Best checkpoint | epoch 12；val acc 0.980741（3972/4050），val loss 0.087566 |
| 最低 val loss | epoch 4：0.074848 |
| Final epoch | train acc 1.000000、train loss 0.000051；val acc 0.980494、val loss 0.093948 |
| 独立复评 | val acc 0.980741、val loss 0.087566、误分78张；test 未使用 |
| 运行目录 | `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_20ep_01/` |
| 独立评价目录 | `runs/resnet18/evaluations/resnet18_layer4_lr0p0001_warmup_cosine_bs64_20ep_01_20260926T222057_780604/` |

## 综合对照

| 实验 | lr / schedule | batch | epoch | Best epoch | Best val acc | 正确/4050 | Val loss @ best acc | Minimum val loss @ epoch | Final val acc / loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 fixed (invalid) | 1e-3 fixed（用户判定不可用） | 128 | 20 | 12 | 0.975309 | 3950 | 0.084821 | 0.084821 @ 12 | 0.967160 / 0.119869 |
| 1e-4 fixed | 1e-4 fixed | 128 | 10 | 8 | 0.977778 | 3960 | 0.077667 | 0.077667 @ 8 | 0.975802 / 0.083827 |
| 1e-4 warmup+cosine 40 | 1e-4 warmup+cosine 40 | 128 | 40 | 20 | 0.978765 | 3964 | 0.100233 | 0.076120 @ 8 | 0.978272 / 0.108228 |
| 1e-5 fixed 20 | 1e-5 fixed 20 | 128 | 20 | 13 | 0.969630 | 3927 | 0.098407 | 0.091585 @ 11 | 0.969136 / 0.096117 |
| 1e-5 cosine 20 | 1e-5 cosine 20 | 128 | 20 | 12 | 0.969630 | 3927 | 0.090324 | 0.090324 @ 12 | 0.968889 / 0.091285 |
| 1e-5 cosine 40 | 1e-5 cosine 40 | 128 | 40 | 29 | 0.971852 | 3936 | 0.100289 | 0.091560 @ 12 | 0.970617 / 0.101677 |
| 5e-6 fixed 40 | 5e-6 fixed 40 | 128 | 40 | 33 | 0.972099 | 3937 | 0.098856 | 0.093184 @ 22 | 0.968642 / 0.103719 |
| 1e-6 fixed 40 | 1e-6 fixed 40 | 128 | 40 | 34 | 0.964444 | 3906 | 0.109247 | 0.105658 @ 40 | 0.963704 / 0.105658 |
| **batch64 reference（首次突破98%）** | batch 64 | 64 | 20 | 12 | 0.980741 | 3972 | 0.087566 | 0.074848 @ 4 | 0.980494 / 0.093948 |
| batch 96 | batch 96 | 96 | 20 | 13 | 0.978765 | 3964 | 0.097356 | 0.080756 @ 5 | 0.978519 / 0.100489 |

## 缩放曲线

学习率主对照包含 train/val accuracy 和 loss；batch size 对照单独成图。accuracy 纵轴固定在 0.90–1.00，loss 固定在 0.00–0.10。区间外数据被裁切，原值仍保留在 CSV 与各 run 的 `metrics.jsonl`。因此 1e-6 初期低于 0.90 的 accuracy 和高于 0.10 的 loss 不会完整显示在缩放图中。

![Layer4+FC learning-rate comparison](../../runs/resnet18/analysis/learning_rate_exploration_zoom.png)

![Batch-size comparison](../../runs/resnet18/analysis/batch_size_comparison_zoom.png)

逐轮可机器读取表：`runs/resnet18/analysis/epoch_comparison.csv`；汇总表：`runs/resnet18/analysis/learning_rate_baseline.csv`。

## 结果对照

### 1e-4 与 warmup + cosine

固定 1e-4、batch 128、10 epoch 的 best 为 0.977778（3960/4050，epoch 8，val loss 0.077667）。新 1e-4 warmup+cosine、batch 128、40 epoch 的 best 为 0.978765（3964/4050，epoch 20，val loss 0.100233），比固定 10 epoch 组多 4 张正确；该 40 epoch 组最低 val loss 为 0.076120（epoch 8），epoch 40 val loss 为 0.108228。两个结果的训练日程和预算不同。

### 1e-5 系列

1e-5 固定 20 epoch 与 1e-5 cosine 20 epoch 的 best accuracy 都是 0.969630（3927/4050）；cosine20 在 best accuracy epoch 的 val loss 为 0.090324，固定20为 0.098407。延长到 cosine40 后 best 为 0.971852（3936/4050，epoch 29），比前两组多 9 张正确；其最低 val loss 0.091560 在 epoch 12，epoch 40 val loss 为 0.101677。

### 5e-6 与 1e-6 延长至 40 epoch

固定 5e-6 的 40 epoch best 为 0.972099（3937/4050，epoch 33）；相对报告 12 中 10 epoch 的 0.963951（3904/4050），增加 33 张正确。最低 val loss 0.093184 在 epoch 22，最后一轮 accuracy 为 0.968642。固定 1e-6 的 40 epoch best 为 0.964444（3906/4050，epoch 34）；相对 20 epoch 的 0.956543（3874/4050）增加 32 张正确。其最低 val loss 0.105658 出现在 epoch 40，最后一轮 accuracy 为 0.963704。

### batch size 64 与 96

两组使用相同的 1e-4 峰值、1 epoch warmup、20 epoch cosine、seed 和其他配置。batch64 的 best 为 0.980741（3972/4050，epoch 12），batch96 的 best 为 0.978765（3964/4050，epoch 13），两组 best 相差 8 张 val 样本。两组各自最低 val loss 分别在 epoch 4（0.074848）与 epoch 5（0.080756）；best accuracy 并未出现在最低 loss epoch。

每轮 optimizer update 数为 batch64: 296、batch96: 197；batch-size 对照同时改变每轮更新次数。batch128/warmup+cosine 的已跑组周期为 40 epoch，不是同 20 epoch 日程的直接 batch128 对照。

### 历史 1e-3 与冻结分类头结果

报告 10 的固定 lr=0.001、20 epoch 结果为 0.975309（3950/4050，epoch 12），用户判定不可用；曲线保留在主学习率图中并标记 `invalid`，不作为有效设定的推荐依据。报告 09 的冻结分类头 ResNet18 best 为 0.953827（3863/4050），训练范围与 layer4+fc 不同，未纳入上面主图。

## 本轮 baseline 记录

- 学习率已覆盖 1e-3（标记不可用）、固定 1e-4、warmup+cosine 1e-4、固定/余弦 1e-5、固定 5e-6 和 1e-6。
- 在报告 10–14 中，目前最高 val accuracy 为 batch64 + 1e-4 warmup/cosine 20 epoch：0.980741（3972/4050）；只按 val 记录，不构成 test 结果。
- 用户当前后续方向：batch size 之后尝试缩小 weight decay；梯度裁剪与 label smoothing 等待后续协作，本 baseline 不包含这些处理。

## 后续协作可采用的对照顺序

1. 如继续 weight decay，固定 batch64、1e-4 warmup+cosine、20 epoch 和其他设置，只变 weight decay；先比较 0.01 基线与一个较小值，之后再决定是否加第二个较小值。
2. 梯度裁剪是在反向传播后、optimizer update 前限制梯度范数；若采用 AdamW + AMP，通常先 `scaler.unscale_(optimizer)`，再按全局 norm 裁剪，然后 `scaler.step()`。可单独试 `max_norm=1.0`，并记录裁剪前范数/触发次数，避免与 weight decay 同时改。
3. Label smoothing 把 one-hot 目标的一部分概率分散到其余类别，通常通过 `CrossEntropyLoss(label_smoothing=...)` 设置；可从 0.05 单变量开始，对比 0.0 基线。它会改变训练目标和 train loss 的解释口径，val loss 仍按普通交叉熵计算。
4. 本报告未授权或启动这些新实验；配置值由用户确认后另开 run，不覆盖本轮结果。

## 产物

- 新实验报告：`resnet18/reports/14_warmup_batch_size_experiments.md`。
- 汇总与曲线：`runs/resnet18/analysis/learning_rate_baseline.csv`、`epoch_comparison.csv`、`learning_rate_exploration_zoom.png`、`batch_size_comparison_zoom.png`。
- 历史数据来源：报告 09–14 及对应 `runs/resnet18/` 下的训练记录；不读取 `artifacts/legacy_eurosat/`。
