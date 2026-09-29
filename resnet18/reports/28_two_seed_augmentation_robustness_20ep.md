# 28：完整数据增强与无随机增强配置的双 seed 对照

记录日期：2026-09-29

## 用户分析与决策

> 本次收益不大，但是可以保留实验结果，在这么高的准确率，且模型对预测置信度这么高的情况下，进行局部随机增强可能的确会造成这种收益与损失相当甚至不如损失的情况；我们现在换两个seed，对目前的完整数据增强配置，非数据增强的表现最好的配置跑相同次数的epoch，如果发现结果波动不大，而且train loss，accuracy也都在最后有下降，那就说明数据增强有效，但是在本实验基线效果已经很高的情况下，收益很难进一步看到效果。

用户确认“非数据增强的表现最好”指 config36：label smoothing 0.05，best val accuracy 0.983210。完整数据增强组指 config41：水平/垂直翻转、随机 90° 倍数旋转、随机裁剪与平移，best val accuracy 0.983210。用户指定两个新 seed：20260929、20260930。两种配置分别以这两个 seed 各运行 20 epoch，共四组新实验。

## 比较口径

两组均为 ImageNet 预训练 ResNet18、`layer4+fc`、AdamW、lr=1e-4、weight decay 0.01、batch64、1 epoch warmup 和 20 epoch cosine、固定原样 train/val split。验证集均使用确定性 Resize 与 Normalize，test 不用于调参。

config36 使用 label smoothing 0.05 且无随机训练数据增强；其 train loss 是平滑目标下的交叉熵。config41 无 label smoothing，使用完整随机数据增强；其 train loss 是普通交叉熵。因此记录各组随 epoch 的 train loss/accuracy 和同口径的 val loss/accuracy，并分别比较不同 seed 的结果；两组 train loss 的绝对值不直接作大小比较。

## 新实验

| 对照 | Seed | 配置 | 运行目录 |
| --- | ---: | --- | --- |
| config36，无随机增强、label smoothing 0.05 | 20260929 | `resnet18/configs/config44_layer4_lr0p0001_warmup_cosine_bs64_ls0p05_seed20260929_20ep.json` | `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_ls0p05_seed20260929_20ep_01/` |
| config41，完整随机增强 | 20260929 | `resnet18/configs/config45_layer4_lr0p0001_warmup_cosine_bs64_fullaug_seed20260929_20ep.json` | `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_fullaug_seed20260929_20ep_01/` |
| config36，无随机增强、label smoothing 0.05 | 20260930 | `resnet18/configs/config46_layer4_lr0p0001_warmup_cosine_bs64_ls0p05_seed20260930_20ep.json` | `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_ls0p05_seed20260930_20ep_01/` |
| config41，完整随机增强 | 20260930 | `resnet18/configs/config47_layer4_lr0p0001_warmup_cosine_bs64_fullaug_seed20260930_20ep.json` | `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_fullaug_seed20260930_20ep_01/` |

每组新建独立运行目录，不覆盖原有实验。

## 训练汇总

best checkpoint 按最高 val accuracy 选取，并列时保留最早 epoch。表中 seed 20260920 的两组为既有记录，新四组为本次训练结果。本次四组训练耗时合计 5403.51 秒（约 90.1 分钟）。

| 配置 | seed | best epoch | best val acc | 正确/4050 | best val loss | min val loss（epoch） | final train acc/loss | final val acc/loss | 秒/轮 | skips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| config36 / LS 0.05 | 20260920 | 7 | 0.983210 | 3982 | 0.117645 | 0.110865（8） | 1.000000 / 0.284326 | 0.981235 / 0.124333 | 47.51 | 0 |
| config41 / 完整增强 | 20260920 | 17 | 0.983210 | 3982 | 0.061789 | 0.061789（17） | 0.993968 / 0.015788 | 0.982469 / 0.063530 | 82.20 | 0 |
| config36 / LS 0.05 | 20260929 | 13 | 0.982222 | 3978 | 0.123532 | 0.112296（6） | 1.000000 / 0.284399 | 0.981728 / 0.122616 | 42.08 | 0 |
| config41 / 完整增强 | 20260929 | 16 | 0.982963 | 3981 | 0.062100 | 0.062100（16） | 0.994550 / 0.016858 | 0.982469 / 0.065963 | 84.77 | 0 |
| config36 / LS 0.05 | 20260930 | 10 | 0.983457 | 3983 | 0.117890 | 0.113661（7） | 1.000000 / 0.284337 | 0.980988 / 0.124149 | 50.87 | 0 |
| config41 / 完整增强 | 20260930 | 7 | 0.980988 | 3973 | 0.064959 | 0.060458（14） | 0.993704 / 0.017841 | 0.980494 / 0.063145 | 92.45 | 0 |

### 独立 val 复评（新四组）

| 配置 | seed | checkpoint epoch | val acc | val loss | 错误数 | 复评目录 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| config36 / LS 0.05 | 20260929 | 13 | 0.982222 | 0.123532 | 72 | `runs/resnet18/evaluations/resnet18_layer4_lr0p0001_warmup_cosine_bs64_ls0p05_seed20260929_20ep_01_20260929T134102_790797/` |
| config41 / 完整增强 | 20260929 | 16 | 0.982963 | 0.062100 | 69 | `runs/resnet18/evaluations/resnet18_layer4_lr0p0001_warmup_cosine_bs64_fullaug_seed20260929_20ep_01_20260929T134116_411215/` |
| config36 / LS 0.05 | 20260930 | 10 | 0.983457 | 0.117890 | 67 | `runs/resnet18/evaluations/resnet18_layer4_lr0p0001_warmup_cosine_bs64_ls0p05_seed20260930_20ep_01_20260929T134129_734138/` |
| config41 / 完整增强 | 20260930 | 7 | 0.980988 | 0.064959 | 77 | `runs/resnet18/evaluations/resnet18_layer4_lr0p0001_warmup_cosine_bs64_fullaug_seed20260930_20ep_01_20260929T134143_069944/` |

所有复评与训练时对应 best epoch 的 val accuracy/loss 一致，仅使用固定 val，未读取 test。

### 按 seed 配对的验证差值

差值为完整增强减去 LS 0.05；两组训练目标不同，train loss 绝对值不参与此表。

| seed | best 正确数差 | best val acc 差 | best val loss 差 | final val acc 差 | final val loss 差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260920 | +0 | +0.000000 | -0.055855 | +0.001235 | -0.060803 |
| 20260929 | +3 | +0.000741 | -0.061432 | +0.000741 | -0.056653 |
| 20260930 | -10 | -0.002469 | -0.052931 | -0.000494 | -0.061004 |

### 三个 seed 内的 best val accuracy 波动

| 配置 | seed 值 | 均值 | 最低 | 最高 | 样本标准差 | 正确数极差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| config36 / LS 0.05 | 20260920, 20260929, 20260930 | 0.982963 | 0.982222 | 0.983457 | 0.000653 | 5 |
| config41 / 完整增强 | 20260920, 20260929, 20260930 | 0.982387 | 0.980988 | 0.983210 | 0.001218 | 9 |

### 后段训练指标

| 配置 | seed | train loss 第10→15→20轮 | train acc 第10→15→20轮 | val acc 第15→20轮 | val loss 第15→20轮 |
| --- | ---: | --- | --- | --- | --- |
| config36 / LS 0.05 | 20260920 | 0.288114 → 0.284910 → 0.284326 | 0.999947 → 1.000000 → 1.000000 | 0.981235 → 0.981235 | 0.122743 → 0.124333 |
| config41 / 完整增强 | 20260920 | 0.050391 → 0.025058 → 0.015788 | 0.982328 → 0.991111 → 0.993968 | 0.980988 → 0.982469 | 0.070363 → 0.063530 |
| config36 / LS 0.05 | 20260929 | 0.288244 → 0.284996 → 0.284399 | 0.999947 → 1.000000 → 1.000000 | 0.981975 → 0.981728 | 0.120882 → 0.122616 |
| config41 / 完整增强 | 20260929 | 0.053075 → 0.026841 → 0.016858 | 0.981429 → 0.990000 → 0.994550 | 0.981975 → 0.982469 | 0.063264 → 0.065963 |
| config36 / LS 0.05 | 20260930 | 0.288134 → 0.284928 → 0.284337 | 0.999947 → 1.000000 → 1.000000 | 0.980988 → 0.980988 | 0.123223 → 0.124149 |
| config41 / 完整增强 | 20260930 | 0.051994 → 0.024370 → 0.017841 | 0.981217 → 0.991323 → 0.993704 | 0.979753 → 0.980494 | 0.063458 → 0.063145 |

LS 0.05 的 train loss 为平滑目标交叉熵；完整增强组为普通交叉熵。两种 train loss 只在各自组内看逐轮变化。

## 逐轮数据与曲线

- `runs/resnet18/analysis/two_seed_augmentation_20260929_01/epoch_comparison.csv`：六组的逐轮 train/val accuracy、loss、学习率与耗时。
- `runs/resnet18/analysis/two_seed_augmentation_20260929_01/run_summary.csv`：六组汇总。
- `runs/resnet18/analysis/two_seed_augmentation_20260929_01/paired_differences.csv`：三组同 seed 验证差值。
- `runs/resnet18/analysis/two_seed_augmentation_20260929_01/seed_comparison.png`：train/val 曲线；accuracy 缩放轴显示0.90–1.00附近变化。

原始逐轮记录、逐类指标、混淆矩阵、曲线与 best/last 权重分别位于四个新的 `runs/resnet18/<experiment_id>/` 目录。训练未使用 test。
