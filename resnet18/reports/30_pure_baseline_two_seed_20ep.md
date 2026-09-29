# 30：纯基线 config32 再换两个 seed 的 20 轮对照

记录日期：2026-09-29

## 用户分析与决策

> 将纯基线 config32（无随机增强、无 label smoothing）分别用 20260929、20260930 跑 20 epoch。学习率、batch size、优化器、调度器和数据划分都保持一致。与既有同 seed 的 label smoothing 和完整随机增强实验对照，查看两项措施能否带来稳定收益。

## 实验口径

三组均为 ImageNet 预训练 ResNet18、`layer4+fc`、AdamW、lr=1e-4、weight decay 0.01、batch64、1 epoch 线性 warmup、20 epoch cosine、相同 Resize/Normalize、固定 train/val 划分。划分清单 `data/manifests/my_split.csv` 的 SHA-256 为 `66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf`；训练只读取 train/val，test 未使用。

纯基线的新配置从 config32 复制，逐字段核对后仅 `experiment_id` 和 `loader.seed` 不同；配置中没有 `label_smoothing` 或 `train_augmentation`。两个实验各创建新目录，不续训、不覆盖既有结果。

| seed | 纯基线配置 | 新运行目录 |
| ---: | --- | --- |
| 20260929 | `resnet18/configs/config48_layer4_lr0p0001_warmup_cosine_bs64_seed20260929_20ep.json` | `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_seed20260929_20ep_01/` |
| 20260930 | `resnet18/configs/config49_layer4_lr0p0001_warmup_cosine_bs64_seed20260930_20ep.json` | `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_seed20260930_20ep_01/` |

既有对照：seed 20260920 的 config32/config36/config41，seed 20260929 的 config44/config45，seed 20260930 的 config46/config47。config36/44/46 相对纯基线只增加 label smoothing 0.05；config41/45/47 相对纯基线只增加完整随机 train 增强，均为独立训练。best checkpoint 以 val accuracy 最大选择，若并列保留较早轮次。val loss 为普通交叉熵；有平滑组的在线 train loss 目标不同，因此三组间不直接比较 train loss 的绝对值。

## 新训练与独立 val 复评

两组各运行完整20轮，优化器跳步均为0。seed 20260929 总训练耗时1122.49秒；seed 20260930 为937.48秒。独立复评重新载入各自 best checkpoint，val accuracy、loss 与训练时对应轮次完全一致，只使用固定 val。

| seed | best epoch | best val 正确/4050 | best val acc | best val loss | 末轮 train acc/loss | 末轮 val acc/loss | 独立复评目录 |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 20260929 | 12 | 3975 | 0.981481 | 0.090102 | 1.000000 / 0.000054 | 0.981481 / 0.094377 | `runs/resnet18/evaluations/resnet18_layer4_lr0p0001_warmup_cosine_bs64_seed20260929_20ep_01_20260929T145313_973839/` |
| 20260930 | 9 | 3969 | 0.980000 | 0.076748 | 1.000000 / 0.000050 | 0.979753 / 0.089947 | `runs/resnet18/evaluations/resnet18_layer4_lr0p0001_warmup_cosine_bs64_seed20260930_20ep_01_20260929T145332_165177/` |

独立复评的误判数分别为75和81。逐轮 `epoch_metrics.csv`、`metrics.jsonl`、逐类指标、混淆矩阵、曲线、best/last 权重在各自实验目录中；复评目录另有 best checkpoint 的错误图片列表和混淆矩阵。

## 三个 seed、三种配置并排对照

| seed | 配置 | best epoch | best val 正确/4050 | best val acc | best val loss | 末轮 train acc | 末轮 val acc |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260920 | 纯基线 config32 | 12 | 3972 | 0.980741 | 0.087566 | 1.000000 | 0.980494 |
| 20260920 | LS 0.05 config36 | 7 | 3982 | 0.983210 | 0.117645 | 1.000000 | 0.981235 |
| 20260920 | 完整增强 config41 | 17 | 3982 | 0.983210 | 0.061789 | 0.993968 | 0.982469 |
| 20260929 | 纯基线 config48 | 12 | 3975 | 0.981481 | 0.090102 | 1.000000 | 0.981481 |
| 20260929 | LS 0.05 config44 | 13 | 3978 | 0.982222 | 0.123532 | 1.000000 | 0.981728 |
| 20260929 | 完整增强 config45 | 16 | 3981 | 0.982963 | 0.062100 | 0.994550 | 0.982469 |
| 20260930 | 纯基线 config49 | 9 | 3969 | 0.980000 | 0.076748 | 1.000000 | 0.979753 |
| 20260930 | LS 0.05 config46 | 10 | 3983 | 0.983457 | 0.117890 | 1.000000 | 0.980988 |
| 20260930 | 完整增强 config47 | 7 | 3973 | 0.980988 | 0.064959 | 0.993704 | 0.980494 |

![三个 seed 的 best val accuracy 与普通 CE](../../runs/resnet18/analysis/three_seed_factor_comparison_20260929_01/three_seed_best_comparison.png)

图左 accuracy 轴缩放为0.975–0.985，图右普通 CE 轴为0.05–0.14。两个指标均取**最高 val accuracy 的 checkpoint**；右图不是各组独立的最低 val loss 轮次。

### 同 seed 相对纯基线的差值

| seed | 措施 | best 正确数差 | best val acc 差 | best val loss 差 | 末轮 val acc 差 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20260920 | LS 0.05 | +10 | +0.002469 | +0.030079 | +0.000741 |
| 20260920 | 完整增强 | +10 | +0.002469 | −0.025776 | +0.001975 |
| 20260929 | LS 0.05 | +3 | +0.000741 | +0.033430 | +0.000247 |
| 20260929 | 完整增强 | +6 | +0.001481 | −0.028002 | +0.000988 |
| 20260930 | LS 0.05 | +14 | +0.003457 | +0.041142 | +0.001235 |
| 20260930 | 完整增强 | +4 | +0.000988 | −0.011789 | +0.000741 |

三个 seed 中，LS 0.05 与完整增强的 best val 正确数都高于各自同 seed 纯基线；LS 分别多10、3、14张，完整增强分别多10、6、4张。普通 val CE 方面，LS 三次均高于纯基线，完整增强三次均低于纯基线。上述为当前固定 val 和这三个 seed 的观测；test 仍保留用于之后的最终检验。

### 同 seed 的错误图片交换

“纠正”是纯基线误判而措施组正确，“新增误判”是纯基线正确而措施组误判；两项之差等于上表 best 正确数差。原图路径去重后比较两组各自独立复评的 `error_samples.csv`。

| seed | 措施 | 共同误判 | 纠正纯基线错误 | 措施组新增误判 | 净正确数增量 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20260920 | LS 0.05 | 54 | 24 | 14 | +10 |
| 20260920 | 完整增强 | 48 | 30 | 20 | +10 |
| 20260929 | LS 0.05 | 56 | 19 | 16 | +3 |
| 20260929 | 完整增强 | 50 | 25 | 19 | +6 |
| 20260930 | LS 0.05 | 56 | 25 | 11 | +14 |
| 20260930 | 完整增强 | 54 | 27 | 23 | +4 |

这些是同一固定 val 上的样本转换数量；不把不同 seed 的同一图片重复计为独立图片。

## 数据落盘

- `runs/resnet18/analysis/three_seed_factor_comparison_20260929_01/run_summary.csv`：九组汇总。
- 同目录 `epoch_comparison.csv`：九组共180轮逐轮 train/val 指标、学习率与耗时。
- 同目录 `paired_differences.csv`、`paired_error_changes.csv`：同 seed 与纯基线的数值差和错误图片交换。
- 同目录 `summary.json`、`three_seed_best_comparison.png`：结构化汇总与缩放轴对照图。
- 同目录 `analyze.py`：从配置、训练逐轮记录及独立复评文件复算上述表格。

源码区只增加 config48/config49 和本报告；运行数据、复评结果、图片和权重均留在 Git 忽略的 `runs/`。本轮未提交 Git。
