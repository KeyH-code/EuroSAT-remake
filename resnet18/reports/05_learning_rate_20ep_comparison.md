# 05 AdamW 学习率单变量对照：各 20 轮

## 实验协议

学习者先将原 0.001 方案保留，再指定 0.002、0.0015、0.003 三个学习率各从 ImageNet 预训练 ResNet-18 **独立从头训练 20 轮**。中间值按学习者纠正后的 **0.0015** 执行，未运行 0.015。三组使用独立 `experiment_id` 和运行目录，原 35 轮实验与 checkpoint 未改动。

除 AdamW 学习率外，三组均为冻结 backbone、只训练新 10 类分类头；输入 ImageNet 均值/标准差、冻结 BN 的 ImageNet 运行统计量、既有 EuroSAT split、batch 128、seed 20260920、AdamW weight decay 0.01、FP16 AMP、无随机增强、`num_workers=0`。每次由 `train.py` 使用 val accuracy 严格升高时保存 `best.pt`；test 不参与。配置依次为 `configs/config06_lr_0.002_20ep.json`、`config07_lr_0.0015_20ep.json`、`config08_lr_0.003_20ep.json`。模型和核心训练函数由学习者编写；三组运行与本报告由 Agent 完成，不替代学习者关键运行与审查。

## 相同 20 轮预算下的结果

原 0.001 运行累计到了 35 轮；下表只截取其**前 20 轮**作同预算参照。准确率分母均为固定 val 的 4050 张。

| AdamW lr | Best epoch ≤20 | Best val acc | 正确/4050 | Best val loss | Epoch 20 val acc | 20 轮 train+val 秒数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001（原方案前 20 轮） | 15 | 0.950617 | 3850 | 0.159865 | 0.946667 | 1620.23* |
| 0.0015 | 19 | 0.951605 | 3854 | 0.153141 | 0.946914 | 774.82 |
| 0.002 | 19 | 0.952099 | 3856 | 0.151327 | 0.946667 | 807.48 |
| **0.003** | **15** | **0.952840** | **3859** | **0.150419** | **0.950864** | **789.26** |

*原 0.001 的第 9 轮异常耗时 591.33 秒，原因未定；三组运行时 GPU 也可能受其他进程影响。计时仅含每轮训练加验证，不含 Conda 启动、checkpoint 与画图，不作为学习率造成速度差异的证据。

三组 `best.pt` 均已独立重载并在同一 val split 复评，得到与逐轮记录一致的 best epoch、loss、accuracy 与错误数：0.0015 为 196 错，0.002 为 194 错，0.003 为 191 错。三组 `metrics.jsonl` 各有完整 20 轮；`last.pt` 对应各自第 20 轮。各自的逐轮 CSV、学习曲线、逐类指标、混淆矩阵、错误清单及 checkpoint 保存在独立 `.cache` 运行目录，不进入 Git。

## 观察边界

在这一次同配置、同 20 轮预算的比较中，0.003 的最高 val accuracy 为三组最高，比原 0.001 前 20 轮多正确 **9 张**（约 **0.22 个百分点**）；0.002 多 6 张，0.0015 多 4 张。这些幅度较小，现有非强制确定性的单次运行不能证明学习率排序具有稳定因果性。

原 0.001 方案在 **第 29 轮**达到更高的 0.953827（3863/4050），但它使用了更长训练预算，不与三组 20 轮结果混成同预算排名。本次三组均未达到 98%；预训练 ResNet 的结果也不替代现行“自定义 CNN ≥98%”的 M2 要求。后续方案由学习者决定。

### 原始运行目录

- 0.0015：`E:\deeplearning\.cache\datasets\eurosat\resnet18_finetune\resnet18_head_lr0p0015_20ep_01`
- 0.002：`E:\deeplearning\.cache\datasets\eurosat\resnet18_finetune\resnet18_head_lr0p002_20ep_01`
- 0.003：`E:\deeplearning\.cache\datasets\eurosat\resnet18_finetune\resnet18_head_lr0p003_20ep_01`
- 原 0.001：`E:\deeplearning\.cache\datasets\eurosat\resnet18_finetune\resnet18_head_imagenet_5ep_01`
