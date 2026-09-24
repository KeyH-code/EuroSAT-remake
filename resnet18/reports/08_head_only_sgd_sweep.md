# 08 冻结分类头：AdamW 与 SGD Momentum 对照

实验日期：2026-09-24

## 实验协议

本报告汇总 config08、config12–16 的同预算结果。所有配置使用独立的 ImageNet 预训练 ResNet-18 起点，冻结 backbone 与 BatchNorm，只训练新建的 10 类分类头。使用同一 EuroSAT split（SHA256：`66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf`）、batch size 128、seed 20260920、224×224 输入、ImageNet mean/std、FP16 autocast 与 GradScaler、无随机增强、`num_workers=0`；每组训练 20 轮。

config08 与 config12–14 的训练和 AdamW 资源细节见 [报告 07](07_head_only_adamw_sweep.md)，最初 config08 学习率比较见 [报告 06](06_parallel_lr_0p004_to_0p006_20ep.md)。config13 曾因内存压力暂停两次，并从自己的 last checkpoint 恢复，最终完成总计 20 轮；其余各组独立从预训练起点开始。config15 与 config16 按序单路运行，训练进程使用 BelowNormal 优先级。

所有选择只看 val accuracy；每组 best checkpoint 又由 `evaluate.py` 独立加载，在固定 val split 上复评。六份独立评价 summary 的 `test_split_used` 均为 `false`。test 未用于选模或调参。

## 20 轮结果

| 配置 | 优化器 / lr / wd / momentum | Best epoch | Best val accuracy | 正确 / 4050 | Best val loss | Epoch 20 val accuracy | Epoch 20 train loss | 逐轮秒数合计 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| config08 | AdamW / 0.003 / 0.01 / — | 15 | 0.952840 | 3859 | 0.150419 | 0.950864 | 0.087052 | 789.3 |
| config12 | AdamW / 0.003 / 0 / — | 15 | 0.952840 | 3859 | 0.150933 | 0.950864 | 0.085140 | 845.7 |
| config13 | AdamW / 0.01 / 0.01 / — | 15 | 0.948395 | 3841 | 0.177915 | 0.940741 | 0.082447 | 815.4 |
| config14 | AdamW / 0.01 / 0 / — | 12 | 0.948148 | 3840 | 0.171800 | 0.942716 | 0.075282 | 801.0 |
| config15 | SGD / 0.01 / 0 / 0.9 | 17 | 0.946914 | 3835 | 0.162531 | 0.945432 | 0.128678 | 1012.4 |
| config16 | SGD / 0.05 / 0 / 0.9 | 11 | 0.950864 | 3851 | 0.155821 | 0.949136 | 0.092515 | 978.0 |

六组逐轮记录里的 best epoch、accuracy 与 loss 均和独立 val 复评一致。config15 的独立复评为 3835/4050 正确、215 错；config16 为 3851/4050 正确、199 错。config16 的 best checkpoint 确认为 epoch 11，config15 为 epoch 17。

末 5 轮（epoch 16–20）的 val accuracy 汇总如下。标准差为总体口径（离差平方和除以 5），单位为百分点：

| 配置 | 末 5 轮均值 | 标准差 |
| --- | ---: | ---: |
| config08 | 94.904% | 0.138 |
| config12 | 94.928% | 0.150 |
| config13 | 93.990% | 0.578 |
| config14 | 94.064% | 0.500 |
| config15 | 94.573% | 0.093 |
| config16 | 94.474% | 0.377 |

config16 的单轮 best 高于 config15，但末五轮均值略低、波动更大；这两个观察可同时成立，不能只凭 best 认定它在各轮都更好。

## 结果解读

- AdamW lr=0.003 时，config12 去掉 weight decay 后与 config08 同为 3859/4050、best epoch 都是 15，epoch 20 accuracy 也相同；本次单次轨迹没有显示去掉 weight decay 提高准确率。config12 的 best val loss 比 config08 高约 0.000514。
- AdamW lr=0.01 的两组 best 分别为 3841 和 3840 张，较 config08 少 18 和 19 张。wd=0.01 的 config13 仅多对 1 张，单次结果不足以判断这一学习率下 weight decay 有稳定优势。
- SGD 两个配置中，lr=0.05 的 best 比 lr=0.01 多对 16 张；但比 config08/config12 少 8 张。与 AdamW lr=0.01 的 config13/14 相比，config16 分别多对 10/11 张，config15 分别少对 6/5 张。
- 这些是不同优化器与不同超参数下的轨迹，不能把相同数值的学习率解释成相同更新幅度。SGD 的 momentum 会改变更新累积；PyTorch SGD 的非零 `weight_decay` 是耦合到梯度的 L2 项，AdamW 则将衰减与自适应更新解耦。本次 SGD 两组 `weight_decay=0`，没有直接比较两种衰减机制。
- 较高学习率配置的末轮 train loss 较低，但不代表 val 指标同步变好。每组只有一个 seed，且未强制确定性；结果只描述这六次固定 20 轮轨迹，不能推断普遍最优值。batch size 始终为 128，本轮没有检验 batch size 的影响。

## 资源与执行记录

| 配置 | 训练方式 | 训练墙钟时间 | 训练中最低可用 RAM 采样 | nvidia-smi 整卡显存采样峰值 | PyTorch allocated / reserved 峰值 |
| --- | --- | ---: | ---: | ---: | ---: |
| config15 | 单进程、BelowNormal | 约 17 分 28 秒 | 约 4.28 GiB | 约 3083 / 8188 MiB | 0.917 / 1.550 GiB |
| config16 | 单进程、BelowNormal | 约 16 分 54 秒 | 约 4.60 GiB | 约 3031 / 8188 MiB | 0.917 / 1.550 GiB |

两次 SGD 训练均通过 `conda run -n dl-reboot python` 启动，config15 完全退出并完成 val 复评后才启动 config16。可用 RAM 没有接近 0.8 GiB 停止线，两组均无 OOM，也没有中断恢复。config15 期间 Windows 系统级硬分页计数器出现过几次短尖峰；一次读到约 7530 Pages/sec 与 883 Page Reads/sec，随后 2 秒样本快速回落为 455/45、0/0、37/4、0/0，彼时可用 RAM 仍约 4.60 GiB。其余时间读取很低。该计数器是系统级数据，不能仅凭它把尖峰归因于训练进程。config16 的分页采样大多为 0–1/sec，偶有短暂低尖峰，可用 RAM 保持在约 4.60 GiB 以上。

AdamW config12–14 的资源过程见报告 07：config12 曾短暂采样到约 0.97 GiB 可用 RAM 后恢复；config13 在可用 RAM 约 472–578 MiB 和 523 MiB 时两次暂停并从本组 checkpoint 恢复；config14 单路运行期间可用 RAM 约 4.1–4.7 GiB。报告 07 记录三组 PyTorch 峰值 allocated/reserved 约 917/1550 MiB。旧 config08 的早期并行资源情况见报告 06，不与本次单路运行直接比较。

## 产物

- config15 配置：`configs/config15_sgd_lr_0.01_momentum_0.9_20ep.json`
- config16 配置：`configs/config16_sgd_lr_0.05_momentum_0.9_20ep.json`
- config15 运行目录：`.cache/datasets/eurosat/resnet18_finetune/resnet18_head_sgd_lr0p01_m0p9_20ep_01/`
- config16 运行目录：`.cache/datasets/eurosat/resnet18_finetune/resnet18_head_sgd_lr0p05_m0p9_20ep_01/`
- 本次启动与复评日志：`.cache/datasets/eurosat/resnet18_finetune/sgd_experiment_logs/`

以上结果没有达到 M2 的自定义 CNN ≥98% 门槛，也不改变该验收条件。Agent 代跑结果只提供实验对照，不替代学习者的关键运行与审查。
