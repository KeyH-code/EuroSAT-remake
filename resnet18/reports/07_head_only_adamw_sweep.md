# 07 冻结分类头：AdamW 学习率与 weight decay 对照

实验日期：2026-09-24

## 实验协议

本轮比较三种新配置，并把 config08 作为同预算历史参照：

| 配置 | AdamW 学习率 | weight decay | 说明 |
| --- | ---: | ---: | --- |
| config08 | 0.003 | 0.01 | 已有基准，结果见 [报告 06](06_parallel_lr_0p004_to_0p006_20ep.md) |
| config12 | 0.003 | 0 | 去除 weight decay |
| config13 | 0.01 | 0.01 | 提高学习率 |
| config14 | 0.01 | 0 | 同时提高学习率并去除 weight decay |

四组都使用缓存的 ImageNet 预训练 ResNet-18，冻结 backbone 与 BatchNorm，只训练新建的 10 类分类头。使用同一 EuroSAT split（SHA256：66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf）、batch size 128、seed 20260920、224×224 输入、ImageNet mean/std、FP16 autocast 与 GradScaler、无随机增强、num_workers=0。每组预算为 20 轮，从预训练权重独立开始；config13 因资源压力从本组 checkpoint 恢复了两次，最终仍完成总计 20 轮。本轮三组新配置依次单路运行，训练 Python 进程使用 BelowNormal 优先级。

优化器只接收 requires_grad=True 的分类头参数，因此这些 weight decay 对照作用在分类头，不更新预训练 backbone。训练每轮由 val accuracy 选择 best checkpoint；各组随后以 evaluate.py 独立重载 best，在固定 val split 复评。test 未用于选模或调参。

配置文件分别为 configs/config08_lr_0.003_20ep.json、configs/config12_adamw_lr_0.003_wd_0_20ep.json、configs/config13_adamw_lr_0.01_wd_0.01_20ep.json 和 configs/config14_adamw_lr_0.01_wd_0_20ep.json。

## 20 轮结果

| 配置 | lr / wd | Best epoch | Best val accuracy | 正确 / 4050 | Best val loss | Epoch 20 val accuracy | Epoch 20 train loss | 逐轮计时合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| config08 | 0.003 / 0.01 | 15 | 0.952840 | 3859 | 0.150419 | 0.950864 | 0.087052 | 历史报告未列 |
| config12 | 0.003 / 0 | 15 | 0.952840 | 3859 | 0.150933 | 0.950864 | 0.085140 | 845.7 秒 |
| config13 | 0.01 / 0.01 | 15 | 0.948395 | 3841 | 0.177915 | 0.940741 | 0.082447 | 815.4 秒 |
| config14 | 0.01 / 0 | 12 | 0.948148 | 3840 | 0.171800 | 0.942716 | 0.075282 | 801.0 秒 |

config08 的数字来自此前 20 轮独立运行及报告 06。config12、13、14 的 best.pt 复评值均与逐轮记录相符；对应错误数为 191、209、210，三份独立评价 summary 均记录 test_split_used=false。表中逐轮计时合计来自各 run 的 20 个 epoch seconds 之和，不包含进程启动、暂停等待或最终 val 复评时间。完整产物目录如下：

- config08：.cache/datasets/eurosat/resnet18_finetune/resnet18_head_lr0p003_20ep_01/
- config12：.cache/datasets/eurosat/resnet18_finetune/resnet18_head_lr0p003_wd0_20ep_01/
- config13：.cache/datasets/eurosat/resnet18_finetune/resnet18_head_lr0p01_wd0p01_20ep_01/
- config14：.cache/datasets/eurosat/resnet18_finetune/resnet18_head_lr0p01_wd0_20ep_01/

## config13 的资源中断与恢复

config12 完整运行；期间一次采样到系统可用内存 0.97 GiB，约 15 秒后恢复到 1.32 GiB，此后运行完成。其峰值 GPU allocated/reserved 为 917/1550 MiB。

config13 的第一次从头运行完成 epoch 1–2 后暂停：可用内存降至约 472–578 MiB，PageReads/sec 约 595，用户当时的 VS Code/WSL 也占用系统内存。epoch 2 的 last.pt 与 metrics 已保存。之后从该 checkpoint 恢复并完成 epoch 3–7；可用内存再次降至约 523 MiB，PageReads/sec 约 965，于是再次暂停。epoch 7 的 metrics 与 last.pt 均已写入。

用户关闭 VS Code 后，复测可用内存约 6.5 GiB、硬分页采样为 0，再从 epoch 7 的 last.pt 恢复并完成 epoch 8–20。该段训练期间可用内存约 4.1–4.9 GiB，未再次触发暂停条件。最终 metrics 共 20 轮，best checkpoint 和 last checkpoint 均已保存，独立 val 复评完成。

config14 在 config13 复评结束后才启动。训练前可用内存约 6.3 GiB，运行期间约 4.1–4.7 GiB，硬分页采样接近 0；完成和复评后可用内存约 6.1–6.4 GiB。三组峰值 GPU allocated/reserved 均约为 917/1550 MiB；训练期间 nvidia-smi 总占用的观察值通常约 2.5–3.5 GiB / 8.19 GiB。各阶段按单路顺序执行，没有训练进程重叠。

## 结果解读

- 在 lr=0.003 时，去除 weight decay 后 best 与 config08 同为 3859/4050，epoch 20 val accuracy 也相同；本次单次轨迹没有显示去除 weight decay 带来准确率提升。
- 在 lr=0.01 时，wd=0.01 的 config13 比 wd=0 的 config14 多正确 1 张。两者只差约 0.025 个百分点，单次运行不足以判断 weight decay 在该学习率下有稳定优势。
- 两组 lr=0.01 的 best 分别比 config08 少正确 18 张和 19 张；它们的 epoch 20 train loss 更低，但末轮 val accuracy 也更低。就这几次固定 20 轮轨迹而言，提高学习率降低了训练 loss，没有提高最佳验证准确率。
- 每种新配置只运行一个 seed，且训练未启用强制确定性。以上是这几次运行的观测，不代表对所有随机轨迹或超参数范围的普遍结论；模型选择和分析仍只使用 val，test 未触碰。
