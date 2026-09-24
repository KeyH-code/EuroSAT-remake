# 06 冻结分类头：0.004–0.006 学习率三路并行对照

## 实验协议

学习者指定 AdamW 学习率 0.004、0.005、0.006 各运行 20 轮，并要求三路并行。Agent 以单次 `conda run -n dl-reboot python` 启动支持脚本，再由同一环境的 Python 同时启动三个独立训练进程；没有并发运行多个 Conda 激活。各进程从已缓存的 ImageNet 预训练 ResNet-18 起点初始化，冻结 backbone，仅训练新 10 类分类头，保留冻结 BN 的 ImageNet 运行统计量。三组均使用既有 EuroSAT split、batch 128、seed 20260920、ImageNet 归一化、AdamW weight decay 0.01、FP16 AMP、无随机增强、`num_workers=0`。每组配置、checkpoint、指标、图表与日志分别留档；仅用 val accuracy 严格升高时选择 `best.pt`，test 未参与。

配置：`configs/config09_lr_0.004_20ep.json`、`config10_lr_0.005_20ep.json`、`config11_lr_0.006_20ep.json`。并行入口为 `src/01_parallel_lr_sweep.py`；每组训练仍由学习者此前编写的 `src/model.py` 与 `src/train.py` 主体执行，Agent 不改其核心训练代码。并行只改变运行调度，不改变单组训练配置；三个进程共享 GPU/CPU/磁盘，因此运行速度不能当作学习率效果。

## 20 轮结果

| AdamW lr | Best epoch | Best val acc | 正确/4050 | Best val loss | Epoch 20 val acc | Epoch 20 train loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.003（先前单独运行） | 15 | **0.952840** | **3859** | **0.150419** | 0.950864 | 0.087052 |
| 0.004 | 15 | 0.951852 | 3855 | 0.153397 | 0.949383 | 0.082513 |
| 0.005 | 11 | 0.949136 | 3844 | 0.160802 | 0.948889 | 0.079668 |
| 0.006 | 11 | 0.951852 | 3855 | 0.155322 | 0.947654 | 0.078312 |

三组各有完整 20 条逐轮指标，训练进程均以退出码 0 结束；各自 `best.pt` 已由 `evaluate.py` 独立重载，在固定 val split 上复评，准确率、loss、checkpoint epoch 与逐轮记录一致，错误数分别为 195、206、195。0.005 的第 16 轮再次达到与第 11 轮相同的 0.949136；保存规则要求严格升高，所以 `best.pt` 仍是第 11 轮。

本次三组的第 20 轮 train loss 随学习率升高而降低，但验证集最佳值没有继续升高：0.004/0.006 均比先前 0.003 少正确 4 张，0.005 少 15 张。0.006 的 val loss 第 11 轮为 0.155322，第 17 轮升至约 0.1995，第 20 轮仍为 0.170419；单轮波动不能单独证明发散，但说明不能只看最低训练损失。这里是固定特征提取器、同一 20 轮预算下的单次非强制确定性结果，不能证明全局最佳学习率恰好是 0.003，也不能把差异归因于参数量。

## 并行资源与耗时

三组各自记录的 train+val 时间为 1692.0、1683.6、1698.2 秒；并行日志创建至最后一组结束约 **28 分 53 秒**。先前 0.0015/0.002/0.003 三个单独运行的 train+val 时间合计约 **39 分 32 秒**。这些实验处在不同时间、不同后台资源状态，且后者合计不包含轮外 checkpoint/绘图，所以这里只能说明此次三路并行可运行、总墙钟时间低于之前三组串行耗时的粗略参照，不能据此给出稳定加速比。

运行中抽查 GPU 整卡已用显存最高约 **7391/8188 MiB**，只剩约 0.6 GiB；系统可用内存曾降至约 315 MiB，出现明显换页。三组单进程 PyTorch `max_memory_reserved` 均为 1550 MiB，但该计数不覆盖驱动与其他进程显存；本次无 OOM，不保证同样后台负载下始终可三路并行。实验结束后进程正常退出，原有实验目录未覆盖。

## 产物

- 0.004：`.cache/datasets/eurosat/resnet18_finetune/resnet18_head_lr0p004_20ep_01/`
- 0.005：`.cache/datasets/eurosat/resnet18_finetune/resnet18_head_lr0p005_20ep_01/`
- 0.006：`.cache/datasets/eurosat/resnet18_finetune/resnet18_head_lr0p006_20ep_01/`
- 并行日志：`.cache/datasets/eurosat/resnet18_finetune/parallel_lr_logs/`
- 同预算 val 对照图：`.cache/datasets/eurosat/resnet18_finetune/lr_0p003_to_0p006_20ep_comparison.png`，由 `src/02_compare_lr_curves.py` 从各运行 CSV 绘制。

三组新实验与先前预训练 ResNet 结果均未达到 98%；现行 M2 的“自定义 CNN ≥98%”条件没有调整，本次 Agent 运行也不替代学习者亲手运行与审查。
