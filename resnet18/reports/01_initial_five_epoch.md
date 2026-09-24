# 01 预训练 ResNet-18 分类头初测：5 轮

## 方案与证据边界

- 运行者：Agent 按学习者指定的初测方案执行；学习者编写了模型和核心训练函数。这次运行不替代学习者亲自完成的正式验收。
- 模型：Torchvision `ResNet18_Weights.DEFAULT`（当前为 ImageNet-1K V1），冻结原 backbone 参数，只训练新建的 10 类 `fc`。训练时 BN 保持 eval，沿用预训练运行统计量；与缓存的预训练权重比较，40 个 BN 统计量缓冲区训练前后完全一致。
- 输入：RGB 转 `[0,1]`、直接缩放到 `224×224`，按 ImageNet `mean=(0.485,0.456,0.406)`、`std=(0.229,0.224,0.225)` 归一化；无随机增强。训练和验证使用同一归一化。
- 数据：既有 `my_split.csv`，train 18,900、val 4,050；未读取 test。仅凭 val accuracy 选择 best。
- 参数：5 epochs；沿用既有训练入口默认 batch size 128、seed 20260920；AdamW 使用框架默认 lr 0.001、weight decay 0.01、betas `(0.9,0.999)`，其余默认。训练前向使用 CUDA FP16 autocast 和 GradScaler；loader `num_workers=0`、`pin_memory=False`。
- 环境：Windows、RTX 4060 Laptop 8GB，PyTorch `2.14.0+cu130`、Torchvision `0.29.0+cu130`。运行前 GPU 已占用约 2452 MiB、利用率约 41%，因此本次耗时包含其他 GPU 负载影响。
- 配置：`configs/config01.json`；运行目录：`E:\deeplearning\.cache\datasets\eurosat\resnet18_finetune\resnet18_head_imagenet_5ep_01`。

## 逐轮结果

| Epoch | Train loss（在线） | Train acc（在线） | Val loss | Val acc | 本轮秒数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7892 | 0.8206 | 0.3873 | 0.9059 | 55.96 |
| 2 | 0.3239 | 0.9121 | 0.2838 | 0.9220 | 38.37 |
| 3 | 0.2564 | 0.9247 | 0.2402 | 0.9306 | 39.27 |
| 4 | 0.2244 | 0.9312 | 0.2163 | 0.9360 | 39.86 |
| 5 | 0.2043 | 0.9359 | 0.2034 | **0.9373** | 51.41 |

训练入口记录的 5 轮训练加验证用时合计 **224.87 秒（约 3 分 45 秒）**，平均 44.97 秒/轮；未包含 Conda 启动、图表和 checkpoint 写盘等轮外时间。峰值 PyTorch allocated 917.0 MiB、reserved 1550.0 MiB；这是进程统计值，不包括其他进程占用。

第 5 轮为 best：val 正确 **3796/4050**，错误 254，loss 0.2034。独立重载 `best.pt` 并在相同 val split 复评，得到一致的 loss、accuracy 和错误数。逐类 recall 最低为 Highway 0.8693、PermanentCrop 0.8907、River 0.8933；最高为 Industrial 0.9787、Forest 0.9778。

同一 split 的小型 `EurosatCNN` 已验证 best 为 0.9402@28（3808/4050），比本次初测多正确 12 张。这里的网络结构、预训练来源、输入尺度和训练轮数等均不同；这组数值只作当前表现对照，不能把差异单独归因于参数量或预训练。本次 5 轮结果也不满足现行的“自定义 CNN ≥98%”条件，不能据此关闭 M2。

原始前 5 轮配置保留在 `configs/config01.json`，逐轮指标保留在上述运行目录 `metrics.jsonl` 的第 1–5 行。该目录后来按学习者要求续训至第 35 轮；当前 checkpoint、图表与 `final_evaluation/` 已更新，详情见 `reports/02_resume_to_epoch15.md` 至 `reports/04_resume_to_epoch35.md`。权重与逐样本文件保留在 `.cache`，不进入 Git。
