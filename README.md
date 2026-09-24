# EuroSAT RGB：独立训练与推理

这个仓库把小 CNN 与 ImageNet 预训练 ResNet18 微调从教学工程复制出来。原教学工程仍在 `E:\deeplearning`，这里运行时不依赖它。历史 `my_split.csv` 的字节未改；程序只在内存中把其中的旧图片路径映射到 `data/EuroSAT_RGB/2750/`。固定划分为 train 18900、val 4050、test 4050；训练过程只看 train/val，test 保留给最终一次评估。

## 目录

- `cnn/`：小 CNN 的 train/evaluate/predict、源码和配置模板。
- `resnet18/`：ResNet18 的 train/evaluate/predict、源码、配置及历史报告。
- `data/`：完整图片与原样清单，Git 忽略。
- `artifacts/legacy_eurosat/`：迁移时从原缓存完整复制的每次运行记录，包含所有配置、日志、指标、图表、checkpoint 和预测输入；不在这里继续写实验。
- `artifacts/resnet18-f37072fd.pth`：原预训练权重；运行缓存副本在 `.cache/torch/hub/checkpoints/`。
- `runs/`：新训练、评价、预测结果，Git 忽略。
- `docs/provenance.md`：迁移核验与历史来源。

## 环境与入口

使用 `conda run -n dl-reboot python`。依赖见 `environment.yml`；本机已存在 `dl-reboot` 时无需重建。命令可从任意工作目录运行，以下以仓库根目录为例：

```powershell
conda run -n dl-reboot python cnn/train.py --config runs/cnn/first_run/config.json
conda run -n dl-reboot python cnn/evaluate.py --checkpoint artifacts/legacy_eurosat/checkpoints/best.pt
conda run -n dl-reboot python cnn/predict.py --input-dir artifacts/legacy_eurosat/predict_inputs/val_32_epoch5

conda run -n dl-reboot python resnet18/train.py --config resnet18/configs/config16_sgd_lr_0.05_momentum_0.9_20ep.json
conda run -n dl-reboot python resnet18/evaluate.py --config resnet18/configs/config16_sgd_lr_0.05_momentum_0.9_20ep.json --checkpoint artifacts/legacy_eurosat/resnet18_finetune/resnet18_head_sgd_lr0p05_m0p9_20ep_01/checkpoints/best.pt
conda run -n dl-reboot python resnet18/predict.py --input-dir artifacts/legacy_eurosat/predict_inputs/val_32_epoch5 --config resnet18/configs/config16_sgd_lr_0.05_momentum_0.9_20ep.json --checkpoint artifacts/legacy_eurosat/resnet18_finetune/resnet18_head_sgd_lr0p05_m0p9_20ep_01/checkpoints/best.pt
```

CNN 新实验要先将 `cnn/configs/baseline.json` 复制到一个全新的 `runs/cnn/<实验名>/config.json`，并按需修改实验名和超参数；训练产物与该配置同目录。ResNet18 按配置的 `experiment_id` 创建新目录，若目录已存在会拒绝覆盖。上述训练命令只是入口示例，迁移验收不启动长时间训练。

历史 checkpoint 可用于验证与预测；历史运行记录是不可变快照，原配置中的旧绝对路径仅作为来源证据。新的运行记录不会写回原教学目录。
