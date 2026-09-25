# EuroSAT RGB：独立训练与推理

这个仓库把小 CNN 与 ImageNet 预训练 ResNet18 微调从教学工程复制出来，并**独立运行**：不读取旧教学项目，也不读取任何历史产物。历史 `my_split.csv` 的字节未改；程序只在内存中把其中的旧图片路径映射到 `data/EuroSAT_RGB/2750/`。固定划分为 train 18900、val 4050、test 4050；训练过程只看 train/val，test 保留给最终一次评估。

## 目录

- `cnn/`：小 CNN 的 train/evaluate/predict、源码和配置模板。
- `resnet18/`：ResNet18 的 train/evaluate/predict、源码、配置及历史报告。
- `data/`：完整图片与原样清单，Git 忽略。
- `runs/`：新训练、评价、预测结果，Git 忽略。**所有产物只写这里。**
- `artifacts/`：仅存原预训练权重 `resnet18-f37072fd.pth`；运行缓存副本在 `.cache/torch/hub/checkpoints/`。
- `artifacts/legacy_eurosat/`：迁移时原样冻结的历史运行记录快照（1.24 GB）。**任何代码都不读它**，也不要在里面训练。内容索引与核对哈希见 `docs/migration-record.md` 与 `docs/legacy-configs/`。
- `docs/`：`migration-record.md`（迁移记录与历史结果）、`legacy-configs/`（旧配置档案）、`provenance.md`（来源核验）。

## 环境与入口

使用 `conda run -n dl-reboot python`。依赖见 `environment.yml`；本机已存在 `dl-reboot` 时无需重建。以下相对路径命令在仓库根目录运行：

```powershell
# 小 CNN：先复制模板到新实验目录，再改实验名与超参数
conda run -n dl-reboot python cnn/train.py --config runs/cnn/first_run/config.json
conda run -n dl-reboot python cnn/evaluate.py --checkpoint runs/cnn/first_run/checkpoints/best.pt
conda run -n dl-reboot python cnn/predict.py --checkpoint runs/cnn/first_run/checkpoints/best.pt --input-dir <图像目录>

# ResNet18：按配置的 experiment_id 新建 runs/resnet18/<experiment_id>/
conda run -n dl-reboot python resnet18/train.py --config resnet18/configs/config16_sgd_lr_0.05_momentum_0.9_20ep.json
conda run -n dl-reboot python resnet18/evaluate.py --config <本次配置> --checkpoint runs/resnet18/<experiment_id>/checkpoints/best.pt
conda run -n dl-reboot python resnet18/predict.py --config <本次配置> --checkpoint runs/resnet18/<experiment_id>/checkpoints/best.pt --input-dir <图像目录>
```

CNN 新实验要先将 `cnn/configs/baseline.json` 复制到一个全新的 `runs/cnn/<实验名>/config.json` 并按需修改；训练产物与该配置同目录。`cnn/evaluate.py` 与 `cnn/predict.py` 的 `--checkpoint` 是必填参数，且必须是本仓库 `runs/` 下的产物。

ResNet18 按 `experiment_id` 创建新目录，目录已存在会拒绝覆盖。`resnet18/configs/config01`–`16` 的 `experiment_id` 已被历史运行占用，复用同名配置前请先改 `experiment_id`。

## 历史产物与重训

历史实验记录**只作档案**：`docs/migration-record.md` 记录了当时每一组配置参数与 val 指标（小 CNN 最佳 0.9402、ResNet18 最佳 0.9538，均未达 98%），`docs/legacy-configs/README.md` 登记了旧配置的位置与 SHA-256。

要复用旧实验，**按记录里的参数在新仓库重新训练**，写入 `runs/`；不要从快照复制 checkpoint 或配置，也不要让新运行写回 `artifacts/legacy_eurosat/`。历史 checkpoint 不参与续训，续训只在本仓库新产物之间进行（`cnn/train.py --resume` 要求 checkpoint 自带配置与哈希）。

新仓库当前已有的基线产物：

- `runs/cnn/baseline_ep2/`：小 CNN 2 轮（`formal_training` 参数，仅缩短轮数），val acc 0.838519。
- `runs/resnet18/resnet18_head_probe_1ep_01/`：ResNet18 冻结分类头 1 轮，配置 `resnet18/configs/config17_probe_1ep.json`，val acc 0.905926。

两者都只用于验证入口链路，不构成 M2 验收证据。
