# EuroSAT RGB：训练与推理

小 CNN 与 ImageNet 预训练 ResNet18 两条线，共用同一份 EuroSAT RGB 数据和固定划分。
程序只在内存中把清单里记录的旧图片路径映射到 `data/EuroSAT_RGB/2750/`。
固定划分为 train 18900、val 4050、test 4050；训练过程只看 train/val，test 保留给最终一次评估。

## 目录

- `cnn/`：小 CNN 的 train/evaluate/predict、源码和配置模板。
- `resnet18/`：ResNet18 的 train/evaluate/predict、源码、配置及实验报告。当前可训练范围为 **`layer4 + fc`**（`layer3` 及更早冻结），见 `resnet18/src/model.py` 的 `TRAINABLE_SCOPE`；结论依据见 `resnet18/reports/09_history_baselines.md`。
- `baseline/`：CNN 与 ResNet18 当前最佳验证结果对应的训练配置参考；基线来源和指标见 `baseline/README.md`。这里仅放配置和说明，不放训练产物。
- `data/`：完整图片与原样清单，Git 忽略。
- `runs/`：训练、评价、预测结果，Git 忽略。**所有产物只写这里。**
- `artifacts/`：原预训练权重 `resnet18-f37072fd.pth`；运行缓存副本在 `.cache/torch/hub/checkpoints/`。
- `artifacts/legacy_eurosat/`：历史运行记录快照（1.24 GB），只读、不被任何代码读取。
- `artifacts/legacy_teaching/`：教学期与迁移期的代码/文档归档，**不可运行、不可 import**；内容清单见该目录的 `README.md`。

两条模型线采用同一套结构约定：

- 顶层 `train.py` / `evaluate.py` / `predict.py` 是**命令行入口**，只做参数解析、路径设置与结果落盘；`resnet18/train.py` 只是把 `main()` 从 `src/runner.py` 转发出来。
- `src/` 放**可复用模块**，不放 `main()`：`data.py` 数据、`model.py` 模型、`transforms.py` 预处理、`optimizer.py` 优化器、`loop.py`（CNN 侧为 `train_fit_loop.py`）训练循环、`runner.py`（CNN 侧为 `experiment_runner.py`）实验编排、`metrics.py` 指标、`checkpoint.py` 存档、`inference.py` 推理、`visualization.py` 图表。
- 所有入口用同一套路径变量：`PROJECT_DIR`（本目录）+ `SRC_DIR = PROJECT_DIR / "src"`。

## 环境与入口

使用 `conda run -n dl-reboot python`。依赖见 `environment.yml`；本机已存在 `dl-reboot` 时无需重建。以下相对路径命令在仓库根目录运行：

```powershell
# 小 CNN：先复制模板到新实验目录，再改实验名与超参数
conda run -n dl-reboot python cnn/train.py --config runs/cnn/first_run/config.json
conda run -n dl-reboot python cnn/evaluate.py --checkpoint runs/cnn/first_run/checkpoints/best.pt
conda run -n dl-reboot python cnn/predict.py --checkpoint runs/cnn/first_run/checkpoints/best.pt --input-dir <图像目录>

# ResNet18：按配置的 experiment_id 新建 runs/resnet18/<experiment_id>/
conda run -n dl-reboot python resnet18/train.py --config resnet18/configs/config18_layer4_lr5e-5_20ep.json
conda run -n dl-reboot python resnet18/evaluate.py --config <本次配置> --checkpoint runs/resnet18/<experiment_id>/checkpoints/best.pt
conda run -n dl-reboot python resnet18/predict.py --config <本次配置> --checkpoint runs/resnet18/<experiment_id>/checkpoints/best.pt --input-dir <图像目录>
```

CNN 新实验要先将 `cnn/configs/baseline.json` 复制到一个全新的 `runs/cnn/<实验名>/config.json` 并按需修改；训练产物与该配置同目录。`cnn/evaluate.py` 与 `cnn/predict.py` 的 `--checkpoint` 是必填参数，且必须是本仓库 `runs/` 下的产物。

ResNet18 按 `experiment_id` 创建新目录，目录已存在会拒绝覆盖。`resnet18/configs/config01`–`17` 的 `experiment_id` 已被运行占用，复用同名配置前请先改 `experiment_id`。

**学习率注意**：`config01`–`16` 的 lr（0.001–0.05）是**冻结分类头**设定下调出来的。解冻 layer4 后 0.001 与 3e-4 会在第一个 batch 之后退化（loss 塌到 0、logits 爆炸），已实测稳定上限约为 5e-5–1e-4。改配置时不要直接沿用旧 lr。

## 实验记录

- `baseline/README.md`：CNN 与 ResNet18 的最佳验证配置来源、选择口径及对应指标。
- `resnet18/reports/01`–`08`：ResNet18 冻结分类头阶段的逐次实验报告。
- `resnet18/reports/09_history_baselines.md`：两个模型全部历史实验的 val 指标与参数索引汇总。

历史实验要复用，按报告里的参数在 `runs/` 下重新训练；续训只在本仓库产物之间进行（`cnn/train.py --resume` 要求 checkpoint 自带配置与哈希，`resnet18/src/runner.py` 还要求 `trainable_scope` 一致）。

当前已有产物：

- `runs/cnn/baseline_ep2/`：小 CNN 2 轮，val acc 0.838519。
- `runs/resnet18/resnet18_head_probe_1ep_01/`：冻结分类头 1 轮（旧设定），val acc 0.905926。

## 测试

仓库当前没有自动化测试。改代码后按下述方式人工验证链路：

```powershell
conda run -n dl-reboot python cnn/train.py --config runs/cnn/<实验名>/config.json      # 跑通一轮
conda run -n dl-reboot python resnet18/train.py --config resnet18/configs/<新配置>.json  # 跑通一轮
conda run -n dl-reboot python -m compileall -q cnn resnet18 eurosat_paths.py
```
