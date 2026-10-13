# EuroSAT RGB：训练、推理与最终评估

小 CNN 与 ImageNet 预训练 ResNet18 两条线，共用同一份 EuroSAT RGB 数据和固定划分。程序只在内存中把清单里记录的旧图片路径映射到 `data/EuroSAT_RGB/2750/`。固定划分为 train 18,900、val 4,050、test 4,050；训练和选模只使用 train/val。

截至 2026-09-29，ResNet18 阶段已锁定完整随机增强的 `layer4+fc` 模型，并完成**一次**最终 test 评估。锁定的 config41、seed 20260920、epoch 17 checkpoint 在 val 上正确 3,982/4,050（98.3210%），在 test 上正确 3,981/4,050（98.2963%）；test macro-F1 为 98.2250%，多分类 OvR macro ROC AUC 为 0.999668。权重仅依据 val 选定，test 未用于选模或调参。完整指标、哈希与逐类结果见 [最终结果](final_result/README.md)。

在同一固定 val 上，完整增强相对无增强、无 label smoothing 的纯基线，三个 seed 分别多正确 10、6、4 张；这是已测三个 seed 的小幅同向收益，不是 test 上的模型间差值。配对过程见 [三 seed 报告](resnet18/reports/30_pure_baseline_two_seed_20ep.md)。后续如研究解冻更多层或其他优化器，仍只用 train/val 决策，不用已查看的 test 结果调整方案。

## 目录

- `cnn/`：小 CNN 的 train/evaluate/predict、源码和配置模板。
- `resnet18/`：ResNet18 的 train/evaluate/predict、源码、配置及实验报告。当前实现的可训练范围为 **`layer4 + fc`**（`layer3` 及更早冻结），见 `resnet18/src/model.py` 的 `TRAINABLE_SCOPE`；后续三 seed 对照见[报告 30](resnet18/reports/30_pure_baseline_two_seed_20ep.md)。
- `baseline/`：CNN 与 ResNet18 已选验证配置的参考；来源和指标见 [baseline 说明](baseline/README.md)。这里仅放配置和说明，不放训练产物。
- `final_result/`：锁定的 ResNet18 模型、val 与单次 test 指标说明及计算脚本。详细逐样本结果与图片留在本机并被 Git 忽略。
- `data/`：完整图片与原样清单，Git 忽略。
- `runs/`：训练、新实验、常规评价及预测结果，Git 忽略；不得覆盖已有实验。已完成的最终 val/test 指标导出另保存在 `final_result/resnet18/` 的 Git 忽略目录中。
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

# ResNet18：新实验先创建未使用过的配置和 experiment_id
conda run -n dl-reboot python resnet18/train.py --config resnet18/configs/<新配置>.json
conda run -n dl-reboot python resnet18/evaluate.py --config <本次配置> --checkpoint runs/resnet18/<experiment_id>/checkpoints/best.pt
conda run -n dl-reboot python resnet18/predict.py --config <本次配置> --checkpoint runs/resnet18/<experiment_id>/checkpoints/best.pt --input-dir <图像目录>
```

CNN 新实验要先将 `cnn/configs/baseline.json` 复制到一个全新的 `runs/cnn/<实验名>/config.json` 并按需修改；训练产物与该配置同目录。`cnn/evaluate.py` 与 `cnn/predict.py` 的 `--checkpoint` 是必填参数，且必须是本仓库 `runs/` 下的产物。

ResNet18 按 `experiment_id` 创建新目录，目录已存在会拒绝覆盖。已有配置中大量 `experiment_id` 已被运行占用；启动新实验前必须使用新名称和新目录。最终模型的权重及配置在 [最终结果](final_result/README.md)中记录；最终 test 已完成，不应为了验证文档再次运行。

**学习率注意**：`config01`–`16` 的 lr（0.001–0.05）是**冻结分类头**设定下调出来的。解冻 layer4 后 0.001 与 3e-4 会在第一个 batch 之后退化（loss 塌到 0、logits 爆炸），已实测稳定上限约为 5e-5–1e-4。改配置时不要直接沿用旧 lr。

## 实验记录

- [最终结果](final_result/README.md)：锁定权重、val/test 汇总、逐类指标、混淆矩阵与本地结果目录。
- [三 seed 纯基线对照](resnet18/reports/30_pure_baseline_two_seed_20ep.md)：纯基线、label smoothing 与完整增强在三个 seed 上的配对 val 结果。
- [项目收尾记录](resnet18/reports/31_final_test_and_documentation_status.md)：最终 test 与后续研究边界。
- [baseline 说明](baseline/README.md)：CNN 与 ResNet18 配置来源及选模口径。
- `resnet18/reports/01`–`08`：ResNet18 冻结分类头阶段的逐次报告；[报告 09](resnet18/reports/09_history_baselines.md)是迁移前历史汇总。报告 10–30 记录后续各阶段实验。

各编号报告按当时完成的实验记录事实；早期报告中的“test 未使用”仅指其记录的训练或分析阶段，不代表项目当前状态。当前阶段状态以本页和最终结果说明为准。

历史实验要复用，按报告里的参数在 `runs/` 下重新训练；续训只在本仓库产物之间进行（`cnn/train.py --resume` 要求 checkpoint 自带配置与哈希，`resnet18/src/runner.py` 还要求 `trainable_scope` 一致）。

当前锁定的 ResNet18 运行目录是 `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_flip_rotate_translate_crop_20ep_01/`；其 `checkpoints/best.pt` 为 epoch 17。小 CNN 仍为历史对照线，本项目最终 test 结果只对应 ResNet18。

`data/`、`runs/` 和权重不进入 Git。只取得源码的读者还需要原始 RGB 图片、原样划分清单与相应权重才能复算最终指标；划分和模型文件的 SHA-256 见 [最终结果](final_result/README.md)。

## 测试

仓库当前没有自动化测试。改代码后按下述方式人工验证链路：

```powershell
conda run -n dl-reboot python cnn/train.py --config runs/cnn/<实验名>/config.json      # 跑通一轮
conda run -n dl-reboot python resnet18/train.py --config resnet18/configs/<新配置>.json  # 跑通一轮
conda run -n dl-reboot python -m compileall -q cnn resnet18 eurosat_paths.py
```
