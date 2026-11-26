# EuroSAT RGB 遥感场景分类

给定一张 Sentinel-2 RGB 图块，预测农田、森林、道路、水体等十种土地利用与覆盖类别。这个个人项目用小型 CNN 建立参考基线，用 ImageNet 预训练 ResNet18 做迁移学习，并在固定数据划分上检验微调和数据增强方案。

**最终测试集 Accuracy 为 98.30%（3,981 / 4,050），Macro-F1 为 98.23%。** 在验证集上确定增强方案并锁定权重后，对保留测试集评价一次。

[快速运行](#快速运行) · [修改与训练](#修改与训练) · [复查结果](#复查结果)

## 目标与结果

项目早期的 Accuracy 目标为 **98%**。沿用 18,900 / 4,050 / 4,050 的固定训练、验证、测试划分。最终模型微调 ResNet18 的 `layer4` 和分类头，同时使用随机翻转、90° 倍数旋转、裁剪和平移。训练预算为 20 轮，锁定的是第 17 轮的权重。

| 模型与划分 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| 小型 CNN，历史验证集 | 94.02%（3,808 / 4,050） | — |
| ResNet18 仅训练分类头，历史验证集 | 95.38%（3,863 / 4,050） | — |
| 最终 ResNet18，验证集 | 98.32%（3,982 / 4,050） | 98.26% |
| 最终 ResNet18，测试集 | **98.30%（3,981 / 4,050）** | **98.23%** |

历史验证结果来自不同阶段的训练设置。相同预算下的增强收益见配对对照。[历史基线](resnet18/reports/09_history_baselines.md) · [最终评估](final_result/README.md)

## 项目贡献与个人职责

- **设计微调和增强对照。** 我确定了从只训练分类头到微调 `layer4 + fc` 的实验路线，并设计三 seed 配对对照。在每个 seed 的相同 20 轮训练条件下，完整增强相对纯基线分别多判对 **10、6、4 张验证图**。[配对对照](resnet18/reports/30_pure_baseline_two_seed_20ep.md)
- **根据错分提出并检验假设。** 我从重复误判和混淆矩阵中识别 Highway / River 的错分方向，再设计定向增强对照。提高这两类的裁剪面积下限后，双向错分从 11 降到 10，整体正确数从 3,982 降到 3,981。最终保留通用增强方案。[定向增强](resnet18/reports/27_highway_river_targeted_augmentation_20ep.md)
- **依据验证证据锁定最终模型。** 我先确定完整增强方案，再从三个该方案的 seed 中按验证表现锁定 config41、seed `20260920`、epoch 17。仓库保留配置、评估脚本、逐类结果和混淆矩阵。[选模与评价记录](resnet18/reports/31_final_test_and_documentation_status.md)

## 结果适用范围

分数适用于本项目固定的 EuroSAT RGB 划分。已核对训练和验证图像无字节或解码 RGB 像素完全重复，现有划分缺少地理元数据，跨地域泛化仍待验证。下一步可在具有位置元数据和人工标签的独立影像上评价冻结模型，单列跨区结果。[数据审计](resnet18/reports/22_fixed_transforms_clean_eval_similarity_audit.md)

## 快速运行

用 CPU 运行一张 RGB 样例，得到文件名、预测类别和最高类别分数。使用 Python 3.12。在 [GitHub Releases](https://github.com/KeyH-code/EuroSAT-remake/releases) 查看 `eurosat-demo.zip` 的发布状态，下载后放在仓库根目录。附件内含 `resnet18-inference.pt`、`sample/Forest_3.jpg` 和样例许可说明。该样例来自固定训练划分，仅用于验证推理链路。

```powershell
python -m venv .venv
$demoPython = '.\.venv\Scripts\python.exe'
& $demoPython -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0' 'torchvision==0.29.0'
& $demoPython -m pip install 'pandas==2.3.3' 'Pillow==11.3.0'
Get-FileHash -LiteralPath .\eurosat-demo.zip -Algorithm SHA256 # 与最终评估页的附件哈希核对
Expand-Archive -LiteralPath .\eurosat-demo.zip -DestinationPath .\runs\demo
$config = (Get-Item -Path .\resnet18\configs\config41_*.json).FullName
& $demoPython resnet18/predict.py --config $config --input-dir runs/demo/sample --inference-weights runs/demo/resnet18-inference.pt --output-csv runs/demo/predictions.csv
Import-Csv -LiteralPath .\runs\demo\predictions.csv
```

预期输出为单行 CSV，`pred_idx=1`、`pred_class=Forest`、`confidence` 验收参考值为 `1.0`。脚本拒绝覆盖已有输出。推理权重只包含模型参数、类别顺序和预处理配置，CPU 路径可用任意 JPEG 或 PNG 目录，不需要本项目的划分清单。源 checkpoint 和推理文件的哈希见[最终评估](final_result/README.md#快速运行附件)。

## 修改与训练

训练和正式验证使用 CUDA，沿用 `dl-reboot` Conda 环境。[环境配置](environment.yml)

1. 从 [EuroSAT 数据页](https://zenodo.org/records/7711810) 获取 RGB 图片，并放在 `data/EuroSAT_RGB/2750/<类别>/<文件名>`。将原样划分清单放到 `data/manifests/my_split.csv`，包含 `filepath,label,label_idx,split`。代码在内存中按类别和文件名映射图片路径，并校验原始 CSV 哈希。原样清单留存于项目本地。
2. 实验配置使用新的运行名。ResNet18 可复制[基线配置](baseline/resnet18_best.json)到 `runs/config41-local.json`，修改 `experiment_id` 为未使用的名称后保存为 UTF-8。小 CNN 可将[配置模板](cnn/configs/baseline.json)复制到 `runs/cnn/first_run/config.json`，并按模板修改参数。
3. 从仓库根目录调用活动入口。ResNet18 训练结果写入 `runs/resnet18/<experiment_id>/`，普通评价默认在 `runs/` 下新建目录。

```powershell
conda run -n dl-reboot python resnet18/train.py --config runs/config41-local.json
conda run -n dl-reboot python resnet18/evaluate.py --config runs/config41-local.json --checkpoint 'runs/resnet18/<experiment_id>/checkpoints/best.pt'
conda run -n dl-reboot python cnn/train.py --config runs/cnn/first_run/config.json
conda run -n dl-reboot python cnn/evaluate.py --checkpoint runs/cnn/first_run/checkpoints/best.pt
conda run -n dl-reboot python cnn/predict.py --checkpoint runs/cnn/first_run/checkpoints/best.pt --input-dir '<图像目录>'
```

ResNet18 的 `config01`–`16` 是当时冻结 backbone、仅训练分类头的历史配置。当前训练器使用 `layer4 + fc`，对照范围和学习率探查见[报告 10](resnet18/reports/10_layer4_lr0p001_20ep.md)和[报告 11](resnet18/reports/11_layer4_lr_magnitude_probe_10ep.md)。原训练 checkpoint 用于验证、续训或 `resnet18/predict.py --checkpoint`，加载时校验清单哈希及类别顺序。续训还校验可训练范围。

## 复查结果

[最终评估](final_result/README.md)包含配置、原 checkpoint 和划分清单的 SHA-256，以及验证集和测试集的逐类指标和混淆矩阵。[三 seed 报告](resnet18/reports/30_pure_baseline_two_seed_20ep.md)记录各配置和验证集对照。完整复算需要原始 RGB、原样划分和对应的原训练 checkpoint。测试集已用于本阶段的一次评价，后续选模和参数调整继续依据训练集与验证集。

## 目录

| 路径 | 内容与命名 |
| --- | --- |
| `cnn/`、`resnet18/` | 模型线。`train.py`、`evaluate.py`、`predict.py` 为命令行入口。`src/` 按职责命名，例如 `inference.py`、`transforms.py`、`runner.py`。 |
| `cnn/configs/`、`resnet18/configs/` | JSON 配置。CNN 使用 `baseline.json`，ResNet18 按 `configNN_<差异>.json` 保留历史。 |
| `resnet18/reports/`、`baseline/` | `NN_<主题>.md` 实验报告与 `<模型>_best.json` 基线参考。 |
| `final_result/` | `README.md` 汇总和 `compute_resnet18_*.py` 评估脚本。`resnet18/{validation,test}/` 为本地指标产物。 |
| `data/`、`runs/`、`logs/`、`error_samples/`、`.cache/` | 本地数据、实验和派生文件。训练用新实验名区分运行，预测、分析等使用各自输出目录。 |
| `artifacts/` | 本地预训练权重和历史档案。`legacy_eurosat/` 冻结保存，`legacy_teaching/` 为教学和迁移参考。活动代码使用项目数据和运行目录。 |

## 测试

改动后可先运行 `python -m compileall -q cnn resnet18 final_result eurosat_paths.py`。推理按快速运行检查输出行数和类别。训练链路可在具备原样清单的 CUDA 环境用新运行名做 train/val 最小验证。

## 数据来源

EuroSAT RGB 由 Patrick Helber、Benjamin Bischke、Andreas Dengel 和 Damian Borth 发布，源自 Copernicus Sentinel-2 影像。[EuroSAT 官方项目与许可](https://github.com/phelber/EuroSAT) · [原始论文](https://arxiv.org/abs/1709.00029)
