# EuroSAT 最终结果：ResNet18 验证集与测试集指标

当前状态（2026-09-29）：**ResNet18 最终微调权重已锁定；固定 val 已审核，最终 test 已执行一次并记录。** test 未用于选择权重或调整参数。后续实验只能继续依据 train/val 决策，不能反过来用本次 test 成绩筛选模型。

## 锁定的模型与数据

| 项目 | 值 |
| --- | --- |
| 模型 | ImageNet 预训练 ResNet18，微调 `layer4+fc` |
| 实验 | config41；seed 20260920；20 epoch 训练中的 epoch 17 `best.pt` |
| 训练配置 | `resnet18/configs/config41_layer4_lr0p0001_warmup_cosine_bs64_flip_rotate_translate_crop_20ep.json` |
| 权重 | `runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_flip_rotate_translate_crop_20ep_01/checkpoints/best.pt` |
| 权重 SHA-256 | `8039bbce2884fa5058d939bf6973c7866492e50b714504442aba87e6af56d025` |
| 配置 SHA-256 | `f6c4da80f50b71f7af303672d06f95fe856617bc4c1233eee80abf6af29da848` |
| 固定划分清单 SHA-256 | `66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf` |
| 评价口径 | val/test 各4050张；确定性 Resize/Normalize，`model.eval()`、FP32，无随机增强 |

权重保留在原 `runs/` 目录，没有复制或改写。`compute_resnet18_validation.py` 在推理前核对上述三个哈希，并检查 checkpoint 的 `epoch=best_epoch=17`。本次新生成的全部 4050 张 val 预测，与先前 `runs/resnet18/analysis/prediction_patterns_20260927_01/predictions_current_best_best.csv` 逐图核对：真实类别、预测类别、最高类别分数均完全相同。

完整增强与无增强、无 label smoothing 的纯基线仅在**固定 val**上做了三个同 seed 配对，best 正确数分别多 10、6、4 张；完整表见[报告 30](../resnet18/reports/30_pure_baseline_two_seed_20ep.md)。三个完整增强运行的 best val 正确数为 3982、3981、3973，因此在读取 test 前选定 seed 20260920 的 config41 权重。纯基线未在本次 test 上评价，不能报告 test 上的相对提升。

## val 总体指标

| 指标 | 数值 |
| --- | ---: |
| Accuracy / micro F1 | 0.983210（3982/4050；误判68） |
| Balanced accuracy / macro recall | 0.982600 |
| Macro precision / recall / F1 | 0.982563 / 0.982600 / 0.982570 |
| Weighted precision / recall / F1 | 0.983240 / 0.983210 / 0.983213 |
| 多分类 ROC AUC：OvR macro / weighted / micro | 0.999647 / 0.999659 / 0.999727 |
| 多分类 average precision：OvR macro / weighted / micro | 0.997308 / 0.997425 / 0.997926 |
| 普通交叉熵 | 0.061789 |
| Top-2 / top-3 accuracy | 0.998272 / 0.999259 |
| 多分类 Brier 分数 / 15 等宽箱 ECE | 0.026717 / 0.006728 |

这里的 ROC AUC 和 average precision 是每类对其他九类的 **one-vs-rest（OvR）概率排序**指标；macro 为十类等权平均，weighted 按各类 val 样本数加权，micro 将十类二元标签与预测分数展平计算。ROC AUC 不等同于按最高分决定类别后的 accuracy。

## val 逐类指标

| 真实类别 | 数量 | Precision | Recall | F1 | ROC AUC OvR | AP OvR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AnnualCrop | 450 | 0.969163 | 0.977778 | 0.973451 | 0.999579 | 0.996644 |
| Forest | 450 | 0.993333 | 0.993333 | 0.993333 | 0.999959 | 0.999698 |
| HerbaceousVegetation | 450 | 0.984199 | 0.968889 | 0.976484 | 0.999419 | 0.996147 |
| Highway | 375 | 0.970976 | 0.981333 | 0.976127 | 0.999366 | 0.994736 |
| Industrial | 375 | 0.997333 | 0.997333 | 0.997333 | 0.999995 | 0.999950 |
| Pasture | 300 | 0.973244 | 0.970000 | 0.971619 | 0.999644 | 0.996609 |
| PermanentCrop | 375 | 0.965426 | 0.968000 | 0.966711 | 0.998788 | 0.991843 |
| Residential | 450 | 0.993348 | 0.995556 | 0.994451 | 0.999953 | 0.999653 |
| River | 375 | 0.978610 | 0.976000 | 0.977303 | 0.999766 | 0.997810 |
| SeaLake | 450 | 1.000000 | 0.997778 | 0.998888 | 0.999999 | 0.999995 |

完整精度、specificity 和各类 support 见 `resnet18/validation/per_class_metrics.csv`。

## val 混淆矩阵

行是真实类别，列是预测类别。类别列顺序：`AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake`。对角线之和为3982，全部格子之和为4050。

| 真实类别 | AC | Fo | HV | Hw | In | Pa | PC | Re | Ri | SL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AnnualCrop | 440 | 0 | 0 | 0 | 0 | 3 | 5 | 0 | 2 | 0 |
| Forest | 0 | 447 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| HerbaceousVegetation | 0 | 2 | 436 | 2 | 0 | 2 | 6 | 1 | 1 | 0 |
| Highway | 2 | 0 | 0 | 368 | 0 | 0 | 1 | 1 | 3 | 0 |
| Industrial | 0 | 0 | 0 | 0 | 374 | 0 | 0 | 1 | 0 | 0 |
| Pasture | 1 | 1 | 5 | 0 | 0 | 291 | 1 | 0 | 1 | 0 |
| PermanentCrop | 10 | 0 | 1 | 0 | 0 | 1 | 363 | 0 | 0 | 0 |
| Residential | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 448 | 0 | 0 |
| River | 1 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 366 | 0 |
| SeaLake | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 449 |

本地图像（不进入 Git）：`resnet18/validation/confusion_matrix.png`，显示数量及行百分比。

本地图像（不进入 Git）：`resnet18/validation/per_class_f1_auc.png`，显示逐类 F1 与 OvR ROC AUC。

## 最终 test 指标

使用同一 `best.pt`、配置与划分清单，在原样 test 4050 张图片上做**一次**推理；不重新选模、不训练、不改阈值。推理前重新核对上述三份 SHA-256 和 checkpoint epoch 17；输出位于 `final_result/resnet18/test/`，与 val 逐样本路径交集为0。此目录新建后拒绝覆盖。

| 指标 | test 结果 |
| --- | ---: |
| Accuracy / micro F1 | 0.982963（3981/4050；误判69） |
| Balanced accuracy / macro recall | 0.982444 |
| Macro precision / recall / F1 | 0.982093 / 0.982444 / 0.982250 |
| Weighted precision / recall / F1 | 0.983022 / 0.982963 / 0.982974 |
| 多分类 ROC AUC：OvR macro / weighted / micro | 0.999668 / 0.999686 / 0.999743 |
| 多分类 average precision：OvR macro / weighted / micro | 0.997234 / 0.997390 / 0.997875 |
| 普通交叉熵 | 0.063983 |
| Top-2 / top-3 accuracy | 0.999012 / 0.999753 |
| 多分类 Brier 分数 / 15 等宽箱 ECE | 0.027847 / 0.008809 |

ROC AUC、average precision、macro/weighted/micro 的定义与上文 val 完全一致。

### test 逐类指标

| 真实类别 | 数量 | Precision | Recall | F1 | ROC AUC OvR | AP OvR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AnnualCrop | 450 | 0.993213 | 0.975556 | 0.984305 | 0.999666 | 0.997574 |
| Forest | 450 | 0.993348 | 0.995556 | 0.994451 | 0.999976 | 0.999803 |
| HerbaceousVegetation | 450 | 0.971111 | 0.971111 | 0.971111 | 0.999419 | 0.994634 |
| Highway | 375 | 0.981283 | 0.978667 | 0.979973 | 0.999495 | 0.996332 |
| Industrial | 375 | 0.989276 | 0.984000 | 0.986631 | 0.999882 | 0.998914 |
| Pasture | 300 | 0.963696 | 0.973333 | 0.968491 | 0.999323 | 0.993980 |
| PermanentCrop | 375 | 0.965699 | 0.976000 | 0.970822 | 0.999330 | 0.994513 |
| Residential | 450 | 0.986813 | 0.997778 | 0.992265 | 0.999957 | 0.999658 |
| River | 375 | 0.978723 | 0.981333 | 0.980027 | 0.999671 | 0.997256 |
| SeaLake | 450 | 0.997763 | 0.991111 | 0.994426 | 0.999962 | 0.999672 |

完整精度、specificity 和 support 见 `resnet18/test/per_class_metrics.csv`。

### test 混淆矩阵

行是真实类别，列是预测类别；列缩写与 val 表一致：AC、Fo、HV、Hw、In、Pa、PC、Re、Ri、SL。对角线之和3981，全部格子之和4050。

| 真实类别 | AC | Fo | HV | Hw | In | Pa | PC | Re | Ri | SL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AnnualCrop | 439 | 1 | 0 | 1 | 0 | 4 | 4 | 0 | 1 | 0 |
| Forest | 0 | 448 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| HerbaceousVegetation | 0 | 0 | 437 | 1 | 0 | 4 | 7 | 1 | 0 | 0 |
| Highway | 0 | 0 | 1 | 367 | 2 | 0 | 2 | 0 | 3 | 0 |
| Industrial | 0 | 0 | 0 | 1 | 369 | 0 | 0 | 5 | 0 | 0 |
| Pasture | 0 | 1 | 5 | 0 | 0 | 292 | 0 | 0 | 2 | 0 |
| PermanentCrop | 2 | 0 | 5 | 0 | 1 | 1 | 366 | 0 | 0 | 0 |
| Residential | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 449 | 0 | 0 |
| River | 1 | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 368 | 1 |
| SeaLake | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 2 | 446 |

本地图像（不进入 Git）：`resnet18/test/confusion_matrix.png`，显示数量及行百分比。

本地图像（不进入 Git）：`resnet18/test/per_class_f1_auc.png`，显示逐类 F1 与 OvR ROC AUC。

### 同一权重的 val/test 对照

| 指标 | val | test | test − val |
| --- | ---: | ---: | ---: |
| 正确/4050 | 3982 | 3981 | −1 |
| Accuracy | 0.983210 | 0.982963 | −0.000247 |
| Macro F1 | 0.982570 | 0.982250 | −0.000320 |
| Macro ROC AUC OvR | 0.999647 | 0.999668 | +0.000021 |
| 普通交叉熵 | 0.061789 | 0.063983 | +0.002194 |

这里的差值是两个不同固定划分上的观测，不涉及 test 上的模型选择或参数调整。

本结果适用于当前 EuroSAT RGB 原样划分。虽然已核对 train/val 无路径、文件字节或 RGB 像素的精确重复，但缺少地理元数据，尚不能确认跨划分的空间独立性，见[数据审计](../resnet18/reports/22_fixed_transforms_clean_eval_similarity_audit.md)。[外部 Sentinel-2 演示](../resnet18/reports/25_field_patrol_live_imagery_demo.md)没有人工真值，不作为外部准确率证据。

## 本地详细文件

`final_result/resnet18/validation/` 保存本次原始评价产物：

- `model_lock.json`：权重、配置、划分的路径与哈希。
- `metrics.json`：总体、macro/micro/weighted、逐类与混淆矩阵全部数值。
- `per_class_metrics.csv`：逐类 precision、recall、specificity、F1、ROC AUC、average precision。
- `confusion_matrix.csv`、`confusion_matrix.png`：误判计数表及数量/行百分比图。
- `per_class_f1_auc.png`：逐类 F1 与 AUC 图，缩放纵轴并保留精确数值于 CSV。
- `predictions.csv`：4050 张 val 图片路径、真值、预测、最高分及十类概率。

`final_result/resnet18/test/` 保存同名文件，内容对应唯一一次最终 test；`model_lock.json` 标记 `split=test` 和 `test_used=true`。本次使用 `compute_resnet18_test.py` 调用相同的指标计算逻辑。生成的逐样本预测、CSV、JSON、图像及权重按仓库规则不进入 Git，两个本地结果目录均由 `final_result/.gitignore` 排除；本文记录完整汇总。
