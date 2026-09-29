# EuroSAT 最终结果：ResNet18 验证集指标

当前状态：**ResNet18 最终微调权重已锁定，固定验证集指标已由用户审核并授权提交。** 最终 test 尚未运行；本报告只记录 val 指标。

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
| 本次评价 | val 4050 张；确定性 Resize/Normalize，`model.eval()`、FP32，无随机增强 |

权重保留在原 `runs/` 目录，没有复制或改写。`compute_resnet18_validation.py` 在推理前核对上述三个哈希，并检查 checkpoint 的 `epoch=best_epoch=17`。本次新生成的全部 4050 张 val 预测，与先前 `runs/resnet18/analysis/prediction_patterns_20260927_01/predictions_current_best_best.csv` 逐图核对：真实类别、预测类别、最高类别分数均完全相同。

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

![val 混淆矩阵：数量及行百分比](resnet18/validation/confusion_matrix.png)

![val 各类别 F1 与 OvR ROC AUC](resnet18/validation/per_class_f1_auc.png)

## 本地详细文件

`final_result/resnet18/validation/` 保存本次原始评价产物：

- `model_lock.json`：权重、配置、划分的路径与哈希。
- `metrics.json`：总体、macro/micro/weighted、逐类与混淆矩阵全部数值。
- `per_class_metrics.csv`：逐类 precision、recall、specificity、F1、ROC AUC、average precision。
- `confusion_matrix.csv`、`confusion_matrix.png`：误判计数表及数量/行百分比图。
- `per_class_f1_auc.png`：逐类 F1 与 AUC 图，缩放纵轴并保留精确数值于 CSV。
- `predictions.csv`：4050 张 val 图片路径、真值、预测、最高分及十类概率。

按仓库规则，逐样本预测、运行记录、图片及权重不进入 Git；此目录内的生成产物已由 `final_result/.gitignore` 排除。本文与计算脚本用于记录本次模型锁定与 val 指标。**最终 test 尚未运行。**
