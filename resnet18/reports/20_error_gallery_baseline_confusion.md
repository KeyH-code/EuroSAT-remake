# 20：各实验误判图册与两份 ResNet18 基线混淆矩阵

整理日期：2026-09-27

## 用户请求

> 把当前统计的各个实验的误判例子给出来，给出基线的混淆矩阵，我得看看这个准确率是不是虚高。

本轮整理报告19的六组实验的全部误判图片，并补充 `baseline/` 当前最佳增强组。分别明确展示首次突破98%的未增强参考基线和当前最佳增强基线。没有启动训练，没有重新选 checkpoint。

## 图册与数据范围

交互图册：`runs/resnet18/analysis/error_review_20260927_01/index.html`。

- 可选择实验、真实类别、预测类别、文件名，也可筛选原六组共同误判的图片。
- 每张图片显示原文件名、真实类别、预测类别、最大 softmax 置信度、原六组误判次数及原始绝对路径。
- 点击图片可放大查看；原图片为64×64 RGB，按原像素显示。图册嵌入原JPEG字节，不修改数据文件。
- 每组同时导出全部错误的CSV、路径清单，以及每页最多24张的PNG图版。PNG按置信度降序排列，不只展示挑选的少数例子。
- 来源均为各组 best checkpoint 对既有 val 的独立复评。没有读取 test 图片，数据划分未变。

| 实验 | Best epoch | 正确/4050 | 准确率 | 全部误判数 | PNG页数 | 子目录 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 未增强 batch64 参考基线 | 12 | 3972 | 98.0741% | 78 | 4 | `reference/` |
| label smoothing 0.05 | 7 | 3982 | 98.3210% | 68 | 3 | `ls005/` |
| label smoothing 0.1 | 11 | 3980 | 98.2716% | 70 | 3 | `ls01/` |
| weight decay 0.005 | 14 | 3974 | 98.1235% | 76 | 4 | `wd0005/` |
| batch48 | 16 | 3972 | 98.0741% | 78 | 4 | `batch48/` |
| 固定lr=1e-4、batch128 | 8 | 3960 | 97.7778% | 90 | 4 | `fixed1e4/` |
| 当前最佳：翻转/旋转/裁剪/平移 | 17 | 3982 | 98.3210% | 68 | 3 | `current_best/` |

原六组共460条“实验×误判图片”记录，加上当前最佳增强组共528条。跨七组去重后137张图片。相同图片在不同组中的误判分别保留，不将528条当作528张不同图片。

## 未增强 batch64 参考基线

运行：`resnet18_layer4_lr0p0001_warmup_cosine_bs64_20ep_01`，best epoch12；配置为config32。此次错误样本分析和label smoothing对照使用这份参考基线。

矩阵行是真实类别，列是预测类别。总计4050张，对角线合计3972张，非对角线合计78张：3972/4050=0.9807407407407407。各类召回率的算术平均为0.9797333333333332，macro F1为0.9801873460915822。

![未增强参考基线的原始计数矩阵](../../runs/resnet18/analysis/error_review_20260927_01/reference/confusion_counts.png)

![未增强参考基线的错误计数矩阵](../../runs/resnet18/analysis/error_review_20260927_01/reference/confusion_errors.png)

错误图隐藏正确对角线，颜色只反映非对角线错误数。各真实类别的百分比矩阵另存为 `reference/confusion_percent.png`。

| 类别 | Val数量 | 正确 | 错误 | 逐类正确率（Recall） |
| --- | ---: | ---: | ---: | ---: |
| AnnualCrop | 450 | 445 | 5 | 98.8889% |
| Forest | 450 | 446 | 4 | 99.1111% |
| HerbaceousVegetation | 450 | 436 | 14 | 96.8889% |
| Highway | 375 | 363 | 12 | 96.8000% |
| Industrial | 375 | 373 | 2 | 99.4667% |
| Pasture | 300 | 290 | 10 | 96.6667% |
| PermanentCrop | 375 | 357 | 18 | 95.2000% |
| Residential | 450 | 447 | 3 | 99.3333% |
| River | 375 | 366 | 9 | 97.6000% |
| SeaLake | 450 | 449 | 1 | 99.7778% |
| 合计 | 4050 | 3972 | 78 | 98.0741% |

该矩阵中数目较多的误判方向：PermanentCrop→AnnualCrop 14张，River→Highway 8张，Pasture→HerbaceousVegetation 6张，Highway→AnnualCrop 5张，Highway→River 5张，HerbaceousVegetation→PermanentCrop 5张。

参考基线错误图片第1页：

![参考基线错误图片第1页](../../runs/resnet18/analysis/error_review_20260927_01/reference/errors_page_01.png)

## baseline/ 当前最佳增强基线

运行：`resnet18_layer4_lr0p0001_warmup_cosine_bs64_flip_rotate_translate_crop_20ep_01`，best epoch17；配置为config41，与 `baseline/resnet18_best.json` 除 experiment_id 外一致。

![当前最佳增强基线的原始计数矩阵](../../runs/resnet18/analysis/error_review_20260927_01/current_best/confusion_counts.png)

该矩阵总数4050、对角线3982、错误68，准确率0.9832098765432099；各类召回率算术平均0.9826，macro F1为0.9825700829334879。错误矩阵和按行百分比矩阵分别为 `current_best/confusion_errors.png`、`current_best/confusion_percent.png`。

## 核对记录与指标边界

七组都完成以下数据核对：

- 训练 best epoch 混淆矩阵元素合计为4050。
- 对角线合计除以4050与独立复评 summary 的 val accuracy 一致。
- 从全部 error_samples.csv 按真实/预测类别重新计数，所得矩阵与训练矩阵的非对角线逐格一致。
- 非对角线合计与全部错误图片数一致。
- 图册引用的每一张原图片存在于 `data/EuroSAT_RGB/2750/`，尺寸均为64×64。

这些核对确认当前记录的准确率与正确/错误计数一致。成绩来源仍是此前反复用于选模和调参的同一验证集；本轮没有测量独立test准确率，也没有进行数据泄漏或空间近邻审计。本报告记录图像和计数，供用户核查，不据此判定泛化成绩是否虚高。

## 产物

所有运行产物位于 `runs/resnet18/analysis/error_review_20260927_01/`：

- `index.html`：自包含误判图册，两份基线的三种混淆矩阵及逐类指标。
- `errors_all_experiments.csv`：七组全部528条错误记录。
- `experiment_summary.json`：运行来源、best epoch、正确/错误计数、整体及宏平均指标。
- 每组子目录的 `errors.csv`、`error_paths.txt`、`errors_page_*.png`：全部误判数据和图版。
- 每组子目录的 `confusion_matrix.csv`、`confusion_matrix.json`、`per_class_metrics.csv`：原始矩阵和逐类计数。
- `reference/`、`current_best/` 中的 `confusion_counts.png`、`confusion_errors.png`、`confusion_percent.png`：两份基线的三种矩阵图。
- `build_review.py`、`metadata.json`：生成过程及本轮核对记录。

运行产物由Git忽略；报告保留请求、来源、展示口径和汇总。
