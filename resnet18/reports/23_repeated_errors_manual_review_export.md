# 23：多次误判样本按真实类别提取，供人工查看

整理日期：2026-09-28

## 用户决策

> 把多次预测错误的样本提取出来，分成类别，放在项目根目录的 `error_samples/<type>/`。`<type>` 是正确类别；文件名体现“正确类别 → 错误预测类别”。之后由用户人工识别错误规律，并决定针对性数据增强。

本轮只做文件整理，不训练、不重新预测、不修改原图/标签/划分。没有读取test图片。

## 选取口径与来源

“多次”沿用报告20–21的**固定七组best checkpoint**对同一val划分的独立预测：同一张原图在至少两组中误判，按原图路径去重。来源文件为：

- `runs/resnet18/analysis/error_review_20260927_01/errors_all_experiments.csv`：七组共528条“实验 × 误判图片”记录。
- `runs/resnet18/analysis/prediction_patterns_20260927_01/seven_run_correctness.csv`：全体4050张val图片在七组中的误判次数与误判组别。

其中98张不同图片至少误判两次，逐张与七组记录核对后再复制。39张入选图片在当前最佳模型中预测正确；“多次误判”描述七组历史结果，不等于这些图片均被当前最佳模型误判。

| 七组中误判次数 | 图片数 |
| --- | ---: |
| 2 | 16 |
| 3 | 12 |
| 4 | 12 |
| 5 | 8 |
| 6 | 17 |
| 7 | 33 |
| **合计** | **98** |

## 目录与命名

输出目录为项目根目录 `error_samples/`。每张图片放入其清单正确类别对应的一级子目录，各类图片数：

| 正确类别 | 图片数 |
| --- | ---: |
| AnnualCrop | 13 |
| Forest | 5 |
| HerbaceousVegetation | 17 |
| Highway | 16 |
| Industrial | 2 |
| Pasture | 9 |
| PermanentCrop | 20 |
| Residential | 3 |
| River | 11 |
| SeaLake | 2 |
| **合计** | **98** |

用户指定的ASCII写法 `<type>-><wrong_type>` 含 `>`，Windows文件名不允许该字符。因此实际使用有效的Unicode箭头，并保留原文件名防止同一类别对的多张图片互相覆盖：

`error_samples/<正确类别>/<正确类别>→<误判类别>__<原文件名>.jpg`

例如 `error_samples/PermanentCrop/PermanentCrop→AnnualCrop__PermanentCrop_773.jpg`。图片是原JPEG逐字节副本，没有重新压缩。

8张图片在七组里曾被判为不止一种错误类别。每张仍只复制一次；文件名中的错误类别取**错误预测里出现最多的类别**。如并列，先取当前最佳模型的误判类别（若它在并列类别中），否则按固定类别顺序取第一类。本轮有2张出现最多次数并列。每组实际错误类别、置信度和次数均写在 `error_samples/index.csv`，文件名不能代替这份完整记录。

## 产物与核对

- `error_samples/<类别>/`：10个按正确类别划分的图片目录，共98张JPEG。
- `error_samples/index.csv`：每张副本/原图绝对路径、七组错误次数与具体预测、文件名使用的错误类别、原图SHA256。
- `error_samples/summary.json`、`README.md`：计数、筛选和命名口径。
- `resnet18/extract_repeated_errors.py`：可复查的提取过程，若目标目录已经存在会拒绝覆盖。

核对记录：98张图均在固定val清单中；七组错误次数与错误类别列表逐一一致；副本的SHA256与对应原图相同。原始 `data/` 没有写入。本轮新增 `/error_samples/` Git忽略规则，图片和索引不进入Git；本报告与提取脚本留在源码区。

错误样本的图像语义规律与后续增强方向由用户人工查看后决定。
