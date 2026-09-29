# 26：Highway / River 颜色与形状线索的数据增强建议

整理日期：2026-09-28

## 用户分析与当前决定

> AnnualCrop 与 PermanentCrop 本身差距不大，暂不给出修改建议。道路与河流值得作为下一步方向：用户观察到河道一般更曲折、颜色偏深，希望尝试围绕这两种线索进行数据增强，并要求本轮给出建议。

本轮仅核对既有预测与训练代码，制定可对照的实验方案；没有写新训练配置、修改训练框架或启动训练。AnnualCrop/PermanentCrop 保持现状。

## 现有数字与口径

| 来源 | Highway→River | River→Highway | 两方向合计 |
| --- | ---: | ---: | ---: |
| `error_samples/` 中98张重复误判图片，按主要误判类别记 | 5 | 10 | 15 |
| 未增强参考best，完整4050张val | 5 | 8 | 13 |
| 当前最佳增强best，完整4050张val | 3 | 8 | 11 |

前一行按已筛出的98张困难样本记，不能充当当前最佳模型的完整混淆矩阵。当前最佳的完整val含Highway 375张、River 375张，两方向合计11张，是后续同split对照的起点。当前最佳干净train的对应方向为Highway→River 3张、River→Highway 11张，仅作已有数据记录，不等同于训练时在线指标。

当前最佳训练配置为layer4+fc、AdamW学习率1e-4、weight decay 0.01、batch64、20 epoch、1 epoch warmup加20 epoch余弦调度、seed20260920；已有水平/垂直翻转、90°整数旋转、80%–100%面积随机裁剪与最多10%的随机平移。后两项有可能裁掉河道或道路的较长连续部分，尚未逐图证明这是本轮误判的原因。

## 对用户观察的边界

“河道更曲折、颜色更深”可作为**待验证的视觉假设**。在真彩色卫星图中，水体通常较暗，但悬浮泥沙、浅水和太阳反光都可使它变亮；颜色不宜写成绝对判别规则。[NASA 真彩色卫星图解读](https://science.nasa.gov/earth/earth-observatory/how-to-interpret-a-satellite-image/)。本轮没有浏览这批错误图片，不能断言11张当前错误是否符合用户观察。

## 建议的两组独立实验

两组均从当前最佳20 epoch配置**新建实验并从头训练**。保持权重初始化方式、seed、学习率、优化器、调度器、batch、训练轮数及已有增强不变；仅改变下述一项处理。新增处理仅在train启用，val仍用原来的确定性Resize＋Normalize。两类分别采用**相同**处理，以免让网络学到“哪一类经过某种处理”的伪线索。

| 组别 | 仅对train的Highway与River新增/调整 | 想测量的问题 |
| --- | --- | --- |
| 颜色对照 | 在现有几何增强之后、Normalize之前，以概率0.5使用 `v2.ColorJitter(brightness=0.15, contrast=0.10, saturation=0.05, hue=0)`；亮度因子约0.85–1.15，既可能变亮也可能变暗 | 轻微改变颜色后，两类是否更少互判；训练不要只依赖“暗=河流” |
| 连续形状对照 | 只把这两类现有 `RandomResizedCrop` 的面积下限从0.80提到0.90，上限仍1.00；裁剪概率仍0.5，比例仍1:1，其余变换不动 | 保留较多连续河道/道路上下文时，两类是否更少互判 |

Torchvision的`ColorJitter`在给定上述浮点参数时从相应因子范围抽样；`RandomResizedCrop.scale`表示裁剪面积占原面积的比例。[Torchvision 0.29 ColorJitter](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ColorJitter.html)、[RandomResizedCrop](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomResizedCrop.html)。这些数值是保守起点，尚无本项目实测收益。

不建议第一轮直接用强弹性变形把道路拉弯，或把河流调暗而道路调亮：前者可能改写形状与边界，后者会强化而不是检验颜色捷径。现有翻转与90°旋转已经覆盖基本方向变化，裁剪面积调整更直接检验“看见足够长的线性地物”是否有影响。两组独立结果明确后，再决定是否需要组合，避免一次改动太多而无法归因。

实现上，当前`resnet18/src/transforms.py`只有单一train transform，`EuroSATDataset`调用transform时也没有传入类别。因此**按类别施加**上述处理需要扩展数据/变换接口和配置记录，不能只改现有JSON数值；新配置不能从旧run续训。实验必须写入新的`runs/`目录，不覆盖当前基线。

## 后续比较口径

每组记录完整val accuracy与普通CE；Highway、River各375张的召回率/CE；Highway→River和River→Highway的**两个方向分别计数**；与当前最佳的逐图片“错→对、对→错”及其他八类是否新增错误。此处方向基线为3张与8张。由于涉及的当前错误仅11张，变化1–2张可以作为观察结果，但单一seed不足以确定其稳定性；出现明确收益后再考虑另seed复跑。不用test调参。

本报告记录用户观察、既有数字及建议，未将建议当成已完成实验。
