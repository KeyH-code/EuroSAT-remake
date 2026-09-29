# 24：多次误判样本的类别方向、双向混淆与类别含义

整理日期：2026-09-28

## 用户分析与请求

> 用户在 `error_samples/` 中先统计了 AnnualCrop、Forest、HerbaceousVegetation、Highway 四类，要求补完十类；找出两个类别互相识别错误的情况、每类最常被识别成什么，并简要介绍正确与错误类别可能的共同点。图片中的具体规律由用户人工判断。

用户前四类的数量与当前文件名统计一致。用户文本中的 `HerbaceousVegetable`、`HernbaceousVegetation` 在仓库中的正式类别名为 `HerbaceousVegetation`；本报告统一按正式类名记录。

## 计数口径

仅统计 `error_samples/index.csv` 的98张不同val原图。每张图片根据其**文件名选定的一种主要错误类别**记1次；它在七个checkpoint中可能有其他错误类别，详见索引的 `wrong_class_counts`、`all_wrong_predictions`。有8张出现过不同错误类别、2张主要错误类别次数并列。因此下面既不是七组共528次误判事件的混淆矩阵，也不是当前最佳模型在4050张val上的混淆矩阵。

98张均在至少两组固定best checkpoint中被误判；它们是一个预先筛出的困难样本集合。各类数量不同，表中比例若出现，只表示在本类**已选入图片**中的构成，不能当成该类完整val错误率。

## 十类方向统计

| 真实类别 | 入选图片 | 错误类别及图片数 | 本类最多的错误类别 |
| --- | ---: | --- | --- |
| AnnualCrop | 13 | Pasture 3；PermanentCrop 6；River 4 | PermanentCrop 6/13 |
| Forest | 5 | HerbaceousVegetation 1；Pasture 2；River 1；SeaLake 1 | Pasture 2/5 |
| HerbaceousVegetation | 17 | Forest 3；Highway 2；Pasture 2；PermanentCrop 7；Residential 1；River 1；SeaLake 1 | PermanentCrop 7/17 |
| Highway | 16 | AnnualCrop 6；HerbaceousVegetation 2；PermanentCrop 1；Residential 2；River 5 | AnnualCrop 6/16 |
| Industrial | 2 | PermanentCrop 1；Residential 1 | PermanentCrop、Residential 并列各1/2 |
| Pasture | 9 | Forest 2；HerbaceousVegetation 6；River 1 | HerbaceousVegetation 6/9 |
| PermanentCrop | 20 | AnnualCrop 13；HerbaceousVegetation 6；Highway 1 | AnnualCrop 13/20 |
| Residential | 3 | Industrial 1；Highway 2 | Highway 2/3 |
| River | 11 | AnnualCrop 1；Highway 10 | Highway 10/11 |
| SeaLake | 2 | AnnualCrop 1；Pasture 1 | AnnualCrop、Pasture 并列各1/2 |
| **合计** | **98** | **33个非零方向** | — |

![98张图片的主要误判方向矩阵](../../runs/resnet18/analysis/repeated_error_category_stats_20260928_01/dominant_error_matrix.png)

左图是张数；右图的百分比以每一行入选图片数为分母。例如River行10/11为90.9%，只描述这11张重复误判图片。

## 双向误判

表内 `A→B / B→A` 是**两批真实类别不同的图片**的数量，并非同一张图片在两个方向上翻转。共11组双向关系，按两个方向合计由多到少：

| 类别对 | A→B | B→A | 合计 |
| --- | ---: | ---: | ---: |
| AnnualCrop ↔ PermanentCrop | 6 | 13 | 19 |
| Highway ↔ River | 5 | 10 | 15 |
| HerbaceousVegetation ↔ PermanentCrop | 7 | 6 | 13 |
| HerbaceousVegetation ↔ Pasture | 2 | 6 | 8 |
| AnnualCrop ↔ River | 4 | 1 | 5 |
| Forest ↔ Pasture | 2 | 2 | 4 |
| HerbaceousVegetation ↔ Highway | 2 | 2 | 4 |
| Highway ↔ Residential | 2 | 2 | 4 |
| Forest ↔ HerbaceousVegetation | 1 | 3 | 4 |
| Highway ↔ PermanentCrop | 1 | 1 | 2 |
| Industrial ↔ Residential | 1 | 1 | 2 |

方向明显不对称的例子包括 PermanentCrop→AnnualCrop 13 对 AnnualCrop→PermanentCrop 6，以及 River→Highway 10 对 Highway→River 5。不能把两个方向相加后当成某个类别的错误率。

## 类别含义与可能重合的视觉线索

下述是**类别定义和单幅RGB遥感图可能出现的视觉重合**，不是本轮98张图片的人工判读，也不是EuroSAT对每张图的标注细则。EuroSAT官方将该数据介绍为Sentinel-2土地利用/覆盖分类；欧盟和ESA的分类定义用于解释词义，不与EuroSAT的十类体系逐项等同。[EuroSAT项目说明](https://github.com/phelber/EuroSAT)、[ESA WorldCover类别定义](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)。

先按名称直译：AnnualCrop 是一年生作物地，PermanentCrop 是多年生作物地，Pasture 是牧场/草场，HerbaceousVegetation 是草本植被，Forest 是树林；Highway 是道路，Industrial 是工业区，Residential 是居住区；River 是河流，SeaLake 是海或湖。前五类主要与植被、农业有关；中间三类是人工建设场景；后两类是水体。这只是便于看图的类别词义速览。

| 类别关系 | 可能的共同点与关键差别 |
| --- | --- |
| AnnualCrop ↔ PermanentCrop（19张） | 都是人为经营的农地，可能有田块边界、种植行列及植被/裸土交替。AnnualCrop偏向周期性播种、收获；PermanentCrop可多年留地，例如果园、葡萄园。使用年限与轮作本身通常不能仅由一张RGB图直接读出；同季节的地块在颜色或行列上可能接近。这是解释两个名字的含义，不断言这19张都呈现这些特征。[ESA WorldCover年度农地说明](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)、[欧盟常年作物释义](https://knowledge4policy.ec.europa.eu/glossary-item/permanent-crops_en)。 |
| HerbaceousVegetation ↔ Pasture（8张）；Forest ↔ HerbaceousVegetation（4张）；Forest ↔ Pasture（4张） | 三者都可能包含绿色植被。HerbaceousVegetation指草本覆盖，Pasture强调牧草地/放牧用途；Forest强调树冠。草本与牧场的用途差别、树下草本或林地边缘的混合，在小块RGB视图中可能难以区分；但是否真有林缘或混合覆盖，需要逐图核查。[ESA树木/草地释义](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)、[Eurostat草地定义](https://ec.europa.eu/eurostat/cache/metadata/EN/apri_lpr_esms.htm)。 |
| HerbaceousVegetation ↔ PermanentCrop（13张） | 都可能显示植物覆盖，且草本作物可出现在多年生种植地之间。差异在于草本地表与多年生木本作物/行列经营；单张RGB画面的绿色比例或纹理不一定表达这种经营差别。[ESA植被类别说明](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)、[欧盟常年作物释义](https://knowledge4policy.ec.europa.eu/glossary-item/permanent-crops_en)。 |
| Highway ↔ River（15张） | 道路和河流都可能在俯视小图中形成穿过画面的细长、连续带状结构。定义上一个是人工道路，一个是水体；仅共享可能的几何形状，并不共享地物性质。应由用户在图片中观察表面、边缘和周边地物是否真的呈现这种情况。[ESA建成区/水体释义](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)。 |
| Highway ↔ Residential（4张）；Industrial ↔ Residential（2张） | 三者都与人工建成环境有关：道路常与住宅或工业建筑共处，工业和住宅都包含建筑及硬化地表。地块内道路与建筑占比可成为人工查看时的区分线索，但本轮未量测这批图片的占比。[ESA建成区释义](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)。 |
| Highway ↔ AnnualCrop（单向6张）、AnnualCrop ↔ River（5张）、SeaLake→AnnualCrop/Pasture（各1张）、Industrial→PermanentCrop（1张） | 这些方向没有一致的土地利用含义可直接解释。田块边界、线性地物、大片相对均一的颜色/纹理或混合场景都只能作为**待看图的假设**；不能从类别名字和计数确定实际误判原因。 |

特别地，本集合里没有River与SeaLake的双向误判记录；虽都属于水体类别，也不能把语义接近直接当成本模型实测混淆。ESA的一套地表覆盖分类甚至将河流、湖泊等共同归于水体大类，进一步说明这里需要区分“类别语义相近”和“当前模型发生误判”。[ESA WorldCover类别定义](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)。

## 产物与边界

派生统计位于 `runs/resnet18/analysis/repeated_error_category_stats_20260928_01/`：

- `category_summary.csv`：每类全部错误方向与最多错误类别；
- `direction_counts.csv`：33个非零方向的张数、类内入选图片比例；
- `reciprocal_pairs.csv`：11组双向误判的方向计数；
- `dominant_error_matrix.png`：计数和行内比例矩阵；
- `statistics.json`、`analyze.py`、`render.py`：统计、来源与绘图过程。

本轮只读 `error_samples/index.csv` 与对应副本以确认文件名；没有打开图片内容、运行模型、改动样本或读取test。图片内容和相似原因仍待用户人工查看。运行数据由Git忽略，本报告记录本次分析轨迹。
