# 19：数据增强前的重复错误与 label smoothing 逐样本对照

整理日期：2026-09-27

## 用户分析与决策

> 当前 layer4 和优化器已经积累多个基线，接下来分析出错样本，为针对性数据增强提供依据。数据源取数据增强前、准确率和 loss 排名前几的尝试。先提取反复出现的错误图片路径，再比较 label smoothing 后预测类别、置信度与高分样本的变化。

本轮为既有 checkpoint 的验证集样本分析；未启动训练，未制定新的增强参数。

## 数据范围与筛选口径

使用固定 val 4050 张图片。manifest SHA-256 为 `66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf`。未读取 test 图片、未修改划分、未读取历史冻结归档。

候选为迁移后 runs/ 中 layer4+fc、没有 train 数据增强的有效实验；排除报告10中用户判定不可用的固定 lr=0.001。分别按 best checkpoint 的 val accuracy 降序和普通 val CE loss 升序排名，取准确率前5组与 loss 前3组的并集，共6组。准确率并列按 loss 排序。loss 指标是最高准确率 checkpoint 的 loss，并非所有 epoch 中最低的 loss；后者对应的中间权重通常没有保留。候选完整排名见 `candidate_ranking.csv`。

| 实验 | Best epoch | Val acc | 普通 val CE loss | 误分 | Acc排名 | Loss排名 |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18_layer4_lr0p0001_warmup_cosine_bs64_label_smoothing0p05_20ep_01 | 7 | 0.983210 | 0.117645 | 68 | 1 | 14 |
| resnet18_layer4_lr0p0001_warmup_cosine_bs64_label_smoothing0p1_20ep_01 | 11 | 0.982716 | 0.179311 | 70 | 2 | 15 |
| resnet18_layer4_lr0p0001_warmup_cosine_bs64_wd0p005_20ep_01 | 14 | 0.981235 | 0.087593 | 76 | 3 | 3 |
| resnet18_layer4_lr0p0001_warmup_cosine_bs64_20ep_01 | 12 | 0.980741 | 0.087566 | 78 | 4 | 2 |
| resnet18_layer4_lr0p0001_warmup_cosine_bs48_20ep_01 | 16 | 0.980741 | 0.094911 | 78 | 5 | 7 |
| resnet18_layer4_lr0p0001_10ep_01 | 8 | 0.977778 | 0.077667 | 90 | 10 | 1 |

重复错误直接合并既有独立复评的 error_samples.csv，每个 run 只计一次；同一 filepath 为同一个样本。六组中四组没有 smoothing、两组有 smoothing。`cohort.csv` 保存复评目录。这里的多组实验共享同一验证集和 seed，不是独立随机重复试验。

## 重复错误与文件路径

六组错误图片的并集为 **128张**，至少两组误判 **95张**，六组全部误判 **40张**。只看四个无 smoothing 实验，四组全部误判48张。三个固定 batch64、仅改变 smoothing 的 best 模型共同误判48张；这两份48张清单定义不同，不应视为同一份。

| 误判组数/6 | 独立图片数 |
| --- | --- |
| 1 | 33 |
| 2 | 18 |
| 3 | 11 |
| 4 | 12 |
| 5 | 14 |
| 6 | 40 |

- [六组全部误判的40张路径](../../runs/resnet18/analysis/sample_error_analysis_20260927_01/all_six_error_paths.txt)
- [至少两组误判的95张路径](../../runs/resnet18/analysis/sample_error_analysis_20260927_01/repeated_error_paths.txt)
- [95张重复错误汇总](../../runs/resnet18/analysis/sample_error_analysis_20260927_01/repeated_errors.csv)：真值、误判组数、误判类别频次、无 smoothing 子集频次。
- [各实验逐样本错误详情](../../runs/resnet18/analysis/sample_error_analysis_20260927_01/repeated_errors_details.csv)：每一张图片在各组中的误判类别与置信度。
- [四个无 smoothing 实验全部误判的48张路径](../../runs/resnet18/analysis/sample_error_analysis_20260927_01/ordinary_all_four_error_paths.txt)

六组共同错误的真实类别分布：PermanentCrop 10、Pasture 6、River 6、AnnualCrop 5、Highway 5、HerbaceousVegetation 3、Forest 3、Residential 2。Industrial 和 SeaLake 没有进入六组共同错误集合。这是交集计数，不是各类总体错误率。

40张中有 **35张**在六组里始终被预测为同一个错误类别，另外5张的错误类别发生过变化。下表按独立图片计数，不把同一张在六组里的误判重复计为6张。

| 真实类别 → 六组一致误判类别 | 图片数 |
| --- | --- |
| PermanentCrop → AnnualCrop | 5 |
| River → Highway | 5 |
| Pasture → HerbaceousVegetation | 5 |
| PermanentCrop → HerbaceousVegetation | 3 |
| AnnualCrop → PermanentCrop | 3 |
| Highway → River | 2 |
| HerbaceousVegetation → Highway | 1 |
| Highway → PermanentCrop | 1 |
| Residential → Industrial | 1 |
| Residential → Highway | 1 |
| Forest → SeaLake | 1 |
| HerbaceousVegetation → Residential | 1 |
| Forest → HerbaceousVegetation | 1 |
| River → AnnualCrop | 1 |
| Highway → Residential | 1 |
| HerbaceousVegetation → Pasture | 1 |
| Highway → AnnualCrop | 1 |
| Pasture → River | 1 |

上述重复模式支持优先查看这些样本，但仅凭预测表无法确定原因是混合地物、标注问题、尺度、纹理或方向；本轮没有作这些视觉原因判断。

## Label smoothing：准确率与置信度

对照 config32 / config36 / config40：batch64、lr=1e-4、1 epoch warmup 后20 epoch cosine、weight decay=0.01、seed、输入预处理、trainable_scope 均一致，仅 experiment_id 和 label_smoothing 不同。分别重新加载 best.pt 和 last.pt，在 shuffle=False 的 val loader 中用 FP32 推理。每张图片保存10类概率、预测类别、最大 softmax 概率、真实类别概率、普通 CE、第二类别和概率间隔。重新推理的准确率、loss、best 错误图片集合与类别及置信度均与原记录核对一致（浮点误差容许1e-6）。

这里“预测分数”定义为最大 softmax 概率（confidence）；“准确率”是与真实标签比较得到的正确比例，两者不能直接等同。confidence=1.0 是当前浮点输出，不表示现实中绝对正确。

| 组别 | Best epoch | Acc | 普通 CE loss | 全体平均置信度 | 正确样本平均置信度 | 错误样本平均置信度 |
| --- | --- | --- | --- | --- | --- | --- |
| 无 smoothing | 12 | 0.980741 | 0.087566 | 0.992697 | 0.995551 | 0.847397 |
| smoothing 0.05 | 7 | 0.983210 | 0.117645 | 0.917994 | 0.923092 | 0.619467 |
| smoothing 0.1 | 11 | 0.982716 | 0.179311 | 0.863799 | 0.869511 | 0.538993 |

smoothing 后绝大多数图片的分数下降：0.05 对照4030/4050张下降，0.1对照4038/4050张下降。预测类别本身只改变40张/45张。说明最广泛的变化是置信度缩小，同时存在少量类别改判。

| 对照基线 | 错→对 | 对→错 | 仍错 | 仍对 | 净增正确数 | 预测类别变化 |
| --- | --- | --- | --- | --- | --- | --- |
| smoothing 0.05 | 24 | 14 | 54 | 3958 | 10 | 40 |
| smoothing 0.1 | 25 | 17 | 53 | 3955 | 8 | 45 |

0.05 净增10张正确，但同时新增14张错误；0.1净增8张正确，同时新增17张错误。不能把净增值理解为只修复了10张或8张。

## 高预测分数的样本有没有变化

采用预先明确的固定阈值0.90、0.95、0.99、0.999。下表括号中的错误数为该阈值集合内误判的图片数。

| 置信度阈值 | 无 smoothing：数量（错） | 0.05：数量（错） | 0.1：数量（错） |
| --- | --- | --- | --- |
| 0.9 | 3963（42） | 3287（3） | 2190（0） |
| 0.95 | 3933（38） | 2212（1） | 156（0） |
| 0.99 | 3851（22） | 28（0） | 1（0） |
| 0.999 | 3704（14） | 0（0） | 0（0） |

两组 smoothing 在90%、95%、99%阈值下的高分集合都完全包含于原基线对应集合，没有新进入的样本。以95%为例，基线3933张，0.05保留2212张、1721张退出；0.1保留156张、3777张退出。高分集合主要收缩，而不是换成一批原来低分的新样本。

0.05的95%以上唯一错误为 PermanentCrop_773.jpg；0.1没有90%以上的错误。0.1达到99%以上的唯一图片是 River_762.jpg，三组均预测正确。零错误只描述当前有限样本，不代表未来高分预测不会出错。

## 同一张图片的变化例子

| 文件名 | 真值 | 基线预测/分数 | 0.05预测/分数 | 0.1预测/分数 |
| --- | --- | --- | --- | --- |
| PermanentCrop_773.jpg | PermanentCrop | AnnualCrop / 1.000000 | AnnualCrop / 0.961159 | AnnualCrop / 0.863853 |
| River_2031.jpg | River | Highway / 0.999906 | Highway / 0.865159 | Highway / 0.746416 |
| Pasture_577.jpg | Pasture | HerbaceousVegetation / 0.999886 | HerbaceousVegetation / 0.820102 | HerbaceousVegetation / 0.790320 |
| PermanentCrop_982.jpg | PermanentCrop | AnnualCrop / 0.937965 | PermanentCrop / 0.523191 | PermanentCrop / 0.346109 |
| HerbaceousVegetation_1817.jpg | HerbaceousVegetation | Forest / 0.959355 | HerbaceousVegetation / 0.450002 | Forest / 0.364564 |
| River_762.jpg | River | River / 1.000000 | River / 0.994968 | River / 0.991799 |

PermanentCrop_773、River_2031、Pasture_577仍然误判，只是错误分数下降。PermanentCrop_982在两组 smoothing 中改判正确。HerbaceousVegetation_1817在0.05正确、0.1仍错，不能将更强 smoothing 理解为逐样本单调改善。

完整逐样本对照见 [label_smoothing_comparison_best.csv](../../runs/resnet18/analysis/sample_error_analysis_20260927_01/label_smoothing_comparison_best.csv)。修复/退步的具体路径见 `corrected_ls005_best.csv`、`regressed_ls005_best.csv`、`corrected_ls01_best.csv`、`regressed_ls01_best.csv`。

## 为什么错误分数降低而总体 val loss 增加

普通 CE 为 -log(真实类别概率)。smoothing 后错误样本的平均 CE 下降，但数量占绝大多数的正确样本 CE 上升；两者相加后，总体 CE 增加。这里使用普通验证 CE，各组计算口径一致。

| 组别 | 正确样本平均CE | 错误样本平均CE | 正确样本对全体平均CE贡献 | 错误样本对全体平均CE贡献 |
| --- | --- | --- | --- | --- |
| 无 smoothing | 0.005228 | 4.280462 | 0.005127 | 0.082439 |
| smoothing 0.05 | 0.085939 | 1.974278 | 0.084496 | 0.033148 |
| smoothing 0.1 | 0.148901 | 1.908333 | 0.146328 | 0.032984 |

置信度压低不等于校准一定改善。当前验证集的整体准确率约98%，但两组 smoothing 的平均分数分别约92%和86%，呈现整体偏保守。以15个等宽分箱的 ECE（分箱内平均置信度与实际准确率差的绝对值，按样本数加权）和多类 Brier（10类概率与one-hot真值的平方误差求和，再对图片平均）补充衡量：

| 组别 | ECE（15等宽箱） | Brier（10类求和） |
| --- | --- | --- |
| 无 smoothing | 0.012569 | 0.031706 |
| smoothing 0.05 | 0.065331 | 0.035828 |
| smoothing 0.1 | 0.118917 | 0.048817 |

这两项在当前验证集上都比基线增大。因此证据支持“高置信误判减少、整体分数下降”，不支持“所有概率质量都变好”。ECE依赖分箱和样本数量，作为本次描述统计使用。

## 同为第20轮的补充对照

三组 best epoch 分别为12/7/11，模型训练进度不同。补充使用三组 last.pt（均为epoch20），观察相同轮数下的输出：

| 组别 | Epoch | Acc | 普通CE loss | 全体平均分数 | 错误平均分数 | ≥95%数量（错） |
| --- | --- | --- | --- | --- | --- | --- |
| 无 smoothing | 20 | 0.980494 | 0.093948 | 0.993397 | 0.855035 | 3943（39） |
| smoothing 0.05 | 20 | 0.981235 | 0.124333 | 0.914728 | 0.586795 | 2150（0） |
| smoothing 0.1 | 20 | 0.980988 | 0.182250 | 0.862997 | 0.518355 | 153（0） |

| 第20轮对照基线 | 错→对 | 对→错 | 净增正确数 |
| --- | --- | --- | --- |
| smoothing 0.05 | 21 | 18 | 3 |
| smoothing 0.1 | 20 | 18 | 2 |

同为第20轮时，置信度下降与高置信误判减少的现象仍然存在，准确率净增缩小到3张/2张。最佳 checkpoint 的10张/8张差异包含各自选模时点的影响。三组都只有一个seed，本轮没有多seed重复训练，不能据此确定稳定收益。

## 图表与产物

![Confidence distribution and reliability](../../runs/resnet18/analysis/sample_error_analysis_20260927_01/confidence_comparison.png)

第一幅图保留全体confidence 0–1范围；第二幅为各组错误样本置信度的累计分布（各自错误集合不同）；第三幅为15等宽分箱的置信度/实际正确率关系，小分箱点可能样本少。配对判断以逐样本对照表为准。

所有分析数据位于 `runs/resnet18/analysis/sample_error_analysis_20260927_01/`：

- `analyze.py`、`finalize.py`：本轮可复查分析过程；analyze.py拒绝覆盖已完成分析。
- `candidate_ranking.csv`、`cohort.csv`：候选排名、入选实验和原复评来源。
- `repeated_error_paths.txt`、`all_six_error_paths.txt`、`ordinary_all_four_error_paths.txt`：原图片绝对路径清单，未复制或修改图片。
- `repeated_errors.csv`、`repeated_errors_details.csv`、`all_error_union.csv`：错误频次及各组预测。
- `predictions_{baseline,ls005,ls01}_{best,last}.csv`：6个checkpoint的全体4050张val预测及10类概率。
- `label_smoothing_comparison_{best,last}.csv`、`persistent_errors_across_ls_{best,last}.csv`：对齐的逐图片变化与共同错误。
- `corrected_*`、`regressed_*`、`left_high95_*`、`high{90,95,99}_*`：改判与高分样本集合；空集合也保存列头。
- `recurrence_summary.json`、`confidence_summary.json`、`paired_summary.json`、`metadata.json`：汇总、权重哈希、数据身份和运行信息。

分析产物和模型权重不进入Git；该报告保存请求、比较口径与汇总。
