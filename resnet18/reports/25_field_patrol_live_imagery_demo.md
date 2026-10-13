# 25：公开 Sentinel-2 影像的乡村土地覆盖巡查演示

整理日期：2026-09-28

## 用户请求与本轮决定

> 用户要求寻找几个类似乡村土地覆盖巡查的真实素材，并实际跑一遍流程供其查看。

本轮选择一景公开 Sentinel-2 L2A 影像，截取七张约 640 m × 640 m 的 64 × 64 RGB 图块，用既有 ResNet18 best checkpoint 运行批量预测，再整理成待人工复核的清单与本地页面。未训练、未改已有模型与数据划分、未读取 EuroSAT test，也未把这些外部图块作为已标注验证集。

## 素材来源与处理

- 提供方：Microsoft Planetary Computer 的 `sentinel-2-l2a` STAC 集合及 data API；影像为 Sentinel-2 L2A `visual` RGB 渲染。
- STAC item：`S2A_MSIL2A_20240922T024531_R132_T50SNB_20240922T132653`。
- 采集时间：2024-09-22 02:45:31 UTC；STAC 整景云量属性约 3.087%，不能代表每个图块的局部云量。
- 预览范围：117.18°–117.24°E、33.08°–33.14°N。从预览图中选定农田、河渠、村落及一个明显云遮挡的观察点。名字只是观察目标，不是人工真值标签。
- 每张图块的边界框、请求 URL、SHA256、采集时间和 STAC item 均保存在 `runs/resnet18/field_patrol_demo_20260928_01/source_metadata.json`。
- 图像是新的外部素材，与当前 EuroSAT RGB 数据的地区、年份和影像处理流程不同；不能用现有 val 指标代替其准确率。

## 运行流程

1. `prepare_materials.py` 请求 STAC 元数据、生成预览范围中的七张 64 × 64 PNG，并生成 `contact_sheet.png`。
2. 使用仓库的 `resnet18/predict.py`、`baseline/resnet18_best.json` 和既有 best checkpoint 批量预测，得到 `predictions.csv`。
3. `build_patrol_demo.py` 核对图片 SHA256 与预测行数，按演示规则排队：人工目视明显云遮挡的图块列为“影像质量待核”；最高类别分数低于 0.90 的列为“优先复核”；其余列为“常规抽查”。0.90 只是演示阈值，不是经校准的风险门槛。全部图块的人类类别结论仍为空。
4. 生成 `patrol_queue.csv`、`summary.json`、模型结果图版 `result_board.png` 和嵌入图片的 `index.html`。页面可填写复核状态、人工类别、备注，并导出当前复核 CSV；默认数据是本次实际运行结果。

## 原始模型输出与排队

| 图块 | 模型预测 | 最高类别分数 | 排队 |
| --- | --- | ---: | --- |
| 01_nw_fields | PermanentCrop | 0.999829 | 常规抽查 |
| 02_west_fields | PermanentCrop | 0.998026 | 常规抽查 |
| 03_canal | River | 0.999843 | 常规抽查 |
| 04_east_fields | River | 0.858262 | 优先复核 |
| 05_south_village | Residential | 0.891579 | 优先复核 |
| 06_east_village | PermanentCrop | 0.997097 | 常规抽查 |
| 07_cloud_example | Industrial | 0.744138 | 影像质量待核 |

七张的预测类别计数为 PermanentCrop 3、River 2、Residential 1、Industrial 1；排队计数为常规抽查 4、优先复核 2、影像质量待核 1。**这些不是正确类别计数或准确率**。第 07 张的云遮挡由人工目视发现；当前模型仍给出 Industrial，流程因此保留原始预测并另标记影像质量问题。

## 产物与核对

全部运行产物位于 `runs/resnet18/field_patrol_demo_20260928_01/`，是新目录，未覆盖既有实验。包含：七张输入图、总览图、联系图、结果图版、STAC 来源元数据、原始预测 CSV、巡查队列 CSV、摘要 JSON、可交互页面及两个复现脚本。

核对结果：预测脚本报告输入 7、输出 7；生成器核对七张图的 SHA256、预测集合、分数范围和队列数量；页面嵌入 7 张有效 PNG、对应 CSV 7 行；内联 JavaScript 通过 `node --check`。受控浏览器的 `file:` URL 策略拒绝预览本地 HTML，因此没有完成浏览器中的实际点击与导出验证；页面可由用户在本机打开检查。

本轮没有取得人工核实标签、实地真值或新地区独立评价，因此不计算准确率，也不据此判断 PermanentCrop、River、Residential 或 Industrial 是否正确。
