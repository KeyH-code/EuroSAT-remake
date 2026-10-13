# 31：最终 test 与文档状态收尾

记录日期：2026-09-29

## 用户决策与本阶段边界

用户将完整随机增强的 ResNet18 作为当前阶段最终模型；解冻 layer3、换优化器和 ResNet50 留作后续研究。先按 val 锁定 config41、seed 20260920、epoch 17 的 `best.pt`，再对原样 test 做一次最终评价。test 不参与模型选择或参数调整。本轮只统一文档口径，不重训、不重跑 test、不更改数据或已落盘的指标。

## 已完成结果

| 指标 | 固定 val | 最终 test |
| --- | ---: | ---: |
| 样本数 | 4050 | 4050 |
| 正确数 | 3982 | 3981 |
| Accuracy | 0.983210 | 0.982963 |
| Macro-F1 | 0.982570 | 0.982250 |
| 多分类 OvR macro ROC AUC | 0.999647 | 0.999668 |
| 普通交叉熵 | 0.061789 | 0.063983 |

配置、权重和原样划分清单的 SHA-256、完整逐类指标与混淆矩阵见 [`final_result/README.md`](../../final_result/README.md)。最终 test 文件留在 Git 忽略的 `final_result/resnet18/test/`。完整增强相对纯基线在相同三个 seed 的固定 val 上分别多正确 10、6、4 张，见[报告 30](30_pure_baseline_two_seed_20ep.md)；没有纯基线 test 对照，因此不将该差值写作 test 收益。

## 文档口径

项目当前状态以根 [`README.md`](../../README.md)、[`baseline/README.md`](../../baseline/README.md)和[`final_result/README.md`](../../final_result/README.md)为入口。编号 01–30 的报告是按各自记录日期形成的实验轨迹：其中“test 未使用”只描述当时的训练或分析范围，不能理解为项目在 2026-09-29 的整体状态。

后续新实验继续只写入 `runs/`，沿用固定原样 train/val 划分，不用本次已经查看的 test 结果调参或筛选模型。空间独立性及外部场景准确率尚未建立，见[报告 22](22_fixed_transforms_clean_eval_similarity_audit.md)与[报告 25](25_field_patrol_live_imagery_demo.md)。
