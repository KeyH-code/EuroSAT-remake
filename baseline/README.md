# EuroSAT 训练基线

此目录保存两条模型线已选验证配置的参考，不保存权重、训练记录或评价产物。新运行仍须在新的 `runs/` 实验目录中创建；不要把训练或评价产物写入此目录。**当前项目状态与最终 test 指标见 [最终结果](../final_result/README.md)**；本页的 CNN 与 ResNet18 结果不是两种模型在 test 上的对照。

## ResNet18

- 配置：[`resnet18_best.json`](resnet18_best.json)
- 来源运行：`resnet18_layer4_lr0p0001_warmup_cosine_bs64_flip_rotate_translate_crop_20ep_01`
- 来源设置：`layer4+fc`，AdamW，lr=1e-4，weight decay=0.01，batch size=64，1 epoch warmup + 20 epoch cosine，训练集随机翻转/90°倍数旋转/裁剪/平移。
- val 选模结果：best epoch 17，accuracy 0.983210（3982/4050），普通交叉熵 0.061789。
- 同一最高 val accuracy 还出现在 label smoothing 0.05 组（epoch 7，val loss 0.117645）和 flip+rotate 组（epoch 18，val loss 0.067543）。本配置在并列最高 val accuracy 的组中对应最低 val loss。
- 完整增强相对纯基线在 seed 20260920、20260929、20260930 的固定 val 上分别多正确 10、6、4 张，见[三 seed 配对报告](../resnet18/reports/30_pure_baseline_two_seed_20ep.md)。三次增强训练中，seed 20260920 的 best val 正确数最高，因此最终锁定该权重。
- 最终单次 test：accuracy 0.982963（3981/4050），macro-F1 0.982250，多分类 OvR macro ROC AUC 0.999668；test 未用于选择上述配置或权重。这里没有纯基线的 test 指标，不将 val 收益写成 test 收益。

## CNN

- 配置：[`cnn_best.json`](cnn_best.json)
- 来源运行：`adamw_003_continue_30ep`，历史训练配置见只读归档 `artifacts/legacy_eurosat/experiments/adamw_003_continue_30ep/config.json`，历史成绩汇总见 `resnet18/reports/09_history_baselines.md`。
- 来源设置：EurosatCNN，AdamW，lr=0.003，weight decay=0.01，batch size=128，30 epoch，首 epoch 从 3e-4 warmup 至 3e-3 后保持，FP32，无数据增强。
- 历史 val 结果：best epoch 28，accuracy 0.940247（3808/4050），loss 0.171498，错误 242；此 CNN 模型未进行本阶段的最终 test 评价。
- 该历史运行从 `warmup1_to_003` 的 epoch 15 checkpoint 续训至 epoch 30。此目录只登记训练配置，不包含该 checkpoint；从头运行本配置不等于复现当时的续训轨迹。

## 使用

这些文件是配置参考。ResNet18 可复制配置后改 `experiment_id`，确保新实验写入新的 `runs/resnet18/<experiment_id>/`。CNN 训练入口要求配置位于实验目录内：先创建新的 `runs/cnn/<experiment_id>/`，再将配置复制为该目录的 `config.json`，然后启动训练。
