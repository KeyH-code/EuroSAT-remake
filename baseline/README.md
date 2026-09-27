# EuroSAT 训练基线

此目录保存两条模型线最佳验证结果对应的配置参考，不保存权重、训练记录或评价产物。新运行仍须在新的 `runs/` 实验目录中创建；不要把训练或评价产物写入此目录。

## ResNet18

- 配置：[`resnet18_best.json`](resnet18_best.json)
- 来源运行：`resnet18_layer4_lr0p0001_warmup_cosine_bs64_flip_rotate_translate_crop_20ep_01`
- 来源设置：`layer4+fc`，AdamW，lr=1e-4，weight decay=0.01，batch size=64，1 epoch warmup + 20 epoch cosine，训练集随机翻转/90°倍数旋转/裁剪/平移。
- val 结果：best epoch 17，accuracy 0.983210（3982/4050），loss 0.061789；test 未使用。
- 同一最高 val accuracy 还出现在 label smoothing 0.05 组（epoch 7，val loss 0.117645）和 flip+rotate 组（epoch 18，val loss 0.067543）。本配置在并列最高 val accuracy 的组中对应最低 val loss。

## CNN

- 配置：[`cnn_best.json`](cnn_best.json)
- 来源运行：`adamw_003_continue_30ep`，历史训练配置见只读归档 `artifacts/legacy_eurosat/experiments/adamw_003_continue_30ep/config.json`，历史成绩汇总见 `resnet18/reports/09_history_baselines.md`。
- 来源设置：EurosatCNN，AdamW，lr=0.003，weight decay=0.01，batch size=128，30 epoch，首 epoch 从 3e-4 warmup 至 3e-3 后保持，FP32，无数据增强。
- val 结果：best epoch 28，accuracy 0.940247（3808/4050），loss 0.171498，错误 242；test 未使用。
- 该历史运行从 `warmup1_to_003` 的 epoch 15 checkpoint 续训至 epoch 30。此目录只登记训练配置，不包含该 checkpoint；从头运行本配置不等于复现当时的续训轨迹。

## 使用

这些文件是配置参考。ResNet18 可复制配置后改 `experiment_id`，确保新实验写入新的 `runs/resnet18/<experiment_id>/`。CNN 训练入口要求配置位于实验目录内：先创建新的 `runs/cnn/<experiment_id>/`，再将配置复制为该目录的 `config.json`，然后启动训练。
