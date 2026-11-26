# EuroSAT 独立实践项目

- `data/` 保存原始 RGB 数据和原样划分清单；不得重划分，也不得将 test 用于调参。
- 新实验一律写入 `runs/`，不得写入 `artifacts/`，也不得覆盖既有实验目录。
- `artifacts/legacy_eurosat/` 是历史运行记录快照，**原样冻结、不删不改、也不被任何代码读取**。
- `artifacts/legacy_teaching/` 是教学期与迁移期的代码/文档归档，**不可运行、不可 import**；不要把它复制回 `cnn/`、`resnet18/`。
- 数据、运行记录、模型权重和缓存不进入 Git。源码、配置模板、说明进入 Git。
- 当前阶段最终模型为 ResNet18 config41、seed 20260920、epoch 17；已锁定权重并完成一次 test 评价（3981/4050）。当前状态和完整指标见 `final_result/README.md`。以后继续研究只用 train/val 决策，不用已查看的 test 结果调参或重新选模。
- 训练和正式评价沿用 `dl-reboot` Conda 环境。公开 CPU 示例使用独立环境，运行和核验说明见 `README.md`。
- 记录实验变量、数据范围、评价结果和选模依据。保留项目决策与对应证据。
