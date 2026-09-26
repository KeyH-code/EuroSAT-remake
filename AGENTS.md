# EuroSAT 独立实践项目

- 本仓库独立于 `E:\deeplearning`；运行代码不得读写旧教学项目。
- `data/` 保存原始 RGB 数据和原样划分清单；不得重划分，也不得将 test 用于调参。
- `artifacts/legacy_eurosat/` 是迁移时的逐次运行记录快照，**原样冻结、不删不改、也不被任何代码读取**；历史结果与旧配置差异见 `docs/migration-record.md`。新实验一律写入 `runs/`，旧实验要复用就按记录里的参数重新训练。
- 数据、运行记录、模型权重和缓存不进入 Git。源码、配置模板、测试、说明进入 Git。
- 沿用 `dl-reboot` Conda 环境；Windows 中文路径、文本按 UTF-8 处理。
- 旧教学项目与它的现有缓存不可删除、覆盖或移动。



