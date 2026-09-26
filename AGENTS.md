# EuroSAT 独立实践项目

- `data/` 保存原始 RGB 数据和原样划分清单；不得重划分，也不得将 test 用于调参。
- 新实验一律写入 `runs/`，不得写入 `artifacts/`，也不得覆盖既有实验目录。
- `artifacts/legacy_eurosat/` 是历史运行记录快照，**原样冻结、不删不改、也不被任何代码读取**。
- `artifacts/legacy_teaching/` 是教学期与迁移期的代码/文档归档，**不可运行、不可 import**；不要把它复制回 `cnn/`、`resnet18/`。
- 数据、运行记录、模型权重和缓存不进入 Git。源码、配置模板、说明进入 Git。
- 沿用 `dl-reboot` Conda 环境；Windows 中文路径、文本按 UTF-8 处理。
- 仓库当前没有自动化测试；改代码后按 `README.md` 的「测试」一节人工验证链路。
