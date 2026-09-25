# 迁移来源与核验（2026-09-24）

**本文件只回答「搬运是否原样」；历史结果、旧配置差异与解耦决定见 [`migration-record.md`](migration-record.md)。**

源教学仓库为 `E:\deeplearning`，迁移时 HEAD `807fec304449f250ec19036048cc6217d81432f9`。本次仅复制，未修改原仓库或其缓存。新仓库的历史快照来自 `E:\deeplearning\.cache\datasets\eurosat`，原样放在 `artifacts/legacy_eurosat/`，其中含每次运行留下的配置、训练日志、逐轮/逐类指标、图表、预测输入、评价产物和 last/best checkpoint。`artifacts/`、`data/`、`runs/` 均被 Git 忽略。

迁移当时逐文件比较源缓存与目标快照：28,094 对文件，双方各 1,239,999,442 字节，缺失 0、SHA-256 不同 0。另将完整 RGB 图像复制到 `data/EuroSAT_RGB/` 供独立运行，对快照内的 RGB 目录逐文件比较：27,000 对文件，缺失 0、SHA-256 不同 0。源缓存最近写入时间为 2026-09-24 15:04:57；核对结束时文件数与字节数仍一致。此核对描述的是当时的快照，之后若原工程继续产生新实验，需要另行增量归档。

原始 `my_split.csv` 在源缓存、快照和运行用 `data/manifests/` 三处的 SHA-256 均为 `66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf`。CSV 中记录的旧绝对路径未被重写，因此旧 checkpoint 的清单哈希保持有效；运行时按类别与文件名映射到新数据目录。ResNet18 原预训练权重及新仓库两份副本的 SHA-256 均为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

## 独立运行核验

- `conda run -n dl-reboot python -m pytest -q cnn/tests resnet18/tests`：**8 项通过**。解耦后测试不再读取历史快照，并新增「源码不得引用快照」的守卫用例。
- CNN、ResNet18 均在复制的真实图像上完成训练与评价；`runs/cnn/baseline_ep2/`（2 轮，val acc 0.838519）与 `runs/resnet18/resnet18_head_probe_1ep_01/`（1 轮，val acc 0.905926）是新仓库自有的首批产物，明细见 `migration-record.md` 第 7 节。这是入口验证，不是整轮训练或 M2 门槛证明。
- 迁移当日曾对 32 张历史预测输入完成推理（CSV 在 `runs/cnn_predictions/` 与 `runs/resnet18/predictions/`）；解耦后用新 checkpoint 对 3,000 张真实图像重跑 `cnn/predict.py` 成功。
- 迁移当日曾用历史 checkpoint 复评并核对指标一致（CNN ep5 val accuracy `0.8911111111111111`、ResNet18 ep11 `0.9508641975308642`），明细见 `migration-record.md` 第 5 节。**该核验只证明搬运无误；历史 checkpoint 不参与新的训练、评价或预测。**
- 两个 train 入口的 `--help` 均可在独立目录启动，`--resume` 契约已实测（新 checkpoint 可续，历史快照 checkpoint 与旧清单路径均被拒绝）。未启动耗时的完整新训练；M2 的 98% 验收仍未通过。

新评价、预测及冒烟结果均单独保存在 `runs/`，不会写回 `artifacts/legacy_eurosat/` 或旧教学工程。完整性核对与指标一致性针对已迁移快照，不保证将来原工程新增的运行自动同步。
