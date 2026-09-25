# 旧配置档案索引

这里**不放配置副本**，只登记旧配置的位置、身份哈希与核对方法。原因：快照 `artifacts/legacy_eurosat/` 本身被原样冻结在同一台机器上，再复制一份进 Git 会产生两个可能各自漂移的副本；记下哈希即可随时确认快照未被改动。

配置内容差异、历史指标与新仓库取值见 [`../migration-record.md`](../migration-record.md)。

## 位置

| 模型 | 路径模式 | 份数 |
| --- | --- | ---: |
| 小 CNN | `artifacts/legacy_eurosat/experiments/<实验名>/config.json` | 18 |
| ResNet18 | `artifacts/legacy_eurosat/resnet18_finetune/<运行名>/config.json` | 12 |

以上均以仓库根为基准。所有旧配置都记有旧仓库绝对路径（`E:\deeplearning\…`），**不能直接在新仓库运行**。

## 核对（只读）

```powershell
$root = 'artifacts\legacy_eurosat'
Get-ChildItem -Recurse -Filter config.json $root |
  ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(),
                                    $_.FullName.Replace((Resolve-Path $root).Path + '\', '') }
```

## 逐文件 SHA-256（迁移时登记）

小 CNN：

| 配置文件 | SHA-256 |
| --- | --- |
| `experiments/formal_training/config.json` | `8893e36cd21c6d1cfb8c68374327af63113156e7fa94b1427c8cc0b40dd4f7aa` |
| `experiments/learning_example_work/`（教学期，无 config.json） | — |
| `experiments/warmup_lr_0.002/config.json` | `732d3c4ca3141a4e689a325093f0a3df759ec261e3d23b9a090b92163a03d525` |
| `experiments/lr_warmup_sweep_15ep/warmup5_to_002/config.json` | `1e766d0a50281e8ecf86c678fbbac9738f0599badc36f27e8938991a937bcacf` |
| `experiments/lr_warmup_sweep_15ep/warmup1_to_002/config.json` | `7ac79cee0a8f674bbc4cef5dc8952403aad95bc6ccfd03be6238cb94622b1b37` |
| `experiments/lr_warmup_sweep_15ep/warmup1_to_003/config.json` | `8291785e27bdf3a7a583dd55f012a23ef2d32624a9371052d72e393de5f4039b` |
| `experiments/lr_004_exploration_15ep/warmup1_to_004/config.json` | `21b1a0f2d39ecc4ecd8360d1883c71163d81bfb6b27a7a74bcaaee8b341638f4` |
| `experiments/lr_004_exploration_15ep/stepup_epoch7_to_004/config.json` | `ce005a7252889bdc6e1ee001099a35ae8c0f37d1298930bb22195ab63b8dfe22` |
| `experiments/batch_size_64_15ep/config.json` | `73a1ff6fef3af9dd745f7e964fffdaeae1b328651823e598babcfbe29c3c56af` |
| `experiments/optimizer_regularization_15ep/weight_decay_0/config.json` | `11576317195382ca74e968aee5c2dca8e3e539239888d421f853256bd4bd1569` |
| `experiments/optimizer_regularization_15ep/weight_decay_005/config.json` | `9450943d8f99bc2e7b14d14cd81d1e56e027f8e5a0b10b032942df55149a770a` |
| `experiments/optimizer_regularization_15ep/momentum_sgd_09/config.json` | `89ecf37de6d010a1a322d7d1f252394787ac216258fc33930ef63c4d93d148a8` |
| `experiments/optimizer_regularization_15ep/adamw_beta1_095/config.json` | `07c6b94ddc7eb963f22b73130388dcda3365f5858f83a6398ab9396809963d70` |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0001/config.json` | `dc2fef2cbe980c89c98601d157355b8997e3fe844cefb8184417506e35435a2a` |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0002/config.json` | `57be3bb46bed6003c940e2826cc1c5248f3274f2a11411a5e7f4a9d52df3e83a` |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0006/config.json` | `f54c80f62516ca574e94331267eedb03532826b1c60d4ba1b274ca7eaf902289` |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0006_extend20/config.json` | `30844e82f0c31d8a21a44e77f2cf6e6c1703ed7de9b6cbe0ab33f6eb30a8fa64` |
| `experiments/sgd_learning_rate_15ep/sgd_lr_0008/config.json` | `3e1d719edd074ebc5ece1f7acd4941987030901ccf9b7025e1faa31fb1d1dd09` |
| `experiments/adamw_003_continue_30ep/config.json` | `c8e8e46bf64ea23247f0df7e5d2b7c2471665f47d0b73f7f70d0c1deefab8575` |

ResNet18：

| 配置文件 | SHA-256 |
| --- | --- |
| `resnet18_finetune/resnet18_head_imagenet_5ep_01/config.json` | `02bb1d4fbde2b1cca7516465849dbaf1d6922c0fc538fd1729fe47ddeda07e52` |
| `resnet18_finetune/resnet18_head_lr0p0015_20ep_01/config.json` | `7c9cc3623e66cd8790be844cdba49ac9955846a2da377cc747972ba784559488` |
| `resnet18_finetune/resnet18_head_lr0p002_20ep_01/config.json` | `67126e8a4d78ff3da4cd52af4193567ebf61bc634c8e22c406ea2fc509e77e17` |
| `resnet18_finetune/resnet18_head_lr0p003_20ep_01/config.json` | `01a8f1c14af859f90ec2f3a7480f98a902bb8f76cdb3a4dad3d5817312818eb5` |
| `resnet18_finetune/resnet18_head_lr0p003_wd0_20ep_01/config.json` | `d2124dbec3c276474968d8bf1c11c2bbbd1c54e1884be400ed76659f23d40048` |
| `resnet18_finetune/resnet18_head_lr0p004_20ep_01/config.json` | `4371379cef3f2eff46d1abcfecd659bdc3cdb7ace280c648fd75e91d2ff6047b` |
| `resnet18_finetune/resnet18_head_lr0p005_20ep_01/config.json` | `0122bcc449259a94013db2625b0135df9fec0acd25dbc6636ff136b22b6cf648` |
| `resnet18_finetune/resnet18_head_lr0p006_20ep_01/config.json` | `b74a8c84883d48854332d956c0c05898a4285b04cae69f16d222c43d2d2357be` |
| `resnet18_finetune/resnet18_head_lr0p01_wd0p01_20ep_01/config.json` | `9b3b49aee1f688c830ce568c88641c10241f6f008d47272037e507a31cfaeab3` |
| `resnet18_finetune/resnet18_head_lr0p01_wd0_20ep_01/config.json` | `9a12452f7ed5ceeb027c69605cfa4f0d9f4ec5b7ca425a524732cef3081f980e` |
| `resnet18_finetune/resnet18_head_sgd_lr0p01_m0p9_20ep_01/config.json` | `10c93b6dfef9f6e840f1765b3b630dea5d4e89fba87914ff58f50781e648d5a6` |
| `resnet18_finetune/resnet18_head_sgd_lr0p05_m0p9_20ep_01/config.json` | `9f4ceac603e62de0c6ae7cc9db4273729e0236f096729659d6b3e74c5c69d5f6` |

## 使用规则

- 这些配置**只读**。不要复制到 `cnn/configs/` 或直接传给训练入口。
- 要重跑同类实验：复制新模板（CNN 用 `cnn/configs/baseline.json`，ResNet18 用 `resnet18/configs/configNN*.json`），改掉 `experiment_id`，产物落在 `runs/`。
- 快照目录不得写入、移动或删除。
