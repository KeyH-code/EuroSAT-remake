"""01 真实图片各训练一个极小批次，验证独立仓库的参数更新链路。"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if not torch.cuda.is_available():
    raise RuntimeError("训练冒烟要求 CUDA；此脚本不作为 CPU 替代验收")
device = torch.device("cuda")

# CNN：从本地复制图像取两个样本，确认一次 optimizer step 真正修改参数。
sys.path.insert(0, str(ROOT / "cnn" / "src"))
from train_fit_loop import EurosatCNN, make_loader, train_one_epoch  # noqa: E402
from transforms import build_train_transform  # noqa: E402

cnn_loader = make_loader("train", build_train_transform(), 2, shuffle=False)
cnn_batch = DataLoader(Subset(cnn_loader.dataset, range(2)), batch_size=2)
cnn = EurosatCNN().to(device)
cnn_optimizer = torch.optim.AdamW(cnn.parameters(), lr=1e-3)
cnn_before = cnn.head.weight.detach().clone()
cnn_loss, cnn_accuracy, cnn_skipped = train_one_epoch(
    cnn, cnn_batch, cnn_optimizer, nn.CrossEntropyLoss(), device, None
)
assert not torch.equal(cnn_before, cnn.head.weight.detach()), "CNN 参数未更新"
print("CNN: 真实图片单步更新通过")

# ResNet18：独立导入项目模型及配置，只更新分类头，不写入历史快照。
sys.path.remove(str(ROOT / "cnn" / "src"))
sys.modules.pop("transforms", None)  # 两个旧工程各有同名模块，依次独立导入。
sys.path.insert(0, str(ROOT / "resnet18" / "src"))
from data import make_loader as make_resnet_loader  # noqa: E402
from model import model as resnet  # noqa: E402
from train import train_one_epoch as train_resnet_one_epoch  # noqa: E402
from transforms import make_transforms  # noqa: E402

config_path = ROOT / "resnet18" / "configs" / "config16_sgd_lr_0.05_momentum_0.9_20ep.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
train_transform, _ = make_transforms(config)
resnet_loader = make_resnet_loader("train", train_transform, 2, shuffle=False)
resnet_batch = DataLoader(Subset(resnet_loader.dataset, range(2)), batch_size=2)
resnet.to(device)
resnet_optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.05, momentum=0.9)
resnet_scaler = torch.amp.GradScaler("cuda", init_scale=1024)  # 小样本冒烟避免默认初始放大值触发跳步。
resnet_before = resnet.fc.weight.detach().clone()
result = train_resnet_one_epoch(resnet, resnet_batch, resnet_optimizer, resnet_scaler)
assert not torch.equal(resnet_before, resnet.fc.weight.detach()), "ResNet18 分类头未更新"
print(f"ResNet18: 真实图片单步更新通过，loss={result['train_loss_ave']:.4f}")

# 冒烟也是一次真实运行；保存可追溯记录，避免只留在终端滚动输出中。
record_dir = ROOT / "runs" / "migration_smoke"
record_dir.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
record = {
    "timestamp": datetime.now().astimezone().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "torch_version": str(torch.__version__),
    "images_per_model": 2,
    "cnn": {"loss": cnn_loss, "accuracy": cnn_accuracy, "skipped_steps": cnn_skipped, "parameters_updated": True},
    "resnet18": {"loss": result["train_loss_ave"], "accuracy": result["train_accuracy"], "parameters_updated": True},
}
record_path = record_dir / f"{stamp}.json"
record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"运行记录：{record_path}")
