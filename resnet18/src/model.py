import os
import torch
import torchvision.models as models
from data import NUM_CLASSES,CLASS_NAMES,REPO_ROOT
from PIL import Image
from torch import nn

# 在请求预训练权重前，把 Torch Hub 的下载与读取目录固定到项目缓存。
# resnet18(weights='DEFAULT') 随后会使用该目录下的 checkpoints/。
TORCH_CACHE_DIR = REPO_ROOT / ".cache" / "torch"
TORCH_HUB_DIR = TORCH_CACHE_DIR / "hub"
TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
torch.hub.set_dir(str(TORCH_HUB_DIR))

model = models.resnet18(weights='DEFAULT')

# 冻结backbone
for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(model.fc.in_features,NUM_CLASSES)

# 只训练分类头不足以产出结果，继续解冻 layer4。
# 决定记录见 docs/migration-record.md 第 8 节：只训分类头历史最佳 0.9538，未达 98%。
# 注意：这一步只改 requires_grad，不改变模型结构，因此旧 checkpoint 仍能 load_state_dict；
# 但可训练范围已经变化，TRAINABLE_SCOPE 用于在续训时拦住不兼容的旧实验。
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# 当前可训练范围。checkpoint 会把它写进 config 快照，续训时据此拒绝跨范围的恢复。
TRAINABLE_SCOPE = "layer4+fc"
