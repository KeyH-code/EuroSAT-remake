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
