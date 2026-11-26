"""构建 EuroSAT 十类 ResNet18，调用方选择是否使用 ImageNet 初始化。"""

from torch import nn
from torchvision.models import resnet18

from data import NUM_CLASSES


def build_resnet18(weights=None):
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
