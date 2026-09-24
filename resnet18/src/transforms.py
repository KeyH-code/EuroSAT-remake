from torchvision.transforms import v2
import torchvision.transforms as tf
from data import EUROSAT_TRAIN_MEAN,EUROSAT_TRAIN_STD
import torch

def make_transforms(config):
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32,scale=True),
            tf.Resize((224,224)),
            # 这里可能以后有随机增强，现在先不执行随机增强
            v2.Normalize(mean = config["transform"]["mean"],std = config["transform"]["std"])
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32,scale=True),
            tf.Resize((224,224)),
            v2.Normalize(mean = config["transform"]["mean"],std = config["transform"]["std"])
        ]
    )

    return train_transform,val_transform
