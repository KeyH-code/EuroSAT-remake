import torch
from torchvision.transforms import v2

TRAIN_MEAN = (0.3438910908404892, 0.37992634527006, 0.40740581022786704)
TRAIN_STD  = (0.20255132079555324, 0.13690415097787975, 0.11545699660990988)

def build_train_transform(mean=TRAIN_MEAN, std=TRAIN_STD):
    # 三步各管一件事，缺一不可：
    #   ToImage()                   PIL -> uint8 张量
    #   ToDtype(float32, scale=1)   uint8 0..255 -> float32 0..1
    #   Normalize()                 (x - mean) / std
    # 注意 v2.ToDtype 对 PIL 输入是空转，不能省掉 ToImage。
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32,scale=True),
            v2.Normalize(mean = mean ,std = std)
        ]
    )
    return transform
    
def build_eval_transform(mean=TRAIN_MEAN, std=TRAIN_STD):
    # 与训练路径内容相同（这一步两者都是确定性处理）。
    # 分开写成两个函数是为了将来把随机增强只加进 build_train_transform。
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32,scale=True),
            v2.Normalize(mean = mean ,std = std)
        ]
    )
    return transform
