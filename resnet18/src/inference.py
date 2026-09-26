"""ResNet-18 批量推理的数据外壳与核心推理函数。"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class UnlabeledImageDataset(Dataset):
    """读取输入目录中的无标签图像，同时保留逐样本 filepath。"""

    def __init__(self, input_dir, transform):
        self.input_dir = Path(input_dir)
        self.transform = transform
        self.images = sorted(
            path
            for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise ValueError(f"输入目录中没有支持的图像：{self.input_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, str(path)


def predict(model, loader, device):
    """返回与 loader 样本顺序一致的 filepath、类别编号、置信度。

    输入契约：
    - loader 每批返回 images [B,3,224,224] 与同顺序的 filepaths；
    - model 输出 logits [B,10]；
    - 推理不能建立梯度，也不能改变模型参数或 BatchNorm 统计量。

    输出契约：
    - 返回三个等长 Python list：filepaths、pred_indices、confidences；
    - pred_indices/confidences 都是一张图对应一个标量，不保留 [B,1] 维度。
    """
    model.eval()
    all_filepaths = []
    all_preds = []
    all_confidences = []

    # predict 没有标签，因此不计算 loss、accuracy 或混淆矩阵。
    with torch.no_grad():
        for batch, filepaths in loader:
            probabilities = torch.softmax(model(batch.to(device)), dim=1)
            confidence, predictions = torch.topk(probabilities, k=1, dim=1)

            all_filepaths.extend(filepaths)
            all_preds.extend(predictions.squeeze(1).cpu().tolist())
            all_confidences.extend(confidence.squeeze(1).cpu().tolist())

    return all_filepaths, all_preds, all_confidences
