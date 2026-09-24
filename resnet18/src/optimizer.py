"""把运行时模型参数与可落盘的优化器超参数接起来。"""

import torch


def make_optimizer(model, config):
    optimizer_config = config["optimizer"]
    # 旧配置没有 name；默认 AdamW 以维持既有实验与 checkpoint 的恢复方式。
    optimizer_name = optimizer_config.get("name", "adamw").lower()

    # Parameter 对象不能放进 JSON；只把学习率等超参数写入 config。
    # 冻结参数不交给优化器，后续若解冻需重新核对参数组。
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"],
        )

    raise ValueError(f"不支持的优化器：{optimizer_name!r}")
