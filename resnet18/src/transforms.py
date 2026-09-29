from torchvision.transforms import v2
import torchvision.transforms as tf
import torch
from data import CLASS_NAMES


class RandomRightAngleRotation:
    """随机选择 0°、90°、180° 或 270°，不做插值重采样。"""

    def __call__(self, image):
        quarter_turns = int(torch.randint(0, 4, ()).item())
        return torch.rot90(image, quarter_turns, dims=(-2, -1))


def _make_train_transform(config, augmentation, color_jitter=None):
    train_operations = [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        tf.Resize((224, 224)),
    ]
    if not isinstance(augmentation, dict):
        raise ValueError("train_augmentation 必须是对象")
    allowed_keys = {
        "horizontal_flip_probability",
        "vertical_flip_probability",
        "rotate_90",
        "translation_probability",
        "translation_fraction",
        "random_crop_probability",
        "random_crop_scale_min",
        "random_crop_scale_max",
    }
    unknown_keys = set(augmentation) - allowed_keys
    if unknown_keys:
        raise ValueError(f"train_augmentation 包含不支持的字段：{sorted(unknown_keys)}")
    for key in (
        "horizontal_flip_probability",
        "vertical_flip_probability",
        "translation_probability",
        "random_crop_probability",
    ):
        probability = augmentation.get(key, 0.0)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError(f"{key} 必须是 [0, 1] 内的数值")
    rotate_90 = augmentation.get("rotate_90", False)
    if not isinstance(rotate_90, bool):
        raise ValueError("train_augmentation.rotate_90 必须是布尔值")

    horizontal_probability = augmentation.get("horizontal_flip_probability", 0.0)
    vertical_probability = augmentation.get("vertical_flip_probability", 0.0)
    if horizontal_probability:
        train_operations.append(v2.RandomHorizontalFlip(p=horizontal_probability))
    if vertical_probability:
        train_operations.append(v2.RandomVerticalFlip(p=vertical_probability))
    if rotate_90:
        train_operations.append(RandomRightAngleRotation())

    crop_scale_min = augmentation.get("random_crop_scale_min", 0.8)
    crop_scale_max = augmentation.get("random_crop_scale_max", 1.0)
    for key, value in (
        ("random_crop_scale_min", crop_scale_min),
        ("random_crop_scale_max", crop_scale_max),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < value <= 1.0
        ):
            raise ValueError(f"{key} 必须是 (0, 1] 内的数值")
    if crop_scale_min > crop_scale_max:
        raise ValueError("random_crop_scale_min 不能大于 random_crop_scale_max")
    crop_probability = augmentation.get("random_crop_probability", 0.0)
    if crop_probability:
        train_operations.append(
            tf.RandomApply(
                [
                    v2.RandomResizedCrop(
                        size=(224, 224),
                        scale=(crop_scale_min, crop_scale_max),
                        ratio=(1.0, 1.0),
                        antialias=True,
                    )
                ],
                p=crop_probability,
            )
        )

    translation_fraction = augmentation.get("translation_fraction", 0.10)
    if (
        isinstance(translation_fraction, bool)
        or not isinstance(translation_fraction, (int, float))
        or not 0.0 <= translation_fraction <= 1.0
    ):
        raise ValueError("translation_fraction 必须是 [0, 1] 内的数值")
    translation_probability = augmentation.get("translation_probability", 0.0)
    if translation_probability:
        train_operations.append(
            tf.RandomApply(
                [
                    tf.RandomAffine(
                        degrees=0,
                        translate=(translation_fraction, translation_fraction),
                        fill=config["transform"]["mean"],
                        interpolation=tf.InterpolationMode.BILINEAR,
                    )
                ],
                p=translation_probability,
            )
        )
    if color_jitter is not None:
        train_operations.append(
            tf.RandomApply(
                [v2.ColorJitter(
                    brightness=color_jitter["brightness"],
                    contrast=color_jitter["contrast"],
                    saturation=color_jitter["saturation"],
                    hue=color_jitter["hue"],
                )],
                p=color_jitter["probability"],
            )
        )
    train_operations.append(
        v2.Normalize(mean=config["transform"]["mean"], std=config["transform"]["std"])
    )
    return v2.Compose(train_operations)


def make_transforms(config):
    augmentation = config.get("train_augmentation", {})
    train_transform = _make_train_transform(config, augmentation)

    targeted = config.get("class_conditional_augmentation")
    if targeted is not None:
        if not isinstance(targeted, dict) or set(targeted) not in (
            {"classes", "color_jitter"}, {"classes", "random_crop_scale_min"}
        ):
            raise ValueError("class_conditional_augmentation 只能指定 classes 和一种增强")
        classes = targeted["classes"]
        if (
            not isinstance(classes, list)
            or not classes
            or len(set(classes)) != len(classes)
            or any(name not in CLASS_NAMES for name in classes)
        ):
            raise ValueError("class_conditional_augmentation.classes 必须是非空且不重复的类别列表")
        selected_augmentation = dict(augmentation)
        color_jitter = targeted.get("color_jitter")
        if color_jitter is not None:
            expected = {"probability", "brightness", "contrast", "saturation", "hue"}
            if not isinstance(color_jitter, dict) or set(color_jitter) != expected:
                raise ValueError(f"color_jitter 字段必须恰好为 {sorted(expected)}")
            for key in expected:
                value = color_jitter[key]
                upper = 0.5 if key == "hue" else 1.0
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= value <= upper
                ):
                    raise ValueError(f"color_jitter.{key} 必须是 [0, {upper}] 内的数值")
        else:
            selected_augmentation["random_crop_scale_min"] = targeted["random_crop_scale_min"]
            if not augmentation.get("random_crop_probability", 0.0):
                raise ValueError("调整裁剪面积下限要求启用 random_crop_probability")
        selected_transform = _make_train_transform(
            config, selected_augmentation, color_jitter=color_jitter
        )
        train_transform = {"default": train_transform}
        for name in classes:
            train_transform[CLASS_NAMES.index(name)] = selected_transform

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            tf.Resize((224, 224)),
            v2.Normalize(mean=config["transform"]["mean"], std=config["transform"]["std"]),
        ]
    )

    return train_transform,val_transform
