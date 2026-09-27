"""Batch-level learning-rate schedules used by ResNet18 experiments."""

import math


class LinearWarmupCosineScheduler:
    """Linear warmup followed by cosine decay, resumable through state_dict.

    ``step_count`` counts successful optimizer updates. The LR applied to update
    zero is ``base_lr * start_factor``; after ``warmup_steps`` updates it reaches
    ``base_lr`` and cosine decay starts. After the final planned update it is
    ``eta_min``.
    """

    step_per_batch = True

    def __init__(
        self, optimizer, *, total_epochs, warmup_epochs, steps_per_epoch,
        start_factor, eta_min,
    ):
        if total_epochs <= warmup_epochs or warmup_epochs <= 0:
            raise ValueError("warmup_epochs 必须大于 0 且小于 total_epochs")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch 必须大于 0")
        if not 0 < start_factor <= 1:
            raise ValueError("start_factor 必须位于 (0, 1] 内")

        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.start_factor = float(start_factor)
        self.eta_min = float(eta_min)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.decay_steps = self.total_steps - self.warmup_steps
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if any(not self.eta_min <= lr for lr in self.base_lrs):
            raise ValueError("eta_min 不得高于任一参数组的基准学习率")
        self.step_count = 0
        self._apply_lrs()

    def _lr_at(self, base_lr, step):
        if step < self.warmup_steps:
            fraction = step / self.warmup_steps
            return base_lr * (self.start_factor + (1 - self.start_factor) * fraction)
        progress = min(step - self.warmup_steps, self.decay_steps) / self.decay_steps
        return self.eta_min + 0.5 * (base_lr - self.eta_min) * (
            1 + math.cos(math.pi * progress)
        )

    def _apply_lrs(self):
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = self._lr_at(base_lr, self.step_count)

    def step(self):
        if self.step_count < self.total_steps:
            self.step_count += 1
            self._apply_lrs()

    def state_dict(self):
        return {
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "start_factor": self.start_factor,
            "eta_min": self.eta_min,
            "base_lrs": self.base_lrs,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state):
        for key in (
            "total_epochs", "warmup_epochs", "steps_per_epoch", "start_factor",
            "eta_min", "base_lrs",
        ):
            if state[key] != getattr(self, key):
                raise ValueError(f"学习率调度器状态不匹配：{key}")
        step_count = int(state["step_count"])
        if not 0 <= step_count <= self.total_steps:
            raise ValueError("checkpoint 的学习率调度进度超出计划范围")
        self.step_count = step_count
        self._apply_lrs()
