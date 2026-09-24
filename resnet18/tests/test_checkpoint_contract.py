"""检查 Agent checkpoint 夹具能恢复模型、优化器与采样随机状态。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402


def test_checkpoint_round_trip(tmp_path):
    # 小线性层只承担状态恢复测试，不作为 ResNet 微调或 GPU 训练证据。
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(7))
    input_batch = torch.ones(2, 3)
    model(input_batch).sum().backward()
    optimizer.step()  # 让 AdamW 的动量状态真实存在。
    expected_logits = model(input_batch).detach().clone()
    expected_generator_state = loader.generator.get_state().clone()

    path = tmp_path / "last.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, scaler=None, epoch=2,
        best_epoch=1, best_val_accuracy=0.75, config={"seed": 7}, train_loader=loader,
    )
    with torch.no_grad():
        model.weight.add_(10)
    loader.generator.manual_seed(99)

    checkpoint = load_checkpoint(
        path, model=model, optimizer=optimizer, map_location="cpu",
        train_loader=loader, restore_rng=True,
    )
    assert checkpoint["epoch"] == 2 and checkpoint["best_epoch"] == 1
    torch.testing.assert_close(model(input_batch), expected_logits)
    torch.testing.assert_close(loader.generator.get_state(), expected_generator_state)
    assert optimizer.state_dict()["state"]  # 恢复后动量缓存仍在。
