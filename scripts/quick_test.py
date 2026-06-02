"""
快速测试: 验证模型可以正常前向传播
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ssm_world_model import SSMWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel


def test_model_forward():
    """测试所有模型的前向传播"""
    batch_size = 4
    seq_len = 16
    state_dim = 28
    action_dim = 7

    # 创建测试数据
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len - 1, action_dim)

    print("Testing model forward pass...")
    print(f"  Input: states={states.shape}, actions={actions.shape}")

    # 测试SSM世界模型
    print("\n1. SSM World Model")
    model_ssm = SSMWorldModel(state_dim=state_dim, action_dim=action_dim)
    pred_ssm = model_ssm(states, actions)
    print(f"   Output: {pred_ssm.shape}")
    print(f"   Params: {sum(p.numel() for p in model_ssm.parameters()) / 1e6:.2f}M")
    assert pred_ssm.shape == (batch_size, state_dim), f"Expected ({batch_size}, {state_dim}), got {pred_ssm.shape}"

    # 测试多步预测
    future_actions = torch.randn(batch_size, 8, action_dim)
    traj_ssm = model_ssm.predict_trajectory(states, actions, future_actions)
    print(f"   Trajectory: {traj_ssm.shape}")
    assert traj_ssm.shape == (batch_size, 8, state_dim)

    # 测试LSTM基线
    print("\n2. LSTM World Model")
    model_lstm = LSTMWorldModel(state_dim=state_dim, action_dim=action_dim)
    pred_lstm = model_lstm(states, actions)
    print(f"   Output: {pred_lstm.shape}")
    print(f"   Params: {sum(p.numel() for p in model_lstm.parameters()) / 1e6:.2f}M")

    # 测试Transformer基线
    print("\n3. Transformer World Model")
    model_tf = TransformerWorldModel(state_dim=state_dim, action_dim=action_dim)
    pred_tf = model_tf(states, actions)
    print(f"   Output: {pred_tf.shape}")
    print(f"   Params: {sum(p.numel() for p in model_tf.parameters()) / 1e6:.2f}M")

    # 测试数据集
    print("\n4. Dataset (synthetic)")
    from src.data.robot_dataset import RobotStateDataset
    dataset = RobotStateDataset("data/test", seq_len=16, split="train")
    sample = dataset[0]
    print(f"   States: {sample['states'].shape}")
    print(f"   Actions: {sample['actions'].shape}")
    print(f"   Target: {sample['target'].shape}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model_forward()
