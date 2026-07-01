"""
MIMO-WM: 多输入多输出状态空间世界模型

MIMO-SSM为每个输入维度独立维护状态空间,
通过并行扫描实现高效序列建模, 引入sigmoid门控增强表达能力.
"""
import torch
import torch.nn as nn
from .ssm_world_model import DiagSSM


class MIMOLayer(nn.Module):
    """MIMO块: LayerNorm + DiagSSM + sigmoid门控 + 残差连接"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.output(x) * torch.sigmoid(self.gate(x))
        return residual + x


class MIMOWorldModel(nn.Module):
    """MIMO世界模型: 编码器 + MIMO块 + 解码器"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList([
            MIMOLayer(d_model, d_state) for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(
                states.shape[0], states.shape[1] - actions.shape[1],
                actions.shape[-1], device=actions.device
            )
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            h = block(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])
