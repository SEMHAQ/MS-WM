"""Mamba世界模型 — 与SSM-WM架构对齐，用Mamba替换DiagSSM"""
import types, sys
fake_gen = types.ModuleType('transformers.generation')
fake_gen.GreedySearchDecoderOnlyOutput = type('x', (), {})
fake_gen.SampleDecoderOnlyOutput = type('x', (), {})
fake_gen.TextStreamer = type('x', (), {})
sys.modules['transformers.generation'] = fake_gen

import torch
import torch.nn as nn
import numpy as np
from mamba_ssm import Mamba
# SSM-WM reference only


class MambaBlock(nn.Module):
    """Mamba块: LayerNorm + Mamba + 门控 + 残差 (与SSMBlock结构一致)"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        mamba_out = self.mamba(x_norm)
        g = torch.sigmoid(self.gate(x_norm))
        out = g * mamba_out + (1 - g) * x_norm
        return residual + out


class MambaWorldModel(nn.Module):
    """Mamba世界模型 — 架构与SSM-WM完全对齐"""
    def __init__(self, state_dim=28, action_dim=7, d_model=128, d_state=16, n_layers=4):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )
        # 与SSM-WM相同的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1],
                              device=actions.device, dtype=actions.dtype)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            x = block(x)
        x = self.norm(x)
        delta_s = self.decoder(x[:, -1, :])
        return states[:, -1, :] + delta_s


if __name__ == '__main__':
    import time

    device = torch.device('cuda')
    state_dim = 376
    action_dim = 17

    model = MambaWorldModel(state_dim=state_dim, action_dim=action_dim,
                            d_model=128, d_state=16, n_layers=4).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Mamba-WM params: {params:.2f}M")

    # 推理时间测试 (B=64)
    model.eval()
    for T in [16, 32, 64, 128, 256, 512]:
        states = torch.randn(64, T, state_dim).to(device)
        actions = torch.randn(64, T - 1, action_dim).to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(states, actions)
        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(states, actions)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
        print(f"T={T}: {np.median(times):.1f} ms")
