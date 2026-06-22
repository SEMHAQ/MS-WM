"""
Fusion-SSM: 一种新型状态空间模型
===================================
核心创新：SSM-Attention混合架构，自适应融合长程和局部特征

动机分析：
- SSM擅长长程建模（Humanoid 348D最优）
- Transformer擅长局部精细建模（Walker2d/Hopper最优）
- 关键洞察：不同维度的任务需要不同的建模策略

设计原则：
1. SSM分支处理长程依赖（O(T log T)）
2. 轻量注意力分支处理局部精细特征（窗口限制，O(TW)）
3. 自适应门控融合两个分支的输出
4. 保持参数效率（<0.3M）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiagSSM(nn.Module):
    """对角SSM（与S4D-WM相同）"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.log_A_real = nn.Parameter(torch.randn(d_model, d_state) * 0.01 - 1.0)
        self.A_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, x):
        batch, L, D = x.shape
        N = self.d_state
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = dt.unsqueeze(-1) * A
        powers = torch.arange(L, device=x.device, dtype=x.dtype)
        dtA_pow = dtA.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)
        CB = self.C * self.B * dt.unsqueeze(-1)
        K = (CB.unsqueeze(-1) * dtA_pow).sum(dim=1).real
        K_fft = torch.fft.rfft(K, n=2*L)
        x_fft = torch.fft.rfft(x.permute(0, 2, 1), n=2*L)
        y = torch.fft.irfft(K_fft.unsqueeze(0) * x_fft, n=2*L)[:, :, :L]
        return y.permute(0, 2, 1) + x * self.D


class LocalAttention(nn.Module):
    """轻量级局部注意力（1D卷积实现，O(TW)）"""
    def __init__(self, d_model, window_size=8):
        super().__init__()
        # 用1D卷积模拟局部注意力
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=window_size,
                              stride=1, padding=window_size//2, groups=d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        # 局部特征提取
        local = self.conv(x.transpose(1, 2))[:, :, :L]  # (B, D, L) - 截断到原长度
        local = local.transpose(1, 2)  # (B, L, D)
        # 门控：学习什么信息有用
        g = torch.sigmoid(self.gate(x))  # (B, L, D)
        return self.out(g * local)


class FusionSSMBlock(nn.Module):
    """融合SSM块：SSM分支 + 注意力分支 + 维度感知自适应选择"""
    def __init__(self, d_model, d_state=16, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        self.attn = LocalAttention(d_model, window_size)
        # 维度感知门控：输入特征 → 决定用哪个分支
        self.dim_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 3),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)

        # 计算输入特征的"复杂度"信号
        # 高方差 = 复杂动力学 → 用SSM
        # 低方差 = 简单动力学 → 用注意力或跳过
        complexity = torch.std(x_norm, dim=-1, keepdim=True)  # (B, L, 1)
        complexity = complexity / (complexity.mean() + 1e-8)  # 归一化

        # SSM分支
        ssm_out = self.ssm(x_norm)

        # 注意力分支
        attn_out = self.attn(x_norm)

        # 维度感知门控
        gate_logits = self.dim_gate(x_norm)  # (B, L, 3)
        # 添加复杂度偏置：高复杂度倾向SSM，低复杂度倾向跳过
        bias = torch.cat([complexity, -complexity, torch.zeros_like(complexity)], dim=-1)  # (B, L, 3)
        weights = F.softmax(gate_logits + bias, dim=-1)

        fused = weights[..., 0:1] * ssm_out + weights[..., 1:2] * attn_out + weights[..., 2:3] * x_norm

        # 残差连接 + FFN
        x = residual + fused
        x = x + self.ffn(self.norm2(x))
        return x


class FSM(nn.Module):
    """
    FSM: Fusion State-space Model

    核心创新：
    1. SSM-Attention混合架构
    2. 自适应门控融合
    3. 窗口注意力保持效率

    优势：
    - 高维任务：SSM主导（类似S4D-WM）
    - 低维任务：注意力主导（类似Transformer）
    - 中维任务：两者融合
    """
    def __init__(self, state_dim=28, action_dim=7, d_model=128, d_state=16, n_layers=4, window_size=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList([
            FusionSSMBlock(d_model, d_state, window_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )
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


class GateSSMBlock(nn.Module):
    """简化版：仅使用改进的门控机制"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        # 改进的门控：双层 + 残差
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        ssm_out = self.ssm(x_norm)
        g = self.gate(x_norm)
        x = residual + g * ssm_out + (1 - g) * x_norm
        x = x + self.ffn(self.norm(x))
        return x


class GSSM(nn.Module):
    """GSSM: Gated SSM（改进门控版，简化版FSM）"""
    def __init__(self, state_dim=28, action_dim=7, d_model=128, d_state=16, n_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList([
            GateSSMBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )
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
