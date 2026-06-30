"""
基线模型: LSTM世界模型, Transformer世界模型 和 GRU世界模型
用于与SSM-WM进行对比实验
"""
import torch
import torch.nn as nn
import math


class LSTMWorldModel(nn.Module):
    """基于LSTM的循环世界模型"""

    def __init__(self, state_dim=28, action_dim=7, hidden_dim=128, n_layers=2):
        super().__init__()
        self.encoder = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.state_dim = state_dim

    def forward(self, states, actions):
        # 对齐长度
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device, dtype=actions.dtype)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        delta_s = self.decoder(x)
        return states[:, -1, :] + delta_s

    def predict_trajectory(self, init_states, init_actions, future_actions):
        states_seq = init_states.clone()
        actions_seq = init_actions.clone()
        predictions = []
        for h in range(future_actions.shape[1]):
            pred = self.forward(states_seq, actions_seq)
            predictions.append(pred)
            states_seq = torch.cat([states_seq[:, 1:], pred.unsqueeze(1)], dim=1)
            actions_seq = torch.cat([actions_seq[:, 1:], future_actions[:, h:h+1]], dim=1)
        return torch.stack(predictions, dim=1)


class GRUWorldModel(nn.Module):
    """基于GRU的循环世界模型"""

    def __init__(self, state_dim=28, action_dim=7, hidden_dim=128, n_layers=2):
        super().__init__()
        self.encoder = nn.Linear(state_dim + action_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.state_dim = state_dim

    def forward(self, states, actions):
        # 对齐长度
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device, dtype=actions.dtype)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = torch.nn.functional.gelu(self.encoder(x))
        out, _ = self.gru(x)
        x = out[:, -1, :]
        delta_s = self.decoder(x)
        return states[:, -1, :] + delta_s

    def predict_trajectory(self, init_states, init_actions, future_actions):
        states_seq = init_states.clone()
        actions_seq = init_actions.clone()
        predictions = []
        for h in range(future_actions.shape[1]):
            pred = self.forward(states_seq, actions_seq)
            predictions.append(pred)
            states_seq = torch.cat([states_seq[:, 1:], pred.unsqueeze(1)], dim=1)
            actions_seq = torch.cat([actions_seq[:, 1:], future_actions[:, h:h+1]], dim=1)
        return torch.stack(predictions, dim=1)


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerWorldModel(nn.Module):
    """基于Transformer的世界模型"""

    def __init__(self, state_dim=28, action_dim=7, d_model=128, nhead=4, n_layers=3):
        super().__init__()
        self.encoder = nn.Linear(state_dim + action_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )
        self.state_dim = state_dim

    def forward(self, states, actions):
        # 对齐长度
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device, dtype=actions.dtype)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        x = self.pos_enc(x)
        # 因果掩码
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        x = x[:, -1, :]
        delta_s = self.decoder(x)
        return states[:, -1, :] + delta_s

    def predict_trajectory(self, init_states, init_actions, future_actions):
        states_seq = init_states.clone()
        actions_seq = init_actions.clone()
        predictions = []
        for h in range(future_actions.shape[1]):
            pred = self.forward(states_seq, actions_seq)
            predictions.append(pred)
            states_seq = torch.cat([states_seq[:, 1:], pred.unsqueeze(1)], dim=1)
            actions_seq = torch.cat([actions_seq[:, 1:], future_actions[:, h:h+1]], dim=1)
        return torch.stack(predictions, dim=1)


class SimpleSSM(nn.Module):
    """普通对角SSM: 随机初始化, 无S4D结构化参数化"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_real = nn.Parameter(torch.randn(d_model, d_state) * 0.1 - 1.0)
        self.A_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, x):
        batch, L, D = x.shape
        N = self.d_state
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.A_real) + 1j * self.A_imag
        dtA = dt.unsqueeze(-1) * A
        # 计算卷积核 K: (D, L)
        K = torch.zeros(D, L, device=x.device, dtype=x.dtype)
        powers = torch.ones(D, N, device=x.device, dtype=x.dtype)
        for t in range(L):
            K[:, t] = (self.B * self.C * powers).real.sum(-1)
            powers = powers * dtA
        K = K * dt.unsqueeze(-1)
        # FFT卷积: x(B,L,D) conv K(D,L) → 每个通道独立卷积
        x_f = torch.fft.rfft(x, 2*L, dim=1)              # (B, L+1, D)
        K_f = torch.fft.rfft(K.T, 2*L, dim=0).unsqueeze(0)  # (1, L+1, D)
        out = torch.fft.irfft(x_f * K_f, 2*L, dim=1)[:, :L, :]  # (B, L, D)
        out = out + x * self.D.unsqueeze(0).unsqueeze(0)
        return out


class SimpleSSMWorldModel(nn.Module):
    """普通SSM世界模型 (无S4D结构化初始化, 无门控)"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.backbone = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                SimpleSSM(d_model, d_state),
                nn.Linear(d_model, d_model),
            ) for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            h = h + block(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])


class MLPWorldModel(nn.Module):
    """MLP世界模型: 无时序建模能力, 作为基线"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        input_dim = state_dim + action_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, state_dim)]
        self.net = nn.Sequential(*layers)
        self.state_dim = state_dim

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = x[:, -1, :]  # 只用最后一步
        delta_s = self.net(x)
        return states[:, -1, :] + delta_s


class TCNWorldModel(nn.Module):
    """时间卷积网络世界模型: 因果卷积做序列建模"""
    def __init__(self, state_dim, action_dim, d_model=128, n_layers=2, kernel_size=3):
        super().__init__()
        self.encoder = nn.Linear(state_dim + action_dim, d_model)
        self.tcn_blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation),
            ))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim)
        )
        self.state_dim = state_dim

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)  # (B, L, D)
        h = h.transpose(1, 2)  # (B, D, L) for Conv1d
        for block in self.tcn_blocks:
            residual = h
            h = block(h)
            h = h[:, :, :residual.shape[2]]  # 因果截断
            h = h + residual
        h = h.transpose(1, 2)  # (B, L, D)
        delta_s = self.decoder(h[:, -1, :])
        return states[:, -1, :] + delta_s
