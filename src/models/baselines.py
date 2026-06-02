"""
基线模型: LSTM世界模型 和 Transformer世界模型
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
