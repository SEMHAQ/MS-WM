import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelectiveSSM(nn.Module):
    """选择性状态空间模型 (Mamba风格)"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D因果卷积
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner
        )

        # SSM参数 (输入依赖)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner)

        # A参数 (对数空间初始化)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D参数 (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # 双分支: 一支过SSM, 一支做门控
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # 因果卷积
        x_conv = rearrange(x_branch, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)

        # 计算输入依赖的SSM参数
        x_ssm = x_conv
        proj = self.x_proj(x_ssm)  # (B, L, 2*d_state + 1)
        B_param, C_param, dt = proj.split([self.d_state, self.d_state, 1], dim=-1)

        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # A矩阵
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # 选择性SSM递推 (并行扫描近似)
        y = self._selective_scan(x_ssm, A, B_param, C_param, dt)

        # skip connection + 门控
        y = y + x_branch * self.D
        y = y * F.silu(z)

        return self.out_proj(y)

    def _selective_scan(self, u, A, B, C, dt):
        """
        选择性扫描: 离散化并递推
        u: (B, L, D) - 输入
        A: (D, N) - 状态矩阵
        B: (B, L, N) - 输入矩阵
        C: (B, L, N) - 输出矩阵
        dt: (B, L, D) - 时间步
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # 离散化
        # deltaA: (B, L, D, N)
        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # deltaB_u: (B, L, D, N)
        deltaB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, 1, N) -> (B, L, D, N)
        deltaB_u = deltaB * u.unsqueeze(-1)

        # 递推
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, D)


class MambaBlock(nn.Module):
    """单个Mamba块: SSM + LayerNorm + 残差"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class SSMWorldModel(nn.Module):
    """
    SSM世界模型 (SSM-WM)

    用于人形机器人状态预测:
    输入: 历史状态序列 s_0:T 和动作序列 a_0:T-1
    输出: 预测的下一状态 s_{t+1}
    """

    def __init__(
        self,
        state_dim: int = 28,      # 机器人状态维度 (关节角度等)
        action_dim: int = 7,      # 动作维度
        d_model: int = 128,       # 隐空间维度
        d_state: int = 16,        # SSM状态维度
        n_layers: int = 4,        # Mamba块层数
        d_conv: int = 4,          # 卷积核大小
        expand: int = 2,          # 扩展因子
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # 状态-动作编码器
        self.encoder = nn.Linear(state_dim + action_dim, d_model)

        # Mamba主干
        self.backbone = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        # 归一化
        self.norm = nn.LayerNorm(d_model)

        # 状态解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        """
        单步预测:
        states:  (B, T, state_dim) - 历史状态序列
        actions: (B, T-1, action_dim) - 历史动作序列 (比states少1步)
        返回: (B, state_dim) - 预测的下一状态
        """
        # 对齐长度: actions比states少1步, 前面补零
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(
                states.shape[0], pad_len, actions.shape[-1],
                device=actions.device, dtype=actions.dtype
            )
            actions = torch.cat([pad, actions], dim=1)

        # 拼接状态和动作
        x = torch.cat([states, actions], dim=-1)  # (B, T, state_dim + action_dim)

        # 编码
        x = self.encoder(x)  # (B, T, d_model)

        # SSM主干
        for block in self.backbone:
            x = block(x)

        x = self.norm(x)

        # 取最后时刻的输出
        x = x[:, -1, :]  # (B, d_model)

        # 解码 + 残差连接
        delta_s = self.decoder(x)  # (B, state_dim)
        pred_state = states[:, -1, :] + delta_s  # 残差: 基于上一状态预测变化量

        return pred_state

    def predict_trajectory(
        self,
        init_states: torch.Tensor,
        init_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ):
        """
        多步展开预测:
        init_states:    (B, T, state_dim) - 初始状态序列
        init_actions:   (B, T, action_dim) - 初始动作序列
        future_actions: (B, H, action_dim) - 未来动作序列
        返回: (B, H, state_dim) - 预测的未来状态轨迹
        """
        states_seq = init_states.clone()
        actions_seq = init_actions.clone()
        predictions = []

        for h in range(future_actions.shape[1]):
            # 预测下一状态
            pred = self.forward(states_seq, actions_seq)  # (B, state_dim)
            predictions.append(pred)

            # 更新序列 (滑动窗口)
            states_seq = torch.cat([states_seq[:, 1:], pred.unsqueeze(1)], dim=1)
            actions_seq = torch.cat([actions_seq[:, 1:], future_actions[:, h:h+1]], dim=1)

        return torch.stack(predictions, dim=1)  # (B, H, state_dim)
