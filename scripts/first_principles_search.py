"""从第一性原理出发的系统性探索
核心问题：什么是世界模型的最优架构？

创新方向：
1. 自适应计算 - 不同维度用不同策略
2. 多尺度建模 - 快/慢动力学分开处理
3. 物理先验 - 嵌入守恒律
4. 几何结构 - 利用状态空间几何
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 50
BS = 256
T = 32

print(f'Device: {device}', flush=True)

# ============================================================
# 基线模型定义
# ============================================================

class BaselineSSM(nn.Module):
    """基线：纯SSM（S4D风格）"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            self.backbone.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
                'gate': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)),
            }))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            residual = x
            x_norm = block['norm'](x)
            ssm_out = block['ssm'](x_norm)
            g = torch.sigmoid(block['gate'](x_norm))
            x = residual + g * ssm_out + (1-g) * x_norm
            x = x + block['ffn'](block['norm'](x))
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

class BaselineTransformer(nn.Module):
    """基线：纯Transformer"""
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, n_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

class LightweightSSM(nn.Module):
    """轻量级SSM（去掉FFN）"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            self.backbone.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
                'gate': nn.Linear(d_model, d_model),
            }))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            residual = x
            x_norm = block['norm'](x)
            ssm_out = block['ssm'](x_norm)
            g = torch.sigmoid(block['gate'](x_norm))
            x = residual + g * ssm_out + (1-g) * x_norm
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs]

def stats(eps):
    a = np.concatenate([s for s,_ in eps])
    return a.mean(0), a.std(0)

def make_data(eps, T, mean, std):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        if len(st) < T+1: continue
        sn = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(sn[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

def train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed=SEED, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 15: break

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

    # 测量推理时间
    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        inf_time = (time.perf_counter() - t0) / 100 * 1000

    return {
        'mse': round(mse, 6),
        'r2': round(r2, 4),
        'params': round(params, 3),
        'inf_time': round(inf_time, 2),
        'train_time': round(elapsed, 1),
        'best_epoch': best_ep
    }

# ============================================================
# 创新架构定义
# ============================================================

class AdaptiveDimensionModel(nn.Module):
    """创新1：维度自适应模型
    核心思想：不同状态维度有不同的预测难度，应该用不同的策略
    - 简单维度（如位置）：用简单线性映射
    - 困难维度（如接触力）：用复杂非线性建模
    - 冗余维度（如对称关节）：共享参数
    """
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # 维度难度评估器
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim),
            nn.Sigmoid(),  # 0-1表示难度
        )
        # 简单分支（线性）
        self.simple_branch = nn.Linear(d_model, state_dim)
        # 困难分支（SSM）
        self.complex_branch = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
            }) for _ in range(n_layers)
        ])
        self.complex_decoder = nn.Linear(d_model, state_dim)
        # 融合权重
        self.fusion_weight = nn.Parameter(torch.ones(state_dim) * 0.5)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)

        # 评估每个维度的难度
        difficulty = self.difficulty_estimator(x)  # (B, L, state_dim)
        avg_difficulty = difficulty.mean(dim=1)  # (B, state_dim)

        # 编码
        h = self.encoder(x)

        # 简单分支
        simple_pred = self.simple_branch(h[:, -1, :])

        # 困难分支
        complex_h = h
        for block in self.complex_branch:
            residual = complex_h
            x_norm = block['norm'](complex_h)
            ssm_out = block['ssm'](x_norm)
            complex_h = residual + ssm_out
        complex_pred = self.complex_decoder(complex_h[:, -1, :])

        # 自适应融合
        alpha = torch.sigmoid(self.fusion_weight)  # (state_dim,)
        # 高难度维度用复杂分支，低难度用简单分支
        weight = avg_difficulty * alpha  # (B, state_dim)
        pred = weight * complex_pred + (1 - weight) * simple_pred

        return states[:, -1, :] + pred

class MultiScaleDynamicsModel(nn.Module):
    """创新2：多尺度动力学模型
    核心思想：不同物理量变化速度不同
    - 位置变化慢 → 用慢速SSM
    - 速度变化快 → 用快速SSM
    - 力变化很快 → 用局部注意力
    """
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # 慢速SSM（长程依赖）
        self.slow_ssm = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
            }) for _ in range(n_layers)
        ])
        # 快速SSM（短程依赖，更小的状态维度）
        self.fast_ssm = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state // 2),
            }) for _ in range(n_layers)
        ])
        # 局部注意力（瞬时依赖）
        self.local_attn = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'conv': nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            }) for _ in range(n_layers)
        ])
        # 多尺度融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)

        # 慢速分支
        slow_h = h
        for block in self.slow_ssm:
            residual = slow_h
            x_norm = block['norm'](slow_h)
            slow_h = residual + block['ssm'](x_norm)

        # 快速分支
        fast_h = h
        for block in self.fast_ssm:
            residual = fast_h
            x_norm = block['norm'](fast_h)
            fast_h = residual + block['ssm'](x_norm)

        # 局部分支
        local_h = h
        for block in self.local_attn:
            residual = local_h
            x_norm = block['norm'](local_h)
            local_h = residual + block['conv'](x_norm.transpose(1,2)).transpose(1,2)

        # 融合三个分支
        fused = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
        pred = self.fusion(fused)

        return states[:, -1, :] + pred

class PhysicsInformedModel(nn.Module):
    """创新3：物理信息模型
    核心思想：嵌入物理先验
    - 能量守恒：预测的能量变化应该小
    - 平滑性：状态变化应该连续
    - 因果性：动作导致状态变化
    """
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
            }) for _ in range(n_layers)
        ])
        self.decoder = nn.Linear(d_model, state_dim)
        # 能量估计器（用于正则化）
        self.energy_estimator = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # 因果掩码（学习哪些动作维度影响哪些状态维度）
        self.causal_mask = nn.Parameter(torch.ones(action_dim, state_dim) * 0.5)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)

        for block in self.backbone:
            residual = h
            x_norm = block['norm'](h)
            h = residual + block['ssm'](x_norm)

        pred = self.decoder(h[:, -1, :])
        return states[:, -1, :] + pred

    def physics_loss(self, states, pred_states, actions):
        """物理正则化损失"""
        # 1. 平滑性损失：状态变化应该连续
        if states.shape[1] > 1:
            velocity = states[:, 1:, :] - states[:, :-1, :]
            pred_velocity = pred_states - states[:, -1, :]
            smoothness_loss = nn.MSELoss()(pred_velocity, velocity[:, -1, :])
        else:
            smoothness_loss = torch.tensor(0.0, device=states.device)

        # 2. 能量守恒损失（近似）
        energy_before = self.energy_estimator(states[:, -1, :])
        energy_after = self.energy_estimator(pred_states)
        energy_loss = torch.mean((energy_after - energy_before) ** 2)

        # 3. 因果性损失：动作应该影响状态
        causal_mask = torch.sigmoid(self.causal_mask)
        action_effect = torch.mean(causal_mask)
        causal_loss = -action_effect  # 鼓励动作有影响

        return smoothness_loss + 0.1 * energy_loss + 0.01 * causal_loss

class HierarchicalModel(nn.Module):
    """创新4：层次化模型
    核心思想：分层处理不同抽象级别
    - 低层：原始状态处理
    - 中层：特征提取
    - 高层：决策/预测
    """
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        # 低层：状态编码
        self.low_level = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # 中层：特征提取（SSM）
        self.mid_level = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
            }) for _ in range(n_layers)
        ])
        # 高层：预测（更简单的结构）
        self.high_level = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, state_dim),
        )
        # 跳跃连接
        self.skip = nn.Linear(state_dim + action_dim, state_dim)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)

        # 跳跃连接
        skip_pred = self.skip(x[:, -1, :])

        # 低层编码
        h = self.low_level(x)

        # 中层特征提取
        for block in self.mid_level:
            residual = h
            x_norm = block['norm'](h)
            h = residual + block['ssm'](x_norm)

        # 高层预测
        pred = self.high_level(h[:, -1, :])

        # 融合
        return states[:, -1, :] + pred + 0.1 * skip_pred

class ResidualDynamicsModel(nn.Module):
    """创新5：残差动力学模型
    核心思想：预测残差而不是绝对状态
    - 大部分状态变化很小（惯性）
    - 只有少数维度变化大（受力）
    - 用残差连接捕捉小变化
    """
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
            }) for _ in range(n_layers)
        ])
        # 残差预测器
        self.residual_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )
        # 状态变化幅度估计器
        self.change_estimator = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)

        for block in self.backbone:
            residual = h
            x_norm = block['norm'](h)
            h = residual + block['ssm'](x_norm)

        # 预测残差
        residual = self.residual_predictor(h[:, -1, :])

        # 估计变化幅度
        change_magnitude = self.change_estimator(h[:, -1, :])  # (B, 1)

        # 缩放残差
        scaled_residual = residual * change_magnitude

        return states[:, -1, :] + scaled_residual

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    # 加载数据
    print('\n加载Humanoid数据...', flush=True)
    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    sd, ad = 348, 17
    os.makedirs('experiments', exist_ok=True)

    # 定义所有架构
    architectures = {
        # 基线
        'Baseline-SSM': lambda: BaselineSSM(sd, ad, d_model=128, d_state=16, n_layers=4),
        'Baseline-Trans': lambda: BaselineTransformer(sd, ad, d_model=128, nhead=4, n_layers=4),
        'LightweightSSM-L2': lambda: LightweightSSM(sd, ad, d_model=128, d_state=16, n_layers=2),
        # 创新架构
        'AdaptiveDimension': lambda: AdaptiveDimensionModel(sd, ad, d_model=128, d_state=16, n_layers=2),
        'MultiScale': lambda: MultiScaleDynamicsModel(sd, ad, d_model=128, d_state=16, n_layers=2),
        'PhysicsInformed': lambda: PhysicsInformedModel(sd, ad, d_model=128, d_state=16, n_layers=2),
        'Hierarchical': lambda: HierarchicalModel(sd, ad, d_model=128, d_state=16, n_layers=2),
        'ResidualDynamics': lambda: ResidualDynamicsModel(sd, ad, d_model=128, d_state=16, n_layers=2),
    }

    # 加载已有结果
    RESULTS_FILE = 'experiments/first_principles_search.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # 测试所有架构
    print('\n' + '='*80, flush=True)
    print('第一性原理架构探索', flush=True)
    print('='*80, flush=True)

    for name, model_fn in architectures.items():
        if name in results:
            print(f'\n{name}: 已有结果，跳过', flush=True)
            continue

        print(f'\n{name}:', flush=True)
        try:
            model = model_fn()
            r = train_eval(model, Xs, Xa, Y, Xv, Xav, Yv)
            results[name] = r
            print(f'  MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}, Params={r["params"]:.3f}M, InfTime={r["inf_time"]:.2f}ms', flush=True)

            # 保存中间结果
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            results[name] = {'error': str(e)}

    # 打印结果汇总
    print('\n' + '='*80, flush=True)
    print('第一性原理架构探索结果汇总', flush=True)
    print('='*80, flush=True)
    print('{:<25} {:<10} {:<10} {:<10} {:<10}'.format('架构', 'MSE', 'R²', '参数(M)', '推理(ms)'))
    print('-'*70)

    # 按MSE排序
    valid_results = {k: v for k, v in results.items() if 'mse' in v}
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['mse'])

    for name, r in sorted_results:
        print('{:<25} {:<10.4f} {:<10.4f} {:<10.3f} {:<10.2f}'.format(
            name, r['mse'], r['r2'], r['params'], r['inf_time']
        ))

    print('\nDone!', flush=True)
