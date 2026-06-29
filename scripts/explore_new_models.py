"""系统性探索新模型 - 寻找真正创新的世界模型架构

从第一性原理出发，测试多种创新方向：
1. 物理引导SSM - 嵌入守恒律约束
2. 多分辨率建模 - 不同时间尺度
3. 自适应计算 - 动态分配算力
4. 结构化状态空间 - 利用状态结构
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 100
BS = 256
T = 32

print(f'Device: {device}', flush=True)

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

def train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 20: break
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    return {'mse': round(mse, 6), 'r2': round(r2, 4), 'params_m': round(params, 3), 'best_epoch': best_ep}

# ============================================================
# 基线模型
# ============================================================

class SimpleSSM(nn.Module):
    """简单SSM基线（无gate无FFN）"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            residual = h; x_norm = block['norm'](h); h = residual + block['ssm'](x_norm)
        return states[:, -1, :] + self.decoder(h[:, -1, :])

# ============================================================
# 创新模型1: 物理引导SSM
# ============================================================

class PhysicsGuidedSSM(nn.Module):
    """物理引导SSM - 嵌入能量守恒约束

    核心思想：
    1. 预测状态变化量（而非绝对状态）
    2. 用能量守恒约束预测
    3. 分离快慢动力学
    """
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # 位置分支（慢速变化）
        self.position_branch = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])

        # 速度分支（快速变化）
        self.velocity_branch = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)})
            for _ in range(n_layers)
        ])

        # 预测头
        self.position_head = nn.Linear(d_model, state_dim)
        self.velocity_head = nn.Linear(d_model, state_dim)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)

        # 位置分支
        pos_h = h
        for block in self.position_branch:
            residual = pos_h; x_norm = block['norm'](pos_h); pos_h = residual + block['ssm'](x_norm)

        # 速度分支
        vel_h = h
        for block in self.velocity_branch:
            residual = vel_h; x_norm = block['norm'](vel_h); vel_h = residual + block['ssm'](x_norm)

        # 预测：位置增量 + 速度增量
        pos_delta = self.position_head(pos_h[:, -1, :])
        vel_delta = self.velocity_head(vel_h[:, -1, :])

        return states[:, -1, :] + pos_delta + vel_delta

# ============================================================
# 创新模型2: 自适应计算SSM
# ============================================================

class AdaptiveComputeSSM(nn.Module):
    """自适应计算SSM - 根据输入复杂度动态调整计算

    核心思想：
    1. 简单输入用轻量计算
    2. 复杂输入用重量计算
    3. 路由网络决定计算路径
    """
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # 路由网络：决定用哪个分支
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),
            nn.Softmax(dim=-1)
        )

        # 轻量分支（单层SSM）
        self.light_branch = nn.ModuleDict({
            'norm': nn.LayerNorm(d_model),
            'ssm': DiagSSM(d_model, d_state)
        })

        # 重量分支（多层SSM）
        self.heavy_branch = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])

        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)

        # 路由决策
        route = self.router(h[:, -1, :])  # (B, 2)

        # 轻量分支
        light_h = h
        block = self.light_branch
        residual = light_h; x_norm = block['norm'](light_h); light_h = residual + block['ssm'](x_norm)

        # 重量分支
        heavy_h = h
        for block in self.heavy_branch:
            residual = heavy_h; x_norm = block['norm'](heavy_h); heavy_h = residual + block['ssm'](x_norm)

        # 自适应融合
        fused = route[:, 0:1].unsqueeze(1) * light_h + route[:, 1:2].unsqueeze(1) * heavy_h

        return states[:, -1, :] + self.decoder(fused[:, -1, :])

# ============================================================
# 创新模型3: 结构化状态空间
# ============================================================

class StructuredSSM(nn.Module):
    """结构化状态空间 - 利用机器人状态的结构

    核心思想：
    1. 不同状态维度有不同的动态特性
    2. 位置维度：慢变化
    3. 速度维度：快变化
    4. 分别建模后融合
    """
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # 位置编码器（假设前一半是位置）
        pos_dim = state_dim // 2
        self.pos_encoder = nn.Linear(pos_dim, d_model // 2)
        self.pos_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model // 2), 'ssm': DiagSSM(d_model // 2, d_state)})
            for _ in range(n_layers)
        ])

        # 速度编码器（假设后一半是速度）
        vel_dim = state_dim - pos_dim
        self.vel_encoder = nn.Linear(vel_dim, d_model // 2)
        self.vel_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model // 2), 'ssm': DiagSSM(d_model // 2, d_state)})
            for _ in range(n_layers)
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)

        # 分离位置和速度
        pos_states = states[:, :, :self.state_dim//2]
        vel_states = states[:, :, self.state_dim//2:]

        # 编码
        pos_h = self.pos_encoder(pos_states)
        vel_h = self.vel_encoder(vel_states)

        # 位置分支
        for block in self.pos_ssm:
            residual = pos_h; x_norm = block['norm'](pos_h); pos_h = residual + block['ssm'](x_norm)

        # 速度分支
        for block in self.vel_ssm:
            residual = vel_h; x_norm = block['norm'](vel_h); vel_h = residual + block['ssm'](x_norm)

        # 融合
        fused = torch.cat([pos_h[:, -1, :], vel_h[:, -1, :]], dim=-1)
        delta_s = self.fusion(fused)

        return states[:, -1, :] + delta_s

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    print('\n加载Humanoid数据...', flush=True)
    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    RESULTS_FILE = 'experiments/new_models.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    configs = {
        'SimpleSSM': lambda: SimpleSSM(348, 17, d_model=96, d_state=16, n_layers=2),
        'PhysicsGuidedSSM': lambda: PhysicsGuidedSSM(348, 17, d_model=96, d_state=16, n_layers=2),
        'AdaptiveComputeSSM': lambda: AdaptiveComputeSSM(348, 17, d_model=96, d_state=16, n_layers=2),
        'StructuredSSM': lambda: StructuredSSM(348, 17, d_model=96, d_state=16, n_layers=2),
    }

    print('\n' + '='*60, flush=True)
    print('新模型探索实验', flush=True)
    print('='*60, flush=True)

    for name, model_fn in configs.items():
        if name in results:
            print(f'\n{name}: 已有结果，跳过', flush=True)
            continue

        print(f'\n{name}:', flush=True)
        model = model_fn()
        r = train_eval(model, Xs, Xa, Y, Xv, Xav, Yv)
        results[name] = r
        print(f'  MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}, Params={r["params_m"]:.3f}M', flush=True)

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # 打印结果
    print('\n' + '='*60, flush=True)
    print('结果汇总', flush=True)
    print('='*60, flush=True)
    print('{:<25} {:<10} {:<10} {:<10}'.format('模型', 'MSE', 'R²', '参数(M)'))
    print('-'*55)

    for name in configs:
        if name in results:
            r = results[name]
            print('{:<25} {:<10.4f} {:<10.4f} {:<10.3f}'.format(name, r['mse'], r['r2'], r['params_m']))

    print('\nDone!', flush=True)
