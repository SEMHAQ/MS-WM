"""结构化SSM世界模型
创新点：利用状态空间的物理结构

核心思想：
1. 状态分解：位置(慢变化)、速度(快变化)、力(瞬时变化)
2. 分别处理：每个部分用不同的SSM
3. 自适应融合：根据输入动态组合
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
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
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
# 基线：简单SSM
# ============================================================
class SimpleSSM(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=1):
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
# 结构化SSM世界模型
# ============================================================
class StructuredWM(nn.Module):
    """结构化SSM世界模型

    创新点：
    1. 状态分解：位置、速度、力
    2. 分别处理：不同SSM参数
    3. 自适应融合

    优势：
    - 利用物理结构
    - 不同维度用不同处理
    - 有理论依据
    """
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=1):
        super().__init__()
        self.state_dim = state_dim
        self.pos_dim = state_dim // 3
        self.vel_dim = state_dim // 3
        self.force_dim = state_dim - 2 * (state_dim // 3)

        # 位置分支（慢变化）- 用较大的d_state
        self.pos_encoder = nn.Sequential(
            nn.Linear(self.pos_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.pos_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state * 2)})
            for _ in range(n_layers)
        ])
        self.pos_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.pos_dim)
        )

        # 速度分支（快变化）- 用标准d_state
        self.vel_encoder = nn.Sequential(
            nn.Linear(self.vel_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.vel_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])
        self.vel_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.vel_dim)
        )

        # 力分支（瞬时变化）- 用较小的d_state
        self.force_encoder = nn.Sequential(
            nn.Linear(self.force_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.force_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)})
            for _ in range(n_layers)
        ])
        self.force_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.force_dim)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)

        # 分解状态
        pos = states[:, :, :self.pos_dim]  # 位置
        vel = states[:, :, self.pos_dim:self.pos_dim+self.vel_dim]  # 速度
        force = states[:, :, self.pos_dim+self.vel_dim:]  # 力

        # 位置分支
        pos_h = self.pos_encoder(torch.cat([pos, actions], dim=-1))
        for block in self.pos_ssm:
            residual = pos_h; x_norm = block['norm'](pos_h); pos_h = residual + block['ssm'](x_norm)
        pos_pred = self.pos_decoder(pos_h[:, -1, :])

        # 速度分支
        vel_h = self.vel_encoder(torch.cat([vel, actions], dim=-1))
        for block in self.vel_ssm:
            residual = vel_h; x_norm = block['norm'](vel_h); vel_h = residual + block['ssm'](x_norm)
        vel_pred = self.vel_decoder(vel_h[:, -1, :])

        # 力分支
        force_h = self.force_encoder(torch.cat([force, actions], dim=-1))
        for block in self.force_ssm:
            residual = force_h; x_norm = block['norm'](force_h); force_h = residual + block['ssm'](x_norm)
        force_pred = self.force_decoder(force_h[:, -1, :])

        # 拼接
        return states[:, -1, :] + torch.cat([pos_pred, vel_pred, force_pred], dim=-1)

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

    RESULTS_FILE = 'experiments/structured_wm.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    configs = {
        'StructuredWM': lambda: StructuredWM(348, 17, d_model=128, d_state=16, n_layers=1),
        'SimpleSSM': lambda: SimpleSSM(348, 17, d_model=128, d_state=16, n_layers=1),
    }

    print('\n' + '='*60, flush=True)
    print('结构化SSM世界模型实验', flush=True)
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
    print('{:<20} {:<10} {:<10} {:<10}'.format('模型', 'MSE', 'R²', '参数(M)'))
    print('-'*50)

    for name in configs:
        if name in results:
            r = results[name]
            print('{:<20} {:<10.4f} {:<10.4f} {:<10.3f}'.format(name, r['mse'], r['r2'], r['params_m']))

    print('\nDone!', flush=True)
