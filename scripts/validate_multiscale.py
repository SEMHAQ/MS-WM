"""验证MultiScale模型在所有数据集上的表现"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
BS = 256
T = 32

print(f'Device: {device}', flush=True)

def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    eps = []
    for i, f in enumerate(fs):
        eps.append((np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']))
        if (i+1) % 200 == 0: print(f'    {i+1}/{len(fs)}...', flush=True)
    print(f'    {len(fs)} episodes loaded', flush=True)
    return eps

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

class MultiScaleDynamicsModel(nn.Module):
    """多尺度动力学模型"""
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
        # 快速SSM（短程依赖）
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

def train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed, epochs=EPOCHS):
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
        if pat >= 20: break

    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

    return {'mse': round(mse, 6), 'r2': round(r2, 4), 'params': round(params, 3), 'best_epoch': best_ep}

# Dataset configs
datasets = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
    'ant': {'dir': 'data/ant', 'sd': 105, 'ad': 8},
    'walker2d': {'dir': 'data/walker2d', 'sd': 17, 'ad': 6},
}

# 加载已有结果
RESULTS_FILE = 'experiments/multiscale_validation.json'
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
else:
    results = {}

# 验证所有数据集
print('\n' + '='*60, flush=True)
print('MultiScale模型验证', flush=True)
print('='*60, flush=True)

for ds_name, ds_cfg in datasets.items():
    print(f'\n{ds_name}:', flush=True)

    # 检查是否已有完整结果
    if ds_name in results and len(results[ds_name]) >= len(SEEDS):
        print(f'  已有完整结果，跳过', flush=True)
        continue

    # 加载数据
    print(f'  加载数据...', flush=True)
    eps_tr = load_eps(ds_cfg['dir'], 'train')
    eps_vl = load_eps(ds_cfg['dir'], 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    if ds_name not in results:
        results[ds_name] = {}

    # 跑5个seed
    for seed in SEEDS:
        seed_key = f'seed{seed}'
        if seed_key in results[ds_name]:
            print(f'  seed={seed} 已有，跳过', flush=True)
            continue

        print(f'  seed={seed}...', end=' ', flush=True)
        model = MultiScaleDynamicsModel(ds_cfg['sd'], ds_cfg['ad'], d_model=128, d_state=16, n_layers=2)
        r = train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed)
        results[ds_name][seed_key] = r
        print(f'MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}', flush=True)

        # 保存中间结果
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

# 打印结果汇总
print('\n' + '='*60, flush=True)
print('MultiScale验证结果汇总', flush=True)
print('='*60, flush=True)
print('{:<12} {:<15} {:<15} {:<10}'.format('数据集', 'MSE', 'R²', '参数(M)'))
print('-'*55)

for ds_name in datasets:
    if ds_name in results:
        valid = [results[ds_name][s] for s in results[ds_name] if 'mse' in results[ds_name][s]]
        if valid:
            mses = [r['mse'] for r in valid]
            r2s = [r['r2'] for r in valid]
            params = valid[0]['params']
            print('{:<12} {:.4f}±{:.4f}  {:.4f}±{:.4f}  {:.3f}'.format(
                ds_name,
                np.mean(mses), np.std(mses),
                np.mean(r2s), np.std(r2s),
                params
            ))
        else:
            print('{:<12} ---'.format(ds_name))
    else:
        print('{:<12} ---'.format(ds_name))

print('\nDone!', flush=True)
