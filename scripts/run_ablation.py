"""MS-WM消融实验 - 测试不同配置"""
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

class MultiScaleModel(nn.Module):
    """多尺度动力学模型 - 可调参数版本"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5, fusion_type='gate'):
        super().__init__()
        self.state_dim = state_dim
        self.fusion_type = fusion_type
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.slow_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])
        self.fast_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)})
            for _ in range(n_layers)
        ])
        self.local_attn = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model)})
            for _ in range(n_layers)
        ])
        if fusion_type == 'gate':
            self.fusion_gate = nn.Sequential(nn.Linear(d_model * 3, 3), nn.Softmax(dim=-1))
            self.fusion_proj = nn.Linear(d_model, state_dim)
        elif fusion_type == 'concat':
            self.fusion = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        slow_h = h
        for block in self.slow_ssm:
            residual = slow_h; x_norm = block['norm'](slow_h); slow_h = residual + block['ssm'](x_norm)
        fast_h = h
        for block in self.fast_ssm:
            residual = fast_h; x_norm = block['norm'](fast_h); fast_h = residual + block['ssm'](x_norm)
        local_h = h
        for block in self.local_attn:
            residual = local_h; x_norm = block['norm'](local_h); local_h = residual + block['conv'](x_norm.transpose(1,2)).transpose(1,2)
        if self.fusion_type == 'gate':
            features = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
            gate = self.fusion_gate(features)
            stacked = torch.stack([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=1)
            fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
            pred = self.fusion_proj(fused)
        elif self.fusion_type == 'concat':
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
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0
    for ep in range(epochs):
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
    return {'mse': round(mse, 6), 'r2': round(r2, 4), 'params': round(params, 3), 'best_epoch': best_ep}

# 消融配置
ablation_configs = {
    'default': {'d_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'},
    'no_fast_ssm': {'d_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'},
    'no_local_attn': {'d_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'},
    'concat_fusion': {'d_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'concat'},
    'd_model_64': {'d_model': 64, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'},
    'd_model_128': {'d_model': 128, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'},
    'n_layers_2': {'d_model': 96, 'd_state': 8, 'n_layers': 2, 'window_size': 5, 'fusion_type': 'gate'},
    'd_state_16': {'d_model': 96, 'd_state': 16, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'},
}

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
    RESULTS_FILE = 'experiments/ablation_results.json'

    # 加载已有结果
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # 运行消融实验
    print('\n' + '='*60, flush=True)
    print('MS-WM消融实验', flush=True)
    print('='*60, flush=True)

    for config_name, config in ablation_configs.items():
        if config_name in results and len(results[config_name]) >= len(SEEDS):
            print(f'\n{config_name}: 已有完整结果，跳过', flush=True)
            continue

        print(f'\n{config_name}:', flush=True)
        if config_name not in results:
            results[config_name] = {}

        for seed in SEEDS:
            seed_key = f'seed{seed}'
            if seed_key in results[config_name]:
                print(f'  seed={seed} 已有，跳过', flush=True)
                continue

            print(f'  seed={seed}...', end=' ', flush=True)
            model = MultiScaleModel(sd, ad, **config)
            r = train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed)
            results[config_name][seed_key] = r
            print(f'MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}', flush=True)

            # 保存中间结果
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 打印结果汇总
    print('\n' + '='*60, flush=True)
    print('消融实验结果汇总', flush=True)
    print('='*60, flush=True)
    print('{:<20} {:<12} {:<12} {:<10}'.format('配置', 'MSE(×10⁻²)', 'R²', '参数(M)'))
    print('-'*55)

    for config_name in ablation_configs:
        if config_name in results:
            valid = [results[config_name][s] for s in results[config_name] if 'mse' in results[config_name][s]]
            if valid:
                mses = [r['mse'] for r in valid]
                r2s = [r['r2'] for r in valid]
                params = valid[0]['params']
                print('{:<20} {:.2f}±{:.2f}    {:.4f}    {:.3f}'.format(
                    config_name,
                    np.mean(mses)*100, np.std(mses)*100,
                    np.mean(r2s),
                    params
                ))
            else:
                print('{:<20} ---'.format(config_name))
        else:
            print('{:<20} ---'.format(config_name))

    print('\nDone!', flush=True)
