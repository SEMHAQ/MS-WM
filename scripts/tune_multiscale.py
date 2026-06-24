"""MultiScale模型调优 - 系统性搜索最优配置"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time, itertools
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 80
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

class MultiScaleModel(nn.Module):
    """多尺度动力学模型 - 可调参数版本"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2,
                 window_size=3, fusion_type='concat'):
        super().__init__()
        self.state_dim = state_dim
        self.fusion_type = fusion_type

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
                'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size,
                                  padding=window_size//2, groups=d_model),
            }) for _ in range(n_layers)
        ])

        # 融合方式
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Linear(d_model, state_dim),
            )
        elif fusion_type == 'gate':
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 3, 3),
                nn.Softmax(dim=-1),
            )
            self.fusion_proj = nn.Linear(d_model, state_dim)
        elif fusion_type == 'attention':
            self.fusion_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.fusion_proj = nn.Linear(d_model, state_dim)

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

        # 融合
        if self.fusion_type == 'concat':
            fused = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
            pred = self.fusion(fused)
        elif self.fusion_type == 'gate':
            features = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
            gate = self.fusion_gate(features)  # (B, 3)
            stacked = torch.stack([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=1)  # (B, 3, D)
            fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)  # (B, D)
            pred = self.fusion_proj(fused)
        elif self.fusion_type == 'attention':
            stacked = torch.stack([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=1)  # (B, 3, D)
            fused, _ = self.fusion_attn(stacked, stacked, stacked)  # (B, 3, D)
            pred = self.fusion_proj(fused.mean(dim=1))

        return states[:, -1, :] + pred

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
        if pat >= 20: break

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

    return {
        'mse': round(mse, 6),
        'r2': round(r2, 4),
        'params': round(params, 3),
        'train_time': round(elapsed, 1),
        'best_epoch': best_ep
    }

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    # 加载所有数据集
    print('\n加载数据...', flush=True)
    datasets = {}
    for ds_name, ds_dir, sd, ad in [('humanoid', 'data/humanoid', 348, 17),
                                     ('ant', 'data/ant', 105, 8),
                                     ('walker2d', 'data/walker2d', 17, 6)]:
        print(f'\n{ds_name}:', flush=True)
        eps_tr = load_eps(ds_dir, 'train')
        eps_vl = load_eps(ds_dir, 'val')
        m, s = stats(eps_tr)
        Xs, Xa, Y = make_data(eps_tr, T, m, s)
        Xv, Xav, Yv = make_data(eps_vl, T, m, s)
        datasets[ds_name] = {'Xs': Xs, 'Xa': Xa, 'Y': Y, 'Xv': Xv, 'Xav': Xav, 'Yv': Yv, 'sd': sd, 'ad': ad}
        print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    os.makedirs('experiments', exist_ok=True)
    RESULTS_FILE = 'experiments/multiscale_tuning.json'

    # 加载已有结果
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # 定义搜索空间
    search_space = {
        'd_model': [64, 96, 128, 192],
        'd_state': [8, 16, 32],
        'n_layers': [1, 2, 3],
        'window_size': [3, 5, 7],
        'fusion_type': ['concat', 'gate', 'attention'],
    }

    # 生成所有配置组合
    configs = []
    for d_model in search_space['d_model']:
        for d_state in search_space['d_state']:
            for n_layers in search_space['n_layers']:
                for window_size in search_space['window_size']:
                    for fusion_type in search_space['fusion_type']:
                        configs.append({
                            'd_model': d_model,
                            'd_state': d_state,
                            'n_layers': n_layers,
                            'window_size': window_size,
                            'fusion_type': fusion_type,
                        })

    print(f'\n总共 {len(configs)} 种配置', flush=True)

    # 搜索
    print('\n' + '='*80, flush=True)
    print('MultiScale调优实验', flush=True)
    print('='*80, flush=True)

    best_config = None
    best_avg_mse = float('inf')

    for i, cfg in enumerate(configs):
        config_key = f"d{cfg['d_model']}_s{cfg['d_state']}_l{cfg['n_layers']}_w{cfg['window_size']}_{cfg['fusion_type']}"

        # 检查是否已有结果
        if config_key in results:
            # 计算平均MSE
            avg_mse = np.mean([results[config_key][ds]['mse'] for ds in datasets if ds in results[config_key]])
            if avg_mse < best_avg_mse:
                best_avg_mse = avg_mse
                best_config = cfg
            continue

        print(f'\n[{i+1}/{len(configs)}] {config_key}:', flush=True)

        config_results = {}
        for ds_name, ds_data in datasets.items():
            print(f'  {ds_name}...', end=' ', flush=True)
            try:
                model = MultiScaleModel(
                    ds_data['sd'], ds_data['ad'],
                    d_model=cfg['d_model'],
                    d_state=cfg['d_state'],
                    n_layers=cfg['n_layers'],
                    window_size=cfg['window_size'],
                    fusion_type=cfg['fusion_type']
                )
                r = train_eval(model, ds_data['Xs'], ds_data['Xa'], ds_data['Y'],
                              ds_data['Xv'], ds_data['Xav'], ds_data['Yv'])
                config_results[ds_name] = r
                print(f'MSE={r["mse"]:.4f}', flush=True)
            except Exception as e:
                print(f'ERROR: {e}', flush=True)
                config_results[ds_name] = {'error': str(e)}

        results[config_key] = config_results

        # 计算平均MSE
        valid_results = [config_results[ds] for ds in config_results if 'mse' in config_results[ds]]
        if valid_results:
            avg_mse = np.mean([r['mse'] for r in valid_results])
            if avg_mse < best_avg_mse:
                best_avg_mse = avg_mse
                best_config = cfg
                print(f'  *** 新最佳! 平均MSE={avg_mse:.4f}', flush=True)

        # 保存中间结果
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # 打印最佳结果
    print('\n' + '='*80, flush=True)
    print('最佳配置', flush=True)
    print('='*80, flush=True)
    print(f'配置: {best_config}', flush=True)
    print(f'平均MSE: {best_avg_mse:.4f}', flush=True)

    # 打印前10名
    print('\n前10名配置:', flush=True)
    print('{:<35} {:<10} {:<10} {:<10} {:<10}'.format('配置', 'Humanoid', 'Ant', 'Walker2d', '平均'))
    print('-'*80)

    # 按平均MSE排序
    ranked = []
    for cfg_key, cfg_results in results.items():
        valid = [cfg_results[ds] for ds in cfg_results if 'mse' in cfg_results[ds]]
        if len(valid) == 3:
            avg_mse = np.mean([r['mse'] for r in valid])
            ranked.append((cfg_key, cfg_results, avg_mse))

    ranked.sort(key=lambda x: x[2])

    for cfg_key, cfg_results, avg_mse in ranked[:10]:
        h = cfg_results.get('humanoid', {}).get('mse', 0)
        a = cfg_results.get('ant', {}).get('mse', 0)
        w = cfg_results.get('walker2d', {}).get('mse', 0)
        print('{:<35} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}'.format(cfg_key, h, a, w, avg_mse))

    print('\nDone!', flush=True)
