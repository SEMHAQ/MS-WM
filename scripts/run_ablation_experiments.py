"""消融实验 - 训练损失消融和阈值函数对比"""
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

# ============================================================
# 不同门控机制
# ============================================================
class HardThresholdGate(nn.Module):
    """硬阈值门控"""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 3)  # 输入d_model*3, 输出3
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 输出3个值，用于三个分支的权重
        gate_values = self.linear(x)
        return (gate_values > self.threshold).float()

class GarroteGate(nn.Module):
    """Garrote阈值门控"""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 3)  # 输入d_model*3, 输出3
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # 输出3个值，用于三个分支的权重
        gate_values = self.linear(x)
        return 1 - (self.scale / (gate_values + 1e-8))**2

class SoftThresholdGate(nn.Module):
    """软阈值门控（默认）"""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 3)  # 输入d_model*3, 输出3

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

# ============================================================
# MS-WM模型（可配置门控机制）
# ============================================================
class MultiScaleModel(nn.Module):
    """MS-WM模型 - 可配置门控机制"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5, gate_type='soft'):
        super().__init__()
        self.state_dim = state_dim
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

        # 门控机制
        if gate_type == 'hard':
            self.fusion_gate = HardThresholdGate(d_model * 3)  # 输入d_model*3, 输出3
        elif gate_type == 'garrote':
            self.fusion_gate = GarroteGate(d_model * 3)  # 输入d_model*3, 输出3
        elif gate_type == 'soft':
            self.fusion_gate = SoftThresholdGate(d_model * 3)  # 输入d_model*3, 输出3

        self.fusion_proj = nn.Linear(d_model, state_dim)

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

        features = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
        gate = self.fusion_gate(features)
        stacked = torch.stack([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=1)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        pred = self.fusion_proj(fused)
        return states[:, -1, :] + pred

# ============================================================
# 训练函数（支持多步损失）
# ============================================================
def train_with_multistep_loss(model, Xs, Xa, Y, Xv, Xav, Yv, seed, lambda_ms=0.0, H_ms=1):
    """训练模型，支持多步展开损失"""
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
            s_batch = torch.FloatTensor(Xs[bi]).to(device)
            a_batch = torch.FloatTensor(Xa[bi]).to(device)
            target = torch.FloatTensor(Y[bi]).to(device)

            # 单步损失
            pred = model(s_batch, a_batch)
            loss_single = loss_fn(pred, target)

            # 多步损失（如果启用）
            loss_ms = torch.tensor(0.0, device=device)
            if lambda_ms > 0 and H_ms > 1:
                cur_s = s_batch.clone()
                cur_a = a_batch.clone()
                for h in range(H_ms):
                    pred_h = model(cur_s, cur_a)
                    # 使用单步目标作为多步目标的近似
                    loss_ms = loss_ms + loss_fn(pred_h, target)
                    cur_s = torch.cat([cur_s[:, 1:], pred_h.unsqueeze(1)], dim=1)
                loss_ms = loss_ms / H_ms

            loss = loss_single + lambda_ms * loss_ms

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 20: break

    # 评估
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

    # 多步预测评估
    multistep_mse = {}
    for H in [1, 4, 8]:
        mse_h = []
        for idx in range(min(50, len(Xv))):
            seq_s = torch.FloatTensor(Xv[idx:idx+1]).to(device)
            seq_a = torch.FloatTensor(Xav[idx:idx+1]).to(device)
            true_next = []
            for h in range(H):
                if idx + h < len(Yv):
                    true_next.append(Yv[idx + h])
            if len(true_next) < H: continue
            preds = []
            cur_s, cur_a = seq_s.clone(), seq_a.clone()
            for h in range(H):
                with torch.no_grad(): p = model(cur_s, cur_a)
                preds.append(p.cpu().numpy()[0])
                cur_s = torch.cat([cur_s[:, 1:], p.unsqueeze(1)], dim=1)
            mse_h.append(np.mean((np.array(preds) - np.array(true_next))**2))
        multistep_mse[f'H{H}'] = round(np.mean(mse_h), 6) if mse_h else None

    return {
        'mse': round(mse, 6),
        'r2': round(r2, 4),
        'params_m': round(params, 3),
        'best_epoch': best_ep,
        'multistep': multistep_mse
    }

# ============================================================
# 实验1: 训练损失消融 (表9)
# ============================================================
def run_loss_ablation():
    print('\n' + '='*60, flush=True)
    print('实验1: 训练损失消融 (Humanoid)', flush=True)
    print('='*60, flush=True)

    RESULTS_FILE = 'experiments/loss_ablation_results.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)

    # 测试不同lambda和H配置
    configs = [
        {'lambda_ms': 0.0, 'H_ms': 1, 'name': 'lambda=0 (单步)'},
        {'lambda_ms': 0.1, 'H_ms': 8, 'name': 'lambda=0.1'},
        {'lambda_ms': 0.5, 'H_ms': 4, 'name': 'lambda=0.5, H=4'},
        {'lambda_ms': 0.5, 'H_ms': 8, 'name': 'lambda=0.5, H=8 (默认)'},
        {'lambda_ms': 0.5, 'H_ms': 16, 'name': 'lambda=0.5, H=16'},
        {'lambda_ms': 1.0, 'H_ms': 8, 'name': 'lambda=1.0'},
    ]

    for cfg in configs:
        config_name = cfg['name']
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
            model = MultiScaleModel(348, 17, d_model=96, d_state=8, n_layers=1, window_size=5, gate_type='soft')
            r = train_with_multistep_loss(model, Xs, Xa, Y, Xv, Xav, Yv, seed,
                                         lambda_ms=cfg['lambda_ms'], H_ms=cfg['H_ms'])
            results[config_name][seed_key] = r
            print(f'MSE={r["mse"]:.4f}, H8={r["multistep"]["H8"]:.3f}', flush=True)

            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 打印结果
    print('\n训练损失消融结果:', flush=True)
    print('{:<25} {:<15} {:<15} {:<10}'.format('配置', 'MSE(×10⁻²)', 'H8 MSE(×10⁻³)', '收敛轮数'))
    print('-'*70)
    for cfg in configs:
        config_name = cfg['name']
        if config_name in results:
            valid = [results[config_name][s] for s in results[config_name] if 'mse' in results[config_name][s]]
            if valid:
                mses = [r['mse'] for r in valid]
                h8_mses = [r['multistep']['H8'] for r in valid if r['multistep']['H8']]
                epochs = [r['best_epoch'] for r in valid]
                print('{:<25} {:.2f}±{:.2f}      {:.2f}±{:.2f}      {:.0f}'.format(
                    config_name,
                    np.mean(mses)*100, np.std(mses)*100,
                    np.mean(h8_mses)*1000, np.std(h8_mses)*1000,
                    np.mean(epochs)
                ))

# ============================================================
# 实验2: 架构消融 (表8)
# ============================================================
def run_architecture_ablation():
    print('\n' + '='*60, flush=True)
    print('实验2: 架构消融 (Humanoid)', flush=True)
    print('='*60, flush=True)

    RESULTS_FILE = 'experiments/architecture_ablation_results.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)

    # 测试不同配置
    configs = {
        'default': {'d_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5},
        'd_model_64': {'d_model': 64, 'd_state': 8, 'n_layers': 1, 'window_size': 5},
        'd_model_128': {'d_model': 128, 'd_state': 8, 'n_layers': 1, 'window_size': 5},
        'd_model_192': {'d_model': 192, 'd_state': 8, 'n_layers': 1, 'window_size': 5},
        'd_state_16': {'d_model': 96, 'd_state': 16, 'n_layers': 1, 'window_size': 5},
        'n_layers_2': {'d_model': 96, 'd_state': 8, 'n_layers': 2, 'window_size': 5},
        'n_layers_3': {'d_model': 96, 'd_state': 8, 'n_layers': 3, 'window_size': 5},
        'window_3': {'d_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 3},
        'window_7': {'d_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 7},
    }

    for config_name, config in configs.items():
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
            model = MultiScaleModel(348, 17, **config)
            r = train_with_multistep_loss(model, Xs, Xa, Y, Xv, Xav, Yv, seed)
            results[config_name][seed_key] = r
            print(f'MSE={r["mse"]:.4f}', flush=True)

            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 打印结果
    print('\n架构消融结果:', flush=True)
    print('{:<20} {:<15} {:<15} {:<10}'.format('配置', 'MSE(×10⁻²)', 'R²', '参数(M)'))
    print('-'*65)
    for config_name in configs:
        if config_name in results:
            valid = [results[config_name][s] for s in results[config_name] if 'mse' in results[config_name][s]]
            if valid:
                mses = [r['mse'] for r in valid]
                r2s = [r['r2'] for r in valid]
                params = valid[0]['params_m']
                print('{:<20} {:.2f}±{:.2f}    {:.4f}    {:.3f}'.format(
                    config_name,
                    np.mean(mses)*100, np.std(mses)*100,
                    np.mean(r2s),
                    params
                ))

# ============================================================
# 实验3: 阈值函数对比 (表12)
# ============================================================
def run_threshold_experiment():
    print('\n' + '='*60, flush=True)
    print('实验3: 阈值函数对比 (Humanoid)', flush=True)
    print('='*60, flush=True)

    RESULTS_FILE = 'experiments/threshold_results.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)

    # 测试不同门控机制
    gate_types = ['hard', 'garrote', 'soft']

    for gate_type in gate_types:
        if gate_type in results and len(results[gate_type]) >= len(SEEDS):
            print(f'\n{gate_type}: 已有完整结果，跳过', flush=True)
            continue

        print(f'\n{gate_type}:', flush=True)
        if gate_type not in results:
            results[gate_type] = {}

        for seed in SEEDS:
            seed_key = f'seed{seed}'
            if seed_key in results[gate_type]:
                print(f'  seed={seed} 已有，跳过', flush=True)
                continue

            print(f'  seed={seed}...', end=' ', flush=True)
            model = MultiScaleModel(348, 17, d_model=96, d_state=8, n_layers=1, window_size=5, gate_type=gate_type)
            r = train_with_multistep_loss(model, Xs, Xa, Y, Xv, Xav, Yv, seed)
            results[gate_type][seed_key] = r
            print(f'MSE={r["mse"]:.4f}', flush=True)

            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 打印结果
    print('\n阈值函数对比结果:', flush=True)
    print('{:<15} {:<15} {:<10}'.format('门控类型', 'MSE(×10⁻²)', 'R²'))
    print('-'*45)
    for gate_type in gate_types:
        if gate_type in results:
            valid = [results[gate_type][s] for s in results[gate_type] if 'mse' in results[gate_type][s]]
            if valid:
                mses = [r['mse'] for r in valid]
                r2s = [r['r2'] for r in valid]
                print('{:<15} {:.2f}±{:.2f}    {:.4f}'.format(
                    gate_type,
                    np.mean(mses)*100, np.std(mses)*100,
                    np.mean(r2s)
                ))

# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print(f'Device: {device}', flush=True)
    print(f'开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    run_loss_ablation()
    run_architecture_ablation()
    run_threshold_experiment()

    print('\n' + '='*60, flush=True)
    print('所有消融实验完成!', flush=True)
    print('='*60, flush=True)
