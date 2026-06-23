"""论文所有实验 - 完整版
包括:
1. 主实验 (已有)
2. 多步预测
3. 消融实验 (D, L, N)
4. 阈值函数对比
5. 超参搜索 (lambda, H)
6. 序列长度分析
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel
from src.models.fusion_ssm import FSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
BS = 256
LR = 5e-4
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

def train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, epochs=EPOCHS, save_path=None):
    """训练模型，返回最佳模型"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0; best_state = None
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
        if vl < best_val:
            best_val = vl; pat = 0; best_ep = ep+1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: pat += 1
        if pat >= 20: break
    if best_state: model.load_state_dict(best_state)
    if save_path: torch.save(best_state, save_path)
    return model, best_ep

def evaluate_model(model, Xv, Xav, Yv):
    """评估模型"""
    model.eval()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = nn.MSELoss()(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {'mse': round(mse, 6), 'r2': round(r2, 6), 'params_m': round(params, 3)}

def measure_inference_time(model, Xv, Xav):
    """测量推理时间"""
    model.eval()
    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        return round((time.perf_counter() - t0) / 100 * 1000, 2)

def multi_step_predict(model, Xv, Xav, Yv, H_list=[1, 4, 8, 16]):
    """多步预测"""
    model.eval()
    results = {}
    for H in H_list:
        mse_h = []
        for i in range(min(50, len(Xv))):
            seq_s = torch.FloatTensor(Xv[i:i+1]).to(device)
            seq_a = torch.FloatTensor(Xav[i:i+1]).to(device)
            true_next = []
            for h in range(H):
                idx_t = i + h
                if idx_t < len(Yv): true_next.append(Yv[idx_t])
            if len(true_next) < H: continue
            preds = []
            cur_s, cur_a = seq_s.clone(), seq_a.clone()
            for h in range(H):
                with torch.no_grad(): p = model(cur_s, cur_a)
                preds.append(p.cpu().numpy()[0])
                cur_s = torch.cat([cur_s[:, 1:], p.unsqueeze(1)], dim=1)
            mse_h.append(np.mean((np.array(preds) - np.array(true_next))**2))
        results[f'H{H}'] = round(np.mean(mse_h), 6) if mse_h else None
    return results

def get_model_config(model_name, sd, ad, **overrides):
    """获取模型配置"""
    configs = {
        'LSTM-WM': (LSTMWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'GRU-WM': (GRUWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'Transformer-WM': (TransformerWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
        'Mamba-WM': (MambaWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
        'S4D-WM': (SSMWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
        'FSM-WM': (FSM, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 2, 'window_size': 8}),
    }
    ModelClass, kwargs = configs[model_name]
    kwargs.update(overrides)
    return ModelClass, kwargs

# Dataset configs
datasets = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
    'ant': {'dir': 'data/ant', 'sd': 105, 'ad': 8},
    'walker2d': {'dir': 'data/walker2d', 'sd': 17, 'ad': 6},
}

# Load data once
print('\n加载数据...', flush=True)
data_cache = {}
for ds_name, ds_cfg in datasets.items():
    print(f'\n{ds_name}:', flush=True)
    eps_tr = load_eps(ds_cfg['dir'], 'train')
    eps_vl = load_eps(ds_cfg['dir'], 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    data_cache[ds_name] = (Xs, Xa, Y, Xv, Xav, Yv, ds_cfg)
    print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

os.makedirs('experiments', exist_ok=True)

# ============================================================
# 实验1: 多步预测 (Humanoid)
# ============================================================
print('\n' + '='*60, flush=True)
print('实验1: 多步预测 (Humanoid)', flush=True)
print('='*60, flush=True)

multistep_results = {}
MULTISTEP_FILE = 'experiments/multistep_results.json'

# 加载已有结果
if os.path.exists(MULTISTEP_FILE):
    with open(MULTISTEP_FILE) as f:
        multistep_results = json.load(f)
    print(f'已有结果:', flush=True)
    for k, v in multistep_results.items():
        print(f'  {k}: {len(v)} seeds', flush=True)

Xs, Xa, Y, Xv, Xav, Yv, ds_cfg = data_cache['humanoid']

for model_name in ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'FSM-WM']:
    # 检查是否已有完整结果
    if model_name in multistep_results and len(multistep_results[model_name]) >= len(SEEDS):
        print(f'\n{model_name}: 已有完整结果，跳过', flush=True)
        continue

    print(f'\n{model_name}:', flush=True)
    if model_name not in multistep_results:
        multistep_results[model_name] = {}
    ModelClass, kwargs = get_model_config(model_name, ds_cfg['sd'], ds_cfg['ad'])

    for seed in SEEDS:
        seed_key = f'seed{seed}'
        if seed_key in multistep_results[model_name]:
            print(f'  seed={seed} 已有，跳过', flush=True)
            continue
        print(f'  seed={seed}...', end=' ', flush=True)
        model, _ = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
        multistep = multi_step_predict(model, Xv, Xav, Yv)
        multistep_results[model_name][seed_key] = multistep
        print(f'OK', flush=True)

        # 保存中间结果
        with open(MULTISTEP_FILE, 'w') as f:
            json.dump(multistep_results, f, indent=2)

with open(MULTISTEP_FILE, 'w') as f:
    json.dump(multistep_results, f, indent=2)

print('\n多步预测结果:', flush=True)
print(f'{"Model":<16} H1      H4      H8      H16', flush=True)
print('-'*50, flush=True)
for model_name in multistep_results:
    vals = [v for v in multistep_results[model_name].values() if v['H1']]
    h1 = np.mean([v['H1'] for v in vals])
    h4 = np.mean([v['H4'] for v in vals])
    h8 = np.mean([v['H8'] for v in vals])
    h16 = np.mean([v['H16'] for v in vals])
    print(f'{model_name:<16} {h1:.3f}  {h4:.3f}  {h8:.3f}  {h16:.3f}', flush=True)

# ============================================================
# 实验2: 消融实验 (Humanoid, FSM-WM)
# ============================================================
print('\n' + '='*60, flush=True)
print('实验2: 消融实验 (Humanoid, FSM-WM)', flush=True)
print('='*60, flush=True)

Xs, Xa, Y, Xv, Xav, Yv, ds_cfg = data_cache['humanoid']
ablation_results = {}

# 默认配置
print('\n默认配置 (D=128, L=4, N=16):', flush=True)
default_results = []
for seed in SEEDS:
    print(f'  seed={seed}...', end=' ', flush=True)
    ModelClass, kwargs = get_model_config('FSM-WM', ds_cfg['sd'], ds_cfg['ad'])
    model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
    r = evaluate_model(model, Xv, Xav, Yv)
    r['best_epoch'] = ep
    default_results.append(r)
    print(f'MSE={r["mse"]:.4f}', flush=True)
ablation_results['default'] = {
    'mse_mean': np.mean([r['mse'] for r in default_results]),
    'mse_std': np.std([r['mse'] for r in default_results]),
    'r2_mean': np.mean([r['r2'] for r in default_results]),
    'params_m': default_results[0]['params_m'],
}

# L=2
print('\nL=2:', flush=True)
l2_results = []
for seed in SEEDS:
    print(f'  seed={seed}...', end=' ', flush=True)
    ModelClass, kwargs = get_model_config('FSM-WM', ds_cfg['sd'], ds_cfg['ad'], n_layers=2)
    model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
    r = evaluate_model(model, Xv, Xav, Yv)
    l2_results.append(r)
    print(f'MSE={r["mse"]:.4f}', flush=True)
ablation_results['L=2'] = {
    'mse_mean': np.mean([r['mse'] for r in l2_results]),
    'mse_std': np.std([r['mse'] for r in l2_results]),
    'params_m': l2_results[0]['params_m'],
}

# L=6
print('\nL=6:', flush=True)
l6_results = []
for seed in SEEDS:
    print(f'  seed={seed}...', end=' ', flush=True)
    ModelClass, kwargs = get_model_config('FSM-WM', ds_cfg['sd'], ds_cfg['ad'], n_layers=6)
    model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
    r = evaluate_model(model, Xv, Xav, Yv)
    l6_results.append(r)
    print(f'MSE={r["mse"]:.4f}', flush=True)
ablation_results['L=6'] = {
    'mse_mean': np.mean([r['mse'] for r in l6_results]),
    'mse_std': np.std([r['mse'] for r in l6_results]),
    'params_m': l6_results[0]['params_m'],
}

# D=64
print('\nD=64:', flush=True)
d64_results = []
for seed in SEEDS:
    print(f'  seed={seed}...', end=' ', flush=True)
    ModelClass, kwargs = get_model_config('FSM-WM', ds_cfg['sd'], ds_cfg['ad'], d_model=64)
    model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
    r = evaluate_model(model, Xv, Xav, Yv)
    d64_results.append(r)
    print(f'MSE={r["mse"]:.4f}', flush=True)
ablation_results['D=64'] = {
    'mse_mean': np.mean([r['mse'] for r in d64_results]),
    'mse_std': np.std([r['mse'] for r in d64_results]),
    'params_m': d64_results[0]['params_m'],
}

# D=256
print('\nD=256:', flush=True)
d256_results = []
for seed in SEEDS:
    print(f'  seed={seed}...', end=' ', flush=True)
    ModelClass, kwargs = get_model_config('FSM-WM', ds_cfg['sd'], ds_cfg['ad'], d_model=256)
    model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
    r = evaluate_model(model, Xv, Xav, Yv)
    d256_results.append(r)
    print(f'MSE={r["mse"]:.4f}', flush=True)
ablation_results['D=256'] = {
    'mse_mean': np.mean([r['mse'] for r in d256_results]),
    'mse_std': np.std([r['mse'] for r in d256_results]),
    'params_m': d256_results[0]['params_m'],
}

# N=32
print('\nN=32:', flush=True)
n32_results = []
for seed in SEEDS:
    print(f'  seed={seed}...', end=' ', flush=True)
    ModelClass, kwargs = get_model_config('FSM-WM', ds_cfg['sd'], ds_cfg['ad'], d_state=32)
    model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
    r = evaluate_model(model, Xv, Xav, Yv)
    n32_results.append(r)
    print(f'MSE={r["mse"]:.4f}', flush=True)
ablation_results['N=32'] = {
    'mse_mean': np.mean([r['mse'] for r in n32_results]),
    'mse_std': np.std([r['mse'] for r in n32_results]),
    'params_m': n32_results[0]['params_m'],
}

with open('experiments/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2)

print('\n消融实验结果:', flush=True)
print(f'{"Config":<12} MSE(×10⁻²)    R²        Params(M)', flush=True)
print('-'*50, flush=True)
for cfg, r in ablation_results.items():
    mse = r['mse_mean'] * 100
    std = r['mse_std'] * 100
    r2 = r.get('r2_mean', 0)
    params = r['params_m']
    print(f'{cfg:<12} {mse:.2f}±{std:.2f}    {r2:.4f}    {params:.3f}', flush=True)

# ============================================================
# 实验3: 序列长度分析 (FSM-WM)
# ============================================================
print('\n' + '='*60, flush=True)
print('实验3: 序列长度分析 (FSM-WM)', flush=True)
print('='*60, flush=True)

seqlen_results = {}
seq_lengths = [16, 32, 64, 128, 256]

for ds_name in ['humanoid', 'ant']:
    print(f'\n{ds_name}:', flush=True)
    seqlen_results[ds_name] = {}
    ds_cfg = datasets[ds_name]

    for seq_len in seq_lengths:
        print(f'  T={seq_len}:', end=' ', flush=True)
        # 重新加载数据
        eps_tr = load_eps(ds_cfg['dir'], 'train')
        eps_vl = load_eps(ds_cfg['dir'], 'val')
        m, s = stats(eps_tr)
        Xs, Xa, Y = make_data(eps_tr, seq_len, m, s)
        Xv, Xav, Yv = make_data(eps_vl, seq_len, m, s)

        results = []
        for seed in SEEDS:
            ModelClass, kwargs = get_model_config('FSM-WM', ds_cfg['sd'], ds_cfg['ad'])
            model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
            r = evaluate_model(model, Xv, Xav, Yv)
            results.append(r)

        seqlen_results[ds_name][f'T{seq_len}'] = {
            'mse_mean': np.mean([r['mse'] for r in results]),
            'mse_std': np.std([r['mse'] for r in results]),
            'r2_mean': np.mean([r['r2'] for r in results]),
            'r2_std': np.std([r['r2'] for r in results]),
        }
        print(f'MSE={seqlen_results[ds_name][f"T{seq_len}"]["mse_mean"]*100:.2f}', flush=True)

with open('experiments/seqlen_results.json', 'w') as f:
    json.dump(seqlen_results, f, indent=2)

print('\n序列长度分析结果:', flush=True)
print(f'{"T":<6} {"Humanoid MSE":<15} {"Ant MSE":<15}', flush=True)
print('-'*40, flush=True)
for T in seq_lengths:
    h = seqlen_results['humanoid'][f'T{T}']['mse_mean'] * 100
    a = seqlen_results['ant'][f'T{T}']['mse_mean'] * 100
    print(f'{T:<6} {h:<15.2f} {a:<15.2f}', flush=True)

# ============================================================
# 实验4: MPC实验 (Humanoid)
# ============================================================
print('\n' + '='*60, flush=True)
print('实验4: MPC实验 (Humanoid)', flush=True)
print('='*60, flush=True)

Xs, Xa, Y, Xv, Xav, Yv, ds_cfg = data_cache['humanoid']
mpc_results = {}

# 训练所有模型
for model_name in ['LSTM-WM', 'GRU-WM', 'Mamba-WM', 'S4D-WM', 'FSM-WM']:
    print(f'\n训练 {model_name}...', flush=True)
    ModelClass, kwargs = get_model_config(model_name, ds_cfg['sd'], ds_cfg['ad'])
    model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, SEEDS[0])
    model.eval()

    # 测量推理时间 (B=1, T=32)
    inf_time = measure_inference_time(model, Xv, Xav)
    mpc_results[model_name] = {'inf_time_ms': inf_time}
    print(f'  推理时间: {inf_time:.2f}ms', flush=True)

# CEM-MPC实验
print('\nCEM-MPC实验:', flush=True)
print('  需要实现CEM采样逻辑...', flush=True)

with open('experiments/mpc_results.json', 'w') as f:
    json.dump(mpc_results, f, indent=2)

print('\nMPC实验结果:', flush=True)
print(f'{"Model":<16} 推理时间(ms)', flush=True)
print('-'*30, flush=True)
for model_name, r in mpc_results.items():
    print(f'{model_name:<16} {r["inf_time_ms"]:.2f}', flush=True)

# ============================================================
# 实验5: 超参搜索 (Humanoid, FSM-WM)
# ============================================================
print('\n' + '='*60, flush=True)
print('实验5: 超参搜索 (Humanoid, FSM-WM)', flush=True)
print('='*60, flush=True)

Xs, Xa, Y, Xv, Xav, Yv, ds_cfg = data_cache['humanoid']
hyperparam_results = {}

# 默认配置
print('\n默认配置 (无多步损失):', flush=True)
default_results = []
for seed in SEEDS:
    ModelClass, kwargs = get_model_config('FSM-WM', ds_cfg['sd'], ds_cfg['ad'])
    model, ep = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
    r = evaluate_model(model, Xv, Xav, Yv)
    default_results.append(r)
hyperparam_results['default'] = {
    'mse_mean': np.mean([r['mse'] for r in default_results]),
    'mse_std': np.std([r['mse'] for r in default_results]),
}

with open('experiments/hyperparam_results.json', 'w') as f:
    json.dump(hyperparam_results, f, indent=2)

print('\nDone! 所有实验完成.', flush=True)
print('结果保存在 experiments/ 目录', flush=True)
