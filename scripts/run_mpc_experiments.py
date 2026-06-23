"""MPC实验 - 独立脚本
包括:
1. 梯度MPC (Adam优化器)
2. CEM采样MPC
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel
from src.models.fusion_ssm import FSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42]
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

def train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, epochs=EPOCHS):
    """训练模型"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_state = None
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
            best_val = vl; pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: pat += 1
        if pat >= 20: break
    if best_state: model.load_state_dict(best_state)
    return model

def get_model_config(model_name, sd, ad):
    """获取模型配置"""
    configs = {
        'LSTM-WM': (LSTMWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'GRU-WM': (GRUWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'Transformer-WM': (TransformerWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
        'Mamba-WM': (MambaWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
        'S4D-WM': (SSMWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
        'FSM-WM': (FSM, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4, 'window_size': 8}),
    }
    return configs[model_name]

# ============================================================
# 梯度MPC (Adam优化器)
# ============================================================
def gradient_mpc(model, init_state, init_actions, ref_state, horizon=10, n_iter=50, lr=0.01):
    """基于梯度的MPC"""
    # 初始化动作序列
    actions = torch.randn(1, horizon, init_actions.shape[-1], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([actions], lr=lr)

    # 保存原始状态
    was_training = model.training

    for _ in range(n_iter):
        optimizer.zero_grad()
        # 前向展开 (需要training mode for RNN backward)
        model.train()
        cur_state = init_state.clone()
        cur_actions = init_actions.clone()
        total_cost = 0

        with torch.enable_grad():
            for h in range(horizon):
                # 拼接当前状态和动作
                pred = model(cur_state, actions[:, h:h+1, :])
                # 计算代价
                cost = torch.sum((pred - ref_state) ** 2)
                total_cost = cost
                # 更新状态
                cur_state = torch.cat([cur_state[:, 1:], pred.unsqueeze(1)], dim=1)

        total_cost.backward()
        optimizer.step()

    # 恢复原始状态
    model.train(was_training)
    return actions.detach()

# ============================================================
# CEM采样MPC
# ============================================================
def cem_mpc(model, init_state, init_actions, ref_state, horizon=10, n_samples=200, n_elite=30, n_iter=2):
    """CEM采样MPC - GPU批量并行版本"""
    model.eval()
    action_dim = init_actions.shape[-1]

    # 初始化采样分布
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device) * 0.5

    for _ in range(n_iter):
        # 采样动作序列
        samples = torch.randn(n_samples, horizon, action_dim, device=device)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * samples
        samples = torch.clamp(samples, -1, 1)  # 限制动作范围

        # 批量评估所有样本 (GPU并行)
        # 扩展初始状态到n_samples个副本
        cur_states = init_state.expand(n_samples, -1, -1)  # (n_samples, T, state_dim)
        total_costs = torch.zeros(n_samples, device=device)

        for h in range(horizon):
            # 批量前向传播
            pred = model(cur_states, samples[:, h:h+1, :])  # (n_samples, state_dim)
            # 计算代价
            costs = torch.sum((pred - ref_state) ** 2, dim=-1)  # (n_samples,)
            total_costs = costs
            # 更新状态
            cur_states = torch.cat([cur_states[:, 1:], pred.unsqueeze(1)], dim=1)

        # 选择精英样本
        elite_idx = torch.topk(total_costs, n_elite, largest=False).indices
        elite_samples = samples[elite_idx]

        # 更新分布
        mean = elite_samples.mean(dim=0)
        std = elite_samples.std(dim=0) + 1e-6

    return mean.unsqueeze(0)

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
    RESULTS_FILE = 'experiments/mpc_results.json'

    # 加载已有结果
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            mpc_results = json.load(f)
        print(f'已有结果:', flush=True)
        for k, v in mpc_results.items():
            print(f'  {k}: {v}', flush=True)
    else:
        mpc_results = {}

    # 训练所有模型
    models = {}
    for model_name in ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'FSM-WM']:
        # 检查是否已有推理时间
        if model_name in mpc_results and 'inf_time_ms' in mpc_results[model_name]:
            print(f'\n{model_name}: 已有推理时间 {mpc_results[model_name]["inf_time_ms"]}ms, 跳过训练', flush=True)
            # 重新加载模型
            ModelClass, kwargs = get_model_config(model_name, sd, ad)
            model = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, SEEDS[0])
            models[model_name] = model
            continue

        print(f'\n训练 {model_name}...', flush=True)
        ModelClass, kwargs = get_model_config(model_name, sd, ad)
        model = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, SEEDS[0])
        models[model_name] = model

        # 测量推理时间
        model.eval()
        with torch.no_grad():
            x_dummy = torch.FloatTensor(Xv[:1]).to(device)
            a_dummy = torch.FloatTensor(Xav[:1]).to(device)
            for _ in range(5): model(x_dummy, a_dummy)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100): model(x_dummy, a_dummy)
            torch.cuda.synchronize()
            inf_time = (time.perf_counter() - t0) / 100 * 1000

        mpc_results[model_name] = {'inf_time_ms': round(inf_time, 2)}
        print(f'  推理时间: {inf_time:.2f}ms', flush=True)

        # 保存中间结果
        with open(RESULTS_FILE, 'w') as f:
            json.dump(mpc_results, f, indent=2)

    # 梯度MPC实验
    print('\n' + '='*60, flush=True)
    print('梯度MPC实验', flush=True)
    print('='*60, flush=True)

    for model_name, model in models.items():
        # 检查是否已有梯度MPC结果
        if model_name in mpc_results and 'gradient_ms' in mpc_results[model_name]:
            print(f'\n{model_name}: 已有梯度MPC结果 {mpc_results[model_name]["gradient_ms"]}ms, 跳过', flush=True)
            continue

        print(f'\n{model_name}:', flush=True)
        times = []
        for trial in range(5):
            # 随机选择初始状态
            idx = np.random.randint(len(Xv))
            init_state = torch.FloatTensor(Xv[idx:idx+1]).to(device)
            init_actions = torch.FloatTensor(Xav[idx:idx+1]).to(device)
            ref_state = torch.FloatTensor(Yv[idx:idx+1]).to(device)

            t0 = time.perf_counter()
            actions = gradient_mpc(model, init_state, init_actions, ref_state)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mpc_results[model_name]['gradient_ms'] = round(np.mean(times), 1)
        mpc_results[model_name]['gradient_hz'] = round(1000 / np.mean(times), 1)
        print(f'  梯度MPC: {np.mean(times):.1f}ms, {1000/np.mean(times):.1f}Hz', flush=True)

        # 保存中间结果
        with open(RESULTS_FILE, 'w') as f:
            json.dump(mpc_results, f, indent=2)

    # CEM-MPC实验
    print('\n' + '='*60, flush=True)
    print('CEM-MPC实验', flush=True)
    print('='*60, flush=True)

    for model_name, model in models.items():
        # 检查是否已有CEM MPC结果
        if model_name in mpc_results and 'cem_ms' in mpc_results[model_name]:
            print(f'\n{model_name}: 已有CEM MPC结果 {mpc_results[model_name]["cem_ms"]}ms, 跳过', flush=True)
            continue

        print(f'\n{model_name}:', flush=True)
        times = []
        for trial in range(5):
            idx = np.random.randint(len(Xv))
            init_state = torch.FloatTensor(Xv[idx:idx+1]).to(device)
            init_actions = torch.FloatTensor(Xav[idx:idx+1]).to(device)
            ref_state = torch.FloatTensor(Yv[idx:idx+1]).to(device)

            t0 = time.perf_counter()
            actions = cem_mpc(model, init_state, init_actions, ref_state)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mpc_results[model_name]['cem_ms'] = round(np.mean(times), 1)
        mpc_results[model_name]['cem_hz'] = round(1000 / np.mean(times), 1)
        print(f'  CEM-MPC: {np.mean(times):.1f}ms, {1000/np.mean(times):.1f}Hz', flush=True)

        # 保存中间结果
        with open(RESULTS_FILE, 'w') as f:
            json.dump(mpc_results, f, indent=2)

    # 保存结果
    with open('experiments/mpc_results.json', 'w') as f:
        json.dump(mpc_results, f, indent=2)

    # 打印结果
    print('\n' + '='*60, flush=True)
    print('MPC实验结果', flush=True)
    print('='*60, flush=True)
    print(f'{"Model":<16} 推理(ms)  梯度MPC(ms) CEM-MPC(ms)', flush=True)
    print('-'*50, flush=True)
    for model_name, r in mpc_results.items():
        inf = r['inf_time_ms']
        grad = r.get('gradient_ms', '-')
        cem = r.get('cem_ms', '-')
        print(f'{model_name:<16} {inf:<10} {grad:<12} {cem}', flush=True)

    print('\nDone!', flush=True)
