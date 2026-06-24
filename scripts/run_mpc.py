"""MPC实验 - MS-WM vs 基线模型
测试梯度MPC和CEM-MPC的控制频率
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel, DiagSSM
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
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
    """MS-WM模型"""
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

def train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed=SEED):
    """训练模型"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_state = None
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
        if vl < best_val:
            best_val = vl; pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: pat += 1
        if pat >= 20: break
    if best_state: model.load_state_dict(best_state)
    return model

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
        return (time.perf_counter() - t0) / 100 * 1000

def cem_mpc(model, init_state, init_actions, ref_state, horizon=10, n_samples=100, n_elite=20, n_iter=1):
    """CEM采样MPC - GPU批量并行版本"""
    model.eval()
    action_dim = init_actions.shape[-1]
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device) * 0.5

    for _ in range(n_iter):
        samples = torch.randn(n_samples, horizon, action_dim, device=device)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * samples
        samples = torch.clamp(samples, -1, 1)

        cur_states = init_state.expand(n_samples, -1, -1)
        total_costs = torch.zeros(n_samples, device=device)

        for h in range(horizon):
            pred = model(cur_states, samples[:, h:h+1, :])
            costs = torch.sum((pred - ref_state) ** 2, dim=-1)
            total_costs = costs
            cur_states = torch.cat([cur_states[:, 1:], pred.unsqueeze(1)], dim=1)

        elite_idx = torch.topk(total_costs, n_elite, largest=False).indices
        elite_samples = samples[elite_idx]
        mean = elite_samples.mean(dim=0)
        std = elite_samples.std(dim=0) + 1e-6

    return mean.unsqueeze(0)

# 模型配置
def get_model_config(model_name, sd, ad):
    configs = {
        'LSTM-WM': (LSTMWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'GRU-WM': (GRUWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'Transformer-WM': (TransformerWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'nhead': 4, 'n_layers': 4}),
        'Mamba-WM': (MambaWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
        'S4D-WM': (SSMWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
        'MS-WM': (MultiScaleModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'}),
    }
    return configs[model_name]

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
    else:
        mpc_results = {}

    # 训练所有模型
    models = {}
    for model_name in ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'MS-WM']:
        # 检查是否已有推理时间
        if model_name in mpc_results and 'inf_time_ms' in mpc_results[model_name]:
            print(f'\n{model_name}: 已有推理时间 {mpc_results[model_name]["inf_time_ms"]}ms, 跳过训练', flush=True)
            ModelClass, kwargs = get_model_config(model_name, sd, ad)
            model = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv)
            models[model_name] = model
            continue

        print(f'\n训练 {model_name}...', flush=True)
        ModelClass, kwargs = get_model_config(model_name, sd, ad)
        model = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv)
        models[model_name] = model

        # 测量推理时间
        inf_time = measure_inference_time(model, Xv, Xav)
        mpc_results[model_name] = {'inf_time_ms': round(inf_time, 2)}
        print(f'  推理时间: {inf_time:.2f}ms', flush=True)

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

    # 打印结果
    print('\n' + '='*60, flush=True)
    print('MPC实验结果', flush=True)
    print('='*60, flush=True)
    print('{:<16} {:<12} {:<12}'.format('模型', '推理(ms)', 'CEM-MPC(Hz)'))
    print('-'*45)

    for model_name in ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'MS-WM']:
        if model_name in mpc_results:
            r = mpc_results[model_name]
            inf = r.get('inf_time_ms', '-')
            cem = r.get('cem_hz', '-')
            print('{:<16} {:<12} {:<12}'.format(model_name, inf, cem))
        else:
            print('{:<16} {:<12} {:<12}'.format(model_name, '---', '---'))

    print('\nDone!', flush=True)
