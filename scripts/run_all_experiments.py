"""统一实验脚本 - 所有模型所有数据集"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel
from src.models.ssm_world_model import DiagSSM

class MultiScaleModel(nn.Module):
    """多尺度动力学模型"""
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

def train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, lambda_ms=0.5, H_ms=4):
    """训练模型，支持多步损失"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
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

            # 多步损失
            loss_ms = torch.tensor(0.0, device=device)
            if lambda_ms > 0:
                cur_s = s_batch.clone()
                cur_a = a_batch.clone()
                for h in range(H_ms):
                    pred_h = model(cur_s, cur_a)
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
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    # inference time
    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        inf_time = (time.perf_counter() - t0) / 100 * 1000
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {'mse': round(mse, 6), 'r2': round(r2, 6), 'best_epoch': best_ep, 'inf_time_ms': round(inf_time, 2), 'params_m': round(params, 3)}

# Dataset configs (Gymnasium MuJoCo medium-v0)
datasets = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
    'humanoid_standup': {'dir': 'data/humanoid_standup', 'sd': 348, 'ad': 17},
    'ant': {'dir': 'data/ant', 'sd': 105, 'ad': 8},
    'walker2d': {'dir': 'data/walker2d', 'sd': 17, 'ad': 6},
}

models = {
    'LSTM-WM': (LSTMWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
    'Transformer-WM': (TransformerWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'nhead': 4, 'n_layers': 4}),
    'GRU-WM': (GRUWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
    'Mamba-WM': (MambaWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
    'S4D-WM': (SSMWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
    'MS-WM': (MultiScaleModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'}),
}

RESULTS_FILE = 'experiments/all_results.json'
os.makedirs('experiments', exist_ok=True)

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
else:
    results = {}

for ds_name, ds_cfg in datasets.items():
    # Check if all models and seeds are done for this dataset
    all_done = True
    for model_name in models:
        key = f'{model_name}_{ds_name}'
        if key not in results:
            all_done = False
            break
        for seed in SEEDS:
            seed_key = f'seed{seed}'
            if seed_key not in results[key] or 'mse' not in results[key][seed_key]:
                all_done = False
                break
        if not all_done:
            break
    if all_done:
        print(f'\n{ds_name}: ALL DONE, SKIP', flush=True)
        continue

    print(f'\n{"="*60}', flush=True)
    print(f'Dataset: {ds_name}', flush=True)
    print(f'{"="*60}', flush=True)

    print(f'  Loading training data...', flush=True)
    eps_tr = load_eps(ds_cfg['dir'], 'train')
    print(f'  Loading validation data...', flush=True)
    eps_vl = load_eps(ds_cfg['dir'], 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    for model_name, (ModelClass, kwargs_fn) in models.items():
        kwargs = kwargs_fn(ds_cfg['sd'], ds_cfg['ad'])
        key = f'{model_name}_{ds_name}'
        if key not in results: results[key] = {}

        for seed in SEEDS:
            seed_key = f'seed{seed}'
            if seed_key in results[key] and 'mse' in results[key][seed_key]:
                print(f'  {model_name} seed={seed} SKIP', flush=True)
                continue
            print(f'  {model_name} seed={seed}...', end=' ', flush=True)
            t0 = time.perf_counter()
            try:
                r = train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
                elapsed = time.perf_counter() - t0
                results[key][seed_key] = r
                print(f'MSE={r["mse"]:.6f} R²={r["r2"]:.6f} ({elapsed/60:.1f}min)', flush=True)
            except Exception as e:
                print(f'ERROR: {e}', flush=True)
                results[key][seed_key] = {'error': str(e)}
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

# Summary
print('\n' + '='*80, flush=True)
print('SUMMARY', flush=True)
print('='*80, flush=True)
print(f'{"Model":<16}', end='', flush=True)
for ds in datasets: print(f'  {ds:<20}', end='', flush=True)
print(flush=True)
print('-'*80, flush=True)

for model_name in models:
    print(f'{model_name:<16}', end='', flush=True)
    for ds_name in datasets:
        key = f'{model_name}_{ds_name}'
        if key in results:
            valid = [results[key][s] for s in results[key] if 'mse' in results[key][s]]
            if valid:
                mses = [r['mse'] for r in valid]
                print(f'  {np.mean(mses):.4f}±{np.std(mses):.4f}', end='', flush=True)
            else:
                print(f'  ---', end='', flush=True)
        else:
            print(f'  ---', end='', flush=True)
    print(flush=True)

print('\nDone! Results saved to experiments/all_results.json', flush=True)
