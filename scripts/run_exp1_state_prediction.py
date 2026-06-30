"""实验1：状态预测对比（公平比较）

模型：LSTM, GRU, Transformer, Mamba, S4D, MIMO-WM
数据集：Humanoid(348D), Ant(105D), Walker2d(17D)
指标：MSE, R², 参数量, 推理时间
种子：5个 [42, 123, 456, 789, 1024]
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time, math
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel, DiagSSM
from src.models.baselines import LSTMWorldModel, GRUWorldModel, TransformerWorldModel, SimpleSSMWorldModel, MLPWorldModel, TCNWorldModel
from src.models.mamba_world_model import MambaWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
BS = 256
LR = 5e-4
T = 32

print(f'Device: {device}', flush=True)

# ============================================================
# 数据加载
# ============================================================
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
# MIMO-WM 模型
# ============================================================
class MIMOLayer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.output(x) * torch.sigmoid(self.gate(x))
        return residual + x

class MIMOWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.backbone = nn.ModuleList([MIMOLayer(d_model, d_state) for _ in range(n_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            h = block(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])

# ============================================================
# 训练+评估
# ============================================================
def train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
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
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
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

    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        inf_time = (time.perf_counter() - t0) / 100 * 1000

    return {
        'mse': round(mse, 6), 'r2': round(r2, 4),
        'params_m': round(params, 3), 'inf_time_ms': round(inf_time, 2),
        'best_epoch': best_ep
    }

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    datasets = {
        'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
        'humanoid_standup': {'dir': 'data/humanoid_standup', 'sd': 348, 'ad': 17},
    }

    models = {
        'LSTM-WM':        (LSTMWorldModel,      lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 96, 'n_layers': 2}),
        'GRU-WM':         (GRUWorldModel,        lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 96, 'n_layers': 2}),
        'Transformer-WM': (TransformerWorldModel,lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'nhead': 4, 'n_layers': 2}),
        'Mamba-WM':       (MambaWorldModel,      lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'n_layers': 2}),
        'MLP-WM':         (MLPWorldModel,        lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 96, 'n_layers': 2}),
        'TCN-WM':         (TCNWorldModel,        lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'n_layers': 2, 'kernel_size': 3}),
        'MIMO-WM':        (MIMOWorldModel,       lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'd_state': 16, 'n_layers': 2}),
    }

    RESULTS_FILE = 'experiments/exp1_state_prediction.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    for ds_name, ds_cfg in datasets.items():
        print(f'\n{"="*60}', flush=True)
        print(f'数据集: {ds_name}', flush=True)
        print(f'{"="*60}', flush=True)

        print(f'  加载数据...', flush=True)
        eps_tr = load_eps(ds_cfg['dir'], 'train')
        eps_vl = load_eps(ds_cfg['dir'], 'val')
        m, s = stats(eps_tr)
        Xs, Xa, Y = make_data(eps_tr, T, m, s)
        Xv, Xav, Yv = make_data(eps_vl, T, m, s)
        print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

        for model_name, (ModelClass, kwargs_fn) in models.items():
            key = f'{model_name}_{ds_name}'
            if key in results and len(results[key]) >= len(SEEDS):
                print(f'\n{model_name}: 已有完整结果，跳过', flush=True)
                continue

            print(f'\n{model_name}:', flush=True)
            if key not in results:
                results[key] = {}

            kwargs = kwargs_fn(ds_cfg['sd'], ds_cfg['ad'])
            for seed in SEEDS:
                seed_key = f'seed{seed}'
                if seed_key in results[key]:
                    print(f'  seed={seed} 已有，跳过', flush=True)
                    continue
                print(f'  seed={seed}...', end=' ', flush=True)
                r = train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
                results[key][seed_key] = r
                print(f'MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}', flush=True)
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

    # 汇总
    print('\n' + '='*60, flush=True)
    print('实验1：状态预测对比结果', flush=True)
    print('='*60, flush=True)

    for ds_name in datasets:
        print(f'\n{ds_name}:', flush=True)
        print('{:<18} {:<18} {:<10} {:<10} {:<10}'.format('模型', 'MSE(×10⁻²)', 'R²', '参数(M)', '推理(ms)'))
        print('-'*70)
        for model_name in models:
            key = f'{model_name}_{ds_name}'
            if key in results:
                valid = [results[key][s] for s in results[key] if 'mse' in results[key][s]]
                if valid:
                    mses = [r['mse'] for r in valid]
                    r2s = [r['r2'] for r in valid]
                    params = valid[0]['params_m']
                    inf_time = valid[0]['inf_time_ms']
                    print('{:<18} {:.2f}±{:.2f}      {:.4f}    {:.3f}    {:.2f}'.format(
                        model_name, np.mean(mses)*100, np.std(mses)*100, np.mean(r2s), params, inf_time
                    ))

    print('\nDone!', flush=True)
