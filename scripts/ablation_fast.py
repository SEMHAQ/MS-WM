"""Fast ablation: run default + key configs with 5 seeds."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel

device = torch.device('cuda')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100; BS = 512; LR = 5e-4; T = 32

def load_eps(d, s, mx=None):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    if mx: fs = fs[:mx]
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

def train_model(model, name, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xs_t = torch.FloatTensor(Xs).to(device); Xa_t = torch.FloatTensor(Xa).to(device); Y_t = torch.FloatTensor(Y).to(device)
    Xv_t = torch.FloatTensor(Xv).to(device); Xav_t = torch.FloatTensor(Xav).to(device); Yv_t = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(Xs_t[bi], Xa_t[bi]); loss = loss_fn(pred, Y_t[bi])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_t, Xav_t), Yv_t).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1; torch.save(model.state_dict(), f'experiments/{name}_seed{seed}.pth')
        else: pat += 1
        if pat >= 20: break
    model.load_state_dict(torch.load(f'experiments/{name}_seed{seed}.pth', map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(Xv_t, Xav_t); mse = loss_fn(pred, Yv_t).item()
        ss_r = torch.sum((Yv_t - pred)**2).item(); ss_t = torch.sum((Yv_t - torch.mean(Yv_t, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    return {'mse': round(mse, 6), 'r2': round(r2, 6), 'best_epoch': best_ep}

eps_tr = load_eps('data/humanoid', 'train', 930)
eps_vl = load_eps('data/humanoid', 'val', 233)
m, s = stats(eps_tr)
Xs, Xa, Y = make_data(eps_tr, T, m, s)
Xv, Xav, Yv = make_data(eps_vl, T, m, s)

# Key configs: default + L=2 + L=6 + D=64 + D=256 + N=32
configs = {
    'default': {'d_model': 128, 'd_state': 16, 'n_layers': 4},
    'L2': {'d_model': 128, 'd_state': 16, 'n_layers': 2},
    'L6': {'d_model': 128, 'd_state': 16, 'n_layers': 6},
    'N32': {'d_model': 128, 'd_state': 32, 'n_layers': 4},
    'D64': {'d_model': 64, 'd_state': 16, 'n_layers': 4},
    'D256': {'d_model': 256, 'd_state': 16, 'n_layers': 4},
}

results = {}
t_start = time.perf_counter()
total = len(configs) * len(SEEDS)
completed = 0

for cfg_name, cfg in configs.items():
    print(f'\n--- {cfg_name} ---')
    key = f'ablation_{cfg_name}'
    results[key] = {}
    for seed in SEEDS:
        print(f'  seed={seed}...', end=' ', flush=True)
        t0 = time.perf_counter()
        model = SSMWorldModel(state_dim=348, action_dim=17, **cfg)
        r = train_model(model, key, Xs, Xa, Y, Xv, Xav, Yv, seed)
        elapsed = time.perf_counter() - t0
        results[key][f'seed{seed}'] = r
        completed += 1
        eta = (time.perf_counter() - t_start) / completed * (total - completed)
        print(f'MSE={r["mse"]:.6f} ({elapsed:.0f}s, ETA {eta/60:.0f}min)')
    with open('experiments/ablation_5seeds.json', 'w') as f:
        json.dump(results, f, indent=2)

print('\n' + '='*60)
print('ABLATION (5 seeds)')
print('='*60)
for key in sorted(results.keys()):
    seeds = results[key]
    mses = [seeds[s]['mse'] for s in sorted(seeds.keys())]
    print(f'{key}: MSE={np.mean(mses):.4f}±{np.std(mses, ddof=1):.4f}')
print(f'Total: {(time.perf_counter()-t_start)/60:.1f}min')
