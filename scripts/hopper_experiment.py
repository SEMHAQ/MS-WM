"""Run experiments on Hopper dataset (11D) with 5 seeds."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel

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

def train_model(ModelClass, name, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
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

# Check if Hopper data exists
if not os.path.exists('data/hopper'):
    print('Hopper data not found. Downloading...')
    os.system('python3 scripts/download_d4rl.py --env hopper')

# Load data
eps_tr = load_eps('data/hopper', 'train')
eps_vl = load_eps('data/hopper', 'val')
m, s = stats(eps_tr)
Xs, Xa, Y = make_data(eps_tr, T, m, s)
Xv, Xav, Yv = make_data(eps_vl, T, m, s)
sd = Xs.shape[-1]; ad = Xa.shape[-1]
print(f'Hopper: state_dim={sd}, action_dim={ad}, Train: {len(Xs)}, Val: {len(Xv)}')

models = {
    'LSTM-WM': (LSTMWorldModel, lambda: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
    'Transformer-WM': (TransformerWorldModel, lambda: {'state_dim': sd, 'action_dim': ad, 'd_model': 64, 'nhead': 4, 'n_layers': 2}),
    'Mamba-WM': (MambaWorldModel, lambda: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
    'S4D-WM': (SSMWorldModel, lambda: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
    'GRU-WM': (GRUWorldModel, lambda: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
}

results = {}
t_start = time.perf_counter()
for model_name, (ModelClass, kwargs_fn) in models.items():
    print(f'\n{model_name}:')
    results[model_name] = {}
    for seed in SEEDS:
        print(f'  seed={seed}...', end=' ', flush=True)
        t0 = time.perf_counter()
        r = train_model(ModelClass, f'{model_name}_hopper', kwargs_fn(), Xs, Xa, Y, Xv, Xav, Yv, seed)
        elapsed = time.perf_counter() - t0
        results[model_name][f'seed{seed}'] = r
        print(f'MSE={r["mse"]:.6f} R²={r["r2"]:.6f} ({elapsed:.0f}s)')
    with open('experiments/hopper_results.json', 'w') as f:
        json.dump(results, f, indent=2)

print('\n' + '='*60)
print('HOPPER RESULTS (5 seeds)')
print('='*60)
for name in sorted(results.keys()):
    seeds = results[name]
    mses = [seeds[s]['mse'] for s in sorted(seeds.keys())]
    r2s = [seeds[s]['r2'] for s in sorted(seeds.keys())]
    print(f'{name}: MSE={np.mean(mses)*100:.2f}±{np.std(mses, ddof=1)*100:.2f} (×10⁻²), R²={np.mean(r2s):.4f}±{np.std(r2s, ddof=1):.4f}')
print(f'Total: {(time.perf_counter()-t_start)/60:.1f}min')
