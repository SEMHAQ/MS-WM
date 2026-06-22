"""10-seed experiments for all models on all datasets.
Results saved to experiments/results_10seeds.json"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel
from src.models.fusion_ssm import FSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
BS = 64
LR = 5e-4
T = 32

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

def train_model(ModelClass, name, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, ds_name, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float('inf'); pat = 0; best_ep = 0
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
        with torch.no_grad():
            vl = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)), torch.FloatTensor(Yv).to(device)).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1; torch.save(model.state_dict(), f'experiments/{name}_{ds_name}_seed{seed}.pth')
        else: pat += 1
        if pat >= 20: break

    # Evaluate on validation
    model.load_state_dict(torch.load(f'experiments/{name}_{ds_name}_seed{seed}.pth', map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        mse = loss_fn(pred, torch.FloatTensor(Yv).to(device)).item()
        ss_r = torch.sum((torch.FloatTensor(Yv).to(device) - pred)**2).item()
        ss_t = torch.sum((torch.FloatTensor(Yv).to(device) - torch.mean(torch.FloatTensor(Yv).to(device), dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

    # Measure inference time (B=1)
    model.eval()
    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        # Warmup
        for _ in range(5): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        inf_time = (time.perf_counter() - t0) / 100 * 1000  # ms

    # Multi-step prediction (H=1,4,8,16)
    multi_step = {}
    for H in [1, 4, 8, 16]:
        mse_h = []
        for i in range(min(50, len(Xv))):
            seq_s = torch.FloatTensor(Xv[i:i+1]).to(device)
            seq_a = torch.FloatTensor(Xav[i:i+1]).to(device)
            true_next = []
            for h in range(H):
                idx_t = i + h
                if idx_t < len(Yv):
                    true_next.append(Yv[idx_t])
            if len(true_next) < H: continue
            preds = []
            cur_s = seq_s.clone()
            cur_a = seq_a.clone()
            for h in range(H):
                with torch.no_grad():
                    p = model(cur_s, cur_a)
                preds.append(p.cpu().numpy()[0])
                cur_s = torch.cat([cur_s[:, 1:], p.unsqueeze(1)], dim=1)
            preds = np.array(preds)
            true_next = np.array(true_next)
            mse_h.append(np.mean((preds - true_next)**2))
        multi_step[f'H{H}'] = round(np.mean(mse_h), 6) if mse_h else None

    return {
        'mse': round(mse, 6),
        'r2': round(r2, 6),
        'best_epoch': best_ep,
        'inf_time_ms': round(inf_time, 2),
        'multi_step': multi_step
    }

# Dataset configs
datasets = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17, 'train_max': 930, 'val_max': 233},
    'ant': {'dir': 'data/ant', 'sd': 105, 'ad': 8, 'train_max': 837, 'val_max': 210},
    'hopper': {'dir': 'data/hopper', 'sd': 11, 'ad': 6, 'train_max': 278, 'val_max': 70},
}

models = {
    'LSTM-WM': (LSTMWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
    'Transformer-WM': (TransformerWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'nhead': 4, 'n_layers': 4}),
    'Mamba-WM': (MambaWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
    'S4D-WM': (SSMWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
    'GRU-WM': (GRUWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
    'FSM-WM': (FSM, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4, 'window_size': 8}),
}

os.makedirs('experiments', exist_ok=True)
results = {}

for ds_name, ds_cfg in datasets.items():
    print(f'\n{"="*60}\nDataset: {ds_name}\n{"="*60}')
    eps_tr = load_eps(ds_cfg['dir'], 'train', ds_cfg['train_max'])
    eps_vl = load_eps(ds_cfg['dir'], 'val', ds_cfg['val_max'])
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'  Train: {len(Xs)}, Val: {len(Xv)}')

    for model_name, (ModelClass, kwargs_fn) in models.items():
        kwargs = kwargs_fn(ds_cfg['sd'], ds_cfg['ad'])
        key = f'{model_name}_{ds_name}'
        results[key] = {}

        for seed in SEEDS:
            print(f'  {model_name} seed={seed}...', end=' ', flush=True)
            t0 = time.perf_counter()
            try:
                r = train_model(ModelClass, model_name, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, ds_name)
                elapsed = time.perf_counter() - t0
                results[key][f'seed{seed}'] = r
                print(f'MSE={r["mse"]:.6f} R²={r["r2"]:.6f} ({elapsed/60:.1f}min)')
            except Exception as e:
                print(f'ERROR: {e}')
                results[key][f'seed{seed}'] = {'error': str(e)}

            # Save intermediate results
            with open('experiments/results_10seeds.json', 'w') as f:
                json.dump(results, f, indent=2)

# Print summary
print('\n' + '='*60)
print('SUMMARY (10 seeds)')
print('='*60)
for key, seeds in results.items():
    valid = [seeds[s] for s in seeds if 'mse' in seeds[s]]
    if not valid: continue
    mses = [r['mse'] for r in valid]
    r2s = [r['r2'] for r in valid]
    times = [r['inf_time_ms'] for r in valid]
    print(f'{key}: MSE={np.mean(mses):.4f}±{np.std(mses):.4f}, R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}, InfTime={np.mean(times):.1f}ms')

print('\nDone! Results saved to experiments/results_10seeds.json')
