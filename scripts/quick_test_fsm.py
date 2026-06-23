"""快速测试不同配置的FSM"""
import torch, torch.nn as nn, numpy as np, sys, os, time
sys.path.insert(0, '.')
from src.models.fusion_ssm import FSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 100
BS = 256
T = 32

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

def train_eval(d_model, n_layers, Xs, Xa, Y, Xv, Xav, Yv):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = FSM(348, 17, d_model=d_model, d_state=16, n_layers=n_layers, window_size=8).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0

    t0 = time.time()
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
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 10: break

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

    return {'mse': round(mse, 6), 'r2': round(r2, 4), 'params': round(params, 3), 'time': round(elapsed, 1), 'epoch': best_ep}

# 加载数据
print('加载Humanoid数据...', flush=True)
eps_tr = load_eps('data/humanoid', 'train', 200)
eps_vl = load_eps('data/humanoid', 'val', 50)
m, s = stats(eps_tr)
Xs, Xa, Y = make_data(eps_tr, T, m, s)
Xv, Xav, Yv = make_data(eps_vl, T, m, s)
print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

# 测试不同配置
configs = [
    {'d_model': 64, 'n_layers': 4, 'name': 'D=64,L=4'},
    {'d_model': 96, 'n_layers': 4, 'name': 'D=96,L=4'},
    {'d_model': 128, 'n_layers': 2, 'name': 'D=128,L=2'},
    {'d_model': 128, 'n_layers': 4, 'name': 'D=128,L=4 (默认)'},
    {'d_model': 192, 'n_layers': 4, 'name': 'D=192,L=4'},
]

print('\n=== FSM不同配置测试 ===', flush=True)
print(f'{"配置":<16} {"MSE":<10} {"R²":<10} {"参数(M)":<10} {"时间(s)":<10}', flush=True)
print('-'*60, flush=True)

for cfg in configs:
    print(f'{cfg["name"]:<16}', end=' ', flush=True)
    r = train_eval(cfg['d_model'], cfg['n_layers'], Xs, Xa, Y, Xv, Xav, Yv)
    print(f'{r["mse"]:<10} {r["r2"]:<10} {r["params"]:<10} {r["time"]:<10}', flush=True)

print('\nDone!', flush=True)
