"""HumanoidStandup 序列长度敏感性实验"""
import torch, torch.nn as nn, numpy as np, sys, os, json
sys.path.insert(0, '.')
from src.models.mimo_world_model import MIMOWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]
TS = [8, 16, 32, 64, 128]

print(f'Device: {device}', flush=True)

# 自动检测维度
f0 = sorted([f for f in os.listdir('data/humanoid_standup/train') if f.endswith('.npz')])[0]
d = np.load('data/humanoid_standup/train/' + f0)
state_dim = d['states'].shape[1]
action_dim = d['actions'].shape[1]
print(f'state_dim={state_dim}, action_dim={action_dim}', flush=True)

def load_eps(directory, split):
    dd = os.path.join(directory, split)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs]

def get_stats(eps):
    a = np.concatenate([s for s,_ in eps])
    return a.mean(0), a.std(0)

def make_data(eps, T, mean, std):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        if len(st) < T+1: continue
        sn = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, max(T, 1)):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(sn[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

def train_model(Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MIMOWorldModel(state_dim, action_dim, d_model=96, d_state=16, n_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0
    for ep in range(100):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), 1024):
            bi = idx[i:i+1024]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0
        else: pat += 1
        if pat >= 20: break
    return model

if __name__ == '__main__':
    RESULTS_FILE = 'experiments/seqlen_standup.json'
    os.makedirs('experiments', exist_ok=True)
    results = {} if not os.path.exists(RESULTS_FILE) else json.load(open(RESULTS_FILE))

    eps_tr = load_eps('data/humanoid_standup', 'train')
    eps_vl = load_eps('data/humanoid_standup', 'val')
    mean, std = get_stats(eps_tr)

    for T in TS:
        key = f'MIMO-WM_T{T}'
        if key in results and len(results[key]) >= len(SEEDS):
            print(f'T={T}: 已有结果，跳过', flush=True)
            continue
        results[key] = {}
        print(f'\nT={T}', flush=True)
        Xs, Xa, Y = make_data(eps_tr, T, mean, std)
        Xv, Xav, Yv = make_data(eps_vl, T, mean, std)
        print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

        for seed in SEEDS:
            sk = f'seed{seed}'
            if sk in results[key]:
                print(f'  seed={seed}: 已有', flush=True)
                continue
            model = train_model(Xs, Xa, Y, Xv, Xav, Yv, seed)
            model.eval()
            with torch.no_grad():
                pred = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
                mse = nn.MSELoss()(pred, torch.FloatTensor(Yv).to(device)).item()
                Yv_g = torch.FloatTensor(Yv).to(device)
                r2 = 1 - ((Yv_g - pred)**2).sum().item() / ((Yv_g - Yv_g.mean(0))**2).sum().item()
            results[key][sk] = {'mse': round(mse, 6), 'r2': round(r2, 4)}
            print(f'  seed={seed}: MSE={mse*100:.2f}, R2={r2:.4f}', flush=True)
            json.dump(results, open(RESULTS_FILE, 'w'), indent=2)

    print('\n' + '='*60)
    print('序列长度敏感性 — HumanoidStandup')
    print('='*60)
    for T in TS:
        key = f'MIMO-WM_T{T}'
        if key in results:
            vals = list(results[key].values())
            mses = [v['mse']*100 for v in vals]
            r2s = [v['r2'] for v in vals]
            print(f'T={T:<4} MSE={np.mean(mses):.2f}±{np.std(mses):.2f}  R2={np.mean(r2s):.4f}±{np.std(r2s):.4f}')
    print('\nDone!', flush=True)
