import torch, torch.nn as nn, numpy as np, sys, time, json
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel

device = torch.device('cuda')
S, A = 376, 17

def gen_ep(n, T, seed):
    eps = []
    for i in range(n):
        rng = np.random.RandomState(seed + i*7)
        s = rng.randn(S)*0.1
        states, actions = [s.copy()], []
        L = T + rng.randint(-5, 6)
        for t in range(L):
            a = 0.7*actions[-1]+0.3*rng.randn(A)*0.5 if actions else rng.randn(A)*0.5
            a376 = np.zeros(S); a376[:A] = a
            s = 0.95*s + 0.1*np.tanh(s*a376) + rng.randn(S)*0.01
            states.append(s.copy()); actions.append(a.copy())
        eps.append((np.array(states), np.array(actions)))
    return eps

def make_data(eps, T):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        if len(st) < T+1: continue
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(st[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(st[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

def train(ModelClass, name, ep_tr, T_tr, epochs=20):
    torch.manual_seed(42); np.random.seed(42)
    m = ModelClass(state_dim=S, action_dim=A, d_model=128, d_state=16, n_layers=4).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xs, Xa, Y = make_data(gen_ep(ep_tr, T_tr, 42), T_tr)
    Xv, Xav, Yv = make_data(gen_ep(80, T_tr, 999), T_tr)
    bs = 64
    for ep in range(epochs):
        m.train(); idx = np.random.permutation(len(Xs)); ls = []
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            p = m(torch.from_numpy(Xs[bi]).float().to(device),
                  torch.from_numpy(Xa[bi]).float().to(device))
            l = nn.functional.mse_loss(p, torch.from_numpy(Y[bi]).float().to(device))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); ls.append(l.item())
        sch.step()
    params = sum(p.numel() for p in m.parameters())/1e6
    print(f'{name}: {params:.2f}M params trained')
    return m

print('Training...')
ssm = train(SSMWorldModel, 'SSM-WM', 500, 64, 20)
lstm = train(LSTMWorldModel, 'LSTM-WM', 500, 64, 20)
mamba = train(MambaWorldModel, 'Mamba-WM', 500, 64, 20)

Ts = [16, 32, 64, 128, 256, 512]
results = {'mse': {}, 'time': {}}

print('\nMSE (x10^-3):')
for T in Ts:
    eps = gen_ep(100, T, 1234)
    Xs, Xa, Yt = make_data(eps, T)
    N = min(300, len(Xs))
    sv = torch.from_numpy(Xs[:N]).float().to(device)
    av = torch.from_numpy(Xa[:N]).float().to(device)
    yv = torch.from_numpy(Yt[:N]).float().to(device)
    with torch.no_grad():
        ms = nn.functional.mse_loss(ssm(sv,av), yv).item()*1000
        ml = nn.functional.mse_loss(lstm(sv,av), yv).item()*1000
        mm = nn.functional.mse_loss(mamba(sv,av), yv).item()*1000
    results['mse'][T] = {'lstm': round(ml,2), 'mamba': round(mm,2), 'ssm': round(ms,2)}
    print(f'  T={T:3d}  LSTM={ml:.2f}  Mamba={mm:.2f}  SSM={ms:.2f}')

print('\nInference Time (ms):')
for T in Ts:
    ds = torch.randn(1, T, S).to(device)
    da = torch.randn(1, T-1, A).to(device)
    tl, tm, ts = [], [], []
    with torch.no_grad():
        for _ in range(30): ssm(ds,da); lstm(ds,da); mamba(ds,da)
        for _ in range(100):
            t0=time.time(); ssm(ds,da); torch.cuda.synchronize(); ts.append((time.time()-t0)*1000)
        for _ in range(100):
            t0=time.time(); lstm(ds,da); torch.cuda.synchronize(); tl.append((time.time()-t0)*1000)
        for _ in range(100):
            t0=time.time(); mamba(ds,da); torch.cuda.synchronize(); tm.append((time.time()-t0)*1000)
    results['time'][T] = {'lstm': round(np.median(tl),1), 'mamba': round(np.median(tm),1), 'ssm': round(np.median(ts),1)}
    print(f'  T={T:3d}  LSTM={np.median(tl):.1f}  Mamba={np.median(tm):.1f}  SSM={np.median(ts):.1f}')

with open('/tmp/fair_results.json', 'w') as f:
    json.dump(results, f)
print('\nDone! Results saved.')
