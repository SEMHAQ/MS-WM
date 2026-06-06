import torch, torch.nn as nn, numpy as np, sys, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel

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

def train_eval(ModelClass, name, ep_tr, ep_vl, T_tr, epochs=20):
    torch.manual_seed(42); np.random.seed(42)
    m = ModelClass(state_dim=S, action_dim=A, d_model=128, d_state=16, n_layers=4).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xs, Xa, Y = make_data(gen_ep(ep_tr, T_tr, 42), T_tr)
    Xv, Xav, Yv = make_data(gen_ep(80, T_tr, 999), T_tr)
    print(f'{name}: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params, train={len(Xs)}, val={len(Xv)}')
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
        if ep == epochs-1:
            m.eval()
            with torch.no_grad():
                vp = m(torch.from_numpy(Xv[:400]).float().to(device),
                       torch.from_numpy(Xav[:400]).float().to(device))
                vl = nn.functional.mse_loss(vp, torch.from_numpy(Yv[:400]).float().to(device)).item()
            print(f'  Epoch {ep+1}: train={np.mean(ls):.6f} val={vl:.6f}')
    return m

# Train both with SAME config
print('=== Training both with identical synthetic data ===')
ssm = train_eval(SSMWorldModel, 'SSM-WM', 400, 64, 64)
mamba = train_eval(MambaWorldModel, 'Mamba-WM', 400, 64, 64)

print('\n=== MSE (×10⁻³) ===')
ssm_mse, mamba_mse = [], []
for T in [16, 32, 64, 128, 256, 512]:
    eps = gen_ep(100, T, 1234)
    Xs, Xa, Yt = make_data(eps, T)
    N = min(300, len(Xs))
    sv = torch.from_numpy(Xs[:N]).float().to(device)
    av = torch.from_numpy(Xa[:N]).float().to(device)
    yv = torch.from_numpy(Yt[:N]).float().to(device)
    with torch.no_grad():
        ms = nn.functional.mse_loss(ssm(sv,av), yv).item()*1000
        mm = nn.functional.mse_loss(mamba(sv,av), yv).item()*1000
    ssm_mse.append(ms); mamba_mse.append(mm)
    print(f'T={T:3d}  SSM={ms:.2f}  Mamba={mm:.2f}')

print('\n=== Inference Time (ms) ===')
ssm_ms, mamba_ms = [], []
for T in [16, 32, 64, 128, 256, 512]:
    dummy_s = torch.randn(1, T, S).to(device)
    dummy_a = torch.randn(1, T-1, A).to(device)
    times_s, times_m = [], []
    with torch.no_grad():
        for _ in range(30): ssm(dummy_s, dummy_a)  # warmup
        for _ in range(100):
            t0 = time.time(); ssm(dummy_s, dummy_a); torch.cuda.synchronize()
            times_s.append((time.time()-t0)*1000)
        for _ in range(30): mamba(dummy_s, dummy_a)
        for _ in range(100):
            t0 = time.time(); mamba(dummy_s, dummy_a); torch.cuda.synchronize()
            times_m.append((time.time()-t0)*1000)
    ssm_ms.append(np.median(times_s)); mamba_ms.append(np.median(times_m))
    print(f'T={T:3d}  SSM={np.median(times_s):.1f}ms  Mamba={np.median(times_m):.1f}ms')

# Check if values are close enough to use in same table
print(f'\n=== Summary ===')
print(f'SSM MSE range:   {min(ssm_mse):.2f} - {max(ssm_mse):.2f}')
print(f'Mamba MSE range: {min(mamba_mse):.2f} - {max(mamba_mse):.2f}')
print(f'SSM Inference:   {min(ssm_ms):.1f} - {max(ssm_ms):.1f} ms')
print(f'Mamba Inference: {min(mamba_ms):.1f} - {max(mamba_ms):.1f} ms')
