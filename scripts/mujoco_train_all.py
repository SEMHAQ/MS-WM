import torch, torch.nn as nn, numpy as np, sys, os, time, json
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel

device = torch.device('cuda')
DATA_DIR = 'data/humanoid'

def load_episodes(split, max_eps=None):
    d = os.path.join(DATA_DIR, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    if max_eps: files = files[:max_eps]
    episodes = []
    for f in files:
        d2 = np.load(os.path.join(d, f))
        episodes.append((d2['states'], d2['actions']))
    return episodes

def make_step_data(episodes, T):
    Xs, Xa, Y = [], [], []
    for st, ac in episodes:
        if len(st) < T+1: continue
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(st[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(st[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

def train_model(ModelClass, name, kwargs, T=32, epochs=30, lr=1e-3):
    torch.manual_seed(42); np.random.seed(42)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    eps_tr = load_episodes('train', 400)
    eps_vl = load_episodes('val', 100)
    Xs, Xa, Y = make_step_data(eps_tr, T)
    Xv, Xav, Yv = make_step_data(eps_vl, T)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'{name}: {params:.2f}M params, train={len(Xs)}, val={len(Xv)}')
    
    best_val = float('inf')
    bs = 64
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        losses = []
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            sb = torch.from_numpy(Xs[bi]).float().to(device)
            ab = torch.from_numpy(Xa[bi]).float().to(device)
            yb = torch.from_numpy(Y[bi]).float().to(device)
            pred = model(sb, ab)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sch.step()
        
        model.eval()
        with torch.no_grad():
            vp = model(torch.from_numpy(Xv[:500]).float().to(device),
                       torch.from_numpy(Xav[:500]).float().to(device))
            vl = nn.functional.mse_loss(vp, torch.from_numpy(Yv[:500]).float().to(device)).item()
        if vl < best_val: best_val = vl
        
        if ep % 10 == 0 or ep == epochs-1:
            print(f'  Epoch {ep+1}: train={np.mean(losses):.6f} val={vl:.6f} best={best_val:.6f}')
    
    # Save checkpoint
    path = f'experiments/{name}_mujoco.pth'
    torch.save(model.state_dict(), path)
    print(f'  Saved to {path}')
    return model, params

# Train all 4 models on MuJoCo
print('=== Training on MuJoCo Humanoid (T=32) ===')
S, A = 376, 17

print('\n--- SSM-WM ---')
ssm, ssm_p = train_model(SSMWorldModel, 'SSM-WM',
    dict(state_dim=S, action_dim=A, d_model=128, d_state=16, n_layers=4))

print('\n--- LSTM-WM ---')
lstm, lstm_p = train_model(LSTMWorldModel, 'LSTM-WM',
    dict(state_dim=S, action_dim=A, hidden_dim=128, n_layers=4))

print('\n--- Mamba-WM ---')
mamba, mamba_p = train_model(MambaWorldModel, 'Mamba-WM',
    dict(state_dim=S, action_dim=A, d_model=128, d_state=16, n_layers=4))

print('\n--- Transformer-WM ---')
trans, trans_p = train_model(TransformerWorldModel, 'Trans-WM',
    dict(state_dim=S, action_dim=A, d_model=128, nhead=4, n_layers=3))

# Save params
with open('/tmp/mujoco_params.json', 'w') as f:
    json.dump({'ssm': ssm_p, 'lstm': lstm_p, 'mamba': mamba_p, 'trans': trans_p}, f)
print('\nAll models trained!')
