"""Train GRU-WM on Humanoid with per-epoch logging for training curves."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.baselines import GRUWorldModel

device = torch.device('cuda')
SEED = 42
EPOCHS = 100
BS = 64
T = 32
LR = 5e-4

# Load data
def load_eps(d, split, mx):
    eps = []
    fd = os.path.join(d, split)
    for f in sorted(os.listdir(fd))[:mx]:
        d2 = np.load(os.path.join(fd, f))
        eps.append((d2['states'], d2['actions']))
    return eps

eps_tr = load_eps('data/humanoid', 'train', 1000)
eps_vl = load_eps('data/humanoid', 'val', 163)
all_s = np.concatenate([e[0] for e in eps_tr])
mean, std = all_s.mean(0), all_s.std(0) + 1e-8

def make(eps, T, m, s):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        sn = (st - m) / s
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(sn[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

Xs, Xa, Y = make(eps_tr, T, mean, std)
Xv, Xav, Yv = make(eps_vl, T, mean, std)
print(f"Train: {len(Xs)}, Val: {len(Xv)}")

torch.manual_seed(SEED); np.random.seed(SEED)
model = GRUWorldModel(state_dim=348, action_dim=17, hidden_dim=128, n_layers=4).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.MSELoss()

train_log = []
t0 = time.perf_counter()
for ep in range(EPOCHS):
    model.train()
    idx = np.random.permutation(len(Xs))
    train_losses = []
    for i in range(0, len(idx), BS):
        bi = idx[i:i+BS]
        pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
        loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        train_losses.append(loss.item())
    sch.step()
    
    tl = np.mean(train_losses)
    model.eval()
    with torch.no_grad():
        vl = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)),
                     torch.FloatTensor(Yv).to(device)).item()
    train_log.append({'epoch': ep+1, 'train': tl, 'val': vl})
    if (ep+1) % 20 == 0 or ep == 0:
        print(f"  Epoch {ep+1}: train={tl:.4f} val={vl:.4f}")

elapsed = time.perf_counter() - t0
print(f"Done in {elapsed/60:.1f}min")

# Save
with open('experiments/d4rl_all_experiments.json') as f:
    data = json.load(f)
data['training_logs']['GRU-WM_d4rl'] = train_log
with open('experiments/d4rl_all_experiments.json', 'w') as f:
    json.dump(data, f, indent=2)
print("Saved GRU-WM training log to d4rl_all_experiments.json")
