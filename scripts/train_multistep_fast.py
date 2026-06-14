"""Fast multi-step training for S4D-WM on D4RL Humanoid.
Reduced data + single best config for speed."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel

device = torch.device('cuda')
STATE_DIM = 348; ACTION_DIM = 17; T = 32; BS = 64; SEED = 42

def load_episodes(data_dir, split, max_eps=None):
    d = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    if max_eps: files = files[:max_eps]
    return [(np.load(os.path.join(d, f))['states'], np.load(os.path.join(d, f))['actions']) for f in files]

def compute_stats(episodes):
    all_s = np.concatenate([st for st, _ in episodes], axis=0)
    return all_s.mean(axis=0), all_s.std(axis=0)

def make_data(episodes, T, mean, std, max_samples=5000):
    Xs, Xa, Y = [], [], []
    for st, ac in episodes:
        if len(st) < T+1: continue
        st_n = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(st_n[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(st_n[j+T])
            if len(Xs) >= max_samples: break
        if len(Xs) >= max_samples: break
    return np.array(Xs), np.array(Xa), np.array(Y)

def make_ms_data(episodes, T, H, mean, std, max_samples=1000):
    Xs, Xa, Ys = [], [], []
    for st, ac in episodes:
        if len(st) < T + H: continue
        st_n = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T-H+1, T):
            if j+T+H > len(st): break
            Xs.append(st_n[j:j+T]); Xa.append(ac[j:j+T-1]); Ys.append(st_n[j+T:j+T+H])
            if len(Xs) >= max_samples: break
        if len(Xs) >= max_samples: break
    return np.array(Xs), np.array(Xa), np.array(Ys)

if __name__ == '__main__':
    print("Fast multi-step training", flush=True)
    eps_tr = load_episodes('data/humanoid', 'train', 930)
    eps_vl = load_episodes('data/humanoid', 'val', 233)
    mean, std = compute_stats(eps_tr)
    
    # Single-step data (subsampled)
    Xs1, Xa1, Y1 = make_data(eps_tr, T, mean, std, 5000)
    Xv1, Xav1, Yv1 = make_data(eps_vl, T, mean, std, 2000)
    print(f"Single-step: train={len(Xs1)}, val={len(Xv1)}", flush=True)
    
    # Multi-step data
    XsH, XaH, YH = make_ms_data(eps_tr[:200], T, 8, mean, std, 1000)
    print(f"Multi-step: {len(XsH)} samples, H=8", flush=True)
    
    kwargs = {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 128, 'd_state': 16, 'n_layers': 4}
    
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = SSMWorldModel(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    loss_fn = nn.MSELoss()
    
    best_val = float('inf'); patience = 0; lam = 0.8
    
    for ep in range(50):
        model.train()
        idx = np.random.permutation(len(Xs1))
        losses = []
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            xs = torch.FloatTensor(Xs1[bi]).to(device)
            xa = torch.FloatTensor(Xa1[bi]).to(device)
            y = torch.FloatTensor(Y1[bi]).to(device)
            pred = model(xs, xa)
            loss_s = loss_fn(pred, y)
            
            # Multi-step rollout
            mbi = np.random.choice(len(XsH), min(BS, len(XsH)), replace=False)
            xs_m = torch.FloatTensor(XsH[mbi]).to(device)
            xa_m = torch.FloatTensor(XaH[mbi]).to(device)
            ys_m = torch.FloatTensor(YH[mbi]).to(device)
            
            loss_ms = 0.0
            cur_s, cur_a = xs_m.clone(), xa_m.clone()
            for h in range(8):
                ph = model(cur_s, cur_a)
                loss_ms += loss_fn(ph, ys_m[:, h, :])
                if h < 7:
                    cur_s = torch.cat([cur_s[:, 1:, :], ph.unsqueeze(1)], dim=1)
            loss_ms /= 8
            
            loss = (1 - lam) * loss_s + lam * loss_ms
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sch.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(torch.FloatTensor(Xv1).to(device), 
                                     torch.FloatTensor(Xav1).to(device)),
                              torch.FloatTensor(Yv1).to(device)).item()
        
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1}: train={np.mean(losses):.6f}, val={val_loss:.6f}", flush=True)
        
        if val_loss < best_val:
            best_val = val_loss; patience = 0
            torch.save(model.state_dict(), 'experiments/S4D-WM_multistep.pth')
        else:
            patience += 1
        if patience >= 15:
            print(f"  Early stop at {ep+1}", flush=True)
            break
    
    print(f"\nBest val: {best_val:.6f}", flush=True)
    
    # Evaluate multi-step
    model.load_state_dict(torch.load('experiments/S4D-WM_multistep.pth'))
    model.eval()
    episodes = eps_vl[:50]
    for H in [1, 4, 8, 16]:
        mses = []
        for st, ac in episodes:
            if len(st) < T + H: continue
            st_n = (st - mean) / (std + 1e-8)
            for j in range(0, len(st)-T-H, T):
                if j+T+H > len(st): break
                cs = st_n[j:j+T].copy(); ca = ac[j:j+T-1].copy()
                for h in range(H):
                    with torch.no_grad():
                        p = model(torch.FloatTensor(cs).unsqueeze(0).to(device),
                                  torch.FloatTensor(ca).unsqueeze(0).to(device)).cpu().numpy()[0]
                    if j+T+h < len(ac):
                        cs = np.concatenate([cs[1:], p.reshape(1,-1)])
                        ca = np.concatenate([ca[1:], ac[j+T+h].reshape(1,-1)])
                mses.append(np.mean((p - st_n[j+T+H-1])**2))
        print(f"  H={H}: MSE={np.mean(mses):.6f}", flush=True)
