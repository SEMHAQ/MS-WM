"""CEM-MPC K-value sensitivity analysis."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel

device = torch.device('cuda')
T = 32; H = 10; N_ITER = 2

def load_model():
    model = SSMWorldModel(state_dim=348, action_dim=17, d_model=128, d_state=16, n_layers=4)
    model.load_state_dict(torch.load('experiments/S4D-WM_humanoid_seed42.pth', map_location=device))
    return model.to(device).eval()

def cem_mpc(model, init_states, init_actions, K, n_elite):
    da = init_actions.shape[-1]
    states = init_states.expand(K, -1, -1).contiguous()
    actions = init_actions.expand(K, -1, -1).contiguous()
    mean = torch.zeros(H, da, device=device)
    std = torch.ones(H, da, device=device) * 0.5

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        eps = torch.randn(K, H, da, device=device)
        act = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(-1, 1)
        cost = torch.zeros(K, device=device)
        s, a = states.clone(), actions.clone()
        for h in range(H):
            with torch.no_grad(): pred = model(s, a)
            cost += torch.sum(pred**2, dim=-1) + 0.01 * torch.sum(act[:, h]**2, dim=-1)
            s = torch.cat([s[:, 1:], pred.unsqueeze(1)], dim=1)
            a = torch.cat([a[:, 1:], act[:, h:h+1]], dim=1)
        idx = cost.topk(n_elite, largest=False).indices
        el = act[idx]; mean = el.mean(0); std = el.std(0).clamp(min=0.01)
    elapsed = (time.perf_counter() - t0) * 1000

    # Compute tracking cost (quality metric)
    with torch.no_grad():
        final_cost = cost[idx].mean().item()

    return elapsed, final_cost

# Load data
eps_tr = [(np.load(f'data/humanoid/train/{f}')['states'], np.load(f'data/humanoid/train/{f}')['actions']) for f in sorted(os.listdir('data/humanoid/train'))[:10]]
eps_vl = [(np.load(f'data/humanoid/val/{f}')['states'], np.load(f'data/humanoid/val/{f}')['actions']) for f in sorted(os.listdir('data/humanoid/val'))[:5]]
m = np.concatenate([s for s,_ in eps_tr]).mean(0); std = np.concatenate([s for s,_ in eps_tr]).std(0)
ep_s, ep_a = eps_vl[0]; ep_sn = (ep_s - m) / (std + 1e-8)

model = load_model()

# Test different K values
K_values = [50, 100, 200, 500, 1000]
results = {}

print('K-value sensitivity analysis (H=10, 2 iter):')
for K in K_values:
    n_elite = max(10, K // 7)  # ~15% elite ratio
    times = []
    costs = []
    for trial in range(5):
        t = min(trial * 50, len(ep_sn) - T - 1)
        init_s = torch.FloatTensor(ep_sn[t:t+T].reshape(T, -1)).unsqueeze(0).to(device)
        init_a = torch.FloatTensor(ep_a[t:t+T-1].reshape(T-1, -1)).unsqueeze(0).to(device)
        elapsed, cost = cem_mpc(model, init_s, init_a, K, n_elite)
        times.append(elapsed)
        costs.append(cost)

    ms = np.mean(times); hz = 1000 / ms; cost_m = np.mean(costs)
    results[f'K{K}'] = {'ms': round(ms, 1), 'hz': round(hz, 1), 'cost': round(cost_m, 4), 'n_elite': n_elite}
    print(f'  K={K:4d} (elite={n_elite:3d}): {ms:.1f}ms ({hz:.1f}Hz), cost={cost_m:.4f}')

with open('experiments/cem_sensitivity.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nDone!')
