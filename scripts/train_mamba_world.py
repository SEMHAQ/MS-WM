"""训练Mamba-WM并在不同序列长度上评估MSE — 接口与SSM-WM对齐"""
import types, sys
fake_gen = types.ModuleType('transformers.generation')
fake_gen.GreedySearchDecoderOnlyOutput = type('x', (), {})
fake_gen.SampleDecoderOnlyOutput = type('x', (), {})
fake_gen.TextStreamer = type('x', (), {})
sys.modules['transformers.generation'] = fake_gen

import torch
import torch.nn as nn
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.mamba_world_model import MambaWorldModel

device = torch.device('cuda')
state_dim = 376
action_dim = 17

def gen_episodes(n_episodes, T, seed=42):
    """生成episode数据, 返回states和actions列表"""
    episodes = []
    for ep in range(n_episodes):
        rng = np.random.RandomState(seed + ep * 7)
        s = rng.randn(state_dim) * 0.1
        ep_len = T + rng.randint(-10, 11)
        states, actions = [s.copy()], []
        for t in range(ep_len):
            a = 0.7 * actions[-1] + 0.3 * rng.randn(action_dim) * 0.5 if actions else rng.randn(action_dim) * 0.5
            a_pad = np.zeros(state_dim)
            a_pad[:action_dim] = a
            s = 0.95 * s + 0.1 * np.tanh(s * a_pad) + rng.randn(state_dim) * 0.01
            states.append(s.copy())
            actions.append(a.copy())
        episodes.append((np.array(states), np.array(actions)))
    return episodes

def make_step_data(episodes, T):
    """构造单步预测数据: 给定T步历史, 预测下一状态"""
    states_batch, actions_batch, targets = [], [], []
    for states, actions in episodes:
        if len(states) < T + 1:
            continue
        for start in range(0, len(states) - T, T):
            s_chunk = states[start:start+T+1]  # T+1个状态
            a_chunk = actions[start:start+T]    # T个动作
            if len(a_chunk) < T:
                continue
            states_batch.append(s_chunk[:T])     # (T, state_dim)
            actions_batch.append(a_chunk[:T-1])   # (T-1, action_dim)
            targets.append(s_chunk[T])            # (state_dim,) 下一状态
    return (np.array(states_batch), np.array(actions_batch),
            np.array(targets))

print("Generating training data (T=64)...")
episodes = gen_episodes(400, 64, seed=42)
episodes_val = gen_episodes(100, 64, seed=999)

X_s, X_a, Y = make_step_data(episodes, 64)
Xv_s, Xv_a, Yv = make_step_data(episodes_val, 64)
print(f"Train: {len(X_s)}, Val: {len(Xv_s)}")

model = MambaWorldModel(state_dim=state_dim, action_dim=action_dim,
                        d_model=128, d_state=16, n_layers=4).to(device)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Params: {params:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

batch_size = 64
for epoch in range(20):
    model.train()
    idx = np.random.permutation(len(X_s))
    losses = []
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i+batch_size]
        s_batch = torch.from_numpy(X_s[bi]).float().to(device)
        a_batch = torch.from_numpy(X_a[bi]).float().to(device)
        y_batch = torch.from_numpy(Y[bi]).float().to(device)
        pred = model(s_batch, a_batch)
        loss = nn.functional.mse_loss(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()

    model.eval()
    with torch.no_grad():
        sv = torch.from_numpy(Xv_s[:500]).float().to(device)
        av = torch.from_numpy(Xv_a[:500]).float().to(device)
        yv = torch.from_numpy(Yv[:500]).float().to(device)
        val_pred = model(sv, av)
        val_loss = nn.functional.mse_loss(val_pred, yv).item()
    if epoch % 5 == 0 or epoch == 19:
        print(f"  Epoch {epoch+1}: train={np.mean(losses):.6f} val={val_loss:.6f}")

torch.save(model.state_dict(), "experiments/Mamba-WM_4layer.pth")
print("Saved Mamba-WM_4layer.pth")

print("\nEvaluating MSE at different T...")
model.eval()
results = {}
for T in [16, 32, 64, 128, 256, 512]:
    eps = gen_episodes(100, T, seed=1234)
    Xs, Xa, Yt = make_step_data(eps, T)
    with torch.no_grad():
        sv = torch.from_numpy(Xs[:300]).float().to(device)
        av = torch.from_numpy(Xa[:300]).float().to(device)
        yv = torch.from_numpy(Yt[:300]).float().to(device)
        pred = model(sv, av)
        mse = nn.functional.mse_loss(pred, yv).item() * 1000
    results[T] = round(mse, 2)
    print(f"  T={T}: MSE={mse:.2f}")

print("\n=== Table 4 Mamba MSE ===")
for T, mse in results.items():
    print(f"{T}\t{mse}")
