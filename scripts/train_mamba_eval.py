"""Train Mamba-WM and evaluate MSE at different sequence lengths."""
import types, sys
fake_gen = types.ModuleType('transformers.generation')
fake_gen.GreedySearchDecoderOnlyOutput = type('GreedySearchDecoderOnlyOutput', (), {})
fake_gen.SampleDecoderOnlyOutput = type('SampleDecoderOnlyOutput', (), {})
fake_gen.TextStreamer = type('TextStreamer', (), {})
sys.modules['transformers.generation'] = fake_gen

import torch
import torch.nn as nn
import numpy as np
from mamba_ssm import Mamba

device = torch.device('cuda')
STATE_DIM = 376
ACTION_DIM = 17
INPUT_DIM = STATE_DIM + ACTION_DIM  # 393
d_model = 128
d_state = 16
T_train = 64

class MambaWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(INPUT_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.dec = nn.Linear(d_model, STATE_DIM)

    def forward(self, x):
        z = self.enc(x)
        z = self.mamba(z)
        z = self.norm(z)
        return self.dec(z)

def gen_data(n_episodes, T, seed=42):
    rng = np.random.RandomState(seed)
    states, actions = [], []
    for _ in range(n_episodes):
        s = rng.randn(T, STATE_DIM).astype(np.float32) * 0.1
        a = rng.randn(T, ACTION_DIM).astype(np.float32) * 0.5
        a_pad = np.zeros((T, STATE_DIM), dtype=np.float32)
        a_pad[:, :ACTION_DIM] = a
        s_next = 0.95 * s + 0.1 * np.tanh(s * a_pad)
        states.append(s)
        actions.append(a)
    return np.array(states), np.array(actions)

print("Training Mamba-WM...")
model = MambaWM().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

s_train, a_train = gen_data(400, T_train, seed=42)
s_val, a_val = gen_data(100, T_train, seed=999)

x_train = np.concatenate([s_train[:, :-1], a_train[:, :-1]], axis=-1)
y_train = s_train[:, 1:]
x_val = np.concatenate([s_val[:, :-1], a_val[:, :-1]], axis=-1)
y_val = s_val[:, 1:]

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

batch_size = 64
for epoch in range(20):
    model.train()
    idx = np.random.permutation(len(x_train))
    losses = []
    for i in range(0, len(idx), batch_size):
        xb = torch.from_numpy(x_train[idx[i:i+batch_size]]).to(device)
        yb = torch.from_numpy(y_train[idx[i:i+batch_size]]).to(device)
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(x_val[:64]).to(device)
        yv = torch.from_numpy(y_val[:64]).to(device)
        val_loss = nn.functional.mse_loss(model(xv), yv).item()
    if epoch % 5 == 0 or epoch == 19:
        print(f"  Epoch {epoch+1}: train={np.mean(losses):.5f} val={val_loss:.5f}")

torch.save(model.state_dict(), "experiments/Mamba-WM.pth")
print("Saved Mamba-WM.pth")

print("\nEvaluating MSE at different sequence lengths...")
model.eval()
results = {}
for T in [16, 32, 64, 128, 256, 512]:
    s_t, a_t = gen_data(100, T, seed=1234)
    x_t = np.concatenate([s_t[:, :-1], a_t[:, :-1]], axis=-1)
    y_t = s_t[:, 1:]
    with torch.no_grad():
        xv = torch.from_numpy(x_t[:64]).to(device)
        yv = torch.from_numpy(y_t[:64]).to(device)
        mse = nn.functional.mse_loss(model(xv), yv).item() * 1000
    results[T] = round(mse, 2)
    print(f"  T={T}: MSE={mse:.2f}")

print("\n=== Table 4 Mamba MSE ===")
for T, mse in results.items():
    print(f"{T}\t{mse}")
