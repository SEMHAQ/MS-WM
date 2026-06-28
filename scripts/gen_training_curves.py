"""生成训练曲线数据和图表（只跑1个seed，快速）"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel, DiagSSM
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 100
BS = 256
T = 32

# MS-WM模型
class MultiScaleModel(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5, fusion_type='gate'):
        super().__init__()
        self.state_dim = state_dim
        self.fusion_type = fusion_type
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.slow_ssm = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)}) for _ in range(n_layers)])
        self.fast_ssm = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)}) for _ in range(n_layers)])
        self.local_attn = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model)}) for _ in range(n_layers)])
        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 3, 3), nn.Softmax(dim=-1))
        self.fusion_proj = nn.Linear(d_model, state_dim)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        slow_h = h
        for b in self.slow_ssm: residual = slow_h; slow_h = residual + b['ssm'](b['norm'](slow_h))
        fast_h = h
        for b in self.fast_ssm: residual = fast_h; fast_h = residual + b['ssm'](b['norm'](fast_h))
        local_h = h
        for b in self.local_attn: residual = local_h; local_h = residual + b['conv'](b['norm'](local_h).transpose(1,2)).transpose(1,2)
        features = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
        gate = self.fusion_gate(features)
        stacked = torch.stack([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=1)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        return states[:, -1, :] + self.fusion_proj(fused)

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

def get_model_config(model_name, sd, ad):
    configs = {
        'LSTM-WM': (LSTMWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'Transformer-WM': (TransformerWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'nhead': 4, 'n_layers': 4}),
        'GRU-WM': (GRUWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'Mamba-WM': (MambaWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
        'S4D-WM': (SSMWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
        'MS-WM': (MultiScaleModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5}),
    }
    return configs[model_name]

def train_with_logging(model_name, Xs, Xa, Y, Xv, Xav, Yv, seed=SEED):
    """训练模型并记录训练曲线"""
    sd, ad = Xs.shape[2], Xa.shape[2]
    ModelClass, kwargs = get_model_config(model_name, sd, ad)
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)

    train_losses = []
    val_losses = []
    best_val = float('inf'); pat = 0

    for ep in range(EPOCHS):
        model.train()
        epoch_loss = 0; n_batches = 0
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            epoch_loss += loss.item(); n_batches += 1
        sch.step()
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        val_losses.append(val_loss)

        if val_loss < best_val: best_val = val_loss; pat = 0
        else: pat += 1
        if pat >= 20: break

    return train_losses, val_losses

# ============================================================
if __name__ == '__main__':
    print('加载Humanoid数据...', flush=True)
    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    RESULTS_FILE = 'experiments/training_curves.json'

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            all_logs = json.load(f)
    else:
        all_logs = {}

    models = ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'MS-WM']
    labels = ['LSTM', 'GRU', 'Trans.', 'Mamba', 'S4D-WM', 'MS-WM']
    colors = ['#d62728', '#9467bd', '#2ca02c', '#ff7f0e', '#1f77b4', '#e91e63']

    for model_name in models:
        if model_name in all_logs:
            print(f'{model_name}: 已有结果，跳过', flush=True)
            continue
        print(f'{model_name}: 训练中...', end=' ', flush=True)
        train_losses, val_losses = train_with_logging(model_name, Xs, Xa, Y, Xv, Xav, Yv)
        all_logs[model_name] = {'train': train_losses, 'val': val_losses}
        print(f'完成 (最终val={val_losses[-1]:.4f})', flush=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_logs, f, indent=2)

    # 生成图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7.0))

    for i, (model_name, label) in enumerate(zip(models, labels)):
        if model_name in all_logs:
            train_losses = all_logs[model_name]['train']
            val_losses = all_logs[model_name]['val']
            epochs = list(range(1, len(train_losses)+1))
            ax1.plot(epochs, train_losses, '-', color=colors[i], linewidth=1.5, label=label, alpha=0.8)
            ax2.plot(epochs, val_losses, '-', color=colors[i], linewidth=1.5, label=label, alpha=0.8)

    ax1.set_xlabel('训练轮次', fontproperties=zhfont)
    ax1.set_ylabel('训练MSE', fontproperties=zhfont)
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('(a) 训练损失变化', fontproperties=zhfont, fontsize=10, pad=8)

    ax2.set_xlabel('训练轮次', fontproperties=zhfont)
    ax2.set_ylabel('验证MSE', fontproperties=zhfont)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('(b) 验证损失变化', fontproperties=zhfont, fontsize=10, pad=8)

    plt.tight_layout()
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/training_curves.pdf', dpi=300, bbox_inches='tight')
    print('Done: training_curves.pdf')
