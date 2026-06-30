"""实验4：MPC规划 — 6个世界模型 × 2种MPC方法

世界模型：LSTM-WM, GRU-WM, Transformer-WM, Mamba-WM, S4D-WM, MIMO-WM
MPC方法：梯度MPC (Adam), CEM-MPC (GPU并行采样)
数据集：Humanoid
指标：控制频率(Hz)、每步耗时(ms)、跟踪误差(MSE)
种子：3个 [42, 123, 456]
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel, DiagSSM
from src.models.baselines import LSTMWorldModel, GRUWorldModel, TransformerWorldModel
from src.models.mamba_world_model import MambaWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 32
SEEDS = [42]

print(f'Device: {device}', flush=True)

def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs[:50]]

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

# ============================================================
# MIMO-WM 模型（与exp1一致）
# ============================================================
class MIMOLayer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.output(x) * torch.sigmoid(self.gate(x))
        return residual + x

class MIMOWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.backbone = nn.ModuleList([MIMOLayer(d_model, d_state) for _ in range(n_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            h = block(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])

# ============================================================
# 梯度MPC
# ============================================================
class GradientMPC:
    """梯度MPC：Adam优化动作序列"""
    def __init__(self, world_model, horizon=10, n_iterations=50, lr=0.01, Q=1.0, R=0.01):
        self.model = world_model
        self.H = horizon
        self.n_iter = n_iterations
        self.lr = lr
        self.Q = Q
        self.R = R

    def plan(self, state_hist, action_hist, target):
        action_dim = action_hist.shape[1]
        action_seq = nn.Parameter(torch.randn(self.H, action_dim, device=device) * 0.1)
        opt = torch.optim.Adam([action_seq], lr=self.lr)

        s_hist = torch.FloatTensor(state_hist).unsqueeze(0).to(device)
        a_hist = torch.FloatTensor(action_hist).unsqueeze(0).to(device)
        tgt = torch.FloatTensor(target).to(device)

        # cuDNN RNN backward 需要 training mode
        was_eval = not self.model.training
        if was_eval:
            self.model.train()

        for _ in range(self.n_iter):
            opt.zero_grad()
            cost = torch.tensor(0.0, device=device)
            s, a = s_hist.clone(), a_hist.clone()
            for h in range(self.H):
                pred = self.model(s, a)
                cost = cost + self.Q * torch.norm(pred - tgt, p=2) + self.R * torch.norm(action_seq[h], p=2)
                s = torch.cat([s[:, 1:], pred.unsqueeze(1)], dim=1)
                a = torch.cat([a[:, 1:], action_seq[h:h+1].unsqueeze(0)], dim=1)
            cost.backward()
            opt.step()

        if was_eval:
            self.model.eval()

        return action_seq[0].detach().cpu().numpy()

# ============================================================
# CEM-MPC（已修复 rollout bug）
# ============================================================
class CEMMPC:
    """CEM-MPC：交叉熵方法，GPU并行采样"""
    def __init__(self, world_model, horizon=10, n_samples=256, n_elite=32, n_iterations=5, Q=1.0, R=0.01):
        self.model = world_model
        self.H = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iter = n_iterations
        self.Q = Q
        self.R = R

    def plan(self, state_hist, action_hist, target):
        action_dim = action_hist.shape[1]
        s_hist = torch.FloatTensor(state_hist).unsqueeze(0).to(device)
        a_hist = torch.FloatTensor(action_hist).unsqueeze(0).to(device)
        tgt = torch.FloatTensor(target).to(device)

        mu = torch.zeros(self.H, action_dim, device=device)
        sigma = torch.ones(self.H, action_dim, device=device)

        for _ in range(self.n_iter):
            samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(self.n_samples, self.H, action_dim, device=device)
            samples = samples.clamp(-1, 1)

            costs = torch.zeros(self.n_samples, device=device)
            s_cur = s_hist.expand(self.n_samples, -1, -1).clone()
            a_cur = a_hist.expand(self.n_samples, -1, -1).clone()
            for h in range(self.H):
                pred = self.model(s_cur, a_cur)
                costs = costs + self.Q * torch.norm(pred - tgt.unsqueeze(0), p=2, dim=-1) + self.R * torch.norm(samples[:, h], p=2, dim=-1)
                s_cur = torch.cat([s_cur[:, 1:], pred.unsqueeze(1)], dim=1)
                a_cur = torch.cat([a_cur[:, 1:], samples[:, h:h+1]], dim=1)

            elite_idx = costs.topk(self.n_elite, largest=False).indices
            elite = samples[elite_idx]
            mu = elite.mean(dim=0)
            sigma = elite.std(dim=0) + 1e-6

        return mu[0].detach().cpu().numpy()

# ============================================================
# 训练世界模型
# ============================================================
def train_world_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
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
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0
        else: pat += 1
        if pat >= 20: break

    return model

# ============================================================
# MPC评估（速度 + 闭环仿真质量）
# ============================================================
def evaluate_mpc_speed(mpc, episodes, mean, std, n_episodes=10, n_steps=50):
    """仅测速度"""
    total_time = 0; count = 0
    for states, actions in episodes[:n_episodes]:
        if len(states) < T + n_steps: continue
        s_norm = (states - mean) / (std + 1e-8)
        target = s_norm[T]
        t0 = time.perf_counter()
        for step in range(n_steps):
            s_hist = s_norm[step:step+T]
            a_hist = actions[step:step+T-1]
            mpc.plan(s_hist, a_hist, target)
            if T + step + 1 < len(s_norm):
                target = s_norm[T + step + 1]
        elapsed = time.perf_counter() - t0
        total_time += elapsed; count += 1
    if count == 0: return {'hz': 0, 'avg_step_time_ms': 0, 'episodes': 0}
    avg_time = total_time / count / n_steps
    return {
        'hz': round(1.0 / avg_time, 2) if avg_time > 0 else 0,
        'avg_step_time_ms': round(avg_time * 1000, 2),
        'episodes': count
    }

def evaluate_mpc_quality(model, episodes, mean, std, n_episodes=10, n_steps=20):
    """闭环仿真质量：用世界模型做仿真器，测MPC规划的跟踪效果"""
    model.eval()
    errors = []
    for states, actions in episodes[:n_episodes]:
        if len(states) < T + n_steps: continue
        s_norm = (states - mean) / (std + 1e-8)
        # 初始历史
        s_hist = s_norm[:T].copy()
        a_hist = actions[:T-1].copy()
        target = s_norm[T]
        ep_errors = []
        for step in range(n_steps):
            # 用 CEM-MPC 规划
            cem = CEMMPC(model, horizon=10, n_samples=64, n_elite=8, n_iterations=3)
            planned_action = cem.plan(s_hist, a_hist, target)
            # 用世界模型仿真一步
            with torch.no_grad():
                s_t = torch.FloatTensor(s_hist).unsqueeze(0).to(device)
                a_t = torch.FloatTensor(a_hist).unsqueeze(0).to(device)
                pred = model(s_t, a_t).cpu().numpy().flatten()
            # 跟踪误差：规划动作产生的预测 vs 目标
            ep_errors.append(np.mean((pred - target) ** 2))
            # 滚动更新历史
            s_hist = np.vstack([s_hist[1:], pred.reshape(1, -1)])
            a_hist = np.vstack([a_hist[1:], planned_action.reshape(1, -1)])
            if T + step + 1 < len(s_norm):
                target = s_norm[T + step + 1]
        if ep_errors:
            errors.append(np.mean(ep_errors))
    if not errors: return {'tracking_mse': 0}
    return {'tracking_mse': round(float(np.mean(errors)), 6)}

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    datasets = {
        'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
        'humanoid_standup': {'dir': 'data/humanoid_standup', 'sd': 348, 'ad': 17},
    }

    models = {
        'LSTM-WM':        (LSTMWorldModel,       {'state_dim': 348, 'action_dim': 17, 'hidden_dim': 96, 'n_layers': 2}),
        'GRU-WM':         (GRUWorldModel,         {'state_dim': 348, 'action_dim': 17, 'hidden_dim': 96, 'n_layers': 2}),
        'Transformer-WM': (TransformerWorldModel,  {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'nhead': 4, 'n_layers': 2}),
        'Mamba-WM':       (MambaWorldModel,        {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'n_layers': 2}),
        'S4D-WM':         (SSMWorldModel,          {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'd_state': 16, 'n_layers': 2}),
        'MIMO-WM':        (MIMOWorldModel,         {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'd_state': 16, 'n_layers': 2}),
    }

    RESULTS_FILE = 'experiments/exp4_mpc.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    for ds_name, ds_cfg in datasets.items():
        print(f'\n{"="*60}', flush=True)
        print(f'数据集: {ds_name}', flush=True)
        print(f'{"="*60}', flush=True)

        eps_tr = load_eps(ds_cfg['dir'], 'train')
        eps_vl = load_eps(ds_cfg['dir'], 'val')
        mean, std = stats(eps_tr)
        Xs, Xa, Y = make_data(eps_tr, T, mean, std)
        Xv, Xav, Yv = make_data(eps_vl, T, mean, std)
        print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

        for model_name, (ModelClass, kwargs) in models.items():
            g_key = f'{model_name}_GradMPC_{ds_name}'
            c_key = f'{model_name}_CEMMPC_{ds_name}'
            q_key = f'{model_name}_Quality_{ds_name}'
            g_done = g_key in results and len(results[g_key]) >= len(SEEDS)
            c_done = c_key in results and len(results[c_key]) >= len(SEEDS)
            q_done = q_key in results and len(results[q_key]) >= len(SEEDS)
            if g_done and c_done and q_done:
                print(f'\n  [{model_name}]: 已有完整结果，跳过', flush=True)
                continue

            print(f'\n  [{model_name}]:', flush=True)
            if g_key not in results: results[g_key] = {}
            if c_key not in results: results[c_key] = {}
            if q_key not in results: results[q_key] = {}

            for seed in SEEDS:
                sg_key = f'seed{seed}'
                if sg_key in results[g_key] and sg_key in results[c_key] and sg_key in results[q_key]:
                    print(f'    seed={seed} 已有，跳过', flush=True)
                    continue

                print(f'    seed={seed} 训练模型...', flush=True)
                model = train_world_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed=seed)
                model.eval()

                if sg_key not in results[g_key]:
                    print(f'    seed={seed} 评估梯度MPC速度...', flush=True)
                    grad_mpc = GradientMPC(model, horizon=10, n_iterations=30, lr=0.01)
                    r = evaluate_mpc_speed(grad_mpc, eps_vl, mean, std)
                    results[g_key][sg_key] = r
                    print(f'      {r["hz"]:.2f} Hz, {r["avg_step_time_ms"]:.1f} ms', flush=True)

                if sg_key not in results[c_key]:
                    print(f'    seed={seed} 评估CEM-MPC速度...', flush=True)
                    cem_mpc = CEMMPC(model, horizon=10, n_samples=256, n_elite=32, n_iterations=5)
                    r = evaluate_mpc_speed(cem_mpc, eps_vl, mean, std)
                    results[c_key][sg_key] = r
                    print(f'      {r["hz"]:.2f} Hz, {r["avg_step_time_ms"]:.1f} ms', flush=True)

                if sg_key not in results[q_key]:
                    print(f'    seed={seed} 评估闭环规划质量...', flush=True)
                    r = evaluate_mpc_quality(model, eps_vl, mean, std)
                    results[q_key][sg_key] = r
                    print(f'      tracking MSE={r["tracking_mse"]:.6f}', flush=True)

                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

    # 汇总
    for ds_name in datasets:
        print('\n' + '='*90, flush=True)
        print(f'实验4：MPC规划结果 — {ds_name}', flush=True)
        print('='*90)
        print('{:<18} {:<20} {:<20} {:<15}'.format('模型', '梯度MPC', 'CEM-MPC', 'CEM加速比'))
        print('-'*75)
        for model_name in models:
            g_key = f'{model_name}_GradMPC_{ds_name}'
            c_key = f'{model_name}_CEMMPC_{ds_name}'
            if g_key in results and c_key in results:
                g_hz = [r['hz'] for r in results[g_key].values()]
                c_hz = [r['hz'] for r in results[c_key].values()]
                g_mean, g_std = np.mean(g_hz), np.std(g_hz)
                c_mean, c_std = np.mean(c_hz), np.std(c_hz)
                speedup = c_mean / g_mean if g_mean > 0 else 0
                print('{:<18} {:.2f}±{:<14.2f} {:.2f}±{:<14.2f} {:.1f}x'.format(
                    model_name, g_mean, g_std, c_mean, c_std, speedup))

    print('\nDone!', flush=True)
