"""系统性架构探索 - 找到最优的世界模型架构
从第一性原理出发，测试多种创新方案
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time, itertools
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 50
BS = 256
T = 32

print(f'Device: {device}', flush=True)

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

def train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed=SEED, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 15: break

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

    # 测量推理时间
    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        inf_time = (time.perf_counter() - t0) / 100 * 1000

    return {
        'mse': round(mse, 6),
        'r2': round(r2, 4),
        'params': round(params, 3),
        'inf_time': round(inf_time, 2),
        'train_time': round(elapsed, 1),
        'best_epoch': best_ep
    }

# ============================================================
# 定义多种架构变体
# ============================================================

class BaselineSSM(nn.Module):
    """基线：纯SSM（S4D风格）"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            self.backbone.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
                'gate': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)),
            }))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            residual = x
            x_norm = block['norm'](x)
            ssm_out = block['ssm'](x_norm)
            g = torch.sigmoid(block['gate'](x_norm))
            x = residual + g * ssm_out + (1-g) * x_norm
            x = x + block['ffn'](block['norm'](x))
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

class BaselineTransformer(nn.Module):
    """基线：纯Transformer"""
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, n_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

class GatedFusion(nn.Module):
    """创新1：门控融合 - SSM和Attention的自适应融合"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=4, window_size=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            self.backbone.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
                'attn': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model),
                'gate': nn.Sequential(nn.Linear(d_model, d_model//4), nn.GELU(), nn.Linear(d_model//4, 2)),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)),
            }))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            residual = x
            x_norm = block['norm'](x)
            ssm_out = block['ssm'](x_norm)
            attn_out = block['attn'](x_norm.transpose(1,2)).transpose(1,2)
            gate = torch.softmax(block['gate'](x_norm), dim=-1)
            fused = gate[..., 0:1] * ssm_out + gate[..., 1:2] * attn_out
            x = residual + fused
            x = x + block['ffn'](block['norm'](x))
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

class MixtureOfExperts(nn.Module):
    """创新2：混合专家 - 多个SSM专家，门控选择"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=4, n_experts=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            experts = nn.ModuleList([DiagSSM(d_model, d_state) for _ in range(n_experts)])
            self.backbone.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'experts': experts,
                'gate': nn.Sequential(nn.Linear(d_model, n_experts)),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)),
            }))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            residual = x
            x_norm = block['norm'](x)
            # 多专家
            expert_outputs = [expert(x_norm) for expert in block['experts']]
            expert_outputs = torch.stack(expert_outputs, dim=-1)  # (B, L, D, n_experts)
            # 门控选择
            gate = torch.softmax(block['gate'](x_norm), dim=-1)  # (B, L, n_experts)
            fused = (expert_outputs * gate.unsqueeze(-2)).sum(dim=-1)  # (B, L, D)
            x = residual + fused
            x = x + block['ffn'](block['norm'](x))
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

class AdaptiveComplexity(nn.Module):
    """创新3：自适应复杂度 - 根据输入自动选择SSM或Attention"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=4, window_size=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            self.backbone.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
                'attn': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model),
                'complexity': nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid()),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)),
            }))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            residual = x
            x_norm = block['norm'](x)
            ssm_out = block['ssm'](x_norm)
            attn_out = block['attn'](x_norm.transpose(1,2)).transpose(1,2)
            # 复杂度自适应
            complexity = block['complexity'](x_norm)  # (B, L, 1)
            # 高复杂度用SSM，低复杂度用Attention
            fused = complexity * ssm_out + (1-complexity) * attn_out
            x = residual + fused
            x = x + block['ffn'](block['norm'](x))
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

class DualBranch(nn.Module):
    """创新4：双分支独立处理，最后融合"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2, window_size=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # SSM分支
        self.ssm_branch = nn.ModuleList()
        for _ in range(n_layers):
            self.ssm_branch.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)),
            }))
        # Attention分支
        self.attn_branch = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_branch.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'attn': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)),
            }))
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        # SSM分支
        ssm_x = x
        for block in self.ssm_branch:
            residual = ssm_x
            x_norm = block['norm'](ssm_x)
            ssm_out = block['ssm'](x_norm)
            ssm_x = residual + ssm_out
            ssm_x = ssm_x + block['ffn'](block['norm'](ssm_x))
        # Attention分支
        attn_x = x
        for block in self.attn_branch:
            residual = attn_x
            x_norm = block['norm'](attn_x)
            attn_out = block['attn'](x_norm.transpose(1,2)).transpose(1,2)
            attn_x = residual + attn_out
            attn_x = attn_x + block['ffn'](block['norm'](attn_x))
        # 融合
        fused = self.fusion(torch.cat([ssm_x[:, -1, :], attn_x[:, -1, :]], dim=-1))
        x = self.decoder(fused)
        return states[:, -1, :] + x

class LightweightSSM(nn.Module):
    """创新5：轻量级SSM - 去掉FFN，极简设计"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            self.backbone.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': DiagSSM(d_model, d_state),
                'gate': nn.Linear(d_model, d_model),
            }))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            residual = x
            x_norm = block['norm'](x)
            ssm_out = block['ssm'](x_norm)
            g = torch.sigmoid(block['gate'](x_norm))
            x = residual + g * ssm_out + (1-g) * x_norm
        x = self.decoder(x[:, -1, :])
        return states[:, -1, :] + x

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    # 加载数据
    print('\n加载Humanoid数据...', flush=True)
    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    sd, ad = 348, 17
    os.makedirs('experiments', exist_ok=True)

    # 定义所有架构
    architectures = {
        'Baseline-SSM': lambda: BaselineSSM(sd, ad, d_model=128, d_state=16, n_layers=4),
        'Baseline-Trans': lambda: BaselineTransformer(sd, ad, d_model=128, nhead=4, n_layers=4),
        'GatedFusion-L4': lambda: GatedFusion(sd, ad, d_model=128, d_state=16, n_layers=4),
        'GatedFusion-L2': lambda: GatedFusion(sd, ad, d_model=128, d_state=16, n_layers=2),
        'MoE-4experts': lambda: MixtureOfExperts(sd, ad, d_model=128, d_state=16, n_layers=4, n_experts=4),
        'MoE-2experts': lambda: MixtureOfExperts(sd, ad, d_model=128, d_state=16, n_layers=4, n_experts=2),
        'AdaptiveComplexity': lambda: AdaptiveComplexity(sd, ad, d_model=128, d_state=16, n_layers=4),
        'DualBranch-L2': lambda: DualBranch(sd, ad, d_model=128, d_state=16, n_layers=2),
        'LightweightSSM-L4': lambda: LightweightSSM(sd, ad, d_model=128, d_state=16, n_layers=4),
        'LightweightSSM-L2': lambda: LightweightSSM(sd, ad, d_model=128, d_state=16, n_layers=2),
    }

    # 加载已有结果
    RESULTS_FILE = 'experiments/architecture_search.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # 测试所有架构
    print('\n' + '='*80, flush=True)
    print('架构搜索实验', flush=True)
    print('='*80, flush=True)

    for name, model_fn in architectures.items():
        if name in results:
            print(f'\n{name}: 已有结果，跳过', flush=True)
            continue

        print(f'\n{name}:', flush=True)
        try:
            model = model_fn()
            r = train_eval(model, Xs, Xa, Y, Xv, Xav, Yv)
            results[name] = r
            print(f'  MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}, Params={r["params"]:.3f}M, InfTime={r["inf_time"]:.2f}ms', flush=True)

            # 保存中间结果
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            results[name] = {'error': str(e)}

    # 打印结果汇总
    print('\n' + '='*80, flush=True)
    print('架构搜索结果汇总', flush=True)
    print('='*80, flush=True)
    print('{:<25} {:<10} {:<10} {:<10} {:<10}'.format('架构', 'MSE', 'R²', '参数(M)', '推理(ms)'))
    print('-'*70)

    # 按MSE排序
    valid_results = {k: v for k, v in results.items() if 'mse' in v}
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['mse'])

    for name, r in sorted_results:
        print('{:<25} {:<10.4f} {:<10.4f} {:<10.3f} {:<10.2f}'.format(
            name, r['mse'], r['r2'], r['params'], r['inf_time']
        ))

    print('\nDone!', flush=True)
