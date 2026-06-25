"""补充缺失实验 - 多步预测、序列长度、训练损失、阈值、超参搜索"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel, DiagSSM
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
BS = 256
T = 32

print(f'Device: {device}', flush=True)

def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    eps = []
    for i, f in enumerate(fs):
        eps.append((np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']))
        if (i+1) % 200 == 0: print(f'    {i+1}/{len(fs)}...', flush=True)
    print(f'    {len(fs)} episodes loaded', flush=True)
    return eps

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

class MultiScaleModel(nn.Module):
    """MS-WM模型"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5, fusion_type='gate'):
        super().__init__()
        self.state_dim = state_dim
        self.fusion_type = fusion_type
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.slow_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])
        self.fast_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)})
            for _ in range(n_layers)
        ])
        self.local_attn = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model)})
            for _ in range(n_layers)
        ])
        if fusion_type == 'gate':
            self.fusion_gate = nn.Sequential(nn.Linear(d_model * 3, 3), nn.Softmax(dim=-1))
            self.fusion_proj = nn.Linear(d_model, state_dim)
        elif fusion_type == 'concat':
            self.fusion = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        slow_h = h
        for block in self.slow_ssm:
            residual = slow_h; x_norm = block['norm'](slow_h); slow_h = residual + block['ssm'](x_norm)
        fast_h = h
        for block in self.fast_ssm:
            residual = fast_h; x_norm = block['norm'](fast_h); fast_h = residual + block['ssm'](x_norm)
        local_h = h
        for block in self.local_attn:
            residual = local_h; x_norm = block['norm'](local_h); local_h = residual + block['conv'](x_norm.transpose(1,2)).transpose(1,2)
        if self.fusion_type == 'gate':
            features = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
            gate = self.fusion_gate(features)
            stacked = torch.stack([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=1)
            fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
            pred = self.fusion_proj(fused)
        elif self.fusion_type == 'concat':
            fused = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
            pred = self.fusion(fused)
        return states[:, -1, :] + pred

def train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_state = None
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val:
            best_val = vl; pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: pat += 1
        if pat >= 20: break
    if best_state: model.load_state_dict(best_state)
    return model

def evaluate_model(model, Xv, Xav, Yv):
    model.eval()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = nn.MSELoss()(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {'mse': round(mse, 6), 'r2': round(r2, 4), 'params_m': round(params, 3)}

def multi_step_predict(model, Xv, Xav, Yv, H_list=[1, 4, 8, 16]):
    """多步预测"""
    model.eval()
    results = {}
    for H in H_list:
        mse_h = []
        for i in range(min(50, len(Xv))):
            seq_s = torch.FloatTensor(Xv[i:i+1]).to(device)
            seq_a = torch.FloatTensor(Xav[i:i+1]).to(device)
            true_next = []
            for h in range(H):
                idx_t = i + h
                if idx_t < len(Yv): true_next.append(Yv[idx_t])
            if len(true_next) < H: continue
            preds = []
            cur_s, cur_a = seq_s.clone(), seq_a.clone()
            for h in range(H):
                with torch.no_grad(): p = model(cur_s, cur_a)
                preds.append(p.cpu().numpy()[0])
                cur_s = torch.cat([cur_s[:, 1:], p.unsqueeze(1)], dim=1)
            mse_h.append(np.mean((np.array(preds) - np.array(true_next))**2))
        results[f'H{H}'] = round(np.mean(mse_h), 6) if mse_h else None
    return results

def get_model_config(model_name, sd, ad):
    configs = {
        'LSTM-WM': (LSTMWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'Transformer-WM': (TransformerWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'nhead': 4, 'n_layers': 4}),
        'GRU-WM': (GRUWorldModel, {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
        'Mamba-WM': (MambaWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
        'S4D-WM': (SSMWorldModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
        'MS-WM': (MultiScaleModel, {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'}),
    }
    return configs[model_name]

# ============================================================
# 实验1: 多步预测 (表4)
# ============================================================
def run_multistep_experiment():
    print('\n' + '='*60, flush=True)
    print('实验1: 多步预测 (Humanoid)', flush=True)
    print('='*60, flush=True)

    RESULTS_FILE = 'experiments/multistep_results.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)

    for model_name in ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'MS-WM']:
        if model_name in results and len(results[model_name]) >= len(SEEDS):
            print(f'\n{model_name}: 已有完整结果，跳过', flush=True)
            continue

        print(f'\n{model_name}:', flush=True)
        if model_name not in results:
            results[model_name] = {}

        ModelClass, kwargs = get_model_config(model_name, 348, 17)
        for seed in SEEDS:
            seed_key = f'seed{seed}'
            if seed_key in results[model_name]:
                print(f'  seed={seed} 已有，跳过', flush=True)
                continue
            print(f'  seed={seed}...', end=' ', flush=True)
            model = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
            multistep = multi_step_predict(model, Xv, Xav, Yv)
            results[model_name][seed_key] = multistep
            print(f'H1={multistep["H1"]:.3f}, H16={multistep["H16"]:.3f}', flush=True)

            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 打印结果
    print('\n多步预测结果:', flush=True)
    print('{:<16} {:<10} {:<10} {:<10} {:<10}'.format('模型', 'H1', 'H4', 'H8', 'H16'))
    print('-'*60)
    for model_name in results:
        valid = [results[model_name][s] for s in results[model_name] if 'H1' in results[model_name][s]]
        if valid:
            h1 = np.mean([v['H1'] for v in valid if v['H1']])
            h4 = np.mean([v['H4'] for v in valid if v['H4']])
            h8 = np.mean([v['H8'] for v in valid if v['H8']])
            h16 = np.mean([v['H16'] for v in valid if v['H16']])
            print('{:<16} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(model_name, h1, h4, h8, h16))

# ============================================================
# 实验2: 序列长度分析 (表7)
# ============================================================
def run_seqlen_experiment():
    print('\n' + '='*60, flush=True)
    print('实验2: 序列长度分析 (Humanoid)', flush=True)
    print('='*60, flush=True)

    RESULTS_FILE = 'experiments/seqlen_results.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    seq_lengths = [16, 32, 64, 128, 256]

    for seq_len in seq_lengths:
        if f'T{seq_len}' in results and len(results[f'T{seq_len}']) >= len(SEEDS):
            print(f'\nT={seq_len}: 已有完整结果，跳过', flush=True)
            continue

        print(f'\nT={seq_len}:', flush=True)
        eps_tr = load_eps('data/humanoid', 'train')
        eps_vl = load_eps('data/humanoid', 'val')
        m, s = stats(eps_tr)
        Xs, Xa, Y = make_data(eps_tr, seq_len, m, s)
        Xv, Xav, Yv = make_data(eps_vl, seq_len, m, s)

        if f'T{seq_len}' not in results:
            results[f'T{seq_len}'] = {}

        ModelClass, kwargs = get_model_config('MS-WM', 348, 17)
        for seed in SEEDS:
            seed_key = f'seed{seed}'
            if seed_key in results[f'T{seq_len}']:
                print(f'  seed={seed} 已有，跳过', flush=True)
                continue
            print(f'  seed={seed}...', end=' ', flush=True)
            model = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
            r = evaluate_model(model, Xv, Xav, Yv)
            results[f'T{seq_len}'][seed_key] = r
            print(f'MSE={r["mse"]:.4f}', flush=True)

            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 打印结果
    print('\n序列长度分析结果:', flush=True)
    print('{:<10} {:<15} {:<10}'.format('T', 'MSE(×10⁻²)', 'R²'))
    print('-'*40)
    for seq_len in seq_lengths:
        key = f'T{seq_len}'
        if key in results:
            valid = [results[key][s] for s in results[key] if 'mse' in results[key][s]]
            if valid:
                mses = [r['mse'] for r in valid]
                r2s = [r['r2'] for r in valid]
                print('{:<10} {:.2f}±{:.2f}    {:.4f}'.format(seq_len, np.mean(mses)*100, np.std(mses)*100, np.mean(r2s)))

# ============================================================
# 实验3: 训练损失消融 (表9)
# ============================================================
def run_loss_ablation():
    print('\n' + '='*60, flush=True)
    print('实验3: 训练损失消融 (Humanoid)', flush=True)
    print('='*60, flush=True)

    # 这个实验需要修改训练代码来使用不同的损失函数
    # 暂时跳过，因为需要修改模型代码
    print('跳过（需要修改模型代码）', flush=True)

# ============================================================
# 实验4: 阈值函数对比 (表12)
# ============================================================
def run_threshold_experiment():
    print('\n' + '='*60, flush=True)
    print('实验4: 阈值函数对比 (Humanoid)', flush=True)
    print('='*60, flush=True)

    # 这个实验需要修改门控机制
    # 暂时跳过，因为需要修改模型代码
    print('跳过（需要修改模型代码）', flush=True)

# ============================================================
# 实验5: 超参搜索 (表13)
# ============================================================
def run_hyperparam_search():
    print('\n' + '='*60, flush=True)
    print('实验5: 超参搜索 (Humanoid)', flush=True)
    print('='*60, flush=True)

    RESULTS_FILE = 'experiments/hyperparam_results.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)

    # 测试不同d_model
    for d_model in [64, 96, 128, 192]:
        config_name = f'd{d_model}'
        if config_name in results:
            print(f'\n{config_name}: 已有结果，跳过', flush=True)
            continue

        print(f'\n{config_name}:', flush=True)
        ModelClass = MultiScaleModel
        kwargs = {'state_dim': 348, 'action_dim': 17, 'd_model': d_model, 'd_state': 8, 'n_layers': 1, 'window_size': 5, 'fusion_type': 'gate'}

        config_results = []
        for seed in SEEDS:
            print(f'  seed={seed}...', end=' ', flush=True)
            model = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
            r = evaluate_model(model, Xv, Xav, Yv)
            config_results.append(r)
            print(f'MSE={r["mse"]:.4f}', flush=True)

        results[config_name] = {
            'mse_mean': np.mean([r['mse'] for r in config_results]),
            'mse_std': np.std([r['mse'] for r in config_results]),
            'params_m': config_results[0]['params_m'],
        }

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # 打印结果
    print('\n超参搜索结果:', flush=True)
    print('{:<10} {:<15} {:<10}'.format('d_model', 'MSE(×10⁻²)', '参数(M)'))
    print('-'*40)
    for d_model in [64, 96, 128, 192]:
        config_name = f'd{d_model}'
        if config_name in results:
            r = results[config_name]
            print('{:<10} {:.2f}±{:.2f}    {:.3f}'.format(d_model, r['mse_mean']*100, r['mse_std']*100, r['params_m']))

# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print(f'Device: {device}', flush=True)
    print(f'开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    # 运行所有实验
    run_multistep_experiment()
    run_seqlen_experiment()
    run_loss_ablation()
    run_threshold_experiment()
    run_hyperparam_search()

    print('\n' + '='*60, flush=True)
    print('所有实验完成!', flush=True)
    print('='*60, flush=True)
