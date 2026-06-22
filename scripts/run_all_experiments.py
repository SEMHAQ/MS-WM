"""
统一实验脚本 - FSM-WM论文所有实验
=====================================
所有模型使用相同配置:
- 序列长度 T=32
- 批大小 B=64
- 训练轮数 100
- 学习率 5e-4 (AdamW + cosine annealing)
- 5个随机种子: 42, 123, 456, 789, 1024

实验内容:
1. 主实验: 6个模型 × 3个数据集 × 5个seed
2. 序列长度分析: FSM-WM × 5个长度 × 2个数据集 × 5个seed
3. 消融实验: FSM-WM变体 × 5个seed

运行方式:
    # 运行所有实验
    python scripts/run_all_experiments.py

    # 只运行主实验
    python scripts/run_all_experiments.py --experiment main

    # 只运行序列长度实验
    python scripts/run_all_experiments.py --experiment seqlen

    # 只运行消融实验
    python scripts/run_all_experiments.py --experiment ablation

    # 快速测试（1个seed）
    python scripts/run_all_experiments.py --quick
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# 添加项目根目录到path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.ssm_world_model import SSMWorldModel
from src.models.baselines import LSTMWorldModel, GRUWorldModel, TransformerWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.fusion_ssm import FSM

# ============================================================
# 配置
# ============================================================
SEEDS = [42, 123, 456, 789, 1024]
DATASETS = {
    'humanoid': {'state_dim': 348, 'action_dim': 17},
    'ant': {'state_dim': 105, 'action_dim': 8},
    'hopper': {'state_dim': 11, 'action_dim': 6},
}
SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 1e-4
D_MODEL = 128
D_STATE = 16
N_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 结果保存目录
RESULTS_DIR = ROOT / 'experiments' / 'paper'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 数据加载
# ============================================================
def load_dataset(dataset_name):
    """加载D4RL数据集"""
    data_dir = ROOT / 'data' / dataset_name
    train_episodes = []
    val_episodes = []

    for split, episode_list in [('train', train_episodes), ('val', val_episodes)]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for f in sorted(split_dir.glob('*.npz')):
            data = np.load(f)
            states = data['states']
            actions = data['actions']
            episode_list.append((states, actions))

    return train_episodes, val_episodes


def normalize_data(train_episodes, val_episodes):
    """Z-score归一化"""
    all_states = np.concatenate([s for s, _ in train_episodes])
    all_actions = np.concatenate([a for _, a in train_episodes])

    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0) + 1e-8
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0) + 1e-8

    def normalize(episodes):
        result = []
        for states, actions in episodes:
            norm_states = (states - state_mean) / state_std
            norm_actions = (actions - action_mean) / action_std
            result.append((norm_states, norm_actions))
        return result

    return normalize(train_episodes), normalize(val_episodes), {
        'state_mean': state_mean, 'state_std': state_std,
        'action_mean': action_mean, 'action_std': action_std
    }


def create_sequences(episodes, seq_len):
    """创建训练序列"""
    sequences = []
    for states, actions in episodes:
        if len(states) < seq_len + 1:
            continue
        for i in range(len(states) - seq_len):
            s_seq = states[i:i+seq_len]
            a_seq = actions[i:i+seq_len]
            s_target = states[i+seq_len]
            sequences.append((s_seq, a_seq, s_target))
    return sequences


# ============================================================
# 模型构建
# ============================================================
def build_model(model_name, state_dim, action_dim):
    """构建模型"""
    if model_name == 'S4D-WM':
        return SSMWorldModel(state_dim, action_dim, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS)
    elif model_name == 'LSTM-WM':
        return LSTMWorldModel(state_dim, action_dim, d_model=D_MODEL, n_layers=N_LAYERS)
    elif model_name == 'GRU-WM':
        return GRUWorldModel(state_dim, action_dim, d_model=D_MODEL, n_layers=N_LAYERS)
    elif model_name == 'Transformer-WM':
        return TransformerWorldModel(state_dim, action_dim, d_model=D_MODEL, n_layers=N_LAYERS)
    elif model_name == 'Mamba-WM':
        return MambaWorldModel(state_dim, action_dim, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS)
    elif model_name == 'FSM-WM':
        return FSM(state_dim, action_dim, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# 训练函数
# ============================================================
def train_model(model, train_seqs, val_seqs, seed, model_name, dataset_name):
    """训练模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_mse = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        np.random.shuffle(train_seqs)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(train_seqs), BATCH_SIZE):
            batch = train_seqs[i:i+BATCH_SIZE]
            if len(batch) < 2:
                continue

            s_batch = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
            a_batch = torch.FloatTensor(np.array([b[1] for b in batch])).to(DEVICE)
            target = torch.FloatTensor(np.array([b[2] for b in batch])).to(DEVICE)

            pred = model(s_batch, a_batch)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for i in range(0, len(val_seqs), BATCH_SIZE):
                batch = val_seqs[i:i+BATCH_SIZE]
                if len(batch) < 2:
                    continue

                s_batch = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
                a_batch = torch.FloatTensor(np.array([b[1] for b in batch])).to(DEVICE)
                target = torch.FloatTensor(np.array([b[2] for b in batch])).to(DEVICE)

                pred = model(s_batch, a_batch)
                val_loss += criterion(pred, target).item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_mse:
            best_val_mse = avg_val_loss
            best_epoch = epoch + 1

    # 测量推理时间
    model.eval()
    state_dim = DATASETS[dataset_name]['state_dim']
    action_dim = DATASETS[dataset_name]['action_dim']
    with torch.no_grad():
        dummy_s = torch.randn(1, SEQ_LEN, state_dim).to(DEVICE)
        dummy_a = torch.randn(1, SEQ_LEN, action_dim).to(DEVICE)

        # Warmup
        for _ in range(10):
            model(dummy_s, dummy_a)

        # 测量
        times = []
        for _ in range(100):
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy_s, dummy_a)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    infer_time = np.mean(times)
    infer_std = np.std(times)

    # 计算参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        'mse': best_val_mse,
        'best_epoch': best_epoch,
        'infer_ms': infer_time,
        'infer_std_ms': infer_std,
        'params_m': params,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


# ============================================================
# 实验1: 主实验 (6模型 × 3数据集 × 5seed)
# ============================================================
def run_main_experiments():
    """运行主实验"""
    print("=" * 60)
    print("实验1: 主实验 (6模型 × 3数据集 × 5seed)")
    print("=" * 60)

    models = ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'FSM-WM']
    results = {}

    for dataset_name, dims in DATASETS.items():
        print(f"\n数据集: {dataset_name}")
        train_eps, val_eps = load_dataset(dataset_name)
        train_norm, val_norm, stats = normalize_data(train_eps, val_eps)
        train_seqs = create_sequences(train_norm, SEQ_LEN)
        val_seqs = create_sequences(val_norm, SEQ_LEN)

        for model_name in models:
            print(f"  {model_name}:", end=" ", flush=True)
            model_results = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = build_model(model_name, dims['state_dim'], dims['action_dim'])
                result = train_model(model, train_seqs, val_seqs, seed, model_name, dataset_name)
                model_results.append(result)
                print(f"seed{seed} MSE={result['mse']:.4f}", end=" ", flush=True)

                # 保存checkpoint
                ckpt_dir = RESULTS_DIR / 'checkpoints'
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"{model_name}_{dataset_name}_seed{seed}.pth")

            # 计算统计
            mses = [r['mse'] for r in model_results]
            results[f"{model_name}_{dataset_name}"] = {
                'mse_mean': np.mean(mses),
                'mse_std': np.std(mses),
                'seeds': {f"seed{SEEDS[i]}": model_results[i] for i in range(len(SEEDS))},
            }
            print(f"-> {np.mean(mses)*100:.2f}±{np.std(mses)*100:.2f}")

    # 保存结果
    with open(RESULTS_DIR / 'main_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ============================================================
# 实验2: 序列长度分析
# ============================================================
def run_seqlen_experiments():
    """运行序列长度实验"""
    print("\n" + "=" * 60)
    print("实验2: 序列长度分析 (FSM-WM × 5长度 × 2数据集 × 5seed)")
    print("=" * 60)

    seq_lengths = [16, 32, 64, 128, 256]
    datasets = ['humanoid', 'ant']
    results = {}

    for dataset_name in datasets:
        dims = DATASETS[dataset_name]
        print(f"\n数据集: {dataset_name}")

        for seq_len in seq_lengths:
            print(f"  T={seq_len}:", end=" ", flush=True)
            train_eps, val_eps = load_dataset(dataset_name)
            train_norm, val_norm, stats = normalize_data(train_eps, val_eps)
            train_seqs = create_sequences(train_norm, seq_len)
            val_seqs = create_sequences(val_norm, seq_len)

            model_results = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = build_model('FSM-WM', dims['state_dim'], dims['action_dim'])
                result = train_model(model, train_seqs, val_seqs, seed, 'FSM-WM', dataset_name)
                model_results.append(result)
                print(f"seed{seed} MSE={result['mse']:.4f}", end=" ", flush=True)

            mses = [r['mse'] for r in model_results]
            results[f"FSM-WM_{dataset_name}_T{seq_len}"] = {
                'mse_mean': np.mean(mses),
                'mse_std': np.std(mses),
            }
            print(f"-> {np.mean(mses)*100:.2f}±{np.std(mses)*100:.2f}")

    with open(RESULTS_DIR / 'seqlen_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ============================================================
# 实验3: 消融实验
# ============================================================
def run_ablation_experiments():
    """运行消融实验"""
    print("\n" + "=" * 60)
    print("实验3: 消融实验 (FSM-WM变体 × 5seed)")
    print("=" * 60)

    dataset_name = 'humanoid'
    dims = DATASETS[dataset_name]
    train_eps, val_eps = load_dataset(dataset_name)
    train_norm, val_norm, stats = normalize_data(train_eps, val_eps)
    train_seqs = create_sequences(train_norm, SEQ_LEN)
    val_seqs = create_sequences(val_norm, SEQ_LEN)

    ablation_configs = {
        'L=2': {'n_layers': 2},
        'L=6': {'n_layers': 6},
        'D=64': {'d_model': 64},
        'D=256': {'d_model': 256},
        'N=32': {'d_state': 32},
    }

    results = {}

    # 默认配置
    print("  默认配置:", end=" ", flush=True)
    default_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model('FSM-WM', dims['state_dim'], dims['action_dim'])
        result = train_model(model, train_seqs, val_seqs, seed, 'FSM-WM', dataset_name)
        default_results.append(result)
    mses = [r['mse'] for r in default_results]
    results['default'] = {'mse_mean': np.mean(mses), 'mse_std': np.std(mses)}
    print(f"{np.mean(mses)*100:.2f}±{np.std(mses)*100:.2f}")

    # 变体
    for config_name, overrides in ablation_configs.items():
        print(f"  {config_name}:", end=" ", flush=True)
        config_results = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 使用自定义配置
            d_model = overrides.get('d_model', D_MODEL)
            d_state = overrides.get('d_state', D_STATE)
            n_layers = overrides.get('n_layers', N_LAYERS)

            model = FSM(dims['state_dim'], dims['action_dim'],
                       d_model=d_model, d_state=d_state, n_layers=n_layers)
            result = train_model(model, train_seqs, val_seqs, seed, 'FSM-WM', dataset_name)
            config_results.append(result)

        mses = [r['mse'] for r in config_results]
        results[config_name] = {
            'mse_mean': np.mean(mses),
            'mse_std': np.std(mses),
            'params_m': config_results[0]['params_m'],
            'infer_ms': config_results[0]['infer_ms'],
        }
        print(f"{np.mean(mses)*100:.2f}±{np.std(mses)*100:.2f}")

    with open(RESULTS_DIR / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='FSM-WM统一实验脚本')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'main', 'seqlen', 'ablation'],
                       help='要运行的实验')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式（只用1个seed）')
    args = parser.parse_args()

    # 快速模式：只用1个seed
    if args.quick:
        global SEEDS
        SEEDS = [42]
        print("⚡ 快速测试模式：只用1个seed")

    print(f"设备: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    if args.experiment in ['all', 'main']:
        run_main_experiments()

    if args.experiment in ['all', 'seqlen']:
        run_seqlen_experiments()

    if args.experiment in ['all', 'ablation']:
        run_ablation_experiments()

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/3600:.1f}小时")
    print(f"结果保存在: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
