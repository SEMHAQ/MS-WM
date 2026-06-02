"""
工具函数: 可视化、日志等
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def plot_prediction_comparison(
    targets, predictions_dict, title="State Prediction Comparison",
    save_path=None, n_steps=100
):
    """
    绘制不同模型的预测对比图

    Args:
        targets: (T, state_dim) 真实状态
        predictions_dict: {model_name: (T, state_dim)} 预测结果
        title: 图标题
        save_path: 保存路径
        n_steps: 显示的步数
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=150)
    state_indices = [0, 1, 2, 3]  # 显示前4个状态维度
    dim_names = [f"Joint {i+1}" for i in state_indices]

    t = np.arange(min(n_steps, len(targets)))

    for ax_idx, (ax, dim_idx) in enumerate(zip(axes, state_indices)):
        ax.plot(t, targets[:n_steps, dim_idx], 'k-', label='Ground Truth', linewidth=2)

        colors = ['#2196F3', '#FF5722', '#4CAF50']
        for (name, pred), color in zip(predictions_dict.items(), colors):
            ax.plot(t, pred[:n_steps, dim_idx], '--', color=color, label=name, linewidth=1.5)

        ax.set_ylabel(dim_names[ax_idx])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")

    plt.close(fig)


def plot_training_curves(log_dir, save_path=None):
    """绘制训练曲线"""
    log_dir = Path(log_dir)
    # TODO: 从tensorboard日志中读取并绘制
    pass


def compute_metrics(pred, target):
    """计算评估指标"""
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))

    # 每个维度的MSE
    per_dim_mse = np.mean((pred - target) ** 2, axis=0)

    return {
        "mse": float(mse),
        "mae": float(mae),
        "per_dim_mse": per_dim_mse.tolist(),
    }


def save_experiment_results(results, save_path):
    """保存实验结果"""
    import json
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results saved: {save_path}")
