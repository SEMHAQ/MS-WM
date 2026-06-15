"""Nature-style figures for S4D-WM paper. All 4 figures in one script."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import numpy as np
import os, json

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

# Nature-style global settings
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Nature color palette
C_SSM   = '#1B5E9B'   # Deep blue (primary)
C_LSTM  = '#C44E52'   # Muted red
C_TRANS = '#55A868'   # Sage green
C_MAMBA = '#8C564B'   # Brown
C_ANNO  = '#E67E22'   # Orange accent

OUTDIR = 'paper/figures'

def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f'{name}.pdf'))
    fig.savefig(os.path.join(OUTDIR, f'{name}.png'), dpi=300)
    plt.close(fig)
    print(f'  Saved {name}')

# ============================================================
# Figure 1: Batch Inference (Lollipop chart)
# ============================================================
def fig_batch_inference():
    batch_sizes = ['1', '8', '32', '64']
    lstm = [2.1, 4.5, 12.3, 27.8]
    mamba = [1.2, 1.8, 2.8, 4.5]
    s4d = [0.9, 1.5, 2.4, 3.8]
    
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(batch_sizes))
    w = 0.22
    
    bars1 = ax.bar(x - w, lstm, w, color=C_LSTM, alpha=0.85, label='LSTM-WM', edgecolor='white')
    bars2 = ax.bar(x, mamba, w, color=C_MAMBA, alpha=0.85, label='Mamba-WM', edgecolor='white')
    bars3 = ax.bar(x + w, s4d, w, color=C_SSM, alpha=0.85, label='S4D-WM', edgecolor='white')
    
    # Value labels on top of S4D bars
    for bar, v in zip(bars3, s4d):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f'{v}',
                ha='center', fontsize=7.5, color=C_SSM, fontweight='bold')
    
    # 10ms threshold (only meaningful at B=64 where LSTM exceeds it)
    ax.axhline(y=10, color='#999', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.text(3.5, 10.8, '10ms阈值', fontproperties=zhfont_s, fontsize=7, color='#999', ha='right')
    
    ax.set_xlabel('批大小 B', fontproperties=zhfont)
    ax.set_ylabel('推理时间 (ms)', fontproperties=zhfont)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, 32)
    
    # Speedup annotation
    ax.annotate('7.3×加速', xy=(3 + w, 3.8), xytext=(2.5, 18),
                fontproperties=zhfont_s, fontsize=8, ha='center', color=C_SSM,
                arrowprops=dict(arrowstyle='->', color=C_SSM, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F0FE', edgecolor=C_SSM, alpha=0.8))
    
    save(fig, 'batch_inference')

# ============================================================
# Figure 3: Ablation (Top-bottom, nature style)
# ============================================================
def fig_ablation():
    configs = ['默认', '无门控', '无残差', 'L=2', 'L=6', 'D=64', 'D=256']
    mse = [0.245, 0.252, 0.249, 0.268, 0.238, 0.263, 0.235]
    params = [0.23, 0.22, 0.24, 0.12, 0.36, 0.08, 0.85]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5))
    
    # (a) MSE
    colors_mse = [C_SSM if i == 0 else '#A0C4E8' for i in range(len(configs))]
    bars = ax1.bar(range(len(configs)), mse, color=colors_mse, alpha=0.9, edgecolor='white', width=0.6)
    bars[0].set_edgecolor(C_SSM)
    bars[0].set_linewidth(1.5)
    
    # Delta annotations
    base = mse[0]
    for i, (bar, v) in enumerate(zip(bars, mse)):
        if i == 0: continue
        delta = (v - base) / base * 100
        color = '#C44E52' if delta > 0 else '#55A868'
        sign = '+' if delta > 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                f'{sign}{delta:.1f}%', fontsize=6.5, ha='center', color=color, fontweight='bold')
    
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, fontproperties=zhfont_s, fontsize=8)
    ax1.set_ylabel('MSE', fontsize=10)
    ax1.set_title('(a) 预测MSE', fontproperties=zhfont, fontsize=10, pad=8)
    
    # (b) Params
    colors_p = [C_SSM if i == 0 else '#A0C4E8' for i in range(len(configs))]
    bars2 = ax2.bar(range(len(configs)), params, color=colors_p, alpha=0.9, edgecolor='white', width=0.6)
    bars2[0].set_edgecolor(C_SSM)
    bars2[0].set_linewidth(1.5)
    
    for bar, v in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f'{v:.2f}M', fontsize=7, ha='center', color='#333')
    
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, fontproperties=zhfont_s, fontsize=8)
    ax2.set_ylabel('参数量 (M)', fontproperties=zhfont)
    ax2.set_title('(b) 参数量', fontproperties=zhfont, fontsize=10, pad=8)
    
    plt.tight_layout()
    save(fig, 'ablation_results')

# ============================================================
# Figure 4: Training Curves (Top-bottom, nature style)
# ============================================================
def fig_training_curves():
    with open('experiments/d4rl_all_experiments.json') as f:
        data = json.load(f)
    logs = data['training_logs']
    
    models = ['S4D-WM_d4rl', 'Mamba-WM_d4rl', 'Trans-WM_d4rl', 'LSTM-WM_d4rl']
    labels = ['S4D-WM', 'Mamba-WM', 'Trans.-WM', 'LSTM-WM']
    colors = [C_SSM, C_MAMBA, C_TRANS, C_LSTM]
    lws = [2.2, 1.5, 1.5, 1.5]  # Primary method thicker
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7.0))
    
    for model, label, color, lw in zip(models, labels, colors, lws):
        entries = logs[model]
        epochs = [e['epoch'] for e in entries]
        train_loss = [e['train'] for e in entries]
        val_loss = [e['val'] for e in entries]
        
        ax1.plot(epochs, train_loss, '-', color=color, linewidth=lw, label=label, alpha=0.85)
        ax2.plot(epochs, val_loss, '-', color=color, linewidth=lw, label=label, alpha=0.85)
    
    ax1.set_xlabel('训练轮次', fontproperties=zhfont)
    ax1.set_ylabel('训练MSE', fontproperties=zhfont)
    ax1.legend(fontsize=8)
    ax1.set_title('(a) 训练损失变化', fontproperties=zhfont, fontsize=10, pad=8)
    
    ax2.set_xlabel('训练轮次', fontproperties=zhfont)
    ax2.set_ylabel('验证MSE', fontproperties=zhfont)
    ax2.legend(fontsize=8)
    ax2.set_title('(b) 验证损失变化', fontproperties=zhfont, fontsize=10, pad=8)
    
    # Highlight best epoch for S4D-WM
    s4d_val = [e['val'] for e in logs['S4D-WM_d4rl']]
    best_ep = np.argmin(s4d_val) + 1
    best_val = min(s4d_val)
    ax2.annotate(f'最优: {best_val:.3f}', xy=(best_ep, best_val), xytext=(best_ep + 15, best_val + 0.03),
                fontproperties=zhfont_s, fontsize=8, color=C_SSM,
                arrowprops=dict(arrowstyle='->', color=C_SSM, lw=1))
    
    plt.tight_layout()
    save(fig, 'training_curves')

# ============================================================
# Figure 5: MPC (Lollipop style)
# ============================================================
def fig_mpc():
    methods = ['LSTM-MPC', 'Mamba-MPC', 'S4D-WM-MPC']
    mse_vals = [0.0032, 0.0041, 0.0043]
    freq_vals = [0.7, 4.3, 5.1]
    colors = [C_LSTM, C_MAMBA, C_SSM]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
    
    # (a) Tracking MSE - horizontal lollipop
    y = np.arange(len(methods))
    ax1.hlines(y, 0, mse_vals, color=colors, linewidth=2.5, zorder=2)
    ax1.scatter(mse_vals, y, s=150, c=colors, zorder=3, edgecolors='white', linewidth=1.5)
    for i, v in enumerate(mse_vals):
        ax1.text(v + 0.0001, i, f'{v:.4f}', fontsize=9, va='center', color=colors[i], fontweight='bold')
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('跟踪MSE', fontproperties=zhfont)
    ax1.set_title('(a) 轨迹跟踪精度', fontproperties=zhfont, fontsize=10, pad=8)
    ax1.set_xlim(0, 0.006)
    
    # (b) Control Frequency - horizontal lollipop
    ax2.hlines(y, 0, freq_vals, color=colors, linewidth=2.5, zorder=2)
    ax2.scatter(freq_vals, y, s=150, c=colors, zorder=3, edgecolors='white', linewidth=1.5)
    for i, v in enumerate(freq_vals):
        ax2.text(v + 0.1, i, f'{v:.1f} Hz', fontsize=9, va='center', color=colors[i], fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('控制频率 (Hz)', fontproperties=zhfont)
    ax2.set_title('(b) 控制频率', fontproperties=zhfont, fontsize=10, pad=8)
    ax2.set_xlim(0, 7)
    
    # Speedup annotation
    ax2.annotate('7.3×', xy=(5.1, 2), xytext=(3.5, 0.5),
                fontsize=10, ha='center', color=C_SSM, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_SSM, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F0FE', edgecolor=C_SSM, alpha=0.8))
    
    plt.tight_layout()
    save(fig, 'mpc_comparison')

# Run all
if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    fig_batch_inference()
    fig_ablation()
    fig_training_curves()
    fig_mpc()
    print("All figures done!")
