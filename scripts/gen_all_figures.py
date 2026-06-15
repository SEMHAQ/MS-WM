"""Regenerate all figures with proper styles."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os, json

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=11)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)
zhfont_l = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=12)

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'font.size': 10,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.8, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C_SSM = '#1B5E9B'; C_LSTM = '#C44E52'; C_TRANS = '#55A868'; C_MAMBA = '#8C564B'
C_GRID = '#E8E8E8'
OUTDIR = 'paper/figures'

def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f'{name}.pdf'))
    fig.savefig(os.path.join(OUTDIR, f'{name}.png'), dpi=300)
    plt.close(fig)
    print(f'  Saved {name}')

# ============================================================
# Figure 2: Seqlen sensitivity - bar+line combo, recommended range
# ============================================================
def fig2():
    T = [16, 32, 64, 128, 256]
    x = np.arange(len(T))
    
    # Humanoid data
    h_mse = [0.291, 0.442, 0.612, 1.213, 2.146]
    h_r2 = [0.656, 0.479, 0.153, -0.623, -1.694]
    # Ant data
    a_mse = [0.542, 0.728, 0.942, 0.934, 0.480]
    a_r2 = [0.302, 0.150, -0.019, 0.139, 0.131]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7.0), gridspec_kw={'hspace': 0.30})
    
    # (a) MSE - bar+line
    w = 0.3
    ax1.bar(x - w/2, h_mse, w, color=C_SSM, alpha=0.3, edgecolor=C_SSM, linewidth=0.8, zorder=2)
    ax1.bar(x + w/2, a_mse, w, color=C_LSTM, alpha=0.3, edgecolor=C_LSTM, linewidth=0.8, zorder=2)
    ax1.plot(x - w/2, h_mse, '-o', color=C_SSM, linewidth=2.0, markersize=6, zorder=5, label='Humanoid (348D)')
    ax1.plot(x + w/2, a_mse, '-s', color=C_LSTM, linewidth=2.0, markersize=6, zorder=5, label='Ant (105D)')
    
    # Recommended ranges
    ax1.axvspan(-0.3, 1.3, alpha=0.06, color=C_SSM, zorder=0)
    ax1.axvspan(2.7, 4.3, alpha=0.06, color=C_LSTM, zorder=0)
    ax1.annotate('Humanoid推荐', xy=(0.5, 1.8), fontproperties=zhfont_s, fontsize=8, color=C_SSM, ha='center', fontstyle='italic')
    ax1.annotate('Ant推荐', xy=(4, 1.8), fontproperties=zhfont_s, fontsize=8, color=C_LSTM, ha='center', fontstyle='italic')
    
    ax1.set_ylabel('MSE', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(t) for t in T], fontsize=10)
    ax1.legend(fontsize=9, handlelength=1.8)
    ax1.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # (b) R² - bar+line
    ax2.bar(x - w/2, h_r2, w, color=C_SSM, alpha=0.3, edgecolor=C_SSM, linewidth=0.8, zorder=2)
    ax2.bar(x + w/2, a_r2, w, color=C_LSTM, alpha=0.3, edgecolor=C_LSTM, linewidth=0.8, zorder=2)
    ax2.plot(x - w/2, h_r2, '-o', color=C_SSM, linewidth=2.0, markersize=6, zorder=5, label='Humanoid (348D)')
    ax2.plot(x + w/2, a_r2, '-s', color=C_LSTM, linewidth=2.0, markersize=6, zorder=5, label='Ant (105D)')
    
    ax2.axvspan(-0.3, 1.3, alpha=0.06, color=C_SSM, zorder=0)
    ax2.axvspan(2.7, 4.3, alpha=0.06, color=C_LSTM, zorder=0)
    ax2.axhline(y=0, color='#999', linestyle=':', linewidth=0.7, alpha=0.8)
    
    ax2.set_xlabel('序列长度 T', fontproperties=zhfont)
    ax2.set_ylabel('R²', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(t) for t in T], fontsize=10)
    ax2.legend(fontsize=9, handlelength=1.8)
    ax2.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    fig.tight_layout()
    save(fig, 'seqlen_sensitivity')

# ============================================================
# Figure 3: Ablation - larger axis labels
# ============================================================
def fig3():
    configs = ['默认', '无门控', '无残差', 'L=2', 'L=6', 'D=64', 'D=256']
    mse = [0.245, 0.252, 0.249, 0.268, 0.238, 0.263, 0.235]
    params = [0.23, 0.22, 0.24, 0.12, 0.36, 0.08, 0.85]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5))
    
    colors_m = [C_SSM if i == 0 else '#A0C4E8' for i in range(len(configs))]
    bars = ax1.bar(range(len(configs)), mse, color=colors_m, alpha=0.9, edgecolor='white', width=0.6)
    bars[0].set_edgecolor(C_SSM); bars[0].set_linewidth(1.5)
    
    base = mse[0]
    for i, (bar, v) in enumerate(zip(bars, mse)):
        if i == 0: continue
        delta = (v - base) / base * 100
        color = '#C44E52' if delta > 0 else '#55A868'
        sign = '+' if delta > 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                f'{sign}{delta:.1f}%', fontsize=7, ha='center', color=color, fontweight='bold')
    
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, fontproperties=zhfont, fontsize=10)
    ax1.set_ylabel('MSE', fontsize=11)
    ax1.set_title('(a) 预测MSE', fontproperties=zhfont, fontsize=11, pad=8)
    
    bars2 = ax2.bar(range(len(configs)), params, color=colors_m, alpha=0.9, edgecolor='white', width=0.6)
    bars2[0].set_edgecolor(C_SSM); bars2[0].set_linewidth(1.5)
    for bar, v in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f'{v:.2f}M', fontsize=8, ha='center', color='#333')
    
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, fontproperties=zhfont, fontsize=10)
    ax2.set_ylabel('参数量 (M)', fontproperties=zhfont, fontsize=11)
    ax2.set_title('(b) 参数量', fontproperties=zhfont, fontsize=11, pad=8)
    
    plt.tight_layout()
    save(fig, 'ablation_results')

# ============================================================
# Figure 4: Training curves - original style with fill_between + zoomed
# ============================================================
def fig4():
    with open('experiments/d4rl_all_experiments.json') as f:
        data = json.load(f)
    logs = data['training_logs']
    
    # Use S4D-WM training curve (single model, like original)
    s4d = logs['S4D-WM_d4rl']
    epochs = np.array([e['epoch'] for e in s4d])
    train_mse = np.array([e['train'] for e in s4d])
    val_mse = np.array([e['val'] for e in s4d])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.8), gridspec_kw={'hspace': 0.25})
    
    # (a) Full training curve
    ax1.plot(epochs, train_mse, '-o', color=C_SSM, linewidth=2.0, markersize=3, label='训练MSE', zorder=3)
    ax1.plot(epochs, val_mse, '-s', color=C_LSTM, linewidth=2.0, markersize=3, label='验证MSE', zorder=3)
    ax1.fill_between(epochs, train_mse * 0.9, train_mse * 1.1, alpha=0.1, color=C_SSM)
    
    best_val = min(val_mse)
    ax1.axhline(y=best_val, color='#999', linestyle=':', linewidth=0.7, alpha=0.8)
    ax1.text(5, best_val + 0.01, f'收敛值 {best_val:.3f}', fontproperties=zhfont_s, fontsize=8, color='#777')
    
    ax1.set_ylabel('MSE', fontsize=11)
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.tick_params(axis='x', labelsize=9)
    ax1.legend(fontsize=10, handlelength=1.8)
    ax1.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)
    ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # (b) Zoomed view (epoch 10-100)
    mask = epochs >= 10
    ax2.plot(epochs[mask], train_mse[mask], '-o', color=C_SSM, linewidth=2.0, markersize=3, label='训练MSE', zorder=3)
    ax2.plot(epochs[mask], val_mse[mask], '-s', color=C_LSTM, linewidth=2.0, markersize=3, label='验证MSE', zorder=3)
    ax2.fill_between(epochs[mask], train_mse[mask] * 0.95, train_mse[mask] * 1.05, alpha=0.1, color=C_SSM)
    ax2.axhline(y=best_val, color='#999', linestyle=':', linewidth=0.7, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MSE', fontsize=11)
    ax2.set_xticks(np.arange(10, 101, 10))
    ax2.tick_params(axis='x', labelsize=9)
    ax2.legend(fontsize=10, handlelength=1.8)
    ax2.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)
    ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    fig.tight_layout()
    save(fig, 'training_curves')

# ============================================================
# Figure 5: MPC - grouped bar with value labels and speedup annotation
# ============================================================
def fig5():
    methods = ['LSTM-MPC', 'Mamba-MPC', 'S4D-WM-MPC']
    mse_vals = [0.0032, 0.0041, 0.0043]
    freq_vals = [0.7, 4.3, 5.1]
    colors = [C_LSTM, C_MAMBA, C_SSM]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # (a) Tracking MSE - horizontal bar with value labels
    y = np.arange(len(methods))
    bars1 = ax1.barh(y, mse_vals, height=0.5, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars1, mse_vals):
        ax1.text(v + 0.00005, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', fontsize=10, va='center', fontweight='bold', color='#333')
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('跟踪MSE', fontproperties=zhfont, fontsize=11)
    ax1.set_title('(a) 轨迹跟踪精度', fontproperties=zhfont, fontsize=11, pad=8)
    ax1.set_xlim(0, 0.006)
    
    # (b) Control Frequency - horizontal bar with speedup
    bars2 = ax2.barh(y, freq_vals, height=0.5, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars2, freq_vals):
        ax2.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                f'{v:.1f} Hz', fontsize=10, va='center', fontweight='bold', color='#333')
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('控制频率 (Hz)', fontproperties=zhfont, fontsize=11)
    ax2.set_title('(b) 控制频率', fontproperties=zhfont, fontsize=11, pad=8)
    ax2.set_xlim(0, 7)
    
    # Speedup bracket
    ax2.annotate('', xy=(5.1, -0.1), xytext=(0.7, -0.1),
                arrowprops=dict(arrowstyle='<->', color=C_SSM, lw=1.5))
    ax2.text(2.9, -0.35, '7.3×加速', fontproperties=zhfont, fontsize=10, ha='center', color=C_SSM, fontweight='bold')
    
    plt.tight_layout()
    save(fig, 'mpc_comparison')

# Run all
if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    fig2()
    fig3()
    fig4()
    fig5()
    print("All figures done!")
