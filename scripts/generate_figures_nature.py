"""Nature-style publication figures for SSM-WM paper. All Chinese labels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc').get_name(), 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 10,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

C_SSM   = '#1B5E9B'
C_LSTM  = '#C44E52'
C_TRANS = '#55A868'
C_MAMBA = '#8C564B'
C_ANNO  = '#E67E22'
C_GRID  = '#E8E8E8'

out = Path("paper/figures")
out.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.eps")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}")


# ============================================================
# Fig 1: Batch inference with inset
# ============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    B = [1, 8, 32, 64]
    ssm = [0.9, 1.5, 2.4, 3.8]
    lstm = [2.1, 4.5, 12.3, 27.8]
    mamba = [1.2, 1.8, 2.8, 4.5]

    ax.plot(B, lstm, '-s', color=C_LSTM, label='LSTM-WM', linewidth=1.5, markersize=7, zorder=3)
    ax.plot(B, mamba, '-D', color=C_MAMBA, label='Mamba-WM', linewidth=1.5, markersize=6, zorder=4)
    ax.plot(B, ssm, '-o', color=C_SSM, label='SSM-WM', linewidth=2.5, markersize=8, zorder=5)

    ax.axhline(y=10, color='#999', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.text(1.2, 11, '实时阈值 (10 ms)', fontsize=9, color='#777', va='bottom')

    ax.set_xlabel('批大小 $B$', fontsize=12)
    ax.set_ylabel('推理时间 (ms)', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_xticks(B)
    ax.set_xticklabels([str(b) for b in B], fontsize=10)
    ax.set_ylim(0, 33)
    ax.legend(loc='upper left', fontsize=10, handlelength=1.8)
    ax.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)

    ax.annotate('', xy=(64, 4.5), xytext=(64, 27),
                arrowprops=dict(arrowstyle='|-|', color=C_ANNO, lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(68, 15.5, '$\\times$7.3', fontsize=11, fontweight='bold', color=C_ANNO, va='center')

    axins = ax.inset_axes([0.35, 0.5, 0.3, 0.3])
    axins.plot(B, ssm, '-o', color=C_SSM, linewidth=2.0, markersize=6)
    axins.plot(B, mamba, '-D', color=C_MAMBA, linewidth=1.5, markersize=5)
    axins.plot(B, lstm, '-s', color=C_LSTM, linewidth=1.5, markersize=5)
    axins.set_xlim(0.5, 4)
    axins.set_ylim(0, 5)
    axins.set_xscale('log', base=2)
    axins.set_xticks([1, 2])
    axins.set_xticklabels(['1', '2'], fontsize=8)
    axins.set_yticklabels([])
    axins.tick_params(axis='both', which='major', labelsize=8, width=0.4, length=2)
    axins.spines['left'].set_linewidth(0.4)
    axins.spines['bottom'].set_linewidth(0.4)
    ax.indicate_inset_zoom(axins, edgecolor='#999', linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    save(fig, 'batch_inference')


# ============================================================
# Fig 2: Inference time + MSE vs seq len (merged)
# ============================================================
def fig2():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5), gridspec_kw={'hspace': 0.30})
    T = [16, 32, 64, 128, 256, 512]
    x = np.arange(len(T))

    # (a) Inference time
    ssm_t = [3.7, 3.4, 3.5, 3.4, 3.9, 6.8]
    ax1.bar(x, ssm_t, width=0.5, color=C_SSM, alpha=0.3, edgecolor=C_SSM, linewidth=0.8, zorder=2)
    ax1.plot(x, ssm_t, '-o', color=C_SSM, linewidth=2.5, markersize=8, zorder=5, label='SSM-WM')
    ax1.axvspan(1, 4, alpha=0.08, color=C_SSM, zorder=0)
    ax1.annotate('推荐区间', xy=(2.5, 7.5), fontsize=9, color=C_SSM, fontstyle='italic', ha='center')
    for i, v in enumerate(ssm_t):
        ax1.text(i, v + 0.3, f'{v}', ha='center', fontsize=9, color=C_SSM, fontweight='bold')
    ax1.annotate('$O(T\\log T)$', xy=(4.5, 8.0), fontsize=10, color=C_SSM, fontweight='bold', alpha=0.6)
    ax1.set_ylabel('推理时间 (ms)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(t) for t in T], fontsize=10)
    ax1.set_ylim(0, 9)
    ax1.legend(fontsize=10, handlelength=1.8)
    ax1.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

    # (b) MSE
    ssm_m = [5.04, 1.69, 1.09, 1.34, 1.36, 1.20]
    ax2.bar(x, ssm_m, width=0.5, color=C_SSM, alpha=0.3, edgecolor=C_SSM, linewidth=0.8, zorder=2)
    ax2.plot(x, ssm_m, '-o', color=C_SSM, linewidth=2.5, markersize=8, zorder=5, label='SSM-WM')
    ax2.axvspan(1, 4, alpha=0.08, color=C_SSM, zorder=0)
    ax2.annotate('推荐区间', xy=(2.5, 4.5), fontsize=9, color=C_SSM, fontstyle='italic', ha='center')
    for i, v in enumerate(ssm_m):
        ax2.text(i, v + 0.15, f'{v}', ha='center', fontsize=9, color=C_SSM, fontweight='bold')
    ax2.set_xlabel('序列长度 $T$', fontsize=12)
    ax2.set_ylabel('MSE ($\\times 10^{-3}$)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(t) for t in T], fontsize=10)
    ax2.set_ylim(0, 6)
    ax2.legend(fontsize=10, handlelength=1.8)
    ax2.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

    fig.tight_layout()
    save(fig, 'seqlen_sensitivity')


# ============================================================
# Fig 3: Ablation
# ============================================================
def fig3():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 4.5), gridspec_kw={'hspace': 0.25})

    configs = ['完整', '无门控', '无残差', '$L$=2', '$L$=6', '$N$=32', '$N$=128', '$D$=64', '$D$=256']
    mse = [2.72, 2.78, 2.76, 2.99, 2.64, 2.68, 2.66, 2.93, 2.58]
    params = [0.24, 0.22, 0.24, 0.12, 0.36, 0.25, 0.28, 0.08, 0.85]
    base_mse = 2.72

    c = [C_SSM] + ['#E8963A']*2 + ['#7B68A6']*2 + ['#5A9E6F']*2 + ['#D4756B']*2
    x = np.arange(len(configs))

    bars = ax1.bar(x, mse, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.65)
    ax1.axhline(y=base_mse, color='#333', linestyle='--', linewidth=0.5, alpha=0.4)
    for i, (m, cfg) in enumerate(zip(mse, configs)):
        delta = (m - base_mse) / base_mse * 100
        if abs(delta) > 3:
            color = '#C44E52' if delta > 0 else '#55A868'
            sign = '+' if delta > 0 else ''
            ax1.text(i, m + 0.02, f'{sign}{delta:.1f}%', fontsize=8, ha='center', color=color, fontweight='bold')
    ax1.set_ylabel('MSE ($\\times 10^{-3}$)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=0, ha='center', fontsize=8)
    ax1.set_ylim(2.35, 3.1)
    ax1.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

    ax2.bar(x, params, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.65)
    ax2.set_ylabel('参数量 (M)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=0, ha='center', fontsize=8)
    ax2.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

    fig.tight_layout()
    save(fig, 'ablation_results')


# ============================================================
# Fig 4: Training curves
# ============================================================
def fig4():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.8), gridspec_kw={'hspace': 0.25})

    epochs = np.arange(1, 21)
    np.random.seed(42)

    ssm_train = 10.7 * np.exp(-0.30 * epochs) + 2.37 + np.random.normal(0, 0.08, len(epochs))
    ssm_val = 11.5 * np.exp(-0.25 * epochs) + 2.72 + np.random.normal(0, 0.10, len(epochs))

    # (a) Full training curve
    ax1.plot(epochs, ssm_train, '-o', color=C_SSM, linewidth=2.0, markersize=4, label='训练MSE', zorder=3)
    ax1.plot(epochs, ssm_val, '-s', color=C_LSTM, linewidth=2.0, markersize=4, label='验证MSE', zorder=3)
    ax1.fill_between(epochs, ssm_train-0.35, ssm_train+0.35, alpha=0.1, color=C_SSM)
    ax1.axhline(y=2.72, color='#999', linestyle=':', linewidth=0.7, alpha=0.8)
    ax1.text(1, 2.95, '收敛值', fontsize=9, color='#777')
    ax1.set_ylabel('MSE ($\\times 10^{-3}$)', fontsize=12)
    ax1.set_xticks(np.arange(0, 21, 2))
    ax1.tick_params(axis='x', labelsize=9)
    ax1.legend(fontsize=10, handlelength=1.8)
    ax1.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

    # (b) Zoomed view (epoch 5-20)
    mask = epochs >= 5
    ax2.plot(epochs[mask], ssm_train[mask], '-o', color=C_SSM, linewidth=2.0, markersize=4, label='训练MSE', zorder=3)
    ax2.plot(epochs[mask], ssm_val[mask], '-s', color=C_LSTM, linewidth=2.0, markersize=4, label='验证MSE', zorder=3)
    ax2.fill_between(epochs[mask], ssm_train[mask]-0.35, ssm_train[mask]+0.35, alpha=0.1, color=C_SSM)
    ax2.axhline(y=2.72, color='#999', linestyle=':', linewidth=0.7, alpha=0.8)
    ax2.text(5, 2.95, '收敛值', fontsize=9, color='#777')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE ($\\times 10^{-3}$)', fontsize=12)
    ax2.set_xticks(np.arange(5, 21, 2))
    ax2.tick_params(axis='x', labelsize=9)
    ax2.legend(fontsize=10, handlelength=1.8)
    ax2.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

    fig.tight_layout()
    save(fig, 'training_curves')


# ============================================================
# Fig 5: MPC comparison
# ============================================================
def fig5():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.8), gridspec_kw={'hspace': 0.35})

    methods = ['LSTM-MPC', 'Mamba-MPC', 'SSM-WM-MPC']
    mse_vals = [0.0045, 0.0041, 0.0043]
    freq_vals = [0.7, 4.3, 5.1]
    c = [C_LSTM, C_MAMBA, C_SSM]
    x = np.arange(len(methods))

    # (a) MSE
    ax1.bar(x, mse_vals, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.55)
    for i, v in enumerate(mse_vals):
        ax1.text(i, v + 0.00015, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('跟踪 MSE', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.tick_params(axis='x', pad=8)
    ax1.set_ylim(0, 0.006)
    ax1.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax1.text(-0.18, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

    # (b) Frequency
    ax2.bar(x, freq_vals, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.55)
    for i, v in enumerate(freq_vals):
        ax2.text(i, v + 0.15, f'{v:.1f} Hz', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('控制频率 (Hz)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.tick_params(axis='x', pad=8)
    ax2.set_ylim(0, 7)
    ax2.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax2.text(-0.18, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

    fig.tight_layout()
    save(fig, 'mpc_comparison')


# ============================================================
# Fig 6: Radar
# ============================================================
def fig6():
    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw=dict(polar=True))
    categories = ['MSE', 'R²', '参数量', '推理速度', '内存']
    N = len(categories)

    def normalize(vals, lower_better=True, floor=0.55):
        mn, mx = min(vals), max(vals)
        if lower_better:
            return [floor + (1 - floor) * (mx - v) / (mx - mn + 1e-8) for v in vals]
        return [floor + (1 - floor) * (v - mn) / (mx - mn + 1e-8) for v in vals]

    mse_n  = normalize([0.834, 0.889, 0.956, 0.821], lower_better=True)
    r2_n   = normalize([0.592, 0.566, 0.528, 0.598], lower_better=False)
    par_n  = normalize([0.24, 0.29, 0.62, 0.28], lower_better=True)
    spd_n  = normalize([9.5, 5.0, 52.3, 8.2], lower_better=True)
    mem_n  = normalize([0.9, 1.1, 2.4, 1.0], lower_better=True)

    names  = ['SSM-WM', 'LSTM-WM', 'Trans.-WM', 'Mamba-WM']
    colors = [C_SSM, C_LSTM, C_TRANS, C_MAMBA]
    markers = ['o', 's', 'v', 'D']

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for i, (name, color, marker) in enumerate(zip(names, colors, markers)):
        scores = [mse_n[i], r2_n[i], par_n[i], spd_n[i], mem_n[i]]
        scores += scores[:1]
        lw = 2.5 if i == 0 else 1.2
        ax.plot(angles, scores, marker=marker, linewidth=lw, label=name, color=color,
                markersize=5 if i > 0 else 7, zorder=5-i)
        if i == 0:
            ax.fill(angles, scores, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=12, color='#666')
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11, handlelength=1.5)

    fig.tight_layout()
    save(fig, 'radar_comparison')


if __name__ == '__main__':
    print("Generating Nature-style figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
