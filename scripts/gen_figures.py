"""生成论文图4(消融实验)和图5(序列长度敏感性)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=7)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
    'mathtext.fontset': 'stix',
})

# ============================================================
# 图4: 消融实验 (水平条形图, 上下分组结构)
# ============================================================
def gen_ablation():
    configs = [
        ('MIMO-WM',       18.69, 0.31, 0.766, 0.208),
        ('w/o gate',      20.36, 0.10, 0.745, 0.142),
        ('w/o residual',  26.14, 0.26, 0.673, 0.208),
        ('w/o LayerNorm', 21.25, 0.48, 0.734, 0.208),
        ('SSM$\\to$LSTM', 38.10, 0.44, 0.523, 0.389),
        ('SSM$\\to$GRU',  32.94, 0.58, 0.588, 0.323),
        ('$D$=64',         21.69, 0.35, 0.729, 0.080),
        ('$D$=256',        18.15, 0.26, 0.773, 0.613),
        ('$L$=1',          19.48, 0.21, 0.756, 0.166),
        ('$L$=4',          18.49, 0.22, 0.769, 0.292),
        ('$N$=8',          18.77, 0.22, 0.765, 0.200),
    ]

    n = len(configs)
    labels = [c[0] for c in configs]
    mses = [c[1] for c in configs]
    mse_stds = [c[2] for c in configs]
    r2s = [c[3] for c in configs]
    params = [c[4] for c in configs]

    BAR = '#4A90D9'
    HL = '#E74C3C'
    # 显式逐个赋色
    colors = []
    for i in range(n):
        colors.append(HL if i == 0 else BAR)

    # 打印验证
    for i in range(n):
        print(f'  {labels[i]:>15s} -> color={colors[i]}', flush=True)

    fig, (ax_mse, ax_r2, ax_params) = plt.subplots(3, 1, figsize=(5.5, 7.5))
    fig.patch.set_facecolor('white')

    y = np.arange(n)
    h = 0.6

    # === 上图: MSE ===
    ax_mse.barh(y, mses, height=h, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
    ax_mse.errorbar(mses, y, xerr=mse_stds, fmt='none', ecolor='#333', capsize=2, linewidth=0.7, zorder=4)
    for i in range(n):
        fw = 'bold' if i == 0 else 'normal'
        ax_mse.text(mses[i] + mse_stds[i] + 0.5, y[i], f'{mses[i]:.1f}',
                    fontsize=7, va='center', color='#222', fontweight=fw)
    ax_mse.set_xlabel('MSE ($\\times 10^{-2}$)', fontsize=9)
    ax_mse.set_xlim(0, 44)
    ax_mse.set_title('(a) MSE', fontsize=9, fontweight='bold', loc='left')
    ax_mse.spines['top'].set_visible(False)
    ax_mse.spines['right'].set_visible(False)
    ax_mse.grid(axis='x', linewidth=0.3, alpha=0.2)
    ax_mse.invert_yaxis()

    # === 中图: R² ===
    ax_r2.barh(y, r2s, height=h, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
    for i in range(n):
        fw = 'bold' if i == 0 else 'normal'
        ax_r2.text(r2s[i] + 0.003, y[i], f'{r2s[i]:.3f}',
                   fontsize=7, va='center', color='#222', fontweight=fw)
    ax_r2.set_xlabel('$R^2$', fontsize=9)
    ax_r2.set_xlim(0.48, 0.80)
    ax_r2.set_title('(b) $R^2$', fontsize=9, fontweight='bold', loc='left')
    ax_r2.spines['top'].set_visible(False)
    ax_r2.spines['right'].set_visible(False)
    ax_r2.grid(axis='x', linewidth=0.3, alpha=0.2)
    ax_r2.invert_yaxis()

    # === 下图: 参数量 ===
    bubble_sizes = [p * 600 for p in params]
    ax_params.scatter(params, y, s=bubble_sizes, c=colors, edgecolors='white',
                      linewidth=0.8, zorder=3, alpha=0.85)
    for i in range(n):
        fw = 'bold' if i == 0 else 'normal'
        ax_params.text(params[i] + 0.02, y[i], f'{params[i]:.1f}M',
                       fontsize=7, va='center', color='#222', fontweight=fw)
    ax_params.set_xlabel('Parameters (M)', fontsize=9)
    ax_params.set_xlim(0, 0.72)
    ax_params.set_title('(c) Parameters', fontsize=9, fontweight='bold', loc='left')
    ax_params.spines['top'].set_visible(False)
    ax_params.spines['right'].set_visible(False)
    ax_params.grid(axis='x', linewidth=0.3, alpha=0.2)
    ax_params.invert_yaxis()

    # Y轴标签只在最上图
    ax_mse.set_yticks(y)
    ax_mse.set_yticklabels(labels, fontsize=8)
    for i, tick in enumerate(ax_mse.get_yticklabels()):
        if i == 0:
            tick.set_fontweight('bold')
            tick.set_color(HL)

    # 中下图隐藏Y轴
    ax_r2.set_yticks([])
    ax_params.set_yticks([])

    plt.tight_layout(pad=0.5, h_pad=0.8)
    plt.savefig('paper/figures/ablation_results.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig('paper/figures/ablation_results.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print('Done: ablation_results.pdf')

# ============================================================
# 图5: 序列长度敏感性 (柱状+折线双轴, 含推荐区间)
# ============================================================
def gen_seqlen():
    # Humanoid 数据 (从之前实验)
    ts = [8, 16, 32, 64, 128]
    humanoid_mse = [20.14, 19.23, 21.18, 21.28, 41.13]
    humanoid_mse_std = [0.13, 0.14, 0.04, 0.16, 0.36]
    humanoid_r2 = [0.765, 0.764, 0.735, 0.708, 0.448]

    # HumanoidStandup 数据 (待补充, 先用占位)
    # 实际需要跑实验后替换
    humanoid_standup_mse = [49.66, 48.5, 53.10, 55.0, 75.0]  # placeholder
    humanoid_standup_r2 = [0.480, 0.490, 0.444, 0.420, 0.300]  # placeholder

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.2, 5.0), sharex=True)
    fig.patch.set_facecolor('white')

    # --- 上图: Humanoid ---
    # 推荐区间背景
    ax1.axvspan(4, 20, alpha=0.08, color='#27ae60', zorder=0)
    ax1.text(12, 42, 'Recommended\n[8, 16]', fontsize=6.5, color='#27ae60',
             ha='center', va='center', style='italic')

    bar_width = 3
    x = np.array(ts)
    bars = ax1.bar(x, humanoid_mse, width=bar_width, color='#3498db', alpha=0.7,
                   edgecolor='white', linewidth=0.5, label='MSE ($\\times 10^{-2}$)')
    ax1.errorbar(x, humanoid_mse, yerr=humanoid_mse_std, fmt='none',
                 ecolor='#555', capsize=2, linewidth=0.8)

    ax1_r = ax1.twinx()
    ax1_r.plot(x, humanoid_r2, 'o-', color='#e74c3c', markersize=5,
               linewidth=1.5, label='$R^2$')
    ax1_r.set_ylim(0.3, 0.85)
    ax1_r.set_ylabel('$R^2$', fontsize=8, color='#e74c3c')
    ax1_r.tick_params(axis='y', labelcolor='#e74c3c', labelsize=7)
    ax1_r.spines['top'].set_visible(False)

    ax1.set_ylabel('MSE ($\\times 10^{-2}$)', fontsize=8, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db', labelsize=7)
    ax1.set_ylim(0, 50)
    ax1.set_title('Humanoid (348-D)', fontsize=8, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.grid(axis='y', linewidth=0.3, alpha=0.2)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6.5,
               framealpha=0.9, edgecolor='gray')

    # --- 下图: HumanoidStandup ---
    ax2.axvspan(16, 40, alpha=0.08, color='#27ae60', zorder=0)
    ax2.text(26, 72, 'Recommended\n[16, 32]', fontsize=6.5, color='#27ae60',
             ha='center', va='center', style='italic')

    bars2 = ax2.bar(x, humanoid_standup_mse, width=bar_width, color='#3498db', alpha=0.7,
                    edgecolor='white', linewidth=0.5, label='MSE ($\\times 10^{-2}$)')

    ax2_r = ax2.twinx()
    ax2_r.plot(x, humanoid_standup_r2, 'o-', color='#e74c3c', markersize=5,
               linewidth=1.5, label='$R^2$')
    ax2_r.set_ylim(0.2, 0.55)
    ax2_r.set_ylabel('$R^2$', fontsize=8, color='#e74c3c')
    ax2_r.tick_params(axis='y', labelcolor='#e74c3c', labelsize=7)
    ax2_r.spines['top'].set_visible(False)

    ax2.set_ylabel('MSE ($\\times 10^{-2}$)', fontsize=8, color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db', labelsize=7)
    ax2.set_ylim(0, 85)
    ax2.set_xlabel('Sequence Length $T$', fontsize=9)
    ax2.set_xticks(ts)
    ax2.set_title('HumanoidStandup (376-D)', fontsize=8, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.grid(axis='y', linewidth=0.3, alpha=0.2)

    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=6.5,
               framealpha=0.9, edgecolor='gray')

    plt.tight_layout(pad=0.5, h_pad=0.3)
    plt.savefig('paper/figures/seqlen_sensitivity.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/seqlen_sensitivity.png', dpi=300, bbox_inches='tight')
    print('Done: seqlen_sensitivity.pdf (HumanoidStandup为占位数据, 需跑实验替换)')

if __name__ == '__main__':
    gen_ablation()
    gen_seqlen()
