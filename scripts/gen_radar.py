"""Radar chart: Humanoid, 6 models, MS-WM highlighted."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=11)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# MS-WM last (highlighted)
models = ['LSTM', 'GRU', 'Trans.', 'Mamba', 'S4D-WM', 'MS-WM']
colors = ['#d62728', '#9467bd', '#2ca02c', '#ff7f0e', '#1f77b4', '#e91e63']
markers = ['D', 'v', '^', 's', 'o', '*']

# Data from experiments (Humanoid)
# R², 1/MSE (higher is better), 1/inf_time (higher is better), 1/params (higher is better)
raw_scores = {
    'LSTM':    [0.489,  1/40.87,  1/2.52,  1/0.64],
    'GRU':     [0.555,  1/35.60,  1/1.81,  1/0.50],
    'Trans.':  [0.677,  1/25.85,  1/9.65,  1/0.15],
    'Mamba':   [0.658,  1/27.34,  1/4.55,  1/0.66],
    'S4D-WM':  [0.661,  1/27.13,  1/4.43,  1/0.23],
    'MS-WM':   [0.737,  1/21.00,  1/2.16,  1/0.085],
}

categories = ['$R^2$', '预测精度', '推理速度', '参数效率']
N = len(categories)

def normalize(col_idx):
    v = np.array([raw_scores[m][col_idx] for m in models])
    mn, mx = v.min(), v.max()
    return 0.2 + 0.8 * (v - mn) / (mx - mn) if mx - mn > 1e-10 else np.full(len(models), 0.5)

all_n = [normalize(i) for i in range(N)]

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(4.5, 4.2), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

for i, m in enumerate(models):
    vals = [all_n[j][i] for j in range(N)] + [all_n[0][i]]
    lw = 2.5 if m == 'MS-WM' else 1.3
    ax.plot(angles, vals, '-', color=colors[i], linewidth=lw, zorder=3)
    ax.fill(angles, vals, color=colors[i], alpha=0.06 if m != 'MS-WM' else 0.15)
    for j in range(N):
        ms = 8 if m == 'MS-WM' else 5.5
        ax.plot(angles[j], vals[j], marker=markers[i], color=colors[i],
                markersize=ms, markeredgecolor='white', markeredgewidth=0.6, zorder=4)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontproperties=zhfont, fontweight='bold')
ax.tick_params(pad=15)

ax.set_ylim(0, 1.15)
ax.set_yticks([])
ax.grid(True, linewidth=0.3, alpha=0.4)

legend_lines = [plt.Line2D([0], [0], color=colors[i], marker=markers[i],
                markersize=6, linewidth=1.5, markeredgecolor='white')
                for i in range(len(models))]
ax.legend(legend_lines, models, loc='upper right', bbox_to_anchor=(1.38, 1.15),
          fontsize=8.5, frameon=True, fancybox=True, framealpha=0.9, edgecolor='gray',
          handletextpad=0.4, handlelength=1.5)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/radar_comparison.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig('paper/figures/radar_comparison.png', bbox_inches='tight', pad_inches=0.1)
print("Done: radar_comparison.pdf")
