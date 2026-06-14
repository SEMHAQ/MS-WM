"""Radar chart: D4RL Humanoid, 4 models, intuitive Chinese labels."""
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

models = ['S4D-WM', 'Mamba-WM', 'Trans.-WM', 'LSTM-WM']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# D4RL Humanoid raw data (Table 8)
# All metrics normalized so that HIGHER = BETTER
raw_scores = {
    #           R²     预测精度(1/MSE)  推理速度(1/ms)  参数效率(1/M)
    'S4D-WM':    [0.694,  1/0.245,        1/3.4,          1/0.23],
    'Mamba-WM':  [0.676,  1/0.259,        1/3.5,          1/0.66],
    'Trans.-WM': [0.653,  1/0.278,        1/1.6,          1/0.15],
    'LSTM-WM':   [0.541,  1/0.367,        1/2.5,          1/0.64],
}

categories = ['R2', '预测精度', '推理速度', '参数效率']
N = len(categories)

def normalize(col_idx):
    v = np.array([raw_scores[m][col_idx] for m in models])
    mn, mx = v.min(), v.max()
    return 0.2 + 0.8 * (v - mn) / (mx - mn) if mx - mn > 1e-10 else np.full(4, 0.5)

all_n = [normalize(i) for i in range(N)]

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(4.5, 4.2), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

for i, m in enumerate(models):
    vals = [all_n[j][i] for j in range(N)] + [all_n[0][i]]
    ax.plot(angles, vals, '-', color=colors[i], linewidth=1.5, zorder=3)
    ax.fill(angles, vals, color=colors[i], alpha=0.08)
    for j in range(N):
        ax.plot(angles[j], vals[j], marker=markers[i], color=colors[i],
                markersize=6, markeredgecolor='white', markeredgewidth=0.6, zorder=4)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontproperties=zhfont, fontweight='bold')
ax.tick_params(pad=15)

ax.set_ylim(0, 1.15)
ax.set_yticks([])
ax.grid(True, linewidth=0.3, alpha=0.4)

legend_lines = [plt.Line2D([0], [0], color=colors[i], marker=markers[i],
                markersize=6, linewidth=1.5, markeredgecolor='white')
                for i in range(N)]
ax.legend(legend_lines, models, loc='upper right', bbox_to_anchor=(1.38, 1.15),
          fontsize=8.5, frameon=True, fancybox=True, framealpha=0.9, edgecolor='gray',
          handletextpad=0.4, handlelength=1.5)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/radar_comparison.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig('paper/figures/radar_comparison.png', bbox_inches='tight', pad_inches=0.1)
print("Done")
