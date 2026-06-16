#!/usr/bin/env python3
"""SeqLen: bar+line combo, recommendation zones as legend entries."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import numpy as np
import json

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
    'mathtext.fontset': 'stix',
})

with open('experiments/seqlen_results_final.json', 'r') as f:
    results = json.load(f)

T_vals = [16, 32, 64, 128, 256]
x = np.arange(len(T_vals))
w = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.2))

mse_h = [r['mse'] for r in results['humanoid']]
mse_a = [r['mse'] for r in results['ant']]
r2_h = [r['r2'] for r in results['humanoid']]
r2_a = [r['r2'] for r in results['ant']]

# ============ Top: MSE ============
ax1.bar(x - w/2, mse_h, w, color='#2E86AB', alpha=0.5, edgecolor='white', linewidth=0.5, zorder=2)
ax1.bar(x + w/2, mse_a, w, color='#A23B72', alpha=0.5, edgecolor='white', linewidth=0.5, zorder=2)
# Lines at bar centers
ax1.plot(x - w/2, mse_h, 'o-', color='#2E86AB', linewidth=1.5, markersize=5, zorder=3, label='Humanoid MSE')
ax1.plot(x + w/2, mse_a, 's-', color='#A23B72', linewidth=1.5, markersize=5, zorder=3, label='Ant MSE')

# Recommendation zones (no text labels, just shading)
ax1.axvspan(-0.5, 1.5, alpha=0.08, color='#2E86AB', zorder=1)
ax1.axvspan(3.5, 4.5, alpha=0.08, color='#A23B72', zorder=1)

ax1.set_ylabel('MSE', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels([])
ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('(a) 预测MSE随序列长度的变化', fontproperties=zhfont, fontsize=10, pad=8)

# ============ Bottom: R2 ============
ax2.bar(x - w/2, r2_h, w, color='#2E86AB', alpha=0.5, edgecolor='white', linewidth=0.5, zorder=2)
ax2.bar(x + w/2, r2_a, w, color='#A23B72', alpha=0.5, edgecolor='white', linewidth=0.5, zorder=2)
ax2.plot(x - w/2, r2_h, 'o-', color='#2E86AB', linewidth=1.5, markersize=5, zorder=3, label='Humanoid $R^2$')
ax2.plot(x + w/2, r2_a, 's-', color='#A23B72', linewidth=1.5, markersize=5, zorder=3, label='Ant $R^2$')

ax2.axvspan(-0.5, 1.5, alpha=0.08, color='#2E86AB', zorder=1)
ax2.axvspan(3.5, 4.5, alpha=0.08, color='#A23B72', zorder=1)

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.6)
ax2.set_xlabel('序列长度 T', fontproperties=zhfont)
ax2.set_ylabel('$R^2$', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(['16', '32', '64', '128', '256'])
ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('(b) $R^2$随序列长度的变化', fontproperties=zhfont, fontsize=10, pad=8)

# Legend: data lines + recommendation zone patches
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
zone_h = [mpatches.Patch(facecolor='#2E86AB', alpha=0.15, label='Humanoid推荐区间'),
          mpatches.Patch(facecolor='#A23B72', alpha=0.15, label='Ant推荐区间')]
all_h = h1 + h2 + zone_h
all_l = l1 + l2 + ['Humanoid推荐区间', 'Ant推荐区间']
fig.legend(all_h, all_l, loc='lower center', ncol=3, fontsize=8,
           bbox_to_anchor=(0.5, -0.04), frameon=True, fancybox=True,
           framealpha=0.9, edgecolor='gray')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig('paper/figures/seqlen_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Done: seqlen_sensitivity.pdf")
