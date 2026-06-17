#!/usr/bin/env python3
"""SeqLen: bar+line, 4-item legend (Humanoid, Ant, zones)."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
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

x = np.arange(5)
w = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5))

mse_h = [r['mse'] for r in results['humanoid']]
mse_a = [r['mse'] for r in results['ant']]
r2_h = [r['r2'] for r in results['humanoid']]
r2_a = [r['r2'] for r in results['ant']]

# ============ Top: MSE ============
ax1.bar(x - w/2, mse_h, w, color='#2E86AB', alpha=0.5, edgecolor='none', zorder=2)
ax1.bar(x + w/2, mse_a, w, color='#A23B72', alpha=0.5, edgecolor='none', zorder=2)
ax1.plot(x - w/2, mse_h, 'o-', color='#2E86AB', linewidth=1.5, markersize=5, zorder=3)
ax1.plot(x + w/2, mse_a, 's-', color='#A23B72', linewidth=1.5, markersize=5, zorder=3)
ax1.axvspan(-0.5, 1.5, alpha=0.08, color='#2E86AB', zorder=1)
ax1.axvspan(3.5, 4.5, alpha=0.08, color='#A23B72', zorder=1)

ax1.set_ylabel('MSE', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels([])
ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.text(0.5, 1.02, '(a) 预测MSE随序列长度的变化', transform=ax1.transAxes,
         fontproperties=zhfont, fontsize=10, ha='center', va='bottom')

# ============ Bottom: R2 ============
ax2.bar(x - w/2, r2_h, w, color='#2E86AB', alpha=0.5, edgecolor='none', zorder=2)
ax2.bar(x + w/2, r2_a, w, color='#A23B72', alpha=0.5, edgecolor='none', zorder=2)
ax2.plot(x - w/2, r2_h, 'o-', color='#2E86AB', linewidth=1.5, markersize=5, zorder=3)
ax2.plot(x + w/2, r2_a, 's-', color='#A23B72', linewidth=1.5, markersize=5, zorder=3)
ax2.axvspan(-0.5, 1.5, alpha=0.08, color='#2E86AB', zorder=1)
ax2.axvspan(3.5, 4.5, alpha=0.08, color='#A23B72', zorder=1)

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.6)
ax2.set_xlabel('序列长度 T', fontproperties=zhfont, fontsize=10)
ax2.set_ylabel('$R^2$', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(['16', '32', '64', '128', '256'])
ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(0.5, 1.02, '(b) $R^2$随序列长度的变化', transform=ax2.transAxes,
         fontproperties=zhfont, fontsize=10, ha='center', va='bottom')

# 4-item legend
legend_elements = [
    Line2D([0], [0], color='#2E86AB', marker='o', linewidth=1.5, markersize=5, label='Humanoid'),
    Line2D([0], [0], color='#A23B72', marker='s', linewidth=1.5, markersize=5, label='Ant'),
    mpatches.Patch(facecolor='#2E86AB', alpha=0.15, label='Humanoid推荐区间'),
    mpatches.Patch(facecolor='#A23B72', alpha=0.15, label='Ant推荐区间'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8.5, prop=zhfont_s,
           bbox_to_anchor=(0.5, -0.03), frameon=True, fancybox=True,
           framealpha=0.9, edgecolor='gray')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.savefig('paper/figures/seqlen_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Done: seqlen_sensitivity.pdf")
