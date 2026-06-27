#!/usr/bin/env python3
"""SeqLen: bar+line, Humanoid only, adjusted proportions for single column."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
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

# Load data
with open('experiments/seqlen_results.json', 'r') as f:
    results = json.load(f)

# Extract data for Humanoid
seq_lengths = ['T16', 'T32', 'T64', 'T128', 'T256']
mse_values = []
r2_values = []

for t in seq_lengths:
    if t in results:
        seeds = [results[t][s] for s in results[t] if isinstance(results[t][s], dict) and 'mse' in results[t][s]]
        if seeds:
            mse_values.append(np.mean([s['mse'] for s in seeds]))
            r2_values.append(np.mean([s['r2'] for s in seeds]))
        else:
            mse_values.append(0)
            r2_values.append(0)
    else:
        mse_values.append(0)
        r2_values.append(0)

x = np.arange(len(seq_lengths))

# Create figure with adjusted proportions (less tall)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 5.0))

# ============ Top: MSE ============
ax1.bar(x, mse_values, 0.6, color='#2E86AB', alpha=0.7, edgecolor='none', zorder=2)
ax1.plot(x, mse_values, 'o-', color='#2E86AB', linewidth=1.5, markersize=5, zorder=3)

ax1.set_ylabel('MSE', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels([])
ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.text(0.5, 1.02, '(a) 预测MSE随序列长度的变化', transform=ax1.transAxes,
         fontproperties=zhfont, fontsize=10, ha='center', va='bottom')

# ============ Bottom: R2 ============
ax2.bar(x, r2_values, 0.6, color='#2E86AB', alpha=0.7, edgecolor='none', zorder=2)
ax2.plot(x, r2_values, 'o-', color='#2E86AB', linewidth=1.5, markersize=5, zorder=3)

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

plt.tight_layout()
plt.savefig('paper/figures/seqlen_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Done: seqlen_sensitivity.pdf")
print(f"MSE: {[f'{v:.4f}' for v in mse_values]}")
print(f"R2: {[f'{v:.4f}' for v in r2_values]}")
