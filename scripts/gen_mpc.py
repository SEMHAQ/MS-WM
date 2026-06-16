"""MPC comparison - top-bottom, Chinese font fixed, 4 models no legend needed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)
zhfont_title = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

methods = ['LSTM', 'GRU', 'Mamba', 'S4D-WM']
loop_time = [299, 1265, 1296, 1298]
freq = [3.3, 0.8, 0.8, 0.8]
colors = ['#B0B0B0', '#B0B0B0', '#B0B0B0', '#1f77b4']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.2, 5.0))

# (a) Loop time
bars1 = ax1.bar(range(len(methods)), loop_time, 0.55, color=colors,
                edgecolor='white', linewidth=0.5, zorder=3)
ax1.set_ylabel('Loop Time (ms)', fontsize=10)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=9)
ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, val in zip(bars1, loop_time):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.set_title('(a) MPC Loop Time', fontsize=10, pad=8)

# (b) Control frequency
bars2 = ax2.bar(range(len(methods)), freq, 0.55, color=colors,
                edgecolor='white', linewidth=0.5, zorder=3)
ax2.set_ylabel('Control Freq. (Hz)', fontsize=10)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=9)
ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=10, color='#CC3333', linestyle='--', alpha=0.6, linewidth=1, zorder=2)
ax2.text(len(methods)-0.5, 10.5, 'Typical requirement (10 Hz)',
         fontsize=7.5, color='#CC3333', ha='right')
for bar, val in zip(bars2, freq):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.set_title('(b) MPC Control Frequency', fontsize=10, pad=8)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/mpc_comparison.pdf', dpi=300, bbox_inches='tight')
print("Done: mpc_comparison.pdf")
