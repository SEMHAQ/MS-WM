"""MPC comparison - top-bottom layout, Nature/Science style."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.2, 4.0))

# (a) Loop time
bars1 = ax1.bar(range(len(methods)), loop_time, 0.55, color=colors,
                edgecolor='none', linewidth=0, zorder=3)
ax1.set_ylabel('回路时间 (ms)', fontproperties=zhfont)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=9)
ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, val in zip(bars1, loop_time):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.set_title('(a) MPC回路时间', fontproperties=zhfont, fontsize=10, pad=8)

# (b) Control frequency
bars2 = ax2.bar(range(len(methods)), freq, 0.55, color=colors,
                edgecolor='none', linewidth=0, zorder=3)
ax2.set_ylabel('控制频率 (Hz)', fontproperties=zhfont)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=9)
ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=10, color='#CC3333', linestyle='--', alpha=0.6, linewidth=1, zorder=2)
ax2.text(len(methods)-0.5, 10.5, '典型控制需求 10Hz', fontproperties=zhfont_s,
         fontsize=7.5, color='#CC3333', ha='right')
for bar, val in zip(bars2, freq):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.set_title('(b) MPC控制频率', fontproperties=zhfont, fontsize=10, pad=8)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/mpc_comparison.pdf', dpi=300, bbox_inches='tight')
print("Done: mpc_comparison.pdf")
