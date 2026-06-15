"""Regenerate MPC comparison figure with S4D-WM naming."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

methods = ['LSTM-MPC', 'Mamba-MPC', 'S4D-WM-MPC']
mse = [0.0032, 0.0041, 0.0043]
freq = [0.7, 4.3, 5.1]
colors = ['#d62728', '#ff7f0e', '#1f77b4']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

# (a) Tracking MSE
bars1 = ax1.bar(methods, mse, color=colors, alpha=0.85, edgecolor='white', width=0.5)
ax1.set_ylabel('跟踪 MSE', fontproperties=zhfont)
ax1.set_title('(a) 轨迹跟踪精度', fontproperties=zhfont, fontsize=10, pad=8)
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, mse):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# (b) Control Frequency
bars2 = ax2.bar(methods, freq, color=colors, alpha=0.85, edgecolor='white', width=0.5)
ax2.set_ylabel('控制频率 (Hz)', fontproperties=zhfont)
ax2.set_title('(b) 控制频率', fontproperties=zhfont, fontsize=10, pad=8)
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, freq):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/mpc_comparison.pdf', dpi=300, bbox_inches='tight')
print("Done: mpc_comparison.pdf")
