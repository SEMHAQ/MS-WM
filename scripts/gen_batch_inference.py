"""Regenerate batch inference figure with correct naming."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# Data from Table 2 (system-level, dataset-independent)
batch_sizes = [1, 8, 32, 64]
lstm_times = [2.1, 4.5, 12.3, 27.8]
mamba_times = [1.2, 1.8, 2.8, 4.5]
s4d_times = [0.9, 1.5, 2.4, 3.8]

x = np.arange(len(batch_sizes))
width = 0.25

fig, ax = plt.subplots(figsize=(5.5, 3.5))
bars1 = ax.bar(x - width, lstm_times, width, label='LSTM-WM', color='#d62728', alpha=0.85)
bars2 = ax.bar(x, mamba_times, width, label='Mamba-WM', color='#ff7f0e', alpha=0.85)
bars3 = ax.bar(x + width, s4d_times, width, label='S4D-WM', color='#1f77b4', alpha=0.85)

from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
ax.set_xlabel('批大小 B', fontproperties=zhfont)
ax.set_ylabel('推理时间 (ms)', fontproperties=zhfont)
ax.set_xticks(x)
ax.set_xticklabels(['1', '8', '32', '64'])
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3, axis='y')

# Add 10ms threshold line
ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax.text(3.3, 10.5, '10ms', fontsize=7, color='red', alpha=0.7)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/batch_inference.pdf', dpi=300, bbox_inches='tight')
print("Done: batch_inference.pdf")
