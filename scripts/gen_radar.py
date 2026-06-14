"""Regenerate radar comparison figure with D4RL Humanoid data (Table 8)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# D4RL Humanoid data from Table 8
models = ['S4D-WM', 'Mamba-WM', 'Trans.-WM', 'LSTM-WM']
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
markers = ['o', 's', '^', 'D']

# 5 dimensions: R², 1/MSE, speed(ms), 1/params, speed/params ratio
raw = {
    #         R²      1/MSE   speed   1/params
    'S4D-WM':    [0.694,  1/0.245, 3.4,  1/0.23],
    'Mamba-WM':  [0.676,  1/0.259, 3.5,  1/0.66],
    'Trans.-WM': [0.653,  1/0.278, 1.6,  1/0.15],
    'LSTM-WM':   [0.541,  1/0.367, 2.5,  1/0.64],
}

categories = ['R²', 'MSE⁻¹', 'Speed\n(↑ better)', 'Params⁻¹\n(M⁻¹)']
N = len(categories)

def normalize(col_idx, higher_better=True):
    """Normalize one metric across all models to [0.15, 1.0]."""
    vals = np.array([raw[m][col_idx] for m in models])
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-10:
        return np.full(4, 0.5)
    normed = 0.15 + 0.85 * (vals - mn) / (mx - mn)
    return normed

# For speed: higher = better (longer bar = faster), but lower ms = better
# Actually raw speed is in ms, so LOWER is better. Use inverse.
# Let's rethink: R² high=good, 1/MSE high=good, speed low_ms=good, 1/params high=good
# For radar, "outer = better" convention
norm_r2 = normalize(0, True)          # R² higher = better
norm_mse = normalize(1, True)         # 1/MSE higher = better
norm_speed = normalize(2, False)      # ms lower = better → invert
norm_params = normalize(3, True)      # 1/params higher = better

all_norms = [norm_r2, norm_mse, norm_speed, norm_params]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(4.2, 3.8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

for i, model in enumerate(models):
    values = [all_norms[j][i] for j in range(N)]
    values += values[:1]
    ax.plot(angles, values, '-', color=colors[i], linewidth=1.6, markersize=0)
    ax.fill(angles, values, color=colors[i], alpha=0.10)
    # Add markers at data points
    for j in range(N):
        ax.plot(angles[j], values[j], marker=markers[i], color=colors[i],
                markersize=5, markeredgecolor='white', markeredgewidth=0.5)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8.5, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['', '', '', ''], fontsize=7, color='gray')
ax.grid(True, linewidth=0.4, alpha=0.5)

legend = ax.legend(models, loc='upper right', bbox_to_anchor=(1.38, 1.15), fontsize=8,
                   frameon=True, fancybox=True, framealpha=0.9, edgecolor='gray',
                   handletextpad=0.4, handlelength=1.5)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/radar_comparison.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('paper/figures/radar_comparison.eps', bbox_inches='tight', pad_inches=0.05)
print("Done: radar_comparison.pdf/.eps")
