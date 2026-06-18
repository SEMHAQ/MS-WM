# S4D-WM: Lightweight State Space World Model for Joint State Prediction in Embodied Intelligence

**面向具身智能关节状态预测的轻量级状态空间世界模型**

[![Paper](https://img.shields.io/badge/Paper-CTA%202026-blue)](https://github.com/SEMHAQ/SSM-World-Model)
[![Code](https://img.shields.io/badge/Code-Python-green)](https://github.com/SEMHAQ/SSM-World-Model)
[![Dataset](https://img.shields.io/badge/Dataset-D4RL-orange)](https://drive.google.com/drive/folders/16TBWD3CeWEmL3Al1M9goMzYN8TDJgKyi?usp=drive_link)

## Overview

S4D-WM is a lightweight world model based on diagonal state space models (S4D-style parameterization) with Mamba-style gated blocks for joint state prediction in embodied intelligence. It achieves **O(T log T) training complexity** and **O(1) single-step inference latency**, with only **0.14--0.23M parameters** across datasets.

## Key Results

### D4RL Humanoid Dataset (348D, T=32)

| Model | MSE (×10⁻²) | R² | Inference (ms) | Params (M) |
|-------|-------------|-----|----------------|------------|
| LSTM-WM | 36.99±0.82 | 0.537±0.010 | 2.9 | 0.64 |
| Trans.-WM | 28.80±0.93 | 0.640±0.012 | 2.9 | 0.15 |
| GRU-WM | 32.14±0.52 | 0.598±0.006 | 2.4 | 0.50 |
| Mamba-WM | 25.84±0.60 | 0.677±0.008 | 9.5 | 0.66 |
| **S4D-WM** | **24.79±0.40** | **0.690±0.005** | 8.3 | 0.23 |

- **33% better MSE than LSTM-WM, 14% better than Trans.-WM, 4% better than Mamba-WM**
- **Smallest standard deviation (0.40) — most stable predictions**
- **0.23M parameters** (36% of LSTM-WM, 35% of Mamba-WM)

### D4RL Ant Dataset (105D, T=32)

| Model | MSE (×10⁻²) | R² | Inference (ms) | Params (M) |
|-------|-------------|-----|----------------|------------|
| LSTM-WM | 112.34±0.90 | 0.037±0.007 | 1.3 | 0.57 |
| Trans.-WM | **72.55±0.78** | **0.398±0.007** | 2.8 | 0.12 |
| GRU-WM | 97.70±0.61 | 0.161±0.005 | 1.1 | 0.44 |
| Mamba-WM | 75.07±0.70 | 0.377±0.006 | 4.7 | 0.59 |
| **S4D-WM** | 73.20±0.76 | 0.389±0.006 | 4.8 | 0.16 |

- **S4D-WM and Trans.-WM are close** (0.9% gap), both significantly outperform LSTM/GRU
- **0.16M parameters** (28% of LSTM-WM)

### D4RL Walker2d Dataset (17D, T=32)

| Model | MSE (×10⁻²) | R² | Inference (ms) | Params (M) |
|-------|-------------|-----|----------------|------------|
| LSTM-WM | 3.49±0.17 | 0.963±0.002 | 1.3 | 0.55 |
| Trans.-WM | **2.76±0.02** | **0.971±0.001** | 2.7 | 0.11 |
| GRU-WM | 3.39±0.07 | 0.964±0.001 | 1.1 | 0.42 |
| Mamba-WM | 3.51±0.03 | 0.963±0.001 | 4.5 | 0.57 |
| **S4D-WM** | 3.04±0.07 | 0.967±0.001 | 4.7 | 0.14 |

- **Trans.-WM best on this simple 17D task**, S4D-WM competitive (10.1% gap)
- **S4D-WM excels on high-dimensional complex dynamics (Humanoid), competitive on mid/low-dimensional tasks**

### Sequence Length Sensitivity (S4D-WM)

| T | Humanoid MSE | Humanoid R² | Ant MSE | Ant R² |
|---|--------------|-------------|---------|--------|
| 16 | **0.291** | **0.656** | 0.542 | **0.302** |
| 32 | 0.442 | 0.479 | 0.728 | 0.150 |
| 64 | 0.612 | 0.153 | 0.942 | -0.019 |
| 128 | 1.213 | -0.623 | 0.934 | 0.139 |
| 256 | 2.146 | -1.694 | **0.480** | 0.131 |

**Key Finding**: Optimal sequence length depends on dataset dimensionality and dynamics complexity:
- **Humanoid (348D)**: T=16 best — high-dimensional dynamics benefit from short sequences
- **Ant (105D)**: T=256 best — medium-dimensional dynamics benefit from longer context
- **Walker2d (17D)**: R² consistently >0.92 across all T — simple dynamics, length has limited impact

## Dataset

The D4RL datasets used in this paper are available at:

**[Google Drive: D4RL Datasets](https://drive.google.com/drive/folders/16TBWD3CeWEmL3Al1M9goMzYN8TDJgKyi?usp=drive_link)**

Contents:
- `humanoid/` — D4RL Humanoid-medium (348D, 1163 episodes)
- `ant/` — D4RL Ant-medium (105D, 1047 episodes)
- `walker2d/` — D4RL Walker2d-medium (17D, 835 train / 209 val episodes)

## Project Structure

```
SSM-World-Model/
├── src/
│   ├── models/
│   │   ├── ssm_world_model.py     # S4D-WM architecture
│   │   ├── baselines.py           # LSTM-WM, Trans.-WM, GRU-WM, Mamba-WM
│   │   └── mpc_controller.py      # Model Predictive Control integration
│   └── train/
│       └── train.py               # Training loop with multi-seed support
├── scripts/
│   ├── run_all_d4rl_experiments.py  # Full experiment pipeline
│   ├── multiseed_train.py           # Multi-seed training (3 seeds: 42, 123, 456)
│   ├── generate_figures_nature.py   # Generate paper figures (Nature style)
│   ├── gen_seqlen_figures.py        # Sequence length sensitivity figures
│   ├── gen_training_curves.py       # Training curves
│   ├── gen_mpc.py                   # MPC comparison figures
│   └── gen_radar.py                 # Radar comparison figures
├── experiments/
│   ├── d4rl_all_experiments.json    # All experiment results + training logs
│   ├── multiseed_results.json       # Multi-seed results (mean±std)
│   ├── seqlen_results_final.json    # Sequence length results
│   └── mpc_results.json             # MPC comparison results
├── data/
│   ├── humanoid/                    # D4RL Humanoid-medium (348D)
│   ├── ant/                         # D4RL Ant-medium (105D)
│   └── walker2d/                    # D4RL Walker2d-medium (17D)
└── paper/
    ├── main.tex                     # Paper source (LaTeX, CTA template)
    ├── main.pdf                     # Compiled PDF (11 pages)
    └── figures/                     # Paper figures (7 figures)
```

## Quick Start

### Requirements

```bash
pip install torch mamba-ssm numpy matplotlib tqdm
```

### Download Dataset

```bash
# Download from Google Drive link above
# Place in data/ directory
```

### Train Models

```bash
# Full experiment pipeline (all datasets, all models, 3 seeds)
python scripts/run_all_d4rl_experiments.py

# Multi-seed training
python scripts/multiseed_train.py
```

### Generate Figures

```bash
python scripts/generate_figures_nature.py
python scripts/gen_seqlen_figures.py
python scripts/gen_training_curves.py
python scripts/gen_mpc.py
python scripts/gen_radar.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2026s4dwm,
  title   = {面向具身智能关节状态预测的轻量级状态空间世界模型},
  author  = {周新民 and 余焕杰 and 张慧慧 and 王伟 and 陈露},
  journal = {控制理论与应用},
  year    = {2026}
}
```

## Acknowledgments

This work was supported by:
- National Social Science Fund of China (Grant No. 21BGL231)
- Major Program of Xiangjiang Laboratory (Grant No. 24XJ01001; 25XJ01001)

## License

This project is for academic research purposes.

## Contact

- 周新民: zhouxinmin2699@163.com
- 余焕杰: semhaqx@gmail.com
- 张慧慧: huihuiz054@gmail.com
