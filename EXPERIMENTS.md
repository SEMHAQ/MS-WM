# FSM-WM 实验指南

## 环境要求

- Python 3.8+
- PyTorch 2.0+ (CUDA版本)
- numpy, einops, pyyaml, matplotlib, tqdm

## 安装依赖

```bash
pip install -r requirements.txt
```

如果需要CUDA支持：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 运行实验

### 1. 运行所有实验（推荐）

```bash
python scripts/run_all_experiments.py
```

预计耗时：4-8小时（取决于GPU）

### 2. 运行单个实验

```bash
# 只运行主实验（6模型 × 3数据集 × 5seed）
python scripts/run_all_experiments.py --experiment main

# 只运行序列长度分析
python scripts/run_all_experiments.py --experiment seqlen

# 只运行消融实验
python scripts/run_all_experiments.py --experiment ablation
```

### 3. 快速测试

```bash
# 只用1个seed，快速验证代码是否正常
python scripts/run_all_experiments.py --quick
```

## 实验配置

所有模型使用相同配置：

| 参数 | 值 |
|------|-----|
| 序列长度 T | 32 |
| 批大小 B | 64 |
| 训练轮数 | 100 |
| 学习率 | 5e-4 |
| 优化器 | AdamW |
| 学习率调度 | Cosine Annealing |
| 随机种子 | 42, 123, 456, 789, 1024 |
| 隐层维度 D | 128 |
| 状态维度 N | 16 |
| 层数 L | 4 |

## 数据集

- Humanoid (348维状态, 17维动作)
- Ant (105维状态, 8维动作)
- Hopper (11维状态, 6维动作)

## 结果保存

所有结果保存在 `experiments/paper/` 目录：

```
experiments/paper/
├── main_results.json      # 主实验结果
├── seqlen_results.json    # 序列长度结果
├── ablation_results.json  # 消融实验结果
├── checkpoints/           # 模型权重
└── logs/                  # 训练日志
```

## 更新论文

实验完成后，运行以下命令更新论文表格：

```bash
python scripts/update_paper_tables.py
```

## 注意事项

1. 确保GPU显存足够（建议8GB以上）
2. 实验过程中不要中断，否则需要重新运行
3. 如果显存不足，可以减小批大小（修改脚本中的BATCH_SIZE）
4. 所有模型使用相同的超参数配置，确保公平比较
