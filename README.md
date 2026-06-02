# SSM-World-Model

基于状态空间模型(SSM)的具身智能世界模型，面向人形机器人状态预测与模型预测控制。

目标期刊：《控制理论与应用》(CTA) - "具身智能与人形机器人"专刊
截稿日期：2026年9月30日
刊登时间：2026年12月

## 项目结构

```
SSM-World-Model/
├── src/
│   ├── models/        # SSM世界模型核心架构
│   ├── data/          # 数据集加载与预处理
│   ├── utils/         # 工具函数
│   └── train/         # 训练脚本
├── paper/
│   ├── sections/      # LaTeX各章节
│   └── figures/       # 论文图表
├── scripts/           # 运行脚本
├── configs/           # 配置文件
├── experiments/       # 实验结果
└── notebooks/         # Jupyter实验笔记本
```

## 技术路线

1. 基于SSM/Mamba构建轻量级世界模型
2. 预测机器人状态转移
3. 结合MPC做模型预测控制
4. 公开数据集验证（Open X-Embodiment, DROID等）

## 环境

- Python 3.10+
- PyTorch 2.0+
