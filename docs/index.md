---
title: SSM-World-Model
---

# 面向人形机器人状态预测的轻量级状态空间世界模型

**周新民 &nbsp; 余焕杰** — 湖南工商大学 / 湘江实验室

*控制理论与应用 (Control Theory & Applications)*, 2026 专刊: 具身智能与人形机器人

[:fontawesome-brands-github: GitHub](https://github.com/SEMHAQ/SSM-World-Model){ .md-button .md-button--primary }

---

## 研究动机

人形机器人状态预测需要**高精度**和**低延迟**的平衡:

| 方法 | 问题 |
|------|------|
| LSTM | 推理慢 (27.8ms), 参数多 (0.29M) |
| Transformer | O(T²)复杂度, 推理 >100ms |
| Mamba | 需要自定义CUDA算子, 部署门槛高 |

**SSM-WM** 用标准PyTorch操作实现了 7倍加速 + 竞争力精度 + 最少参数量。

## 核心贡献

1. **轻量级SSM世界模型** — S4D对角参数化 + Mamba门控块结构
2. **O(T log T)训练, O(1)推理** — FFT卷积 + 递推模式双模计算
3. **MPC集成** — 合成数据集5.1Hz, MuJoCo 2.1Hz控制频率
4. **门控阈值实验** — 软阈值 > Garrote > 硬阈值

## 一句话总结

> SSM-WM在MuJoCo Humanoid上优于LSTM 6%、优于Transformer 13%，与Mamba持平但实现更简单，推理快16%，参数少14%。
