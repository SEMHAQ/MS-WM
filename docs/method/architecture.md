---
title: SSM-WM架构
---

# SSM-WM 架构

## 整体结构

SSM-WM由三部分组成: **状态-动作编码器** → **SSM主干网络** → **状态解码器**

## 状态-动作编码器

将状态 $\bm{s}_t$ 和动作 $\bm{a}_t$ 拼接后通过两层全连接网络投影到隐空间:

$$
\bm{z}_t' = \bm{W}_1[\bm{s}_t; \bm{a}_t] + \bm{b}_1
$$

$$
\bm{z}_t = \bm{W}_2\text{GELU}(\bm{z}_t') + \bm{b}_2
$$

其中 $\bm{W}_1 \in \mathbb{R}^{D \times (d_s + d_a)}$，$\bm{W}_2 \in \mathbb{R}^{D \times D}$ 为可学习参数，$D$ 为隐空间维度。

高斯误差线性单元 (Gaussian error linear unit, GELU) 激活函数相比ReLU具有更平滑的梯度特性，有利于训练稳定性。

## SSM主干网络

采用 $L$ 层SSM块对编码后的序列进行建模。每个SSM块包含:

1. **LayerNorm** — 归一化
2. **对角SSM** — S4D风格，$\bm{A} = \text{diag}(\bm{a})$，通过FFT计算因果卷积
3. **门控机制** — Mamba风格，$\bm{g}_t = \sigma(\bm{W}_g\bm{z}_t + \bm{b}_g)$
4. **残差连接** — $\bm{z}_t^{(l)} = \bm{z}_t^{(l-1)} + \bm{g}_t \odot \text{SSM}(\bm{z}_t^{(l-1)})$

## 状态解码器

$$
\hat{\bm{s}}_{t+1} = \bm{W}_d\bm{z}_t + \bm{b}_d
$$

## 训练目标

$$
\mathcal{L} = (1-\lambda)\mathcal{L}_{\text{single}} + \lambda\mathcal{L}_{\text{multi}}
$$

- 单步损失: $\mathcal{L}_{\text{single}} = \|\hat{\bm{s}}_{t+1} - \bm{s}_{t+1}\|^2$
- 多步损失: 自回归展开 $H$ 步的累积误差
- 默认: $\lambda=0.5$，$H=8$

## 计算复杂度

| 组件 | 复杂度 |
|------|--------|
| LayerNorm | O(TD) |
| 对角SSM (FFT) | O(TD log T) |
| 门控 | O(TD²) |
| **总计** | **O(TD log T + D²L)** |
