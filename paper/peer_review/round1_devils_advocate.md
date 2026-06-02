# Devil's Advocate Report

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Review Date**: 2026-06-02
- **Review Round**: Round 1

---

## Reviewer Role
Devil's Advocate

---

## Strongest Counter-Argument (200-300 words)

本文的核心论点是"SSM可以替代LSTM/Transformer用于机器人世界模型, 因为它更快且精度相当". 但这一论点存在根本性缺陷:

首先, 精度差距并不"相当". SSM-WM的MSE(0.001317)比LSTM(0.000852)高55%, 这不是"同一数量级"的差距, 而是实质性差距. 在控制任务中, 预测误差会被累积放大, 55%的初始误差差距可能导致控制性能的显著下降.

其次, 推理速度的优势被夸大. 3.8ms vs 27.8ms的对比是在GPU上测量的. 在实际部署中, LSTM的推理可以通过ONNX Runtime、TensorRT等优化工具大幅加速, 差距可能缩小到2-3倍而非7倍. 同时, SSM的FFT卷积在CPU上可能并不比LSTM的矩阵运算快.

第三, 论文的实验设计存在选择性报告嫌疑. 仅在T=64下报告结果, 而未展示T=16/32下的对比(在这些长度下, LSTM可能更快). 这种选择性报告削弱了论文的可信度.

最后, 合成数据的使用严重削弱了论文的说服力. 在真实机器人场景中, 状态转移函数远比合成数据复杂, SSM的线性假设可能不成立.

---

## Issue List

### CRITICAL Issues

**C1: 摘要与正文数据不一致**
- **Dimension**: 学术诚信
- **Location**: 摘要第3-4行 vs 正文4.1节
- **Description**: 摘要声称使用Open X-Embodiment, 正文使用合成数据

**C2: 论文描述与代码实现不一致**
- **Dimension**: 方法学
- **Location**: 论文3.1节 vs src/models/ssm_world_model.py
- **Description**: 论文描述Mamba块, 代码实现DiagSSM

### MAJOR Issues

**M1: 选择性报告**
- **Dimension**: 实验设计
- **Location**: 表1
- **Description**: 仅在T=64下报告, 未展示其他序列长度

**M2: 精度差距被淡化**
- **Dimension**: 结论
- **Location**: 实验结果分析
- **Description**: 55%的MSE差距被描述为"同一数量级, 可接受"

**M3: 缺少消融实验数据**
- **Dimension**: 方法学
- **Location**: 4.3节
- **Description**: 消融实验无数据支撑

### MINOR Issues

**m1: 参考文献不足**
- **Dimension**: 文献
- **Location**: 参考文献
- **Description**: 仅8篇, 建议补充到15-20篇

---

## Ignored Alternative Explanations

1. **LSTM的门控机制可能更适合机器人状态预测**: LSTM的遗忘门可以选择性地丢弃不相关的历史信息, 而SSM的线性递推可能保留过多噪声.
2. **SSM的速度优势可能在CPU上消失**: FFT卷积在CPU上可能不如LSTM的矩阵运算高效.
3. **Transformer的OOM可能是实现问题**: 论文未说明Transformer是否尝试过gradient checkpointing或flash attention等优化.

---

## Missing Stakeholder Perspectives

1. **控制工程师**: 55%的MSE差距在实际控制任务中意味着什么?
2. **嵌入式开发者**: 在资源受限设备上的实际部署体验如何?
3. **机器人操作员**: 模型的鲁棒性和安全性如何保证?

---

## Observations (Non-Defects)

1. 残差预测设计(s_{t+1} = s_t + delta_s)是良好的实践.
2. 门控SSM块的设计合理, 有助于信息流控制.
3. GitHub代码公开有利于可复现性.
