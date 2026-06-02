# Peer Review Report - EIC

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Review Date**: 2026-06-02
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Editor-in-Chief (EIC), 《控制理论与应用》

### Reviewer Identity
控制理论与应用领域资深编辑, 专注于智能控制、机器人学和机器学习的交叉研究.

### Review Focus
期刊适配性、原创性、整体质量和读者价值.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [x] **Minor Revision**
- [ ] Major Revision
- [ ] Reject

### Confidence Score
4/5

### Summary Assessment
本文提出了一种基于状态空间模型(SSM)的轻量级世界模型(SSM-WM), 用于人形机器人的状态预测, 并将其嵌入MPC框架实现在线控制. 论文选题紧扣"具身智能与人形机器人"专刊方向, 具有较好的时效性. SSM在推理速度上的优势(7.3x加速)是本文的核心卖点. 然而, 论文存在若干需要改进的问题: (1) 摘要中声称使用Open X-Embodiment数据集, 但正文实验使用的是合成数据, 存在不一致; (2) 消融实验部分仅列出实验设置而无具体数据; (3) 缺少架构图和实验可视化. 这些问题不影响论文的核心贡献, 但需要修正以确保学术严谨性.

---

## Strengths

### S1: 选题契合专刊方向
论文聚焦于具身智能领域的世界模型构建, 与"具身智能与人形机器人"专刊高度契合. SSM作为新兴序列建模方法, 在机器人控制领域的应用具有前瞻性.

### S2: 推理速度优势显著
实验表明SSM-WM的推理时间仅为3.8ms, 相比LSTM的27.8ms加速7.3倍, 满足实时控制的<10ms要求. 这一优势在实际部署中具有重要价值.

### S3: 参数效率高
SSM-WM仅需0.24M参数, 比LSTM少17%, 比Transformer少61%, 适合资源受限的边缘部署场景.

---

## Weaknesses

### W1: 摘要与正文数据不一致
**Problem**: 摘要第3-4行声称"在Open X-Embodiment公开数据集上的实验结果表明", 但正文4.1节明确使用"合成的机器人状态-动作数据集".
**Why it matters**: 这是学术不端的红线, 审稿人和读者会质疑论文的可信度.
**Suggestion**: 必须统一. 要么下载并使用真实的Open X-Embodiment数据集, 要么修改摘要为"合成数据集".
**Severity**: Critical

### W2: 消融实验缺少数据支撑
**Problem**: 第4.3节"消融实验"仅列出3个消融设置, 结论仅一句"实验结果表明, 各组件均对最终性能有正向贡献", 无任何数据表格或图表.
**Why it matters**: 消融实验是验证方法有效性的关键证据, 缺少数据等于没有消融.
**Suggestion**: 补充完整的消融实验表格, 包含每个消融设置的MSE/MAE数值.
**Severity**: Major

### W3: 缺少架构图
**Problem**: 第3.1节提到"整体架构如图~1所示", 但全文没有图1.
**Why it matters**: 架构图是理解模型设计的关键, 缺失严重影响可读性.
**Suggestion**: 补充SSM-WM的架构图, 清晰展示编码器、SSM主干、解码器的数据流.
**Severity**: Major

---

## Detailed Comments

### Title & Abstract
- 标题准确, 但摘要中Open X-Embodiment的声称需要修正.
- 英文摘要中"3--5x faster inference"与实际数据(7.3x)不一致, 需要更新.

### Introduction
- 引言结构清晰, 研究动机充分.
- 贡献点3声称"在Open X-Embodiment公开数据集上验证", 需要与实际数据一致.

### Methodology
- SSM的数学描述清晰, 但实际实现(对角SSM + FFT卷积)与论文描述(Mamba块 + 选择性SSM)存在差异, 需要统一.

### Results
- 表1数据清晰, 但缺少图形化展示(如推理时间对比柱状图).
- Transformer的"---"和">100"需要补充实际数据或说明原因.

### References
- 仅8篇参考文献偏少, 建议补充到15-20篇, 特别是SSM在机器人领域的近期应用.

---

## Questions for Authors

1. 摘要中声称使用Open X-Embodiment数据集, 但正文使用合成数据, 请解释原因并统一.
2. 论文描述的架构(Mamba块)与实际代码实现(对角SSM)是否一致? 请明确说明.
3. Transformer基线在T=64时的MSE是多少? ">100ms"的推理时间是否因OOM导致无法测量?

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor |
|-----------|--------------|------------|
| Originality | 62 | Adequate |
| Methodological Rigor | 55 | Weak |
| Evidence Sufficiency | 50 | Weak |
| Argument Coherence | 68 | Adequate |
| Writing Quality | 65 | Adequate |
| **Weighted Average** | **58** | **Major Revision** |
