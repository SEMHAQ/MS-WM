# Peer Review Report - Reviewer 1 (Methodology)

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Review Date**: 2026-06-02
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
深度学习与序列建模方法学专家, 专注于SSM/Transformer架构对比研究.

### Review Focus
研究设计严谨性、实验方法合理性、统计有效性、可复现性.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

### Confidence Score
5/5

### Summary Assessment
本文提出SSM-WM用于机器人状态预测, 实验结果显示推理速度优势(7.3x)但精度落后LSTM 55%. 从方法学角度看, 存在以下关键问题: (1) 合成数据的动力学设计过于简单, 不足以验证SSM的长序列优势; (2) 论文描述的Mamba架构与实际DiagSSM实现不一致; (3) 缺少统计显著性检验和多次运行的方差报告; (4) 消融实验无数据. 这些问题严重影响论文的方法学严谨性.

---

## Strengths

### S1: 速度-精度权衡的实验设计
论文同时报告MSE、参数量和推理时间, 为读者提供了全面的方法对比视角.

### S2: 残差预测设计
解码器采用残差连接(预测变化量而非绝对状态), 这是状态预测领域的良好实践, 有助于加速收敛.

### S3: 门控SSM块设计
SSMBlock中的门控机制(g * ssm_out + (1-g) * x_norm)是合理的架构选择, 有助于信息流控制.

---

## Weaknesses

### W1: 论文描述与代码实现不一致
**Problem**: 论文3.1节描述"多层Mamba块"和"选择性SSM机制", 但实际代码(src/models/ssm_world_model.py)使用的是DiagSSM(对角SSM + FFT卷积), 完全不同的架构.
**Why it matters**: 论文描述的架构与实现不一致是严重的学术诚信问题. 审稿人无法验证论文声称的贡献.
**Suggestion**: 必须统一论文描述和实际实现. 如果使用DiagSSM, 则需要更新论文中的方法描述.
**Severity**: Critical

### W2: 缺少统计显著性检验
**Problem**: 表1仅报告单次运行结果, 未报告多次运行的均值和标准差.
**Why it matters**: 深度学习实验具有随机性, 单次运行结果可能不具代表性.
**Suggestion**: 至少运行3次, 报告均值±标准差, 并进行t检验或Wilcoxon检验.
**Severity**: Major

### W3: 合成数据设计过于简单
**Problem**: 合成数据使用线性动力学+tanh耦合, 状态维度仅28, 序列长度64. 这不足以验证SSM在长序列上的优势.
**Why it matters**: SSM的核心优势是长序列建模, 但实验中的序列长度太短, 无法充分体现这一优势.
**Suggestion**: 增加序列长度(128/256/512)的对比实验, 展示不同序列长度下各方法的速度和精度变化趋势.
**Severity**: Major

### W4: Transformer基线数据缺失
**Problem**: 表1中Transformer-WM的MSE和MAE列填写"---", 推理时间写">100ms".
**Why it matters**: 缺少关键基线数据, 无法完整评估SSM-WM的相对优势.
**Suggestion**: 补充Transformer在T=64下的完整数据. 如果因OOM无法运行, 需要明确说明并讨论.
**Severity**: Major

### W5: 缺少序列长度敏感性分析
**Problem**: 仅在T=64下进行实验, 未展示不同序列长度下各方法的性能变化.
**Why it matters**: SSM的优势随序列长度增加而增大, 缺少这一分析无法证明SSM的长序列优势.
**Suggestion**: 在T=16/32/64/128/256下对比各方法的MSE和推理时间, 绘制趋势图.
**Severity**: Major

---

## Detailed Comments

### Methodology
- 训练设置合理(AdamW, 余弦退火, 梯度裁剪).
- 缺少学习率搜索和超参数敏感性分析.
- batch_size=32偏小, 可能影响训练稳定性.

### Results
- 表1格式规范, 但数据不完整.
- 缺少训练曲线图(loss vs epoch).
- 缺少预测结果的可视化(如预测轨迹 vs 真实轨迹).

### Reproducibility
- 代码已上传GitHub(SEMHAQ/SSM-World-Model), 可复现性良好.
- 但论文描述与代码不一致, 影响可复现性.

---

## Questions for Authors

1. 论文描述的Mamba架构与实际DiagSSM实现不一致, 请解释并统一.
2. 为什么Transformer在T=64下无法获得MSE数据? 是OOM还是其他原因?
3. 是否可以补充不同序列长度(16/32/64/128)的对比实验?

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor |
|-----------|--------------|------------|
| Originality | 58 | Weak |
| Methodological Rigor | 45 | Weak |
| Evidence Sufficiency | 42 | Weak |
| Argument Coherence | 62 | Adequate |
| Writing Quality | 60 | Adequate |
| **Weighted Average** | **51** | **Major Revision** |
