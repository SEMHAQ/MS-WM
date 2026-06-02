# Peer Review Report - Reviewer 2 (Domain)

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Review Date**: 2026-06-02
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
具身智能与机器人控制领域专家, 专注于世界模型和模型预测控制研究.

### Review Focus
文献覆盖度、理论框架适配性、领域贡献.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

### Confidence Score
4/5

### Summary Assessment
本文将SSM应用于人形机器人世界模型构建, 选题具有前瞻性. 然而, 从领域角度看, 存在以下问题: (1) 文献综述不够全面, 缺少SSM在机器人领域的近期工作; (2) MPC集成仅停留在理论层面, 缺少实验验证; (3) 合成数据无法验证方法在真实机器人场景下的有效性. 论文的核心贡献(SSM用于机器人世界模型)有价值, 但需要更充分的实验支撑.

---

## Strengths

### S1: SSM在机器人领域的应用探索
将SSM应用于机器人世界模型是一个有价值的研究方向, 目前该领域研究较少, 具有一定的原创性.

### S2: MPC集成的理论框架
将SSM-WM嵌入MPC框架的理论描述清晰, 优化问题的数学建模合理.

### S3: 实时性优势的论证
3.8ms的推理时间满足实时控制要求, 这在实际机器人部署中具有重要价值.

---

## Weaknesses

### W1: 文献覆盖不足
**Problem**: 仅引用8篇文献, 缺少SSM在机器人领域的近期应用(如S4WM, Mamba for control等).
**Why it matters**: 无法准确定位本文在现有研究中的位置.
**Suggestion**: 补充15-20篇文献, 包括: (1) SSM在时序预测中的应用; (2) 世界模型在机器人控制中的应用; (3) 轻量级序列建模方法.
**Severity**: Major

### W2: MPC集成缺少实验验证
**Problem**: 第3.3节描述了MPC框架, 但实验部分完全没有MPC相关的实验结果.
**Why it matters**: MPC是论文的贡献点之一, 缺少实验验证等于该贡献未被证明.
**Suggestion**: 补充MPC控制实验, 如轨迹跟踪任务、避障任务等, 展示SSM-WM在闭环控制中的表现.
**Severity**: Major

### W3: 合成数据的局限性
**Problem**: 使用合成数据进行实验, 无法验证方法在真实机器人场景下的有效性.
**Why it matters**: 人形机器人控制涉及复杂的物理交互, 合成数据可能无法捕捉真实世界的复杂性.
**Suggestion**: 至少在一个公开的真实机器人数据集上进行验证(如DROID, RoboSet等).
**Severity**: Major

### W4: 缺少与SSM变体的对比
**Problem**: 仅与LSTM和Transformer对比, 缺少与S4、Mamba、S5等SSM变体的对比.
**Why it matters**: 无法判断所提DiagSSM相对于其他SSM变体的优势.
**Suggestion**: 补充与S4、Mamba等SSM变体的对比实验.
**Severity**: Minor

---

## Detailed Comments

### Literature Review
- 引言部分对RNN和Transformer的局限性描述准确.
- 但缺少SSM在机器人领域的近期工作, 如:
  - S4WM (ICML 2024)
  - Mamba for continuous control
  - World Models for Robot Learning (survey)

### Methodology
- MPC框架的数学建模合理, 但缺少实际实现细节.
- 建议补充MPC的求解方法(如iLMP、CEM等).

### References
- 仅8篇, 需要大幅扩充.
- 建议按类别组织: SSM基础(4-5篇)、世界模型(3-4篇)、机器人控制(3-4篇)、具身智能(2-3篇).

---

## Questions for Authors

1. MPC集成部分是否有实验结果? 如果有, 请补充; 如果没有, 建议删除该贡献点或说明是未来工作.
2. 是否考虑在真实机器人数据集上验证?
3. DiagSSM与S4、Mamba的性能对比如何?

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor |
|-----------|--------------|------------|
| Originality | 65 | Adequate |
| Methodological Rigor | 52 | Weak |
| Evidence Sufficiency | 48 | Weak |
| Argument Coherence | 64 | Adequate |
| Writing Quality | 62 | Adequate |
| Literature Integration | 40 | Weak |
| **Weighted Average** | **54** | **Major Revision** |
