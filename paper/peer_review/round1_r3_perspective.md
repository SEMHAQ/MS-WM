# Peer Review Report - Reviewer 3 (Perspective)

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Review Date**: 2026-06-02
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
边缘计算与实时系统领域专家, 关注轻量级AI模型在嵌入式系统中的部署.

### Review Focus
跨学科视角、实际应用价值、技术可迁移性.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [x] **Minor Revision**
- [ ] Major Revision
- [ ] Reject

### Confidence Score
3/5

### Summary Assessment
本文从边缘部署的角度来看, SSM-WM的3.8ms推理时间和0.24M参数量确实具有吸引力. 然而, 论文缺少对实际部署场景的深入讨论, 如内存占用、功耗、不同硬件平台的适配性等. 此外, 论文的核心卖点(推理速度)需要更全面的分析, 包括不同batch size下的延迟、不同硬件平台(GPU/CPU/NPU)的对比等.

---

## Strengths

### S1: 实时性优势明确
3.8ms的推理时间在机器人控制领域具有明确的应用价值, 特别是对于需要高频控制(>100Hz)的场景.

### S2: 参数效率高
0.24M参数量使得模型可以在资源受限的边缘设备上部署, 如NVIDIA Jetson、树莓派等.

### S3: 选题具有产业价值
轻量级世界模型在工业机器人、服务机器人等领域有广泛的应用前景.

---

## Weaknesses

### W1: 缺少实际部署分析
**Problem**: 论文仅报告GPU上的推理时间, 未讨论在CPU、NPU等边缘设备上的性能.
**Why it matters**: 实际部署场景中, GPU可能不可用, 需要在CPU或NPU上运行.
**Suggestion**: 补充在不同硬件平台(GPU/CPU/NPU)上的推理时间和内存占用.
**Severity**: Minor

### W2: 缺少batch size敏感性分析
**Problem**: 推理时间在batch_size=32下测量, 未讨论不同batch size的影响.
**Why it matters**: 实际部署中batch_size通常为1, 需要了解单样本推理延迟.
**Suggestion**: 补充batch_size=1/4/16/32下的推理时间对比.
**Severity**: Minor

### W3: 精度差距需要更充分的讨论
**Problem**: SSM-WM的MSE比LSTM高55%, 论文仅用"同一数量级, 可接受"一笔带过.
**Why it matters**: 在控制任务中, 预测精度直接影响控制质量, 55%的差距可能不可接受.
**Suggestion**: 讨论在不同精度要求的场景下, SSM-WM的适用性. 提供精度-速度权衡的分析.
**Severity**: Minor

---

## Questions for Authors

1. 在CPU或NPU上的推理时间是多少?
2. batch_size=1时的推理延迟是多少?
3. 55%的MSE差距在实际控制任务中的影响有多大?

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor |
|-----------|--------------|------------|
| Originality | 60 | Adequate |
| Methodological Rigor | 55 | Weak |
| Evidence Sufficiency | 52 | Weak |
| Argument Coherence | 66 | Adequate |
| Writing Quality | 64 | Adequate |
| Significance & Impact | 68 | Adequate |
| **Weighted Average** | **58** | **Minor Revision** |
