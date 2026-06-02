# Editorial Decision - Round 1

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Decision Date**: 2026-06-02
- **Decision Round**: Round 1

---

## Editorial Decision

### Decision: **Major Revision**

### Decision Rationale
论文选题契合专刊方向, SSM在推理速度上的优势(7.3x加速)具有实际价值. 但存在以下关键问题需要解决:

1. **学术诚信问题(必须修复)**: 摘要与正文数据不一致, 论文描述与代码实现不一致
2. **实验完整性问题(必须修复)**: 消融实验无数据, Transformer数据缺失
3. **论文完整性问题(必须修复)**: 缺少架构图, 参考文献不足

### Consensus Issues (All Reviewers Agree)
1. 摘要与正文数据不一致 (Critical)
2. 消融实验缺少数据 (Major)
3. 缺少架构图 (Major)
4. 参考文献不足 (Major)

### Disagreement Issues
- Reviewer 1认为方法学问题严重(Critical), EIC和Reviewer 3认为是Major
- Reviewer 3对精度差距的容忍度高于其他审稿人

---

## Revision Roadmap

### Priority 1 (Critical - Must Fix)
1. **统一数据来源**: 修改摘要, 使用合成数据的准确描述; 或使用真实Open X-Embodiment数据集
2. **统一方法描述**: 修改论文3.1节, 准确描述DiagSSM架构; 或修改代码实现Mamba

### Priority 2 (Major - Strongly Recommended)
3. **补充消融实验**: 运行消融实验并报告具体数据
4. **补充架构图**: 绘制SSM-WM架构图
5. **补充Transformer数据**: 补充T=64下的完整实验数据
6. **扩充参考文献**: 补充到15-20篇

### Priority 3 (Minor - Suggested)
7. **补充统计检验**: 多次运行, 报告均值±标准差
8. **补充序列长度敏感性分析**: T=16/32/64/128对比
9. **讨论精度差距**: 更充分地讨论55%的MSE差距

---

## Score Summary

| Reviewer | Recommendation | Weighted Score |
|----------|---------------|----------------|
| EIC | Minor Revision | 58 |
| R1 (Methodology) | Major Revision | 51 |
| R2 (Domain) | Major Revision | 54 |
| R3 (Perspective) | Minor Revision | 58 |
| Devil's Advocate | N/A | N/A |
| **Average** | **Major Revision** | **55** |
