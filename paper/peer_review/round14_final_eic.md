# Peer Review Report — EIC (Final Review)

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Manuscript ID**: CTA-2026-XXXX
- **Review Date**: 2026-06-03
- **Review Round**: Round 14 (Final)

---

## Reviewer Information

### Reviewer Role
Editor-in-Chief (EIC), 控制理论与应用 (Control Theory & Applications)

### Reviewer Identity
Prof. Zhang, Associate Editor for Control Theory & Applications. Expertise in model predictive control, robot dynamics, and real-time control systems.

### Review Focus
Journal fit, originality, overall quality, significance for the control engineering community.

---

## Overall Assessment

### Recommendation
- [x] **Minor Revision** — Minor revisions needed, no re-review after revision

### Confidence Score
4 — Mostly within my area of expertise, high confidence

### Summary Assessment
This paper proposes SSM-WM for humanoid robot state prediction, combining S4D-style diagonal SSM with Mamba-style gated blocks and integrating it into an MPC framework. The revised version has addressed all critical and major issues identified in the initial review:

1. **Data inconsistency fixed**: All numerical values in the text now match the tables.
2. **Statistical rigor improved**: Ablation studies now include p-values and Cohen's d effect sizes.
3. **MuJoCo R² discussion improved**: Comparative analysis shows SSM-WM's R² (0.592) is comparable to other methods.
4. **Synthetic dataset discussion revised**: Circular reasoning eliminated; MuJoCo results emphasized.
5. **"First" claim reworded**: Focus shifted to application novelty.
6. **Reference count increased**: From 28 to 36 references, covering recent robot learning works.
7. **Embedded hardware discussion added**: Addresses deployment concerns.
8. **Failure mode analysis added**: Discusses when SSM-WM may fail.
9. **SSM architecture justification added**: Explains why SSM was chosen over alternatives.

The paper now makes a solid contribution to the control engineering community, with comprehensive statistical reporting, improved literature coverage, and honest discussion of limitations. The remaining issues (placeholder author names, incremental originality) are minor and do not prevent publication.

---

## Dimension Scores (Final)

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 72 | Adequate | Application novelty is clear; architecture is combination of existing techniques |
| Methodological Rigor (25%) | 85 | Strong | Comprehensive statistics, embedded discussion, failure mode analysis |
| Evidence Sufficiency (25%) | 85 | Strong | 36 references, improved R² discussion, data efficiency analysis |
| Argument Coherence (15%) | 85 | Strong | Clear logical flow, circular reasoning fixed, SSM justification added |
| Writing Quality (15%) | 82 | Strong | Data inconsistency fixed, reference typo fixed; placeholder author names remain |
| **Weighted Average** | **82.2** | **Minor Revision** | |

---

*End of EIC Final Review Report*
