# Editorial Decision (Final)

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Manuscript ID**: CTA-2026-XXXX
- **Submission Date**: 2026-06-03
- **Decision Date**: 2026-06-03
- **Review Round**: Round 14 (Final)

---

## Decision *

### Minor Revision

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence | Weighted Score |
|----------|------|---------------|------------|----------------|
| EIC | 控制理论与应用 Editor | Minor Revision | 4 | 82.2 |
| Reviewer 1 | Methodology Expert | Minor Revision | 5 | 82.4 |
| Reviewer 2 | Domain Expert (Robotics) | Minor Revision | 5 | 84.0 |
| Reviewer 3 | Cross-Disciplinary (Efficient Computing) | Minor Revision | 4 | 84.4 |
| Devil's Advocate | Adversarial Critic | Minor Revision | — | 80.0 |

**Average Score**: 82.6 (Minor Revision range: 65-79, but close to Accept threshold of 80)

---

## Score Progression

| Reviewer | Initial Score | Final Score | Improvement |
|----------|--------------|-------------|-------------|
| EIC | 69.7 | 82.2 | +12.5 |
| R1 Methodology | 68.4 | 82.4 | +14.0 |
| R2 Domain | 71.2 | 84.0 | +12.8 |
| R3 Perspective | 72.4 | 84.4 | +12.0 |
| Devil's Advocate | 62.8 | 80.0 | +17.2 |
| **Average** | **68.9** | **82.6** | **+13.7** |

---

## Consensus Analysis *

### Points of Agreement (Consensus)

**[CONSENSUS-5]** (All reviewers agree):
1. **All critical issues fixed**: The text-table data inconsistency has been resolved. All numerical values are now consistent.
2. **Statistical rigor is exemplary**: The 5-seed experiments with p-values, CIs, and Cohen's d set a high standard for the control engineering literature.
3. **MPC integration is a genuine strength**: The real-time control demonstration (5.1Hz/2.1Hz) is practically meaningful.

**[CONSENSUS-4]** (4/5 reviewers agree):
1. **Literature coverage is now adequate**: The addition of 8 new references brings the total to 36, covering recent robot learning works.
2. **Embedded hardware discussion is valuable**: Acknowledging that GPU benchmarks may not transfer to embedded platforms is important for practical deployment.
3. **Failure mode analysis strengthens the paper**: Discussing when SSM-WM may fail is important for safety-critical applications.

### Points of Disagreement

**Disagreement 1: Originality assessment**
- **DA view**: Originality is "Adequate" (72) — the architecture is a combination of existing techniques.
- **EIC/R1/R2/R3 view**: Originality is "Adequate" to "Strong" (72-78) — the application to robot world models with MPC integration is novel.
- **Disagreement type**: Severity disagreement
- **Editor's Resolution**: Originality is "Adequate" (72-78 range). The novelty is in the application domain and MPC integration, not in the architecture itself. This is acceptable for 控制理论与应用.
- **Resolution Rationale**: The paper makes a solid contribution to the field, even if the architecture is incremental. The MPC integration and comprehensive experimental validation add significant value.

**Disagreement 2: Embedded hardware benchmarks**
- **R3/DA view**: Actual embedded benchmarks are needed to support the speed claims.
- **EIC/R1/R2 view**: GPU benchmarks are sufficient for a first paper; embedded evaluation can be future work.
- **Disagreement type**: Perspective difference
- **Editor's Resolution**: The paper should acknowledge that all benchmarks are GPU-specific and discuss expected performance on embedded hardware. Actual embedded benchmarks can be future work.
- **Resolution Rationale**: Requiring embedded benchmarks would significantly delay publication. The paper's contribution is in the architecture and MPC integration, not in embedded deployment.

---

## Decision Rationale *

The paper proposes SSM-WM for humanoid robot state prediction, combining S4D diagonal SSM with Mamba-style gated blocks and integrating it into an MPC framework. All reviewers agree that the paper makes a solid contribution to the field, with practical relevance for real-time robot control.

The revised version has addressed all critical and major issues:
1. **Data inconsistency fixed**: All numerical values in the text now match the tables.
2. **Statistical rigor improved**: Ablation studies now include p-values and Cohen's d effect sizes.
3. **MuJoCo R² discussion improved**: Comparative analysis shows SSM-WM's R² is comparable to other methods.
4. **Synthetic dataset discussion revised**: Circular reasoning eliminated; MuJoCo results emphasized.
5. **"First" claim reworded**: Focus shifted to application novelty.
6. **Reference count increased**: From 28 to 36 references.
7. **Embedded hardware discussion added**: Addresses deployment concerns.
8. **Failure mode analysis added**: Discusses when SSM-WM may fail.
9. **SSM architecture justification added**: Explains why SSM was chosen over alternatives.

The remaining issues (placeholder author names, incremental originality) are minor and do not prevent publication. The paper's contributions (SSM for robot world models, MPC integration, systematic experiments) are valuable for the control engineering community.

---

## Required Revisions * (Must Fix)

| # | Revision Item | Source Reviewer | Severity | Section | Estimated Effort |
|---|--------------|----------------|----------|---------|-----------------|
| R1 | Replace placeholder author names | EIC | Minor | Title page | 0.5 days |

### Required Item Details

**R1: Replace Placeholder Author Names**
- **Problem**: The paper still has "Author Name" placeholders on the title page.
- **Source**: EIC
- **Requirement**: Replace with actual author names before submission.
- **Acceptance criteria**: Author names are present on the title page.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Source Reviewer | Priority | Section | Expected Improvement |
|---|--------------|----------------|----------|---------|---------------------|
| S1 | Consider adding threshold function comparison experiment | DA | P2 | Section 5 | Boost originality |
| S2 | Add actual embedded hardware benchmarks | R3, DA | P2 | Section 5 | Strengthen deployment claims |
| S3 | Improve abstract structure (lead with key finding) | EIC | P3 | Abstract | Better readability |

---

## Revision Roadmap *

### Priority 1 — Final Polish (Estimated total effort: 1 day)
- [ ] R1: Replace placeholder author names

### Priority 2 — Optional Enhancements (Estimated total effort: 3-5 days)
- [ ] S1: Consider adding threshold function comparison experiment
- [ ] S2: Add actual embedded hardware benchmarks (if available)

### Priority 3 — Text and Formatting (Estimated total effort: 0.5 days)
- [ ] S3: Improve abstract structure
- [ ] Final language polishing

### Total Estimated Effort
- **Minor Revision**: 1-2 days (required only)
- **With optional enhancements**: 4-7 days

---

## Closing

We are pleased to invite you to submit a revised version of your manuscript. The reviewers have unanimously agreed that the paper makes a solid contribution to the field, and all critical and major issues have been addressed. The remaining revision (replacing placeholder author names) is straightforward and should not delay publication.

The paper's contributions (SSM for robot world models, MPC integration, comprehensive experimental validation) are valuable for the control engineering community. We look forward to receiving your final revision.

---

## Appendix: Reviewer Score Summary (Final)

| Reviewer | Originality | Methodology | Evidence | Coherence | Writing | Weighted Avg |
|----------|------------|-------------|----------|-----------|---------|--------------|
| EIC | 72 | 85 | 85 | 85 | 82 | 82.2 |
| R1 Methodology | 70 | 87 | 85 | 85 | 82 | 82.4 |
| R2 Domain | 76 | 85 | 85 | 87 | 84 | 84.0 |
| R3 Perspective | 78 | 85 | 85 | 87 | 84 | 84.4 |
| Devil's Advocate | 72 | 82 | 82 | 82 | 82 | 80.0 |
| **Average** | **73.6** | **84.8** | **84.4** | **85.2** | **82.8** | **82.6** |

---

*End of Editorial Decision (Final)*
