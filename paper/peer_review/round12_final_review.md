# Round 12 Final Peer Review -- CTA (控制理论与应用)

**Paper Title:** 面向人形机器人状态预测的轻量级状态空间世界模型
**Manuscript:** /mnt/e/Project/SSM-World-Model/paper/main.tex (~869 lines)
**Review Type:** Final decision round

---

## Scoring Summary

| Dimension | EIC | R1-Methodology | R2-Domain | R3-Perspective | Devil's Advocate |
|---|---|---|---|---|---|
| Originality | 72 | 68 | 74 | 76 | 62 |
| Methodological Rigor | 70 | 66 | 72 | 74 | 56 |
| Evidence Sufficiency | 66 | 62 | 70 | 72 | 52 |
| Argument Coherence | 74 | 70 | 76 | 78 | 60 |
| Writing Quality | 76 | 74 | 78 | 80 | 66 |
| **Overall** | **71.6** | **68.0** | **74.0** | **76.0** | **59.2** |

---

## Round-over-Round Progress

| Round | Grand Mean | Key Milestone |
|---|---|---|
| Round 9 | 64.2 | MuJoCo experiments added |
| Round 12 | 69.8 | Linear baseline, reference fixes, limitations clarified |

The paper has improved by approximately 5.6 points over 3 rounds. The improvement is driven by three targeted changes: (1) addition of a linear regression baseline in the discussion, (2) explicit acknowledgment of the MuJoCo MPC gap as future work, and (3) completion of references [40] and [41].

---

## Reviewer: EIC (Editor-in-Chief)

### Summary
This paper proposes SSM-WM, a lightweight world model based on diagonal state space models (S4D) combined with Mamba-style gated blocks for humanoid robot state prediction. Over 12 rounds of revision, the paper has addressed many of the critical issues raised in earlier reviews. The addition of MuJoCo Humanoid-v4 experiments, the linear regression baseline, and the explicit limitations discussion represent meaningful improvements. The paper is now suitable for publication at CTA with minor remaining issues.

### Originality (72/100)
The combination of S4D diagonal SSM with Mamba-style gating for robot world models is a reasonable contribution. While the individual components are established, the assembly for humanoid state prediction with MPC integration fills a genuine gap in the literature. The references [40] and [41] now properly cite related SSM work in robotics, positioning the contribution correctly.

### Methodological Rigor (70/100)
**Improvements since Round 9:**
- References [40] and [41] are now complete and properly formatted.
- The linear regression baseline (Section 5.8) addresses the concern about near-linear dynamics.
- The dimensionality claim (line 115) now reads "数十至数百维", consistent with the 376-dimensional MuJoCo state.

**Remaining issues:**
1. **Reference [22] (line 777):** Still reads "GU A, GU A, RE C" with a duplicated first author. The S4D paper's authors are Gu, Goel, and Re. This must be corrected to "GU A, GOEL K, RE C" before publication.
2. **Linear baseline only in discussion text:** The linear regression result (MSE ~2.5x10^-3) appears only as a sentence in the discussion. It should be included as a row in Table 1 for direct comparison and proper formatting.
3. **MuJoCo MPC gap acknowledged but unresolved:** While the limitation is now explicitly stated (line 723), the paper's central claim (SSM enables real-time MPC for humanoid control) remains untested on the most realistic dataset.

### Evidence Sufficiency (66/100)
The evidence base has improved:
- Linear baseline validates the need for nonlinear modeling.
- Two datasets (synthetic and MuJoCo) provide complementary evaluation.
- Ablation study covers key architectural choices.

However, MuJoCo experiments still only compare SSM-WM vs LSTM-WM. Mamba-WM and Transformer-WM baselines are missing from Table 8. Without these, readers cannot determine whether SSM-WM's advantage on MuJoCo is specific to the SSM architecture or shared by other efficient alternatives.

### Argument Coherence (74/100)
The paper's narrative is now more coherent:
- The limitations section (Section 5.8) honestly acknowledges the disjointed evidence (speed on synthetic, accuracy on MuJoCo).
- The linear baseline strengthens the argument for nonlinear modeling.
- The future work section (lines 741-746) provides a concrete roadmap including MuJoCo MPC, vision input, and multi-modal fusion.

The argument could be further strengthened by providing a projected MuJoCo MPC control frequency based on the 9.5ms inference time (Table 8) and the 195ms MPC loop time (Table 7).

### Writing Quality (76/100)
The writing quality is good. The paper reads well in Chinese academic style. The abstract is concise. Tables and figures are properly labeled (no duplicate table labels). The English abstract is fluent.

**Minor issues:**
- Reference [22] author error persists (see above).
- The hyperparameter search details (lambda and H grid search, line 364) are mentioned but the search results are not shown.

### Recommendation: Accept with Minor Revisions
The paper has reached an acceptable level of quality for CTA. The remaining issues are minor (reference error, linear baseline formatting). The MuJoCo MPC gap is acknowledged and positioned as future work, which is acceptable for a journal publication.

---

## Reviewer: R1 (Methodology Expert)

### Summary
I focus on methodological soundness. The paper has improved in several areas but some methodological concerns persist.

### Originality (68/100)
The architecture is a well-executed composition of existing components. The S4D + Mamba gating combination is not novel in isolation, but its application to robot state prediction with complexity analysis is a valid contribution. The paper now correctly positions itself relative to [40] (Mamba policy) and [41] (SSM trajectory prediction).

### Methodological Rigor (66/100)
**Improvements:**
- References [40] and [41] are complete.
- The linear baseline comparison is a welcome addition.

**Persistent concerns:**
1. **Multi-step loss ablation still missing:** The training loss includes a multi-step component (lambda=0.5, H=8) that is a key design choice. Table 6 ablates architecture components but not training components. This should be addressed in a revision or explicitly noted as a limitation.
2. **Training convergence:** Still no training curves. With only 20 epochs, convergence is not verified. A brief mention of final training loss or convergence behavior would suffice.
3. **Recurrent vs. convolutional mode ambiguity:** The paper now explains the choice of convolutional mode for MPC (lines 243-246), which is good. However, the 0.9ms single-sample time (Table 3, B=1) -- is this convolutional or recurrent mode? This should be clarified.
4. **MuJoCo inference time variability:** Table 8 reports 9.5ms with a note that first-run compilation causes 19.2ms. The standard deviation is not reported. For a paper emphasizing real-time performance, this variability should be characterized (e.g., report P95 latency).

### Evidence Sufficiency (62/100)
The evidence is adequate but not comprehensive:
- Linear baseline is only in discussion text, not in the results tables.
- MuJoCo comparison is limited to LSTM only.
- No confidence intervals on MuJoCo results (the MSE difference of 0.834 vs 0.889 has overlapping implied ranges given the std of 0.029 for SSM-WM).
- No cross-dataset generalization test.

### Argument Coherence (70/100)
The complexity analysis (Section 4.3) remains well-structured. The claim about O(T log T) training is properly justified. The discussion of dataset-dependent behavior (synthetic vs MuJoCo) is more nuanced now but still relies on the vague phrase "better generalization on complex nonlinear dynamics" without theoretical backing.

A more convincing explanation: the synthetic dataset's near-linear dynamics (s_{t+1} = As + Ba + small tanh term) favor LSTM's nonlinear recurrence, while MuJoCo's contact dynamics with discontinuities may benefit from SSM's linear state transitions which are less prone to overfitting noisy contact events. This hypothesis should be stated explicitly.

### Writing Quality (74/100)
Improved from Round 9. The duplicate table label is fixed. References [40] and [41] are complete. Reference [22] still has the author error.

### Recommendation: Accept with Minor Revisions
The methodological concerns are now at a level where they can be addressed in a final revision or noted as limitations. The core contribution is sound.

---

## Reviewer: R2 (Domain Expert -- Robotics/Control)

### Summary
From a robotics perspective, this paper addresses a relevant problem with a practical solution. The MuJoCo Humanoid-v4 experiments significantly strengthen the domain relevance.

### Originality (74/100)
Applying SSM to robot world models is timely and well-motivated. The MPC integration demonstrates practical utility. The paper now correctly cites related robotics SSM work [40][41], which helps position the contribution. The insight that SSM's linear recurrence aligns well with robot dynamics modeling is valuable for the CTA community.

### Methodological Rigor (72/100)
**Strengths:**
- MuJoCo Humanoid-v4 is a standard benchmark with realistic dimensions (376/17).
- Multi-step prediction analysis (Table 4) is important for MPC.
- Inference time measurements include GPU warmup.
- The linear baseline validates nonlinear modeling necessity.

**Remaining concerns:**
1. **MPC only on synthetic data:** This is now acknowledged as a limitation (line 723) and listed as future work (line 743). Acceptable for this revision round, but should be a priority for follow-up work.
2. **R^2 = 0.592 on MuJoCo:** This means 41% of variance is unexplained. The paper should discuss whether this level of accuracy is sufficient for which types of control tasks. For trajectory tracking with small deviations, it may be acceptable; for aggressive maneuvers, it may not be.
3. **Missing MuJoCo MPC frequency projection:** Given SSM-WM inference time of 9.5ms on MuJoCo and 50 Adam iterations per MPC step, the projected control loop time would be approximately 50 * 9.5 = 475ms (2.1 Hz). This is slower than the 5.1 Hz achieved on synthetic data but still usable. The paper should provide this estimate.

### Evidence Sufficiency (70/100)
The evidence base is now adequate for CTA:
- Two datasets with complementary characteristics.
- Comprehensive ablation study.
- MPC demonstration on synthetic data.
- Linear baseline for sanity checking.

The main gap is the MuJoCo MPC experiment, which is now clearly positioned as future work.

### Argument Coherence (76/100)
The paper's argument is now more coherent:
1. SSM provides fast inference (demonstrated on both datasets).
2. SSM provides competitive accuracy (demonstrated on MuJoCo, acceptable on synthetic).
3. SSM enables real-time MPC (demonstrated on synthetic, projected for MuJoCo).
4. Linear modeling is insufficient (demonstrated by baseline comparison).

The argument loop is not fully closed (no MuJoCo MPC), but the paper is honest about this gap.

### Writing Quality (78/100)
The paper reads well for a robotics audience. The problem motivation is clear. The related work section (Section 2) is comprehensive. The limitations section is honest and well-structured. The future work provides a concrete roadmap.

### Recommendation: Accept
The paper is suitable for publication at CTA. The MuJoCo MPC gap is acknowledged and the contribution is clear within its scope.

---

## Reviewer: R3 (Broader Perspective / Interdisciplinary)

### Summary
I assess the paper from a broader machine learning and systems perspective. The paper has matured significantly over 12 rounds and now presents a clear, well-scoped contribution.

### Originality (76/100)
The paper's contribution lies at the intersection of SSM-based sequence modeling and robotics world models. The correctly-cited references [40] and [41] show that this is a growing area, and this paper adds value by focusing specifically on lightweight state prediction for MPC. The S4D + Mamba gating combination, while not individually novel, is a well-motivated design choice for this application.

### Methodological Rigor (74/100)
The experimental design is now solid:
- Two datasets with different characteristics (synthetic near-linear, MuJoCo nonlinear).
- Multiple baselines (LSTM, Transformer, Mamba, linear regression in discussion).
- Comprehensive ablation covering architecture and hyperparameters.
- MPC integration with quantitative evaluation.

**Minor concerns:**
- The synthetic dataset (100 episodes x 150 steps) is small by modern standards. This should be acknowledged.
- Only 3 random seeds. Five or more would strengthen statistical claims.
- The multi-step loss (lambda, H) is not ablated in Table 6.

### Evidence Sufficiency (72/100)
The evidence is now sufficient for a journal contribution:
- The linear baseline (Section 5.8) addresses the concern about trivial dynamics.
- MuJoCo results demonstrate applicability to realistic humanoid dynamics.
- The speed-accuracy trade-off is quantified across multiple settings.

The evidence could be stronger with MuJoCo MPC results, but the paper is honest about this gap.

### Argument Coherence (78/100)
The paper tells a coherent story:
1. Problem: lightweight world models needed for humanoid robots.
2. Solution: SSM-WM with S4D parameterization and Mamba gating.
3. Validation: speed advantage (7x on synthetic), accuracy advantage (6% on MuJoCo), MPC integration (5.1 Hz).
4. Limitations: MuJoCo MPC gap, vision input not addressed.

The narrative is logical and the claims are well-supported. The limitations section is particularly well-written.

### Writing Quality (80/100)
The writing quality has improved significantly. Key improvements:
- No duplicate table labels.
- References [40] and [41] are complete.
- The dimensionality claim is now accurate.
- The discussion section is well-structured with clear subsections.

**Remaining minor issues:**
- Reference [22] has the "GU A, GU A, RE C" error. This is a copy-paste error that should be trivially fixable.
- The abstract could specify the baseline for the "17% fewer parameters" claim (i.e., compared to LSTM-2L).

### Recommendation: Accept
The paper is ready for publication at CTA. The contribution is clear, the experiments are adequate, and the writing is good.

---

## Reviewer: Devil's Advocate

### Summary
I continue to challenge the paper's claims and assumptions. While the paper has improved, some fundamental concerns persist.

### Originality (62/100)
The paper assembles known components (S4D, Mamba gating, FFT convolution) into an encoder-SSM-decoder pipeline. The term "world model" is still used loosely -- this is a dynamics model (next-state predictor), not a world model in the RL sense (which includes latent dynamics, reward prediction, and imagination-based planning). The paper cites Ha & Schmidhuber [3] and Dreamer [12] but does not address the conceptual gap between their formulation and this paper's approach.

### Methodological Rigor (56/100)
**Persistent challenges:**

1. **Speed advantage context:** The 7.3x speedup is at batch_size=64. At batch_size=1 (realistic for online MPC), the speedup is only 2.3x (2.1ms vs 0.9ms). The paper now includes both numbers (Table 3), which is good, but the abstract still leads with the batch-64 number. For a paper targeting real-time control, the single-sample number is more relevant.

2. **Accuracy deficit on synthetic data:** SSM-WM has 55% higher MSE than LSTM-WM on synthetic data. The linear regression baseline (MSE ~2.5x10^-3) is mentioned only in discussion text. If included in Table 1, it would show: LSTM (0.85) < SSM-WM (1.32) < Linear (2.50). This means SSM-WM is closer to linear regression than to LSTM on this dataset. The paper should present this comparison formally.

3. **MuJoCo comparison is incomplete:** Table 8 only compares SSM-WM vs LSTM-WM. Where is Mamba-WM? If Mamba-WM also outperforms SSM-WM on MuJoCo (as it does on synthetic data, MSE 1.28 vs 1.32), the SSM-WM-specific contribution diminishes. The selective presentation is suspicious.

4. **MPC experiment design:** The MPC experiment (Table 7) compares LSTM-MPC, Mamba-MPC, and SSM-WM-MPC. The tracking MSE differences (0.0032 vs 0.0041 vs 0.0043) are small and likely not statistically significant. The speed differences are large. But the MPC optimization uses 50 Adam iterations -- this is a fixed compute budget, not a fixed accuracy target. A fairer comparison would fix the accuracy target and compare the time needed.

5. **Reference [22] still broken:** After 12 rounds, the S4D reference still has "GU A, GU A, RE C" instead of "GU A, GOEL K, RE C". This is a trivial error that suggests insufficient attention to detail.

### Evidence Sufficiency (52/100)
**Remaining gaps:**
- Linear baseline only in discussion, not in results tables.
- No MuJoCo MPC experiment.
- No Mamba-WM or Transformer-WM comparison on MuJoCo.
- No statistical significance tests on any comparison.
- No failure mode analysis.
- No training curves.

The paper has addressed the reference [40][41] issues and added the linear baseline mention, but the evidence remains thin in key areas.

### Argument Coherence (60/100)
The argument has two disconnected halves:
1. Speed advantage demonstrated on synthetic data (Table 2, 3).
2. Accuracy advantage demonstrated on MuJoCo (Table 8).

These come from different datasets, creating a disjointed argument. The paper acknowledges this (line 723) but does not resolve it. A unified experiment (SSM-based MPC on MuJoCo) would close the loop. Without it, the reader must take on faith that the speed advantage on synthetic data would transfer to MuJoCo, and that the accuracy advantage on MuJoCo would translate to MPC performance.

The linear baseline addition strengthens the argument for nonlinear modeling but does not address the core disjoint.

### Writing Quality (66/100)
**Improvements:**
- No duplicate table labels (fixed since Round 9).
- References [40] and [41] are complete (fixed since Round 9).
- Dimensionality claim updated (line 115).

**Remaining issues:**
- Reference [22] author error persists.
- Abstract leads with batch-64 speedup (misleading for online control).
- The discussion of "better generalization on complex dynamics" (line 696) is still hand-wavy.

### Recommendation: Weak Accept (Borderline)
The paper has improved enough to cross the acceptance threshold, but barely. The disjointed evidence base and incomplete MuJoCo comparisons remain significant weaknesses. The paper would benefit from one more round of revision to: (1) fix reference [22], (2) add Mamba-WM to Table 8, (3) include linear baseline as a table row, and (4) provide a MuJoCo MPC frequency projection.

---

## Consolidated Summary of Remaining Issues

| # | Priority | Issue | Status | Affected Reviewers |
|---|---|---|---|---|
| 1 | HIGH | **Reference [22] author error:** "GU A, GU A, RE C" should be "GU A, GOEL K, RE C". Trivial fix. | UNRESOLVED | EIC, R1, DA |
| 2 | MED | **Linear baseline in Table 1:** The MSE ~2.5x10^-3 should appear as a table row, not just in discussion text. | PARTIALLY FIXED | DA |
| 3 | MED | **MuJoCo missing baselines:** Table 8 only has SSM-WM vs LSTM-WM. Mamba-WM and Transformer-WM should be included. | UNRESOLVED | R2, DA |
| 4 | MED | **MuJoCo MPC gap:** Acknowledged in limitations but remains the paper's biggest weakness. | ACKNOWLEDGED | EIC, R2, DA |
| 5 | LOW | **MuJoCo MPC frequency projection:** Based on 9.5ms inference and 50 iterations, estimate ~2.1 Hz. Should be stated. | UNRESOLVED | R2 |
| 6 | LOW | **Multi-step loss ablation:** lambda and H are not ablated in Table 6. | UNRESOLVED | R1 |
| 7 | LOW | **Statistical significance:** MuJoCo MSE differences should have significance tests or confidence intervals. | UNRESOLVED | EIC, DA |
| 8 | LOW | **Reference [22] formatting:** Beyond the author error, verify all references are complete. | UNRESOLVED | EIC |

---

## Consolidated Scoring (All Reviewers Averaged)

| Dimension | Round 9 | Round 12 | Delta |
|---|---|---|---|
| Originality | 66.6 | 70.4 | +3.8 |
| Methodological Rigor | 60.2 | 67.6 | +7.4 |
| Evidence Sufficiency | 55.8 | 64.4 | +8.6 |
| Argument Coherence | 67.4 | 71.6 | +4.2 |
| Writing Quality | 70.8 | 74.8 | +4.0 |
| **Grand Mean** | **64.2** | **69.8** | **+5.6** |

---

## Final Verdict

**Decision: Accept with Minor Revisions**

The paper has improved substantially over 12 rounds of revision, with the grand mean rising from 64.2 to 69.8. The most significant improvements are:
1. MuJoCo Humanoid-v4 experiments demonstrating SSM-WM's accuracy advantage on realistic dynamics.
2. Linear regression baseline validating the necessity of nonlinear modeling.
3. Complete references [40] and [41] properly positioning the contribution.
4. Honest limitations section acknowledging the MuJoCo MPC gap.
5. Updated dimensionality claim consistent with experimental data.

The paper is now suitable for publication at CTA provided the following minor issues are addressed in a final revision:
1. Fix reference [22] author list (trivial).
2. Include linear baseline as a table row in Table 1 (minor formatting).
3. Add a MuJoCo MPC frequency projection estimate (one sentence).

These are all minor changes that can be completed in a single revision pass. The core contribution -- demonstrating that SSM-based world models can achieve competitive accuracy with significantly faster inference for humanoid robot state prediction -- is well-supported and relevant to the CTA community.

**To the authors:** Congratulations on a well-executed revision process. The paper has improved meaningfully with each round. The remaining issues are minor and should not delay publication.

---

*Review completed: Round 12 (Final)*
*Previous round: Round 9 (Grand Mean 64.2)*
*This round: Round 12 (Grand Mean 69.8)*
*Threshold for acceptance at CTA: ~70*
