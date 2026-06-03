# Final Figure Contract - SSM-WM Architecture

## Figure purpose
This figure replaces the original `paper/figures/architecture.pdf` used in Fig. 1 of `paper/main.tex`. It explains the proposed SSM-WM as a code-aligned method overview rather than a decorative block diagram.

## Required narrative
1. Humanoid robot state and action histories are concatenated as the sequence input.
2. A two-layer state-action encoder maps each time step to latent tokens.
3. The core contribution is the stacked gated diagonal SSM backbone.
4. Each SSM block contains LayerNorm, diagonal SSM/FFT causal convolution, gate fusion, and residual update.
5. The prediction head decodes the final hidden token and predicts a residual next state.
6. The same predicted state supports both training rollout loss and MPC-time planning.

## Mandatory visual elements
- Input histories: `s_{t-T:t}` and `a_{t-T:t-1}`.
- Encoder: Linear, GELU, Linear.
- Backbone: `L x Gated Diag-SSM` stack.
- Block inset: `LayerNorm -> DiagSSM -> Gate -> Fuse`, with a residual skip path.
- Output: `s_hat_{t+1} = s_t + delta_s`.
- MPC usage: training rollout and receding-horizon MPC.
- Complexity badges: `O(T log T)`, `O(1) step`, and `0.24M params`.

## Style constraints
- Publication schematic style, not cartoon or hand-drawn.
- Industrial blue-gray palette with orange only for gating/control-loop emphasis.
- White/light-gray background, rounded cards, thin technical arrows.
- Text kept short enough for paper readability.
- No extra modules not supported by the paper or code.

## Deliverables
- `architecture_redraw.svg`: editable vector source.
- `architecture_redraw.pdf`: LaTeX-ready vector figure.
- `architecture_redraw.png`: high-resolution raster preview.
