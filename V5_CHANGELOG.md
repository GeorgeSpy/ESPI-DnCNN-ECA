# DnCNN Lite ECA v5 - Changelog

## New in v5

1. Dual pooling ECA (GAP + GMP)
- Added max-pooling branch in ECA for peak-sensitive channel descriptors.
- Why: ESPI fringes often include sharp maxima; GAP-only can miss those peaks.

2. Learnable temperature and gain (optional)
- New flag: `--eca-learnable-temp-gain`.
- Learnable params: `log_temp`, `raw_gain` with safe transforms/clamps.
- Why: per-layer adaptation instead of fixed global scalars.

3. Multi-scale ECA kernels (optional)
- New flag: `--eca-multi-scale`.
- Parallel kernels: k=3,5,7 in channel-attention path.
- Why: capture short and wider cross-channel dependencies.

4. ECA placement presets
- New flag: `--eca-preset {none,shallow3,dense_shallow}`.
- `dense_shallow` preset: [0,1,2,3,6,10,14] (auto-clipped by depth).
- Why: denser attention in shallow blocks where fringe/edge info dominates.

5. ECA order option
- New flag: `--eca-order {post,pre}`.
- Allows attention after or before ConvBlock.
- Why: test whether pre-activation modulation helps on your data.

6. ECA optimizer param groups (optional)
- New flags:
  - `--separate-eca-param-optim`
  - `--eca-param-lr-scale`
  - `--eca-param-weight-decay`
- Applies separate LR/WD to `temp/gain` learnable params.
- Why: stabilize newly learnable ECA scalars.

## Kept from v4
- `--no-eca` fair baseline mode
- GN control via `--gn-groups`
- non-finite guards (`--nan-action`, `--max-nonfinite-batches`)
- strict resume default
- expanded diagnostics in CSV/TensorBoard

## Expected improvements

Primary expected gains:
- better edge/fringe preservation (EdgeF1 up)
- modest SSIM gain
- similar or slightly improved PSNR

Typical realistic range vs no-ECA:
- dPSNR: -0.05 to +0.40 dB
- dSSIM: +0.00 to +0.04
- dEdgeF1: +0.01 to +0.04

## Risk notes
- Multi-scale + learnable params increases optimization complexity.
- If unstable:
  - reduce LR
  - keep `--eca-order post`
  - use `--eca-preset shallow3`
  - keep `--nan-action stop`.
