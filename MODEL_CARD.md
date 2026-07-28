# Model Card - ESPI DnCNN-ECA Variants

> **2026 revision:** the historical three-run robustness result is a pilot with
> incomplete seed propagation. Use
> `results/revision_2026_corrected_robustness/` for current robustness claims.

## Overview

This repository contains lightweight DnCNN-style denoisers for ESPI imagery, with emphasis on Efficient Channel Attention (ECA) ablations and final thesis-era V4/V5 comparisons.

The public codebase includes:

1. **Baseline DnCNN-Lite ECA script**
2. **V4 fair-ablation / stable thesis script**
3. **V5 extended research script**

The repository also preserves historical baseline material associated with the V3 stage for traceability, but the final thesis interpretation is tied to the curated V4/V5 package rather than to the earlier baseline stage alone.

## Task

The task is denoising of ESPI measurements or aligned ESPI-derived image pairs, with evaluation in both direct image-quality metrics and downstream classification impact.

## Corrected evidence summary

The corrected evidence supports the following high-level conclusions:

- Real-aligned supervision and output-contract design matter at least as much as
  architecture complexity.
- Reconstruction metrics alone do not guarantee downstream utility.
- The corrected five-seed in-distribution sweep favors **V5R aggressive ECA**
  over **V4R light ECA**.
- The six-board transfer audit does not show a universal denoising advantage over
  a noise-adapted Raw baseline.
- A matched U-Net GroupNorm ablation suggests that ECA reduces board-specific
  instability and improves Macro-F1, but the current U-Net result uses seed 42.
- NAFNet-Tiny with the audited proxy-target contract is a negative control for
  whitening-induced downstream collapse.

Representative corrected five-seed means:

- **Raw:** 93.28% Accuracy, 84.39% Macro-F1
- **V4R light ECA:** 92.54% Accuracy, 82.69% Macro-F1
- **V5R aggressive ECA:** 94.29% Accuracy, 85.89% Macro-F1

## Inputs and outputs

### Inputs

Depending on the script and evaluation regime, the main inputs are:

- clean reference images (`--clean-root`)
- optional real noisy images (`--real-noisy-root`)
- optional checkpoint file (`--resume`)

### Outputs

Typical outputs include:

- training logs
- checkpoints
- optional ONNX export
- final CSV result tables and plot-ready tables for the thesis package

## Intended use

These models are intended for:

- thesis support and reproducibility
- ESPI denoising research
- ablation studies on lightweight attention mechanisms
- downstream pipeline analysis where denoising quality is evaluated jointly with classification impact

They are not intended to be treated as production-ready denoisers without project-specific validation, data auditing, and deployment hardening.

## Limitations

The main limitations are the following:

- the raw project datasets are not included in the public repository,
- exact thesis data curation pipelines live partly outside this repository,
- denoising metrics alone are not sufficient to select the best model for downstream use,
- board-grouped transfer currently uses seed 42,
- the U-Net matched ablation is not yet a multi-seed denoiser-training estimate,
- class 2 remains difficult under the grouped protocol,
- the final conclusions rely on multiple protocols and must not pool their
  statistical units.

## Scientific notes

This repository should be interpreted as the **denoising component** of a larger thesis workflow. The pseudo-noisy generator and downstream classification repositories remain separate and are part of the complete research pipeline.
