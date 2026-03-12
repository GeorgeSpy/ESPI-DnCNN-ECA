# Model Card - ESPI DnCNN-ECA Variants

## Overview

This repository contains lightweight DnCNN-style denoisers for ESPI imagery, with emphasis on Efficient Channel Attention (ECA) ablations and final thesis-era V4/V5 comparisons.

The public codebase includes:

1. **Baseline DnCNN-Lite ECA script**
2. **V4 fair-ablation / stable thesis script**
3. **V5 extended research script**

## Task

The task is denoising of ESPI measurements or aligned ESPI-derived image pairs, with evaluation in both direct image-quality metrics and downstream classification impact.

## Reported thesis conclusions

The curated final thesis package supports the following high-level conclusions:

- Real-aligned denoiser supervision is more important than adding architectural complexity alone.
- Synthetic pseudo-noisy supervision can improve denoising metrics while still degrading downstream classification.
- The best overall downstream results were obtained by the **V4R ECA** configuration trained on real-aligned pairs.
- The more aggressive **V5** design increases cost substantially without becoming the preferred overall thesis choice.

Representative thesis-level downstream results from the canonical package:

- **Raw classification pipeline:** 97.70% accuracy, 93.99% Macro-F1
- **V4R no-ECA denoised pipeline:** 98.76% accuracy, 96.07% Macro-F1
- **V4R ECA denoised pipeline:** 98.87% accuracy, 96.64% Macro-F1

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
- the final thesis conclusions rely on curated result tables, not on any single metric in isolation.

## Scientific notes

This repository should be interpreted as the **denoising component** of a larger thesis workflow. The pseudo-noisy generator and downstream classification repositories remain separate and are part of the complete research pipeline.