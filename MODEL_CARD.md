# Model Card - ESPI DnCNN-ECA Variants

## Overview

This repository provides lightweight DnCNN-style convolutional denoisers for Electronic Speckle Pattern Interferometry (ESPI), focusing on Efficient Channel Attention (ECA) configurations and their impact on downstream vibration-mode classification.

The public codebase includes:

1. **Baseline DnCNN-Lite ECA script** (`espi_dncnn_lite_eca.py`)
2. **V4 light ECA script** (`espi_dncnn_lite_eca_FULL_PATCH_v4.py`)
3. **V5 extended ECA research script** (`espi_dncnn_lite_eca_FULL_PATCH_v5.py`)

## Task

Restoration of single-shot ESPI interferograms corrupted by optical speckle noise, evaluated across both image-domain reconstruction metrics and downstream 5-class vibration-mode recognition.

## Performance Summary

Multi-seed evaluation across five random seeds (`42, 13, 37, 101, 202`) under heavy additive stress noise ($\sigma = 25$):

- **Raw (Unprocessed baseline):** 93.28% Accuracy, 84.39% Macro-F1
- **V4R (Light ECA, 3 layers):** 92.54% Accuracy, 82.69% Macro-F1
- **V5R (Dense ECA, 7 layers):** **94.29% Accuracy, 85.89% Macro-F1**

Key findings:
- Real-aligned supervision with genuine ESPI speckle statistics significantly outperforms synthetic pseudo-noise supervision.
- Denser attention placement (V5R) provides enhanced regularization under severe synthetic stress noise (+1.75% Acc, +3.19% Macro-F1 over V4R).
- Downstream task performance on unseen physical specimens is material-dependent, with the unprocessed Raw input remaining highly competitive on carbon specimens.
- In residual U-Net architectures, channel attention efficacy is normalization-dependent, showing substantial gains under GroupNorm (+14.29% Macro-F1) and neutral/attenuated effects under BatchNorm.

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
- final CSV result tables and plot-ready tables

## Intended use

These models are intended for:

- scientific imaging and interferometry research
- ESPI denoising benchmarks
- ablation studies on lightweight attention mechanisms
- downstream pipeline analysis evaluating joint restoration and classification impact

## Limitations

- The raw experimental interferograms are proprietary and not bundled with the public repository.
- Image restoration metrics (PSNR/SSIM) alone are insufficient to predict downstream mode recognition accuracy.
- Physical specimen transfer demonstrates material-dependent performance, requiring validation on target experimental setups.
- U-Net matched sensitivity is evaluated across the six specimen boards at seed 42.

## Scientific notes

This repository should be interpreted as the **denoising component** of a larger thesis workflow. The pseudo-noisy generator and downstream classification repositories remain separate and are part of the complete research pipeline.
