# DnCNN-ECA Version Comparison Report

This report summarizes the architectural evolution of the public DnCNN-Lite ECA variants, from the earliest CPU-safe versions to the later V4 and V5 scripts, and links that evolution to both denoising metrics and downstream classification behavior.

---

## 1. Architecture Evolution and Features

| File Version | Attention Type | Pooling | Spatial Attention | Config Style | Main Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`FIXED.py`** *(V1)* | Simple ECA (fixed kernel) | Global Average Pooling | No | Hardcoded | Minimal CPU-safe ECA integration after convolution blocks. |
| **`PATCHED.py` / `v2.py`** | Simple ECA (fixed kernel) | Global Average Pooling | Yes (`SpatialLiteAttention`) | Hardcoded | Added lightweight spatial attention to model not only which channels matter, but also where to focus spatially. |
| **`..._v3.py`** *(fixed-patched branch)* | Squeeze-and-Excitation (SE) | Global Average Pooling | Yes | Hardcoded | Experimental stage where ECA was temporarily replaced by an SE block with linear layers. |
| **`FULL_PATCH_v3.py`** | Advanced ECA | Global Average Pooling | Yes | Config class | Major refactor introducing `DnCNNLiteECAConfig`, temperature/gain controls, and mixed-precision support. |
| **`espi_dncnn_lite_eca_FULL_PATCH_v4.py`** | Advanced ECA | Global Average Pooling | Yes | Config class | Stable thesis-era version with cleaner I/O, fair ECA vs no-ECA controls, and better experiment reliability. |
| **`espi_dncnn_lite_eca_FULL_PATCH_v5.py`** | Extended ECA | Dual Pooling (avg + max) | Yes | Config class | Research-oriented extension with dual pooling, optional learnable temp/gain, multi-scale kernels, and placement presets. |

---

## 2. Denoising Metrics

The values below summarize comparative evaluation on both **synthetic validation** and **real ESPI pairs / averages**.

| Model | Training Regime | Val PSNR (Synthetic) | Val SSIM | Val EdgeF1 | Real PSNR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V4 Baseline (NoECA)** | Pseudo-noisy synthetic supervision | 27.24 dB | 0.7846 | 0.7686 | 34.58 dB |
| **V4 ECA** | Pseudo-noisy synthetic supervision | **27.47 dB** | **0.7972** | **0.7787** | 34.50 dB |
| **V5 Baseline (NoECA)** | Pseudo-noisy synthetic supervision | 27.24 dB | 0.7846 | 0.7686 | **34.58 dB** |
| **V5 ECA (Advanced)** | Pseudo-noisy synthetic supervision | 27.24 dB | 0.7852 | 0.7712 | 34.22 dB |
| **V4R Baseline (NoECA)** | Real pairs (23,891 images) | *N/A* | *N/A* | *N/A* | 23.76 dB *(real validation)* |
| **V4R ECA** | Real pairs (23,891 images) | *N/A* | *N/A* | *N/A* | **23.85 dB** *(real validation)* |

### Key denoising observations

- **V4 ECA** is clearly strongest on the reported synthetic denoising metrics, with gains of +0.23 dB PSNR and +0.0126 SSIM over the V4 baseline.
- **V5 ECA**, despite being architecturally more ambitious, does not consistently outperform the simpler V4 ECA configuration in the reported experiments.
- On **real-aligned training pairs**, adding ECA in the V4R regime provides a small but stable denoising gain over the no-ECA counterpart.

---

## 3. Downstream Classification Task

The table below summarizes how each denoising regime affected the downstream 5-class ResNet-18 classifier.

| Pre-processing Pipeline | Denoiser Training Data | ECA Enabled | Classification Accuracy (%) | Classification Macro-F1 (%) | dAcc vs Raw |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No denoising (Raw)** | None | No | 97.70 | 93.99 | 0.00 |
| **V4 denoised** | Pseudo-noisy synthetic supervision (243 images) | No | 96.39 | 89.06 | -1.31 |
| **V4 denoised** | Pseudo-noisy synthetic supervision (243 images) | Yes | 94.77 | 84.21 | -2.93 |
| **V4R denoised (real-trained)** | Real pairs (23,891 images) | No | 98.76 | 96.07 | +1.06 |
| **V4R denoised (real-trained)** | Real pairs (23,891 images) | Yes (V4 ECA) | **98.87** | **96.64** | **+1.17** |

### Key downstream observations

1. **Training regime dominates architecture alone.** The denoisers trained on pseudo-noisy synthetic data degraded downstream classification, even when image-quality metrics looked competitive.
2. **Real-aligned training is the decisive factor for downstream benefit.** The real-trained V4R models improved the classification pipeline over the raw baseline.
3. **V4R ECA is the best overall system-level result.** It achieved the strongest reported downstream Accuracy and Macro-F1 in the final thesis package.