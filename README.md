# ESPI-DnCNN-ECA: Lightweight Denoising for ESPI Interferometry

A specialized PyTorch implementation of **DnCNN-Lite** enhanced with **ECA (Efficient Channel Attention)**, designed for denoising **Electronic Speckle Pattern Interferometry (ESPI)** images.

This repository contains both:
- the core denoising code and utilities, and
- a **curated V4/V5 thesis result package** (canonical CSV tables + plotting scripts) used for the final thesis conclusions.

---

## 🌟 Key Features

- **Lightweight Architecture:** Optimized DnCNN-Lite model suitable for CPU training and inference.
- **Physics-Aware Evaluation:** Supports REAL evaluation mode (single-shot noisy inputs vs averaged references) with proper metric calculation (**PSNR, SSIM, Edge-F1**).
- **ECA Attention:** Integrated Efficient Channel Attention blocks for feature refinement.
- **Tile-Based Inference:** Full-resolution denoising using tiled processing + Hann-window blending to avoid boundary artifacts.
- **Robust Metrics:** Includes PSNR, SSIM, and fringe-oriented Edge-F1 evaluation.
- **Hardware Agnostic:** Automatic CUDA/CPU support; mixed precision (AMP) when available.

---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.8+ and PyTorch.

---

## 🚀 Usage

### Training on Synthetic Data

Training requires a folder of clean reference images. The script can generate pseudo-noisy pairs on the fly.

```bash
python espi_dncnn_lite_eca.py \
    --clean-root /path/to/clean_images \
    --output-dir ./training_output \
    --epochs 50 \
    --batch-size 4 \
    --sigma-g 0.05 \
    --speckle 0.02
```

### Real Data Evaluation (Single-shot vs Averaged)

To evaluate on real experimental data:

```bash
python espi_dncnn_lite_eca.py \
    --clean-root /path/to/averaged_references \
    --real-noisy-root /path/to/single_shot_noisy \
    --resume ./checkpoints/best.pth \
    --val-ratio 0.0 \
    --epochs 0
```

### ONNX Export

```bash
python espi_dncnn_lite_eca.py ... --export-onnx model.onnx
```

---

## 📊 Thesis Final Results (V4/V5 Canonical Package)

The **final thesis conclusions** are based on the **V4/V5 canonical package** (see `results/v4v5_final/`).

### Main Findings (Final Thesis)

* **Training Data Dominance (Regime-dependent behavior):**
  The supervision regime matters more than the architecture variant. Denoisers trained on **pseudo-noisy (synthetic)** data can degrade downstream classification, while denoisers trained on **real-aligned pairs** improve downstream performance.

* **Optimal Architecture (V4R ECA):**
  The lightweight **V4R ECA** design (3 attention layers), when trained on real-aligned data, achieves the best overall downstream performance (**98.87% Accuracy, 96.64% Macro-F1**).

* **Robustness vs Complexity Trade-off:**
  The more aggressive **V5** design (7 attention layers, dual-pooling) incurs a large latency overhead (~**+360%**) and higher variability without consistent gains. **V4** remains the best balance of performance, robustness, and cost.

### Canonical thesis package (what to use)

* `results/v4v5_final/` — consolidated CSV tables (downstream metrics, robustness, latency)
* `scripts/` — minimal plotting scripts to reproduce thesis figures from the canonical CSV files
* `docs/THESIS_RESULTS_NOTES.md` — mapping of repository files to thesis chapter/appendix tables/figures

---

## 🧭 Legacy v3 / Pilot Results (Historical Context)

Earlier **v3** experiments (including initial ECA vs non-ECA ablations and pilot evaluation tracks) are preserved for **historical traceability** and research development context.

> These legacy findings are useful for understanding the project evolution, but **they do not represent the final thesis pipeline or final thesis conclusions**.

---

## 📈 Regenerate Thesis Figures (from canonical CSVs)

From the repository root:

```bash
python scripts/plot_downstream_v4v5.py --input results/v4v5_final/plots_data_accuracy_macrof1.csv --out figures
python scripts/plot_robustness.py --input results/v4v5_final/plots_data_robustness.csv --out figures
python scripts/plot_latency.py --input results/v4v5_final/latency_params_summary.csv --out figures --with-params
```

This generates PNG/SVG figures for:

* downstream Accuracy
* downstream Macro-F1
* robustness (mean±std error bars)
* latency (and optional parameter count)

---

## 📄 License

MIT License — see `LICENSE` for details.
