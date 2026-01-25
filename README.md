# ESPI-DnCNN-ECA: Lightweight Denoising for ESPI Interferometry

A specialized PyTorch implementation of **DnCNN-Lite** enhanced with **ECA (Efficient Channel Attention)**, designed specifically for denoising Electronic Speckle Pattern Interferometry (ESPI) images.

## 🌟 Key Features

*   **Lightweight Architecture**: Optimized `DnCNNLite` model suitable for CPU training and inference.
*   **Physics-Aware Evaluation**: Supports "REAL" evaluation mode, comparing single-shot noisy inputs against averaged clean references with proper metric calculation (PSNR, SSIM, EdgeF1).
*   **ECA Attention**: Integrated Efficient Channel Attention blocks for feature refinement.
*   **Tile-Based Inference**: Full-resolution processing using tile-based denoising with Hann window blending to avoid artifacts.
*   **Robust Metrics**: Includes custom implementations for PSNR, SSIM, and Fringe Edge F1 score.
*   **Hardware Agnostic**: Automatic switching between CUDA and CPU; supports mixed-precision (AMP).

## 🛠 Installation

```bash
pip install -r requirements.txt
```

*(Requires Python 3.8+ and PyTorch)*

## 🚀 Usage

### Training on Synthetic Data

Training requires a folder of clean reference images. The script automatically generates pseudo-noisy pairs on the fly.

```bash
python espi_dncnn_lite_eca.py \
    --clean-root /path/to/clean_images \
    --output-dir ./training_output \
    --epochs 50 \
    --batch-size 4 \
    --sigma-g 0.05 \
    --speckle 0.02
```

### Real Data evaluation

To evaluate on real experimental data (Single-shot vs Averaged pairs):

```bash
python espi_dncnn_lite_eca.py \
    --clean-root /path/to/averaged_references \
    --real-noisy-root /path/to/single_shot_noisy \
    --resume ./checkpoints/best.pth \
    --val-ratio 0.0 \
    --epochs 0
```

### ONNX Export

Export the trained model for portable inference:

```bash
python espi_dncnn_lite_eca.py ... --export-onnx model.onnx
```

## 📊 Performance (v3 Findings)

Recent A/B testing revealed:
*   **Denoising Benefit**: Significant improvement from Noisy → Denoised (e.g., Random Forest classification accuracy increased from 13% to 62%).
*   **Architecture**: The `Lite` version performs identically to heavier counterparts for this specific task, validating the efficiency focus.
*   **Attention**: While ECA is implemented, ablation studies showed practically identical performance to vanilla DnCNN for this specific physics-driven data, suggesting the core CNN capabilities are the primary driver of quality.

## 📄 License

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
