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

## 📊 Thesis Final Results (V4/V5 Canonical Package)

The final conclusions of the thesis are based on the **V4 / V5 canonical package**. Essential findings from the ablation and downstream analysis include:

*   **Training Data Dominance**: The source of supervision matters more than the architecture variant. Denoisers trained on pseudo-noisy (synthetic) data *degrade* downstream classification (Accuracy −1.3% to −2.9%). Denoisers trained on real-aligned pairs *improve* it (+1.1%).
*   **Optimal Architecture (V4R ECA)**: The lightweight V4 ECA (3 attention layers), when trained on real data, achieves the highest downstream performance (**98.87% Accuracy**, **96.64% Macro-F1**).
*   **Robustness vs Complexity**: The aggressive V5 design (7 attention layers with dual-pooling) incurs a significant latency overhead (~+360%) and higher variability without corresponding performance gains. The V4 design remains the optimal trade-off.

For curated canonical data, reporting tables, and plotting scripts, please see:
*   [**`results/v4v5_final/`**](results/v4v5_final/) - Consolidated `CSV` tables for downstream metrics, robustness, and latency.
*   [**`scripts/`**](scripts/) - Minimal plotting utilities to reproduce thesis figures from the CSV tables.

*(Legacy exploratory findings (v3) are preserved elsewhere for historical traceability but do not represent the final thesis pipeline).*

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
