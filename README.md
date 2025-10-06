# ESPI-DnCNN: Enhanced Vibration Mode Classification

A comprehensive pipeline for ESPI (Electronic Speckle Pattern Interferometry) vibration mode classification using deep learning denoising and enhanced feature engineering.

## 🎯 Key Results

- **Pattern-only classification:** 90.15% accuracy (69.91% macro-F1) - **34.85% improvement** over baseline
- **Hybrid classification:** 97.85% accuracy (95.15% macro-F1) - establishes upper performance bound
- **Robust validation:** LODO, LOBO, bootstrap CI, and negative controls confirm reliability

## 📊 Performance Comparison

| Method | Features | Accuracy | Macro-F1 | Improvement |
|--------|----------|----------|----------|-------------|
| **Baseline** | 10 nodal | 55.3% | 22.0% | - |
| **Pattern-only** | 16 enhanced | **90.15%** | **69.91%** | **+34.85%** |
| **Hybrid** | 22 + prior | **97.85%** | **95.15%** | **+42.55%** |

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv ESPI_VENV
ESPI_VENV\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Enhanced Classification
```bash
# Pattern-only vs Hybrid evaluation
python rf_evaluate_two_setups.py

# Robustness analysis
python rf_robustness_analysis.py

# LODO/LOBO validation
python rf_lodo_lobo.py

# Negative control tests
python rf_negative_control.py
```

### 3. View Results
Results are saved in `results/` directory:
- `report_pattern.json` - Pattern-only detailed results
- `report_hybrid.json` - Hybrid detailed results
- `confusion_matrix_*.png` - Confusion matrices
- `robustness_*.json` - Bootstrap confidence intervals

## 🔬 Methodology

### Fair Comparison Approach
To ensure fair evaluation without label leakage, we conducted two distinct experiments:

1. **Pattern-only:** Enhanced morphological/topological features without frequency information
2. **Hybrid:** Pattern features + frequency priors

### Enhanced Features
- **Symmetry features:** `hv_ratio`, `diag_ratio`
- **Nodal complexity:** Combined change measures
- **Normalized features:** `grad_cv`, `lapz`
- **Distance priors:** Distance to bin centers (hybrid only)

### Validation Strategy
- **StratifiedGroupKFold:** Dataset + frequency grouping
- **Bootstrap 95% CI:** 1000 resamples for confidence intervals
- **LODO/LOBO:** Cross-dataset and cross-frequency validation
- **Negative control:** Label shuffle validation

## 📁 Project Structure

```
ESPI-DnCNN/
├── run_espi_pipeline.py          # Main pipeline
├── batch_denoise_pytorch_v3.py   # Core denoising
├── phase_extract_fft.py          # Phase extraction
├── robust_align_irls.py          # IRLS alignment
├── espi_features_nodal.py        # Feature extraction
├── rf_evaluate_two_setups.py     # Enhanced evaluation
├── rf_robustness_analysis.py     # Robustness tests
├── rf_lodo_lobo.py              # LODO/LOBO validation
├── rf_negative_control.py        # Negative controls
├── feature_pack_aug.py          # Feature augmentation
├── results/                      # All results and visualizations
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## 🎓 Thesis Results

The enhanced feature engineering approach successfully improves ESPI vibration mode classification from 55.3% to 90.15% accuracy in a fair, leakage-free evaluation. The hybrid model achieves 97.85% accuracy when frequency information is available.

**Key Contributions:**
1. **Leakage-free evaluation** with clear pattern-only vs hybrid distinction
2. **Robust validation** through multiple cross-validation strategies  
3. **Significant improvement** (34.85%) through enhanced feature engineering
4. **Comprehensive analysis** with confidence intervals and stability measures

## 📊 Results Files

- `results/FINAL_THESIS_RESULTS_REPORT.md` - Comprehensive analysis report
- `results/compare_baseline_pattern_hybrid.json` - Performance comparison
- `results/robustness_*.json` - Bootstrap confidence intervals
- `results/lodo_results.json` - Leave-One-Dataset-Out analysis
- `results/lobo_results.json` - Leave-One-Bin-Out analysis
- `results/negative_control.json` - Label shuffle validation
- `results/feature_stability.json` - Cross-seed stability analysis

## 🔧 Dependencies

- Python 3.8+
- PyTorch 2.5.1+cu121
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- OpenCV

## 📄 License

This project is part of academic research. Please cite appropriately if used in your work.

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{espi_dncnn_2025,
  title={Enhanced ESPI Vibration Mode Classification using Deep Learning and Feature Engineering},
  author={[Your Name]},
  year={2025},
  institution={[Your Institution]}
}
```

---

*For detailed methodology and results, see `results/FINAL_THESIS_RESULTS_REPORT.md`*
