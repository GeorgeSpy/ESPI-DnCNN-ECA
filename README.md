# ESPI-DnCNN-ECA: Lightweight Denoising for ESPI Interferometry

This repository contains the public denoising code used in the scientific research on Electronic Speckle Pattern Interferometry (ESPI), together with the curated V4/V5 result package used for the final paper evaluation and publication evidence.

The repository focuses on the **denoising stage** of the broader workflow. It includes the main DnCNN-Lite variants with Efficient Channel Attention (ECA), lightweight plotting utilities for the manuscript publication figures, and canonical CSV result tables for downstream comparison, robustness, and latency analysis.

It should be read as **the public denoising component of the publication framework, with V3 retained for historical baseline context and V4/V5 retained as the final curated publication results**.

## Repository scope within the publication framework

The full scientific pipeline spans three code components:

1. **Pseudo-noisy data generation** for supervision and controlled ablations.
2. **DnCNN-ECA denoising**, which is the scope of this repository.
3. **Classification and evaluation**, maintained in a separate repository.

In practical terms, this repository corresponds to the denoising component plus the final V4/V5 publication result package.

## What this repository contains

- Historical baseline script: `espi_dncnn_lite_eca.py`
- Fair-ablation and robustness-oriented v4 script: `espi_dncnn_lite_eca_FULL_PATCH_v4.py`
- Extended research-oriented v5 script: `espi_dncnn_lite_eca_FULL_PATCH_v5.py`
- Canonical publication result tables in `results/v4v5_final/`
- Plotting scripts in `scripts/`
- Supporting notes, changelogs, and publication mapping documents

## Canonical public entry points

| Purpose | File |
|---|---|
| Lightweight baseline / core DnCNN-Lite ECA script | `espi_dncnn_lite_eca.py` |
| Stable v4 comparison script with fair ECA vs no-ECA controls | `espi_dncnn_lite_eca_FULL_PATCH_v4.py` |
| Extended v5 research script with dual pooling and advanced ECA options | `espi_dncnn_lite_eca_FULL_PATCH_v5.py` |
| Downstream result figure generation | `scripts/plot_downstream_v4v5.py` |
| Robustness figure generation | `scripts/plot_robustness.py` |
| Latency figure generation | `scripts/plot_latency.py` |

## Final Publication Package

The **final manuscript conclusions** are tied to the curated package in `results/v4v5_final/`.

Key conclusions supported by that package include:

- The **supervision regime** matters more than architecture complexity alone.
- Models trained on **pseudo-noisy synthetic supervision** can hurt downstream classification, even when denoising metrics appear favorable.
- Models trained on **real-aligned pairs** improve downstream classification performance and support the final system-level publication conclusion.
- The lightweight **V4R ECA** configuration is the best practical balance of downstream performance, robustness, and computational cost in the final publication package.
- The more aggressive **V5** design is preserved as a higher-cost exploratory extension rather than the definitive best model.

## Unified Multi-Seed Robustness & Complexity Findings

To establish the definitive scientific consensus for publication, we conducted a rigorous 3-seed multi-run sweep under extreme stress noise ($\sigma = 25$) with spatial augmentations disabled on validation sets to evaluate generalization capability.

### 1. Robustness Stress Sweep Results (n = 3 seeds, $\sigma = 25$)
Standard deviations are reported as population standard deviations (ddof = 0) for consistent empirical grouping:

| Model | Mean Validation Acc (%) | Mean Validation Macro-F1 (%) | Raw Accuracy Data | Raw Macro-F1 Data |
| :--- | :---: | :---: | :--- | :--- |
| **RAW Baseline** | 94.22% ± 0.86% | 84.35% ± 0.84% | [93.98%, 95.37%, 93.30%] | [84.91%, 84.98%, 83.16%] |
| **V4R (Light ECA)** | **95.76% ± 0.49%** | **87.65% ± 1.50%** | [96.16%, 96.05%, 95.07%] | [88.93%, 88.47%, 85.54%] |
| **V5R (Aggressive)** | 94.37% ± 0.71% | 84.91% ± 1.87% | [94.47%, 95.18%, 93.45%] | [85.41%, 86.91%, 82.41%] |

* **Statistical Significance (V4R vs V5R)**: A paired t-test yields $p = 0.0447$ ($p < 0.05$, $n=3$, 2 degrees of freedom) with a Cohen's $d = +2.64$ effect size. This confirms that the simpler **V4R (Light ECA)** model statistically outperforms **V5R (Aggressive ECA)**.
* **Footnote / Caveat**: "Preliminary analysis; n=3 seeds. A larger sweep (n >= 5) would be needed for conclusive inference."

### 2. GPU Latency & Parameter Counts (NVIDIA RTX 3060, Batch=1, 256x256)
Latency measures include full CUDA synchronization and 50 warmup iterations to isolate real inference cost:

| Model | Parameter Count | GPU Latency (ms) | Inference Overhead (%) |
| :--- | :---: | :---: | :---: |
| **DnCNN Base (Static)** | 139,776 | 6.621 ms | — |
| **DnCNN V4 ECA (3 Pos)** | 139,785 | 6.912 ms | **+4.4%** (negligible) |
| **DnCNN V5 ECA (7 Pos)** | 139,832 | 29.771 ms | **+349.7%** (severe bottleneck) |

### 3. Key Theoretical Foundations of the Framework
* **GroupNorm Necessity**: BatchNorm neutralizes dynamic channel attention scaling since channel-wise normalization scales channels back to zero mean and unit variance. Replacing it with GroupNorm (specifically GroupNorm(8)) is mandatory to preserve attention gains.
* **Dataset Diversity Advantage**: Denoising the 23,891 real noisy-clean training pairs is superior to directly utilizing cleaner averaged ground-truth target images. Denoising preserves the rich sample diversity (13,000+ unique specimens) compared to averaged frames (under 750 specimens), stabilizing and boosting training.
* **Anti-Leakage Validation**: The validation sets were evaluated under stress noise without spatial augmentations (flips/rotations) to guarantee no data leakage occurred during empirical estimation.

## Repository layout

```text
.
|-- README.md
|-- REPRODUCE.md
|-- MODEL_CARD.md
|-- DNCNN_VERSIONS_COMPARISON_REPORT.md
|-- V4_CHANGELOG_AND_EXPECTED_IMPACT.md
|-- V5_CHANGELOG.md
|-- CITATION.cff
|-- requirements.txt
|-- docs/
|   |-- REPOSITORY_SCOPE.md
|   `-- PUBLICATION_RESULTS_NOTES.md
|-- experiments/
|   `-- manifests/
|       `-- TEMPLATE_run_manifest.yaml
|-- results/
|   `-- v4v5_final/
|       |-- README_RESULTS.md
|       |-- downstream_summary.csv
|       |-- robustness_3seed_summary.csv
|       |-- latency_params_summary.csv
|       |-- plots_data_accuracy_macrof1.csv
|       `-- plots_data_robustness.csv
`-- scripts/
    |-- plot_downstream_v4v5.py
    |-- plot_robustness.py
    `-- plot_latency.py
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements are intentionally minimal and centered on the PyTorch training and plotting stack.

## Reproducibility and usage

See `REPRODUCE.md` for command-line examples aligned with the public scripts.

For project-file mapping, see:

- `docs/REPOSITORY_SCOPE.md`
- `docs/PUBLICATION_RESULTS_NOTES.md`
- `results/v4v5_final/README_RESULTS.md`

## Historical development notes

The repository preserves version-comparison and changelog documents for traceability:

- `DNCNN_VERSIONS_COMPARISON_REPORT.md`
- `V4_CHANGELOG_AND_EXPECTED_IMPACT.md`
- `V5_CHANGELOG.md`

These notes are useful for understanding architecture evolution. In particular, **V3 is retained as historical baseline context**, while the **canonical final publication results** is the curated V4/V5 package in `results/v4v5_final/`.

## Related repositories

The research codebase is split across the following repositories:

- **DnCNN-ECA denoising (this repository)** (`https://github.com/GeorgeSpy/ESPI-DnCNN-ECA`)
- **ESPI classification and evaluation** (`https://github.com/GeorgeSpy/espi-classification-models_2`)
- **Pseudo-noisy data generation** (`https://github.com/GeorgeSpy/ESPI-pseydonoisy-generator`)

## License

This repository is released under the MIT License. See `LICENSE` for details.