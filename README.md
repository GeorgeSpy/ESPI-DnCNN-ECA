# ESPI-DnCNN-ECA: Lightweight Denoising for ESPI Interferometry

This repository contains the public denoising code used in the thesis work on Electronic Speckle Pattern Interferometry (ESPI), together with the curated V4/V5 result package used for the final thesis interpretation.

The repository focuses on the **denoising stage** of the broader workflow. It includes the main DnCNN-Lite variants with Efficient Channel Attention (ECA), lightweight plotting utilities for the final thesis figures, and canonical CSV result tables for downstream comparison, robustness, and latency analysis.

## Repository scope within the thesis

The full thesis spans three code components:

1. **Pseudo-noisy data generation** for supervision and controlled ablations.
2. **DnCNN-ECA denoising**, which is the scope of this repository.
3. **Classification and evaluation**, maintained in a separate repository.

In practical terms, this repository corresponds to the denoising component plus the final V4/V5 thesis result package.

## What this repository contains

- Baseline denoising script: `espi_dncnn_lite_eca.py`
- Fair-ablation and robustness-oriented v4 script: `espi_dncnn_lite_eca_FULL_PATCH_v4.py`
- Extended research-oriented v5 script: `espi_dncnn_lite_eca_FULL_PATCH_v5.py`
- Canonical thesis result tables in `results/v4v5_final/`
- Plotting scripts in `scripts/`
- Supporting notes, changelogs, and thesis mapping documents

## Canonical public entry points

| Purpose | File |
|---|---|
| Lightweight baseline / core DnCNN-Lite ECA script | `espi_dncnn_lite_eca.py` |
| Stable v4 comparison script with fair ECA vs no-ECA controls | `espi_dncnn_lite_eca_FULL_PATCH_v4.py` |
| Extended v5 research script with dual pooling and advanced ECA options | `espi_dncnn_lite_eca_FULL_PATCH_v5.py` |
| Downstream result figure generation | `scripts/plot_downstream_v4v5.py` |
| Robustness figure generation | `scripts/plot_robustness.py` |
| Latency figure generation | `scripts/plot_latency.py` |

## Final thesis package

The **final thesis conclusions** are tied to the curated package in `results/v4v5_final/`.

Key conclusions supported by that package include:

- The **supervision regime** matters more than architecture complexity alone.
- Models trained on **pseudo-noisy synthetic supervision** can hurt downstream classification.
- Models trained on **real-aligned pairs** improve downstream classification performance.
- The lightweight **V4R ECA** configuration gives the best overall balance of downstream performance, robustness, and cost.
- The more aggressive **V5** design increases latency substantially without delivering consistently better overall behavior.

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
|   `-- THESIS_RESULTS_NOTES.md
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

For thesis-file mapping, see:

- `docs/REPOSITORY_SCOPE.md`
- `docs/THESIS_RESULTS_NOTES.md`
- `results/v4v5_final/README_RESULTS.md`

## Historical development notes

The repository preserves version-comparison and changelog documents for traceability:

- `DNCNN_VERSIONS_COMPARISON_REPORT.md`
- `V4_CHANGELOG_AND_EXPECTED_IMPACT.md`
- `V5_CHANGELOG.md`

These notes are useful for understanding architecture evolution, but the **canonical final thesis evidence** is the curated package in `results/v4v5_final/`.

## Related repositories

The thesis codebase is split across the following repositories:

- **DnCNN-ECA denoising (this repository)** (`https://github.com/GeorgeSpy/ESPI-DnCNN-ECA`)
- **ESPI classification and evaluation** (`https://github.com/GeorgeSpy/espi-classification-models_2`)
- **Pseudo-noisy data generation** (`https://github.com/GeorgeSpy/ESPI-pseydonoisy-generator`)

## License

This repository is released under the MIT License. See `LICENSE` for details.
