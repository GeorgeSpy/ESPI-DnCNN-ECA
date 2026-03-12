# Reproducibility Guide

This document describes how to work with the public scripts and the curated thesis result package in this repository. It is intentionally aligned with the current file layout and public entry points.

## 1. Environment setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

On Windows, activate the environment with either:

```powershell
.\.venv\Scripts\Activate.ps1
```

or:

```bat
.\.venv\Scripts\activate.bat
```

## 2. Main scripts and roles

The repository exposes three main model scripts:

- `espi_dncnn_lite_eca.py`: lightweight baseline public script
- `espi_dncnn_lite_eca_FULL_PATCH_v4.py`: fair-ablation and thesis-oriented v4 script
- `espi_dncnn_lite_eca_FULL_PATCH_v5.py`: extended v5 script with more aggressive ECA options

Use the baseline script when you want the simplest public entry point. Use v4/v5 when you need the thesis-era comparison controls.

## 3. Example baseline training run

```bash
python espi_dncnn_lite_eca.py \
  --clean-root path/to/clean_images \
  --output-dir outputs/baseline_run \
  --epochs 50 \
  --batch-size 4 \
  --sigma-g 0.05 \
  --speckle 0.02 \
  --tensorboard
```

## 4. Example v4 training run

```bash
python espi_dncnn_lite_eca_FULL_PATCH_v4.py \
  --clean-root path/to/clean_images \
  --output-dir outputs/v4_eca \
  --device cpu \
  --epochs 50 \
  --batch-size 2 \
  --use-eca \
  --gn-groups 0 \
  --sigma-g 0.02 \
  --speckle 0.2 \
  --tensorboard
```

For a fair no-ECA comparison in v4, replace `--use-eca` with `--no-eca`.

## 5. Example v5 training run

```bash
python espi_dncnn_lite_eca_FULL_PATCH_v5.py \
  --clean-root path/to/clean_images \
  --output-dir outputs/v5_eca \
  --device cpu \
  --epochs 50 \
  --batch-size 2 \
  --use-eca \
  --eca-use-maxpool \
  --eca-multi-scale \
  --eca-preset shallow3 \
  --tensorboard
```

## 6. Resume and evaluation-oriented runs

All three main scripts support checkpoint-based workflows through `--resume`. The v4 and v5 scripts additionally expose stricter resume controls and real-evaluation scheduling options.

Representative pattern:

```bash
python espi_dncnn_lite_eca_FULL_PATCH_v4.py \
  --clean-root path/to/clean_images \
  --real-noisy-root path/to/real_noisy \
  --output-dir outputs/v4_eval \
  --resume path/to/checkpoint.pth \
  --device cpu
```

The exact input directory structure remains project-specific and is not bundled in the public repository.

## 7. Regenerate thesis figures from canonical CSVs

From the repository root:

```bash
python scripts/plot_downstream_v4v5.py --input results/v4v5_final/plots_data_accuracy_macrof1.csv --out figures
python scripts/plot_robustness.py --input results/v4v5_final/plots_data_robustness.csv --out figures
python scripts/plot_latency.py --input results/v4v5_final/latency_params_summary.csv --out figures --with-params
```

These commands reproduce the public result figures from the canonical V4/V5 package.

## 8. Experiment manifests

A template manifest is provided at:

```text
experiments/manifests/TEMPLATE_run_manifest.yaml
```

Use it to record run provenance, training regime, evaluation settings, and output artifacts.

## 9. Scope clarification

This repository does **not** contain the full end-to-end thesis pipeline by itself. The pseudo-noisy generator and the downstream classification code are maintained in separate repositories. The public material here should therefore be interpreted as the **denoising component plus curated final V4/V5 result evidence**.