# Reproducibility Guide

This document describes how to work with the public scripts and curated result
packages in this repository. The historical V4/V5 package, the corrected
robustness package, and the August 2026 endpoint-validity package have different
interpretation boundaries.

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

## 7. Regenerate historical figures

From the repository root:

```bash
python scripts/plot_downstream_v4v5.py --input results/v4v5_final/plots_data_accuracy_macrof1.csv --out figures
python scripts/plot_robustness.py --input results/v4v5_final/plots_data_robustness.csv --out figures
python scripts/plot_latency.py --input results/v4v5_final/latency_params_summary.csv --out figures --with-params
```

These commands reproduce the historical V4/V5 figures. The robustness plot is
based on the historical three-run pilot and must not be used as the current
independent-seed robustness result.

## 8. Corrected seed-aware protocol

The corrected revision uses seeds:

```text
42, 13, 37, 101, 202
```

Every native training/evaluation command must receive an explicit
`--seed <SEED>` argument. A compliant runner must seed Python, NumPy, PyTorch
CPU/CUDA, data splitting, additive noise, classifier training, and checkpoint
selection. Native stderr must be logged but only a non-zero process exit code
is considered failure.

The public, machine-independent run matrix is recorded in:

```text
experiments/manifests/corrected_seed5_public_manifest.csv
```

The corresponding result tables are in:

```text
results/revision_2026_corrected_robustness/
```

The full downstream evaluator remains in the separate classification
repository. Absolute Windows paths, datasets, checkpoints, and classifier
weights are intentionally excluded from this public package.

## 9. Validate the endpoint-validity package

The current downstream endpoint tables can be verified without the private
dataset or model checkpoints:

```bash
python scripts/validate_endpoint_validity_results.py
```

The validator checks:

- the 11,706-repeat endpoint closure;
- five-class support on all six boards;
- deterministic Raw/V4R/V5R view-lock summaries;
- the frozen F-only and F+I headline metrics;
- all 90 board/seed incremental deltas;
- all 90 leakage-audit rows;
- board-first and condition-balanced paired contrasts;
- the SHA-256 public artifact manifest.

The full sample-level tensors, logits, data manifests, checkpoints, and private
paths are intentionally not distributed in this repository.

## 10. Experiment manifests

A template manifest is provided at:

```text
experiments/manifests/TEMPLATE_run_manifest.yaml
```

Use it to record run provenance, training regime, evaluation settings, and output artifacts.

## 11. Interpretation boundary

This repository does **not** contain the full end-to-end pipeline by itself. The
pseudo-noisy generator and downstream classification code are maintained in
separate repositories.

Do not pool the corrected random-split five-seed sweep with the locked
board-grouped audit. The former estimates seed robustness under the original
in-distribution protocol; the latter estimates transfer to unseen physical
boards at seed 42.

The current nested endpoint adds a third estimand: image information conditional
on a matched frequency-only model. Its frozen absolute result is inconclusive,
while the later paired descriptive audit finds a consistent Raw advantage over
the denoised views. See `docs/CURRENT_CLAIM_BOUNDARIES_2026.md` before using any
of these results in a manuscript.
