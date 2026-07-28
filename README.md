# ESPI-DnCNN-ECA: Lightweight Denoising for ESPI Interferometry

This repository contains the public DnCNN-Lite/ECA denoising code used in the
ESPI study, together with curated result tables for reconstruction-aware and
downstream classification analysis.

## 2026 reproducibility correction

The previous three-run robustness result is retained only as a historical pilot.
A protocol audit found incomplete seed propagation, so it must not be treated as
an independent three-seed estimate and its old p-value must not be used in a
revised paper.

The current revision evidence is in
[`results/revision_2026_corrected_robustness/`](results/revision_2026_corrected_robustness/):

- corrected seed-aware five-seed retraining/evaluation with seeds
  `42, 13, 37, 101, 202`;
- a locked six-board grouped transfer audit;
- a matched U-Net GroupNorm no-ECA/ECA sensitivity analysis;
- an output-contract audit including a NAFNet-Tiny failure-mode control.

The historical package in [`results/v4v5_final/`](results/v4v5_final/) remains
available for traceability but is no longer the canonical robustness evidence.

## Repository scope

The complete research workflow spans three components:

1. pseudo-noisy and real-aligned supervision data generation;
2. DnCNN-ECA denoising, which is the scope of this repository;
3. downstream classification and evaluation, maintained separately.

Raw datasets, checkpoints, classifier weights, and machine-specific experiment
outputs are intentionally not included.

## Public model entry points

| Purpose | File |
|---|---|
| Lightweight historical baseline | `espi_dncnn_lite_eca.py` |
| V4 fair-ablation/light-ECA model | `espi_dncnn_lite_eca_FULL_PATCH_v4.py` |
| V5 aggressive/research ECA model | `espi_dncnn_lite_eca_FULL_PATCH_v5.py` |

The corrected V4R configuration uses GroupNorm(8) with ECA positions
`[0,1,2]`. The corrected V5R configuration uses GroupNorm(8) with positions
`[0,1,2,3,6,10,14]`.

## Corrected five-seed result

The corrected replication explicitly propagates each seed through Python,
NumPy, PyTorch CPU/CUDA, the data split, additive stress noise, classifier
training, and checkpoint selection.

| Model | Accuracy mean +/- SD | Macro-F1 mean +/- SD |
|---|---:|---:|
| Raw | 0.9328 +/- 0.0158 | 0.8439 +/- 0.0164 |
| V4R light ECA | 0.9254 +/- 0.0178 | 0.8269 +/- 0.0227 |
| V5R aggressive ECA | **0.9429 +/- 0.0144** | **0.8589 +/- 0.0314** |

V5R is higher than V4R on all five seeds. Mean paired V5R-minus-V4R effects
are `+0.0175` Accuracy and `+0.0319` Macro-F1. This suggests that the aggressive
V5R configuration is more robust than V4R under the original in-distribution
protocol. It does not isolate the presence of ECA because both models contain
ECA and differ in attention density/configuration.

V5R is only modestly higher than Raw on average, with intervals crossing zero;
therefore the corrected sweep does not establish a universal denoising benefit.

## Locked six-board transfer result

| Model | Accuracy mean | Macro-F1 mean |
|---|---:|---:|
| Raw | 0.8314 | **0.4970** |
| V4R light ECA | 0.8294 | 0.4395 |
| V5R aggressive ECA | **0.8321** | 0.4681 |

The board-grouped audit does not show a universal transfer advantage for either
denoiser. Results are board- and material-dependent: V5R is more favorable on
several wood boards, whereas Raw is stronger on the carbon subset. This audit
must not be pooled with the random-split five-seed result because the protocols
estimate different quantities.

## Matched U-Net ECA sensitivity

A residual U-Net-Lite with GroupNorm was evaluated at the same epoch, seed,
classifier protocol, and common-parameter initialization.

| U-Net variant | Board-balanced Accuracy | Board-balanced Macro-F1 |
|---|---:|---:|
| GN no ECA | 0.5940 | 0.1833 |
| GN ECA at enc0/enc1/enc2 | **0.7525** | **0.3262** |

Macro-F1 improves on all six boards, with a mean paired effect of `+0.1429`
(exploratory 95% interval `[0.0350, 0.2508]`). Accuracy improves on four of six
boards. The result suggests that ECA can reduce architecture-specific
instability, but it is currently a seed-42 architecture sensitivity rather than
a five-seed U-Net confirmation.

## Reconstruction is not downstream utility

The output-contract audit shows that DnCNN V4R/V5R preserve substantially more
of the original ESPI input than the tested U-Net and NAFNet configurations.
NAFNet-Tiny native SCA fits nearly white averaged proxy targets but collapses on
C01 downstream evaluation (`0.0901` Accuracy, `0.0778` Macro-F1).

The supported conclusion is therefore conditional:

> Denoising can improve downstream classification when it preserves
> class-discriminative ESPI structure. ECA may stabilize a residual architecture,
> but reconstruction quality or architecture modernity alone does not guarantee
> downstream benefit.

## Repository layout

```text
.
|-- README.md
|-- REPRODUCE.md
|-- MODEL_CARD.md
|-- docs/
|   |-- REPOSITORY_SCOPE.md
|   |-- PUBLICATION_RESULTS_NOTES.md
|   `-- REVISION_RESULTS_2026.md
|-- experiments/
|   `-- manifests/
|       |-- TEMPLATE_run_manifest.yaml
|       `-- corrected_seed5_public_manifest.csv
|-- results/
|   |-- v4v5_final/                         # historical package
|   `-- revision_2026_corrected_robustness/ # current revision evidence
|-- scripts/
|-- espi_dncnn_lite_eca.py
|-- espi_dncnn_lite_eca_FULL_PATCH_v4.py
`-- espi_dncnn_lite_eca_FULL_PATCH_v5.py
```

## Installation and reproduction

```bash
pip install -r requirements.txt
```

See [`REPRODUCE.md`](REPRODUCE.md) for script usage and the interpretation
boundary of the public result tables.

## Related repositories

- DnCNN-ECA denoising: <https://github.com/GeorgeSpy/ESPI-DnCNN-ECA>
- ESPI classification/evaluation: <https://github.com/GeorgeSpy/espi-classification-models_2>
- Pseudo-noisy generation: <https://github.com/GeorgeSpy/ESPI-pseydonoisy-generator>

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
