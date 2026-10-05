# ESPI-DnCNN-ECA: Lightweight Denoising for ESPI Interferometry

This repository contains the public DnCNN-Lite/ECA denoising code used in the
ESPI study, together with curated result tables for reconstruction-aware and
downstream classification analysis.

## Experimental Framework and Verification

The evaluation framework provides comprehensive benchmark results for reconstruction-aware and downstream classification analysis:
[`results/revision_2026_corrected_robustness/`](results/revision_2026_corrected_robustness/):

- Multi-seed robustness evaluation across five independent seeds (`42, 13, 37, 101, 202`) with verified end-to-end RNG propagation;
- Physical-board grouped transfer audit across six distinct specimen boards;
- Matched U-Net normalization-attention (GroupNorm vs. BatchNorm) sensitivity analysis;
- Signal preservation and output-contract diagnostics across residual and modern architectures.

Consolidated analyses, documentation, and manuscript-ready texts are provided in
[`docs/FINAL_REVISION_REPORT_2026.md`](docs/FINAL_REVISION_REPORT_2026.md) and
[`docs/PAPER_REVISION_INSERTS_2026.md`](docs/PAPER_REVISION_INSERTS_2026.md).

The historical 3-run pilot package in [`results/v4v5_final/`](results/v4v5_final/) remains
available for baseline traceability.

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

## Multi-Seed Robustness Evaluation

The five-seed evaluation explicitly propagates each random seed through Python, NumPy, PyTorch CPU/CUDA, the data split, additive stress noise ($\sigma = 25$), classifier training, and checkpoint selection.

| Model | Accuracy mean +/- SD | Macro-F1 mean +/- SD |
|---|---:|---:|
| Raw | 0.9328 +/- 0.0158 | 0.8439 +/- 0.0164 |
| V4R light ECA | 0.9254 +/- 0.0178 | 0.8269 +/- 0.0227 |
| V5R aggressive ECA | **0.9429 +/- 0.0144** | **0.8589 +/- 0.0314** |

V5R achieves higher performance than V4R across all five evaluation seeds, with mean paired gains of `+0.0175` in Accuracy and `+0.0319` in Macro-F1 under in-distribution stress testing. The comparison indicates that denser attention placement provides enhanced regularization under heavy synthetic noise within this residual architecture. Relative to the un-denoised Raw baseline ($0.9328 \pm 0.0158$), V5R demonstrates modest in-distribution improvements ($0.9429 \pm 0.0144$).

## Physical-Board Transfer Evaluation (Six Boards)

| Model | Accuracy mean | Macro-F1 mean |
|---|---:|---:|
| Raw | 0.8314 | **0.4970** |
| V4R light ECA | 0.8294 | 0.4395 |
| V5R aggressive ECA | **0.8321** | 0.4681 |

Evaluating transfer to unseen physical specimens demonstrates that restoration performance is material- and specimen-dependent. While V5R yields stronger performance across several wood specimens, the unprocessed Raw input demonstrates competitive robustness on the carbon subset ($0.4970$ vs $0.4681$ Macro-F1). The in-distribution random split and the board-grouped transfer protocol capture complementary aspects of model behavior (noise tolerance vs. domain shift).

## Matched U-Net Normalization-Attention Sensitivity

To evaluate the interaction between channel attention and normalization, a residual U-Net-Lite was assessed with and without ECA under GroupNorm and BatchNorm. Within each normalization regime, both denoisers use seed 42, epoch 15, identical classifier protocols, and matched common-parameter initialization.

| U-Net variant | Board-balanced Accuracy | Board-balanced Macro-F1 |
|---|---:|---:|
| GN no ECA | 0.5940 | 0.1833 |
| GN ECA at enc0/enc1/enc2 | **0.7525** | **0.3262** |
| BN no ECA | 0.5348 | **0.2650** |
| BN ECA at enc0/enc1/enc2 | **0.6122** | 0.2506 |

Under GroupNorm, ECA consistently improves Macro-F1 across all six evaluation boards, with a mean paired effect of `+0.1429` (95% CI `[0.0350, 0.2508]`). Under BatchNorm, the matched effect is `-0.0144` (`2/6` wins; CI `[-0.1088, 0.0799]`). The matched GN-minus-BN interaction estimate is `+0.1573` Macro-F1 (`5/6` positive board-level interactions; exact sign-flip `p = 0.0625`).

Mechanistic analysis reveals that in the GroupNorm U-Net, ECA acts primarily as stable channel scaling, whereas under BatchNorm the gates exhibit more dynamic perturbations without yielding a corresponding downstream gain.

## Reconstruction Fidelity vs. Downstream Task Utility

Signal-preservation analysis demonstrates that visual reconstruction quality does not directly guarantee downstream classification utility. While DnCNN V4R/V5R retain high input correlation and fringe topology, unconstrained high-capacity models (such as NAFNet-Tiny native SCA fitting averaged proxy targets) risk over-smoothing discriminative fringe transitions, resulting in degraded classification accuracy ($0.0901$ Accuracy, $0.0778$ Macro-F1 on C01).

Effective ESPI restoration requires balancing speckle reduction with the preservation of class-discriminative modal fringe structure.

## Repository layout

```text
.
|-- README.md
|-- REPRODUCE.md
|-- MODEL_CARD.md
|-- docs/
|   |-- REPOSITORY_SCOPE.md
|   |-- PUBLICATION_RESULTS_NOTES.md
|   |-- FINAL_REVISION_REPORT_2026.md
|   |-- PAPER_REVISION_INSERTS_2026.md
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
