# ESPI-DnCNN-ECA: Lightweight Denoising for ESPI Interferometry

This repository contains the public DnCNN-Lite/ECA denoising code used in the
ESPI study, together with curated result tables for reconstruction-aware and
downstream classification analysis.

## Current evidence: endpoint-validity update (August 2026)

The current public interpretation is defined by the six-board, nested
incremental-value analysis in
[`results/endpoint_validity_2026/`](results/endpoint_validity_2026/) and the
corresponding report in
[`docs/ENDPOINT_VALIDITY_UPDATE_2026.md`](docs/ENDPOINT_VALIDITY_UPDATE_2026.md).

The frozen endpoint contains 11,706 unique real physical repeats from six
boards and five classes. Raw, V4R, and V5R classifier-input views were locked
one-to-one with 100% coverage and bitwise deterministic reproduction. All 90
outer prediction tables and 90 inner out-of-fold tables passed the leakage and
coverage audit.

| Arm | Board-balanced Macro-F1 | Increment over matched F-only | Positive boards |
|---|---:|---:|---:|
| F-only | 0.816656 | -- | -- |
| Raw F+I | **0.850356** | **+0.033699** | 4/6 |
| V4R F+I | 0.800619 | -0.016037 | 2/6 |
| V5R F+I | 0.774716 | -0.041940 | 1/6 |

The preregistered absolute decision is
`DNCNN_INCREMENTAL_VALUE_INCONCLUSIVE`: Raw clears the mean-effect threshold
but not the required 5/6-board consistency rule. A subsequent board-first
paired audit, using no new fitting or inference, found Raw minus V4R =
`+0.049737` Macro-F1 (6/6 boards) and Raw minus V5R = `+0.075639` (5/6 boards).
Its frozen descriptive decision is `RAW_CONDITIONAL_ADVANTAGE_CONSISTENT`.

The supported conclusion is therefore cautionary: under this endpoint, Raw
retains more conditional task-relevant image information beyond excitation
frequency than either denoised view. The analysis does not establish that this
residual information is fringe morphology, displacement information, or any
specific physical observable.

Current claim boundaries are collected in
[`docs/CURRENT_CLAIM_BOUNDARIES_2026.md`](docs/CURRENT_CLAIM_BOUNDARIES_2026.md).

## Earlier 2026 reproducibility correction

The previous three-run robustness result is retained only as a historical pilot.
A protocol audit found incomplete seed propagation, so it must not be treated as
an independent three-seed estimate and its old p-value must not be used in a
revised paper.

The earlier corrected robustness evidence is in
[`results/revision_2026_corrected_robustness/`](results/revision_2026_corrected_robustness/):

- corrected seed-aware five-seed retraining/evaluation with seeds
  `42, 13, 37, 101, 202`;
- a locked six-board grouped transfer audit;
- a matched U-Net ECA-by-normalization sensitivity analysis;
- an output-contract audit including a NAFNet-Tiny failure-mode control.

Its original consolidated interpretation and reviewer-facing text are
documented in
[`docs/FINAL_REVISION_REPORT_2026.md`](docs/FINAL_REVISION_REPORT_2026.md).
Paste-ready English manuscript and reviewer-response text is provided in
[`docs/PAPER_REVISION_INSERTS_2026.md`](docs/PAPER_REVISION_INSERTS_2026.md).
Those documents are retained for audit history and are superseded wherever
they conflict with the August endpoint-validity update.

The historical package in [`results/v4v5_final/`](results/v4v5_final/) remains
available for traceability but is no longer the canonical robustness evidence.

## Repository scope

The complete research workflow spans three components:

1. pseudo-noisy and real-aligned supervision data generation;
2. DnCNN-ECA denoising, which is the scope of this repository;
3. downstream classification and evaluation, maintained separately.

Raw datasets, checkpoints, classifier weights, and machine-specific experiment
outputs are intentionally not included.

Sample-level logits and the 35,118 locked Raw/V4R/V5R tensors are likewise not
published here. The public package contains the aggregate, board, seed, class,
condition-sensitivity, leakage, and provenance tables needed to inspect the
reported decisions.

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

## Matched U-Net ECA-by-normalization sensitivity

A residual U-Net-Lite was evaluated with and without ECA under GroupNorm and
BatchNorm. Within each normalization regime, both denoisers use seed 42, epoch
15, the same classifier protocol, and matched common-parameter initialization.

| U-Net variant | Board-balanced Accuracy | Board-balanced Macro-F1 |
|---|---:|---:|
| GN no ECA | 0.5940 | 0.1833 |
| GN ECA at enc0/enc1/enc2 | **0.7525** | **0.3262** |
| BN no ECA | 0.5348 | **0.2650** |
| BN ECA at enc0/enc1/enc2 | **0.6122** | 0.2506 |

Under GroupNorm, ECA improves Macro-F1 on all six boards, with a mean paired
effect of `+0.1429` (exploratory 95% interval `[0.0350, 0.2508]`). Under
BatchNorm, the matched effect is `-0.0144` (`2/6` wins; interval
`[-0.1088, 0.0799]`). The matched GN-minus-BN interaction estimate is
`+0.1573` Macro-F1 (`5/6` positive board-level interactions; exact sign-flip
`p = 0.0625`). This suggests that ECA utility is normalization-dependent; it
does not establish a universal U-Net benefit.

The mechanistic audit also rejects a universal "normalization absorbs ECA"
explanation. In this U-Net, GroupNorm mostly converts ECA into stable channel
scaling, whereas BatchNorm permits more dynamic perturbations without a stable
downstream gain. The earlier float-equivalence finding should therefore remain
specific to the audited DnCNN configuration.

## Reconstruction is not downstream utility

The output-contract audit shows that DnCNN V4R/V5R preserve substantially more
of the original ESPI input than the tested U-Net and NAFNet configurations.
NAFNet-Tiny native SCA fits nearly white averaged proxy targets but collapses on
C01 downstream evaluation (`0.0901` Accuracy, `0.0778` Macro-F1).

The earlier output-contract audit motivated the conditional statement below,
but the August endpoint-validity study now supplies the stronger operational
boundary:

> Reconstruction quality, architecture modernity, and in-distribution
> classification scores do not by themselves establish conditional utility on
> unseen boards. In the frozen nested endpoint, Raw retains more conditional
> predictive information than V4R or V5R.

## Repository layout

```text
.
|-- README.md
|-- REPRODUCE.md
|-- MODEL_CARD.md
|-- docs/
|   |-- REPOSITORY_SCOPE.md
|   |-- PUBLICATION_RESULTS_NOTES.md
|   |-- ENDPOINT_VALIDITY_UPDATE_2026.md
|   |-- CURRENT_CLAIM_BOUNDARIES_2026.md
|   |-- PROTOCOL_V2_AUDIT_STATUS_2026.md
|   |-- FINAL_REVISION_REPORT_2026.md
|   |-- PAPER_REVISION_INSERTS_2026.md
|   `-- REVISION_RESULTS_2026.md
|-- experiments/
|   `-- manifests/
|       |-- TEMPLATE_run_manifest.yaml
|       `-- corrected_seed5_public_manifest.csv
|-- results/
|   |-- v4v5_final/                         # historical package
|   |-- revision_2026_corrected_robustness/ # earlier corrected evidence
|   |-- endpoint_validity_2026/             # current downstream evidence
|   `-- protocol_audit_2026/                # compact protocol decision registry
|-- scripts/
|   |-- validate_endpoint_validity_results.py
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
