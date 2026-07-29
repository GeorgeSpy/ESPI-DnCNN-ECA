# 2026 corrected robustness and downstream revision

## Why this revision exists

The earlier robustness workflow was audited after reviewer concern about the
small number of runs. The audit found that the old orchestration did not prove
that `--seed` reached every native Python run. Consequently, the historical
three-run table cannot be treated as a valid independent seed estimate.

The experiment was replaced with a fresh, isolated, seed-aware five-seed
replication. All outputs were kept outside the historical result package, and
the old files remain unchanged for traceability.

## Corrected seed-aware replication

Seeds: `42, 13, 37, 101, 202`.

The corrected protocol controls:

- Python `random`;
- NumPy RNG;
- PyTorch CPU and CUDA RNGs;
- deterministic CuDNN behavior where applicable;
- data splitting;
- additive stress-noise realization;
- classifier initialization/training;
- seed-specific checkpoint selection.

Five-seed means are:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Raw | 0.9328 | 0.8439 |
| V4R light ECA | 0.9254 | 0.8269 |
| V5R aggressive ECA | 0.9429 | 0.8589 |

V5R exceeds V4R on all five seeds. The mean paired effect is `+0.0175`
Accuracy and `+0.0319` Macro-F1. With only five pairs, exact two-sided sign-flip
inference is coarse (`p = 0.0625`), so this should be described as a robust
effect-size pattern rather than definitive population-level confirmation.

V5R-minus-Raw effects are smaller (`+0.0101` Accuracy and `+0.0150`
Macro-F1) and their intervals cross zero.

## Board-grouped transfer audit

Six physical boards were evaluated using locked validation/test boards and a
frozen noise realization. Raw, V4R, and V5R have nearly identical mean
Accuracy. Raw has the highest board-balanced Macro-F1.

The material split is heterogeneous: Raw is stronger on the three carbon
boards, while V5R is more favorable on several wood boards. With three boards
per material, this remains an effect-size observation rather than confirmation
of a material interaction.

## Matched ECA ablation in U-Net

To isolate an ECA effect in another residual architecture, U-Net GN no-ECA and
U-Net GN ECA3 were compared at epoch 15 under common initialization and the
same downstream protocol.

| Board | ECA-minus-no-ECA Accuracy | ECA-minus-no-ECA Macro-F1 |
|---|---:|---:|
| C01 | +0.0004 | +0.0588 |
| C02 | -0.0250 | +0.0703 |
| C03 | +0.4292 | +0.1349 |
| W01 | +0.3938 | +0.3425 |
| W02 | +0.1664 | +0.1237 |
| W03 | -0.0138 | +0.1272 |

Macro-F1 improves on all six boards. The mean paired change is `+0.1429`, with
an exploratory interval `[0.0350, 0.2508]`, Cohen's `dz = 1.39`, and exact
sign-flip `p = 0.03125`. These calculations are descriptive because the folds
share training boards and only seed 42 was used.

Per-class analysis shows improvements for classes 0, 1, 3, and 4. Class 2
remains near zero, so ECA does not solve the full class-imbalance problem.

## Architecture/output-contract audit

DnCNN V5R retains higher input-output correlation and gradient content than the
tested U-Net and NAFNet variants. The U-Net variants produce near-white outputs
but ECA makes their downstream behavior substantially more stable. NAFNet-Tiny
native SCA produces an almost constant near-white output and collapses in C01
classification.

This demonstrates why reconstruction metrics alone are insufficient for ESPI
model selection. A powerful denoiser can fit the averaged proxy target while
removing the signal needed for classification.

## Paper-ready interpretation

Recommended wording:

> The corrected five-seed replication suggests that the aggressive V5R ECA
> configuration is more robust than the light V4R configuration under the
> original in-distribution stress protocol. In the locked leave-one-board-out
> audit, however, denoising gains are heterogeneous and do not consistently
> exceed a noise-adapted Raw baseline. A matched U-Net ablation shows a
> consistent board-level Macro-F1 improvement with ECA, suggesting that channel
> attention can reduce architecture-specific instability, while the NAFNet
> negative control demonstrates that reconstruction-optimized whitening may
> destroy downstream-discriminative ESPI structure.

## Claims to avoid

- Do not claim that the old three runs were independent seeds.
- Do not retain the old three-run p-value.
- Do not claim that denoising universally beats Raw.
- Do not claim that ECA always improves Accuracy.
- Do not treat the six boards as six independent conventional replicates.
- Do not describe the seed-42 U-Net result as a five-seed robustness test.
