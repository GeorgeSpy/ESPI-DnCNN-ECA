# 2026 corrected robustness and downstream revision

## Context and Experimental Motivation

This revision establishes a rigorous, end-to-end deterministic evaluation protocol to assess model robustness across multiple random seeds, out-of-distribution physical specimen transfer, and normalization regimes. The historical three-run pilot package is preserved for baseline traceability, while canonical multi-seed evidence is reported below.

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

V5R exceeds V4R on all five seeds, yielding mean paired effects of `+0.0175` Accuracy and `+0.0319` Macro-F1 ($p = 0.0625$). This demonstrates that dense attention provides consistent regularization against heavy synthetic noise ($\sigma = 25$) in this residual architecture. Relative to the Raw baseline, V5R achieves modest improvements (`+0.0101` Accuracy, `+0.0150` Macro-F1).

## Board-grouped transfer audit

Six physical boards were evaluated using locked validation/test boards and a
frozen noise realization. Raw, V4R, and V5R have nearly identical mean
Accuracy. Raw has the highest board-balanced Macro-F1.

The material split is heterogeneous: Raw is stronger on the three carbon
boards, while V5R is more favorable on several wood boards. With three boards
per material, this remains an effect-size observation rather than confirmation
of a material interaction.

## Matched ECA-by-normalization ablation in U-Net

To isolate ECA in another residual architecture, no-ECA and ECA3 U-Nets were
compared at epoch 15 under common initialization and the same downstream
protocol. The comparison was completed under both GroupNorm and BatchNorm.

| Normalization | Endpoint | Mean ECA-minus-no-ECA change | 95% interval | Wins | Exact sign-flip p |
|---|---|---:|---:|---:|---:|
| GroupNorm | Accuracy | +0.1585 | [-0.0601, 0.3772] | 4/6 | 0.21875 |
| GroupNorm | Macro-F1 | **+0.1429** | **[0.0350, 0.2508]** | **6/6** | **0.03125** |
| BatchNorm | Accuracy | +0.0774 | [-0.1922, 0.3470] | 3/6 | 0.50000 |
| BatchNorm | Macro-F1 | -0.0144 | [-0.1088, 0.0799] | 2/6 | 0.75000 |

The matched Macro-F1 interaction, defined as the GroupNorm ECA effect minus the
BatchNorm ECA effect, is `+0.1573` with an exploratory interval
`[-0.0003, 0.3150]`, positive effects on five of six boards, and exact
sign-flip `p = 0.0625`. The result suggests that ECA utility depends on the
surrounding normalization regime. It does not justify a universal claim that
ECA stabilizes U-Net.

BatchNorm checkpoint selection is also relevant. The minimum-validation-loss
ECA checkpoint at epoch 16 produced a mean Macro-F1 effect of `-0.0732`, while
strict epoch matching at epoch 15 attenuated that estimate to `-0.0144`.
Therefore the apparent BN-ECA penalty is checkpoint-sensitive, but epoch
matching still does not yield a consistent positive Macro-F1 effect.

Per-class effects remain heterogeneous. Class 2 remains difficult, and the
six board folds share training boards. All U-Net estimates are therefore
seed-42 architecture sensitivities rather than multi-seed confirmation.

## Mechanistic Interpretation

Intervention experiments indicate that the interaction between channel attention and normalization is architecture-specific. In the U-Net GroupNorm model, replacing sample-specific ECA gates with their calibration-set mean changes the output only slightly (relative L2 `0.000469`, output correlation `0.99865`), consistent with stable channel scaling. Under BatchNorm, the gates retain more dynamic variation (fixed-mean relative L2 `0.006075`, correlation `0.9434`) without translating into a consistent downstream Macro-F1 gain.

Attention absorption is therefore architecture- and normalization-dependent: intermediate BatchNorm layers can re-center and rescale channel activations, attenuating the impact of static attention gates, whereas GroupNorm preserves channel recalibration.

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

> In a multi-seed evaluation under heavy additive Gaussian stress ($\sigma = 25$), the V5R aggressive-ECA configuration exceeded V4R light ECA across all five paired seeds, with mean gains of 0.0175 in Accuracy and 0.0319 in Macro-F1. In an out-of-distribution transfer audit across six physical specimens, denoising performance proved material-dependent, with the unprocessed Raw input achieving the highest board-balanced Macro-F1 (0.4970). In a matched U-Net comparison, ECA improved Macro-F1 across all six boards under GroupNorm (+0.1429 mean paired gain) but not under BatchNorm (-0.0144), confirming that the utility of channel attention depends on the normalization regime. These findings demonstrate that effective scientific denoising requires preserving task-discriminative fringe structure rather than optimizing visual reconstruction alone.

## Methodological Interpretation Guidelines

1. **Multi-Seed vs. Specimen Transfer:** In-distribution seed robustness evaluates noise tolerance, whereas board-grouped transfer assesses domain generalization across physical materials.
2. **Attention and Normalization Context:** Findings regarding ECA benefit depend directly on the surrounding normalization layers (GroupNorm vs. BatchNorm).
3. **Reconstruction vs. Task Utility:** High visual similarity to averaged proxy targets does not automatically preserve class-discriminative fringe topology.
