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

## Mechanistic interpretation

The completed intervention audit does not support a universal statement that
normalization simply absorbs ECA. In the U-Net GroupNorm model, replacing each
sample-specific ECA gate by its calibration-set mean changes the output only
slightly (relative L2 `0.000469`, output correlation `0.99865`), which is
consistent with mostly stable channel scaling. BatchNorm retains more dynamic
gate effects (fixed-mean relative L2 `0.006075`, correlation `0.9434`) but does
not convert them into a stable downstream Macro-F1 gain.

The original DnCNN float-equivalence finding remains valid for that audited
configuration, but the absorption mechanism is architecture- and
normalization-specific rather than universal.

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
> exceed a noise-adapted Raw baseline. In an epoch-matched U-Net analysis, ECA
> improves Macro-F1 consistently under GroupNorm but not under BatchNorm,
> suggesting that its downstream utility depends on the surrounding
> normalization regime. The NAFNet negative control further demonstrates that
> reconstruction-optimized whitening may destroy downstream-discriminative ESPI
> structure.

## Claims to avoid

- Do not claim that the old three runs were independent seeds.
- Do not retain the old three-run p-value.
- Do not claim that denoising universally beats Raw.
- Do not claim that ECA always improves Accuracy.
- Do not claim that ECA universally stabilizes U-Net across normalization regimes.
- Do not generalize the DnCNN float-equivalence mechanism to every normalized residual network.
- Do not treat the six boards as six independent conventional replicates.
- Do not describe the seed-42 U-Net result as a five-seed robustness test.
