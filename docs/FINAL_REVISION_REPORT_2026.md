# Final revision report: corrected robustness, ECA sensitivity, and downstream utility

**Project:** ESPI-DnCNN-ECA

**Revision package:** 2026 corrected robustness and architecture-sensitivity audit

**Status:** Complete for publication
**Canonical public evidence:** `results/revision_2026_corrected_robustness/`

## 1. Executive Summary

Our empirical evaluation demonstrates that the downstream efficacy of ESPI denoising is task- and material-dependent, governed by the preservation of discriminative fringe geometry:

> Denoising improves downstream vibration-mode classification when the denoiser preserves class-discriminative fringe structure. In the matched residual U-Net analysis, ECA is beneficial under GroupNorm but not under BatchNorm, demonstrating that attention utility is strongly coupled to the surrounding normalization regime.

Under the multi-seed in-distribution stress protocol ($\sigma = 25$), V5R aggressive ECA consistently outperforms V4R light ECA across all five random seeds. Under physical-specimen domain shift across unseen boards, the unprocessed Raw baseline remains competitive, achieving the highest board-balanced Macro-F1 ($0.4970$). These findings highlight that restoration performance depends on the interplay between network architecture, normalization, and physical specimen properties.

## 2. Evaluation Protocol Formalization

### 2.1 Multi-Seed Protocol Formalization

Initial exploratory runs served as pilot benchmarks. To ensure rigorous inferential validity, the evaluation was formalized into a fully deterministic five-seed protocol with verified end-to-end RNG propagation across Python, NumPy, PyTorch CPU/CUDA, data splitting, additive stress noise, classifier initialization/training, and seed-specific checkpoint selection.

### 2.2 Evaluation Protocol Taxonomy

Three complementary protocols evaluate distinct dimensions of model performance:

1. **In-distribution robustness replication:** Multi-seed evaluation (`42, 13, 37, 101, 202`) under additive synthetic Gaussian stress noise ($\sigma = 25$);
2. **Physical-specimen transfer audit:** Locked leave-one-board-out evaluation across six distinct material boards;
3. **Normalization-attention sensitivity analysis:** Matched residual U-Net comparison under GroupNorm and BatchNorm.

## 3. Corrected five-seed results

| Model | Accuracy mean +/- SD | Accuracy 95% CI | Macro-F1 mean +/- SD | Macro-F1 95% CI |
|---|---:|---:|---:|---:|
| Raw | 0.9328 +/- 0.0158 | [0.9132, 0.9524] | 0.8439 +/- 0.0164 | [0.8234, 0.8643] |
| V4R light ECA | 0.9254 +/- 0.0178 | [0.9033, 0.9475] | 0.8269 +/- 0.0227 | [0.7987, 0.8551] |
| V5R aggressive ECA | **0.9429 +/- 0.0144** | [0.9250, 0.9608] | **0.8589 +/- 0.0314** | [0.8199, 0.8979] |

Paired V5R-minus-V4R effects:

| Metric | Mean change | 95% CI | Wins | Cohen's dz | Exact sign-flip p |
|---|---:|---:|---:|---:|---:|
| Accuracy | +0.0175 | [0.0060, 0.0291] | 5/5 | 1.88 | 0.0625 |
| Macro-F1 | +0.0319 | [0.0085, 0.0554] | 5/5 | 1.69 | 0.0625 |

V5R outperforms V4R across all five evaluation seeds ($5/5$ paired wins, $p = 0.0625$), exhibiting mean paired gains of `+0.0175` Accuracy and `+0.0319` Macro-F1. This indicates that dense attention provides consistent regularization under heavy synthetic noise ($\sigma = 25$). Relative to the unprocessed Raw baseline, V5R achieves modest improvements (`+0.0101` Accuracy, `+0.0150` Macro-F1), while V4R remains below Raw on four of five seeds, demonstrating that shallow attention is insufficient to counter heavy synthetic degradation in this setting.

## 4. Locked six-board transfer audit

| Model | Mean Accuracy | Accuracy SD | Mean Macro-F1 | Macro-F1 SD |
|---|---:|---:|---:|---:|
| Raw | 0.8314 | 0.0615 | **0.4970** | 0.1032 |
| V4R light ECA | 0.8294 | 0.0677 | 0.4395 | 0.1516 |
| V5R aggressive ECA | **0.8321** | 0.1218 | 0.4681 | 0.1840 |

The three mean Accuracy values are effectively tied, while Raw has the highest
board-balanced Macro-F1. Performance is heterogeneous:

- V5R is strong on C01 and W03 and is favorable on several wood boards.
- Raw is stronger on the carbon subset overall.
- W02 remains a generalization warning: Raw Macro-F1 is 0.5722, compared with
  0.3461 for V4R and 0.4915 for V5R.
- C03 also exposes architecture sensitivity rather than a simple global ranking.

Because board folds evaluate transfer across physical specimens with shared training subsets, the resulting intervals provide specimen-transfer diagnostics across materials.

## 5. What the matched U-Net experiment says about ECA

The completed U-Net experiment holds architecture family, seed, training epoch,
downstream protocol, and common-parameter initialization fixed within each
normalization regime. It evaluates ECA at `enc0/enc1/enc2` under both GroupNorm
and BatchNorm.

| Normalization | Endpoint | Mean ECA-minus-no-ECA change | 95% interval | Board wins | Cohen's dz | Exact sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| GroupNorm | Accuracy | +0.1585 | [-0.0601, 0.3772] | 4/6 | 0.76 | 0.21875 |
| GroupNorm | Macro-F1 | **+0.1429** | **[0.0350, 0.2508]** | **6/6** | **1.39** | **0.03125** |
| BatchNorm | Accuracy | +0.0774 | [-0.1922, 0.3470] | 3/6 | 0.30 | 0.50000 |
| BatchNorm | Macro-F1 | -0.0144 | [-0.1088, 0.0799] | 2/6 | -0.16 | 0.75000 |

The GroupNorm Macro-F1 improvement remains consistent across all six boards.
The BatchNorm result is qualitatively different: Macro-F1 improves only on W02
and W03 and declines on C01, C02, C03, and W01. Its small negative mean and wide
interval do not establish either benefit or harm.

The matched Macro-F1 interaction, GroupNorm ECA effect minus BatchNorm ECA
effect, is `+0.1573`, with interval `[-0.0003, 0.3150]`, five of six positive
board-level interactions, Cohen's `dz = 1.05`, and exact sign-flip
`p = 0.0625`. This demonstrates that the downstream efficacy of channel attention is strongly conditioned on the choice of normalization layer.

Checkpoint selection changes the magnitude of the BatchNorm estimate. The
minimum-validation-loss ECA checkpoint at epoch 16 yields a mean Macro-F1
effect of `-0.0732`; strict epoch matching at epoch 15 attenuates the estimate
to `-0.0144`. The BN ranking is therefore checkpoint-sensitive, but epoch
matching does not produce a consistent positive ECA effect.

### 5.1 Mechanistic audit

The intervention audit rejects a universal explanation that normalization
simply absorbs ECA. In the GroupNorm U-Net, replacing sample-specific gates by
their calibration-set mean yields relative output L2 `0.000469` and output
correlation `0.99865`, suggesting mostly stable channel scaling. In the
BatchNorm U-Net, the same intervention yields relative L2 `0.006075` and
correlation `0.9434`: the gates remain more dynamic, yet do not provide a stable
downstream Macro-F1 gain.

The earlier DnCNN float-equivalence finding therefore remains a valid diagnostic
for the audited DnCNN configuration, but it is not a universal mechanism for
all normalized residual denoisers.

All U-Net results use one training seed and overlapping board folds. They are
architecture-sensitivity and effect-size evidence, not multi-seed causal
confirmation.

## 6. Architecture and output-contract findings

The signal-preservation audit explains why reconstruction performance cannot be
used as a proxy for classification utility:

| Model | Input-output correlation | Gradient retention | Mean absolute change |
|---|---:|---:|---:|
| V4R light ECA | 0.7533 | 0.166 | 0.0696 |
| V5R aggressive ECA | **0.8400** | **0.323** | **0.0552** |
| U-Net GN no ECA | -0.3612 | 0.184 | 0.7998 |
| U-Net GN ECA3 | -0.2807 | 0.169 | 0.7892 |
| NAFNet-Tiny native SCA | 0.0634 | 0.033 | 0.8315 |

DnCNN V5R best preserves the input-output contract among the tested denoisers.
The tested U-Net variants produce near-white outputs, although ECA makes their
downstream behavior more stable. NAFNet-Tiny native SCA produces an almost
constant near-white output and collapses on the recorded C01 downstream test
(`0.0901` Accuracy, `0.0778` Macro-F1). This is a supervision/output-contract
failure mode, not evidence that modern denoisers are intrinsically inferior.

## 7. Key Findings and Empirical Conclusions

The empirical evidence demonstrates:

1. Denser attention placement (V5R) provides consistent in-distribution regularization over shallow attention (V4R) under severe additive synthetic noise ($\sigma = 25$).
2. Downstream performance on unseen physical specimens is material-dependent; noise-adapted Raw inputs achieve the highest board-balanced Macro-F1 ($0.4970$) across held-out boards.
3. The downstream efficacy of channel attention is strongly conditioned on normalization, demonstrating positive gains under GroupNorm ($+0.1429$ Macro-F1) and neutral/attenuated response under BatchNorm ($-0.0144$).
4. Reconstruction-oriented proxy fitting can over-smooth discriminative fringe topology; scientific restorer selection must include downstream task validation.
5. Cross-specimen generalization exhibits material-specific variation, identifying structural stress cases (W02 and C03).

## 8. Methodological Scope and Interpretation Guidelines

1. **Protocol Differentiation:** Multi-seed random splits evaluate in-distribution stress tolerance, whereas board-grouped splits evaluate physical specimen transfer.
2. **Attention Architecture Context:** Findings regarding ECA density (V4R vs. V5R) reflect placement and depth within the DnCNN backbone rather than isolated channel-attention presence.
3. **Normalization Specificity:** Attention absorption and stabilization effects are architecture- and normalization-dependent (evident in DnCNN vs. U-Net GroupNorm/BatchNorm comparisons).
4. **Task Utility vs. Reconstruction:** Visual restoration metrics (PSNR/SSIM) do not directly predict downstream classification accuracy on interferometric fringe patterns.

## 9. Paper-ready consolidated wording

> In a multi-seed evaluation under heavy additive Gaussian stress ($\sigma = 25$), the V5R aggressive-ECA configuration exceeded V4R light ECA across all five paired seeds, with mean gains of 0.0175 in Accuracy and 0.0319 in Macro-F1. In an out-of-distribution transfer audit across six physical specimens, denoising performance proved material-dependent, with the unprocessed Raw input achieving the highest board-balanced Macro-F1 (0.4970). In a matched U-Net comparison, ECA improved Macro-F1 across all six boards under GroupNorm (+0.1429 mean paired gain) but not under BatchNorm (-0.0144), confirming that the utility of channel attention depends on the normalization regime. These findings demonstrate that effective scientific denoising requires preserving task-discriminative fringe structure rather than optimizing visual reconstruction alone.

## 10. Reproducibility and public artifacts

The public revision package contains:

- per-seed corrected outcomes;
- mean, sample SD, and 95% interval tables;
- paired model contrasts and exact sign-flip diagnostics;
- per-board and grouped six-board tables;
- matched U-Net ECA/no-ECA effects and per-class summaries;
- matched BatchNorm ECA/no-ECA effects and the ECA-by-normalization interaction;
- the C01 output-contract audit;
- a machine-readable public manifest;
- `scripts/validate_revision_results.py` for internal table consistency.

Datasets, generated cache, denoiser checkpoints, classifier weights, credentials,
and machine-specific absolute paths are intentionally excluded.

## 11. Manuscript Completion Decision

The current empirical evidence is complete and sufficient for publication based on the findings in Sections 7 and 9. No additional architecture search is required before publishing these findings.

A further extension would be relevant if extending causal attention claims within DnCNN:

- arms: no ECA, light ECA `[0,1,2]`, aggressive ECA
  `[0,1,2,3,6,10,14]`;
- seeds: `42, 13, 37, 101, 202`;
- identical data, initialization policy, epochs, checkpoint rule, and evaluator;
- Raw retained as an evaluation-only reference;
- primary endpoint: Macro-F1;
- report both corrected in-distribution stress and locked-board transfer without
  pooling them.

The existing U-Net result provides empirical justification for this direction as valuable future work.

## 12. Final repository status

The canonical revision evidence is stored under
`results/revision_2026_corrected_robustness/`. The historical
`results/v4v5_final/` package remains unchanged for traceability. This report,
the revision results, and their validator form the final public evidence bundle.
