# Final revision report: corrected robustness, ECA sensitivity, and downstream utility

**Project:** ESPI-DnCNN-ECA

**Revision package:** 2026 corrected robustness and architecture-sensitivity audit

**Status:** Complete for publication with the claim boundaries stated below
**Canonical public evidence:** `results/revision_2026_corrected_robustness/`

## 1. Executive conclusion

The completed experiments do not support the broad claim that denoising always
improves ESPI classification. They support a narrower and more useful result:

> Denoising can improve downstream classification when the denoiser preserves
> class-discriminative ESPI structure. In the matched residual U-Net analysis,
> ECA is beneficial under GroupNorm but not under BatchNorm, suggesting that its
> downstream utility depends on the surrounding normalization regime.

The corrected five-seed replication supersedes the earlier three-run robustness
summary for inferential purposes. Under the corrected in-distribution stress
protocol, V5R aggressive ECA is consistently stronger than V4R light ECA. On
unseen physical boards, however, Raw remains highly competitive and has the
highest board-balanced Macro-F1. The effect of denoising is therefore dependent
on architecture, board, material, and evaluation regime.

## 2. Evidence hierarchy and protocol correction

### 2.1 Withdrawn inferential use of the historical three-run result

The old orchestration did not prove that `--seed` reached every native Python
training and evaluation command. The old result is retained for traceability,
but only as a pilot diagnostic. Its historical p-value must not be used.

Required paper statement:

> The earlier three-run robustness result is treated as a pilot diagnostic
> because the audit showed that seed propagation was incomplete. The corrected
> seed-aware five-seed replication supersedes it for inferential purposes.

### 2.2 Corrected five-seed replication

The corrected run uses seeds `42, 13, 37, 101, 202`. Each seed is explicitly
propagated to Python, NumPy, PyTorch CPU/CUDA, the data split, additive stress
noise, classifier initialization/training, and seed-specific checkpoint
selection. Native stdout and stderr are logged separately, and native commands
fail only on a non-zero process exit code.

### 2.3 Protocols that must remain separate

Three completed protocols estimate different quantities and must not be pooled:

1. the five-seed random-split/in-distribution stress replication;
2. the seed-42 locked six-board transfer audit;
3. the seed-42 matched U-Net ECA-by-normalization sensitivity analysis.

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

V5R is higher than V4R on every corrected seed. With only five non-zero pairs,
the exact two-sided sign-flip test is necessarily coarse. The result is best
reported as a consistent effect-size pattern, not as definitive population
confirmation.

Relative to Raw, V5R gains only `+0.0101` Accuracy and `+0.0150` Macro-F1 on
average. Both paired intervals cross zero. V4R is below Raw on four of five
seeds. Thus the corrected sweep supports the V5R-over-V4R ranking within this
protocol, but not a universal denoising-over-Raw claim.

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

The board folds share training boards, so their intervals are exploratory and
must not be described as six fully independent conventional replicates.

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
`p = 0.0625`. This is evidence that the ECA response is normalization-dependent,
not confirmation of a universal interaction.

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

## 7. Supported paper claims

The evidence supports the following statements:

1. The corrected five-seed replication suggests that V5R aggressive ECA is more
   robust than V4R light ECA under the original in-distribution stress protocol.
2. Denoising does not consistently outperform a noise-adapted Raw baseline on
   unseen physical boards.
3. A matched U-Net ablation suggests that the downstream response to ECA depends
   on normalization: it is favorable under GroupNorm but not under BatchNorm.
4. Reconstruction-oriented whitening can remove downstream-discriminative ESPI
   structure; model selection must include downstream evaluation.
5. Generalization varies by board and material, with W02 and C03 serving as
   important stress cases.

## 8. Claims that must not be made

- Do not call the historical three runs independent seeds.
- Do not reuse the historical three-run p-value.
- Do not claim that denoising universally improves classification.
- Do not claim that ECA always improves Accuracy.
- Do not claim that ECA universally stabilizes U-Net.
- Do not generalize the DnCNN float-equivalence/absorption mechanism to every
  normalized residual architecture.
- Do not interpret V5R versus V4R as a pure presence/absence ECA ablation;
  both models use ECA and differ in attention density/configuration.
- Do not describe the U-Net seed-42 result as a five-seed ECA confirmation.
- Do not pool the random-split, locked-board, and matched-U-Net protocols.

## 9. Paper-ready consolidated wording

> We audited the earlier robustness protocol and replaced it with a corrected
> seed-aware five-seed replication. Under the corrected in-distribution stress
> protocol, the V5R aggressive-ECA configuration exceeded V4R light ECA on all
> five paired seeds, with mean differences of 0.0175 in Accuracy and 0.0319 in
> Macro-F1. The advantage over Raw was smaller and uncertain. In a locked
> six-board transfer audit, denoising gains were heterogeneous and Raw achieved
> the highest board-balanced Macro-F1. In a matched seed-42 U-Net analysis, ECA
> improved Macro-F1 on all six boards under GroupNorm but not under BatchNorm,
> yielding a positive GroupNorm-minus-BatchNorm interaction estimate. These
> findings suggest that ECA utility depends on normalization and support
> conditional downstream utility rather than a universal denoising advantage.

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

## 11. Completion decision and next step

The current revision is complete for a paper that uses the cautious claims in
Sections 7 and 9. No additional architecture search is required before
publishing those claims.

A further experiment is required only if the paper must make a stronger causal
claim about ECA. The highest-value next experiment remains a matched DnCNN
GroupNorm ablation, not another unrelated modern denoiser:

- arms: no ECA, light ECA `[0,1,2]`, aggressive ECA
  `[0,1,2,3,6,10,14]`;
- seeds: `42, 13, 37, 101, 202`;
- identical data, initialization policy, epochs, checkpoint rule, and evaluator;
- Raw retained as an evaluation-only reference;
- primary endpoint: Macro-F1;
- report both corrected in-distribution stress and locked-board transfer without
  pooling them.

The existing U-Net result justifies this experiment, but does not need to delay
the current revision if its wording remains explicitly exploratory.

## 12. Final repository status

The canonical revision evidence is stored under
`results/revision_2026_corrected_robustness/`. The historical
`results/v4v5_final/` package remains unchanged for traceability. This report,
the revision results, and their validator form the final public evidence bundle.
