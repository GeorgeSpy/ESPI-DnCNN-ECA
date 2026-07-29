# Final revision report: corrected robustness, ECA sensitivity, and downstream utility

**Project:** ESPI-DnCNN-ECA

**Revision package:** 2026 corrected robustness and architecture-sensitivity audit

**Status:** Complete for publication with the claim boundaries stated below
**Canonical public evidence:** `results/revision_2026_corrected_robustness/`

## 1. Executive conclusion

The completed experiments do not support the broad claim that denoising always
improves ESPI classification. They support a narrower and more useful result:

> Denoising can improve downstream classification when the denoiser preserves
> class-discriminative ESPI structure. ECA appears to reduce instability in a
> matched residual U-Net comparison, but its causal effect still requires a
> multi-seed matched ablation for definitive confirmation.

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
3. the seed-42 matched U-Net GN no-ECA versus ECA sensitivity analysis.

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

The matched U-Net experiment holds architecture family, GroupNorm, seed,
training epoch, downstream protocol, and common-parameter initialization fixed.
Only the ECA condition changes.

| Endpoint | Mean ECA-minus-no-ECA change | 95% interval | Board wins | Cohen's dz | Exact sign-flip p |
|---|---:|---:|---:|---:|---:|
| Accuracy | +0.1585 | [-0.0601, 0.3772] | 4/6 | 0.76 | 0.21875 |
| Macro-F1 | **+0.1429** | **[0.0350, 0.2508]** | **6/6** | **1.39** | **0.03125** |

Macro-F1 improves on every board. Accuracy is less consistent. The largest
Accuracy changes occur where no-ECA is unstable: C03 rises from 0.0441 to
0.4733, and W01 rises from 0.4925 to 0.8862. The C03 result should therefore be
described as stabilization of a collapsed baseline, not as proof of a typical
43-point ECA gain.

Per-class board-balanced F1 improves for classes 0, 1, 3, and 4. Class 2 remains
near zero (`0.0109` without ECA, `0.0155` with ECA), so ECA does not solve the
class-imbalance or class-separability problem.

This is the cleanest completed evidence that ECA itself matters, but it uses one
training seed and overlapping board folds. It is supportive evidence, not a
multi-seed causal confirmation.

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
3. A matched U-Net ablation suggests that ECA can reduce architecture-specific
   instability, particularly for Macro-F1.
4. Reconstruction-oriented whitening can remove downstream-discriminative ESPI
   structure; model selection must include downstream evaluation.
5. Generalization varies by board and material, with W02 and C03 serving as
   important stress cases.

## 8. Claims that must not be made

- Do not call the historical three runs independent seeds.
- Do not reuse the historical three-run p-value.
- Do not claim that denoising universally improves classification.
- Do not claim that ECA always improves Accuracy.
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
> the highest board-balanced Macro-F1. A matched seed-42 U-Net GroupNorm
> ablation improved Macro-F1 on all six boards when ECA was enabled, suggesting
> that channel attention can reduce architecture-specific instability. These
> findings support conditional downstream utility rather than a universal
> denoising advantage.

## 10. Reproducibility and public artifacts

The public revision package contains:

- per-seed corrected outcomes;
- mean, sample SD, and 95% interval tables;
- paired model contrasts and exact sign-flip diagnostics;
- per-board and grouped six-board tables;
- matched U-Net ECA/no-ECA effects and per-class summaries;
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
claim about ECA. The highest-value next experiment is a matched DnCNN GroupNorm
ablation, not another unrelated modern denoiser:

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
