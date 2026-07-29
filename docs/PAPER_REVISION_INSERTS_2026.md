# Paper revision inserts: reviewer-facing English text

This document provides paste-ready text for the SETN revision and a longer
follow-up paper. The wording follows the corrected seed audit, the locked
six-board transfer experiment, the matched U-Net ECA-by-normalization study,
and the mechanistic intervention audit.

## 1. Positioning and contribution

The contribution should be framed as a diagnostic study rather than a new
state-of-the-art denoising architecture:

> This work studies when channel attention remains functionally useful in an
> ESPI denoising-to-classification pipeline. The contribution is a controlled
> diagnostic analysis of supervision realism, normalization, attention
> placement, and downstream signal preservation. We do not claim a universally
> superior denoiser. Instead, we identify conditions under which ECA is neutral,
> beneficial, or insufficient for preserving class-discriminative fringe
> structure.

## 2. Domain bridge for a general machine-learning audience

Suggested introduction paragraph:

> Electronic speckle pattern interferometry (ESPI) records deformation as dense
> fringe patterns whose spacing, curvature, discontinuities, and spatial
> topology encode the vibration mode of a physical structure. Single-shot
> measurements are corrupted by speckle, illumination variation, sensor noise,
> and acquisition-specific artifacts. The downstream task is not generic image
> restoration: a classifier must retain subtle fringe geometry that separates
> vibration-mode classes. A denoiser can therefore improve pixel-level
> appearance while simultaneously removing information required for
> classification. This makes ESPI a useful case study for evaluating denoisers
> by downstream utility rather than reconstruction metrics alone.

Recommended figure layout:

1. real single-shot ESPI input;
2. averaged proxy target;
3. representative DnCNN, U-Net, and modern-denoiser outputs;
4. enlarged fringe regions showing preserved or lost topology;
5. downstream prediction and class label.

Suggested caption:

> ESPI denoising is constrained by downstream fringe preservation. Averaged
> proxy targets reduce acquisition noise but may suppress contrast or topology
> that remains discriminative for vibration-mode classification. The diagnostic
> pipeline therefore reports both reconstruction behavior and locked downstream
> classification performance.

## 3. Corrected robustness protocol

Paste-ready methods text:

> We audited the earlier robustness protocol and replaced it with a corrected
> seed-aware five-seed replication. Seeds 42, 13, 37, 101, and 202 were
> propagated explicitly to Python, NumPy, PyTorch CPU and CUDA generators, data
> splitting, additive stress-noise generation, classifier training, and
> seed-specific checkpoint selection. Native command outputs were isolated by
> run, and test metrics were reported only after validation-based model
> selection. The earlier three-run result is retained only as a pilot diagnostic
> because seed propagation was incomplete; the corrected five-seed replication
> supersedes it for inferential purposes.

Paste-ready results text:

> Under the corrected in-distribution stress protocol, V5R aggressive ECA
> exceeded V4R light ECA on all five paired seeds. Mean V5R-minus-V4R effects
> were +0.0175 Accuracy and +0.0319 Macro-F1. Because the exact two-sided
> sign-flip test is coarse with five non-zero pairs, these results are reported
> as a consistent effect-size pattern rather than definitive population-level
> confirmation. V5R's mean advantage over the Raw baseline was smaller and its
> paired interval crossed zero.

## 4. Modern-backbone response: matched U-Net analysis

Recommended table:

| Normalization | U-Net condition | Accuracy mean | Macro-F1 mean |
|---|---|---:|---:|
| GroupNorm | no ECA | 0.5940 | 0.1833 |
| GroupNorm | ECA at enc0/enc1/enc2 | 0.7525 | 0.3262 |
| BatchNorm | no ECA | 0.5348 | 0.2650 |
| BatchNorm | ECA at enc0/enc1/enc2 | 0.6122 | 0.2506 |

Paste-ready results text:

> We additionally evaluated ECA in a residual U-Net-Lite to determine whether
> the diagnostic finding was specific to DnCNN-Lite. Within each normalization
> regime, the no-ECA and ECA models used common-parameter initialization, seed
> 42, epoch 15, and the same locked six-board classifier protocol. Under
> GroupNorm, ECA improved Macro-F1 on all six boards (mean paired effect
> +0.1429, exploratory 95% CI [0.0350, 0.2508]). Under BatchNorm, the matched
> Macro-F1 effect was -0.0144 (95% CI [-0.1088, 0.0799]) with improvements on
> only two of six boards. The matched GroupNorm-minus-BatchNorm interaction
> estimate was +0.1573 (95% CI [-0.0003, 0.3150]; five of six positive
> board-level interactions; exact sign-flip p=0.0625). These findings suggest
> that ECA utility depends on the surrounding normalization regime rather than
> being a universal property of the U-Net backbone.

Checkpoint-selection sensitivity should be disclosed:

> The minimum-validation-loss BatchNorm ECA checkpoint occurred at epoch 16
> and yielded a mean Macro-F1 effect of -0.0732. Fixing both BatchNorm models at
> epoch 15 attenuated this estimate to -0.0144. We therefore treat the magnitude
> of the BN-ECA ranking as checkpoint-sensitive and use the strict epoch-matched
> analysis for mechanistic interpretation.

## 5. Mechanistic interpretation of ECA absorption

The original mechanism must be narrowed rather than generalized:

> Float-level equivalence in the audited DnCNN configuration demonstrates that
> ECA can become functionally neutral in a normalized residual denoiser, but the
> U-Net intervention audit shows that this is not a universal absorption
> mechanism. In the GroupNorm U-Net, replacing sample-specific ECA gates by
> their calibration-set mean caused only a small output change (relative L2
> 0.000469; output correlation 0.99865), consistent with mostly stable channel
> scaling. BatchNorm retained more dynamic gate effects (relative L2 0.006075;
> correlation 0.9434), yet these effects did not produce a stable downstream
> Macro-F1 gain. Thus, the presence of dynamic attention is not sufficient for
> downstream utility, and the functional role of ECA depends on both backbone
> and normalization.

Recommended mechanistic figure panels:

1. gate multiplier distributions for GN and BN ECA models;
2. output change after replacing gates by calibration means;
3. output change after forcing gates to one;
4. normalization affine-parameter distributions;
5. board-level ECA effects under GN and BN.

## 6. Downstream utility and modern-denoiser control

Paste-ready discussion text:

> Reconstruction quality did not reliably predict classification utility. The
> DnCNN variants preserved substantially more input correlation and gradient
> structure than the tested U-Net and NAFNet configurations. The NAFNet-Tiny
> negative control fitted the averaged proxy target but produced near-white
> outputs and collapsed on the recorded C01 downstream evaluation. This result
> should not be interpreted as evidence that modern denoisers are intrinsically
> inferior; it demonstrates that a powerful reconstruction model can exploit a
> supervision target that is misaligned with the downstream signal-preservation
> objective.

## 7. Statistical limitations

Paste-ready limitations text:

> The corrected DnCNN robustness experiment uses five independent training
> seeds. The U-Net normalization analysis instead uses six held-out physical
> boards at one training seed. Because board folds overlap in their training
> sets, their t intervals and exact sign-flip tests are exploratory effect-size
> diagnostics rather than conventional independent-replicate inference. We
> therefore use “suggests” and “effect-size estimate” for the U-Net findings and
> avoid claims of universal ECA benefit, harm, or causal interaction.

## 8. Reviewer-response summary

### Reviewer 1: inaccessible ESPI context

Addressed by the domain-bridge paragraph and a new visual explanation of fringe
geometry, averaged proxy supervision, and the downstream vibration-mode task.

### Reviewer 1: outdated architecture baseline

Addressed by the matched residual U-Net analysis and the NAFNet-Tiny
failure-mode control. The results show that the central diagnostic issue is not
restricted to DnCNN-Lite, while avoiding an unsupported claim that the tested
models define the current denoising state of the art.

### Reviewer 1: unproven absorption mechanism

Addressed by gate-replacement interventions, gate-to-one interventions, gate
distribution analysis, and normalization affine-parameter analysis. The revised
claim is architecture-specific: DnCNN can absorb ECA, whereas U-Net GN mostly
uses stable scaling and U-Net BN retains dynamic but downstream-inconsistent
attention.

### Reviewers 1 and 2: weak statistics

Addressed by the corrected five-seed DnCNN replication, explicit withdrawal of
the old three-run inferential claim, paired effect sizes, sample SDs, confidence
intervals, and exact sign-flip diagnostics. The seed-42 U-Net analysis remains
clearly labeled exploratory.

### Reviewer 2: missing experimental detail

The revised methods should explicitly report dataset counts, physical-board
splits, classifier architecture, acquisition setup, averaging procedure,
epochs, optimizer, learning rate, batch size, seed propagation, checkpoint
selection, validation/test roles, and the locked stress-noise protocol.

## 9. Final claim boundary

Recommended concluding statement:

> The corrected experiments support a conditional downstream-utility claim:
> denoising can improve ESPI classification when class-discriminative fringe
> structure is preserved. ECA is neither universally neutral nor universally
> beneficial. Its effect varies with supervision, backbone, normalization,
> checkpoint selection, and physical-board regime. This diagnostic perspective,
> rather than a claim of architectural state of the art, is the central
> contribution of the study.
