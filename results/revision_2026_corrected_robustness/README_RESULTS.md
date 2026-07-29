# Corrected robustness and architecture-sensitivity results (2026 revision)

This package supersedes the historical three-run robustness summary for
inferential purposes. It contains lightweight, publication-ready tables only;
datasets, checkpoints, classifier weights, and machine-specific paths are not
included.

## Protocols kept separate

### Corrected five-seed in-distribution sweep

- Seeds: `42, 13, 37, 101, 202`
- Stress noise: additive Gaussian `sigma = 25/255`
- Denoisers are retrained per seed.
- The downstream classifier and stress-noise RNG receive the same explicit run
  seed.
- Model selection uses validation Macro-F1; test metrics are reported once.
- V4R: GroupNorm(8), light ECA at positions `[0,1,2]`.
- V5R: GroupNorm(8), aggressive ECA at positions
  `[0,1,2,3,6,10,14]`.

### Locked six-board transfer audit

- Boards: `C01, C02, C03, W01, W02, W03`.
- One physical board is held out for test and a different board for validation.
- Seed 42 and a frozen additive-noise realization are used.
- Board folds overlap in their training boards, so board-level intervals and
  sign-flip tests are exploratory effect-size diagnostics.

### Matched U-Net ECA-by-normalization sensitivity

- Residual U-Net-Lite with GroupNorm and BatchNorm arms.
- Same seed, epoch 15, classifier protocol, and common-parameter
  initialization.
- Comparison: no ECA versus ECA at encoder stages `enc0/enc1/enc2`.
- This is a matched architecture sensitivity analysis, not a five-seed U-Net
  robustness estimate.

## Main results

The corrected five-seed sweep suggests that V5R is more robust than V4R under
the original in-distribution protocol. Mean paired V5R-minus-V4R effects are
`+0.0175` Accuracy and `+0.0319` Macro-F1, with V5R higher on all five seeds.
The two-sided exact sign-flip p-value is `0.0625`, the smallest attainable with
five non-zero paired effects; wording should therefore remain cautious.

In the locked six-board audit, Raw, V4R, and V5R have nearly tied mean Accuracy,
while Raw has the highest board-balanced Macro-F1. This does not support a
universal denoising advantage on unseen physical boards.

Under GroupNorm, the matched U-Net ablation improves Macro-F1 on all six boards.
The mean paired ECA-minus-no-ECA effect is `+0.1429` (exploratory 95% CI
`[0.0350, 0.2508]`). Under BatchNorm, the matched Macro-F1 effect is `-0.0144`
(`2/6` wins; CI `[-0.1088, 0.0799]`). The GN-minus-BN interaction estimate is
`+0.1573` (`5/6` positive board-level interactions; exact sign-flip
`p = 0.0625`). The result suggests normalization-dependent ECA utility and must
not be presented as a universal U-Net benefit.

The NAFNet-Tiny native-SCA negative control reaches only `0.0901` Accuracy and
`0.0778` Macro-F1 on C01. An output-contract audit shows near-white outputs and
very low gradient retention. This is interpreted as a supervision/output-
contract failure mode, not as evidence that modern denoisers are intrinsically
inferior.

## Files

- `corrected_seed5_summary.csv`: five-seed means, sample SDs, and t intervals.
- `corrected_seed5_public_manifest.csv`: per-seed downstream outcomes.
- `corrected_seed5_paired_effects.csv`: paired model contrasts.
- `grouped_six_board_summary.csv`: board-balanced Raw/V4R/V5R summary.
- `grouped_six_board_results.csv`: per-board Raw/V4R/V5R outcomes.
- `unet_matched_epoch15_paired_board_effects.csv`: per-board U-Net ECA effects.
- `unet_matched_epoch15_paired_summary.csv`: aggregate paired effects.
- `unet_matched_epoch15_per_class_summary.csv`: board-balanced per-class F1.
- `unet_bn_matched_epoch15_paired_board_effects.csv`: strict epoch-15 BN
  no-ECA/ECA board effects.
- `unet_bn_matched_epoch15_paired_summary.csv`: aggregate strict BN paired
  effects.
- `unet_eca_normalization_matched_epoch15_summary.csv`: matched BN, GN, and
  GN-minus-BN interaction effects.
- `output_contract_audit_c01.csv`: matched signal-preservation diagnostics.

See `../../docs/REVISION_RESULTS_2026.md` for the consolidated interpretation
and paper-ready wording.

## Required paper correction

Use the following statement:

> The earlier three-run robustness result is treated as a pilot diagnostic
> because the audit showed that seed propagation was incomplete. The corrected
> seed-aware five-seed replication supersedes it for inferential purposes.
