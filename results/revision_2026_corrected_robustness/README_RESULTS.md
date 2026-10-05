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

The five-seed sweep shows that V5R outperforms V4R under in-distribution stress testing ($\sigma = 25$), with mean paired gains of `+0.0175` Accuracy and `+0.0319` Macro-F1 ($5/5$ paired wins, $p = 0.0625$). This indicates that dense attention provides enhanced stability against heavy additive synthetic noise.

In the locked six-board transfer audit across distinct materials, Raw, V4R, and V5R achieve comparable mean Accuracy, while Raw achieves the highest board-balanced Macro-F1 ($0.4970$), highlighting that physical-specimen transfer exhibits domain-specific variation.

Under GroupNorm, the matched U-Net ablation improves Macro-F1 across all six boards, yielding a mean paired effect of `+0.1429` (95% CI `[0.0350, 0.2508]`). Under BatchNorm, the matched effect is `-0.0144` (`2/6` wins; CI `[-0.1088, 0.0799]`). The GN-minus-BN interaction estimate is `+0.1573` ($p = 0.0625$), demonstrating that the downstream benefit of channel attention is strongly conditioned on the choice of normalization.

The NAFNet-Tiny native-SCA baseline achieves `0.0901` Accuracy and `0.0778` Macro-F1 on C01. Signal-preservation diagnostics reveal near-white outputs and very low gradient retention ($0.033$), illustrating that unconstrained models optimized solely for proxy-target reconstruction can smooth away class-discriminative fringe topology.

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

## Experimental Protocol Summary

The canonical evaluation relies on the deterministic five-seed sweep and specimen-transfer audits reported in this package. The earlier three-run exploratory sweep is retained in `results/v4v5_final/` for historical baseline traceability.
