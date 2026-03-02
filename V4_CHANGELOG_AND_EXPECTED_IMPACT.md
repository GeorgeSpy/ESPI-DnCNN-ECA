# DnCNN Lite ECA v4 - Changelog And Expected Impact

## Scope

This changelog outlines the changes made from `v3` to `v4` in:
- `C:\ESPI_DnCNN\espi_dncnn_lite_eca_FULL_PATCH_v4.py`

and explains why these changes are expected to improve the model's behavior.

---

## 1. New Feature: Clean no-ECA baseline

### What changed
- Added CLI flags:
  - `--use-eca`
  - `--no-eca`
- Added `use_eca` to the config (`DnCNNLiteECAConfig`) and runtime args.
- During model construction, when `use_eca=False`, all attention blocks are replaced with identity mappings.

### Why
- Provides a true A/B comparison without having to switch scripts or architecture families.

### Expected Impact
- Fair comparison between ECA and no-ECA.
- Fewer "artifact" discrepancies caused by running entirely different scripts.

---

## 2. New Feature: Controlled GroupNorm

### What changed
- Added `--gn-groups` flag.
- New helper function `make_norm_with_groups(...)`:
  - `gn_groups=0` -> auto mode (tests 8/4/2/1 divisors)
  - `gn_groups>0` -> explicitly selects the largest valid divisor.
- The `ConvBlock` now accepts `gn_groups`.

### Why
- Group Normalization (GN) strongly affects attention and activation dynamics.
- We require a controlled parameter sweep rather than hard-coded behavior.

### Expected Impact
- More stable training.
- Potential improvements in SSIM/EdgeF1 on challenging samples.

---

## 3. Non-finite Guards in Training Loop

### What changed
- `run_epoch_train(...)` was rewritten to:
  - Check for finite inputs/loss/gradients.
  - Skip bad batches (`nan_action=skip`) or halt completely (`nan_action=stop`).
  - Track non-finite occurrences.
  - Return structured statistics.
- Added arguments:
  - `--nan-action {skip,stop}`
  - `--max-nonfinite-batches`
  - `--grad-clip`
  - `--log-grad-norm`

### Why
- Previous runs experienced NaN collapses starting as early as epoch 2.
- Without these guards, misleading logs and corrupted checkpoints are saved.

### Expected Impact
- Run stability and reliability.
- Immediate fail-fast behavior when something goes wrong.

---

## 4. Non-finite Guards in Validation/Real Eval

### What changed
- `run_validation_fullres(...)` now:
  - Checks for finite inputs/loss/metrics.
  - Tracks `samples_seen`/`samples_used`/`nonfinite_batches`.
  - Returns a dictionary containing metrics and counters.
- `run_real_evaluation(...)` now:
  - Skips non-finite samples.
  - Returns `used` and `skipped` counts.

### Why
- Validation is the source of truth for model selection.
- It is critical to know if a metric was derived from a complete or a "broken" validation pass.

### Expected Impact
- Cleaner and more accurate selection of the best checkpoint.
- Better diagnostics when outliers or instabilities occur.

---

## 5. Safer Checkpoint Loading

### What changed
- Added flags:
  - `--resume-strict` (default)
  - `--resume-nonstrict`
- Resuming a model now loads strictly by default.
- In strict mode, a mismatch causes a hard error (instead of silently proceeding).

### Why
- Non-strict loading frequently obscures architecture mismatches.
- It causes subtle, "sneaky" errors during evaluation.

### Expected Impact
- Consistent resume behavior.
- Fewer pseudo-results caused by partially matched checkpoints.

---

## 6. Logging/Observability Upgrade

### What changed
- New columns in `train_log.csv`:
  - `train_nonfinite`, `val_nonfinite`
  - `train_batches`, `val_samples`
  - `grad_norm`
- New TensorBoard scalars:
  - `debug/train_nonfinite`
  - `debug/val_nonfinite`
  - `opt/grad_norm_mean`

### Why
- The v3 logs were insufficient for explaining why a run degraded.

### Expected Impact
- Rapid root-cause analysis.
- More reliable experiment tracking.

---

## 7. Default Behavior Changes

### What changed
- `--freeze-norm-epoch` default value changed from `3` to `0`.

### Why
- For fair ablation studies, we prefer initial tests without early freezing.
- Premature freezing can hinder layer adaptation.

### Expected Impact
- Clearer conclusions regarding the impact of ECA and GN.

---

## 8. Checkpoint Selection Hardening

### What changed
- `best` and `best_ssim` metric updates are only executed when the validation metrics are finite.

### Why
- Prevents saving a "best model" derived from a corrupted validation state.

### Expected Impact
- More valid and robust final checkpoints.

---

## 9. Expected Impact Summary

| Change | Primary Benefit | Expected Result |
|---|---|---|
| `--no-eca` / `--use-eca` | Fair ablation | Clean comparison of ECA vs baseline |
| `--gn-groups` | Norm control | Better tuning for SSIM/EdgeF1 |
| Non-finite guards | Stability | Fewer NaN collapse runs |
| Strict resume | Correctness | Fewer checkpoint mismatch errors |
| Expanded logs | Observability | Faster debugging and reproducibility |
| Default no freeze | Fairness | More honest initial A/B results |

---

## 10. Realistic Expectations from v4

Version 4 is primarily a reliability and fairness upgrade, not a "magical" architectural leap.

Realistically, the difference (ECA vs noECA) after proper tuning is expected to be:
- `ΔPSNR`: approximately `-0.10` to `+0.30` dB
- `ΔSSIM`: approximately `-0.01` to `+0.03`
- `ΔEdgeF1`: approximately `+0.005` to `+0.03`

The greatest advantage of v4:
- The results obtained will be significantly more reliable and reproducible.
