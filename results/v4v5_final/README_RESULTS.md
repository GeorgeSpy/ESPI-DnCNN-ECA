# V4/V5 historical publication results

> **Revision notice (2026-07-28):** this directory is retained unchanged for
> historical traceability. The earlier three-run robustness table must be
> treated as a pilot diagnostic, not an independent multi-seed estimate,
> because a protocol audit found incomplete seed propagation. It is superseded
> for inferential purposes by
> `../revision_2026_corrected_robustness/`.

This directory contains the **canonical final result tables** used in the publication, primarily for Chapter 4 and Appendix Theta.

## Files

- `downstream_summary.csv`  
  Consolidated downstream 5-class classification results (ResNet-18) for Raw, pseudo-noisy, and real-aligned pipelines.

- `robustness_3seed_summary.csv`  
  Historical three-run pilot at `sigma = 25`. Do not use this table for
  independent-seed inference or retain its old p-value in a revised paper.

- `latency_params_summary.csv`  
  Parameter-count and inference-latency summary for Base, V4 ECA, and V5 ECA.

- `plots_data_accuracy_macrof1.csv`  
  Plot-ready table for downstream Accuracy and Macro-F1 figures.

- `plots_data_robustness.csv`  
  Plot-ready table for the robustness figure, including Macro-F1@BestF1 and Accuracy@BestF1 summaries.

## publication mapping

- Chapter 4.3: DnCNN / ECA / downstream impact
- Appendix Theta: extended V4/V5 analysis, diagnostics, robustness, and cost

## Interpretation boundary

These files document the original curated package. They are not the current
canonical robustness evidence after the seed-propagation audit.

Use `results/revision_2026_corrected_robustness/` for the corrected five-seed
replication and the locked board-grouped sensitivity analyses. Do not pool the
random-split seed sweep and board-grouped folds; they estimate different
quantities.
