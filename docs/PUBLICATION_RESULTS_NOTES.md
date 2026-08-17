# publication Results Notes (Repository-to-publication Mapping)

This note maps repository result files to publication tables and figures.

## Current endpoint-validity evidence

- `results/endpoint_validity_2026/board_balanced_summary.csv`
  - matched frequency-only, image-only, and frequency-plus-image summaries;
- `results/endpoint_validity_2026/incremental_deltas.csv`
  - all board/seed conditional Macro-F1 deltas;
- `results/endpoint_validity_2026/paired_board_contrasts.csv`
  - board-first Raw/V4R/V5R contrasts;
- `results/endpoint_validity_2026/per_class_paired_contrasts.csv`
  - descriptive localization by endpoint class;
- `results/endpoint_validity_2026/leakage_audit.csv`
  - inner-OOF coverage and held-out-board exclusion checks.

These files and `docs/ENDPOINT_VALIDITY_UPDATE_2026.md` define the current
downstream evidence. The absolute decision remains
`DNCNN_INCREMENTAL_VALUE_INCONCLUSIVE`; the later paired descriptive result is
`RAW_CONDITIONAL_ADVANTAGE_CONSISTENT`.

## Chapter 4 (DnCNN / ECA / downstream)

- `results/v4v5_final/downstream_summary.csv`
  - Used for the consolidated downstream comparison table (Raw / pseudo / real-aligned)
  - Used as the source for downstream Accuracy and Macro-F1 figures

- `results/v4v5_final/plots_data_accuracy_macrof1.csv`
  - Plot source for:
    - downstream Accuracy figure
    - downstream Macro-F1 figure

- `results/v4v5_final/robustness_3seed_summary.csv`
  - Used for the robustness summary table (3 seeds, sigma = 25)

- `results/v4v5_final/plots_data_robustness.csv`
  - Plot source for the robustness figure (mean +/- std error bars)

- `results/v4v5_final/latency_params_summary.csv`
  - Used for the cost and latency summary table
  - Optional latency plot source

## Interpretation note for publication alignment

The V4/V5 findings belong to the **mature 5-class downstream phase** of the
publication and should not be interpreted as a direct numerical continuation
of the earlier Chapter 4 ablation stage. The historical tables remain valid for
their original protocols, but the August nested endpoint is required for claims
about image value conditional on excitation frequency.

## Appendix Theta

Appendix Theta contains the extended V4/V5 analysis narrative, including regime-dependent behavior, robustness interpretation, cost-performance trade-offs, and diagnostic context.

## Legacy v3 note

Earlier v3 and pilot-stage findings are preserved for historical traceability, but the final publication conclusions rely on the curated V4/V5 package.
