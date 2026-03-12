# Thesis Results Notes (Repository-to-Thesis Mapping)

This note maps repository result files to thesis tables and figures.

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

## Interpretation note for thesis alignment

The V4/V5 findings belong to the **mature 5-class downstream phase** of the thesis and should not be interpreted as a direct numerical continuation of the earlier Chapter 4 ablation stage. They reflect the later curated evaluation package used for the final thesis interpretation.

## Appendix Theta

Appendix Theta contains the extended V4/V5 analysis narrative, including regime-dependent behavior, robustness interpretation, cost-performance trade-offs, and diagnostic context.

## Legacy v3 note

Earlier v3 and pilot-stage findings are preserved for historical traceability, but the final thesis conclusions rely on the curated V4/V5 package.