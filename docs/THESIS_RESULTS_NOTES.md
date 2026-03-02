# Thesis Results Notes (Repo ↔ Thesis Mapping)

This note maps repository result files to thesis tables/figures.

## Chapter 4 (DnCNN / ECA / downstream)
- `results/v4v5_final/downstream_summary.csv`
  - Used for the consolidated downstream comparison table (Raw / pseudo / real-aligned)
  - Used as source for downstream Accuracy and Macro-F1 figures

- `results/v4v5_final/plots_data_accuracy_macrof1.csv`
  - Plot source for:
    - Downstream Accuracy figure
    - Downstream Macro-F1 figure

- `results/v4v5_final/robustness_3seed_summary.csv`
  - Used for robustness summary table (3 seeds, sigma=25)

- `results/v4v5_final/plots_data_robustness.csv`
  - Plot source for robustness figure (mean±std error bars)

- `results/v4v5_final/latency_params_summary.csv`
  - Used for cost/latency summary table
  - Optional latency plot source

## Appendix Θ
Appendix Θ contains the extended analysis narrative (V4/V5 regime-dependent behavior, robustness interpretation, trade-off discussion, and diagnostic context).

## Legacy v3 note
v3 pilot/legacy results are preserved for historical context and traceability, but the final thesis conclusions rely on the curated V4/V5 package.
 
