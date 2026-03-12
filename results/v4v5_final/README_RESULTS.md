# V4/V5 Final Thesis Results (Canonical Package)

This directory contains the **canonical final result tables** used in the thesis, primarily for Chapter 4 and Appendix Theta.

## Files

- `downstream_summary.csv`  
  Consolidated downstream 5-class classification results (ResNet-18) for Raw, pseudo-noisy, and real-aligned pipelines.

- `robustness_3seed_summary.csv`  
  Added-noise robustness evaluation at `sigma = 25` with 3 seeds, reported as mean +/- std using best-Macro-F1 epoch selection.

- `latency_params_summary.csv`  
  Parameter-count and inference-latency summary for Base, V4 ECA, and V5 ECA.

- `plots_data_accuracy_macrof1.csv`  
  Plot-ready table for downstream Accuracy and Macro-F1 figures.

- `plots_data_robustness.csv`  
  Plot-ready table for the robustness figure, including Macro-F1@BestF1 and Accuracy@BestF1 summaries.

## Thesis mapping

- Chapter 4.3: DnCNN / ECA / downstream impact
- Appendix Theta: extended V4/V5 analysis, diagnostics, robustness, and cost

## Canonical note

Only the files in this directory should be treated as the **final curated thesis evidence**.

Exploratory or legacy outputs from older evaluator tracks may exist elsewhere for historical traceability, but they should not override the package documented here.