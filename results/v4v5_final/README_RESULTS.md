# V4/V5 Final Thesis Results (Canonical Package)

This directory contains the **canonical final result tables** used in the thesis (Chapter 4 and Appendix Θ).

## Files
- `downstream_summary.csv`  
  Consolidated downstream 5-class classification results (ResNet-18) for Raw / pseudo / real-aligned pipelines.

- `robustness_3seed_summary.csv`  
  Added-noise robustness evaluation (sigma=25) with 3 seeds, reported as mean±std, using Best Macro-F1 epoch selection.

- `latency_params_summary.csv`  
  Parameter count and inference latency measurements for Base / V4 ECA / V5 ECA.

- `plots_data_accuracy_macrof1.csv`  
  Plot-ready table for thesis downstream figures (Accuracy and Macro-F1).

- `plots_data_robustness.csv`  
  Plot-ready table for robustness figure (Macro-F1@BestF1 and Accuracy@BestF1, mean±std).

## Thesis Mapping
- Chapter 4.3 (DnCNN / ECA / downstream impact)
- Appendix Θ (extended V4/V5 analysis, diagnostics, robustness, cost)

## Canonical Note
Only the files in this directory should be treated as the **final curated thesis evidence**.
Exploratory / legacy outputs from older evaluator tracks are preserved elsewhere for historical traceability.
 
