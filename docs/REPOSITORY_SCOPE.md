# Repository Scope and Thesis Mapping

This note clarifies how the public repositories map to the thesis workflow.

## What this repository contains

`ESPI-DnCNN-ECA` is the **denoising repository**. It contains:

- public DnCNN-Lite denoising scripts,
- ECA ablation variants,
- plotting utilities for the final V4/V5 thesis figures,
- canonical final CSV tables for downstream, robustness, and latency analysis,
- supporting documents that map files to thesis sections.

## What this repository does not contain

The thesis also includes two additional technical components that are maintained separately:

- the pseudo-noisy data generator,
- the downstream classification and evaluation code.

Those components are intentionally isolated in separate repositories because they belong to different stages of the overall workflow.

## Thesis-level interpretation

At thesis level, the complete workflow is:

1. Generate pseudo-noisy supervision where needed.
2. Train or apply the DnCNN-ECA denoising stage.
3. Produce denoised or aligned inputs for downstream analysis.
4. Train and evaluate the classification models.

This repository corresponds primarily to step **2**, plus the curated final denoising-to-classification result package used in later thesis analysis.

## Practical guidance for readers

If a reader wants to understand the denoising architecture evolution and the final V4/V5 denoising conclusions, this repository is the correct entry point.

If a reader wants to reproduce the full thesis pipeline, they should also consult the separate pseudo-noisy-generator and classification repositories linked in the main `README.md`.