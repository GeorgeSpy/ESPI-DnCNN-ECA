# Repository Scope and publication Mapping

This note clarifies how the public repositories map to the publication workflow.

The publication codebase is organized into three code components:

1. **Pseudo-noisy data generation**
2. **Denoising**, which is the scope of this repository
3. **Classification and evaluation**

## What this repository contains

`ESPI-DnCNN-ECA` is the **denoising repository**. It is one of the three publication code components and contains:

- public DnCNN-Lite denoising scripts,
- ECA ablation variants,
- plotting utilities for the final V4/V5 publication figures,
- canonical final CSV tables for downstream, robustness, and latency analysis,
- the August 2026 endpoint-validity and compact protocol-audit tables,
- supporting documents that map files to publication sections.

## What this repository does not contain

The publication also includes two additional technical components that are maintained separately:

- the pseudo-noisy data generator,
- the downstream classification and evaluation code.

Those components are intentionally isolated in separate repositories because they belong to different stages of the overall workflow.

## publication-level interpretation

At publication level, the complete workflow is:

1. Generate pseudo-noisy supervision where needed.
2. Train or apply the DnCNN-ECA denoising stage.
3. Produce denoised or aligned inputs for downstream analysis.
4. Train and evaluate the classification models.

This repository corresponds primarily to step **2**, plus curated
denoising-to-classification result packages. The current downstream
interpretation is the nested frequency-conditional endpoint documented in
`ENDPOINT_VALIDITY_UPDATE_2026.md`; earlier result packages remain available for
estimand-specific traceability.

## Practical guidance for readers

If a reader wants to understand the denoising architecture evolution and the
V4/V5 denoising evidence, this repository is the correct entry point. Current
claims must also follow `CURRENT_CLAIM_BOUNDARIES_2026.md`.

If a reader wants to reproduce the full publication pipeline, they should also consult the separate pseudo-noisy-generator and classification repositories linked in the main `README.md`.
