# Endpoint-validity result package

This directory is the current public evidence package for the DnCNN downstream
endpoint. It contains no images, logits, checkpoints, personal paths, or
sample-level identifiers.

## Frozen decisions

- Absolute incremental-value gate: `DNCNN_INCREMENTAL_VALUE_INCONCLUSIVE`.
- Paired descriptive audit: `RAW_CONDITIONAL_ADVANTAGE_CONSISTENT`.
- Manuscript recommendation: `CAUTIONARY_METHODS_PAPER_SUPPORTED`.

## Files

| File | Role |
|---|---|
| `endpoint_freeze_summary.csv` | sample, label, class, and fold closure |
| `view_lock_summary.csv` | deterministic Raw/V4R/V5R materialization |
| `contract_hashes.csv` | SHA-256 of the frozen local analysis contracts |
| `source_artifact_provenance.csv` | source revision and bitwise-copy hashes |
| `class_support_by_board.csv` | five-class support on every held-out board |
| `board_balanced_summary.csv` | primary and diagnostic aggregate metrics |
| `board_seed_mean_metrics.csv` | board-first seed-averaged arm metrics |
| `incremental_deltas.csv` | all board/seed conditional deltas |
| `per_class_metrics.csv` | per-board, per-seed, per-arm class metrics |
| `condition_balanced_sensitivity.csv` | condition-balanced sensitivity results |
| `leakage_audit.csv` | inner-OOF coverage and held-out-board checks |
| `paired_board_contrasts.csv` | board-first Raw/V4R/V5R paired contrasts |
| `paired_contrast_summary.csv` | board-balanced paired effects and intervals |
| `condition_balanced_paired_contrasts.csv` | paired condition-balanced sensitivity |
| `per_class_paired_contrasts.csv` | localization of paired effects by class |
| `board_bootstrap_results.csv` | descriptive board bootstrap |
| `exact_signflip_results.csv` | descriptive exact six-board sign flips |
| `public_artifact_sha256.csv` | hashes of this public result package |

`per_class_metrics.csv` contains the complete 5-seed table. The
historical-compatible I-only values quoted in the report use seed 42; the
I-only rows in `board_balanced_summary.csv` average seed-level board means.

## Validation

From the repository root:

```bash
python scripts/validate_endpoint_validity_results.py
```

The validator checks frozen counts, headline metrics, decision inputs, leakage
rows, schemas, and the public hash manifest without requiring the private
corpus.
