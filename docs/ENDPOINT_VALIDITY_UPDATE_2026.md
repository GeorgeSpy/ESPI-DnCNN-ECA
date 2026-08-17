# Endpoint-validity update: nested frequency-conditional evaluation

**Revision date:** 2026-08-17  
**Primary endpoint:** fixed five-class Macro-F1, unweighted across six held-out boards  
**Frozen absolute decision:** `DNCNN_INCREMENTAL_VALUE_INCONCLUSIVE`  
**Paired descriptive decision:** `RAW_CONDITIONAL_ADVANTAGE_CONSISTENT`

## 1. Why this update was required

The earlier downstream analyses established image-only performance under
random or board-grouped protocols, but excitation frequency alone predicts the
five-class endpoint strongly. Image-only scores therefore do not answer the
more relevant question: does an image view add predictive information after a
matched frequency-only model is already available?

This update evaluates that conditional estimand without changing the historical
denoisers:

```text
Delta Macro-F1(image | frequency)
    = Macro-F1(frequency + image) - Macro-F1(frequency only)
```

The experiment compares Raw, V4R, and V5R views using identical samples,
labels, folds, frequency features, fusion-model family, and outer evaluation
boards.

## 2. Frozen endpoint and provenance

- Observation unit: real physical repeat.
- Samples: 11,706.
- Boards: C01, C02, C03, W01, W02, W03.
- Class order: `1_1H`, `1_1T`, `1_2`, `2_1`, `higher`.
- Cross-project label differences audited: 71.
- Label corrections: 0.
- Unresolved labels: 0.
- Interpretation of all 71 cases:
  `CROSS_PROJECT_DIFFERENCE_BUT_BOTH_PROTOCOL_VALID`.
- Raw/V4R/V5R locked-view coverage: 100%.
- V4R materialized views: 11,706, bitwise deterministic.
- V5R materialized views: 11,706, bitwise deterministic.
- Raw materialized views: 11,706, bitwise deterministic.
- Historical hard-prediction agreement: 100%.
- Canonical classifier-input representation: float32 array before classifier
  preprocessing.

The path-bearing sample manifest, logits, checkpoints, and locked tensors are
not included in the public repository. Their aggregate provenance and the
hashes of the frozen contracts are retained in the public tables.

## 3. Nested evaluation design

Each outer fold holds out exactly one physical board. For each view and seed,
condition-grouped inner out-of-fold image logits were created using only the
five development boards. Frequency-only and frequency-plus-image fusion heads
used the same balanced multinomial logistic-regression family and were fit only
on inner out-of-fold records. The outer board was used once for evaluation.

Five locked seeds were used for the image branches: `42`, `13`, `37`, `101`,
and `202`.

The completed model inventory comprised:

- 225 newly fitted inner models;
- 72 newly fitted outer models;
- 18 verified historical outer models reused;
- 297 newly fitted models in total.

All 90 outer prediction tables and 90 inner out-of-fold tables passed exact
coverage, uniqueness, held-out-board exclusion, and condition-group checks.
No in-sample image logits entered fusion fitting.

## 4. Absolute incremental-value result

| Regime | Historical-compatible seed-42 I-only Macro-F1 | F+I Macro-F1 | Delta from matched F-only | Positive boards |
|---|---:|---:|---:|---:|
| Raw | 0.497028 | 0.850356 | +0.033699 | 4/6 |
| V4R | 0.439515 | 0.800619 | -0.016037 | 2/6 |
| V5R | 0.468059 | 0.774716 | -0.041940 | 1/6 |

The matched F-only board-balanced Macro-F1 is `0.816656`. The deterministic
nearest-frequency diagnostic reaches `0.861237`; it is reported separately and
is not substituted for the matched fusion baseline.

The preregistered support rule required a mean delta of at least `+0.030` and a
positive delta on at least five of six boards. Raw clears the mean threshold
but is positive on only four boards. V4R and V5R do not clear either criterion.
Consequently, the immutable absolute decision is:

```text
DNCNN_INCREMENTAL_VALUE_INCONCLUSIVE
```

The I-only entries in `board_balanced_summary.csv` average the five seed-level
board means and therefore differ from the historical-compatible seed-42 values
shown above. This is an intentional estimand distinction, not a discrepancy.

## 5. Board-first paired conditional contrasts

The paired audit reused the completed Stage-B tables. It performed no fitting,
inference, denoising, or prediction generation. Seeds were averaged within each
board, and the held-out board remained the inferential unit.

| Board | Raw delta | V4R delta | V5R delta | Raw - V4R | Raw - V5R |
|---|---:|---:|---:|---:|---:|
| C01 | +0.047259 | -0.023479 | +0.000695 | +0.070738 | +0.046564 |
| C02 | -0.011488 | -0.069391 | -0.003221 | +0.057903 | -0.008267 |
| C03 | -0.043435 | -0.067855 | -0.135509 | +0.024420 | +0.092074 |
| W01 | +0.077612 | +0.039852 | -0.031632 | +0.037760 | +0.109244 |
| W02 | +0.112323 | +0.026702 | -0.022049 | +0.085621 | +0.134372 |
| W03 | +0.019924 | -0.002054 | -0.059924 | +0.021978 | +0.079849 |

Board-balanced contrasts:

- Raw minus V4R: `+0.0497367` Macro-F1, positive on 6/6 boards;
  descriptive board-bootstrap interval `[0.0314100, 0.0684046]`.
- Raw minus V5R: `+0.0756394`, positive on 5/6 boards;
  descriptive interval `[0.0351426, 0.1091660]`.
- V4R minus V5R: `+0.0259027`, positive on 4/6 boards;
  descriptive interval `[-0.0178238, 0.0635114]`.

The direction is unchanged under condition-balanced sensitivity: Raw exceeds
V4R on 6/6 boards and V5R on 5/6 boards.

The largest class-level Raw advantages are concentrated in `1_2` and `2_1`:

| Class | Raw - V4R incremental F1 | Raw - V5R incremental F1 |
|---|---:|---:|
| 1_1H | -0.002556 | -0.001033 |
| 1_1T | +0.013609 | +0.029551 |
| 1_2 | +0.134003 | +0.150423 |
| 2_1 | +0.102045 | +0.193755 |
| higher | +0.001582 | +0.005501 |

The paired descriptive decision is:

```text
RAW_CONDITIONAL_ADVANTAGE_CONSISTENT
```

This decision does not alter the preregistered absolute result.

## 6. Scientific interpretation

The safe conclusion is that, for this five-class endpoint and these six held-out
boards, the Raw views retain more conditional predictive information beyond
excitation frequency than V4R or V5R. The historical Raw/V4R/V5R image-only
ranking is therefore insufficient as a stand-alone test of fringe-preserving
denoiser utility.

The result does **not** establish:

- that the residual predictive information is fringe morphology;
- that it represents physical displacement or another calibrated measurand;
- that denoising universally removes useful information;
- that either denoiser is universally inferior for reconstruction;
- that the legacy averaged proxy is a unique clean target.

## 7. Related audit boundaries

The acquisition-provenance audit remains `H4_WEAK_PROXY_ONLY`: no independent,
calibrated, repeat-level observable with exact image mapping was found. A later
pixel-overlay readback addendum remained `READBACK_EXTRACTION_NOT_VALIDATED`, so
it did not upgrade H4.

The C01 target-validity audit found pixel repeat variability and at least one
repeatable/discriminative candidate descriptor, but it did not find systematic
aggregate-target structural excess. Its decision was `GATE_1A_NO_GO` for that
specific target-ambiguity framing.

Details of the broader protocol program are summarized in
[`PROTOCOL_V2_AUDIT_STATUS_2026.md`](PROTOCOL_V2_AUDIT_STATUS_2026.md).

## 8. Public artifacts

The complete public aggregate package is in
[`results/endpoint_validity_2026/`](../results/endpoint_validity_2026/). Run:

```bash
python scripts/validate_endpoint_validity_results.py
```

to verify schemas, row counts, frozen headline values, leakage checks, and the
public SHA-256 manifest.

