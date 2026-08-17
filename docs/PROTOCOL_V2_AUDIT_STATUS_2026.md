# Protocol V2 audit status

This is a compact public record of the protocol work that followed the earlier
DnCNN-ECA revision. It documents decisions and scope boundaries; it does not
publish raw data, checkpoints, sample-level paths, logits, or machine-specific
experiment trees.

## Data and split closure

- Nine of nine confirmed provenance mappings were applied.
- All 96 provenance/manifest acceptance checks passed.
- The primary real-only contract contains 1,382 conditions with at least nine
  repeats and 23,557 real frames.
- Pseudo pairs in primary manifests: 0.
- Shared proxy groups, conditions, byte hashes, and canonical-pixel hashes
  across protected splits: 0.
- C01 remains development-only.
- Frozen registration policy: R0 identity only.
- Confirmatory technical attrition: 0 of 18,695 frames.

### V2-A split sizes

| Split | Frames | Conditions |
|---|---:|---:|
| C01 development | 4,862 | 232 |
| Train | 15,101 | 928 |
| Validation | 1,970 | 122 |
| Test | 1,624 | 100 |

### V2-B board folds

| Held-out board | Train frames / conditions | Validation frames / conditions | Test frames / conditions |
|---|---:|---:|---:|
| C02 | 10,833 / 683 | 3,272 / 222 (C03) | 4,590 / 245 |
| C03 | 10,833 / 683 | 4,590 / 245 (C02) | 3,272 / 222 |
| W01 | 10,791 / 658 | 4,013 / 253 (W02) | 3,891 / 239 |
| W02 | 11,753 / 706 | 2,929 / 191 (W03) | 4,013 / 253 |
| W03 | 11,875 / 720 | 3,891 / 239 (W01) | 2,929 / 191 |

Each fold also retains the separate C01 development set of 4,862 frames and 232
conditions.

## QC and repeat-role contracts

Carbon-board line structure and small fringe displacement were treated as
diagnostic morphology, not automatic corruption. Registration estimation was
not used to transform real images. The frozen production decision is R0 identity;
R1 remains a negative control and rigid/non-rigid registration is excluded.

The deterministic reference builder uses one input repeat, four reconstruction
references, and four independent structural references. Eligibility requires
at least nine technically valid real repeats.

## Loss and architecture contracts

The SET-C primary and SET-D backup loss interfaces passed numerical,
permutation, autograd, finite-difference, mixed-precision, and diagnostic
non-interference tests after the V2.5 revisions. The architecture-design gate
defined a common NAFNet-ESPI-B0 backbone and an experimental RSG-NAFNet design.
These are interface results, not trained-model performance findings.

## Training-pilot status

The later direct set-valued pilots did not complete their frozen budgets:

| Arm | Frozen outcome |
|---|---|
| A0-N0 SET-C, tau 0.02 | Early confirmed output-coherence event; no tau selected |
| A0-N0 SET-C, tau 0.05 | Early confirmed output-coherence event; no tau selected |
| A0-N0 SET-D, tau 0.05 | Safety gate failed after 72 committed updates |
| Legacy P0 control | Not executed; exact historical scalar loss was not uniquely recoverable |

These are protocol outcomes, not evidence of universal loss infeasibility. The
hard-stop detector had synthetic and functional controls but was not validated
with labeled real mid-training outputs and a frozen ROC/PR operating point. The
incidents are therefore described as unresolved early output-coherence events.

No completed learned result from this experimental route is promoted as a
canonical model contribution in this repository.

## Target and physical-observable audits

- H1, pixel repeat variability: supported.
- H2, repeatability plus broad condition discriminability for at least one
  candidate descriptor: supported.
- H3, systematic aggregate-target structural excess: not supported.
- Gate 1A decision: `GATE_1A_NO_GO`.
- Physical-observable status: `H4_WEAK_PROXY_ONLY`.
- Pixel-overlay readback addendum: `READBACK_EXTRACTION_NOT_VALIDATED`.

Accordingly, between-repeat variability cannot be attributed uniquely from the
available records either to conditional acquisition noise or to variation of
the underlying mechanical field.

## Current program consequence

The publication-relevant completed learned evidence remains the historical
DnCNN-ECA family plus the corrected robustness, grouped transfer, nested
endpoint, and paired conditional analyses. The untrained RSG design and the
early-stopped set-valued pilots must not be presented as validated model
improvements or general negative results.

