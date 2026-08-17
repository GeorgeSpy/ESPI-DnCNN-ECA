# Current claim boundaries

This document defines the current public interpretation of the repository. It
supersedes earlier language wherever a conflict exists.

## Supported statements

1. The corrected five-seed in-distribution experiment estimates seed robustness
   for its original protocol; it does not estimate transfer to unseen boards.
2. The six-board nested endpoint uses 11,706 real physical repeats with complete
   five-class support and locked Raw/V4R/V5R identity.
3. Excitation frequency is a strong predictor of the five-class label, so image
   utility must be evaluated conditionally rather than from image-only scores.
4. The absolute incremental-value gate is inconclusive: Raw has mean delta
   `+0.033699` but only 4/6 positive boards; V4R and V5R have negative mean
   deltas.
5. In the matched paired audit, Raw exceeds V4R on 6/6 boards and V5R on 5/6
   boards. The condition-balanced sensitivity has the same direction.
6. Under this endpoint, Raw retains more conditional task-relevant image
   information than either denoised view.
7. Reconstruction, in-distribution classification, and conditional held-out
   board utility are different estimands and must not be pooled.

## Required qualifiers

- The board-bootstrap intervals and exact sign-flip calculations are
  descriptive six-board analyses. The boards are not asserted to be a random
  population sample.
- The paired audit was performed after the frozen absolute gate and cannot
  replace its `DNCNN_INCREMENTAL_VALUE_INCONCLUSIVE` decision.
- The residual image information has not been identified as fringe morphology,
  displacement information, or any other calibrated physical observable.
- H4 remains `H4_WEAK_PROXY_ONLY`.
- The legacy averaged target has incomplete operator provenance. Cross-fitted
  means and medoids are analytical candidates, not proven reproductions of the
  legacy target.
- Target ambiguity does not automatically invalidate historical downstream
  results; it changes the interpretation of reconstruction metrics and clean
  target semantics.

## Unsupported statements

Do not claim that:

- V4R or V5R universally improves ESPI recognition;
- V5R is a universally superior denoiser because it won the earlier
  in-distribution five-seed comparison;
- Raw's conditional advantage proves that denoising destroys physical signal;
- repeat variability is known to be physical signal or known to be only noise;
- averaged references are universally invalid;
- medoids are universally superior to means;
- the direct set-valued training pilots prove that SET-C or SET-D is generally
  impossible;
- the untrained RSG-NAFNet design is a validated restoration contribution.

## Manuscript consequence

The current evidence supports a cautionary methods/evaluation framing. Any
strong reconstruction-versus-recognition claim must include the frequency-only
control, held-out-board results, and the distinction between the frozen
absolute decision and the paired descriptive audit.

