# Probability calibration and prior policy

The pilot experiment runner supports an explicit natural target-prior declaration
for evaluating a classifier trained under either a natural or an enriched stress
prior. The declaration is part of the run configuration and therefore part of its
configuration digest and manifest.

```yaml
prior_shift:
  schema: loslab.prior_shift.v1
  training_prior_kind: enriched_stress
  training_prior_id: frozen-training-prior-study-v1
  target_prior_kind: natural
  target_prior_id: frozen-target-population-study-v1
  target_positive_prevalence: 0.12
  target_prior_evidence_sha256: <sha256-of-the-reviewed-prior-evidence>
```

Replace the identifiers, prevalence, and digest with the values from the frozen
study evidence. The digest is a caller-supplied evidence reference; the runner
does not fetch, authenticate, or scientifically validate that evidence.

When a signed split registry includes a calibration partition, the runner uses
those rows. Otherwise, it deterministically reserves a group-disjoint subset of
the training pool and fits the model on the remaining training rows. Group-
balanced Platt calibration and the source positive prevalence are both estimated
from the calibration partition. Validation rows select the operating threshold
after calibration and prior adjustment. Test labels are used only for the final
metrics. Prediction artifacts retain the raw, calibrated, and prior-adjusted
probabilities and the exact partition indices. The saved model threshold is
converted back to raw model-score space for inference.

The odds correction assumes label shift: class-conditional score distributions
must remain stable while class prevalence changes. The runner cannot establish
that assumption. Equal weighting of independent groups is the declared pilot
estimator; it is not a claim that the supplied group IDs represent independent
physical acquisition units. This workflow does not establish natural-prior
calibration, qualify any NR receiver, or authorize sealed-test release. It is
implemented only in the legacy common experiment runner; other model and
evaluation routes remain open work.

## Optional low-confidence coverage audit

An experiment config may also set `low_confidence_margin` to a preregistered
absolute probability distance in `[0, 0.5)`. The runner applies the fixed rule
`abs(test_probability - validation_threshold) <= low_confidence_margin` after
the validation-selected threshold has been frozen. It reports flagged row and
caller-supplied group counts overall and by source family and domain, validates
the report against `loslab.low_confidence_audit.v1`, and stores the row mask in
the prediction artifact. The margin is included in the hashed run config. This
is descriptive coverage only: test labels do not select the threshold or
margin, and the audit does not estimate error risk or prove group independence.
