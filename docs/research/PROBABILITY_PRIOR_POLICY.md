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
