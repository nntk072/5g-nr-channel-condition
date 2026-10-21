# Grouped scalar-regression uncertainty evaluation

`scripts/evaluate_regression_uncertainty.py` evaluates frozen scalar predictions
with calibration and test data separated by independent group. A group should be
the unit that is plausibly independent for the claim (usually a scene, trajectory
family, or measurement session), not an individual correlated radio frame.

The input is UTF-8 JSON with this schema:

```json
{
  "schema": "loslab.scalar_regression_predictions.v1",
  "target_name": "relative_path_delay",
  "target_unit": "ns",
  "predictions": [
    {
      "sample_id": "route-a:0001",
      "group_id": "scene-family-a",
      "split": "calibration",
      "target": 12.0,
      "prediction": 11.4,
      "uncertainty_scale": 2.0,
      "identifiability_stratum": "resolved"
    },
    {
      "sample_id": "route-b:0001",
      "group_id": "scene-family-b",
      "split": "test",
      "target": 20.0,
      "prediction": 18.7,
      "uncertainty_scale": 3.0,
      "identifiability_stratum": "ambiguous"
    }
  ]
}
```

`uncertainty_scale` must be positive and is the model's relative scale; the
calibrator learns a multiplier from the maximum standardized absolute residual
within each calibration group. It reports split-conformal intervals at 50%, 80%,
90%, and 95%, group-simultaneous as well as sample coverage, interval width,
selective MAE/RMSE by uncertainty ranking, a test-group bootstrap interval for
macro group RMSE, and optional interval-coverage breakdown by
`identifiability_stratum` using the same pooled calibration. If there are too few
independent calibration groups for a
requested finite-sample level, the mathematically correct interval is unbounded
and is reported as such instead of silently clipping the quantile.

Example:

```bash
uv run --no-sync --locked python scripts/evaluate_regression_uncertainty.py \
  predictions.json outputs/uncertainty-report.json --seed 42 --bootstrap-draws 2000
```

The predictor and `uncertainty_scale` function must be frozen using training
groups before this file is created; no calibration or test row may be used to
fit or tune them. The calibration guarantee requires calibration and test
groups to be disjoint and exchangeable. Scene-family shifts, systematic simulator mismatch, and
non-exchangeable sessions invalidate that guarantee. The evaluator explicitly
rejects overlapping groups; it cannot establish exchangeability for the user.
Risk-coverage curves and intervals from a single held-out scene remain
descriptive and are not evidence of scene-level generalization. Calibration and
test rows must be fixed before inspecting test metrics. Never use path-oracle
quantities or simulator-only metadata as model inputs at inference.
