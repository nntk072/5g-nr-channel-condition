# evaluation.scoring

Metrics and statistics for evaluating channel-condition predictions: accuracy and
calibration metrics, paired-seed statistics, grouped regression uncertainty,
class-support and low-confidence audits, prior-shift corrections, and coverage
of target families.

## Invariants

- Functions take explicit arrays and seeds; no module reads global state.
- Statistics report their inputs' counts, so small-sample results remain visible.
