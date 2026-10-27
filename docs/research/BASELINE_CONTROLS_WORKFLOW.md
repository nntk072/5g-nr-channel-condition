# Shared-split power-only baseline

The canonical `loslab experiment` runner includes an opt-in `power_only`
baseline. It derives a real-valued feature for each receiver link and
subcarrier from the observed CFR, using `10 log10(max(|H|², 1e-12))`. This
removes complex phase while preserving the measured frequency-resolved power
profile. It does not read labels, source IDs, domains, path truth, or other
metadata as features.

Use one invocation to compare methods on the exact same per-seed split:

```sh
python -m loslab experiment \
  --input <receiver-observations.npz> \
  --output <new-run-directory> \
  --observation precomputed \
  --methods power_only mlp raw_mlp \
  --seeds 11 23 47 59 71 \
  --bootstrap 200
```

The runner saves power-only models as sklearn logistic-regression baselines,
binds the exact feature transform and full/partition matrix digests in model
input artifact contract v3, and reports paired seed comparisons when both
methods are present. The schema is
`datasets/schema/model-input-artifact-contract.v3.schema.json`; v1/v2 readers
remain available. Results are simulation-only diagnostics and do not show
that power is sufficient for measured or deployment channels.

G8.08 is only partially implemented. Source-ID-only, metadata-only,
time/row-shuffled, permuted-label, randomized-phase, and explicit oracle-leak
controls still need isolated routes and audit contracts before benchmarking.
