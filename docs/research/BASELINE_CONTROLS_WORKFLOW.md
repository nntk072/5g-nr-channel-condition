# Shared-split power-only baseline

The canonical `loslab experiment` runner includes an opt-in `power_only`
baseline. It derives a real-valued feature for each receiver link and
subcarrier from the observed CFR, using `10 log10(max(|H|², 1e-12))`. This
removes complex phase while preserving the measured frequency-resolved power
profile. It does not read labels, source IDs, domains, path truth, or other
metadata as features.

It also includes an opt-in `phase_randomized` control. This independently
randomizes phase for every sample, receiver link, and subcarrier while
preserving observed CFR magnitude, then supplies the flattened real/imaginary
matrix to the same MLP implementation as `raw_mlp`. Pin its independent
`phase_randomization_seed` (default `2027`) in experiment YAML or with
`--phase-randomization-seed` to reproduce the transform.

Use one invocation to compare methods on the exact same per-seed split:

```sh
python -m loslab experiment \
  --input <receiver-observations.npz> \
  --output <new-run-directory> \
  --observation precomputed \
  --methods power_only phase_randomized mlp raw_mlp \
  --seeds 11 23 47 59 71 \
  --bootstrap 200
```

The runner saves power-only models as sklearn logistic-regression baselines,
binds the exact feature transform and full/partition matrix digests in model
input artifact contract v3. The phase-randomized route shares the raw-MLP
classifier and split, records its seed and transformed matrix in contract v4,
and reports paired seed comparisons against raw CFR when both methods are
present. The `label_permuted` control uses the same flattened observed CFR and
MLP implementation as `raw_mlp`, but deterministically permutes only the
training labels for each run seed. It preserves the training class counts and
leaves validation, calibration, and test labels unchanged. Contract v5 binds
the transform algorithm, seeds, source-target digest, permutation digest,
permuted-target digest, and positive count; verification replays the
permutation from the retained training labels. Compare it on the same split:

```sh
python -m loslab experiment \
  --input <receiver-observations.npz> \
  --output <new-run-directory> \
  --observation precomputed \
  --methods label_permuted raw_mlp \
  --seeds 11 23 47 59 71 \
  --label-permutation-seed 2028 \
  --bootstrap 200
```

The `row_shuffled` control keeps the training labels fixed and permutes only
the observed-CFR feature rows inside the training partition. Validation,
calibration, and test features remain in their original rows. It uses the same
MLP implementation and split as `raw_mlp`; contract v6 binds the source and
permuted training-matrix digests, permutation indices digest, seeds, and
permutation implementation digest, and verifies the operation from the
retained source observations. Run it with `--methods row_shuffled raw_mlp` and
pin `--row-shuffle-seed` (default `2029`).

The `source_only` control uses only the dataset's categorical `source`
identifier. Its sorted one-hot vocabulary is fit on the training rows; a source
category found only in validation/calibration/test maps to an all-zero vector.
It uses logistic regression and the common split, so compare it with
`--methods source_only raw_mlp`. This is an intentional metadata-only leakage
diagnostic, not an admissible observation model. Contract v7 records the
source-field exception, training vocabulary, exact source roster digest, and
encoder implementation digest, and verifies the vocabulary from training
rows.

The versioned schemas are
`datasets/schema/model-input-artifact-contract.v3.schema.json` through
`datasets/schema/model-input-artifact-contract.v7.schema.json`; v1-v7 readers
remain available. Results are simulation-only diagnostics and do not show
that any control predicts measured or deployment channels.

G8.08 remains partial. Other metadata-only, explicit oracle-leak controls, and
physical classical estimators still need isolated routes and audit contracts
before benchmarking. Row shuffling is restricted to the training partition
and is not a time-series or temporal-dependence test.
