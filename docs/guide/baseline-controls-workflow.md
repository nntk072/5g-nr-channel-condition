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

The `metadata_only` control is a broader leakage diagnostic over exactly the
`group`, `domain`, and `source` identifiers. It uses logistic regression and
the same per-seed split as `raw_mlp`. Each field is namespaced before its
training-only vocabulary is one-hot encoded; field values unseen during
training map to zero. This route is deliberately not receiver-observable and
must not be used as a deployment model. The dense diagnostic matrix is capped
at 16,000,000 cells and fails before allocation when the selected cohort would
exceed that limit. Compare with:

```sh
python -m loslab experiment \
  --input <receiver-observations.npz> \
  --output <new-run-directory> \
  --observation precomputed \
  --methods metadata_only raw_mlp \
  --seeds 11 23 47 59 71 \
  --bootstrap 200
```

Model-input contract/schema v8 records the exact metadata field roster,
training vocabulary, full source-roster digests, implementation identity, and
feature-matrix digests. Verification replays the transform using the retained
split indices and dataset identifiers.

`group_only` and `domain_only` isolate the two principal metadata sources when
the combined control signals leakage. Each route consumes only its named
identifier, fits a sorted vocabulary on training rows, maps unseen held-out
values to zero, and uses the same split and logistic-regression baseline as
`raw_mlp`. Their v11/v12 artifact contracts bind the sole allowed source field,
training vocabulary, identifier roster digest, implementation, and feature
matrix, then replay from that field and the retained split. Run either
diagnostic with `--methods group_only raw_mlp` or
`--methods domain_only raw_mlp`; neither is receiver-observable or deployable.

The `oracle_label_leak` route intentionally exposes the true binary target as a
two-column one-hot input. It uses the common split and scoring path only to
check evaluator ceilings and detect leakage-sensitive scoring mistakes. The
route is explicitly non-deployable and must never be included as a predictive
model result. Its v9 artifact contract binds the full target roster and derived
feature matrix and rejects replay when any target changes. Run this isolated
control with:

```sh
python -m loslab experiment \
  --input <receiver-observations.npz> \
  --output <new-run-directory> \
  --observation precomputed \
  --methods oracle_label_leak raw_mlp \
  --seeds 11 23 47 59 71 \
  --bootstrap 200
```

The v9 contract is specific to this intentionally leaking route; earlier
method-specific contract versions remain readable. Do not interpret its
paired score delta as a performance estimate or qualification result.

## Receiver-visible delay and sparse multipath baselines

The opt-in `delay_k_proxy` route derives normalized circular-IFFT delay
moments, PDP concentration, and normalized entropy from observed complex CFR.
Its strongest-bin to remainder ratio is a coarse K-like proxy only: the peak
is not identified as a physical LOS path, and the route has no carrier/DMRS
calibration or path oracle. Delay coordinates are fractions of the CFR record
length, not seconds. The `sparse_multipath` route applies bounded OMP/BIC to
each observed CFR link and feeds normalized fit order, delay spread, residual,
and dominant-fit-power summaries to logistic regression. Its normalized
frequency/delay grid is a mathematical baseline; it does not establish
physical path resolvability or substitute for a calibrated NR receiver.

Both routes consume only `observed_cfr`, share the same per-seed group split
with other methods in one runner invocation, and bind transform and feature
matrix identities in model-input contract v13. Example software workflow:

```sh
python -m loslab experiment \
  --input <receiver-observations.npz> \
  --output <new-run-directory> \
  --observation precomputed \
  --methods delay_k_proxy sparse_multipath observable_features raw_mlp \
  --seeds 11 23 47 59 71 \
  --bootstrap 200
```

The runner's shared split makes these routes comparable on the same sample
partition, but their representations differ. For a receiver-input
parameter-count-matched neural trio, request `--capacity matched` with all
three methods in one run. The runner uses the medium raw-CFR MLP width tuple as
the reference budget, derives real-CNN and complex-CNN widths from the CFR
link/tone shape, and rejects either CNN if its nearest channel-width
configuration differs by more than 10%. Each result records resolved widths,
target/actual trainable parameter counts, and relative difference. This
matches parameter count, not optimizer compute, receptive field, or
representational capacity; it does not imply scientific fairness or
qualification. Do not launch the cohort until the relevant input, split, and
research acceptance gates are ready.

```sh
python -m loslab experiment \
  --input <receiver-observations.npz> \
  --output <new-run-directory> \
  --observation precomputed \
  --methods raw_mlp cnn cvnn \
  --capacity matched \
  --seeds 11 23 47 59 71 \
  --bootstrap 200
```

Independent-site evidence, receiver qualification, and scientific performance
claims remain open. Production generation and advanced training remain gated.

The versioned schemas are
`datasets/schema/model-input-artifact-contract.v3.schema.json` through
`datasets/schema/model-input-artifact-contract.v13.schema.json`; v1-v13 readers
remain available. Results are simulation-only diagnostics and do not show
that any control predicts measured or deployment channels.

G8.08 remains partial. The group-only, domain-only, and combined metadata routes
attribute leakage among group/domain/source identifiers. The two receiver-CFR
physical-feature routes above are software baselines only; other metadata
controls and physical estimators still need isolated routes and audit contracts
before benchmarking. The oracle-label leak is only a non-deployable
evaluator ceiling. Row shuffling is restricted to the training partition and is
not a time-series or temporal-dependence test.
