# Benchmark partition materialization

loslab-materialize-benchmark-partitions builds a deterministic, non-copying row index for six benchmark cohorts: train, validation, calibration, site-held-out, site+array-held-out, and real-external.

The command requires a signed canonical Release Split Policy v2 registry and a separately signed benchmark-cohort declaration. The declaration assigns every frozen test parent to exactly one of the three benchmark test cohorts, supplies exact site and array identities for every parent, and binds three evidence digests for each real_external parent. The implementation rejects a site-held-out identity that appears in another cohort, rejects a site+array-held-out array identity that appears in another cohort, and rejects any frozen independent component spanning multiple benchmark cohorts.

The output contains source dataset SHA-256 values and row indices only; it does not duplicate channel tensors or physical parents. Every source row must resolve to a parent in the signed split lock and must agree with its frozen parent type, source family, partition identity when present, and any declared leakage relations. The complete frozen parent roster must be represented by the supplied datasets.

The declaration contract is datasets/schema/benchmark-cohort-declaration.v1.schema.json and the output contract is datasets/schema/benchmark-partition-manifest.v1.schema.json. The declaration signature namespace is umcd-benchmark-cohort-v1@github.com; the split lock keeps the existing umcd-split-policy-v2@github.com namespace.

## Cohort-specific scoring consumer

`experiments.runner.run_experiment` can consume the materialized manifest with
`benchmark_manifest_path`, the signed declaration path/signature/trusted
keyring/signer, `benchmark_dataset_names` in the same order as `paths`, and
`benchmark_test_cohort` set to `site_held_out`, `site_array_held_out`, or
`real_external`. It also requires the signed split-lock arguments. For example:

```python
from experiments.runner import run_experiment

run_experiment(
    ["datasets/benchmark.npz"],
    "outputs/site-held-out-run",
    split_lock_path="artifacts/split-registry.json",
    split_signature_path="artifacts/split-registry.json.sig",
    trusted_split_signers_path="trust/split-allowed-signers",
    split_signer_id="split-reviewer@example.org",
    benchmark_manifest_path="artifacts/benchmark-partitions.json",
    benchmark_declaration_path="artifacts/benchmark-cohorts.json",
    benchmark_declaration_signature_path="artifacts/benchmark-cohorts.json.sig",
    benchmark_declaration_allowed_signers_path="trust/cohort-allowed-signers",
    benchmark_declaration_signer_id="cohort-reviewer@example.org",
    benchmark_dataset_names=["benchmark"],
    benchmark_test_cohort="site_held_out",
)
```

The consumer snapshots each named dataset's bytes and recomputes the manifest
from both verified attestations, the registry, the declaration, and row
metadata. Dataset snapshots reject symlinks/hard links, enforce compressed and
uncompressed size ceilings, and are the same bytes used to build model input.
It confirms that train, validation, and calibration still match the
signed lock, then evaluates only the selected cohort as test rows. Content
addressing through `build_cache_root` is deliberately unsupported in this
route because the exact named input snapshots must remain visible. The run
manifest and per-model result retain selected cohort provenance and list any
unsupported test cohorts with reason codes and external dependencies. If
preflight fails before a run manifest exists, the runner
creates `benchmark-cohort-preflight.json` exclusively and records either a
failed pre-manifest state or an unsupported selected cohort; an existing
receipt is preserved. This is an input-scoped replay and evaluation route only;
it does not turn a candidate manifest or signed external-data claim into
scientific qualification.

This workflow produces CANDIDATE_NOT_SCIENTIFICALLY_QUALIFIED. Signature validity authenticates the supplied declarations but does not establish independent signer custody, authoritative site/array inventories, measurement validity, rights sufficiency, or external-domain scientific qualification.
