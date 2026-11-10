# Benchmark partition materialization

loslab-materialize-benchmark-partitions builds a deterministic, non-copying row index for six benchmark cohorts: train, validation, calibration, site-held-out, site+array-held-out, and real-external.

The command requires a signed canonical Release Split Policy v2 registry and a separately signed benchmark-cohort declaration. The declaration assigns every frozen test parent to exactly one of the three benchmark test cohorts, supplies exact site and array identities for every parent, and binds three evidence digests for each real_external parent. The implementation rejects a site-held-out identity that appears in another cohort, rejects a site+array-held-out array identity that appears in another cohort, and rejects any frozen independent component spanning multiple benchmark cohorts.

The output contains source dataset SHA-256 values and row indices only; it does not duplicate channel tensors or physical parents. Every source row must resolve to a parent in the signed split lock and must agree with its frozen parent type, source family, partition identity when present, and any declared leakage relations. The complete frozen parent roster must be represented by the supplied datasets.

The declaration contract is datasets/schema/benchmark-cohort-declaration.v1.schema.json and the output contract is datasets/schema/benchmark-partition-manifest.v1.schema.json. The declaration signature namespace is umcd-benchmark-cohort-v1@github.com; the split lock keeps the existing umcd-split-policy-v2@github.com namespace.

This workflow produces CANDIDATE_NOT_SCIENTIFICALLY_QUALIFIED. Signature validity authenticates the supplied declarations but does not establish independent signer custody, authoritative site/array inventories, measurement validity, rights sufficiency, or external-domain scientific qualification.
