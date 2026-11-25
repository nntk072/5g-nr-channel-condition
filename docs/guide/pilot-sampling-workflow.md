# UMCD pilot sampling workflow

`experiments.pilot_sampling` builds a deterministic pilot roster from a frozen
design and candidate roster, then audits observed results against that exact
roster. It is a planning and accounting workflow: it does not invoke Sionna,
generate channel data, qualify a receiver, or authorize production D1 runs.

## Freeze coverage before sampling

The design uses `datasets/schema/pilot-design.v1.schema.json`. It declares the
mandatory channel-family and antenna-topology inventories, low and high
difficulty (optionally medium), both `natural_prevalence` and `enriched_stress`
prior types, and an explicit applicable family/topology roster. Every declared
family/topology pair has cells for each required difficulty/prior combination.
Each cell sets the requested number of independent groups. Natural-prior cells
also declare a target-population probability; these probabilities must sum to
one. Enriched-stress cells use null because they are not a prevalence sample.
The prior provenance digest binds the separately reviewed evidence used to
construct the source candidate roster; the tool cannot authenticate that
evidence or infer a deployment prior.

The candidate roster uses
`datasets/schema/pilot-candidate-roster.v1.schema.json`. Each candidate binds a
candidate ID, independent-group ID, all four cell dimensions, and a digest for
its source record. The candidate inventory must already represent the declared
prior. A natural-prior roster must come from the declared deployment sampling
process; an enriched roster is a stress-coverage design and is not prevalence
evidence. The report records each cell's share of the selected roster and
computes a cell-level design weight as target-population probability divided by
sample share. This corrects the declared between-cell allocation conditional on
the supplied candidate frame; it does not correct candidate-frame selection,
within-cell representativeness, or an unverified deployment prior.

## Build the sample

```bash
loslab-pilot sample \
  --design config/pilot/design.json \
  --candidates data/pilot/candidate-roster.json \
  --output reports/pilot/sample.json
```

Selection ranks independent groups by SHA-256 over the frozen seed, cell ID,
and group ID, without replacement. If multiple candidate rows represent one
group within a cell, a second stable rank selects one. Reordering input rows
does not change the sample. The output binds exact canonical input digests,
records requested/available/selected counts and per-cell shortfalls, and stays
`NOT_QUALIFIED`. It binds source-file digests for the sampler and CLI. It exits
nonzero after writing the report when a cell lacks the requested number of
groups.

## Audit observed runs

The outcome manifest uses
`datasets/schema/pilot-outcomes.v1.schema.json`. It has at most one row per
selected candidate, with a result digest, runtime seconds, storage bytes, and
one of `included`, `no_path`, `empty`, `rejected`, or `failed`.

```bash
loslab-pilot audit \
  --sample reports/pilot/sample.json \
  --outcomes reports/pilot/outcomes.json \
  --output reports/pilot/audit.json
```

`datasets/schema/pilot-audit.v1.schema.json` describes the result. The auditor
rejects outcomes outside the frozen sample, preserves missing runs as
`not_run`, and reports per-cell outcome counts plus total, p50, p90, and p99
runtime/storage. A complete diagnostic requires a complete design roster and
one outcome for every selected candidate. Failed, no-path, empty, and rejected
cases count as observed outcomes; they are not silently discarded. Reports
remain `NOT_QUALIFIED` and describe supplied costs only; they do not establish
hardware representativeness, physical truth, release suitability, or scientific
acceptance. The deterministic design seed is not collision-audited against
scientific experiment rosters and does not authorize an experiment. All commands
refuse to overwrite existing outputs.
