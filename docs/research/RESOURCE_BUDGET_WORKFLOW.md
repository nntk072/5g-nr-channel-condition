# Measurement-driven resource budget preflight

The `loslab resource-budget` workflow converts a caller-frozen population
design and measured pilot rows into conditional solver, receiver, preprocessing,
downstream-analysis, memory, and retained-storage estimates. It performs no RT,
receiver, or model execution and cannot authorize production generation.

## Build a request

Start from [`compute/schema/resource-budget-request.v1.schema.json`](../../compute/schema/resource-budget-request.v1.schema.json).
Each planned stratum identifies its scene family and exact number of intended
units. Supply 10–100 measured, distinct scene/link conditions for every
stratum. Each pilot observation records digests for the exact configuration
and source measurement record, plus solver, receiver, preprocessing, and
downstream wall time; per-process GPU peak memory; and retained raw-path, CFR,
waveform, manifest, and other artifact bytes. The request includes Python,
NumPy, Sionna, Torch, OS/machine, solver backend, accelerator, driver, and
runtime identities; the workflow recomputes the digest of the canonical
environment object. These are caller-declared pilot measurements and
identities; the workflow does not independently authenticate their instruments
or source artifacts.

Do not fill pilot rows with modeled or generated estimates. Use native measured
observations and retain their source reports separately. The schema requires a
2×–3× headroom factor, following the research plan's resource-budget guidance.

## Run

```bash
python -m loslab resource-budget \
  --request /path/to/measured-pilot-budget-request.json \
  --output /path/to/new-resource-budget-report.json
```

The workflow reads the request as a bounded, stable, single-link regular file,
rejects duplicate JSON keys, hashes the exact request bytes, validates both
schemas, and publishes the report without replacing an existing file. The
report binds the request, both schemas, workflow and CLI sources, runtime
environment, and measurement batch.

## Interpretation

Per-stratum empirical higher p50, p90, and p99 values are multiplied by planned
unit counts and headroom. Device peak memory is reported per process and is not
multiplied by population size. Storage byte arithmetic is integer-exact. Totals
are conditional on the supplied pilot distribution representing its planned
stratum and on linear per-unit scaling; they are not confidence bounds, capacity
guarantees, costs, or approval to scale. With only 10–100 pilot conditions,
empirical p99 is descriptive. The report remains
`MEASUREMENT_CONDITIONAL_NOT_QUALIFIED`.
