# compute

**Purpose.** Measurement-driven resource-budget preflight for scientific runs.

**Public API.**

- `compute.resource_budget` — estimates memory and wall-clock needs from
  measured per-unit costs and refuses runs that exceed the declared budget.
- `compute/schema/` — JSON schema for budget reports.

**Invariants.**

- A budget is a non-qualifying engineering estimate. It never changes a
  scientific status.
- Estimates use measured costs supplied by the caller or recorded in the
  repository; no hidden default is treated as a measurement.
