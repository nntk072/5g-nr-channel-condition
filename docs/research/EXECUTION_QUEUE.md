# Active implementation queue

Checkpoint: 2026-11-11, based on `42e9d08` plus the preserved working-tree
changes. This queue is the executor's next-action list; the research and dataset
plans remain authoritative for scope and acceptance. Update this file only when
a task closes, its dependency changes, or a concrete blocker appears; retain
chronological detail in evidence artifacts rather than duplicating it across plans. Gate IDs in the ledger use
`DS.*` and `RS.*` namespaces so equal historical numbers cannot be confused.

The first ledger snapshot is [REQUIREMENT_LEDGER.json](REQUIREMENT_LEDGER.json).
It indexes 182 backlog rows and 30 named governing gates. Most entries remain
`unknown` / `not_run`: that is deliberate until a source-level audit links an
implementation, verifier and exact evidence. The feature-map parser does not
determine ledger status.

## Current task

### E00 — finish common-runner replay provenance

- **Requirements:** `PKG01-006` (partial common-runner implementation); replay
  contract and runnable producer integration.
- **Status:** focused implementation and verification complete (37 passed).
- **Changed files:** `artifacts/run_replay.py`, its JSON schema, manifest and
  experiment runner integration, CLI replay arguments, and focused tests.
- **Delivered:** a digest-bound command/source/config/input/backend/RNG/attempt
  contract; complete and failed terminal transitions; a common API wrapper
  that writes failure receipts when a started manifest exists; CLI retry
  identity arguments.
- **Known boundary:** failures before the initial manifest exists have no
  replay receipt. Process interruption leaves a durable `running` receipt for
  recovery; no recovery/reconciliation workflow is implemented yet. Other
  experiment producers were found in
  `scripts/run_observable_multipath_experiment.py` and
  `experiments/capacity_sweep.py` and have not been audited for replay coverage.
- **Verification command:**

  ```bash
  ./.venv/bin/python -m pytest --no-cov -o addopts='' -q \
    tests/test_artifacts/test_run_replay.py \
    tests/test_artifacts/test_manifest.py \
    tests/test_experiments/test_runner.py
  ```

- **Closure:** command passes; changed-file Ruff and schema checks pass; check
  the feature-map freshness and `git diff --check`; ledger records common-runner
  coverage as partial, with non-run producers and interrupted-run recovery
  explicitly open. This task does not qualify a scientific run.

## Ready queue

### E01 — reconcile the full requirement ledger

- **Dependencies:** E00 focused verification result.
- **Scope:** inspect every backlog row and every named requirement under both
  governing plans. Replace generic/unreviewed ledger entries with stable source
  references, mandatory/conditional/later scope plus justification, source
  integration evidence, exact dependencies, verification status, separate
  scientific qualification status, closure rule and next task. Add any
  governing requirement missing from the backlog; do not delete or soften a
  requirement. Reconcile split gate naming and distinguish later gated work
  from prerequisite implementation.
- **Deliverable:** complete, source-audited `REQUIREMENT_LEDGER.json`, a concise
  blocker register and an ordered ready queue with finite task cards.
- **Verification:** ledger validator/schema check and a cross-reference audit
  that every backlog row and plan requirement has exactly one ledger record or
  a documented alias. Run
  `./.venv/bin/python -m scripts.validate_requirement_ledger` for structural
  and source-file coverage before the manual source audit.
- **Exit rule:** no `unknown` status remains without a specific inspection task
  and owner/dependency; external blockers are separate from code status.

This audit is divided into fixed batches so it cannot silently become a
general-purpose “review everything” task. `E01.PKG00` is closed: all four
package-0 rows were traced to the feature-map builder/schema/tests and the new
ledger validator. The inventory is structurally executable, but its
classifications remain heuristics and the duplicate-history/helper-wiring
requirements remain open. The validator reports exact row/gate coverage; it
does not claim source audit completion.

The next batch is `E01.PKG01`, then packages 02–10 in numeric order, followed by
`E01.DS.GATES`, `E01.RS.PHASES`, `E01.RS.APPENDIX_E`, and `E01.CROSSWALK`.
Each package batch is the fixed set of `PKGnn-*` IDs in the ledger and closes
only after every row has inspected source/test integration, implementation
state, exact verification, qualification state, dependencies and closure
rule. The three gate batches collect the exact mandatory/conditional scope and
acceptance definitions from their governing sections. `E01.CROSSWALK` checks
complete one-to-one requirement coverage and resolves dependencies/blockers.

For every batch, retain the exact source files inspected in that ledger row;
do not copy the whole chronological status history. If a row is already
implemented, reconcile it and identify its exact verifier rather than
reimplementing it.

### E02 — close the split-to-evaluation integration slice

- **Dependencies:** E01 audit of split, benchmark, lineage and evaluator
  requirements; retain existing signed partition materializer and result
  promotion gates.
- **Scope:** trace frozen parent/child assignments through actual experiment
  consumers, including benchmark cohort availability, descendant membership,
  row/parent/component bindings and inference-time leakage. Implement only gaps
  demonstrated in this end-to-end route; do not add another parallel split
  helper if existing contracts can be wired in.
- **Deliverable:** one runnable consumer path, retained audit artifact and
  finite unsupported-cohort list with precise external dependencies.
- **Verification:** targeted tests for every changed integration boundary and
  a small deterministic fixture covering assignment through evaluation.
- **Exit rule:** consumers use the signed partition identity and validate exact
  parent/child lineage before scoring; the artifact reports unsupported or
  missing cohorts instead of silently treating them as qualified.

### E03 — canonical path, target and storage contract audit

- **Dependencies:** E01 source map for `DS.G1`, `DS.G2`, `DS.G7`, targets and
  serialization; use analytical and already-retained evidence first.
- **Scope:** identify concrete gaps in same-path reconstruction, units/gauge,
  dtype/masks, missingness, arrays, retry/failure state, and model-input
  consumers. Each change must name an original requirement and downstream
  consumer.
- **Deliverable:** closed implementation cards for verified gaps plus exact
  remaining qualification dependencies; no broad rewrite.
- **Verification:** focused round-trip, analytic-oracle and failure-path checks
  for changed contracts.

### E04 — bounded receiver integration and G3 readiness

- **Dependencies:** E01 task-level scope audit; existing G5 receiver work and
  task-specific G3 acceptance definitions.
- **Scope:** map exact required waveform/receiver configurations, implement
  missing interface/criteria plumbing, and distinguish code gaps from backend
  limitations. Freeze run-specific G3 contracts and sampling design before
  any qualifying experiment.
- **Deliverable:** bounded support matrix, receiver-to-evaluator path, explicit
  external blockers and any concrete user decision required for unsupported
  mandatory modes.
- **Entry rule:** do not launch scientific qualification until the applicable
  G3 contract and sampling are frozen. Production D1 RT and advanced training
  remain gated until their existing criteria pass.

## Scientific and release guardrails

- Keep the 2x2 single-V CDL-B covariance miss failed and visible; a diagnostic
  or different cohort cannot replace it.
- Do not reopen City43 Phase-0 with convergence sweeps or call a finite ray
  budget ground truth.
- Use Etoile, Florence and Munich under the accepted ODbL policy. Public
  redistribution still needs release-specific source/extract, conversion,
  attribution/source-offer, third-party-rights and review evidence.
- Do not launch broad CI or expensive/native qualification during prerequisite
  implementation. Run the consolidated software verification only after its
  implementation entry conditions are met.
- Do not mark a phase, requirement or mission complete from plan text, parser
  counts, synthetic fixtures or a passed software contract alone.
