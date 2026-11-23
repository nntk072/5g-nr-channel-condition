# Repository architecture

The repository separates reusable Python code, declarative inputs, research
assets, generated evidence, historical work, and tests:

- `configuration/` loads and identifies config; `config/` stores declarations.
- `artifacts/` owns common provenance and run layout mechanics.
- `datasets/`, `propagation/`, and `receivers/` own data, channel physics, and
  observable receiver outputs.
- `models/` defines classifiers; `learning/` owns fitting objectives.
- `evaluation/` owns metrics and statistics; `experiments/` assembles runs.
- `integrations/sionna/` isolates version-sensitive Sionna calls.
- `data/`, `scenes/`, `cache/`, and `outputs/` store inputs and generated state.
- `notebooks/legacy_2025/` and `archive/legacy_2025/` preserve the earlier thesis.
- `tests/` mirrors the active package boundaries.

See [the refactor plan](refactor-plan.md) for ownership rules, migration history,
and the phased acceptance gates. The active research workflow is in
[`docs/guide/quickstart.md`](../guide/quickstart.md).

The automated checks and matching local commands are documented in
[`CI.md`](ci.md).

## Agent control plane

The repository's scientific data plane remains in the existing dataset, propagation,
receiver, learning, evaluation, experiment, artifact, and compute packages.
`agentops/` is an engineering control plane for bounded task execution.

Canonical policy lives in `AGENTS.md` and `.agents/`. `.codex/` and `.claude/`
are thin role adapters. `config/agentops/default.yaml` lists reviewed task cards.
Runtime state is written only beneath ignored `outputs/agentops/` and logs beneath
ignored `logs/agents/`.

The control plane reuses the existing requirement ledger, `artifacts/` provenance
helpers, `experiments/runner.py`, split policy, and `compute/resource_budget.py`.
It does not implement alternate scientific manifests, split registries, experiment
runners, qualification gates, or requirement databases.

Execution is source/card/worktree-bound. Durable task state, provider receipts,
verification receipts, telemetry, and handoffs remain beneath ignored operational
roots. The curated queue reconciles those operational states without rewriting
`docs/research/EXECUTION_QUEUE.md` or `docs/research/REQUIREMENT_LEDGER.json`.

Native resource classes are fail-closed behind explicit resource authorization and
measured host inventory. Portable CPU engineering paths do not import Sionna RT or
allocate GPU resources. See [Agent Operations](../operations/agent-operations.md)
for the operator workflow.
