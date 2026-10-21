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

See [the refactor plan](REFACTOR_PLAN.md) for ownership rules, migration history,
and the phased acceptance gates. The active research workflow is in
[`docs/research/QUICKSTART.md`](../research/QUICKSTART.md).

The automated checks and matching local commands are documented in
[`CI.md`](CI.md).
