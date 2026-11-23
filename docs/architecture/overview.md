# Architecture overview

This page describes how the packages fit together. The repository-level summary
is in [ARCHITECTURE.md](../../ARCHITECTURE.md). The API reference is generated
from the docstrings in [the API section](../api/index.md).

## Layers

Packages are grouped by responsibility. A package may import from the packages
listed above it, never from a package below it that depends on it.

| Layer | Package | Responsibility |
| --- | --- | --- |
| Foundation | `artifacts` | Provenance, manifests, run layout, rebinding of recorded paths and digests |
| Foundation | `configuration` | Loads declarative configuration from `config/` |
| Physics | `propagation` | Channel geometry, visibility, and identifiability |
| Physics | `receivers` | NR receiver chains and pilot handling |
| Data | `datasets` | Episode stores, split policy, scene families, lineage, and public-release checks |
| Integration | `integrations` | Sionna RT, PHY, SRS, PRS, and PBCH adapters; runtime inventory |
| Models | `models` and `learning` | Classifiers and fitting objectives (real and complex CNN, coral, supervised) |
| Evaluation | `evaluation` | Metrics, acceptance criteria, joint evidence, sample-size planning |
| Experiments | `experiments` | Run drivers, acceptance contracts, and audits of retained results |
| Support | `compute` | Scientific resource estimates |
| Control plane | `agentops` | Engineering task queue, verification receipts, and ownership rules |
| Entry points | `scripts`, `loslab` | Command-line tools that call the packages above |

## Flows

1. **Scene and channel generation.** Scene assets and configuration enter
   `datasets`. Ray tracing runs through `integrations/sionna`, and the
   oracle path data is stored with its identity (`datasets/path_oracle.py`,
   `integrations/sionna/rt_identity.py`).
2. **Receiver observation.** Channels pass through `integrations/sionna` and
   `receivers`, which produce the observations a model may use. Geometric
   labels and oracle path data are targets only; they are never classifier inputs.
3. **Evaluation and evidence.** `evaluation` computes metrics and acceptance
   decisions. `experiments` records provenance through `artifacts`. A software
   pass is an engineering status only; it does not change a scientific
   qualification status.

## Classes and data structures

Records are dataclasses or `TypedDict` values, so their fields are explicit and
serializable. Examples are the split-policy records in `datasets/split_policy`,
the episode records in `datasets/episode_store`, and the receiver configuration
in `receivers/nr`. Operations are module-level functions. Classes hold state
only where a stateful resource exists, such as an episode store or a Sionna
adapter. The public functions keep their names: callers, retained result
files, and frozen protocols refer to them by name.

## Verification

- `tools/check` runs the gates. CI runs the same commands; see
  [Continuous integration](ci.md).
- `tests/architecture` checks the package-boundary and structure rules,
  including the absence of runtime import cycles and a README in every package.
- Retained evidence is verified against recorded digests. Renames and moves are
  recorded in `docs/reference/digest-rebindings.json`.
