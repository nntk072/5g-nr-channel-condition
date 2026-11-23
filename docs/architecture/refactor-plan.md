# Repository Refactor Plan

## Goal

Refactor `long_nguyen_thesis-main` into a research-grade repository with the same architectural discipline used in `Aalto_MS_Thesis`, while keeping the domain specific to LOS/NLOS channel classification, Sionna PHY/RT, receiver modeling, domain shift, and publication-grade experimentation.

The refactor must preserve scientific behavior first. Folder movement is useful only when ownership, imports, configuration, experiment provenance, tests, and generated artifacts become clearer.

The migration sequence is:

**freeze current behavior -> establish repository boundaries -> move shared infrastructure -> split `loslab` by domain -> migrate experiments/tests -> quarantine legacy notebooks/code -> remove compatibility paths**

## Implementation status

The structural refactor phases have been implemented on this branch:

- Phase 0 recorded the pre-migration baseline: 47 research tests passed.
- Phases 1-2 added `pyproject.toml`/`uv.lock`, package metadata, configuration
  loading, artifact identity, canonical run folders, resolved-config snapshots,
  manifests, and a completion marker.
- Phases 3-7 moved the research modules into their owning packages, split model
  definitions from fitting objectives, isolated Sionna, and added YAML-driven
  experiment execution.
- Phase 8 moved active tests under `tests/` and added package-boundary checks.
- Phase 9 moved historical notebooks and superseded code under
  `notebooks/legacy_2025/` and `archive/legacy_2025/`.
- Phase 10 retains `loslab` as the documented CLI and compatibility facade.

The final code edits after the last full test run still need a complete test
pass. Native Sionna execution, standardized NR receiver implementation, and
reconstruction of the old CNN/CVNN baselines remain research tasks rather than
folder-migration tasks. The upstream Sionna source reference is documented at
[`SIONNA_REFERENCE.md`](SIONNA_REFERENCE.md) and cloned outside this repository.

## Current repository problems

The current tree mixes several generations of the project:

- the 2025 thesis notebooks and scripts live at repository root;
- the 2026 research implementation is concentrated in one flat `loslab/` package;
- `cnn/`, `mlp/`, and `nr_pusch/` mix model code with historical experiments;
- generated figures live in `output/` while current research commands use `runs/` and `data/` paths;
- research tests use a separate `tests_research/` namespace;
- `recycle_bin/` and `redundant/` remain inside the active source tree;
- dependency and pytest configuration are split across `requirements-research.txt` and `pytest-research.ini`;
- scene assets currently live under `examples/research_scenes/` while the quickstart documents `scenes/`;
- provenance exists, but artifact ownership and output layouts are not yet first-class repository concepts.

The current `loslab` package is a good research baseline and should be decomposed incrementally rather than rewritten.

## Architecture principles inherited from Aalto_MS_Thesis

1. The repository root exposes the major domain boundaries.
2. Importable code lives in explicit Python packages with `__init__.py`.
3. Configuration assets are separate from configuration-loading code.
4. Data, caches, generated outputs, scenes, and documentation are not Python packages.
5. `scripts/` contains thin command wrappers only; reusable logic belongs in packages.
6. Tests mirror package ownership and include architecture/import checks.
7. Generated research evidence has a manifest, provenance, immutable identity, and completion state.
8. Historical notebooks and abandoned implementations cannot be imported by production/research code.
9. Sionna remains an optional integration boundary so feature/evaluation tests can run without native RT dependencies.
10. Every migration phase must retain a working CLI and passing characterization tests.

## Target repository structure

```text
long_nguyen_thesis-main/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── uv.lock                         # once dependency resolution is frozen
│
├── configuration/                  # config loading, schema, hashing, path resolution
│   ├── __init__.py
│   ├── loader.py
│   ├── schema.py
│   ├── hashing.py
│   └── paths.py
│
├── artifacts/                      # manifests, provenance, atomic output handling
│   ├── __init__.py
│   ├── identity.py
│   ├── manifest.py
│   ├── provenance.py
│   ├── atomic.py
│   └── layout.py
│
├── datasets/                       # dataset schema, IO, validation, split ownership
│   ├── __init__.py
│   ├── contracts.py
│   ├── io.py
│   ├── splits.py
│   ├── synthetic.py
│   └── validation.py
│
├── propagation/                    # physical channel and scene-independent physics
│   ├── __init__.py
│   ├── geometry.py
│   ├── physics.py
│   ├── channel_features.py
│   └── visibility.py
│
├── receivers/                      # what the classifier can actually observe
│   ├── __init__.py
│   ├── pilots.py
│   ├── impairments.py
│   ├── estimation.py
│   └── nr/                         # future PUSCH/DMRS/PRS receiver path
│       ├── __init__.py
│       ├── pusch.py
│       ├── dmrs.py
│       └── prs.py
│
├── models/                         # model definitions only
│   ├── __init__.py
│   ├── base.py
│   ├── mlp.py
│   ├── cnn.py
│   ├── classical.py
│   └── factory.py
│
├── learning/                       # fitting objectives/adaptation algorithms
│   ├── __init__.py
│   ├── supervised.py
│   ├── coral.py
│   ├── group_dro.py                # future work, only when implemented
│   └── calibration.py
│
├── evaluation/                     # metrics, statistics, reporting, predictions
│   ├── __init__.py
│   ├── metrics.py
│   ├── calibration.py
│   ├── statistics.py
│   ├── bootstrap.py
│   ├── reporting.py
│   └── plotting.py
│
├── experiments/                    # experiment declarations and orchestration
│   ├── __init__.py
│   ├── contracts.py
│   ├── runner.py
│   ├── variants.py
│   ├── campaigns.py
│   └── cli.py
│
├── integrations/
│   ├── __init__.py
│   └── sionna/
│       ├── __init__.py
│       ├── compatibility.py        # version gates / lazy imports
│       ├── phy.py                  # TDL/CDL generation
│       ├── rt.py                   # ray tracing
│       └── scenes.py
│
├── compute/                        # environment checks and optional HPC wrappers
│   ├── __init__.py
│   ├── doctor.py
│   └── environment.py
│
├── loslab/                         # temporary public CLI / compatibility facade
│   ├── __init__.py
│   └── __main__.py
│
├── config/                         # declarative assets only
│   ├── base/
│   ├── datasets/
│   ├── receivers/
│   ├── models/
│   ├── scenes/
│   └── experiments/
│
├── data/
│   ├── raw/
│   ├── curated/
│   ├── manifests/
│   └── examples/
│
├── scenes/                         # Sionna/Mitsuba scene assets, XML/OBJ/materials
│   ├── generated/
│   ├── reference/
│   └── manifests/
│
├── cache/                          # disposable derived state
│   ├── datasets/
│   ├── channels/
│   └── features/
│
├── outputs/
│   ├── runs/
│   ├── evaluation/
│   ├── figures/
│   ├── reports/
│   └── debug/
│
├── scripts/                        # thin wrappers only
│   ├── data/
│   ├── scenes/
│   ├── train/
│   ├── eval/
│   └── research/
│
├── notebooks/
│   ├── research/                   # active exploratory notebooks
│   └── legacy_2025/                # historical thesis notebooks
│
├── docs/
│   ├── architecture/
│   ├── decisions/
│   ├── research/
│   ├── thesis/
│   └── operations/
│
├── tests/
│   ├── architecture/
│   ├── test_artifacts/
│   ├── test_datasets/
│   ├── test_propagation/
│   ├── test_receivers/
│   ├── test_models/
│   ├── test_learning/
│   ├── test_evaluation/
│   ├── test_experiments/
│   └── test_integrations/
│
└── archive/
    └── legacy_2025/                 # non-notebook superseded source if retained
```

## Ownership rules

### `configuration/` vs `config/`

`config/` stores YAML/TOML/CSV experiment inputs. `configuration/` owns parsing, validation, deterministic merges, path resolution, schema versions, and canonical hashes.

Every experiment should save its fully resolved configuration into its run directory so old runs are never reinterpreted using new defaults.

### `artifacts/`

Move provenance mechanics out of experiment code. Every retained run should have a manifest with at least:

- run ID;
- source revision or source-tree hash;
- resolved config hash;
- dataset identities and SHA256 values;
- scene identity/hash when RT is involved;
- Sionna/Python/framework versions;
- seed;
- model/method identifier;
- parent artifacts;
- completion state.

Use staging paths followed by atomic promotion. A partially written directory must never look like a completed experiment.

### `datasets/`

The existing research contract `h`, `y`, `group`, `domain`, `source`, and metadata becomes an explicit versioned dataset schema.

Feature extraction must continue to consume only observable channel information. Ground-truth geometry or visibility metadata must stay inaccessible to model-input code except in explicitly named oracle diagnostics.

### `propagation/`

Own geometry, physical consistency checks, LOS/NLOS visibility definitions, path-power calculations, delay/path descriptors, and scene-independent physical transformations.

Do not place Sionna API calls here. Sionna-specific adapters belong to `integrations/sionna/`.

### `receivers/`

This becomes a major research boundary because the planned paper needs realistic receiver observations rather than direct perfect CFR access.

The current generic comb-LS implementation moves here first. Later standardized PUSCH/DMRS/PRS estimation, AWGN, CFO, phase noise, timing error, quantization, and interference can be introduced without changing model APIs.

### `models/`, `learning/`, and `experiments/`

`models/` defines architectures. `learning/` defines fitting/adaptation objectives such as CORAL. `experiments/` decides which dataset, model, learning rule, split, seed, and evaluation protocol are combined in a run.

This avoids another monolithic `experiment.py` becoming the center of the project.

### `integrations/sionna/`

All version-sensitive Sionna 2.x imports and API translation live here. Unit tests for the rest of the repository must remain runnable without Sionna installed.

## Current-to-target migration map

| Current path | Target owner |
| --- | --- |
| `loslab/dataset.py` | `datasets/` |
| `loslab/synthetic.py` | `datasets/synthetic.py` or `propagation/` depending on responsibility |
| `loslab/geometry.py` | `propagation/geometry.py` |
| `loslab/physics.py` | `propagation/physics.py` |
| `loslab/pilots.py` | `receivers/pilots.py` |
| `loslab/models.py` | `models/` + `learning/` |
| `loslab/evaluation.py` | `evaluation/metrics.py` + reporting |
| `loslab/statistics.py` | `evaluation/statistics.py` / `bootstrap.py` |
| `loslab/experiment.py` | `experiments/runner.py` |
| `loslab/provenance.py` | `artifacts/provenance.py` |
| `loslab/sionna2.py` | `integrations/sionna/` |
| `mlp/` | canonical `models/mlp.py` after parity tests; then archive/remove |
| `cnn/` | canonical `models/cnn.py` after baseline reconstruction; then archive/remove |
| `nr_pusch/` | `receivers/nr/` + `integrations/sionna/phy.py` |
| root `evaluation.py` | migrate useful code to `evaluation/`; archive superseded remainder |
| root `data_preprocessing.py` | `datasets/` / `receivers/` according to responsibility |
| root `ml_algorithms.py` | `models/classical.py` / `learning/` |
| root `visualization.py` | `evaluation/plotting.py` |
| root `utils.py` | eliminate generic dumping ground; move functions to owning package |
| `tests_research/` | `tests/` mirrored by package |
| `examples/research_scenes/` | `scenes/generated/` or `data/examples/` |
| `output/` | `outputs/figures/` or run-owned artifacts |
| `recycle_bin/`, `redundant/` | delete after Git verification or move required historical evidence to `archive/legacy_2025/` |
| root notebooks | `notebooks/legacy_2025/` unless still part of active research |
| `NguyenLong.pdf` | `docs/thesis/prior_work/NguyenLong.pdf` |

## Refactor phases

### Phase 0 — freeze the baseline

Before moving imports:

1. Run and record `python -m pytest tests_research -q`.
2. Run `python -m loslab doctor` in the supported environment.
3. Run one deterministic debug-data + experiment smoke pipeline.
4. Record hashes for `manifest.json`, result schema, prediction array shapes, and model checkpoint metadata.
5. Save the baseline report under `outputs/refactor_baseline/` or `docs/architecture/evidence/`.

No architecture phase proceeds while unexplained baseline failures remain.

### Phase 1 — repository scaffolding

Create `pyproject.toml` and establish:

- Python version;
- package build configuration;
- runtime dependencies;
- optional Sionna/RT dependencies;
- dev dependencies (`pytest`, `ruff`, `mypy` where practical);
- pytest configuration;
- Ruff import rules;
- CLI entry points.

Create the empty target asset directories (`config`, `data`, `scenes`, `cache`, `outputs`, `scripts`, `notebooks`, `archive`) with short ownership READMEs only where useful.

Acceptance gate: current tests still pass and `python -m loslab ...` still works.

### Phase 2 — artifact and configuration infrastructure

Extract `loslab.provenance` into `artifacts/` and introduce a canonical run layout:

```text
outputs/runs/<run-id>/
├── manifest.json
├── config/
│   └── resolved.json
├── data/
│   └── identities.json
├── models/
├── predictions/
├── evaluation/
├── diagnostics/
├── logs/
└── COMPLETE
```

Do this before broad module movement so later packages share one output contract.

Acceptance gate: existing experiment output fields remain available and newly generated artifacts are atomic and provenance-complete.

### Phase 3 — datasets and propagation

Move dataset schema/IO first, then geometry/physics.

Add characterization tests for:

- exact dataset fields and dtypes;
- group-disjoint split behavior;
- no test-label leakage;
- visibility oracle separation;
- deterministic city generation for a fixed seed;
- scene and dataset hashes;
- LOS examples requiring direct + indirect paths where configured.

Acceptance gate: dataset and geometry tests produce equivalent outputs for fixed fixtures.

### Phase 4 — receiver boundary

Move generic comb-pilot LS code into `receivers/` and define one observation interface used by experiments.

Suggested contract:

```text
Channel realization -> Receiver observation model -> Observable tensor -> Feature/model pipeline
```

Future standardized NR estimation then replaces or augments the receiver implementation without changing experiment orchestration.

Acceptance gate: ideal-CFR and comb-LS experiment modes remain reproducible.

### Phase 5 — models and learning algorithms

Split the flat model module into explicit baselines and learning methods.

The current RF, logistic regression, MLP, and CORAL MLP implementations are
split across `models/` and `learning/`. The old CNN/CVNN material remains
historical under `archive/legacy_2025/`; reconstructing it as a controlled,
publication-ready baseline is a separate research task.

Priority order for that future baseline work:

1. logistic regression and RF;
2. MLP;
3. CORAL MLP;
4. reconstructed CNN baseline from the 2025 work;
5. complex/CVNN baseline only if it remains scientifically required;
6. new methods such as group-DRO only after the baseline architecture is stable.

Add parameter-count and inference-latency reporting here because they are required for a strong paper comparison.

Acceptance gate: fixed-seed model predictions and metrics match the pre-refactor baseline within justified numerical tolerance.

### Phase 6 — Sionna integration split

Replace `loslab/sionna2.py` with small adapters under `integrations/sionna/`.

Keep three boundaries explicit:

- compatibility/version detection;
- PHY TDL/CDL generation;
- RT scene/path generation.

Mock integration tests should target adapter contracts. Native Sionna tests should be separately marked and skipped when dependencies/hardware are unavailable.

Acceptance gate: non-Sionna tests do not import Mitsuba/Dr.Jit/Sionna transitively.
The upstream source reference lives outside the repository at
`/home/nguyenl37/sionna`; native runtime behavior still requires validation on
the intended Linux/CUDA host.

### Phase 7 — experiment orchestration and evaluation

Move experiment execution into `experiments/` and metrics/statistics into `evaluation/`.

Experiment declarations are configuration-driven through the `--config` CLI
option. Prespecified seeds, holdout scenes, model variants, receiver modes,
SNR grids, and bootstrap settings belong in versioned experiment configs.

Every run must persist predictions so statistics can be recomputed without retraining.

Acceptance gate: the E0-E7 research program can be represented as configs/campaigns without editing package source.

### Phase 8 — tests and architecture enforcement

Rename `tests_research/` to `tests/` and mirror domain ownership.

Add architecture tests that fail when:

- production packages import from `scripts/`, `notebooks/`, or `archive/`;
- non-integration modules directly import Sionna;
- tests import reusable code from scripts;
- generated outputs are written to arbitrary root paths;
- `utils.py`-style cross-domain dumping grounds reappear;
- model-input paths access geometric oracle labels.

### Phase 9 — legacy migration

Classify every old notebook/script as one of:

1. reproduced baseline evidence;
2. reusable implementation to migrate;
3. historical thesis artifact;
4. redundant/dead code.

Move historical notebooks to `notebooks/legacy_2025/`. Keep the original thesis PDF under `docs/thesis/prior_work/`. Remove dead files once Git history proves they are recoverable.

`recycle_bin/` and `redundant/` should disappear from the active repository root.

### Phase 10 — compatibility cleanup

During migration, `loslab/__main__.py` should delegate to the new packages so the quickstart continues to work.

`loslab` remains the public umbrella CLI and a small compatibility layer.
Reusable implementation lives in the new domain packages.

## Dependency direction

Target dependency flow:

```text
configuration ─────┐
artifacts ─────────┼────────────────────────────┐
                   │                            │
datasets ──────────┼──> propagation             │
                   │       │                    │
                   │       v                    │
                   └──> receivers               │
                           │                    │
                           v                    │
                        models <── learning     │
                           │          │         │
                           └────┬─────┘         │
                                v               │
                           evaluation           │
                                ^               │
                                │               │
integrations/sionna ────────────┼── experiments│
                                │               │
compute ────────────────────────┴───────────────┘
```

`experiments/` may orchestrate the lower-level packages. Lower-level packages must not import `experiments/`.

## Immediate implementation order

The safest first implementation slice is:

1. add `pyproject.toml` and consolidate pytest/dependency configuration;
2. create `artifacts/` and migrate provenance without changing artifact schema;
3. create `datasets/` and migrate `loslab/dataset.py`;
4. create `propagation/` and migrate geometry/physics;
5. create `receivers/` and migrate pilots;
6. create `evaluation/` and migrate metrics/statistics;
7. create `models/` + `learning/` and split model code;
8. create `integrations/sionna/` and split the Sionna adapter;
9. create `experiments/` and move orchestration last;
10. migrate tests and only then archive/remove legacy paths.

This order minimizes circular dependencies and keeps the existing `loslab` CLI usable throughout the refactor.

## Definition of done

The structural repository refactor is complete when all of the following are true:

- repository root contains no active research notebooks or loose reusable Python modules;
- each importable package has a documented responsibility and public API;
- `python -m loslab` remains functional or is replaced by documented project entry points;
- all tests live under `tests/` and pass from a clean environment;
- Sionna is isolated behind `integrations/sionna/` and remains optional for non-native tests;
- dataset schemas, split identities, scene identities, config hashes, seeds, and source revisions are recorded for every retained experiment;
- runs are written under canonical `outputs/` layouts with manifests and completion state;
- active code never imports from `archive/`, `notebooks/`, or `scripts/`;
- the 2025 thesis implementation is preserved under the legacy archive without polluting the active architecture;
- MLP remains a first-class baseline; controlled CNN/CVNN reconstruction is tracked as follow-up research;
- receiver realism can evolve from ideal CFR to standardized NR observations without changing model/evaluation interfaces;
- configured experiments can be executed from versioned YAML and produce regenerated artifacts rather than manual notebook state.

## Non-goals of the refactor

The folder refactor itself must not be presented as scientific novelty. It exists to make the real research contributions testable and reproducible: physically grounded LOS/NLOS labels, realistic direct-plus-indirect propagation, receiver observability, domain-shift evaluation, scene-level holdouts, uncertainty/robustness methods, and eventual measured-channel validation.
