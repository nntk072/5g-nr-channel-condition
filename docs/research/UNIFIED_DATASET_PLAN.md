# Final Unified Dataset Plan

**Project:** 5G NR Channel Condition / Multipath State Intelligence

**Plan status:** research and engineering design; not a generated dataset, an empirical result, a simulator certification, or a scientific qualification.

**Plan basis:** repository state inspected 2026-11-26; dated evidence and ledger entries retain their own recorded dates.

**Canonical dataset concept:** Unified Multipath Channel Dataset (UMCD), a logical, versioned collection of heterogeneous datasets with explicit provenance and truth boundaries.

## 1. Executive decision

UMCD is one **logical dataset program**, not one homogeneous tensor, one universal label, or one mandatory storage row containing every field. For physical multi-node RT, its canonical parent unit is a `NetworkEpisode`: scene/runtime state, network topology with directed links, an explicitly referenced time/frequency configuration, and source identities needed to interpret propagation. Other families retain their existing atomic record (standardized draw/NR record, legacy `ChannelDataset` row, analytical trial, GBSM drop, CAEZ capture/window, or specialized NR trajectory record). Separate immutable layers represent path-oracle output, physical channel views, receiver observations, targets, counterfactuals, and provenance where those layers exist. A family provides only layers it can truthfully support.

The repository already implements a substantial portion of this architecture. `NetworkEpisode`, `PhysicalChannelView`, `PathOracleTable`, typed `TargetDefinition`/`TargetValue`, receiver observation records, provenance, split lineage, and immutable episode bundles form the current canonical path. The older UMCD-1 plan proposed a new schema and broad package layout against an uploaded source archive. This plan replaces that stale architecture proposal with an evidence-based design around current contracts. It does not authorize a rewrite of those contracts.

The primary research objective is **physically grounded geometric LOS/NLOS classification under 3GPP-to-ray-tracing domain shift**, using receiver-side NR PUSCH/DMRS observations and geometric direct-path visibility as a separately sourced target. The repository's current classifier evidence is a six-scene, RT-only leave-one-scene-out pilot; it does not yet demonstrate the proposed TDL/CDL-source-to-RT transfer experiment. Keep that pilot as a baseline/reproduction track and run the transfer question as a separately identified comparison when its exact data, feature, split and evaluation contracts are frozen. The plan also covers three existing, non-interchangeable research tracks: standardized TDL/CDL profile-proxy analysis; analytical multipath order/identifiability experiments; and measured CAEZ CSI-magnitude forecasting. They share governance, lineage, and some storage/evaluation machinery, but they do not become one benchmark or one metric. TDL/CDL profile class is not geometric LOS. Path-solver output is simulator truth, not physical measurement. Receiver-facing inputs must not contain oracle, scene, source-family, target, or future information unavailable at inference time.

The program advances by evidence gates. Analytical fixtures and standardized-model conformance establish bounded software behavior. Native RT, geometry/material fidelity, receiver performance, held-out scenes, measurement transfer, and release eligibility each require their own evidence. A software `PASS`, a runnable producer, or a schema-valid artifact never promotes scientific qualification status.

## 2. Authority, purpose, and scope

### 2.1 Authority order

Apply the repository’s governing order when this plan conflicts with other material:

1. Signed or frozen scientific protocols and exact normative contracts.
2. Current source code and tests.
3. The active requirement ledger and execution queue.
4. Repository agent constitution and research-integrity/qualification rules.
5. Domain maps and architecture guidance.
6. Platform adapters and reviewed task cards.

This document defines the coherent dataset design and its evidence boundaries. It does not supersede an exact protocol, weaken a ledger requirement, silently change a schema contract, or assert that a queue task is closed. Historical plan prose and results remain historical evidence; this document identifies their current disposition.

**Operational research records.** In this checkout, the active ledger, queue, and detailed research notes reside under git-ignored `dev-docs/research/`; the repository constitution names `docs/research/` for the ledger and queue. Follow the authority paths in the active checkout and confirm the actual files before work; the local ignored records are not distributed by this plan. This plan records no mutable requirement status and does not copy or relocate the operational records.

### 2.2 Research questions and intended claims

UMCD is organized around the current primary objective below, then supports the separately scoped tasks that already exist in the repository. None of the claims is established by a dataset schema or software pass.

| Research question | Required observation/target distinction | Minimum evidence boundary |
|---|---|---|
| Primary objective: does a receiver-observation model trained on controlled TDL/CDL data transfer to geometric direct-path visibility across independent RT scenes? | Standardized source-domain observations and geometric-visibility target-domain observations are different cohorts; receiver-visible PUSCH/DMRS input must be constructed consistently and profile metadata is excluded | Freeze source/target cohorts, waveform and feature parity, exact scene/site independence, validated geometry labels, non-leaky features, powered independent-group inference; report separately from RT-only LOSO |
| Current classifier baseline: does an RT-trained classifier generalize to an unseen RT scene? | Six-scene RT PUSCH/DMRS LOSO pilot and geometric visibility labels | Exploratory only: current reported held-out sets are small and not independently geometry/material-qualified; reproduce from artifacts before comparison |
| Separate: how much profile identity is learned in TDL/CDL A–E? | Profile grouping is a synthetic class proxy, not LOS/NLOS | Profile-held-out evaluation and explicit undefined-metric handling |
| Separate: which multipath order/delays are identifiable from finite-bandwidth pilots? | Analytical path order and relative-delay truth versus observed complex pilots | Analytical model assumptions, gauge/noise/bandwidth contract and group-disjoint evaluation |
| Separate: how well can measured CSI magnitude be forecast? | CAEZ magnitude-only target and time-aligned observation; no geometric truth | Campaign/session chronology, measurement rights, external validity limits and baseline comparison |
| Extension: can dynamic channel/propagation state be forecast or linked to receiver impact? | Causal past observations, future timestamps, dynamic truth, and exact receiver outcomes | Synchronized trajectories, qualified receiver and applicable G3/G5 contracts |

These are potential claims, not promised outcomes. The historical 2025 thesis metrics remain reported historical results until reproduced under a controlled matched-input, matched-budget evaluation. Novelty of a model or representation is an empirical question, not a dataset property.

The repository's currently reported pilot metrics and their limits are summarized below from the tracked [task/output inventory](../architecture/output-tasks-and-channels.md). They are historical results in that report, not rerun or independently qualified by this plan.

| Existing track | Reported result | What it establishes and what it does not |
|---|---|---|
| Six-scene RT LOSO classifier | Large CNN macro AUROC 0.814; raw-IQ MLP 0.806; only 40 held-out links and 4–20 spatial groups per held-out scene | Exploratory synthetic held-out-scene behavior; not independent geometry/material qualification and not TDL/CDL-to-RT transfer |
| TDL/CDL profile proxy | Leave-one-profile-out profile-macro recall 0.932 over five profile groups | Profile grouping is a proxy; a held-out profile has one proxy class, so pooled AUROC is undefined; says nothing about geometric LOS |
| Analytical multipath order | Exact order in 791/960 trials (82.4%); per-order accuracy decreases from 94.6% at order 0 to 62.5% at order 3 | Synthetic identifiability stress under the declared analytical model, not physical path recovery evidence |
| CAEZ CSI-magnitude forecast | Compact indoor-trained Ridge beats persistence indoors/outdoors and loses in hallway; full-tensor indoor persistence beats linear trend | Exploratory magnitude-only forecast; chronological tail and site/UE/protocol confounding, with no phase or LOS/NLOS truth |

### 2.3 External technical and publication basis

The plan was checked against a targeted set of primary technical and publication sources. This is a focused related-work review, not a systematic literature review; publication novelty must be rechecked before a manuscript submission.

| Source | What it establishes | Consequence for this project |
|---|---|---|
| [3GPP TR 38.901 V19.2.0](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.02.00_60/tr_138901v190200p.pdf) and the [Sionna PHY implementation guide](https://nvlabs.github.io/sionna/v2.2.0/phy/api/channel/wireless/3gpp.html) | CDL-A/B/C and TDL-A/B/C are standardized NLOS channel profiles; CDL-D/E and TDL-D/E are LOS profiles. Sionna 2.2 documents the TR 38.901 V16.1 and V19.2 implementations. | A profile label is a model-construction category, not a scene-derived geometric LOS observation or a measured link label. Keep profile-proxy experiments separate; use the repository-pinned V19.2 references until an approved version change. |
| [Sionna RT paper (Hoydis et al., 2023)](https://arxiv.org/abs/2303.11103), [Sionna RT Technical Report (Aït Aoudia et al., 2025)](https://arxiv.org/abs/2504.21719), and the [version 2.2 PathSolver](https://nvlabs.github.io/sionna/v2.2.0/rt/api/paths_solvers.html) and [Paths](https://nvlabs.github.io/sionna/v2.2.0/rt/api/paths.html) API documentation | Sionna RT is a differentiable propagation simulator. Its `PathSolver` returns simulated paths and coefficients; `Paths` derives CIR/CFR representations. In the current API, a fixed seed alone does not ensure determinism unless deterministic mode is enabled; CIR/CFR/taps can normalize delays. | RT output is simulator evidence, not measurement or automatic physical validation. Record solver/runtime settings, determinism mode, normalization, and absolute-delay convention. Test path-to-CFR parity against the exact installed version. |
| [Di Cicco et al., “Machine Learning-Based Line-Of-Sight Prediction in Urban Manhattan-Like Environments,” EuCAP 2023](https://doi.org/10.23919/EuCAP57121.2023.10133145) | A GBDT predicts LOS from link and environment descriptors on synthetic ray-traced Manhattan topologies; the study evaluates unseen locations/topologies and reports AUCPR under strong class imbalance. | LOS classification and unseen-topology evaluation are prior work. The distinction to investigate here is receiver-observation-based NR inference and the precisely controlled statistical-channel-to-independent-RT transfer question, not “first ML LOS classifier” or “first unseen-scene test.” Include deployable geometry/link-feature baselines only as a separately declared information tier. |
| [Gao et al., “CFM-Bench: A Unified Multi-Domain, Multi-Task Benchmark for Channel Foundation Models,” arXiv:2607.14975 (2026)](https://arxiv.org/abs/2607.14975) | The preprint curates six channel domains and multiple tasks, with independence-aware partitions; LOS/NLOS is supported only for its S1 3GPP and R2 MOCSID domains. | Do not claim UMCD is the first unified/multi-domain channel benchmark or first benchmark with LOS/NLOS labels. Compare scope, semantic contracts, project source families, NR observation construction, producer provenance, and qualification boundaries. Treat CFM-Bench as a preprint until its publication status is checked again. |
| [Almutairi et al., “Deep Transfer Learning for Cross-Device Channel Classification in mmWave Wireless,” IEEE MSN 2021](https://doi.org/10.1109/MSN53354.2021.00034) | Prior work studies LoS/nLoS classification using mmWave beam-training messages and transfer across device domains. | Transfer learning for channel-condition classification is also established. Any claimed contribution must isolate its exact input semantics, source/target domains, split unit, and evidence strength. |

**Novelty boundary.** A new classifier, a ray-traced LOS target, domain adaptation, and a multi-family dataset are individually established ideas. The project can claim a contribution only after literature review and experiments show a specific, falsifiable advance—potentially the combination of standards-constrained NR receiver observations, controlled TDL/CDL-to-independent-scene RT transfer, explicit geometry/profile target separation, and auditable source-to-target evidence. This combination is a candidate framing, not a novelty finding. Compare the proposed result to the above sources and to the relevant baselines in the local research program before choosing a manuscript claim. The initial comparison matrix must include: (a) the reproduced six-scene RT LOSO baseline; (b) TDL/CDL-source to independent RT-scene zero-shot transfer; and (c) any adaptation method as a separately declared protocol with its target-data access (unlabeled versus labeled), validation groups, and prohibition on test-group adaptation. Add an environment/link-feature GBDT comparator from prior LOS-prediction work only as a distinct information tier; it is not an observation-only receiver baseline. Use natural-prevalence test cohorts and report PR-based and per-class results alongside AUROC, with uncertainty resampled at the independent scene/site level.

### 2.4 Out of scope without separate authorization

- Production D1 RT generation, broad native sweeps, advanced training, or qualification runs without their existing authorization and frozen criteria.
- Claiming absolute physical truth from a simulator, a scene label, a path solver, or a successful software test.
- A universal binary LOS label across analytical, standardized, scenario, RT, and measured domains.
- Assigning geometric oracle truth to measured data lacking the relevant geometry and calibration.
- Treating RCS/ISAC targets, radar observations, or material inference as ordinary communication-channel labels.
- Promising a fixed corpus size, scene count, throughput, or publication result before profiling, power planning, source eligibility, and release review.

## 3. Repository setup, rules, and implementation ownership

### 3.1 Project setup and command surface

- Python support is 3.11+; portable development uses `uv` with the committed [`uv.lock`](../../uv.lock) and [`pyproject.toml`](../../pyproject.toml). Sionna PHY and Sionna RT are optional extras pinned to the repository’s current 2.2.0 integration target. A native result is valid only for the host/backend and runtime it records.
- [`config/packages.toml`](../../config/packages.toml) is the single package/test-root inventory used by [`tools/sync_packages.py`](../../tools/sync_packages.py) to synchronize package lists in packaging, lint, mypy, coverage, and build settings. The shipped packages are `artifacts`, `configuration`, `datasets`, `propagation`, `receivers`, `models`, `learning`, `evaluation`, `experiments`, `integrations`, `compute`, and `scripts`; `tests` is the test root. Do not add a dataset package by editing only [`pyproject.toml`](../../pyproject.toml) or by inventing a compatibility package.
- [`tools/check`](../../tools/check) is the supported gate entry point. Its current gates include `lock`, `lint`, `typecheck`, `architecture`, `test`/`core`, `neural`, `build`, `docs`, and changed-file `fast`. [`CONTRIBUTING.md`](../../CONTRIBUTING.md) requires lock, lint, typecheck, architecture and portable tests before a code PR. Fast PR CI and nightly full coverage have different roles; consult [GitHub Actions](../../.github/workflows/ci.yml) and [GitLab CI](../../.gitlab-ci.yml) before any implementation change.
- `pyproject.toml` excludes `native_sionna` and `evidence_integrity` from default pytest selection and defines the product coverage source. The latter is cryptographic evidence integrity rather than 5G NR behavior. Do not claim a skipped native/evidence suite passed. Native RT needs the separate RT extra and host evidence; no generic test/check may launch it.
- The docs build uses Sphinx with warnings as errors, but [`docs/conf.py`](../conf.py) explicitly excludes `research/**` from the published site. The final dataset plan is repository research guidance, not an end-user site page. Keep it out of Sphinx toctrees; make its repository location discoverable from [`docs/README.md`](../README.md).
- The root [`README.md`](../../README.md) states that no repository license has been chosen. Do not imply that source code, generated data, weights, scene assets, or this plan have a repository-wide redistribution license. Per-asset and per-dataset rights remain separate decisions.

### 3.2 Current engineering and scientific rules

The dataset plan must be carried out under the current root [constitution](../../AGENTS.md) and [`.agents/rules/`](../../.agents/rules/README.md), not by introducing its own workflow:

- Preserve dirty work, active jobs, logs, checkpoints and evidence. Inspect callers before behavior changes; keep one named writer per mutable file. These are the current [core execution rules](../../.agents/rules/core-execution.md).
- Frozen protocols and normative contracts outrank source/tests only in the authority order specified by the constitution; the requirement ledger and queue track current dispositions and work. In this checkout the constitution points to `docs/research/REQUIREMENT_LEDGER.json` and `EXECUTION_QUEUE.md`, while actual copies were found only under ignored `dev-docs/research/`. This is a real repository-policy inconsistency and must be resolved by the repository owner before the plan can cite a clone-accessible mutable ledger.
- Never convert software `PASS`, mock/analytic fixtures, or parser output into qualification. G3 requires exact accepted G5 receiver evidence and a task-specific powered design. Native Sionna, GPU, scheduler, broad sweep, and production D1 work require their governing authorization and selected host.
- Preserve negative results, including retained G4 and City43 evidence. Do not replace a failed or negative cohort with a favorable cohort. Internal research permission is not public redistribution permission.
- The [branch and review workflow](../../.agents/rules/branch-pr-workflow.md) uses purpose-based names such as `docs/`, `refactor/`, or `fix/`, not agent/tool prefixes. This document does not create gate-named files: G0–G8 are requirement/gate cross-references, not filenames. The rules do not authorize commits, pushes, merges, remote workflow reruns, or release publication.
- Operational task state and logs stay under ignored state roots. `artifacts/`, the actual `experiments/runner/` package, split policy, and `compute/resource_budget.py` keep their existing semantic ownership; do not move those responsibilities into the dataset package. The supplied constitution names `experiments/runner.py` as the semantic owner; current source and imports implement that responsibility in the `experiments/runner/` package. Follow the actual package entry points and callers; no compatibility module is needed for dataset work.

### 3.3 Package ownership and dependency direction

The actual package graph in `ARCHITECTURE.md` and `docs/architecture/overview.md` is the implementation boundary. The final dataset approach uses each package for its existing responsibility:

| Package | Existing responsibility | Dataset-plan rule |
|---|---|---|
| `configuration` | Resolve YAML/JSON configs, reject unknown/duplicate keys, identify resolved settings | Use it for declarative run configuration; do not put producer logic in config loaders. |
| `artifacts` | Run identities, write-once layouts, manifests, build-DAG mechanics and replay provenance | Use for experiment-run mechanics; dataset object semantics remain with their producer/store. |
| `propagation` | Geometry, independent visibility, physics primitives and identifiability | Keep geometry oracle independent of Sionna path output; do not import native RT here. |
| `receivers` | Framework-independent NR receiver contracts and pilot processing | Do not label the generic pilot adapter standards-compliant NR; Sionna receiver code belongs in integrations. |
| `datasets` | Episode/channel/path/target/provenance/split/release contracts and schemas | Own canonical dataset meaning and immutable bundle interfaces. |
| `integrations` | Optional Sionna PHY/RT adapters and runtime identity | Keep native imports lazy; record conversions and host-specific evidence. |
| `models` | Model definitions and classical estimators | No training loops or oracle features in model definitions. |
| `learning` | Training and fitting | Consume frozen split registries; never generate their own split. |
| `evaluation` | Metrics, statistics, audits, acceptance and qualification contracts | An acceptance scorer does not qualify its inputs or source dataset by itself. |
| `experiments` | Experiment and runner lifecycle (`experiments/runner/`) | Own experiment execution and retained result workflow; do not move it into dataset storage. The supplied constitution's `experiments/runner.py` path differs from the current package layout; follow the actual package entry points and callers. |
| `compute` | Measurement-driven resource estimates | Use measured costs; an estimate is not authorization or qualification. |
| `scripts` | Thin CLI wrappers and audit/report entry points | Keep reusable logic in packages. Current scripts/console entries are the CLI surface; `loslab/` is not in `config/packages.toml` or present as a package in this checkout despite stale prose elsewhere. Do not invent it as part of this plan. |

Module dependencies follow the tested architecture direction in `ARCHITECTURE.md`; do not introduce import cycles or reverse package ownership to make a producer easier to call. New packages require a demonstrated domain boundary, a `config/packages.toml` entry, package README, synchronized configuration, and architecture-gate evidence. This plan currently proposes no new package.

**Configuration and asset inventory.** The committed configuration surface is presently selective, not a complete catalog of every proposed UMCD family: `config/packages.toml` is the package source; `config/scenes/scene-family-registry.v1.json` and `config/reference/licenses/odbl-osm-scene-release-policy-20261023.json` define candidate and ODbL policy inputs; `config/datasets/rt_trajectories_pilot/` contains bounded RT trajectory pilot configs for box screen, Étoile, simple street canyon, and Urban41/42/43; `config/experiments/debug_smoke.yaml` is a debug smoke configuration; and `config/reference/` holds standardized reference/evidence JSON and the target-family producer-consumer matrix. `config/models/`, `config/receivers/`, and `config/datasets/` also contain guidance READMEs, not a complete family-wide parameter registry. `scenes/generated/research_scenes/` currently contains City41/42/43 XML and manifests; a directory name does not establish per-asset completeness, independence or redistribution permission, and the registry may identify candidates outside this directory. Do not propose a full Cartesian configuration matrix or mark a scene family supported merely because a config or directory exists. Add each configuration only for a queue-backed, producer-supported, contract-reviewed use; bind its resolved content through `configuration/loader.py` and the applicable run identity.

`archive/legacy_2025/` and `notebooks/legacy_2025/` are referenced by historical planning material but are absent in this checkout; do not treat those references as active producer paths or verified current results. `data/` exists but is git-ignored; never rely on a local dataset there as release evidence. Results and scientific artifacts remain with their existing `artifacts/`, `outputs/`, and research evidence owners, not embedded into this plan.

### 3.4 Actual producer-to-consumer routes

The plan’s first execution route is to extend these real routes rather than invent a second generation framework. Exact calls must be rechecked at implementation time; this map records current source anchors and the intended contract boundary.

| Family/workflow | Current entry and implementation route | Output/consumer boundary |
|---|---|---|
| Legacy single-link static RT dataset | `scripts/generate_rt_dataset.py` → `integrations/sionna/adapter.py`: `generate_raytraced()` or `generate_raytraced_cached()` → `ChannelDataset` artifact | This is the existing compact legacy route. It is not the multi-node episode writer. Preserve it for its consumers and candidate/cache identities; do not relabel its output as an episode bundle. |
| Static multi-node RT episode | `integrations/sionna/rt_network.py`: `build_rt_network_generation_request()`, `run_authorized_static_network_episode()`, path/view conversion and `write_sionna_rt_episode_bundle_from_paths()` → `datasets.episode_store.write_episode_bundle()` | Canonical multi-node episode library route with existing D1 authorization and receipt. Repository search found no `scripts/` CLI calling `run_authorized_static_network_episode()`; do not claim `generate_rt_dataset.py` invokes it or create a new CLI without a queue-backed requirement. |
| Dynamic RT trajectory | `scripts/generate_rt_nr_trajectory.py` → `integrations/sionna/rt_trajectory.py`: `generate_rt_nr_trajectory()` | Writes the specialized NR trajectory store; with optional `umcd_store_root`, the same retraced solves additionally publish path truth, absolute channel views, geometric-LOS targets and receiver observations as UMCD layers. Without that argument, do not claim a UMCD bundle was written. |
| Standardized PUSCH/DMRS | `scripts/build_standardized_pusch_dataset.py` → `integrations/sionna/tdl.py` or `cdl.py` → `datasets.standardized.standardized_run_from_channel()` → `integrations/sionna/standardized_pusch.build_standardized_pusch_dmrs_records()` → `datasets.nr_store.write_nr_record()` | Specialized NR record store, not the episode bundle. Keep profile proxy, generated channel truth and receiver observation distinct. The current profile LOS proxy is not geometric LOS and diagnostics are not receiver qualification. |
| RT-derived DMRS legacy records | `scripts/build_rt_dmrs_observation_dataset.py` → `integrations/sionna/rt_dmrs.py`: request builder and `build_rt_dmrs_observation_dataset()` | Current compact `ChannelDataset`/NR observation path remains a compatibility producer. Preserve `label_semantics`, child split binding and absolute/relative power wording; migrate only with a consumer-backed bundle adapter. |
| Scene-family sample registry | `scripts/sample_scene_families.py` → `datasets/scene_families.py` | Candidate registry binds eligible asset tree and split relations; current report is not scene qualification. Other producers/caches must not bypass the registry where the active task requires it. |
| Split lock and child audit | `scripts/build_channel_split_registry.py`, `scripts/build_standardized_split_registry.py`, `scripts/bind_split_registry_children.py` → `datasets/split_policy/`, `datasets/split_lineage.py` | Freeze parents before children/model access; signed locks are not silently regenerated. |
| Benchmark materialization | `scripts/materialize_benchmark_partitions.py` → `evaluation/benchmark_partitions.py` and dataset readers | Materialized cohorts inherit exact locked membership and carry support/audit reports; materialization does not prove independent sites by itself. |
| Public release | `scripts/export_public_release.py` → `datasets/public_release.py` and release schemas | Release validation is separate from internal research eligibility and requires rights, inventory, lineage and approval evidence. |
| Model experiment | `scripts/run_channel_experiment.py` / `experiments/runner.py` → `datasets` reader and split contract → `models` / `learning` → `evaluation` → `artifacts` | Experiment runner owns execution and evidence; path/oracle/source/scene metadata stay outside model inputs. |

**Prescribed integration strategy:** keep the existing episode bundle as the canonical multi-node storage route; retain family-specific typed records when their source semantics differ; bridge a producer only where a concrete consumer needs a canonical episode layer and the mapping is lossless or explicitly lossy. The bridge must bind its source artifact, axes, target semantics, validity, provenance and split parent. No global “convert everything to one tensor” step, no parallel `UMCDv2` core, and no gate-named implementation file are part of the design.

### 3.5 Current implementation audit and design disposition

#### 3.5.1 Canonical components to retain

| Concern | Current source of truth | Current behavior and plan disposition |
|---|---|---|
| Multi-node episode and topology | `datasets/network.py` | Explicit node roles, array specs, port-element weights, directed link direction/role, per-frame poses and time axes. Use as canonical parent/link model; preserve directed TX→RX semantics. |
| Physical complex channel | `datasets/channel_view.py` | Per-link named layout `[frame, rx_port, tx_port, subcarrier]`; explicit carrier and baseband offsets, delay reference, power scale, optional phase gauge/time-alignment digest and element-validity mask. Invalid entries are zero-stored and mask-declared. Keep named axes and metadata; never silently flatten for storage or comparison. |
| Path truth | `datasets/path_oracle.py` | CSR-style ragged path groups keyed by link/frame/RX/TX port, absolute delay, complex gain convention, explicit `SOLVED`, `NO_PATH`, `SOLVER_FAILURE`, `UNCONVERGED`, and optional Doppler/angles/interactions. Keep solve outcome for every expected group. |
| Targets | `datasets/targets.py` | Versioned semantic definitions and element-wise validity/reason codes. Canonical validators bind recognized meanings to producer/reference/aggregation. Extend only with producer-backed meaning; keep historical target versions readable where allowed. |
| Receiver observations | `datasets/nr.py`, `datasets/observations.py` | Typed NR and frequency-domain observations, model-input structures, timing and resource metadata. Maintain separation between received measurements, receiver estimates, oracle truth, and model inputs. |
| Provenance and lineage | `datasets/provenance.py`, `datasets/lineage.py`, `datasets/split_lineage.py` | Immutable run provenance, source/config identity, asset digests and lineage claims; signed split and child-artifact relationships. A digest establishes byte identity, not scientific correctness or signer authority. |
| Bundle storage | `datasets/episode_store/` | `umcd.bundle.v1` immutable layer objects; manifest is published last; pickle-free JSON/NPZ; deterministic iteration and size estimation. Retain reader/writer behavior and layer separation. |
| Splits | `datasets/split_policy/`, `datasets/splits.py`, `datasets/independent_splits.py` | Signed policy/registries and leakage relations; derive splits at parent independence identities before model exposure. Do not regenerate a frozen lock silently. |
| Compact compatibility dataset | `datasets/channel.py` | Legacy `ChannelDataset` is dense `[N, antenna_or_link, frequency]` with binary `y`, labels requiring `label_semantics`, and row metadata. Retain as a compatibility/benchmark adapter; do not treat as canonical UMCD episode storage. |
| Producer families | `datasets/analytical.py`, `standardized.py`, `scenario_gbsm.py`, `scene_families.py`, Sionna integrations and scripts | Producers are heterogeneous and at differing maturity. Each must explicitly declare the layers and semantics it produces; registration is not qualification. |

The package inventory includes 35 Python source files at depth two under `datasets/` (29 directly under it and six in subpackages) and 195 top-level schema files at inspection. That breadth is a reason to inventory callers and contracts before consolidation; file count alone is not evidence of duplication. `datasets/episode_store/README.md` documents compatibility-preserving module extraction and a broad export surface that includes private helpers. Future cleanup should reduce accidental coupling carefully, not rename/remove exports based only on style.

### 3.6 Dataset module inventory

The dataset plan uses the following existing modules as the implementation inventory. This is a responsibility map, not a proposal to merge modules by naming similarity.

| Existing module(s) | Current responsibility | Prescribed use in UMCD |
|---|---|---|
| `datasets/analytical.py`, `datasets/synthetic.py` | Analytic/control data and synthetic debug fixtures | Keep analytic truths as bounded control families; keep synthetic fixtures out of scientific corpus claims. |
| `datasets/caez.py` | CAEZ-5G measured dataset access/processing | Treat the available magnitude modality as a distinct measured-observation family. Do not infer complex phase, absolute ToA, path count, or geometric LOS. Preserve source/license/protocol and chronological/site confounding limitations. |
| `datasets/channel.py` | Legacy compact `ChannelDataset` and model-input wrapper | Compatibility adapter for existing classifier/experiment workflows; labels always carry semantics and metadata remain outside the model payload. |
| `datasets/channel_view.py`, `datasets/network.py`, `datasets/time_axis.py` | Physical per-link view, network episode/topology and time alignment | Canonical episode/channel axis contracts; retain declared port ordering, carrier, delay/power reference and clocks. |
| `datasets/path_oracle.py`, `datasets/path_evolution.py` | Ragged path solve truth and temporal path evolution | Oracle/evidence layer only; define path matching and continuity only under its own audit contract. |
| `datasets/targets.py`, `datasets/observations.py`, `datasets/nr.py` | Target definitions/values, frequency-domain observations and NR receiver records | Model inputs, targets, receiver estimates and hidden truth remain distinct typed records. |
| `datasets/nr_store.py` | Immutable stored NR observation records | Keep as the specialized NR record store unless a specific consumer requires a validated bundle bridge; do not duplicate it with generic storage. |
| `datasets/standardized.py`, `datasets/standardized_specs.py` | Standardized TDL/CDL records and configuration/specification semantics | Preserve model/profile identity and table-version semantics; do not derive geometric scene truth. |
| `datasets/scenario_gbsm.py` | Scenario GBSM records | Keep scenario-model output separate from ray-traced path truth and from empirical geometry. |
| `datasets/counterfactual.py`, `datasets/frequency_pair.py` | Direct-path interventions and paired frequency views | Preserve parent/child pairing, changed/held-constant variables and shared split membership. |
| `datasets/scene_families.py`, `datasets/rt_split_identity.py`, `datasets/splits.py`, `datasets/independent_splits.py`, `datasets/legacy_rt_split.py` | Candidate families, RT identities and split derivations | Build identities and family/site relations before generation/evaluation; legacy split support never overrides signed current policy. |
| `datasets/split_policy/`, `datasets/split_lineage.py` | Signed split policies, registries and child-artifact lineage | Sole authority for frozen membership and inherited child assignment. |
| `datasets/provenance.py`, `datasets/lineage.py` | Run-layer provenance and lineage claims | Bind the producing layer and source identity; do not imply trust or validity from hash fields alone. |
| `datasets/episode_store/` | Immutable bundle publication/read and shard iteration | Canonical multi-layer bundle route. Keep store mechanics separate from model-training and experiment execution. |
| `datasets/model_input_audit.py` | Oracle/metadata feature-boundary audit | Audit serialized model inputs and all feature-construction paths. |
| `datasets/public_release.py` | Release manifests/approval and public artifact checks | Apply only after corpus inventory, rights, privacy and lineage admission; no implicit release by writing a bundle. |

Additional data-plane owners are outside `datasets/`: `propagation/` owns geometry and independent visibility; `receivers/` owns framework-independent receiver contracts; `integrations/sionna/` owns optional simulator/waveform adapters; `evaluation/` owns audits and acceptance; `experiments/` owns run execution; `artifacts/` owns run artifact mechanics; `compute/` owns measured resource estimates. Do not relocate these domains into `datasets/` for superficial unification.

### 3.7 Status snapshot and interpretation

The local ledger snapshot inspected for this plan contains 316 requirement rows: 185 marked implementation-complete, 125 partial, and 6 absent; verification is 207 pass, 87 not run, and 22 incomplete; qualification is 306 not qualified, 8 qualified within bounded scope, 1 not applicable, and 1 failed. These are ledger classifications at inspection, not a fresh test run and not corpus-level completion. The current research edition log last records Edition 3.775 dated 2026-11-12. Recompute counts from the current active ledger before making any later status declaration.

The scene-family candidate registry evidence in Edition 3.775 reports only 2 of 12 registered scene families with eligible candidates, S00 analytic roster gaps, S03–S11 unsupported, producers beyond the direct RT path not fully wired, no family qualified, and G1/G3/G7 plus production RT still open/gated. This is a dated software/source audit, not a current re-audit or authorization to generate.

### 3.8 Disposition of historical greenfield proposals

The historical master plan's illustrative “canonical storage schema v1,” proposed `umcd_*` APIs, and greenfield file tree describe an earlier design state. The episode/store contracts now exist. This plan instead provides a **contract map**: which current type owns each concept, which family producer feeds it, and what capability or qualification remains missing. A schema change is justified only by a concrete producer/consumer requirement that cannot be expressed by the current versioned contract. A breaking migration is not the default.

## 4. Dataset ontology and entity relationships

### 4.1 Logical organization

UMCD has three orthogonal taxonomies that must not be conflated:

1. **Source/data family** says how channel or observation data are produced: analytical, standardized stochastic, scenario-based stochastic, geometric RT, NR receiver, measured, or derived/stress data.
2. **Scene-family registry** says what environment/asset family a geometry candidate represents (the current candidate registry names S00–S11). A source family can contain several scene families; a scene family is not a source family.
3. **Task/target family** says what inference/evaluation question is being asked: geometric visibility, channel estimation, multipath structure, forecasting, receiver impact, and so on.

An artifact must preserve all applicable classifications and must not infer one taxonomy from another.

### 4.2 Canonical episode and family-specific atomic units

`NetworkEpisode` is the canonical parent for physical multi-node RT episodes: scene/runtime state, carrier and frequency grid, time axis, nodes/arrays, directed links, and supported pose state. It is not a universal row or mandatory wrapper for every family. Use the existing family-specific atomic unit where that is the actual data contract: standardized channel draw/NR record, legacy single-link `ChannelDataset` row, analytical trial, scenario GBSM drop, CAEZ capture/window, or specialized NR trajectory record. Bind derived records to a common parent only when the source contains the required relationship and the adapter can express it without inventing values. A generated/observed channel view, path solve, target, receiver observation, or counterfactual in an episode-backed family is a child layer bound to that exact parent and its own provenance.

**Episode identity rules (apply only to episode-backed families):**

- Episode identity binds the full topology and physical configuration plus source/scene/runtime identity and time/frequency context. Identity construction must use the current canonical identity helpers; a human-readable ID alone is not identity evidence.
- Link identity is parent-scoped and directed. Reciprocal directions are separate unless an explicit physical reciprocal contract binds them.
- Array identity distinguishes physical elements, their world pose, polarization, logical ports, and port-to-element weights. RF chains are not inferred from element count or port count.
- Frame identity binds episode time-axis index and timestamp; a trajectory sample is not an independent episode merely because it has a different frame ID.
- A channel view is keyed by episode/link and exact frame, port, carrier, baseband-frequency, delay, power, and gauge contract.
- A path group exists for every required link/frame/RX-port/TX-port solve, including explicit empty and failed outcomes. Omitting failed groups is not equivalent to no path.
- A receiver observation binds waveform/resource mapping, receive chain, estimator, noise/impairment context, output axes, and exact source channel/time alignment.
- A target definition and its values bind semantic name/version, units, axes, source/reference, aggregation, validity, and any censoring/confidence contract.
- Counterfactuals bind the parent episode, changed variable, unchanged variables, paired child identities, and the causal interpretation permitted by the intervention.

### 4.3 Access layers and trust boundaries

| Layer | Meaning | Allowed consumer | Forbidden shortcut |
|---|---|---|---|
| Source/config/scene | Producer inputs, assets, solver setup, runtime and license | Reproduction, stratification, auditing under policy | Treat source metadata as receiver input |
| Path oracle | Solver-provided path events, geometry visibility, interactions, solve outcomes | Target creation, physical audit, oracle baselines explicitly marked as such | Feed hidden truth to deployed model or call returned rays measured physical paths |
| Physical channel | Complex channel view derived from a declared path/model result | Upper-bound or channel-domain task when named as ideal/simulated | Call it receiver-observed CSI without a receiver path |
| Receiver observation | Received waveform/pilots/estimated CSI with declared receiver path | Deployment-facing model input if acquisition assumptions match | Use future samples, labels, profile, scene, path or hidden source fields |
| Target | Semantically defined outcome and validity | Loss/scoring only for eligible task/family | Universalize labels across incompatible sources |
| Provenance/lineage | Immutable origin, identity, derivation, split and custody evidence | Reproduction, audit, leakage controls | Equate hashes/signatures with scientific validity |

Model-input audits must check actual serialized feature payloads, not only dataclass field names. Oracle layers remain separately stored and access-scoped. Feature builders receive explicit model-input records, never the bundle wholesale.

## 5. Physical, temporal, frequency, and topology contracts

### 5.1 Channel tensor contract

The current canonical `PhysicalChannelView` is per directed link with axes exactly `("frame", "rx_port", "tx_port", "subcarrier")`. It binds:

- strictly increasing baseband subcarrier offsets including DC and positive absolute RF frequencies after adding `carrier_hz`;
- explicit frame timestamps and episode alignment;
- explicit RX/TX port IDs in the link-local order declared by the parent topology;
- finite complex values and a boolean element-wise validity mask with zero storage for masked elements;
- `delay_reference` and `power_scale`; relative-delay or normalized-power views require the exact normalization recipe;
- optional named `phase_gauge_id`; matching gauge declarations are necessary but not proof of phase calibration;
- source-time-alignment digest when the source timelines require alignment.

Do not silently conjugate, transpose, reorder, reshape, center, normalize, interpolate, or phase-align channel data. Any such derivation is a named operation with a new view identity and provenance. Preserve the carrier-phase term when reconstructing path coefficients into CFR. Comparisons reject unmatched axes, time, ports, carrier, grid, delay reference, power reference, or gauge unless a frozen method explicitly defines an admissible transform.

Dense `PhysicalChannelView` is the present channel-view contract, not a universal promise that every source can provide every dense axis. Variable-size path lists remain ragged. Missing channels or unavailable estimates use masks/reasons, not fabricated zero values. Sparse variable-size channel encoding and full legacy single-link migration remain open requirements in the ledger.

### 5.2 Path-oracle contract

Path delays are absolute propagation delay under the declared time reference; coefficients use the pinned `sionna_rt_path_coefficient_v1` convention unless a new separately validated convention is added. Optional fields—Doppler, AoA/AoD, path kind, object/primitive IDs, interaction type and vertices—are all-or-none at table scope in the current type where specified. Do not infer physical scatterer identity from solver interaction IDs outside their scene/runtime namespace.

Each expected solve key has one status:

- `SOLVED`: one or more valid paths under the declared solver contract.
- `NO_PATH`: solver completed with an explicit empty result; it is not automatically a claim of physical radio outage.
- `SOLVER_FAILURE`: solve failed; never convert empty backing arrays to zero channel.
- `UNCONVERGED`: solver did not meet its declared convergence criterion; never use as physical truth without an explicit qualified policy.

Numerical path-to-CFR derivation must define delay phase, carrier phase, frequency offset, complex precision, summation order/tolerance, direct/indirect masking source, and path status handling. The shared `reconstruct_path_oracle_cfr()` route is the current code path; native `Paths.cfr()` parity and calibrated received-power truth remain separate evidence needs. Sionna 2.2 exposes delay normalization options on CIR/CFR/tap derivations and defaults `normalize_delays=True`; preserve absolute delay for canonical path truth and record every normalized view as a distinct derivation. `PathSolver` is nondeterministic by default: a fixed seed alone is not a deterministic replay guarantee, so use and record deterministic mode where promised and bind the runtime/backend identity. Record enabled path mechanisms, `max_depth`, `samples_per_src`, and `synthetic_array`; the API documents `max_depth=0` as LoS-only candidate generation, and synthetic arrays apply per-element phase shifts from a point-source representation. The RT 2.2 solver supports diffraction, but the API describes it as first-order edge/wedge diffraction, so do not describe this as general higher-order diffraction physics.

### 5.3 Time and clock semantics

- Each episode declares the primary time axis in seconds, a canonical ordered timestamp sequence, and any source-clock alignment report. No implicit frame-rate inference.
- Node, antenna-array, scene-object, channel, and waveform timestamps must state whether they share a clock. Asynchronous sources require explicit alignment, uncertainty, and validity intervals; do not force them onto one grid without an operation record.
- A trajectory contains ordered frames, motion/pose state and discontinuity/gap edges. Histories and targets may not cross a declared gap unless the task specifically models gap recovery.
- Ray retracing is needed when geometry or blockage changes. A Doppler phase approximation is not a model of path creation/destruction, moving obstruction, or path rerouting.
- Forecast targets declare origin time, horizon, future support and mask. All features are causal at that origin. Frame windows from one route/site are grouped for split independence.
- Repeated snapshots from one propagation event, multiple pilot patterns, SNR versions, receiver decodes, and counterfactual children are correlated descendants—not independent samples.

### 5.4 Frequency, bandwidth, and waveform semantics

Record carrier, subcarrier offsets, occupied/effective grid, numerology, PRB allocation, FFT size, guard/DC layout, cyclic prefix, sample rate, symbol timing, pilot positions, antenna ports, layers, precoding, and mapping type as applicable. Distinguish effective-subcarrier CFR from sample-domain channel convolution and from a receiver’s estimated CFR. Distinguish baseband offsets from absolute RF frequency. State how out-of-band bins, null/guard bins, and invalid bins are represented.

If an NR task uses only a bounded implementation mode, state it precisely. The 2026-11-12 receiver support matrix records Sionna PHY 2.2 PUSCH frequency-grid support with hopping set to `neither` and transform precoding false; caller-resolved intra-slot grid relocation is bounded software plumbing, not general hopping support. PRS frequency-domain known-resource search is not raw-sample acquisition or first-arrival ToA. SRS positioning contracts do not imply full RRC/PER decode, timing calibration or receiver qualification. Update these facts from source and frozen APIs at plan revision time.

## 6. Target ontology and task eligibility

### 6.1 Target specification

Every target must declare: semantic identifier and version; human-readable meaning; type/units; source family and producer; reference plane (oracle, ideal channel, receiver estimate, decoded result, or measurement); axes and rank; aggregation; valid range; validity mask and reason codes; censoring; confidence/calibration status; required companion metadata; eligible tasks/families; and explicit known exclusions. A target version’s identity/digest does not validate arbitrary semantics; recognized versions stay pinned to producer, source and aggregation at admission.

### 6.2 Propagation and visibility labels

Keep at least these meanings separate:

- `geometric_los`: direct segment visibility under a named geometry oracle and loaded asset/runtime state. Current NumPy triangle visibility and Mitsuba/PathSolver fail-closed cross-checks provide software and selected native evidence; scene-wide clearance, source completeness, material transmission, and release review remain separate.
- `profile_los_proxy`: categorical profile-component proxy supplied by a standardized channel family. It is not geometry and is not a generic LOS label.
- `detectable_direct_path`: receiver-/threshold-dependent detectable first/direct path; requires a detector, noise, bandwidth, power floor and timing contract. Current power/aggregation criteria are open.
- `blocked`, `outage`, `no solver path`, and `solver failure`: separate concepts; one must never stand in for another.
- `weak_los` or ambiguous visibility: only publish where a frozen physical/detection rule defines the boundary and reason/mask.

### 6.3 Channel and multipath targets

Targets can include full complex CFR/CIR, path delays and gains, RMS/mean excess delay, angular and Doppler measures, K factor, path loss/received power, path-set recovery, coherence descriptors, uncertainty, or identifiability. Each needs a definition of reference, noise floor, power weighting, bandwidth/time resolution, port aggregation, path matching, and censoring. In particular:

- RMS delay spread is power-weighted relative to a stated reference and only over explicitly admitted path groups. The current path-oracle helper supports one link-local RX/TX pair and does not define multiport pooling or a power-floor rule.
- Delay spread from a TDL/CDL profile constructor is a generation parameter, not a receiver estimate or necessarily oracle statistic.
- IFFT/PDP bins are finite-grid observables with bandwidth/windowing and periodicity limits, not exact physical path truth.
- Solver ray count, diffuse sample count, resolvable signal paths, clusters and physical objects are different quantities.
- Absolute ToA or carrier phase is not identifiable without a declared clock/gauge/calibration reference.
- Received power must state transmit power, gains, losses, noise figure/temperature and propagation scale; per-sample relative SNR is not an absolute link budget.

### 6.4 Receiver and task target families

Receiver estimates (CSI, delay, angle, sync), decoded outcomes (CRC/BLER/throughput), localization, beam/outage decisions, forecasting and uncertainty are distinct. Each target must cite the applicable waveform/receiver implementation and exact operating strata. G3 receiver-impact admission requires exact accepted G5 evidence plus the task-specific powered design and governing contract. Synthetic receiver curves and software loopbacks are diagnostics only.

### 6.5 Task-to-family eligibility matrix

This is an eligibility rule, not a statement that an eligible family is presently qualified. “Eligible with evidence” means the target is conceptually supportable if its exact producer and gate pass. “Conditional” means acquisition or task assumptions must establish a missing truth/observation contract. “No” means the family does not provide that kind of truth merely by belonging to UMCD.

| Task/target | Analytical | TDL/CDL | Scenario GBSM | Static RT | Dynamic RT | NR observations | Measured/external |
|---|---|---|---|---|---|---|---|
| Numerical channel/path recovery | Eligible with exact analytic oracle | Eligible for the model’s declared taps/channel | Eligible for declared model state | Eligible with native path/channel contract | Eligible with native temporal contract | Conditional on receiver/measurement channel reference | Conditional on calibrated truth modality |
| Geometric direct visibility | Eligible for explicitly constructed geometry | No | No, unless an independent geometry oracle is present | Eligible with scene-bound geometry oracle | Eligible with per-frame geometry oracle | Not from waveform/CSI alone | Conditional on independent surveyed geometry and clock alignment |
| TDL/CDL profile category | No | Eligible as profile metadata/proxy only | No | No | No | No | No |
| Detectable direct path / first arrival | Conditional on explicit bandwidth/noise/detector model | Conditional on detector applied to channel | Conditional on detector applied to channel | Conditional on calibrated receiver/detector | Conditional plus synchronization and time state | Conditional on exact receiver/acquisition | Conditional on acquisition/calibration and timing |
| Multipath parameter/set estimation | Eligible for resolvable analytic construction | Conditional; taps are model components, not automatically physical paths | Conditional on model’s own component semantics | Eligible only within solver/evidence limits | Eligible with trajectory path identity/evolution rules | Targets require separate reference; not available from observations alone | Conditional on measurement resolution/calibration and target modality |
| Geometric LOS/NLOS classification from NR PUSCH/DMRS observations | Control/diagnostic only | Source-domain candidate only; profile proxy is not target truth | Separate model family unless paired geometric oracle exists | Primary target family when scene-bound geometry and receiver observations pass their contracts | Extension only with per-frame geometry and synchronized observations | Observation family/consumer boundary; not a source of geometric truth by itself | External validation only when independently surveyed geometry and acquisition support the target |
| Sparse-pilot channel estimation | Eligible as controlled forward model | Eligible with exact waveform/receiver | Eligible with exact waveform/receiver | Eligible with exact channel-to-waveform route | Eligible with synchronized channel evolution | Receiver-estimation task; exact mode must be stated | Conditional on measurement chain and labels |
| Forecasting channel/propagation state | Only for declared analytic temporal process | Conditional on model temporal process | Conditional on spatial/temporal consistency | No from a static episode | Eligible with contiguous synchronized trajectories | Eligible only when observations are time ordered and causal | Conditional on time-aligned continuous capture |
| Receiver impact / BLER / link outcome | Diagnostic until exact receiver evidence | Diagnostic until exact G5 scope passes | Diagnostic until exact G5 scope passes | Diagnostic until channel and receiver gates pass | Diagnostic until dynamic receiver scope passes | Requires exact accepted G5 receiver and applicable G3 design | Requires validated acquisition and receiver evidence |
| Cross-family/domain transfer | Control source only; not external test domain | Source domain | Source domain | Independent source only if scene/site is held out | Independent route/scene split required | Separate receiver/measurement domain | Preferred external domain only if genuinely independent |
| Public release | Software fixtures only if explicitly packaged | Model terms/version/license review | Model source/license review | Scene, asset, material, output and license review | Same plus trajectory/privacy review | Waveform and derivation provenance review | Data rights, privacy, consent/custody and modality review |

Do not combine eligible rows into one score unless target semantics, population, reference plane, and independence unit are compatible and a frozen analysis justifies the aggregation. An “eligible” cell never inherits another family’s qualification.

## 7. Source family and scene-family catalog

Source family taxonomy and the S00–S11 scene registry remain separate. The table below states program roles; availability and eligibility are recomputed from the candidate registry and source audits before each release.

| UMCD source family | Intended role | Truth it may provide | Main limitations / admission conditions |
|---|---|---|---|
| Analytical controls | Numerical invariants, unit tests, identifiability studies | Exact analytic paths/channels for named idealized geometry | Not realistic-scene evidence; use to test arithmetic and limits, not external validity |
| Standardized TDL/CDL | Controlled stochastic baselines and table/covariance conformance | Model/profile-conditioned channel variables; bounded physical array semantics where evidenced | Profile LOS proxy is not geometric LOS; table implementation coverage is version-specific; Sionna ≠ complete standard certification |
| 3GPP scenario GBSM | Scenario/statistical transfer and spatial consistency studies | Outputs under a named model version, scenario and configuration | Not geometry-specific ray ground truth or independent city measurement; keep distinct from RT |
| Static geometric RT | Geometry-conditioned path/channel investigations and direct/indirect studies | Solver paths/channels and geometric visibility from named loaded scene | Requires native backend evidence, scene/material audits, convergence, calibration/power and source/license review |
| Dynamic RT | Time-varying scene/channel and blockage/forecast tasks | Time-indexed solver outputs and declared geometry labels | Requires synchronized motion/retracing, object state, temporal identity and independent trajectory splits |
| Counterfactual/intervention | Paired causal software/physics studies | Difference under explicit changed/held-constant factors | Solver and intervention fidelity must be validated; paired children are not independent samples |
| NR receiver observations | Deployment-facing pilot/waveform inference | Receiver-produced observations and outcomes within supported receiver envelope | Exact waveform/estimator/noise/sync assumptions; software path is not receiver qualification |
| Stress/OOD | Robustness, failure and calibrated-abstention studies | Known shift/control variable and outcome | Must be declared stress distribution; cannot represent ordinary prevalence without sampling rationale |
| Measured/external/field | Transfer and real-world validation | Only modalities and targets directly supported by acquisition/calibration | Acquisition, timing, hardware, privacy, license, custody and population independence all required |

The old master catalog also names S00 analytic, S01 urban canyon, S02 dense urban, S03 suburban/rural, S04 indoor office, S05 factory/warehouse, S06 station/airport/mall, S07 outdoor-to-indoor, S08 tunnel/garage, S09 mmWave street, S10 trajectory/mobility, and S11 measured/external. These registry identifiers must be reconciled with the current S00–S11 candidate catalog before use; labels such as “core,” “expansion,” or “held-out” do not establish a candidate exists or is eligible. Edition 3.775 records two families with eligible candidates and no qualified families at that time. Do not inflate candidate variants from one procedural scene into independent sites.

### 7.1 Scene/asset acceptance record

For every scene candidate, preserve source name/version/date, license and use permissions, exact asset closure, mesh/material/texture identities, conversion and repair steps, coordinate system/units, transform history, missing-asset report, material assignment, visual QA, geometry checks, source family relation, site/family independence relation, and allowed release scope. Sionna RT 2.2 models transmission through a surface using a radio material and assumes negligible ray angular deflection; thick-object behavior may be inaccurate. Bind material thickness/properties and treat geometry visibility and electromagnetic transmission as different semantics. Hashes provide byte identity and cache inputs; they do not prove physical realism, ownership, license, or independent provenance. Public redistribution permission and internal research permission are separate.

## 8. Generation architecture, producer and data contracts

### 8.1 DAG

Every family uses an explicit derivation DAG:

`source/config/assets → runtime scene/model setup → physical solve or channel draw → path/channel layer → waveform and receiver view → targets/derived views → split binding → audit/release manifest`.

An edge must identify parent artifact(s), producer implementation and version, configuration, input layer and exact output semantics. Reprocessing creates a new child identity; it never overwrites an immutable record or silently refreshes provenance on historical results.

### 8.2 Producer responsibilities

Each producer declares:

- supported family, configurations, backend/device and version range;
- required source and asset inputs and exact input identity rules;
- layers and targets produced, axis/order/unit conventions, validity semantics and known limitations;
- random seed/stream policy and reproducibility envelope;
- solver/receiver success, no-data, failure and partial outcomes;
- calibration and normalization operations;
- cacheability conditions and output identity;
- focused software and native evidence needed for admission.

The dataset store remains the semantic owner of immutable bundles; `experiments/runner.py` remains execution owner; `compute/resource_budget.py` remains scientific resource-estimation owner. Do not move these responsibilities into a generic dataset orchestrator. Producer audits must distinguish a callable helper from a production-wired route.

### 8.3 Identity, hashes, signatures, and caching

Content hashes are appropriate for artifact identity, source/asset closure, input/output integrity and cache keys when every behavior-changing input is included. A hash cannot establish that the hashed data are meaningful, physically accurate, authorized, correctly split, or honestly described. Signatures bind a signer to bytes under a trust policy; reviewer authority and custody are separate. Preserve these semantics and do not add hash checks whose only purpose is tracking whether a prose plan has changed.

Cache reuse requires complete producer inputs, software/source, runtime/backend, scene/material and configuration identity, random-state policy, and declared numerical reproducibility. Incomplete provenance blocks reuse. Different deterministic guarantees must be named (byte deterministic, numerical tolerance, or stochastic replicate). Solver seed alone is insufficient when the backend documents additional deterministic controls.

### 8.4 Failure and recovery

Persist distinct outcomes for ineligible source, preflight failure, solver failure, unconverged solve, explicit no path, invalid observation, receiver failure, partial artifact write, and complete publication. Fail closed at scientific boundaries. Atomic immutable publication writes content objects before final manifest; interrupted runs may leave unreferenced objects but no published partial bundle. Recovery tools may audit/replay exact prior bytes; they may not relabel failure as success or silently mutate an immutable result.

## 9. Splits, leakage, sampling, and statistics

### 9.1 Independence hierarchy

Build split components from all known shared causes: measurement session/site, original asset/source family, exact mesh/scene, procedural seed/template lineage, topology/configuration where it creates near-duplicate content, trajectory/route/spatiotemporal footprint, episode, standardized random draw, and paired counterfactual/receiver/noise descendants. The exact applicable namespaces depend on family; document omitted axes and why they do not induce leakage.

Assign parent independence identities before generating or exposing derived observations. Children inherit their parent’s split. A pilot, SNR point, pilot pattern, repeated seed-index, view, or frame must not cross partitions. Signed locks are immutable and verified; a new policy/version produces a new lock with a documented relation to prior versions.

### 9.2 Sampling and balance

Freeze the eligible population and strata before sampling. Retain base prevalence, rejection reasons, empty/outage/failure counts, and unsupported strata. If class balancing or conditional sampling is used, report original and sampled prevalence, retention, and any reweighting assumptions. Never reject valid no-coverage or difficult NLOS samples solely to meet a preferred class count. Do not fill missing family candidates with synthetic variants and call them independent sites.

Use deterministic, permutation-invariant selection where the sampling contract requires replay. Declare seed namespace and collision-audit scope. Report coverage/shortfall by family, scene/site, topology, carrier, waveform, receiver, path mechanism and other preregistered strata. No sample size or power is selected from convenience: derive it from the exact estimand, independent unit, variance/acceptance margin, multiplicity, and applicable protocol.

### 9.3 Evaluation and uncertainty

Save row-level predictions bound to immutable sample IDs and exact model/config identities. Tune on development only; preserve locked test data from iterative selection. Report class prevalence, PR-AUC/AUPRC, macro-F1 and per-class precision/recall alongside AUROC where it is defined, calibrated uncertainty, abstention, failure coverage, and subgroup/domain results where relevant. Bootstrap/resample at the independent scene/site/session or trajectory level, not per correlated ray/frame/pilot. Distinguish per-episode diagnostics from independent inferential claims. Freeze metric definitions, aggregation, censoring, threshold policy, confidence level, multiplicity correction and missing-value handling before confirmatory runs.

## 10. Native and scientific validation program

The gate names below follow the established research structure. Exact requirement-level criteria remain in the governing ledger and frozen protocols; this table summarizes each gate without weakening it.

| Gate | Dataset question | Minimum evidence classes | Current boundary from inspected evidence |
|---|---|---|---|
| G0: environment/source/provenance | Did the declared implementation, runtime, assets and backend actually produce this artifact? | Exact source/runtime inventory; native host/backend preflight; immutable provenance and trusted receipt where required | Software identity and several source contracts exist; Halti/native runtime and scene-specific production prerequisites remain separate. |
| G1: geometry/LOS/scenes | Are geometry and visibility labels defined and source scenes eligible? | Independent geometry oracle; boundary/grazing/contact fixtures; scene closure, QA, candidate independence and license review | NumPy/Mitsuba visibility path and selected checks exist; complete scene/source and qualification coverage remains open. |
| G2: path extraction/channel semantics | Do native paths map to the declared absolute-delay, phase, power and axes correctly? | Native API/dtype/mask/padding checks; analytic path/CFR parity; same-path decomposition; convergence and calibration evidence | Shared reconstruction and analytical regressions exist; native `Paths.cfr()` parity/calibrated power remain open. |
| G3: physical/receiver relevance | Does a claimed physical or receiver-impact difference satisfy the frozen task contract? | Exact G5-qualified receiver receipt; task-specific powered independent design; signed/reviewed contract and raw-outcome audit | G3 contracts/evaluators are software-plumbed in part; run manifests remain `NOT_QUALIFIED`; no admission from synthetic curves. |
| G4: analytical/standardized conformance | Does implementation match the bounded model/reference contract? | Frozen analytic references, exact table/version, topology and covariance checks, independent seeds, multiplicity-aware criteria | Numerous finite Sionna/TR 38.901 configuration-specific results exist; bounded passes do not establish general conformance; a retained failure stays visible. |
| G5: NR waveform/receiver | Does the exact supported NR configuration observe/decode correctly across the claimed envelope? | Standards-derived mapping, estimator/receiver contract, calibrated channel/noise, powered clusters, signed acceptance and independent audit | Several bounded software and native paths exist; support matrix lists unsupported modes and says G5 remains `NOT_QUALIFIED`. |
| G6: trajectory/forecast/intervention | Are time ordering, synchronization, causal inputs and future labels valid? | Timestamp/gap/clock contracts; retracing or justified dynamics; independent trajectory test; intervention checks | Typed time and trajectory support exists; clock calibration, broad dynamic validity and task-specific evidence remain open. |
| G7: storage/labels/privacy/splits | Are artifacts complete, interpretable, immutable, legal and leakage-safe? | Schema/round-trip/integrity checks; target-producer crosswalk; authoritative inventory; signed split audit; privacy/license review | Bundle and split contracts exist; broad corpus inventory, lineage closure, release authority and some legacy migration remain open. |
| G8: independent statistics/fairness | Are reported estimates supported by independent units and coverage? | Frozen population, powered design, independent site/session/trajectory groups, uncertainty and shift/fairness audit | Statistical tooling and finite references exist; dataset-level independent coverage and release claims remain unqualified. |

Analytical unit cases, schema validation and software integration tests establish software behavior only. Native evidence is host/backend-specific. A finite conformance family is bounded to its frozen configurations. Physical scene calibration and measurement transfer need evidence beyond simulator self-consistency. Retained failures and negative cohorts remain visible; no favorable replacement cohort can erase them.

## 11. Dataset release, versioning, and publication

### 11.1 Release tiers

Use release tiers to communicate evidence rather than imply one corpus-wide qualification:

- **Development/software fixtures:** tiny analytical or synthetic records for tests and demos. Never presented as empirical dataset evidence.
- **Model-family research artifacts:** standardized or scenario families with exact model/version scope and model-specific truth semantics.
- **RT candidate research artifacts:** geometry/RT bundles admitted only for declared internal research scope, with native/source/material and scene-license status visible.
- **Receiver-observation study:** exact waveform, channel, receiver, synchronization and noise context; qualify only within the admitted run protocol.
- **Measured/external cohort:** acquisition, modality, calibration, privacy, data rights and independent-site/session provenance reviewed.
- **Public release:** only artifacts whose source and every child layer have compatible redistribution rights, privacy review, complete manifests, stable readers, split and target documentation, and authorized release review.

No tier uses “qualified” without the exact scope and authority that granted it. Internal-use eligibility does not imply redistribution permission.

### 11.2 Dataset card and inventory

Every release card/inventory should state: dataset version and schema versions; date and code/runtime/backend; family and candidate registry versions; episode/link/frame/observation counts with denominators; unique independent sites/sessions/trajectories; carrier/waveform/topology/receiver coverage; target availability and reason codes; split definitions and signed lock IDs; source/asset licenses; exclusions and failures; generation and calibration limitations; integrity and reproduction instructions; known leakage/coverage limitations; scientific qualification status per family/task; and permitted claims/use.

Counts are not evidence of diversity. Report both raw records and independent population units. Report family shortfalls and unusable/unreleased artifacts explicitly.

### 11.3 Version and migration policy

- Keep schema meaning immutable. A change to axis, units, semantics, aggregation, identity, or trust boundary receives a new schema/target version.
- Readers may support multiple historical versions where safe, but reading a historical artifact never upgrades its scientific meaning.
- Write admission may refuse legacy versions whose axes or provenance are not sufficiently bound; preserve those bytes and document the refusal/migration path.
- Migrations are new derived artifacts with source identity, transform implementation/version, before/after audit and parent-child lineage. Never overwrite source evidence.
- Storage format evolution must preserve pickle-free safe loading, bounded resource use, atomic publication, integrity checks, and compatibility adapters as needed.

## 12. Compute, storage, and campaign sizing

Estimate cost only after a representative, authorized pilot measures the actual native backend and storage behavior. Separate scene load/build, ray tracing, path serialization, channel reconstruction, waveform/receiver processing, target derivation, model training, audit, and report time. Record hardware/device, host contention, memory peak, solver budget, rays/paths, output bytes, compression ratio, file count, wall time, retries and failure rate.

For each candidate campaign, estimate:

- uncompressed and compressed bytes for each layer independently;
- path cardinality and interaction metadata distribution, not a mean alone;
- dense channel dimensions from actual frame/port/grid counts;
- receiver-observation and model-input multiplicity per physical episode;
- immutable derived-view growth and retained failed outcomes;
- manifest/inventory/audit index costs and safe iteration memory;
- backup/retention, transfer, and release packaging costs.

Scale in gated stages: analytical/contract pilot → bounded family pilot → independent-site pilot → preregistered campaign. Each stage has a stop decision if source closure, convergence, runtime, data quality, split independence, license, or acceptance criteria fail. A compute quota is not scientific authorization, and a cheaper backend fallback must not silently alter a frozen run.

## 13. Implementation and research roadmap

This is the prescribed dependency order for turning the design into a dataset. The implementation owner follows the existing queue task and its named source owner; this plan does not create alternate task cards or gate-named files. The current queue and ledger are local ignored inputs in this checkout, so re-read them at the start of each task and stop if their canonical accessible version disagrees.

### Step 0 — Resolve active control-plane inputs before implementation

1. Read the active constitution and navigation rules, then locate the authoritative ledger, execution queue and relevant source evidence in the current checkout. At this edition those records are present under git-ignored `dev-docs/research/`; do not infer mutable status from this plan or assume those files ship with a clone.
2. Run the existing E01 source and requirement reconciliation against the active ledger. Keep the historical plan crosswalk distinct from current implementation, verification, qualification and closure status.
3. Build each software task from an existing reviewed queue card. Do not invent an AgentOps runtime dependency, an unreviewed G3 task file, or a second backlog.
4. **Exit:** the implementer has identified the exact active ledger and queue, reconciled source/requirement authority for the selected task, and recorded its existing owner, dependencies and next action in the operational task state.

### Step 1 — Complete target and producer meaning before adding data

Use the existing `M1.TARGET_FAMILY_SEMANTICS` card under E01/PKG08, then complete its six dependency-ordered items: (a) audit producer-backed range/gauge/reference/mask/missingness rules; (b) inspect producer semantics before adding axes; (c) define `detectable_direct_path` only after detector, bandwidth, noise/SNR and false-alarm criteria are frozen; (d) define `relative_first_path_delay` only after timing, acquisition, resolution and censoring rules are fixed; (e) define dominant received-path power only after power/link-budget references are fixed; (f) define receiver-estimated delay spread separately from path-oracle RMS. Retain historical read support and refuse under-specified canonical writes.

Implementation is confined first to current owners `datasets/targets.py`, the producer modules listed in §3.4, their existing schemas under `datasets/schema/`, and focused tests mirroring those routes. Do not create another target package or new G3-named file. Change a target schema version only when meaning or axes change; never infer axes for existing vectors.

**Exit:** every target admitted to a producer has exact producer/source/reference/aggregation, type, units, axes, validity, missingness and reader behavior; unavailable semantics remain unsupported.

### Step 2 — Finish family candidate and leakage wiring

Continue `M1.SCENE_FAMILY_REGISTRY` using the existing `config/scenes/scene-family-registry.v1.json`, `datasets/scene_families.py`, `scripts/sample_scene_families.py`, `evaluation/near_duplicate_audit.py`, and signed split adapters. Do not add a second sampler or split allocator. The 2026-11-12 queue evidence says direct/cached RT and trajectory producer identity has advanced, but only two of twelve families had eligible candidates at the one-group design point; re-audit the live registry before claiming support. Complete the exact S00 analytic roster. Add candidates only after source, asset-tree, independence, license, internal-use and public-release dispositions are independently recorded. Report unsupported family shortfalls.

Complete `M1.SCENE_VISIBILITY_EDGE_CASES` only within its bounded software contract, then separately satisfy G1 scene-wide/source/native requirements. Complete `M1.SCENE_STATE_TIMELINE` for declared pose/material interpolation, gaps and identity before temporal claims. Do not turn a visibility unit-test pass into scene/material qualification.

**Exit:** every eligible family candidate reaches existing split policy with the same source/asset/family/site identity, and unsupported families stay explicit. No new D1 generation is started by this step.

### Step 3 — Make canonical episode storage the multi-node route

Continue the existing `M1.MULTINODE_RT_SUPPORT_MATRIX` route. The software path already has `NetworkEpisode` extraction, exact link roles, `sionna_bindings_to_channel_views()`, `build_sionna_rt_episode_bundle_from_paths()`, `write_sionna_rt_episode_bundle_from_paths()`, `run_authorized_static_network_episode()`, and `episode_channels_to_network_phy_input()`. Use these APIs for the declared finite supported topology; preserve exact TX/RX port maps, all link/frame solve outcomes, live scene/run identity, candidate split relations, and the existing D1 destination/receipt gate. The entry is currently a library API, not the legacy `scripts/generate_rt_dataset.py` command.

Do not add a generic new writer, flatten topology to one dense global tensor, imply temporal evolution by repeating one CFR over OFDM symbols, or silently mask failed solves as zeros. Extend `datasets/episode_store/` only when a concrete layer cannot be safely represented, and keep immutable object/manifest publication and legacy readers intact.

**Exit:** the same episode identity connects scene extraction, path and channel layer publication, split lineage, and the supported network-PHY consumer; unsupported topology is rejected with a reason. The queue’s software closure and test counts remain dated evidence; native multi-link qualification and the full topology roster remain separate.

### Step 4 — Reconcile every producer through a declared adapter

Use the exact route map in §3.4. Keep the legacy single-link static RT CLI and the separate authorized multi-node episode library route distinct. Dynamic RT already accepts optional `umcd_store_root` and publishes UMCD layers from the same retraced solves when supplied; preserve the specialized trajectory store as well. Standardized TDL/CDL, scenario GBSM, CAEZ measured magnitude, and compact `ChannelDataset` producers retain their family-specific record semantics. If a downstream consumer needs a common interface, implement the smallest adapter at the producer/consumer boundary; do not rewrite a source family merely to make file layouts look uniform.

For each route, inventory both production callers and test callers. Validate exact layer/target mapping, split-child inheritance, manifest read-back, model-input audit and release classification. Tests establish software contracts only and must use the repository’s meaningful 5G NR behavior fixtures; `evidence_integrity` and native Sionna remain separately marked according to `pyproject.toml`/CONTRIBUTING.

**Exit:** each supported producer has one documented entrypoint, output contract, consumer and limitation. Unwired helpers are marked as such; test references do not count as production wiring.

### Step 5 — Use existing split-to-evaluation integration; do not create another path

E02’s six-cohort scoring software slice was closed on 2026-11-12. Preserve its signed cohort declaration, canonical signed split registry, exact dataset-byte snapshot, cohort-index replay, split-to-row checks, selected-test digest, unsupported-cohort reasons and pre-manifest failure receipt. Do not add a parallel split utility or treat the bounded E02 closure as proof of authoritative corpus inventory, signer custody, or scientific independence. Continue any remaining PKG08 lineage/inventory items through the current queue.

**Exit:** every benchmark/experiment consumer resolves exact parent and child identity from the frozen lock before scoring; missing or unsupported cohorts remain visible.

### Step 6 — Admit receiver work only through E04/G5 then G3

Follow E04’s current sequence: freeze the exact task-specific waveform, receiver/decoder, SNR/noise/link budget, strata, independent channel cluster, sample design, acceptance family and reviewer authority; close software gaps only in the existing receiver/integration/evaluation owners; then require a signed/audited G5 result for that exact mode. Only afterward can a G3 contract bind that receipt and run its powered candidate/challenger comparison. The 2026-11-12 support matrix says PUSCH hopping/transform precoding and other receiver scope remain bounded or unsupported, and G5/G3 manifests remain unqualified. No loopback, synthetic two-path curve, finite decoder pass or support matrix is a qualification receipt.

**Exit:** the exact evidence chain satisfies existing signed contracts and auditor rules, or the mode remains `NOT_QUALIFIED`. No receiver campaign is run from this plan alone.

### Step 7 — Freeze independent cohorts, evaluate, and release by family

For the LOS/NLOS research objective, preregister three separately named comparisons: (a) reproduce the existing six-scene RT-only LOSO baseline; (b) train on controlled TDL/CDL-derived NR observations and test zero-shot on independent RT scenes; and (c) if adaptation is studied, declare whether target observations are unlabeled or labeled, define adaptation/validation groups, and keep every test scene sealed from adaptation and model selection. Use identical observation construction, NR numerology, pilot/DMRS exposure, preprocessing and compute budget where the comparison claims allow it. Keep profile identity and geometry labels out of inference features. Compare with the repository's established neural and classical baselines plus an environment/link-feature baseline as a distinct information tier; no score can collapse these input regimes into one result. Use the natural target prevalence for final evaluation, report class counts and per-scene results, AUCPR/PR-AUC and macro-F1 as well as AUROC when defined, and compute uncertainty across independent scenes/sites rather than links alone.

Only after the relevant software/source/qualification prerequisites pass, freeze a family/task-specific population, sampling design, independent unit, split lock, target roster, metrics, uncertainty method, multiplicity and stopping rule. Profile resource costs with the existing `compute/resource_budget.py`; use existing experiment runner and artifact manifest mechanics. Retain failures and negative cohorts. Release each family/task only after G7/G8, rights, privacy, inventory, lineage, independent review and `datasets/public_release.py` approval conditions for that exact scope pass.

**Exit:** publish a family- and task-scoped dataset card with actual coverage and unsupported shortfalls. Do not declare UMCD globally qualified because one family or task passes.

No step automatically authorizes the next. Production D1, native sweeps, qualification runs, advanced training, commits, pushes and public releases remain subject to the repository’s existing user/task authorization rules.

## 14. Risks and explicit decisions

| Risk or unresolved choice | Why it matters | Plan disposition / required decision point |
|---|---|---|
| Active ledger/queue are operational local records and not shipped in a clean clone | A plan reader must not mistake the static crosswalk for mutable requirement status | Locate the active checkout records at task start; retain operational authority and mutable status there, not in this plan. |
| Existing literature already covers ray-traced LOS prediction, domain transfer, and multi-domain channel tasks | Broad novelty claims would be unsupported and could misstate the field | Re-run the focused bibliography and comparison matrix before manuscript claims; narrow novelty to demonstrated differences in observations, source/target domains, semantics and evidence. |
| S00–S11 registry may not match old scene labels exactly | Confusing scene identity creates invalid coverage and independence claims | Reconcile IDs/names and semantic definitions against the current registry before candidate sampling. |
| Only a limited subset of families currently has eligible candidates | Broad “unified dataset” language can imply unsupported coverage | Publish family/candidate availability and shortfall table with every release; do not pretend all planned families are built. |
| `ChannelDataset` and UMCD episode bundle serve different use cases | Premature consolidation can break mature experiment consumers | Keep compatibility adapter; only deprecate after caller inventory, replacement path, migration and versioned evidence exist. |
| Dense physical views may be expensive or unavailable for some families | Naive universal schema can balloon storage or fabricate missingness | Keep path lists ragged and views optional; profile before deciding any chunking/sparse format. |
| Many schema revisions may obscure compatibility | Version count alone does not show redundant meaning | Maintain semantic change log and read/write support matrix; consolidate only when exact semantics and provenance can be preserved. |
| Native backend availability and reproducibility differ by host | Local test success may not reproduce target runtime | Pin exact environment/device receipt and keep native evidence host-scoped; do not assume fallback equivalence. |
| RT geometry/material realism and solver convergence are open | Visual plausibility or a correct LOS segment test is not propagation validation | Gate RT claims on source closure, material model, path mechanisms, convergence, calibration and independent review. |
| Receiver integrations cover bounded modes | A usable adapter can invite overbroad receiver claims | Keep exact support matrix current and scope every G5/G3 result to frozen configs and acceptance. |
| Measured data may have limited modality or rights | Magnitude-only or restricted data cannot validate all phase/path/geometric targets | Maintain modality-specific target eligibility and separate research-use from redistribution status. |
| Correlated samples inflate apparent precision | Frames, rays, pilots, seeds and views share parent causes | Freeze independent unit and resampling hierarchy per estimand before confirmatory collection. |

## 15. Requirement and source traceability

The appendix below cross-references each current requirement ID to this plan. It intentionally does not repeat mutable implementation, verification, qualification, dependency or closure fields; those stay in the active ledger. The requirement descriptions are short labels to identify rows, not replacements for the full requirements.

At minimum, crosswalk the plan to DS.G0–DS.G8 and their child criteria, target/producer/storage requirements, scene-family registry requirements, split/near-duplicate requirements, NR/receiver requirements, G3 admission, lineage/release/license requirements, and applicable frozen family protocols. Include non-gate package requirements only where they change a dataset producer, reader, semantic contract or release path. The ledger’s complete 316-row content is not duplicated in this document; the crosswalk below is a navigation aid and does not replace reading the authoritative requirement text and closure fields.

### 15.1 Maintained source and test links

These tracked files are directly inspectable from a repository clone and form the source map for implementation behavior:

| Contract or behavior | Code/schema | Focused regression references |
|---|---|---|
| Episode, node, array, directed-link, pose and time contracts | [`datasets/network.py`](../../datasets/network.py), [`datasets/time_axis.py`](../../datasets/time_axis.py), [`datasets/schema/umcd-episode-metadata.v1.schema.json`](../../datasets/schema/umcd-episode-metadata.v1.schema.json) | [`test_network_episode_rules.py`](../../tests/test_datasets/test_network_episode_rules.py), [`test_network_geometry_rules.py`](../../tests/test_datasets/test_network_geometry_rules.py) |
| Named-axis physical channel and phase comparison | [`datasets/channel_view.py`](../../datasets/channel_view.py), [`datasets/schema/umcd-physical-channel-layer.v1.schema.json`](../../datasets/schema/umcd-physical-channel-layer.v1.schema.json) | [`test_umcd_channel_receiver_views.py`](../../tests/test_datasets/test_umcd_channel_receiver_views.py) |
| Ragged path oracle, solve outcomes, and layer metadata | [`datasets/path_oracle.py`](../../datasets/path_oracle.py), [`datasets/schema/umcd-path-oracle-layer.v1.schema.json`](../../datasets/schema/umcd-path-oracle-layer.v1.schema.json) | [`test_path_oracle_table_rules.py`](../../tests/test_datasets/test_path_oracle_table_rules.py), [`test_umcd_path_oracle.py`](../../tests/test_datasets/test_umcd_path_oracle.py), [`test_path_oracle_layer_metadata.py`](../../tests/test_datasets/test_path_oracle_layer_metadata.py) |
| Target definitions, producer pins and validity reasons | [`datasets/targets.py`](../../datasets/targets.py), [`datasets/schema/umcd-target-layer.v1.schema.json`](../../datasets/schema/umcd-target-layer.v1.schema.json) | [`test_umcd_targets.py`](../../tests/test_datasets/test_umcd_targets.py) |
| Model-input boundary and oracle isolation | [`datasets/model_input_audit.py`](../../datasets/model_input_audit.py), [`datasets/nr.py`](../../datasets/nr.py) | [`test_model_input_audit.py`](../../tests/test_datasets/test_model_input_audit.py), [`test_nr_model_input_firewall.py`](../../tests/test_datasets/test_nr_model_input_firewall.py) |
| Provenance and immutable episode bundles | [`datasets/provenance.py`](../../datasets/provenance.py), [`datasets/episode_store/`](../../datasets/episode_store/), [`datasets/schema/umcd-bundle-v1.schema.json`](../../datasets/schema/umcd-bundle-v1.schema.json) | [`test_umcd_store_integrity.py`](../../tests/test_datasets/test_umcd_store_integrity.py) |
| Signed split policy and leakage controls | [`datasets/split_policy/`](../../datasets/split_policy/), [`datasets/split_lineage.py`](../../datasets/split_lineage.py) | [`test_split_policy_policy_rules.py`](../../tests/test_datasets/test_split_policy_policy_rules.py), [`test_split_policy_signing.py`](../../tests/test_datasets/test_split_policy_signing.py) |
| RT network, trajectory and path-to-channel producer routes | [`integrations/sionna/rt_network.py`](../../integrations/sionna/rt_network.py), [`integrations/sionna/rt_trajectory.py`](../../integrations/sionna/rt_trajectory.py), [`integrations/sionna/path_oracle.py`](../../integrations/sionna/path_oracle.py) | [`test_rt_network.py`](../../tests/test_integrations/test_rt_network.py), [`test_rt_trajectory_contract.py`](../../tests/test_integrations/test_rt_trajectory_contract.py), [`test_path_oracle_adapter_contract.py`](../../tests/test_integrations/test_path_oracle_adapter_contract.py) |
| NR PUSCH receiver boundary | [`receivers/nr/pusch.py`](../../receivers/nr/pusch.py), [`integrations/sionna/nr_pusch.py`](../../integrations/sionna/nr_pusch.py) | [`test_nr_pusch.py`](../../tests/test_receivers/test_nr_pusch.py), [`test_nr_pusch_channel_estimation.py`](../../tests/test_integrations/test_nr_pusch_channel_estimation.py) |

Test names above are evidence locations, not assertions that these tests were rerun for this documentation change. Consult the current test source and exact run receipt before relying on a result.

### 15.2 Checkout-local research inputs

The following inputs were inspected locally but currently live under the ignored `dev-docs/research/` tree, so links to them would not resolve in a clean clone. They are named here as provenance for this edition, not published authorities available to every reader: `REQUIREMENT_LEDGER.json`, `EXECUTION_QUEUE.md`, `PLAN_ARCHITECTURE.md`, `PLAN_DATA_GENERATION.md`, `PLAN_RECEIVERS_AND_TARGETS.md`, `PLAN_QUALIFICATION_GATES.md`, `PLAN_OPEN_QUESTIONS.md`, `RECEIVER_SUPPORT_MATRIX_2026-11-12.md`, and `UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md`. The tracked plan must be refreshed against the active versions when the active local research records or their authority change.

### 15.3 Source map for this edition

- Additional canonical types and invariants: `datasets/nr.py`, `datasets/observations.py`, `datasets/provenance.py`, `datasets/split_policy/`, and `datasets/episode_store/`.
- Dataset orientation: `datasets/README.md`; storage contract: `datasets/episode_store/README.md`.
- Producer and receiver examples: `integrations/sionna/rt_network.py`, `integrations/sionna/rt_trajectory.py`, `integrations/sionna/standardized_pusch.py`, `integrations/sionna/network_receiver.py`, `scripts/generate_rt_dataset.py`, `scripts/build_rt_dmrs_observation_dataset.py`, `scripts/build_standardized_pusch_dataset.py`.
- Versioned contracts: `datasets/schema/`, `evaluation/schema/`, and producer/reader tests under `tests/`.
- Working checkout-local research sources inspected are listed in §15.2.
- Historical public-workspace copy: `outputs/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md`; treat it as historical source material, not the current implementation inventory.

### 15.4 Individual requirement-to-plan crosswalk

The exact 316 requirement IDs below were inventoried from the checkout-local ledger on 2026-11-26. Each row has an explicit section route selected from its requirement meaning and implementation boundary; similar requirements may share sections, but the mapping is not a substitute for reading the full ledger wording. Implementation, verification, qualification, dependency, and closure status remain in the ledger and are intentionally not duplicated here. This mapping is a design cross-reference, not a substitute for the source audit or owner/dependency decisions in E01. The ledger is local and git-ignored in this checkout, so revalidate the ID set and requirement text against the active ledger before using this appendix in another checkout.

| Requirement ID | Ledger requirement (short label) | Plan sections |
|---|---|---|
| `DS.G0` | Gate G0 — environment, source and provenance | §3, §7, §8, §11, §12, §13 |
| `DS.G0.01` | pinned `sionna`, `sionna-rt`, `mitsuba`, `drjit`, PyTorch, Python, NumPy and expected GPU/CPU backend versions print and match… | §3, §7, §8, §11, §12, §13 |
| `DS.G0.02` | selected polarized Mitsuba variant and native `PathSolver` can compute a known free-space path on selected host; CUDA OptiX… | §3, §7, §8, §11, §12, §13 |
| `DS.G0.03` | scene asset closure complete: XML/mesh/MTL/texture/associated metadata content hashed; unresolved assets invalidate production… | §3, §7, §8, §11, §12, §13 |
| `DS.G0.04` | dynamically altered radio material properties, scene-object pose/velocity, antenna pattern and frequency enter runtime scene hash. | §3, §7, §8, §11, §12, §13 |
| `DS.G0.05` | RNG streams independent and deterministic where promised; repeat for same hardware/backend identity. | §3, §7, §8, §11, §12, §13 |
| `DS.G0.06` | versioned TDL/CDL profile-table hash and external 3GPP spec revision recorded separately. | §3, §7, §8, §11, §12, §13 |
| `DS.G0.07` | no future-dated or unverifiable benchmark report promoted on filename alone; every result links to raw manifests, commit,… | §3, §7, §8, §11, §12, §13 |
| `DS.G0.08` | licensing and permitted field redistribution validated; measured datasets may require nonpublic local storage. | §3, §7, §8, §11, §12, §13 |
| `DS.G1` | Gate G1 — geometry, LOS semantics and physical scenes | §5, §6, §7, §10, §13 |
| `DS.G1.01` | free-space straight path has the expected geometric visibility and propagation delay `d/c` within numerical tolerance. | §5, §6, §7, §10, §13 |
| `DS.G1.02` | independent segment-intersection oracle vs depth-zero `PathSolver` agrees in well-conditioned configurations. | §5, §6, §7, §10, §13 |
| `DS.G1.03` | near-edge/grazing/mesh-contact links flagged ambiguous, not forced into confident labels. | §5, §6, §7, §10, §13 |
| `DS.G1.04` | antennas outside forbidden solids; units/coordinate frames and object transforms consistent. | §5, §6, §7, §10, §13 |
| `DS.G1.05` | geometry manipulations (add opaque blocker, open door) produce expected visibility **when physical model supports it**. | §5, §6, §7, §10, §13 |
| `DS.G1.06` | transmissive thin wall handled separately from geometrically unobstructed LOS; physical transmission is not 'direct free-space LOS'. | §5, §6, §7, §10, §13 |
| `DS.G1.07` | blocked scene with diffraction/reflection retains indirect paths when actual geometry supports them. | §5, §6, §7, §10, §13 |
| `DS.G1.08` | LOS-rich scene contains valid direct **and** multiple indirect paths after configured solver convergence. | §5, §6, §7, §10, §13 |
| `DS.G1.09` | scene motion interpolation never causes discontinuous teleportation; timestamps/poses monotonic. | §5, §6, §7, §10, §13 |
| `DS.G1.10` | scene families are independent; no hashed assets/mesh variants duplicate into frozen held-out site unintentionally. | §5, §6, §7, §10, §13 |
| `DS.G2` | Gate G2 — native path extraction, phase, timing, power | §5, §8, §10, §13 |
| `DS.G2.01` | `Paths` native API version compatibility: required properties exist with asserted shapes and dtypes. | §5, §8, §10, §13 |
| `DS.G2.02` | `valid` masks actually gate padded paths; zero/uninitialized entries excluded from computations. | §5, §8, §10, §13 |
| `DS.G2.03` | explicit `normalize_delays=False` on canonical `cir`, `cfr`, and `taps` where applicable. | §5, §8, §10, §13 |
| `DS.G2.04` | explicit `normalize=False` for canonical absolute-power CFR. | §5, §8, §10, §13 |
| `DS.G2.05` | `cfr()` manual reconstruction from the same path set equals API output after correctly applying carrier-to-baseband phase… | §5, §8, §10, §13 |
| `DS.G2.06` | `H_full = H_direct+H_indirect` for a **same-path-set** partition at identical reference, array and frequency axes. | §5, §8, §10, §13 |
| `DS.G2.07` | frequency bin ordering (center/DC/guard/negative offsets) matches declared NR resource grid, including odd/even FFT-size fixtures. | §5, §8, §10, §13 |
| `DS.G2.08` | first physical arrival delay invariant to optional regridding/receiver-view generation; normalized view cannot overwrite original. | §5, §8, §10, §13 |
| `DS.G2.09` | lossless canonical complex serialization roundtrips within dtype error without sign conjugation or axis permutation. | §5, §8, §10, §13 |
| `DS.G2.10` | path loss/free-space coefficient follows a validated link-budget convention for test geometry and antenna gains. | §5, §8, §10, §13 |
| `DS.G2.11` | received coherent power not erroneously computed as sum of powers of individual paths where interference cross terms matter. | §5, §8, §10, §13 |
| `DS.G2.12` | manually reconstructed single-, two- and multiple-path CFRs match analytical test cases including close delays, nulls and… | §5, §8, §10, §13 |
| `DS.G2.13` | MIMO explicit vs synthetic array differences quantified on representative near-/far-field cases. | §5, §8, §10, §13 |
| `DS.G2.14` | dual polarization/basis rotation controls yield expected channel transformation. | §5, §8, §10, §13 |
| `DS.G2.15` | unused solver path metadata/padded values never enter target computations; last array dimension not assumed to mean path without… | §5, §8, §10, §13 |
| `DS.G3` | Gate G3 — ray budget and physics fidelity | §6, §9, §10, §13 |
| `DS.G3.01` | `max_depth` sweep stabilizes relevant delay/power/angular results for representative LOS and NLOS links. | §6, §9, §10, §13 |
| `DS.G3.02` | `samples_per_src` sweep stabilizes quantiles/distributions, separately for specular/diffuse components. | §6, §9, §10, §13 |
| `DS.G3.03` | diffuse model enabled **and** nonzero material scattering coefficient validated; flag-only enablement insufficient. | §6, §9, §10, §13 |
| `DS.G3.04` | diffuse ray count changes with sample budget without being mistaken for changing physical path count. | §6, §9, §10, §13 |
| `DS.G3.05` | `max_num_paths_per_src` clipping is audited; no silent path truncation in stress cases. | §6, §9, §10, §13 |
| `DS.G3.06` | refraction penetration/slab assumptions recorded; thick-object results not called high-fidelity without calibration. | §6, §9, §10, §13 |
| `DS.G3.07` | first-order edge diffraction limitations acknowledged; deep diffraction NLOS stress marked model-limited. | §6, §9, §10, §13 |
| `DS.G3.08` | deterministic solver replay passes on pinned exact environment; numerical backend changes recorded and quantified. | §6, §9, §10, §13 |
| `DS.G3.09` | weak-power tails do not spuriously vanish solely from aggressive fixed threshold; threshold sensitivity reported. | §6, §9, §10, §13 |
| `DS.G3.10` | empty/full/direct path outcomes audited by class with retries and reference budgets. | §6, §9, §10, §13 |
| `DS.G3.11` | geometric-LOS/solver-direct disagreement excluded from headline until resolved or labeled low-confidence. | §6, §9, §10, §13 |
| `DS.G3.12` | realized natural vs deliberately stratified class priors both logged. | §6, §9, §10, §13 |
| `DS.G4` | Gate G4 — analytical and standardized model conformance | §7, §10, §13 |
| `DS.G4.01` | TDL A–E nominal tap delays and powers match pinned resource tables (CPU-native Sionna 2.2.0/TR 38.901 19.2 check verifies… | §7, §10, §13 |
| `DS.G4.02` | scalable TDL A–E tap delays change correctly with configured RMS delay spread (CPU-native Sionna 2.2.0 test checks the generated… | §7, §10, §13 |
| `DS.G4.03` | CDL A–E angular/delay nominal params match pinned tables and defined array orientations (the existing native 20-case A–E ×… | §7, §10, §13 |
| `DS.G4.04` | model-D/E specular/scattered power partition and configured K settings produce expected empirical behavior (a seeded CPU Monte… | §7, §10, §13 |
| `DS.G4.05` | 3GPP scenario state and profile category are not written into a `geometric_los` target field (TDL/CDL adapter metadata declares… | §7, §10, §13 |
| `DS.G4.06` | independent batch draws and temporal evolution samples are explicitly distinguished (TDL/CDL adapter metadata identifies the… | §7, §10, §13 |
| `DS.G4.07` | empirical delay, Doppler, spatial and covariance distributions have confidence intervals under independent draws (CPU-native… | §7, §10, §13 |
| `DS.G4.08` | TDL/CDL restrictions (valid carrier, BW, array/model limitations) are checked at standardized-record admission… | §7, §10, §13 |
| `DS.G4.09` | profile-only classification is explicitly scoped to the A/B/C-versus-D/E `PROFILE_PROXY` target, not geometric LOS.… | §7, §10, §13 |
| `DS.G5` | Gate G5 — NR waveform, channel and receiver correctness | §5, §6, §10, §13 |
| `DS.G5.01` | generated pilot/resource grid conforms to specified applicable TS 38.211 version and Sionna native API; no generic comb pattern… | §5, §6, §10, §13 |
| `DS.G5.02` | effective subcarrier indices map exactly to transmitted FFT bins and receiver channel estimate without DC/guard shifts (native… | §5, §6, §10, §13 |
| `DS.G5.03` | PUSCH perfect-channel/noiseless case behaves as expected under valid CP and waveform settings (native CPU 6-RB waveform round… | §5, §6, §10, §13 |
| `DS.G5.04` | CP-exceedance stress reveals physically plausible ISI/ICI where time-domain model predicts it; static frequency multiplication… | §5, §6, §10, §13 |
| `DS.G5.05` | OFDM symbol clock and RT Doppler sampling use correct real duration and frequency units (normal-CP timing tests, 30-kHz… | §5, §6, §10, §13 |
| `DS.G5.06` | physical thermal-noise PSD to complex sample covariance matches injected and measured covariance statistically (proper-complex… | §5, §6, §10, §13 |
| `DS.G5.07` | precise SNR definition (pre-/post-beam, per RE/antenna, plus NF) travels with each view (ideal standardized PUSCH AWGN mode… | §5, §6, §10, §13 |
| `DS.G5.08` | absolute path loss retained in physical receiver view; normalized diagnostic explicitly separate (native CPU analytic… | §5, §6, §10, §13 |
| `DS.G5.09` | CFO/phase-noise/timing/IQ impairments injected in physically consistent order with known reference (partial: ordered… | §5, §6, §10, §13 |
| `DS.G5.10` | LS estimator bias/variance and LMMSE noise behavior verified against analytical fixtures (native CPU 1,024-draw pilot checks… | §5, §6, §10, §13 |
| `DS.G5.11` | estimated CSI stored on correct antenna/layer/symbol/frequency axes, with pilot mask and observation missingness (typed… | §5, §6, §10, §13 |
| `DS.G5.12` | same parent channel with different noise seeds has independent receiver views but identical hidden propagation truth… | §5, §6, §10, §13 |
| `DS.G5.13` | BLER/BER/CRC/beam outcomes derived from real matched receiver pipeline, not posthoc channel score alone (native CPU Sionna PUSCH… | §5, §6, §10, §13 |
| `DS.G5.14` | PRS/SRS/SSB implementations only admitted after independently validating mapping and observable timing claims (partial SRS… | §5, §6, §10, §13 |
| `DS.G5.15` | no future/fixed oracle CSI is fed into ordinary model input adapter (adversarial NR record with extreme oracle target and… | §5, §6, §10, §13 |
| `DS.G6` | Gate G6 — trajectories, forecasting and interventions | §5, §8, §9, §13 |
| `DS.G6.01` | time series increasing and synchronized; expected channel coherence fits chosen time grid. | §5, §8, §9, §13 |
| `DS.G6.02` | `t+h` targets align by timestamps (not arbitrary frame offset when gaps exist). | §5, §8, §9, §13 |
| `DS.G6.03` | estimated Doppler agrees with time evolution under short stationary-path conditions. | §5, §8, §9, §13 |
| `DS.G6.04` | short-horizon Doppler-only evolution compared with physical re-tracing across displacement in wavelengths. | §5, §8, §9, §13 |
| `DS.G6.05` | physical blockage intervention changes full path set and creates/destroys indirect components where geometry predicts. | §5, §8, §9, §13 |
| `DS.G6.06` | no path index tracking assumption across retraces; use association uncertainty and birth/death logic. | §5, §8, §9, §13 |
| `DS.G6.07` | future phase comparison uses a consistent clock reference or phase-invariant metric. | §5, §8, §9, §13 |
| `DS.G6.08` | future event right-censoring handled (no label leakage from 'not observed' to 'did not occur'). | §5, §8, §9, §13 |
| `DS.G6.09` | no frames from same route, parent physical realization, or intervention pair straddle split. | §5, §8, §9, §13 |
| `DS.G6.10` | causal prediction only uses interventions known at forecast time, not post-event geometry if claiming deployment realism. | §5, §8, §9, §13 |
| `DS.G7` | Gate G7 — storage, label validity, privacy, split integrity | §4, §8, §9, §11, §13 |
| `DS.G7.01` | all required fields present and typed; missing target fields carry reason codes rather than invented zero (partial: NR record v2… | §4, §8, §9, §11, §13 |
| `DS.G7.02` | each numeric target's units/gauge/aggregation source is registered and versioned. | §4, §8, §9, §11, §13 |
| `DS.G7.03` | nulls/infs/NaNs, negative powers, invalid durations, missing array axes handled explicitly. | §4, §8, §9, §11, §13 |
| `DS.G7.04` | target-mask shapes broadcast correctly and selection counts reported (partial: `NRChannelRecord` checks broadcastability, zero… | §4, §8, §9, §11, §13 |
| `DS.G7.05` | empty physical channels are valid records; loss functions handle them without divide-by-zero. | §4, §8, §9, §11, §13 |
| `DS.G7.06` | immutable writes verified by re-read and SHA checks including compressed chunk integrity. | §4, §8, §9, §11, §13 |
| `DS.G7.07` | input adapter cannot access oracle labels, file paths, source profile category or scene IDs unless experiment explicitly allows… | §4, §8, §9, §11, §13 |
| `DS.G7.08` | train/val/cal/test split lock signed before model tuning; neighboring spatial buffers and overlapping time windows detected.… | §4, §8, §9, §11, §13 |
| `DS.G7.09` | counterfactual twins, noise variants, resampled bandwidth copies and repeated routes inherit parent split. | §4, §8, §9, §11, §13 |
| `DS.G7.10` | exact cross-split geometry/site hashes and context-compatible channel correlation flag review candidates across the complete… | §4, §8, §9, §11, §13 |
| `DS.G7.11` | measured license and non-redistribution enforced by export routine, not only written documentation (partial:… | §4, §8, §9, §11, §13 |
| `DS.G7.12` | compressed data does not materially change physical target values; dtype and quantization error audited (partial:… | §4, §8, §9, §11, §13 |
| `DS.G7.13` | dataset can be loaded in shards with consistent deterministic order and no process-wide 100 GB materialization (partial:… | §4, §8, §9, §11, §13 |
| `DS.G7.14` | target-level valid-count coverage report generated for every split and family. | §4, §8, §9, §11, §13 |
| `DS.G7.15` | released dataset 'main' tests do not silently combine source-family-specific labels into fake universal classes. | §4, §8, §9, §11, §13 |
| `DS.G8` | Gate G8 — independent statistical and fairness checks | §9, §10, §13 |
| `DS.G8.01` | held-out scenes have both label classes for AUROC or explicitly mark AUROC undefined. | §9, §10, §13 |
| `DS.G8.02` | effective group sample counts used for CIs; not millions of correlated frames treated as millions of trials. | §9, §10, §13 |
| `DS.G8.03` | natural deployment priors and balanced stress priors evaluated separately; calibration corrected only with documented priors. | §9, §10, §13 |
| `DS.G8.04` | missing/low-confidence oracle items tracked and selection-bias audited. | §9, §10, §13 |
| `DS.G8.05` | dataset release includes at least one unseen site/material/antenna domain and a frozen external measured evaluation appropriate… | §9, §10, §13 |
| `DS.G8.06` | calibrations, normalization transforms and synthetic material fits trained on train-only data (or explicitly on calibration… | §9, §10, §13 |
| `DS.G8.07` | independent classic estimator baselines (PDP, peak delay, MUSIC/ESPRIT where applicable, OMP/SAGE/matched-filter etc.) used to… | §9, §10, §13 |
| `DS.G8.08` | null/falsification controls: time shuffled, phase randomized, source-ID-only, power-only, metadata-only, label-permuted and… | §9, §10, §13 |
| `DS.G8.09` | preregistered threshold selection and calibration do not query sealed test outcomes. | §9, §10, §13 |
| `DS.G8.10` | externally measured artifacts are independently source-verified, correctly timestamp-aligned and have explicit permitted target… | §9, §10, §13 |
| `PKG00-001` | Turn this backlog into a fully traceable feature map. `scripts.build_implementation_feature_map` indexes every… | §3, §13, §15, §16 |
| `PKG00-002` | Reconcile duplicate/chronological status entries in `MASTER_RESEARCH_PLAN.md` and `MVP_STATUS.md`; maintain one current summary… | §3, §13, §15, §16 |
| `PKG00-003` | Identify APIs that exist only as isolated helpers and wire them into the canonical generation, storage, audit, or evaluation path. | §3, §13, §15, §16 |
| `PKG00-004` | Mark every feature as code-complete, partial, absent, or externally blocked; do not use test-pass status to imply scientific… | §3, §13, §15, §16 |
| `PKG01-001` | Read RT cache entries from one stable single-link regular-file snapshot, enforce a configurable payload byte ceiling, reject… | §3, §8, §11, §13 |
| `PKG01-002` | Discover CPU/LLVM and CUDA/OptiX capabilities, probe CUDA/OptiX before deterministic LLVM fallback, and expose the selected… | §3, §8, §11, §13 |
| `PKG01-003` | Provide a standalone JSON capability report over the existing probed CUDA/OptiX-to-LLVM selector, retaining installed variants,… | §3, §8, §11, §13 |
| `PKG01-004` | Finish dependency/API compatibility matrix and exact native source, wheel, runtime, and model-table identities.… | §3, §8, §11, §13 |
| `PKG01-005` | Bind every material/property override, pose/motion state, antenna/pattern, carrier, solver option, and asset closure into… | §3, §8, §11, §13 |
| `PKG01-006` | Complete RNG stream and run-replay provenance contracts, including command, UTC time, commit/source/config hashes, input hashes,… | §3, §8, §11, §13 |
| `PKG01-007` | Add strict result-corpus promotion rules while retaining honest diagnostics for historical artifacts whose provenance cannot be… | §3, §8, §11, §13 |
| `PKG01-008` | Complete release-specific ODbL asset/extract/conversion provenance, attribution, source offer, and third-party license closure… | §3, §8, §11, §13 |
| `PKG02-001` | Complete the independent geometric segment-visibility oracle and its explicit clear/blocked/ambiguous outcomes, including… | §4, §7, §10, §13 |
| `PKG02-002` | Complete geometry and runtime-material consistency checks, antenna placement constraints, scene motion interpolation, and… | §4, §7, §10, §13 |
| `PKG02-003` | Finish scene-family/site independence and authoritative duplicate-asset inventory coverage before held-out allocation. Exact… | §4, §7, §10, §13 |
| `PKG02-004` | Implement the planned independent scene-family registry and sampling support for urban, suburban/rural, indoor office/factory,… | §4, §7, §10, §13 |
| `PKG02-005` | Complete generation support for single-link, multi-UE, multi-cell, and shared-interference topology layouts, with explicit… | §4, §7, §10, §13 |
| `PKG03-001` | Complete native `Paths` API/version adapters with shape, dtype, valid-mask, padding, normalization, and path-source assertions.… | §5, §8, §10, §13 |
| `PKG03-002` | Complete absolute delay/phase/power reconstruction and same-path-set direct/indirect decomposition; preserve raw solver-valid… | §5, §8, §10, §13 |
| `PKG03-003` | Complete analytical single/multiple-path checks for close delays, coherent nulls, nonzero reference offsets, link-budget… | §5, §8, §10, §13 |
| `PKG03-004` | Complete explicit/synthetic array comparison for near/far field, dual polarization, basis rotations, port order, and named axis… | §5, §8, §10, §13 |
| `PKG03-005` | Complete sparse variable-size path/channel serialization, lossless dtype-safe round trips, zero-path handling, checksums, atomic… | §5, §8, §10, §13 |
| `PKG03-006` | Integrate geometry-derived LOS separately from solver-found direct paths and standardized-model LOS proxies. | §5, §8, §10, §13 |
| `PKG04-001` | Require canonical, already-trimmed family, stratum, and path-episode labels in the shared acceptance-family builders and… | §6, §9, §10, §13 |
| `PKG04-002` | Define a frozen G3 qualification-run contract requiring candidate/challenger configurations, task scope, waveform/resources,… | §6, §9, §10, §13 |
| `PKG04-003` | Bind the frozen contract digest into the joint evaluator run receipt and cross-check its acceptance family, receiver… | §6, §9, §10, §13 |
| `PKG04-004` | Version the G3 run-manifest contract through v3: v2 binds `pyproject.toml` and `uv.lock`; v3 additionally records the… | §6, §9, §10, §13 |
| `PKG04-005` | Add a contract-bound evaluation entry point that rejects missing/mismatched preregistration before any evaluator branch executes… | §6, §9, §10, §13 |
| `PKG04-006` | Add optional detached OpenSSH signature verification for exact G3 joint run-manifest bytes against a caller-supplied… | §6, §9, §10, §13 |
| `PKG04-007` | Add optional external role/purpose authorization for the verified run-manifest signer; bind the policy ID and digest into the… | §6, §9, §10, §13 |
| `PKG04-008` | Give the G3 run-manifest verification report a strict v1 schema with a digest of the exact schema bytes, validate every verifier… | §6, §9, §10, §13 |
| `PKG04-009` | Add opt-in strict signed-reviewer admission to G3 run-manifest verification: require a v3 receipt, detached receipt signature,… | §6, §9, §10, §13 |
| `PKG04-010` | Require the `split_preregistration` attestation subject roster to include the exact evaluator-bound split-manifest file, whose… | §6, §9, §10, §13 |
| `PKG04-011` | Require the G3 joint evaluator's complex-CSI, propagation, receiver-impact, and path-equivalence branches to declare identical… | §6, §9, §10, §13 |
| `PKG04-012` | Add joint-evaluator input v3 with receiver scene/condition/episode IDs and require the observed episode set to equal the frozen… | §6, §9, §10, §13 |
| `PKG04-013` | Bind candidate/challenger configurations, waveform, and link-budget/noise declarations in evaluator-input v2/v3 to the frozen… | §6, §9, §10, §13 |
| `PKG04-014` | Require the frozen contract's path-equivalence coverage threshold to equal the shared acceptance-family criterion, which is… | §6, §9, §10, §13 |
| `PKG04-015` | Preserve the frozen G3 v2 run contract for historical diagnostic receipts; add v3 with embedded design digest, family/stratum… | §6, §9, §10, §13 |
| `PKG04-016` | Compose all four evaluator-aligned G3 assurance outputs with family-bound branch planning into a per-stratum design artifact,… | §6, §9, §10, §13 |
| `PKG04-017` | Bound the G3 paired receiver evaluator's retained bootstrap output arrays by a declared float64-cell ceiling, reject oversized… | §6, §9, §10, §13 |
| `PKG04-018` | Add qualification contract v4 with a hash-bound path-analysis parameter set for vertex/direction tolerances, complex-CSI floor,… | §6, §9, §10, §13 |
| `PKG04-019` | Add contract-bound evaluator input v5 for solver-valid deterministic paths and full CFR components; recompute effective classes,… | §6, §9, §10, §13 |
| `PKG04-020` | Add externally trusted runtime/source/measurement provenance and signature admission checks before any scientific qualification… | §6, §9, §10, §13 |
| `PKG04-021` | Complete scientific admission for the shared family: validate material-scattering acquisition/split coverage against its… | §6, §9, §10, §13 |
| `PKG04-022` | Implement and integrate material-scattering scorer/assurance software for held-out angle/material/polarization strata,… | §6, §9, §10, §13 |
| `PKG04-023` | Implement and integrate the complex-CSI evaluator for the frozen unnormalized active-grid/port/symbol/gauge definition,… | §6, §9, §10, §13 |
| `PKG04-024` | Add acquisition-package v0.4 with a fixed `noise_only_cfr_estimate_rms_v1` definition, bounded exact-byte capture of referenced… | §6, §9, §10, §13 |
| `PKG04-025` | Implement and integrate the propagation evaluator for first-arrival delay, RMS delay, 3D spherical angular spreads,… | §6, §9, §10, §13 |
| `PKG04-026` | Integrate diffuse stochastic paths into contract-bound evaluator input v6 and roster diagnostic v2. The joint scorer keeps… | §6, §9, §10, §13 |
| `PKG04-027` | Implement and integrate the paired NR receiver evaluator for common-SNR BLER differences, familywise cluster-bootstrap… | §6, §9, §10, §13 |
| `PKG04-028` | Add a hash-bound joint numerical-criterion roll-up for material pilot margins, complex CSI, propagation metrics, receiver… | §6, §9, §10, §13 |
| `PKG04-029` | Complete report/run-manifest signature and trust-policy plumbing. Detached exact-byte verification, run-manifest signer roles,… | §6, §9, §10, §13 |
| `PKG04-030` | Add an explicit strict G3.03 attestation-set mode that requires verified signatures for all five declared types, including… | §6, §9, §10, §13 |
| `PKG04-031` | Add optional detached OpenSSH verification over the exact external run-manifest signer-policy bytes, under a separate namespace,… | §6, §9, §10, §13 |
| `PKG04-032` | Complete the bounded City43 Phase-0 diagnosis from existing path-level reports in… | §6, §9, §10, §13 |
| `PKG04-033` | Extend the signed G3.03 attestation verifier to support v0.4 acquisition packages and add single-attestation and complete-set… | §6, §9, §10, §13 |
| `PKG05-001` | Retain the preregistered G4.10 physical dual-cross handheld CDL covariance family and its 10 passing native cases and 3,200… | §7, §8, §10, §13 |
| `PKG05-002` | Inventory profile applicability in one canonical, versioned TDL/CDL ledger and use it in record admission; expose a… | §7, §8, §10, §13 |
| `PKG05-003` | Publish a separate hash-bound generator capability matrix derived from the canonical profile ledger. It records every supported… | §7, §8, §10, §13 |
| `PKG05-004` | Convert explicit scenario-GBSM outputs to per-drop records and publish them through a bounded, content-addressed scenario run… | §7, §8, §10, §13 |
| `PKG05-005` | Connect the UMi/UMa/RMa/InH/InF Sionna scenario adapter to a bounded CLI generation path. The v1 config names every split-parent… | §7, §8, §10, §13 |
| `PKG05-006` | Inventory each remaining G4 gate against existing native checks and implement only genuine gaps: hash-pinned tables,… | §7, §8, §10, §13 |
| `PKG05-007` | Bind TDL resolved-profile metadata to a canonical SHA-256 digest and verify it at standardized-record admission when present.… | §7, §8, §10, §13 |
| `PKG05-008` | Forward the requested `num_sinusoids` value to the Sionna 2.2 TDL constructor. A dependency-isolated regression definition… | §7, §8, §10, §13 |
| `PKG05-009` | Record the SHA-256 of the exact packaged Sionna TDL JSON resource selected for scalable or fixed-delay profiles, bind its… | §7, §8, §10, §13 |
| `PKG05-010` | Read CDL parameter tables as package `Traversable` resources, parse and hash one exact byte snapshot, record canonical TR 38.901… | §7, §8, §10, §13 |
| `PKG05-011` | Pin exact ETSI source identities for TR 38.901 v16.1.0/v19.2.0 and TS 38.101-4 v19.2.2 without checking the PDFs into the… | §7, §8, §10, §13 |
| `PKG05-012` | Add independent TS 38.101-4 Annex B RMS-delay-spread scalar, tap-summary, and per-component delay/power/fading references for… | §7, §8, §10, §13 |
| `PKG05-013` | Derive fixed TDL-D10/D30 LOS K-factor expectations from the separately listed Annex B LOS and Rayleigh component powers, with… | §7, §8, §10, §13 |
| `PKG05-014` | Reject Boolean and non-real physical scalar inputs at the TDL/CDL adapter boundary, including frequency, delay, speed,… | §7, §8, §10, §13 |
| `PKG05-015` | Complete confidence-controlled tests for remaining profile/configuration/topology combinations and criteria at the gate's… | §7, §8, §10, §13 |
| `PKG05-016` | Publish a strict source/version/capability matrix for every TDL/CDL scalable and fixed profile, declare supported single-link… | §7, §8, §10, §13 |
| `PKG05-017` | Add an independent, source-hash-bound compact scalar reference for CDL-A–E LOS status, component/output-cluster counts,… | §7, §8, §10, §13 |
| `PKG05-018` | Make the CDL-A–E draw-level ensemble PDP/RMS confidence regression compare empirical channel draws against powers and delays… | §7, §8, §10, §13 |
| `PKG05-019` | Remove fixed-TDL self-reference from the draw-level PDP/RMS confidence regression. Its fixed A10/A30/B100/C60/C300/D10/D30… | §7, §8, §10, §13 |
| `PKG05-020` | Rebase the all-profile dual-VH CDL polarization power regressions on the byte-pinned ETSI per-cluster and scalar reference… | §7, §8, §10, §13 |
| `PKG05-021` | Rebase the all-profile four-angle sector-power confidence regression on the byte-pinned ETSI per-cluster… | §7, §8, §10, §13 |
| `PKG05-022` | Rebase the CDL directional-element and dual-end antenna-pattern power regressions on byte-pinned ETSI cluster… | §7, §8, §10, §13 |
| `PKG05-023` | Rebase CDL complex-pseudocovariance family accounting on the byte-pinned ETSI profile roster for output cluster counts and LOS… | §7, §8, §10, §13 |
| `PKG05-024` | Extend the four-element half-wavelength receive-array covariance reference across CDL-A–E, deriving each expectation from the… | §7, §8, §10, §13 |
| `PKG05-025` | Add an independent normative reference and confidence-controlled regression for CDL-D/E LOS dual-polarized two-element receive… | §7, §8, §10, §13 |
| `PKG05-026` | Extend independent normative references to remaining multi-element array coupling, covariance, and temporal branches. The… | §7, §8, §10, §13 |
| `PKG06-001` | Implement and focused-validate the TS 38.211 type-1 SRS frequency-tree hopping recurrence and its per-symbol scheduled-counter… | §5, §6, §10, §13 |
| `PKG06-002` | Validate shared AWGN inputs before dtype conversion, reject Boolean/complex/sparse/nonnumeric and nonphysical variances, and… | §5, §6, §10, §13 |
| `PKG06-003` | Complete PUSCH/DMRS resource mapping and decoder/CRC paths over remaining allocation, mapping, CDM/port, hopping,… | §5, §6, §10, §13 |
| `PKG06-004` | Complete realistic signal scaling, multi-link/multi-cell waveform superposition, interference/contamination/SINR truth, per-link… | §5, §6, §10, §13 |
| `PKG06-005` | Reject short or trailing sample sequences at the network PUSCH handoff by binding the configured scalar cyclic-prefix length and… | §5, §6, §10, §13 |
| `PKG06-006` | Complete impairment composition and receiver measurement paths for timing, CFO, phase noise, IQ imbalance, AWGN, Doppler, ISI/CP… | §5, §6, §10, §13 |
| `PKG06-007` | Derive DCI 0_3 field widths for format identifier, scheduled-cell-set and scheduled-cell-combination indicators, BWP indicator,… | §5, §6, §10, §13 |
| `PKG06-008` | Derive DCI 0_3 priority-indicator and minimum-applicable-scheduling-offset widths from decoded configuration-presence flags,… | §5, §6, §10, §13 |
| `PKG06-009` | Derive every field width in the ordered DCI 0_3 layout from decoded configuration or fixed format rules, including TPMI codebook… | §5, §6, §10, §13 |
| `PKG06-010` | Finish remaining G5.14 SRS gaps: connect the current DCI 0_3 decoded-state width resolver to real ASN.1/RRC decoding and… | §5, §6, §10, §13 |
| `PKG06-011` | Bind DCI 0_3 BWP-aware SRS schedules to caller-supplied per-BWP frequency extents and reject out-of-BWP allocations; retain… | §5, §6, §10, §13 |
| `PKG06-012` | Add a bounded same-grid SRS inter-cell resource-element overlap audit with a versioned report schema, pairwise overlap counts, a… | §5, §6, §10, §13 |
| `PKG06-013` | Add the bounded multiport SRS receive primitive for mapped ordinary/hopped/positioning grids, with rank-aware per-subcarrier LS… | §5, §6, §10, §13 |
| `PKG06-014` | Extend multiport SRS estimation to accept a caller-supplied per-RX/per-symbol noise-variance profile and use weighted least… | §5, §6, §10, §13 |
| `PKG06-015` | Report exact one-sided 95% independent-channel prevalence bounds for any CRC failure in the clean/corrected/uncorrected… | §5, §6, §10, §13 |
| `PKG06-016` | Bind new G5.09 reports to the complete adapter receiver metadata, selected decoder/estimator contract, and exact runtime Sionna… | §5, §6, §10, §13 |
| `PKG06-017` | Bound independent-channel G5.09 receiver work to at most 16,384 total transport blocks before constructing the Sionna receiver;… | §5, §6, §10, §13 |
| `PKG06-018` | Bound SNR-curve input preprocessing as well as receiver work: cap the SNR count before materializing it, derive the maximum… | §5, §6, §10, §13 |
| `PKG06-019` | Calculate CRC-failure totals and block BLER numerators directly from retained Boolean transport-block outcomes instead of… | §5, §6, §10, §13 |
| `PKG06-020` | Enforce a real, finite, nonnegative PUSCH receiver noise-variance boundary before float32 conversion, reject… | §5, §6, §10, §13 |
| `PKG06-021` | Apply the TS 38.211 periodic positioning-SRS Hyper-SFN parity rule for configured periods at or above `2^mu * 10240` slots, and… | §5, §6, §10, §13 |
| `PKG06-022` | Compose decoded periodic/semi-persistent SRS resources into one resource-set schedule, allowing each resource's… | §5, §6, §10, §13 |
| `PKG06-023` | Extend periodic/semi-persistent SRS composition through selected-UL-BWP frequency feasibility. The exported BWP-aware API checks… | §5, §6, §10, §13 |
| `PKG06-024` | Compose decoded periodic/semi-persistent SRS resource sets over a frozen multi-cell roster, with a per-cell absolute slot and… | §5, §6, §10, §13 |
| `PKG06-025` | Finish SSB/PBCH generalization: general ASN.1/PER extension decoding, case/band/beam/candidate and frame integration, multiple… | §5, §6, §10, §13 |
| `PKG06-026` | Reject Boolean and string aliases in PSS/SSS acquisition's physical sample rate, detection thresholds, CFO grid, and… | §5, §6, §10, §13 |
| `PKG06-027` | Implement PRS only as a standards-defined signal and receiver with independently specified mapping and timing observables; no… | §5, §6, §10, §13 |
| `PKG06-028` | Implement the TS 38.211 V19.2.0 downlink PRS Gold/QPSK sequence, supported symbol/comb pairs, per-symbol comb-offset mapping… | §5, §6, §10, §13 |
| `PKG06-029` | Add the bounded NR DL-RSTD computation from already-estimated received subframe-start timestamps: select the unique nearest… | §5, §6, §10, §13 |
| `PKG06-030` | Resolve downlink PRS slot occasions from decoded TS 38.214 periodicity, resource slot offset, repetitions/time gap, Option-1 bit… | §5, §6, §10, §13 |
| `PKG06-031` | Add a bounded paired diagnostic SNR-curve path for the independent ordered-impairment fixture. Every point reuses channel,… | §5, §6, §10, §13 |
| `PKG06-032` | Add a pre-run G5.09 SNR-family acceptance contract that binds the exact SNR grid, channel-seed roster, estimator, blocks per… | §5, §6, §10, §13 |
| `PKG06-033` | Keep G5 SNR-family contract freezing and interval scoring importable without Torch or the NR receiver engine; verify roster,… | §5, §6, §10, §13 |
| `PKG06-034` | Keep retained G5.09 result auditing importable without Torch by loading the receiver only at the deterministic-replay call… | §5, §6, §10, §13 |
| `PKG06-035` | Finish receiver evaluation APIs and artifact/report binding for exact waveform, decoder/source configuration, multi-point SNR… | §5, §6, §10, §13 |
| `PKG06-036` | Retain batch-seed cluster identity in sample-domain PUSCH SNR curves and add a paired LS-versus-finite-delay-DFT comparator.… | §5, §6, §10, §13 |
| `PKG07-001` | Finish synchronized time axes across TX/RX/objects, timestamp-based future-target alignment, monotonicity/gap handling, and… | §5, §8, §9, §13 |
| `PKG07-002` | Complete Doppler checks against coefficient evolution and bounded physical retracing, including path birth/death and association… | §5, §8, §9, §13 |
| `PKG07-003` | Complete future phase/gauge and censoring semantics, observability/missing-reason rules, and causal feature eligibility. | §5, §8, §9, §13 |
| `PKG07-004` | Complete paired frequency-view solves with band/material-specific propagation; do not extrapolate measured magnitude-only data… | §5, §8, §9, §13 |
| `PKG07-005` | Implement paired mathematical direct-path ablation from one frozen `PathOracleTable`, preserving absolute delay/coefficient… | §5, §8, §9, §13 |
| `PKG08-001` | Make split builders and audits reject missing, blank, non-string, and padded sample/group IDs instead of coercing them into… | §4, §7, §9, §13 |
| `PKG08-002` | Reject split-fraction type coercion at `SplitPolicy` construction: require finite real numeric values and reject booleans,… | §4, §7, §9, §13 |
| `PKG08-003` | Apply the same strict numeric boundary to spatial-buffer derivation: accept finite `numbers.Real` scalars (including NumPy real… | §4, §7, §9, §13 |
| `PKG08-004` | Add a candidate split-registry build CLI that derives the canonical leakage graph from bounded saved ChannelDataset snapshots,… | §4, §7, §9, §13 |
| `PKG08-005` | Wire complete-provenance legacy Sionna RT ChannelDatasets through the canonical RT split-identity migration before… | §4, §7, §9, §13 |
| `PKG08-006` | Derive exact shared-mesh leakage relations from complete, digest-verified scene manifests when building saved-ChannelDataset… | §4, §7, §9, §13 |
| `PKG08-007` | Require every newly persisted NR v3 target to have a complete, digest-verified versioned semantic definition; retain v1/v2 read… | §4, §6, §8, §10, §13 |
| `PKG08-008` | Complete domain-specific target units/reference/gauge/aggregation semantics, shapes, allowed ranges, and canonical migration… | §4, §6, §8, §10, §13 |
| `PKG08-009` | Allow finite complex arrays in versioned UMCD `TargetValue` contracts when their canonical definition declares complex CFR,… | §4, §6, §8, §10, §13 |
| `PKG08-010` | Bind all four canonical NR CFR target versions to their observation shapes at versioned storage admission: standardized/TDL… | §4, §6, §8, §10, §13 |
| `PKG08-011` | Enforce value-domain rules for the currently canonical standardized profile-LOS proxy, NR CFR, and geometric-LOS definition… | §4, §6, §8, §10, §13 |
| `PKG08-012` | Enforce zero storage for every NR target element excluded by its broadcasted false mask, matching the canonical `TargetValue`… | §4, §6, §8, §10, §13 |
| `PKG08-013` | Require one explicit `TargetValidity` reason per NR target element whenever a mask excludes that element, enforce exact… | §4, §6, §8, §10, §13 |
| `PKG08-014` | Implement NR trajectory target-support aggregation by explicit split × family × target inventory, with present/missing records,… | §7, §9, §11, §13 |
| `PKG08-015` | Establish trusted custody for the inventory-signature key and exact scene/band/topology/antenna/task inventories, and provide… | §7, §9, §11, §13 |
| `PKG08-016` | signed near-duplicate dispositions to the canonical UMCD corpus audit. Report v8 hashes each candidate and the complete… | §7, §9, §11, §13 |
| `PKG08-017` | Complete model-input leakage audit across every dataset adapter, feature transform, experiment runner, training path, and… | §4, §6, §9, §10, §13 |
| `PKG08-018` | a fail-closed inference-input field-roster audit for legacy CFR, analytical multipath, standardized-channel, and NR receiver… | §4, §6, §9, §10, §13 |
| `PKG08-019` | idate values as well as field names at registered model-input boundaries: allow only finite dense numeric arrays in the declared… | §4, §6, §9, §10, §13 |
| `PKG08-020` | olve allowlisted model-input and nested NR configuration identities against the actual class objects exported by their… | §4, §6, §9, §10, §13 |
| `PKG08-021` | end observation-only feature provenance through the specialized profile-holdout runner: bind the registered `ChannelModelInputs`… | §4, §6, §9, §10, §13 |
| `PKG08-022` | d the canonical experiment runner's saved models to the exact full feature matrix, transform implementation digest, and… | §4, §6, §9, §10, §13 |
| `PKG08-023` | Implement immutable content-addressed NR and UMCD episode stores with manifest-last publication, deterministic reader batches,… | §3, §8, §11, §13 |
| `PKG08-024` | Complete cross-format corruption and retry/failure accounting, audit every promised storage format under interrupted writes, and… | §3, §8, §11, §13 |
| `PKG08-025` | Finish signed split-lock/release audit integration: exact parent/child roster, all child artifacts and row bindings, complete… | §4, §9, §11, §13 |
| `PKG08-026` | Require frozen-split proximity audits to bind an explicit canonical coordinate-frame ID, reject cross-frame comparisons and… | §4, §9, §11, §13 |
| `PKG08-027` | Add an explicit complete-split-integrity mode to the signed lineage audit. It requires a signed frozen corpus inventory and… | §4, §9, §11, §13 |
| `PKG08-028` | Version the seed-roster collision report to v2 and continue after result files exceed the declared input/decompression ceiling;… | §4, §9, §11, §13 |
| `PKG08-029` | Verify signed child-artifact rosters against exact frozen child bindings, hash each artifact beneath the roster directory… | §4, §9, §11, §13 |
| `PKG08-030` | Report `child_artifact_semantic_rows_verified` as true only when every rostered artifact has a semantic validator; mixed legacy… | §4, §9, §11, §13 |
| `PKG08-031` | Fail closed when a signed split-lock audit receives provisional RT scene identities; allow the unsigned diagnostic path to… | §4, §9, §11, §13 |
| `PKG08-032` | Bound manifest memory in split-lineage audit reads for the child-binding roster, frozen inventory, and legacy reconstruction… | §4, §9, §11, §13 |
| `PKG08-033` | Bound split-policy registry/payload, detached-signature, and allowed-signers reads to 256 MiB, 1 MiB, and 4 MiB respectively… | §4, §9, §11, §13 |
| `PKG08-034` | Make split-policy lock publication enforce the same 256 MiB ceiling as the canonical reader, preventing creation of immutable… | §4, §9, §11, §13 |
| `PKG08-035` | Fail closed at the split-registry JSON boundary on invalid UTF-8/JSON, non-object roots/containers, non-integer schema versions,… | §4, §9, §11, §13 |
| `PKG08-036` | Add signed child-artifact roster v2 for multiple uniquely identified typed artifacts per frozen child. Require at least one… | §4, §9, §11, §13 |
| `PKG08-037` | Add signed child-artifact roster v3 with an exact regular-file inventory under one declared artifact subtree. Reject symlinks,… | §4, §9, §11, §13 |
| `PKG08-038` | Add roster v4 for exact-tree UMCD episode stores: semantically read each episode bundle under a conservative 512 MiB… | §4, §9, §11, §13 |
| `PKG08-039` | Implement the fail-closed ODbL release exporter and strict v3 per-artifact rights/provenance manifest, including exact field… | §7, §11, §13 |
| `PKG09-001` | Reject normalized profile-holdout metadata identities at its model-input boundary: profile labels must be canonical uppercase… | §4, §6, §9, §10, §13 |
| `PKG09-002` | Require explicit independent-group identity for target-coverage counts; the canonical UMCD auditor supplies the frozen… | §4, §6, §9, §10, §13 |
| `PKG09-003` | Complete class availability and effective independent-group reporting; retain undefined metrics rather than inventing values.… | §4, §6, §9, §10, §13 |
| `PKG09-004` | Separate natural and enriched stress priors; make calibration prior-aware and train-only. the legacy common… | §4, §6, §9, §10, §13 |
| `PKG09-005` | Add selection/missingness/low-confidence audits and held-out site/material/antenna/external-domain eligibility. the… | §4, §6, §9, §10, §13 |
| `PKG09-006` | Implement train-only normalization/calibration enforcement, sealed-test protections, and preregistered threshold-selection… | §4, §6, §9, §10, §13 |
| `PKG09-007` | Implement classic propagation/estimation baselines and falsification/leakage controls (metadata/source-only, power-only,… | §4, §6, §9, §10, §13 |
| `PKG09-008` | Add a deliberately leaky `oracle_label_leak` falsification route to the canonical runner using the same split and scorer as… | §4, §6, §9, §10, §13 |
| `PKG09-009` | Complete uncertainty calibration and clustered inference at the independent physical episode/site/acquisition-unit level. | §4, §6, §9, §10, §13 |
| `PKG10-001` | Connect versioned configuration/schema registries to an immutable build DAG with content-addressed cache keys and idempotent… | §3, §8, §12, §13 |
| `PKG10-002` | Add the shared build-node manifest/API foundation: JSON-Schema-validate each node configuration; bind configuration-schema… | §3, §8, §12, §13 |
| `PKG10-003` | Wire ordered channel-dataset concatenation in the canonical experiment runner through the shared build-node API. The run… | §8, §9, §13 |
| `PKG10-004` | Wire candidate split-registry construction through an optional immutable build-DAG node. The key binds named dataset bytes,… | §8, §9, §13 |
| `PKG10-005` | Implement the planned generation tiers and explicit data lineage/status labels (`proposed`, `synthetic_debug`,… | §8, §12, §13 |
| `PKG10-006` | Implement a deterministic pilot sampler/report over the declared mandatory channel-family, antenna-topology, difficulty, and… | §8, §12, §13 |
| `PKG10-007` | Implement resource budgeting/reporting for solver, receiver, raw path, CFR, manifests, storage, and downstream analysis before… | §8, §12, §13 |
| `PKG10-008` | Implement benchmark materialization and scoring consumption for… | §8, §9, §10, §13 |
| `PKG10-009` | Implement baseline training/evaluation for physical estimators, existing MLP, parameter-matched MLP/CNN/CV-CNN, and sparse… | §9, §10, §13 |
| `PKG10-010` | Implement the proposed phase-symmetry/physics-reconstruction model and uncertainty/observability heads only after inputs,… | §9, §10, §13 |
| `PKG10-011` | Generate paper-ready tables/figures/reproduction bundle from frozen data/code identities after all required evidence gates pass. | §10, §11, §13 |
| `RS.AppendixE.G0` | repo integrity | §3, §8, §10, §13 |
| `RS.AppendixE.G1` | native physics | §5, §7, §10, §13 |
| `RS.AppendixE.G2` | receiver validity | §5, §6, §10, §13 |
| `RS.AppendixE.G3` | data validity | §4, §6, §7, §8, §9, §11, §13 |
| `RS.AppendixE.G4` | strong baselines | §2, §9, §10, §13 |
| `RS.AppendixE.G5` | credible novelty | §2, §13, §15 |
| `RS.AppendixE.G6` | external evidence | §2, §7, §10, §11, §13 |
| `RS.AppendixE.G7` | added forecasting claim | §5, §9, §13 |
| `RS.AppendixE.G8` | system/measurement claim | §6, §10, §13 |
| `RS.AppendixE.G9` | release | §7, §11, §13 |
| `RS.Phase0` | Phase 0 — Freeze and audit (P0, immediate) | §2, §13, §14 |
| `RS.Phase1` | Phase 1 — Physical truth/units and native Sionna contract (P0) | §5, §8, §10, §13 |
| `RS.Phase10` | Phase 10 — Manuscript and review-readiness (P0) | §2, §11, §13, §16 |
| `RS.Phase2` | Phase 2 — Standardized and RT data generator integrity (P0) | §7, §8, §10, §13 |
| `RS.Phase3` | Phase 3 — NR receiver and timing calibration (P0) | §5, §6, §10, §13 |
| `RS.Phase4` | Phase 4 — Matched legacy + modern baselines (P0) | §2, §6, §9, §10, §13 |
| `RS.Phase5` | Phase 5 — Invent and validate one novel mechanism (P0) | §2, §9, §10, §13 |
| `RS.Phase6` | Phase 6 — Cross-domain robustness (P0) | §2, §7, §9, §10, §13 |
| `RS.Phase7` | Phase 7 — Dynamic RT and forecasting (P1; conditionally flagship) | §5, §6, §9, §10, §13 |
| `RS.Phase8` | Phase 8 — Independent measurement campaign (P1 but pivotal for journal strength) | §7, §9, §10, §11, §13 |
| `RS.Phase9` | Phase 9 — Full sensitivity/cost/reproduction (P0 before submission) | §8, §10, §12, §13 |

### 15.5 External standards, simulator documentation, and related work

The following sources inform the design and novelty boundary above. Versioned standards and simulator documentation govern exact behavior; publications contextualize research claims but do not override project contracts. The CFM-Bench source is an arXiv preprint at this plan edition.

- 3GPP / ETSI, [TR 138 901 V19.2.0 (Release 19)](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.02.00_60/tr_138901v190200p.pdf), especially clauses 7.7.1–7.7.2 for CDL/TDL profiles. The project currently pins the Sionna 2.2 integration against V19.2; do not silently update that contract because a later TR revision exists.
- NVIDIA, [Sionna PHY 2.2 TR 38.901 documentation](https://nvlabs.github.io/sionna/v2.2.0/phy/api/channel/wireless/3gpp.html); [Sionna RT 2.2 PathSolver API](https://nvlabs.github.io/sionna/v2.2.0/rt/api/paths_solvers.html); [Paths API](https://nvlabs.github.io/sionna/v2.2.0/rt/api/paths.html); and [Introduction to Sionna RT](https://nvlabs.github.io/sionna/v2.2.0/rt/tutorials/Introduction.html). These are the versioned API references for the repository integration target.
- Hoydis et al., [“Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling” (2023)](https://arxiv.org/abs/2303.11103), and Aït Aoudia et al., [“Sionna RT: Technical Report” (2025)](https://arxiv.org/abs/2504.21719). Use them to understand simulator algorithms and limitations, not as evidence that a particular scene is physically calibrated.
- Di Cicco et al., [“Machine Learning-Based Line-Of-Sight Prediction in Urban Manhattan-Like Environments,” EuCAP 2023](https://doi.org/10.23919/EuCAP57121.2023.10133145). The paper evaluates a feature-based GBDT over unseen synthetic Manhattan topologies and motivates imbalance-aware PR metrics.
- Almutairi et al., [“Deep Transfer Learning for Cross-Device Channel Classification in mmWave Wireless,” IEEE MSN 2021](https://doi.org/10.1109/MSN53354.2021.00034). This is adjacent prior work using mmWave beam-training messages and device-domain transfer; it is not a standards-constrained NR PUSCH/DMRS transfer study.
- Gao et al., [“CFM-Bench: A Unified Multi-Domain, Multi-Task Benchmark for Channel Foundation Models,” arXiv:2607.14975 (2026)](https://arxiv.org/abs/2607.14975). The benchmark covers six channel domains and several tasks; its LoS/NLoS task is supported in only two domains, including a 3GPP statistical domain and a geometry-labeled RT domain. Compare against its scope without claiming priority over the broad multi-domain benchmark idea.

## 16. Definition of a complete plan and plan maintenance

### 16.1 Review record

- **2026-11-26 — Independent repository and targeted literature review.** Reconciled the headline research objective with current RT-only LOSO evidence and the separate profile proxy; distinguished episode parents from family-specific atomic records; reviewed current Sionna 2.2 API behavior, 3GPP TR 38.901 V19.2.0, prior ray-traced LOS prediction and channel-transfer work, and the CFM-Bench preprint. Updated the novelty boundary, Sionna path/delay/determinism handling, comparison protocol, and external bibliography (§2, §5.2, §9, §13, §14, §15.5). This was a targeted review, not a systematic literature review, a test run, or a scientific rerun.


This plan is complete as a design authority when:

1. Each entity, layer, target, family, producer, task, split, gate and release artifact has one named semantic owner or an explicit unsupported disposition.
2. Current source facts are distinguished from intended capabilities and proposed research choices.
3. All target eligibility and forbidden interpretations are explicit enough that two producers cannot silently publish semantically different values under one name/version.
4. Every active requirement is cross-referenced or dispositioned; no plan obligation has an unowned implementation or qualification dependency.
5. Scope, sampling, compute, and release claims are conditional on evidence and authorization; no invented fixed dataset size or success threshold remains.
6. Scientific truth, receiver observability, lineage, licensing, split independence, and qualification boundaries are preserved.
7. Links, dates, schema versions, source locations, and requirement references are checked against the checkout used for the edition.

Review this plan when a governing protocol changes, canonical dataset contract changes, a family gains or loses eligibility, a producer/reader path is added or retired, or a release scope is proposed. Add a short dated change record with changed sections and evidence references. Do not append routine test counts or every ledger status update to the plan; keep those in execution/evidence records.

**Completion statement:** This document specifies the intended unified dataset program and the evidence required to realize it. It does not declare UMCD complete, authorize a generation campaign, qualify any family, or establish a publication claim.
