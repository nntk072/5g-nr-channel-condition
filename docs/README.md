# Documentation

Start here. This index lists every document under `docs/`.

## Guide

How-to guides and workflows.

- [Shared-split power-only baseline](guide/BASELINE_CONTROLS_WORKFLOW.md)
- [Luna execution guide for the 5G NR Multipath MVP](guide/LUNA_EXECUTION_GUIDE_2026-11-11.md)
- [Sample-domain network waveform to PUSCH receiver](guide/NETWORK_PUSCH_RECEIVER_WORKFLOW.md)
- [UMCD pilot sampling workflow](guide/PILOT_SAMPLING_WORKFLOW.md)
- [Start here — reproducible LOS/NLOS research upgrade](guide/QUICKSTART.md)
- [Measurement-driven resource budget preflight](guide/RESOURCE_BUDGET_WORKFLOW.md)
- [Runtime identity inventory](guide/RUNTIME_INVENTORY_WORKFLOW.md)
- [G4 scenario GBSM generation command v1](guide/SCENARIO_GBSM_GENERATION_CLI_V1.md)

## Architecture

System design, package map, CI, and decision records.

- [Refactor baseline](architecture/BASELINE.md)
- [Benchmark partition materialization](architecture/BENCHMARK_PARTITION_MATERIALIZATION.md)
- [Continuous integration](architecture/CI.md)
- [Model input route inventory — working audit](architecture/MODEL_INPUT_ROUTE_INVENTORY.md)
- [Module split design notes](architecture/MODULE_SPLIT_DESIGN_NOTES.md)
- [Naming exceptions](architecture/NAMING_EXCEPTIONS.md)
- [Multi-link RT to network PHY support](architecture/NETWORK_TOPOLOGY_SUPPORT.md)
- [Analytical two-path decoder: implementation contract](architecture/OBSERVABLE_MULTIPATH_MODEL.md)
- [Model outputs and channel families](architecture/OUTPUT_TASKS_AND_CHANNELS.md)
- [Repository Refactor Plan](architecture/REFACTOR_PLAN.md)
- [Sionna RT backend selection](architecture/RT_BACKEND_SELECTION.md)
- [Sionna RT run identity](architecture/RT_RUN_IDENTITY.md)
- [RT scene asset identity](architecture/RT_SCENE_ASSET_IDENTITY.md)
- [External Sionna source reference](architecture/SIONNA_REFERENCE.md)
- [Grouped scalar-regression uncertainty evaluation](architecture/UNCERTAINTY_EVALUATION.md)

## Operations

Agent operations, task protocol, compute policy, and release acceptance.

- [Agent Compute Policy](operations/AGENT_COMPUTE_POLICY.md)
- [Agent Operations](operations/AGENT_OPERATIONS.md)
- [AgentOps P14 Release Acceptance Report](operations/AGENT_RELEASE_ACCEPTANCE_2026-11-19.md)
- [Agent Task Protocol](operations/AGENT_TASK_PROTOCOL.md)
- [Probability calibration and prior policy](operations/PROBABILITY_PRIOR_POLICY.md)
- [Production-structure refactor: verification record](operations/REFACTOR_VERIFICATION.md)
- [UMCD release split policy v2](operations/RELEASE_SPLIT_POLICY_V2.md)
- [Research result promotion](operations/RESULT_PROMOTION_POLICY.md)
- [Research upgrade changelog — 2026-10-21](operations/UPGRADE_CHANGELOG.md)

## Reference

Generated matrices, registries, and the rebinding record.

- [Helper-to-canonical-route wiring matrix](reference/HELPER_WIRING_MATRIX.md)
- [Scene-family registry and candidate sampling](reference/SCENE_FAMILY_REGISTRY.md)
- [Canonical target producer and consumer matrix](reference/TARGET_FAMILY_PRODUCER_CONSUMER_MATRIX.md)
- [digest-rebindings.json](reference/digest-rebindings.json)
- [refactor-digest-changes.tsv](reference/refactor-digest-changes.tsv)

## Thesis

Thesis document.

- [NguyenLong.pdf](thesis/NguyenLong.pdf)

## Research

Research notes, protocols, contracts, and status: [research/INDEX.md](research/INDEX.md).
Retained result files are indexed in [research/results/INDEX.md](research/results/INDEX.md).
Result files are retained evidence and are not edited.

Historical records:

- The master research plan is no longer part of the tree. It is kept at the git tag `docs/master-research-plan-archive`.
- The legacy 2025 code is kept at the git tag `archive/legacy-2025`.
