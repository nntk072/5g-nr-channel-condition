# Documentation

Start here. This index lists every document under `docs/`.

## Guide

Setup and workflows for users of the framework.

- [Shared-split power-only baseline](guide/baseline-controls-workflow.md)
- [Luna execution guide for the 5G NR Multipath MVP](guide/luna-execution-guide-2026-11-11.md)
- [Sample-domain network waveform to PUSCH receiver](guide/network-pusch-receiver-workflow.md)
- [UMCD pilot sampling workflow](guide/pilot-sampling-workflow.md)
- [Quickstart](guide/quickstart.md)
- [Measurement-driven resource budget preflight](guide/resource-budget-workflow.md)
- [Runtime identity inventory](guide/runtime-inventory-workflow.md)
- [G4 scenario GBSM generation command v1](guide/scenario-gbsm-generation-cli-v1.md)

## Architecture

Concepts and output contracts for users of the framework.

- [Benchmark partition materialization](architecture/benchmark-partition-materialization.md)
- [Multi-link RT to network PHY support](architecture/network-topology-support.md)
- [Analytical two-path decoder: implementation contract](architecture/observable-multipath-model.md)
- [Model outputs and channel families](architecture/output-tasks-and-channels.md)
- [Architecture overview](architecture/overview.md)
- [Sionna RT backend selection](architecture/rt-backend-selection.md)
- [Sionna RT run identity](architecture/rt-run-identity.md)
- [RT scene asset identity](architecture/rt-scene-asset-identity.md)
- [External Sionna source reference](architecture/sionna-reference.md)
- [Grouped scalar-regression uncertainty evaluation](architecture/uncertainty-evaluation.md)

## Reference

Reference data and the digest rebinding record.

- [digest-rebindings.json](reference/digest-rebindings.json)
- [Scene-family registry and candidate sampling](reference/scene-family-registry.md)

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
