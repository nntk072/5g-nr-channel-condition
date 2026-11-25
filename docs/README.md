# Documentation

Start here. This index lists every document under `docs/`. Development records,
evidence and plans are kept outside the published tree.

## Guide

Setup and workflows for users of the framework.

- [Quickstart](guide/quickstart.md)
- [Getting started](guide/getting-started.md)
- [Shared-split power-only baseline](guide/baseline-controls-workflow.md)
- [Sample-domain network waveform to PUSCH receiver](guide/network-pusch-receiver-workflow.md)
- [Pilot sampling workflow](guide/pilot-sampling-workflow.md)
- [Resource budget preflight](guide/resource-budget-workflow.md)
- [Runtime identity inventory](guide/runtime-inventory-workflow.md)
- [Scenario GBSM generation](guide/scenario-gbsm-generation-cli-v1.md)

## Architecture

Concepts and output contracts for users of the framework.

- [Architecture overview](architecture/overview.md)
- [Benchmark partition materialization](architecture/benchmark-partition-materialization.md)
- [Multi-link RT to network PHY support](architecture/network-topology-support.md)
- [Analytical two-path decoder](architecture/observable-multipath-model.md)
- [Model outputs and channel families](architecture/output-tasks-and-channels.md)
- [Sionna RT backend selection](architecture/rt-backend-selection.md)
- [Sionna RT run identity](architecture/rt-run-identity.md)
- [RT scene asset identity](architecture/rt-scene-asset-identity.md)
- [External Sionna source reference](architecture/sionna-reference.md)
- [Grouped scalar-regression uncertainty evaluation](architecture/uncertainty-evaluation.md)

## Reference

- [Command reference](reference/cli.md): console commands and scripts
- [Schema index](reference/schemas.md): every JSON schema and its identity
- [Scene-family registry and candidate sampling](reference/scene-family-registry.md)

## Contributing

- [Contributing, gates and rules](../CONTRIBUTING.md)

## API

- [API reference](api/index.md): generated from the docstrings by Sphinx.

## Index and project pages

- [Documentation home](index.md): the Sphinx entry page for this index.
- [Architecture package map](architecture/README.md): how the packages depend on one another.
- [Module size](architecture/module-size.md): modules above the size target and why they stay whole.
- [Coverage policy](architecture/coverage-policy.md): the only reviewed coverage exclusions.
- [Project README](../README.md), [Contributing](../CONTRIBUTING.md), [Security](../SECURITY.md), [Changelog](../CHANGELOG.md), [Citation](../CITATION.cff)
