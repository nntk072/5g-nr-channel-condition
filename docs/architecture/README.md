# Architecture for users

The framework separates reusable Python code, declarative inputs, research
assets, generated evidence, and tests. The pages below explain how the
pieces fit together and what each output means.

- [Architecture overview](overview.md): the packages, their layers, and the main flows.
- [Output tasks and channel families](output-tasks-and-channels.md): what each model output contains.
- [Observable multipath model](observable-multipath-model.md): the analytical two-path decoder contract.
- [Network topology support](network-topology-support.md): multi-link ray tracing to network PHY.
- [Benchmark partition materialization](benchmark-partition-materialization.md): how train, validation and held-out partitions are built.
- [Uncertainty evaluation](uncertainty-evaluation.md): grouped scalar-regression uncertainty.
- [Sionna RT backend selection](rt-backend-selection.md), [run identity](rt-run-identity.md) and [scene asset identity](rt-scene-asset-identity.md).
- [External Sionna reference](sionna-reference.md): the Sionna version the framework targets.

For setup and workflows, start at the [documentation index](../README.md).
