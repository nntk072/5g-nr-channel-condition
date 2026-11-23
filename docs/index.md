# 5G NR Channel Condition

Research code for NR channel-condition learning: standardized and ray-traced
multipath channels, PUSCH receiver evaluation, and the protected evaluation
contracts that govern when a result counts as evidence.

This site is the entry point for developers and reviewers. The repository
keeps the scientific protocols and retained evidence; this site describes how
the code is organized and how to run it.

```{toctree}
:caption: Start here
:maxdepth: 1

guide/quickstart
```

```{toctree}
:caption: Guides
:maxdepth: 1

guide/baseline-controls-workflow
guide/luna-execution-guide-2026-11-11
guide/network-pusch-receiver-workflow
guide/pilot-sampling-workflow
guide/resource-budget-workflow
guide/runtime-inventory-workflow
guide/scenario-gbsm-generation-cli-v1
```

```{toctree}
:caption: Architecture
:maxdepth: 1

architecture/overview
architecture/ci
architecture/naming-exceptions
architecture/module-split-design-notes
architecture/baseline
architecture/benchmark-partition-materialization
architecture/model-input-route-inventory
architecture/network-topology-support
architecture/observable-multipath-model
architecture/output-tasks-and-channels
architecture/refactor-plan
architecture/rt-backend-selection
architecture/rt-run-identity
architecture/rt-scene-asset-identity
architecture/sionna-reference
architecture/uncertainty-evaluation
```

```{toctree}
:caption: Operations
:maxdepth: 1

operations/agent-operations
operations/agent-task-protocol
operations/agent-compute-policy
operations/agent-release-acceptance-2026-11-19
operations/probability-prior-policy
operations/refactor-verification
operations/release-split-policy-v2
operations/result-promotion-policy
operations/upgrade-changelog
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/helper-wiring-matrix
reference/scene-family-registry
reference/target-family-producer-consumer-matrix
```

```{toctree}
:caption: API
:maxdepth: 1

api/index
```
