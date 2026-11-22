# Agent Architecture

The repository keeps engineering orchestration separate from scientific behavior.

```text
approved research program
        |
REQUIREMENT_LEDGER.json + EXECUTION_QUEUE.md
        |
engineering control plane
AGENTS.md -> .agents/rules -> .agents/domain_maps -> roles/task cards
        |
agentops and tools/check
        |
existing scientific data plane
configuration, datasets, scenes, propagation, integrations/sionna,
receivers, models, learning, evaluation, experiments, artifacts, compute
```

`agentops/` owns engineering task policy, state, verification receipts, provider execution, review, and handoff. It does not own scientific manifests, experiment replay, split registries, qualification decisions, or production data.

`loslab/` remains the public CLI and compatibility package. `scripts/` remains wrapper and audit tooling. `integrations/sionna/` remains an optional integration boundary whose native behavior must be qualified separately from portable software checks.

Package direction follows current source contracts. Domain maps are navigation aids and cannot override source, tests, or governed research documents.
