---
trigger: model_decision
description: Use when changing CI, verification dispatch, packaging, or release acceptance.
---

# CI verification

Preserve the existing Python 3.11 and 3.12 portable test matrix, the separate Python 3.12 neural/coverage job with CPU PyTorch, and the package build job. The additive `agent-policy` job runs `tools/check agent` without native Sionna or GPU work.

`tools/check` is the local dispatcher. It must propagate required failures, execute argv without a shell, and keep native smoke separate from portable CI.
