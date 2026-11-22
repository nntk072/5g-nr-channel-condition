---
trigger: model_decision
description: Use to choose the smallest exact verification gate that matches the changed contract.
---

# Validation routing

Use `tools/check agent` for agent-policy changes, `tools/check fast` for active software paths plus `agentops`, exact task-card `scoped_verification` for a bounded implementation, `tools/check core` for the portable software suite, and `tools/check full` for release parity. Native checks remain explicit and separately authorized.

A skipped, dry-run, blocked, interrupted, or timed-out check never counts as `PASS`. Keep the exact argv, exit code, source SHA, task-card SHA, and receipt path.
