---
trigger: model_decision
description: Use for subprocesses, providers, external tools, secrets, and path handling.
---

# Security

Use argument arrays with `shell=False` for provider and verification subprocesses. Apply finite timeouts and output limits. Whitelist inherited environment variables and redact credentials from receipts and logs.

Reject absolute task-owned paths, parent traversal, symlink escape, duplicate JSON keys, non-finite numbers, unknown schema versions, and stale source bindings. Optional developer tools do not become authority over current source.
