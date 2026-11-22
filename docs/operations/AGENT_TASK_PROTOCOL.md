# Agent Task Protocol

`nr-agent` executes only reviewed cards listed in `config/agentops/default.yaml`. The live scientific queue and requirement ledger remain `docs/research/EXECUTION_QUEUE.md` and `docs/research/REQUIREMENT_LEDGER.json`.

Each execution is bound to the current Git revision and the raw task-card SHA-256. Independent review approvals and verification receipts additionally bind the exact current worktree digest. Engineering status and scientific qualification are separate fields. A successful software gate cannot promote G0–G8, G5, G3, publication, or production D1 status.

The supported flow is `PENDING -> PREFLIGHT -> IMPLEMENTING -> REVIEWING -> VERIFYING -> REPORTING -> DONE`, with a bounded `FIXING -> REVIEWING` loop. `BLOCKED`, `FAILED`, `STALE_SOURCE`, `NEEDS_HUMAN_APPROVAL`, and `DRY_RUN_ONLY` are terminal non-success states. `DONE` requires a successful non-skipped verification receipt for the same task card and source revision.

Read-only commands are `doctor`, `validate-tasks`, `list`, `status`, and `plan`. `run --dry-run` renders policy and creates only ignored operational state. `verify` runs the named engineering gate and emits a source/worktree-bound receipt. Source-mutating provider execution requires a reviewed task card with `permissions.source_write: true`, an explicitly configured provider executable, and any required resource authorization.

`finalize-manual` is the canonical handoff for authorized work already implemented in the current worktree. It requires a strict JSON review file and the exact dirty owned paths. AgentOps validates that file's shape and binding fields; it does not verify that an independent reviewer wrote it (see the release acceptance report, Limitations 1). It does not accept unrelated dirty files as task output. `resume` continues only implemented safe checkpoints and otherwise fails closed while preserving state.

Native classes require `--resource-authorization-file` on `run`, `resume`, or `finalize-manual`. The JSON receipt is revalidated against task ID, source/card digests, current host, measured capacity, selected device, runtime metadata, expiry, and, for `scientific_native`, scientific-gate authority; that class is always refused because no scientific-gate verifier exists. CPU-only task classes do not require a native receipt.

Typical usage:

```bash
uv run --locked nr-agent doctor
uv run --locked nr-agent validate-tasks
uv run --locked nr-agent list
uv run --locked nr-agent plan AGENT.INFRA.SELFTEST
uv run --locked nr-agent run AGENT.INFRA.SELFTEST --dry-run
uv run --locked nr-agent run AGENT.INFRA.SELFTEST
uv run --locked nr-agent status AGENT.INFRA.SELFTEST
uv run --locked nr-agent verify E04.PKG06-003.HOP_EST_DEMAP --gate scoped
uv run --locked nr-agent resume E04.PKG06-003.HOP_EST_DEMAP
uv run --locked nr-agent handoff AGENT.INFRA.SELFTEST
```

For a reviewed manual implementation, the equivalent finalization shape is:

```bash
uv run --locked nr-agent finalize-manual TASK_ID \
  --review-file /path/to/independent-review.json \
  --changed-path owned/source.py \
  --changed-path tests/test_owned_source.py
```

The review JSON contract is exactly `reviewer_id`, `verdict`, `findings`, `task_id`, `source_revision`, `task_card_sha256`, and `worktree_digest`. The four binding fields must identify the exact source state the reviewer inspected. A changed worktree invalidates the approval and requires a fresh independent review. On the provider review path, the reviewer must be a configured `reviewer_provider` that does not resolve to the implementer's executable, so the implementer cannot approve its own change. On the `finalize-manual` path anyone with write access to the repository can produce the review file, including the implementing agent or card owner; AgentOps validates its shape, binding fields, and reviewer identity string but not its authorship, so that path is an attestation, not an enforced independent review (see the release acceptance report, Limitations). A failed or interrupted verification leaves the task in durable state with its receipts retained for diagnosis and safe recovery.

No command commits, pushes, merges, deletes research evidence, cancels jobs, or launches native scientific qualification by default.
