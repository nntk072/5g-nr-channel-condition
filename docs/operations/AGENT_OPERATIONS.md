# Agent Operations

## Baseline

The agent architecture was introduced from the clean target revision `2e6780dc7f006526a811fc48156358ff39f361ba`. The exact source and governing-document hashes are recorded in `config/agentops/baseline-lock.json`.

Halti now has user-scoped `uv 0.13.9` and managed Python 3.12.16 on `x86_64`. A host `python3.11` executable is unavailable. Slurm is unavailable, and the read-only NVIDIA process inventory reported no active compute process at the baseline refresh. The previous `.venv` was preserved under the user's cache before recreation.

The restored toolchain exposes existing baseline failures that predate agent infrastructure. Ruff lint and package build pass. Ruff format reports 83 existing files that would be reformatted. Mypy reports 269 existing errors in 54 files. Portable core pytest collection without the neural dependency stops because a test imports `torch`. Agent work must not hide or mass-edit these unrelated baseline failures.

## Operating rule

Use the active execution queue and requirement ledger to choose scientific software work. Agent infrastructure may coordinate engineering tasks but may not alter scientific qualification state by inference.

## Runtime evidence

`config/agentops/default.yaml` is a curated engineering view over the existing research authorities. Task cards remain source-bound and reference existing requirement IDs. Durable state lives under ignored `outputs/agentops/tasks/<task_id>/`. Provider and verification receipts, handoffs, and `telemetry.jsonl` are operational evidence only.

Every verification receipt binds the Git source revision, raw task-card SHA-256, and a digest of the current worktree. A receipt becomes stale when source or worktree content changes. Failed verification receipts are retained alongside later successful receipts instead of being overwritten.

`nr-agent list` reconciles a card with durable state. A reviewed card declared `ready` is shown as historically `done` after a matching operational `DONE`, while `verification_evidence_status` separately reports whether the retained final receipt still matches the current source and worktree. A state file from an older version of a task card is surfaced as `STALE_TASK_CARD`. Historical completion is preserved, and stale receipts are never reused as current verification.

## Manual implementation finalization

Manual source changes can enter the same state, review, verification, and handoff path with `nr-agent finalize-manual`. The command accepts only dirty paths owned by the reviewed task card, requires a structured `APPROVE` review file whose binding fields match the current worktree (its authorship is not verified; the release acceptance report describes the attestation limit), runs every mandatory card gate, and records engineering `DONE` only after successful non-skipped verification. The scientific disposition is copied from the task card and is not inferred from test success.

The E04 `PKG06-003` hop estimator/demapper pilot exercised this path on Halti. Its engineering handoff is `DONE`, while its scientific disposition remains `NOT_QUALIFIED` and the broader G5/G3 prerequisites remain open under the research authorities.

## Resume and interruption recovery

Task state is atomic and guarded by an exclusive writer lock. Resume verifies the same task-card digest, source revision, workspace, and resource authority. The implemented automatic resume checkpoints are deliberately narrow. A retained manual task at `VERIFYING` can rerun its exact card gates, and a retained manual task at `REPORTING` can finish from validated receipts. Earlier implementation phases fail closed for operator review instead of replaying source mutation.

Native-resource tasks require the same explicit resource authorization to be revalidated on resume. The authorization is checked against the selected host, measured RAM and disk capacity, Sionna 2.2 runtime metadata, selected CUDA device, expiry, and the task/source/card identity. Resource authorization permits a bounded engineering action only and never qualifies a scientific result.

## Diagnostics and telemetry

`nr-agent doctor` and `tools/agent/doctor` are read-only. They inspect Python, `uv`, AgentOps imports, provider availability, ignored/writable operational roots, and installed Sionna distributions without importing native GPU libraries or allocating compute.

Task telemetry records source/card identity, role, phase, provider result metadata, verification commands and hashes, and provider token/cost fields only when the provider actually reports them. Credential-like values are redacted by pattern matching, which is not exhaustive (release acceptance report, Limitation 8). Telemetry never supplies scientific qualification evidence.
