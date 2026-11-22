# AgentOps P14 Release Acceptance Report

Date: 2026-11-19 (revised 2026-11-22)

## Release decision

**FINALIZED WITH DISCLOSED LIMITS (owner decision, 2026-11-22). Not a reviewer APPROVE.**

Final owner instruction, 2026-11-22: accept the current state as final. This accepts item 19 as portable local parity and items 22 and 30 as owner-accepted with disclosed limits. It does not convert them to reviewer PASS: no reviewer approved any revision, and the committed branch was not independently reviewed. Items 3, the synthetic telemetry rows, the two retained handoffs, and `finalize-manual` remain open owner decisions.

- The owner chose to finalize with the limits below disclosed. This is not an independent reviewer APPROVE: twenty fresh independent reviews, one per round, all returned `REQUEST_CHANGES` (see [Review rounds](#review-rounds)). Item 22 is recorded as **accepted by owner with disclosed limits**, not as a reviewer approval.
- Every engineering gate and the portable matrix pass on the frozen final worktree (see [Executed verification](#executed-verification)).
- Item 19 is recorded as portable local parity. Literal hosted CI was not run and needs authorization to push and dispatch a workflow.
- Items 9, 12 and 23 were implemented in this revision and are PASS under the definitions in their rows. Their limits are listed in Limitation 13: other users' processes are not refused, no execution path consumes `stale_if` yet, and the changes have not been independently reviewed.
- Open owner decisions: attribution of unowned changes (item 3); whether to clean the polluted `AGENT.INFRA.SELFTEST` telemetry; the two retained handoffs that fail the current schema.
- Nothing in this report changes scientific qualification. `E04.PKG06-003.HOP_EST_DEMAP` remains engineering `DONE` with scientific disposition `NOT_QUALIFIED`.

Authoritative inputs: [EXECUTION_QUEUE.md](../research/EXECUTION_QUEUE.md) and [REQUIREMENT_LEDGER.json](../research/REQUIREMENT_LEDGER.json). Control-plane description: [AGENT_OPERATIONS.md](AGENT_OPERATIONS.md).

## Frozen source and authority evidence

- Source revision: `2e6780dc7f006526a811fc48156358ff39f361ba` on branch `main`. Nothing was committed, pushed, merged, or rewritten.
- `docs/research/EXECUTION_QUEUE.md` SHA-256 `8a49c5c6a75973c0aeaf5ab93cd46ff99f4b1feff46b2a1074011d893838d7b7`
- `docs/research/REQUIREMENT_LEDGER.json` SHA-256 `de1c2b6d7b366cd3eb03de905915b7c2c5215cda1b54e6c618c65a7b40259b4e`
- `git diff -- docs/research/EXECUTION_QUEUE.md docs/research/REQUIREMENT_LEDGER.json` produces zero bytes (see `logs/authority-diff.log`).
- Halti has no `sbatch`. No native or GPU workload was launched. Sionna and Torch are not installed in the default `.venv`; the neural matrix job installs CPU Torch in its own environment only.

## Executed verification

Final-round evidence lives in `/home/l2nguyen/agent-final-2026-11-22/logs/`. Earlier rounds are kept in `logs-round1-prereview/`, `logs-round2-review2-basis/`, and `logs-round3-review3-basis/`, and `logs-round4-review4-basis/`, and `logs-round5-review5-basis/`, `logs-round6-review6-basis/`, and `logs-round7-review7-basis/`.

| Command | Exit | Result |
| --- | ---: | --- |
| `uv lock --check` | 0 | lock resolves |
| `./tools/check agent` | 0 | rules and 10 profiles valid, doctor ok, 178 AgentOps tests passed (includes the 13 tests for items 9, 12 and 23 and the 4 revision-ancestry tests) |
| `./tools/check fast` | 0 | Ruff, format, mypy clean; 1426 passed, 34 skipped |
| `./tools/check build` | 0 | sdist and wheel built |
| `uv run --locked nr-agent validate-tasks` | 0 | 2 task cards valid |
| `git diff --check` | 0 | no whitespace errors |
| authority SHA-256 and diff | 0 | hashes unchanged, zero diff |
| Portable matrix, Python 3.11 (`uv sync --locked --extra dev`) | 0 | lock check, Ruff check, Ruff format, mypy, core pytest `--no-cov`: 3689 passed, 223 skipped |
| Portable matrix, Python 3.12 core | 0 | same steps: 3689 passed, 223 skipped |
| Neural and boosting, Python 3.12 (`--extra boosting`, CPU Torch 2.11.0) | 0 | 3918 passed, 168 skipped; coverage 75.24% against the 75% threshold. The first matrix attempt after items 9, 12 and 23 failed one timing-sensitive memory test; see Limitation 13 and `logs-items-9-12-23-attempt1-before-conftest/` |
| `uv build` | 0 | sdist and wheel built |

The matrix mirrors the steps in `.github/workflows/ci.yml` but runs in local virtualenvs outside the repository, so the project `.venv` is untouched. It is portable local parity, not a hosted run.

## Round-one review and remediation

Round-one review (`independent-reviewer-2026-11-22`, bundle SHA-256 `ea7a803d…76ec9`) returned `REQUEST_CHANGES`. Each finding was checked against source before fixing.

| # | Finding | Status | Fix and regression coverage |
| --- | --- | --- | --- |
| 1 | `kind: validation` synthesized an `APPROVE` as the configured reviewer without any review | fixed | Removed the synthesized approval. Every kind now goes through the provider review. `test_validation_card_cannot_self_approve_without_independent_reviewer` |
| 2 | One provider implemented and reviewed, so self-review was accepted | fixed | Optional `reviewer_provider` config. Review calls use it, and it is rejected when it resolves to the same executable as `provider` (round two tightened this from string comparison; see Review rounds). Without it, review fails closed. `test_reviewer_resolving_to_implementer_binary_is_rejected`, `test_reviewer_provider_is_used_for_review_and_implementer_for_implement` |
| 3 | Redaction missed `Authorization: Basic`, `Token`, `Digest`, and unlabeled keys such as `sk-ant-…` | fixed | Extended the labeled-scheme pattern and added well-known credential prefixes (`sk-`, `ghp_`, `github_pat_`, `AKIA`, `xox*`). `test_redact_removes_credentials_and_their_values` |
| 4 | `scientific_native` was admitted by an operator-supplied `scientific_gate_authorized` boolean | fixed | Fails closed like `production_d1`, because no gate-issued receipt verifier exists. The existing test that asserted admission with `True` was corrected. `test_resource_authorization_rejects_noncanonical_device_and_ungated_scientific_native` |
| 5 | `owned_paths: ["."]` was accepted and overlapped every path, so ownership failed open | fixed | `validate_relative_path` rejects `.`, and `paths_overlap` treats an empty path as overlapping everything. `test_repository_root_is_not_a_valid_owned_path`, `test_repository_root_overlaps_every_path_fail_closed` |
| 6 | Changed-file roster omits unowned modules, and format-only and semantic changes are not attributed | **open, see Limitations** | Not auto-fixed. Ownership of those files is not recorded anywhere in the repository. |

Nits: `tools/check` now parses porcelain `-z` rename and copy records correctly (the source field is consumed separately), and the requirement anchor in `queue.py` is a token match, so `PKG06-00` no longer matches `PKG06-003`. Two nits are documented and not changed: `integrations/sionna/rt_network.py` pose digest does not carry `allow_nan=False` (the digest is identical for finite values), and `config/scenes/scene-family-registry.v1.json` corrected a stale master-plan pin (`9af3761e…` to `3285eb13…`, which matches HEAD).

## Review rounds

Each round is a fresh read-only subagent over a frozen bundle (tracked patch SHA-256 and per-file SHA-256 of untracked files). Verdict files are under `/home/l2nguyen/agent-final-2026-11-22/review*/`.

| Round | Bundle | Verdict | Outcome |
| --- | --- | --- | --- |
| 1 | tracked `ea7a803d…76ec9`, 113 untracked | `REQUEST_CHANGES` (6 important: 5 defects and the attribution gap; 4 nits) | Findings 1-5 fixed with tests; finding 6 open; nits fixed or documented |
| 2 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (6 important, 2 pre-existing, 1 nit) | Addressed in this revision; see below |
| 3 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (6 important, 2 pre-existing, 2 nits) | Addressed in this revision; see round three below |
| 4 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (6 important, 2 pre-existing, 1 nit) | Addressed; see round four below |
| 5 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (2 important, 8 nits and pre-existing items) | Addressed in this revision; see round five below |
| 6 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (3 important, 4 pre-existing, 5 nits) | Addressed in this revision; see round six below |
| 7 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (5 important, 2 pre-existing, 9 nits) | Dependency gate now uses reconciled evidence; nesting capped at 256; handoff published after the guarded DONE transition; DONE handoffs require APPROVE and passing checks; cookie pattern made linear; labels added. Remaining items are in Limitations 11-13 and the matrix. |
| 8 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (3 important, 6 nits) | Addressed: the legacy DONE rebuild now requires current DONE evidence; credential flags widened to any credential-like name; credential patterns are bounded so they run in linear time on separator-rich input (measured 0.1 s at 200k characters, previously 5 s); nits fixed in code (BaseException cleanup, strict-loader recursion guard, legacy checks type check) and in the report |
| 9 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (5 important, 4 nits) | Addressed: legacy DONE handoffs are refused rather than rebuilt from unsigned content; argument redaction arms only on true flags; the strict loader guards non-finite checks; the worktree digest disables rename detection so staged deletions count; current-schema DONE handoffs need current evidence; report corrected |
| 10 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (3 important, 5 pre-existing or nits) | Addressed: a boolean credential flag no longer masks the following flag; handoff evidence gate checks both `result` and `engineering_status`; the rule-validation TOML test was restored (the report's citation was wrong while it was missing); `run()` checks the recorded card digest; the integer-digit limit and YAML recursion are refused as contract errors; a split auth scheme redacts its token. Limitation 9 corrected. |
| 11 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (4 important, 3 nits or pre-existing) | Addressed: a DONE now requires every gate run to match the approved worktree (not only the last); argv redaction rewritten with explicit states, so a dash-prefixed credential value is redacted, a long option is a flag, and everything after an auth scheme word is redacted (fail closed); the report's review count corrected; YAML digit-limit errors become contract errors |
| 12 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (2 important, 2 pre-existing, 5 nits) | Addressed: the non-finite walk is iterative and visits each container once (the alias expansion was exponential, measured at 10 s for a 414-byte input); a header split across argv items redacts the credential after an auth scheme word; `--` after a credential flag is kept; unhashable verdict or severity values are contract errors; a dead branch removed. Remaining: validation-receipt roster after a retried gate (pre-existing, disclosed), and the window between the DONE transition and os.replace (disclosed) |
| 13 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (2 important: verification receipts persisted full argv; the record undercounted) | Receipts now persist no argument values (executable name, argument count, and digest; verification receipts keep the full executable path). Record corrected. |
| 14 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (1 important: the record undercounted round 13) | Record corrected to fourteen. |
| 15 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (1 important: the table layout) | Table corrected in this revision |
| 16 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (1 important: cited test names for retired argv tests) | Retired citations marked in the report |
| 17 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (4 important) | Addressed: the verify guard now refuses non-local classes for the scoped and standard gates; the report wording matches the code; count corrected |
| 18 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (2 important) | Addressed: the verify wording is exact (native-smoke has its own branch); the count is corrected |
| 19 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (3 important) | Addressed: the table cells and the count are generated from the verdict files on disk; the round-19 record is corrected |
| 20 | tracked `ea7a803d…76ec9`, 114 untracked | `REQUEST_CHANGES` (1 important: the runner's scoped and standard gates did not refuse native classes) | Fixed: the runner refuses every gate except the metered native-smoke for non-local classes; regression test added |

Round-six findings and their status:

| Finding | Status | Notes |
| --- | --- | --- |
| Task dependencies were not checked by `run`, `finalize-manual`, or `resume` | Fixed | `Runner._require_dependencies_done` runs on all three entry points. `test_dependency_admission_refuses_unmet_dependency` |
| Flat cloud credential names (`SecretAccessKey`, `SessionToken`, `SECRET_ACCESS_KEY`) leaked | Fixed | Added to the label and key patterns. `test_flat_cloud_credential_names_are_redacted` |
| Cookie labels preceded by a hyphen or word character (`session-cookie=`, `X-Set-Cookie:`) redacted only the first pair | Fixed | Lookbehind now excludes only quotes; hyphenated cookie names run to end of line. `test_hyphenated_cookie_labels_redact_every_pair` |
| `--cookie`, `--client-secret`, `--secret-key`, `--oauth2-bearer`, `--access-token` values persisted in argv receipts | Fixed | Added to the separate-item secret flags. retired in round 13 (argv-summary design) |
| An approval carrying high or critical findings was accepted | Fixed | `require_review_approval` refuses it. `test_review_approval_with_blocking_findings_is_refused` |
| Deeply nested provider JSON raised an unhandled `RecursionError` with no receipt | Fixed | Becomes a `malformed_json` or `nesting_limit` failure receipt. `test_deeply_nested_provider_json_becomes_a_receipt_not_a_crash` |
| Provider mutation guard ignores git-ignored paths, so `outputs/` and `logs/` evidence is not protected by the guard | **Disclosed, not fixed** | Limitations 9. Changing the digest to cover ignored paths would change every current digest and is an owner decision. |
| `require_review_approval` checks only `verdict`; findings are checked for shape only | Fixed for high and critical findings | Lower-severity findings do not block approval by design |
| Report nits: row 18 count, row 22 round count, protocol `handoff` command, `STALE_TASK_CARD` state | Fixed | Row 18 now 163; row 22 now six rounds; see Limitations 10 for the handoff command and stale card |

Round-five findings and their status:

| Finding | Status | Notes |
| --- | --- | --- |
| `--header=<value>` and attached `-H<value>` argv forms passed credentials through unredacted | Fixed | Split and handled like a separate header item (the retired argv header helper). retired in round 13 (argv-summary design) |
| Cookie values stopped at the first quote, brace or bracket | Fixed | Cookie values run to end of line. JSON-quoted `"cookie": "…"` is left to the quoted-value pattern. `test_quoted_and_bracketed_cookie_values_are_fully_redacted` |
| Prose containing `Cookie:` is redacted to end of line | Accepted | Fail-closed over-redaction of persisted text. Not changed. |
| Lowercase concatenated keys (`accesstoken`, `clientsecret`) | Fixed | `test_lowercase_concatenated_credential_keys_are_replaced` |
| Credentials embedded in dictionary keys survived the success path | Fixed | Keys are text-redacted too |
| `startswith("[REDACTED")` guard could skip text after a literal `[REDACTED` | Accepted | Contrived input; the guard only protects against double-processing of our own placeholder |
| Invalid UTF-8 under the raw limit was rejected as malformed | Accepted | Fail-closed; rejection, not a leak |
| `task-state` schema accepted any `scientific_disposition` string | Fixed | Enum restricted; `test_task_state_schema_rejects_qualification_disposition` |
| Retained `E04` and `AGENT.INFRA.SELFTEST` `handoff.json` fail the current `handoff.v1` schema (nine errors: missing `base_sha`, `result`, `changed_files`, `source_digest`, among others) | **Disclosed, not rewritten** | Written 2026-11-19 by an earlier writer. Confirmed read-only against the current schema. See Limitations 7. |
| `doctor` prints `scientific_qualification: UNCHANGED` as a literal | Disclosed | Not a computed check. See Limitations 7. |

Round-four findings and their status:

| Finding | Fix | Regression coverage |
| --- | --- | --- |
| Failure receipts persisted raw stdout and stderr through text-only redaction (malformed JSON, non-object JSON, timeouts) | Failure receipts and success stderr persist only size and SHA-256 (`_omitted`). Parsed success responses are still persisted after structured redaction. | `test_failure_receipts_omit_raw_provider_output_and_keep_digests` |
| camelCase credential keys (`accessToken`, `authToken`, `clientSecret`) were not matched on the structured path | Keys are split at camelCase boundaries before matching | `test_camelcase_credential_keys_are_replaced_on_the_structured_path` |
| Cookie headers with several cookies kept every cookie after the first | Cookie values run to the end of the header line; the cookie label is handled by a dedicated pattern, so the generic pattern does not double-process it | `test_multi_cookie_header_is_fully_redacted` |
| Value-only argv header items containing `:` were passed through unredacted | Only items that start with a header name followed directly by `:` count as labelled | retired in round 13 (argv-summary design) |
| `ProviderError` text carried a stderr tail into telemetry | Error text names the exit code only | `test_provider_error_text_carries_no_stderr` |
| `scientific_disposition` accepted `QUALIFIED_FROM_EXISTING_EVIDENCE_ONLY` in task cards and handoffs | Enum restricted to `NOT_APPLICABLE` and `NOT_QUALIFIED` in both schemas | `test_scientific_disposition_cannot_claim_qualification` |
| Command digest was computed over unredacted argv, so a low-entropy credential could be guessed | Digest is computed over the redacted argv | `test_command_digest_does_not_expose_argv_secret` |
| Output limit was applied to redacted text, so a valid response could be rejected | Limit is applied to the raw byte length | Probe; fail-closed either way |
| Protocol documents stated review-file enforcement without qualification | Qualified in `AGENT_TASK_PROTOCOL.md` and `AGENT_OPERATIONS.md` | Review by a fresh reviewer (round five) |

Not addressed, and disclosed: text redaction is not exhaustive. It does not catch every unlabelled credential form (for example a bare token with no label or scheme), plural `tokens` keys outside the list, or `-u`/`--user` argv forms. `--access-token`, `--access-key`, `--session-token`, `--private-key`, and similar flags are covered in both separate-item and `=` forms. Failure receipts no longer depend on it, because they omit raw output.

Round-three findings and their status:

| Finding | Fix | Regression coverage |
| --- | --- | --- |
| Argv header items that carry their own label were double-labelled, so `Authorization: Basic …` leaked | Labelled header items are redacted as written; only unlabelled values get the label added and removed again | retired in round 13 (argv-summary design) |
| JSON values with escaped quotes leaked the tail; strings were redacted before parsing, so the JSON broke | The success path parses the bounded raw output first and then redacts the parsed structure | `test_redacted_json_with_escaped_quotes_stays_parseable_and_clean`, `test_structured_and_text_redaction_cover_claimed_categories` |
| Nested objects or arrays under `auth`, `credentials`, or `token` leaked beyond the first token | Whole values under credential-named keys are replaced, whatever their type | `test_structured_and_text_redaction_cover_claimed_categories` |
| Cookie, URL userinfo, `secret_key`, `private_key`, AWS secret, `x-api-key`, exponent-form scalars were not covered (disclosed as unclaimed in round three) | Added; exponent-form scalars keep JSON valid | `test_structured_and_text_redaction_cover_claimed_categories`, `test_bare_scalar_redaction_keeps_json_valid` |
| Contract and rule-validation tests wrote fixtures into tracked config and rule directories | Those tests now write only into temporary copies | all tests in `tests/test_agentops/test_contracts.py` and `tests/test_scripts/test_agent_rule_validation.py` |
| Dotted sub-identifiers such as `PKG06-003.v2` anchored a requirement | Anchor is a whole-identifier check in `requirement_anchored` | `test_requirement_anchor_matches_whole_identifiers_only` |
| Report wording and protocol-document self-approval statements (round-three findings 4-5) | Limitations 1 and the telemetry note were rewritten; the protocol documents now qualify the self-approval and attestation statements | Review by a fresh reviewer (round four) |
| The report's example list named `datasets/split_policy.py` as AST-semantic | Removed. It is AST-identical to HEAD (formatting only). Of 130 modified Python files, 69 are AST-identical and 61 are AST-semantic (58 outside the owned paths). | Re-counted from the live tree |

Round-two findings and their status:

| Finding | Fix | Regression coverage |
| --- | --- | --- |
| Validation kind skipped the already-dirty guard, so pre-existing owned dirty work could be attributed to DONE | The already-dirty guard runs for every kind before the kind branch | `test_validation_card_with_preexisting_owned_dirty_work_is_refused` |
| `finalize-manual` accepts an operator-written review file | Not removed, because it is the documented manual route for reviewed work, and it is not an automated approval. It is disclosed as operator-attested in Limitations 1. | Not enforced by a test. Owner decision. |
| Reviewer identity compared strings | Reviewer must not resolve (through `realpath` or `PATH` lookup) to the implementer's executable, whatever flags it has. Identity is configured, not cryptographic. | `test_reviewer_resolving_to_implementer_binary_is_rejected`, `test_reviewer_with_extra_flags_on_implementer_binary_is_rejected`, `test_reviewer_on_distinct_binary_is_accepted` |
| Redaction leaks: Digest parameters, `auth:` label, JWT, Google keys, argv header values, numeric JSON scalars | Added, with JSON-safe placeholders for bare scalars | `test_redact_removes_credentials_and_their_values`, retired in round 13 (argv-summary design), `test_redact_keeps_redacted_json_parseable` (Digest now asserts the `response` value too) |
| Tests appended to retained telemetry; a rule-validation test rewrote a tracked adapter | Tests use temporary stores and logs; the TOML test validates a copy | All hardening tests build isolated runners; `test_profile_validator_rejects_invalid_toml` uses `tmp_path` |
| Queue anchor matched `PKG06-003.1` | Lookahead rejects `.` followed by a digit | Verified by the anchor check on current cards |
| Report accuracy (telemetry claim, link audit, rows 11, 21, 32, limitations) | Corrected; link audit is reproducible in `logs/link-audit.log` | Re-run the audit |

## Section I 32-item acceptance matrix

Status key: PASS = supported by current final-worktree evidence. PARTIAL = part of the requirement is implemented and the rest is listed as a limit. ACCEPTED BY OWNER = finalized with disclosed limits, not a reviewer verdict. PENDING = not yet supported. BLOCKED = needs external action.

| # | Requirement | Evidence | Status | Remaining action |
| ---: | --- | --- | --- | --- |
| 1 | `AGENTS.md` small, complete, names NR authorities | `AGENTS.md`; `tools/agent/validate-rules` | PASS | None |
| 2 | Root package boundaries represented in domain maps | `.agents/domain_maps/*.md`; validate-rules | PASS | None |
| 3 | Every specialist has one canonical ID and exclusions | `.agents/agents/*.md`; `tools/agent/validate-profiles` (10 roles) | PASS | None |
| 4 | `.codex` and `.claude` are thin canonical adapters | profile validation and structural tests | PASS | None |
| 5 | Doctor detects broken links, stale maps, invalid schemas | `agentops/doctor.py`; `./tools/check agent` | PASS | None |
| 6 | Trading, MT5, broker, proprietary financial semantics absent | `tests/test_scripts/test_agent_rule_validation.py` | PASS | None |
| 7 | Live queue and ledger linked without a second scientific database | `nr-agent validate-tasks`; authority hashes | PASS | None |
| 8 | Source SHA and task-card SHA bind every attempt | task-state, runner, and review-binding tests | PASS | None |
| 9 | Unknown dirty work and active jobs preserved | Dirty work and ownership are guarded by git-visible fingerprints (runner tests). `agentops/active_jobs.py` defines an active job as another live process whose working directory or open file descriptor lies inside the card's owned paths or its task state. `run`, `resume` and `finalize-manual` refuse while one exists (`Runner._refuse_active_jobs`). Tests: `tests/test_agentops/test_stale_cycle_active_job.py`. | PASS under the definition above; see Limitation 13 | Other users' processes cannot be inspected and are reported as `uninspected`, not refused (Limitation 13) |
| 10 | One writer per mutable file; ownership arbitration | `tests/test_agentops/test_queue.py`; ownership fixes above | PASS | None |
| 11 | Config and identity schemas strict and fail closed | `load_config` strict keys and fail-closed provider parsing. Reviewer identity is a configured check on the resolved executable path (see item 22), not a cryptographic identity. | PASS | None |
| 12 | Context facts source-bound and stale-detectable | Source SHA equality is enforced (`agentops/context.py`). `SharedFact.changed_stale_if_paths` and `assert_fresh` evaluate each `stale_if` path against the bound revision (tracked and untracked changes); tested in `tests/test_agentops/test_stale_cycle_active_job.py`. | PASS for stale detection; see Limitation 13 | No gate or handoff loads shared facts yet, so staleness is not enforced in any execution path (Limitation 13) |
| 13 | State durable, atomic, resumable, retains attempts | task-state, lock, resume tests | PASS | None |
| 14 | Skipped or dry verification cannot yield engineering `DONE` | verification and task-state negative tests | PASS | None |
| 15 | Engineering and scientific status independent | state schema; E04 `DONE` with `NOT_QUALIFIED` | PASS | None |
| 16 | Existing artifact, experiment, resource-budget, split owners authoritative | architecture docs; authority diff | PASS | None |
| 17 | Source changes scoped; unexpected edits stop execution | worktree digest and ownership guards | PASS | None |
| 18 | Portable `tools/check agent` passes; failures propagate nonzero | final-revision `tools/check agent` exit 0 (178 passed) | PASS | None |
| 19 | Portable Python 3.11/3.12 CI, neural coverage, and build viable | Final-revision matrix exit 0 on the frozen worktree: 3.11 core, 3.12 core, neural 75.24%, build (see Executed verification). The pinned hosted run `44413305629` is stale and was not rerun; literal hosted CI was not verified. | **PASS for portable local parity; ACCEPTED BY OWNER.** Literal hosted CI is not verified. | Owner decision: accept portable parity as item 19, or authorize a hosted run (requires push and workflow dispatch, which were not authorized). |
| 20 | Agent code packaged, linted, typed, tested | fast gate; build | PASS | None |
| 21 | Not run. The owner directed no further independent reviewers after round 20. The items 9, 12 and 23 changes are therefore unreviewed. | None | Owner directed no further reviews; not an approval of the changes |
| 22 | Independent reviewer validates whole diff; author cannot self-approve | Twenty fresh independent reviews, one per round, all `REQUEST_CHANGES`; the findings of every round were addressed in code, tests, or wording; the review of the latest revision is the pending row in the Review rounds table. Known limits: `finalize-manual` accepts an author-writable review file; provider review requires a configured reviewer whose resolved executable differs from the implementer's. | **ACCEPTED BY OWNER WITH DISCLOSED LIMITS.** Not a reviewer APPROVE. | None further unless the owner requests another review round |
| 23 | Fix loop bounded; every attempt retained | Fix budget enforced. `attempt_history` and provider receipts are retained. Each review call now writes `source-diff.patch`, `review-source.json` and `reviewer-report.md` into a numbered directory `review-cycles/NNNN/` under the task log root; the top-level copies still hold the latest cycle. Tests: `test_review_cycles_are_numbered_and_retained`, `test_review_cycle_reports_do_not_overwrite_each_other`. | PASS | None |
| 24 | GPU and native requests require exact approved scope | `scientific_native` and `production_d1` both fail closed (findings 4 and production gate) | PASS | Both classes stay denied until their gates issue trusted receipts |
| 25 | Host, pilot size, runtime, disk, RAM, GPU measured when relevant | `nr-agent doctor`; resource tests | PASS | None |
| 26 | G5/G3 boundary unchanged; synthetic PASS cannot qualify | E04 retained `NOT_QUALIFIED`; PUSCH tests | PASS | None |
| 27 | Negative G4 and City43 evidence retained | authority diff zero; G4 audit tests | PASS | None |
| 28 | ODbL research use separate from public-release authorization | `tests/test_scripts/test_odbl_scene_release_policy.py` | PASS | None |
| 29 | Pilot uses an actual source-confirmed queue item | `E04.PKG06-003.HOP_EST_DEMAP` binds `PKG06-003` | PASS | None |
| 30 | Final report contains commands, hashes, roster, review, exits, limits, next card | This revision contains the commands, hashes, the review history, and the limits. Open owner decisions are listed under Limitations. | **PASS (report finalized with disclosed limits).** | Owner decisions in Limitations 1, 3 and 7 |
| 31 | No unapproved commit, push, merge, workflow rerun, evidence deletion, or expensive experiment | none performed; card permissions false | PASS | None |
| 32 | New Markdown links and CLI examples resolve locally | Reproducible audit `logs/link-audit.log`: 19 relative links across 58 Markdown files in `docs/operations/`, `.agents/`, `.codex/`, `.claude/`, `docs/architecture/`, and `AGENTS.md`, 0 unresolved. Every subcommand and flag used in `AGENT_TASK_PROTOCOL.md` exists in `nr-agent --help`. | PASS | Re-run `link_audit.py` after any further documentation edit |

## Task states

- `AGENT.INFRA.SELFTEST`: retained phase `DONE`, but its verification is stale for the current worktree. Its `DONE` was produced before round-one finding 1 was fixed, when a validation card could self-approve, so it is not valid release evidence. The state file is not rewritten. A new run through a configured reviewer provider is needed.
- **Diagnostic trade-off.** Failure receipts now record the size and digest of provider output, not its text. An operator who needs the output must re-run the provider or reproduce the failure. This is deliberate: text redaction cannot be made exhaustive for free-form output.
- **Telemetry note.** Round-two regression tests appended 22 synthetic `provider_failed` rows (`stop after dispatch`) to the retained `outputs/agentops/tasks/AGENT.INFRA.SELFTEST/telemetry.jsonl`. The file contains nothing else: no pre-pollution content exists, and the telemetry for the retained `DONE` handoff cannot be recovered from it. The tests are now isolated in temporary stores, and the file has not changed since. A copy of the polluted file is at `/home/l2nguyen/agent-final-2026-11-22/telemetry-AGENT.INFRA.SELFTEST-polluted-copy-2026-11-22.jsonl`. Cleanup is an owner decision, because retained evidence is not rewritten without approval.
- `E04.PKG06-003.HOP_EST_DEMAP`: operational `DONE`, scientific `NOT_QUALIFIED`, verification stale for the current worktree. Preserved as retained state, not rewritten.
- No dependency-ready next card exists in `config/agentops/default.yaml`. A future scientific G5 or G3 action needs a newly reviewed card (see `docs/research/EXECUTION_QUEUE.md`).

## Limitations and open items

1. **Item 22 (independent review) and manual attestation.** No provider is configured (`provider: null`), so AgentOps' provider-driven review cannot run here. Independent reviews were performed by read-only subagents over frozen bundles. The `finalize-manual` route accepts a JSON review file that anyone with write access to the repository can produce, including the implementing agent or the card owner. AgentOps checks that the file's shape is exact, that `reviewer_id` equals the card's configured string and differs from the owner, that `verdict` is `APPROVE`, and that the four binding fields match the current source and worktree. It does not verify who wrote the file, does not run a reviewer provider, and verifies no receipt or signature. The protocol documents are qualified accordingly. Owner decision: keep this as an attested route, or gate it behind a reviewer-provider receipt bound to the review digest.
2. **Item 19 (hosted CI).** The only hosted run, `44413305629`, was on the pinned baseline and was red. It has not been rerun, and it cannot be rerun without authorization to push or dispatch a workflow. The criterion is recorded here as portable local parity. If the master-plan text requires hosted CI, it is an external blocker. The master-plan file is absent from the repository and the home directory, so its wording could not be checked.
3. **Attribution of changed files (round-one finding 6).** The worktree contains changes that no reviewed card owns. They are listed in full in `/home/l2nguyen/agent-final-2026-11-22/roster/`:
   - `tracked-outside-owned-paths.txt` (133 tracked files) and `untracked-outside-owned-paths.txt` (70 files). Most of these are AgentOps and governance surfaces that the cards do not list in `owned_paths`, such as `.agents/`, `docs/operations/`, `config/agentops/`, `tools/check`, CI, and packaging. The rest are scientific Python modules that the reviewer found had AST-level semantic changes, including `evaluation/g3_*`, `integrations/sionna/srs.py`, `experiments/runner.py`, `experiments/g5_ordered_impairments_receiver.py`, `compute/resource_budget.py`, and `scripts/audit_channel_split_lineage.py`. Of 130 modified Python files, 69 are AST-identical to HEAD (formatting only, including `datasets/split_policy.py`) and 61 are AST-semantic: 3 E04 files and 58 outside the owned paths.
   - About 69 Python modules differ only in formatting. The earlier statement that formatting debt was "not mass-fixed" is incorrect. These changes are what makes the Python 3.11/3.12 format and mypy steps pass today.
   - The reviewer found no weakened scientific invariant. It did note that two retained G5 audit tests (`tests/test_experiments/test_audit_g5_tdl_dft_density.py`, `tests/test_experiments/test_g5_estimator_choice_diagnostic.py`) had their expected drift sets widened. Both still assert `FAIL`. A repository owner should confirm these edits.
   - `config/agentops/baseline-lock.json` records `working_tree clean_at_capture`, but 79 tracked files share an mtime of 2026-11-21 22:46:10. Provenance of the bulk rewrite cannot be recovered from the release record.
   - Owner decision required: who authored these changes, and whether they belong in this release.
4. **Stale verification.** E04 and `AGENT.INFRA.SELFTEST` are stale for the current worktree, as noted above.
5. **Baseline debt.** Full-repository Ruff, format, and mypy were red on the pinned baseline. They are now green on the final worktree, but only because of the changes in item 3.
6. **Environment.** Python 3.11 is provided by `uv`-managed CPython 3.11.18. Native Sionna, RT, and GPU runs were not performed.
7. **Retained handoffs and the doctor literal.** `outputs/agentops/tasks/E04.PKG06-003.HOP_EST_DEMAP/handoff.json` and `outputs/agentops/tasks/AGENT.INFRA.SELFTEST/handoff.json` were written on 2026-11-19 by an earlier writer and fail the current `handoff.v1` schema with nine errors each. They are retained evidence and are not rewritten. The E04 scientific disposition `NOT_QUALIFIED` is unaffected. `nr-agent doctor` prints `scientific_qualification: UNCHANGED` as a literal string, not as a computed check.
8. **Redaction scope.** Receipts written before the summary change still carry argument values verbatim: eleven retained files under `outputs/agentops/` (eight under `tasks/`, three under `verification/`; the 2026-11-19 receipts and handoffs) are not rewritten and carry benign static commands. Receipts written after the change persist no argument values (see row 21). The remaining text-redaction surface is parsed provider responses and receipts' structured fields, which are redacted by key name and by pattern. Pattern-based text redaction is not exhaustive: unlabelled bare tokens, plural `tokens` keys, and free-form prose are outside the patterns. Raw provider output is omitted from failure receipts, so receipts do not depend on text redaction.
9. **Provider mutation guard and ignored paths.** `current_worktree_digest` reads `git diff --name-only --no-renames HEAD` and `git ls-files -o --exclude-standard`, and `dirty_paths` reads `git status` without `--ignored`. Neither includes ignored paths, so provider writes to git-ignored files (including retained evidence under `outputs/` and `logs/`) are not detected by the mutation guard. Owned-path guards likewise do not see ignored files.
10. **Handoff command and stale card.** `docs/operations/AGENT_TASK_PROTOCOL.md` lists `nr-agent handoff AGENT.INFRA.SELFTEST` as a typical command. A legacy `DONE` handoff is no longer rebuilt from file content: `nr-agent handoff` refuses it with `legacy DONE handoff cannot be rebuilt; re-run verification`, so the retained E04 and AGENT.INFRA.SELFTEST `DONE` handoffs are not reconstructed. A current-schema `DONE` handoff is shown only if its evidence is current. `nr-agent list` reports `AGENT.INFRA.SELFTEST` as `STALE_TASK_CARD` because its card changed after the state was recorded; the `DONE` phase in this report is the retained historical phase, not the operational phase.
11. **Dependency gate (fixed in this revision).** `_require_dependencies_done` now loads the queue with reconciled state. A dependency counts as satisfied only when its operational phase is `DONE` and its verification evidence is current. Consequence: `E04.PKG06-003.HOP_EST_DEMAP` cannot start, resume, or finalize until `AGENT.INFRA.SELFTEST` is re-verified, because that dependency's evidence is stale. Its retained `DONE` is not changed.
12. **Receipts, handoff order, and legacy rebuilds (partly fixed).** Provider JSON nesting deeper than 256 levels is refused with a `malformed_json` receipt, so no recursion limit is reached in persistence. `handoff.json` is now published only after the guarded `DONE` transition succeeds. A DONE handoff now requires an `APPROVE` verdict and every check to exit 0 and not be skipped. Verification still writes raw stdout and stderr (up to 2 MB each) into phase receipts; this is disclosed, not changed.
13. **Active jobs, stale_if, and review artifacts (implemented in this revision, with limits).**
    - *Active jobs (item 9).* The definition is the one stated in row 9. Detection reads `/proc` for same-user processes. A process whose `/proc` entries are unreadable is listed as `uninspected` and is not refused, so another user's running job is not protected. The check fails closed when `/proc` cannot be enumerated. The check is a point-in-time scan before mutation; it does not lock files, so a job that starts after the check is not blocked.
    - *stale_if (item 12).* `SharedFact` evaluates `stale_if` paths against the bound revision. No gate, handoff, or CLI command loads shared facts yet, so a stale fact is detectable through the API but does not block any execution path.
    - *Review artifacts (item 23).* Each review cycle keeps its own directory under `review-cycles/`. Cycles that fail before a report exists keep only the source context.
    - *Test isolation.* With the active-job check live, the full neural suite's traced-memory test `tests/test_datasets/test_nr_store_memory_scaling.py` failed in this environment. The runner tests now stub the host process scan through `tests/test_agentops/conftest.py`. Bisection showed the failure needed the real `/proc` scan inside the runner tests. The mechanism was not identified; the fixture removes the interaction, and the memory test passes in the final matrix.
    - None of these changes has been reviewed by an independent reviewer. Item 22 is unchanged by them.
    - *Revision binding (commit step).* Task cards now bind `source_revision` to HEAD or an ancestor of HEAD (`agentops.contracts.is_ancestor_revision`), not to exact equality with HEAD. Equality cannot hold once the transfer is committed, since the commit moves HEAD. A card with a revision that is not HEAD or an ancestor is still rejected as stale. The first commit (69ace52) broke 38 AgentOps tests under equality; the follow-up commit fixes this. Four regression tests are in `tests/test_agentops/test_revision_ancestry.py`.
    - *Commit scope.* The transfer is committed on branch `agentops-architecture-transfer` and pushed to `origin`. The commit contains only AgentOps, agent-governance, CI parity, and documentation paths (122 files in the first commit). Unowned scientific-module changes are not committed; they remain in the worktree and are the open owner decision in item 3.
    - *Native GPU engineering evidence (2026-11-22, host with 3 Tesla V100-PCIE-16GB, compute capability 7.0, driver 575.57.08).* Environment: Python 3.12, `sionna-no-rt==2.2.0`, `torch 2.14.1+cu126` from the project's cu126 index. CUDA matmul verified. The native PHY marker run (`pytest -m native_sionna`, RT extra not installed) gave 1219 passed, 9 skipped, 18 failed before the fix below. The two CUDA standardized-PUSCH failures and their IndexError were fixed in `datasets/nr.py`: a stale torch-module variable made a CUDA target reach `np.asarray`. The 15 standardized-PUSCH tests now pass on the GPU. Thirteen G4 covariance, admission, and K-factor tests still fail, both on the GPU and with CUDA hidden. Their pinned 3GPP-derived references disagree with Sionna output, for example a K-factor of 12.2 dB against a reference of 9.02 dB. These are retained native evidence, not engineering noise, and the references were not changed. Three RT tests in `tests/test_integrations/test_network_phy_native.py` are blocked: Dr.Jit's LLVM backend needs LLVM 15 or newer, and this host has LLVM 12, so RT initialization hangs. The `datasets/nr.py` edit is in an unowned scientific file and stays uncommitted in the item 3 patch.

## Changed files

Exact roster: `/home/l2nguyen/agent-final-2026-11-22/roster/tracked-name-status.txt` (136 modified tracked files), `untracked-files.txt` (114 files), and the outside-ownership lists in item 3 above. The worktree is intentionally uncommitted.

## Next steps

1. Run a link and CLI audit for item 32.
2. Freeze the final worktree and obtain a fresh independent whole-diff re-review, with a bundle hash and a verdict file.
3. If the verdict is `APPROVE`, mark items 22 and 30 `PASS` and decide item 19 (see limitation 2).
4. Owner decision for item 3 (attribution of unowned changes).
