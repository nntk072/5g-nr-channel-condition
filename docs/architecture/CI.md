# Continuous integration

GitHub Actions runs `.github/workflows/ci.yml` on every push, every pull request,
and manual dispatch. Each job runs one named gate from `tools/check`, so the same
command runs locally and in CI.

| Job | Gate | What it checks |
| --- | --- | --- |
| `lock` | `./tools/check lock` | `uv.lock` matches `pyproject.toml`. Runs first; every other job waits for it. |
| `lint` | `./tools/check lint` | Ruff lint and Ruff formatting over the source, scripts, AgentOps, and tests. |
| `typecheck` | `./tools/check typecheck` | mypy over the configured packages. |
| `architecture` | `./tools/check architecture` | Package-boundary tests in `tests/architecture`. |
| `test` | `./tools/check test` | Core pytest suite on Python 3.12 with coverage disabled. |
| `neural` | `./tools/check neural` | Neural and boosted baseline tests with CPU-only PyTorch and the configured 75% coverage threshold. |
| `build` | `./tools/check build` | Wheel and source distribution, plus an import smoke test of the packages. |
| `ci-success` | (aggregator) | Fails if any job above failed or was skipped. Make this the required check on the branch. |

Python 3.12 is the only supported CI version. Sionna's native ray-tracing stack is
an optional research dependency and is not installed on CI runners. Its adapter is
covered with mocked boundaries, and native checks run only through the opt-in
gates described below.

Run the same checks locally from the repository root:

```bash
./tools/check lock
uv sync --locked --extra dev
./tools/check lint
./tools/check typecheck
./tools/check architecture
./tools/check test
./tools/check build
```

The neural gate needs the boosting extra and CPU-only PyTorch, which CI installs
before running it. Locally:

```bash
uv sync --locked --extra dev --extra boosting
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
./tools/check neural
```

The core and test gates use `--no-cov`. The neural gate enforces the configured
coverage threshold.

## Agent governance checks (local only)

`tools/check agent` validates canonical agent rules and profiles, runs the
read-only doctor, and executes the isolated `tests/test_agentops` suite and
rule-validation tests without native Sionna. It is a local gate. CI does not run it
as a separate job: the AgentOps tests already run in the `test` job. `tools/check fast`
adds Ruff, formatting, mypy, and the active changed Python tests.

`tools/check native-smoke` remains an explicit opt-in gate and has no generic native
workload.

Agent verification receipts record exact argv, exit code, bounded output hashes,
source revision, task-card SHA-256, and worktree digest. A later worktree mutation
invalidates a prior receipt for `DONE`. Failed gates propagate nonzero status and
are retained as operational evidence.
