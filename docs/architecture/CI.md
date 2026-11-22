# Continuous integration

GitHub Actions runs `.github/workflows/ci.yml` for pushes to `main` or
`master`, pull requests, and manual dispatches.

The test job checks that `uv.lock` matches `pyproject.toml`, installs the
project with its development dependencies, runs Ruff lint and formatting
checks, runs mypy, and executes pytest with coverage disabled on Python 3.11 and
3.12. A separate Python 3.12 job installs CPU PyTorch and runs pytest with
coverage enabled to exercise neural training and checkpoint reloads; this job
enforces the configured 75% total coverage threshold. The build job creates
both a wheel and a source distribution. Sionna's native ray-tracing stack is an
optional research dependency and is not installed on standard CI runners; its adapter is covered
with mocked boundaries, with a separate native smoke command documented in
the research quickstart.

Run the same checks locally from the repository root:

```bash
uv lock --check
uv sync --locked --extra dev
uv run --locked ruff check artifacts configuration datasets propagation receivers models learning evaluation experiments integrations compute loslab scripts agentops tests
uv run --locked ruff format --check artifacts configuration datasets propagation receivers models learning evaluation experiments integrations compute loslab scripts agentops tests
uv run --locked mypy
uv run --locked python -m pytest
uv build
```

To run the optional neural checks locally, install the CPU-only PyTorch wheel
and avoid resynchronizing the base environment for that invocation:

```bash
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync --locked python -m pytest
```

The core workflow jobs use `--no-cov`; the neural job enforces the configured
coverage threshold. Native Sionna/RT remains an opt-in environment check
because its Mitsuba and Dr.Jit runtimes are not part of the standard CPU CI
environment.

## Agent policy and control-plane checks

`tools/check agent` validates canonical rules and profiles, runs the read-only
doctor, and executes the isolated `tests/test_agentops` suite plus structural
rule-validation tests without native Sionna. `tools/check fast` adds Ruff,
formatting, mypy, and the active changed Python tests.

The existing Python 3.11/3.12 test jobs, neural/coverage job, and package build remain
the authoritative portable CI surfaces. The additive `agent-policy` job runs on
Python 3.12 and is CPU-only. `tools/check native-smoke` remains an explicit opt-in
gate and has no generic native workload.

Agent verification receipts record exact argv, exit code, bounded output hashes,
source revision, task-card SHA-256, and worktree digest. A later worktree mutation
invalidates a prior receipt for `DONE`. Failed gates propagate nonzero status and
are retained as operational evidence.
