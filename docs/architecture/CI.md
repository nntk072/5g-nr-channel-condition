# Continuous integration

GitHub Actions runs `.github/workflows/ci.yml` for pushes to `main` or
`master`, pull requests, and manual dispatches.

The test job checks that `uv.lock` matches `pyproject.toml`, installs the
project with its development dependencies, runs Ruff lint and formatting
checks, runs mypy, and executes pytest on Python 3.11 and 3.12. A separate
Python 3.12 job installs CPU PyTorch and runs the same suite to exercise neural
training and checkpoint reloads. The build job creates both a wheel and a
source distribution. Sionna's native ray-tracing stack is an optional research
dependency and is not installed on standard CI runners; its adapter is covered
with mocked boundaries, with a separate native smoke command documented in
the research quickstart.

Run the same checks locally from the repository root:

```bash
uv lock --check
uv sync --locked --extra dev
uv run --locked ruff check artifacts configuration datasets propagation receivers models learning evaluation experiments integrations compute loslab tests
uv run --locked ruff format --check artifacts configuration datasets propagation receivers models learning evaluation experiments integrations compute loslab tests
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

Pytest reports line coverage for the active packages. Native Sionna/RT remains
an opt-in environment check because its Mitsuba and Dr.Jit runtimes are not
part of the standard CPU CI environment.
