# Verified Command Surface

The portable development environment uses Python 3.11 or 3.12 and `uv` with the committed lockfile.

```bash
uv lock --check
uv sync --locked --extra dev
uv run --locked ruff check artifacts configuration datasets propagation receivers models learning evaluation experiments integrations compute loslab scripts tests
uv run --locked ruff format --check artifacts configuration datasets propagation receivers models learning evaluation experiments integrations compute loslab scripts tests
uv run --locked mypy
uv run --locked python -m pytest --no-cov
uv build
```

The existing CI also has a Python 3.12 neural and boosted job that installs CPU PyTorch before running the full suite. Native Sionna PHY and RT checks require their optional extras and are separate from the portable gate.

As of the 2026-11-19 Halti baseline, `uv` and managed Python 3.12.16 are restored. The current source baseline still has existing format and Mypy failures, and portable pytest collection without the neural dependency fails because a test imports `torch`. These baseline failures must stay distinguishable from new agent-infrastructure regressions.
