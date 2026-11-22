from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_ci_keeps_portable_neural_build_and_result_gate() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for marker in (
        'python-version: "3.12"',
        "uv lock --check",
        "ruff check artifacts configuration datasets propagation receivers",
        "ruff format --check artifacts configuration datasets propagation receivers",
        "uv run --locked mypy",
        "uv run --locked python -m pytest --no-cov",
        "torch==2.11.0",
        "uv run --no-sync --locked python -m pytest",
        "uv build",
        "ci-success:",
        "needs: [dependency-lock, lint, test, neural, build]",
    ):
        assert marker in workflow


def test_ci_result_job_requires_every_other_job() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    required = set(jobs) - {"ci-success"}
    assert set(jobs["ci-success"]["needs"]) == required
    assert jobs["ci-success"]["if"] == "always()"
    assert "agent-policy" not in jobs
