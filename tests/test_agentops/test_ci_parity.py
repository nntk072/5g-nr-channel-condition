from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ci_keeps_portable_neural_build_and_agent_policy_jobs() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for marker in (
        'python-version: ["3.11", "3.12"]',
        "uv lock --check",
        "ruff check artifacts configuration datasets propagation receivers",
        "ruff format --check artifacts configuration datasets propagation receivers",
        "uv run --locked mypy",
        "uv run --locked python -m pytest --no-cov",
        "torch==2.11.0",
        "uv run --no-sync --locked python -m pytest",
        "uv build",
        "agent-policy:",
        "./tools/check agent",
    ):
        assert marker in workflow
