from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GATES = ("lock", "lint", "typecheck", "architecture", "test", "neural", "build")


def _workflow() -> dict:
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def test_every_ci_gate_is_a_named_tools_check_command() -> None:
    gate_source = (ROOT / "tools/check").read_text(encoding="utf-8")
    for gate in GATES:
        assert f'gate == "{gate}"' in gate_source or f'"{gate}"' in gate_source, gate
    jobs = _workflow()["jobs"]
    for gate in GATES:
        job_runs = [step.get("run", "") for job in jobs.values() for step in job.get("steps", [])]
        assert any(f"./tools/check {gate}" in run for run in job_runs), gate


def test_ci_is_python_312_only_and_keeps_torch_and_build_pins() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in workflow
    assert 'python-version: ["3.11", "3.12"]' not in workflow
    assert "torch==2.11.0" in workflow


def test_ci_result_job_requires_every_other_job() -> None:
    jobs = _workflow()["jobs"]
    required = set(jobs) - {"ci-success"}
    assert set(jobs["ci-success"]["needs"]) == required
    assert jobs["ci-success"]["if"] == "always()"
    assert "agent-policy" not in jobs
