from __future__ import annotations

from artifacts.manifest import mark_complete, run_manifest


def test_manifest_transitions_to_complete_without_mutating_running_record() -> None:
    running = run_manifest("run-001", {"seed": 42}, input_count=3)
    complete = mark_complete(running)

    assert running["state"] == "running"
    assert "completed_utc" not in running
    assert complete["state"] == "complete"
    assert complete["completed_utc"]
    assert complete["config_identity"] == running["config_identity"]
    assert complete["input_count"] == 3
