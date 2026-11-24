"""Manifest failure transitions for runs that carry a replay contract."""

from __future__ import annotations

from artifacts.manifest import mark_failed, run_manifest
from tests.test_artifacts.test_manifest import _replay_fixture


def test_failure_of_a_manifest_without_replay_keeps_no_replay_block() -> None:
    manifest = run_manifest("run-plain", {"seed": 1})
    failed = mark_failed(manifest, error_type="ValueError", error_message="boom", stage="train")
    assert "run_replay" not in failed


def test_failure_transitions_a_nested_replay_contract() -> None:
    manifest = run_manifest("run-replay", {"seed": 1}, run_replay=_replay_fixture())
    failed = mark_failed(manifest, error_type="RuntimeError", error_message="fail", stage="score")
    assert failed["run_replay"]["state"] == "failed"
    assert failed["state"] == "failed"
