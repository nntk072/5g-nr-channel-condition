"""Shared AgentOps test fixtures."""

from __future__ import annotations

import pytest
from agentops import runner as runner_module


@pytest.fixture(autouse=True)
def _isolate_runner_from_host_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Runner tests use temporary stores and do not test the host process table. Scanning the real
    # /proc from inside the shared pytest process changes the traced-memory measurements of later
    # memory-profile tests (tests/test_datasets/test_nr_store_memory_scaling.py). The real scan is
    # covered directly in tests/test_agentops/test_stale_cycle_active_job.py.
    monkeypatch.setattr(
        runner_module,
        "find_active_jobs",
        lambda roots, **kwargs: {"active": [], "uninspected": []},
    )
