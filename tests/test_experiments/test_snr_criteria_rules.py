"""SNR acceptance criteria: schema, roster, timestamp and execution-context admission rules."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from experiments.acceptance import (
    SNR_EXECUTION_ACCEPTANCE_SCHEMA,
    freeze_snr_acceptance_criteria,
    validate_preregistration_order,
)
from tests.test_experiments.test_acceptance import (
    _criteria,
    _execution_bound_criteria,
    _timestamped_criteria,
)


def _freeze(criteria: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "snr_values": (0.0, 5.0),
        "channel_seeds": tuple(range(100, 120)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 4,
    }
    arguments.update(overrides)
    return freeze_snr_acceptance_criteria(criteria, **arguments)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(lambda c: c.update(schema="umcd.other.v1"), "unsupported", id="schema"),
        pytest.param(
            lambda c: c.update(extra_field=1), "missing or unknown fields", id="unknown-field"
        ),
        pytest.param(lambda c: c.pop("snr_db"), "missing or unknown fields", id="missing-field"),
        pytest.param(
            lambda c: c.update(snr_db=[0.0, float("nan")]), "canonical JSON values", id="snr-nan"
        ),
        pytest.param(
            lambda c: c.update(channel_estimator="other"),
            "channel estimator is unsupported",
            id="estimator",
        ),
        pytest.param(lambda c: c.update(blocks_per_channel=0), "positive integer", id="blocks"),
        pytest.param(
            lambda c: c.update(minimum_independent_channels=99),
            "exceeds the frozen channel-seed roster",
            id="channels",
        ),
        pytest.param(
            lambda c: c.update(snr_db=[5.0, 0.0]),
            "exactly match the requested curve",
            id="snr-order",
        ),
    ],
)
def test_criteria_are_frozen_only_when_they_match_the_contract(mutate, message) -> None:
    criteria = copy.deepcopy(_criteria())
    mutate(criteria)
    with pytest.raises(ValueError, match=message):
        _freeze(criteria)


def test_criteria_must_be_an_object() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        _freeze(["not", "an", "object"])


def test_timestamped_contract_requires_a_utc_preregistration_time() -> None:
    criteria = copy.deepcopy(_timestamped_criteria())
    criteria["preregistered_utc"] = "2026-10-25T09:00:00"
    with pytest.raises(ValueError, match="ending in Z|must use UTC|ISO-8601"):
        _freeze(criteria)
    criteria["preregistered_utc"] = "2026-10-25T09:00:00+02:00"
    with pytest.raises(ValueError, match="ending in Z|must use UTC"):
        _freeze(criteria)


def test_execution_context_digest_must_match_its_contents() -> None:
    criteria = copy.deepcopy(_execution_bound_criteria())
    criteria["execution_context_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="execution context|digest|differs"):
        _freeze(criteria)


def test_execution_context_entries_must_be_objects() -> None:
    criteria = copy.deepcopy(_execution_bound_criteria())
    criteria["execution_context"] = ["not", "object"]
    with pytest.raises((TypeError, ValueError)):
        _freeze(criteria)


def test_execution_bound_schema_is_recognised() -> None:
    assert SNR_EXECUTION_ACCEPTANCE_SCHEMA.endswith(".v3")


def test_preregistration_order_must_precede_the_run() -> None:
    criteria = copy.deepcopy(_timestamped_criteria())
    criteria["preregistered_utc"] = "2026-10-25T09:00:00Z"
    with pytest.raises(ValueError):
        validate_preregistration_order(criteria, receiver_run_started_utc="2026-10-24T09:00:00Z")
    criteria["preregistered_utc"] = "not-a-time"
    with pytest.raises(ValueError):
        validate_preregistration_order(criteria, receiver_run_started_utc="2026-10-24T09:00:00Z")
    criteria["preregistered_utc"] = "2026-10-01T09:00:00Z"
    assert validate_preregistration_order(criteria, receiver_run_started_utc="2026-10-24T09:00:00Z")
