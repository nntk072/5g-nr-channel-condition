"""D1 generation gate: pilot request limits, production denial and destination binding."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from integrations.sionna.d1_generation_gate import authorize_d1_generation

GOOD_RT: dict[str, Any] = {
    "operation": "rt_data",
    "config": {
        "per_class": 20,
        "max_attempts": 40,
        "samples_per_src": 100,
        "max_depth": 3,
        "total_work_units": 4_000,
    },
    "destinations": ["/tmp/d1-pilot-scratch"],
}


def _request(**changes: Any) -> dict[str, Any]:
    request = copy.deepcopy(GOOD_RT)
    request.update(changes)
    return request


def test_bounded_pilot_request_is_authorized() -> None:
    authorization = authorize_d1_generation(GOOD_RT, pilot=True)
    assert authorization.mode == "pilot"


@pytest.mark.parametrize(
    ("config_change", "message"),
    [
        pytest.param({"per_class": 21}, "per_class must be an integer in", id="per-class-over"),
        pytest.param({"max_depth": 0}, "max_depth must be an integer in", id="depth-zero"),
        pytest.param(
            {"samples_per_src": True}, "samples_per_src must be an integer in", id="samples-bool"
        ),
        pytest.param(
            {
                "max_attempts": 2_000,
                "samples_per_src": 5_000_000,
                "total_work_units": 10_000_000_000,
            },
            "total_work_units must be in",
            id="work-over",
        ),
        pytest.param({"total_work_units": 999}, "total_work_units must equal", id="work-mismatch"),
    ],
)
def test_pilot_rt_limits_are_enforced(config_change, message) -> None:
    request = _request()
    request["config"].update(config_change)
    with pytest.raises(ValueError, match=message):
        authorize_d1_generation(request, pilot=True)


def test_pilot_rt_requires_its_full_config() -> None:
    with pytest.raises(ValueError, match="requires the full scene config"):
        authorize_d1_generation(_request(config="not-a-mapping"), pilot=True)


def test_pilot_refuses_unrecognized_operations() -> None:
    with pytest.raises(ValueError, match="limited to recognized RT generation requests"):
        authorize_d1_generation({"operation": "something_else", "config": {}}, pilot=True)


def test_pilot_destinations_must_be_a_json_array() -> None:
    with pytest.raises(ValueError, match="pilot destinations must be a JSON array"):
        authorize_d1_generation(_request(destinations="/tmp/x"), pilot=True)
