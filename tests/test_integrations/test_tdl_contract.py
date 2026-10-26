from typing import Any, cast

import pytest
from integrations.sionna.tdl import generate_tdl_frequency_response


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"num_subcarriers": 4}, "num_subcarriers"),
        ({"batch_size": 0}, "batch_size"),
        ({"num_time_steps": 0}, "num_time_steps"),
        ({"subcarrier_spacing_hz": 0}, "subcarrier_spacing_hz"),
        ({"los_angle_of_arrival_rad": float("nan")}, "los_angle_of_arrival_rad"),
        ({"seed": -1}, "seed"),
        ({"spec_version": "99.0"}, "unsupported 3GPP TR 38.901 profile-table version"),
    ],
)
def test_tdl_adapter_rejects_invalid_inputs_before_optional_import(values, message):
    config = {
        "model": "A",
        "seed": 1,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 64,
    }
    config.update(values)
    with pytest.raises(ValueError, match=message):
        generate_tdl_frequency_response(**cast(Any, config))
