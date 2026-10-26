from typing import Any, cast

import pytest
from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"model": "F"}, "model must be one of"),
        ({"spec_version": "99.0"}, "unsupported 3GPP TR 38.901 profile-table version"),
        ({"seed": -1}, "seed must be a nonnegative integer"),
        ({"num_subcarriers": 4}, "num_subcarriers must be >= 8"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"direction": "sideways"}, "direction must be"),
        ({"normalize_delays": False}, "normalize_delays=False is unsupported"),
        ({"delay_spread_s": -1.0}, "delay_spread_s must be finite and nonnegative"),
        ({"ut_orientation_rad": (0.0, float("nan"), 0.0)}, "components must be finite"),
        ({"ut_velocity_mps": (1.0, 2.0)}, "exactly three components"),
        ({"ut_velocity_mps": (1.0, 0.0, 0.0), "min_speed_mps": 1.0}, "set ut_velocity_mps"),
    ],
)
def test_cdl_adapter_rejects_invalid_contract_before_optional_import(values, message):
    config = {
        "model": "A",
        "seed": 1,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 64,
        "delay_spread_s": 100e-9,
    }
    config.update(values)
    with pytest.raises(ValueError, match=message):
        generate_cdl_frequency_response(**cast(Any, config))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"num_cols_per_panel": 0}, "num_cols_per_panel must be a positive integer"),
        ({"polarization": "dual", "polarization_type": "V"}, "polarization_type must be one of"),
        ({"element_horizontal_spacing": float("inf")}, "must be finite and positive"),
    ],
)
def test_cdl_array_config_rejects_invalid_panel_geometry(values, message):
    with pytest.raises(ValueError, match=message):
        CDLArrayConfig(**values)


def test_cdl_array_config_is_immutable_and_json_ready():
    config = CDLArrayConfig(num_cols_per_panel=2, polarization="dual", polarization_type="VH")

    assert config.num_cols_per_panel == 2
    assert config.polarization == "dual"
    with pytest.raises(AttributeError):
        config.num_cols_per_panel = 3
