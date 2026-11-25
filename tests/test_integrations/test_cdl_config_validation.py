"""Portable CDL configuration and preflight validation tests."""

from __future__ import annotations

import pytest
from integrations.sionna.cdl import (
    CDLArrayConfig,
    CDLHandheldUTArrayConfig,
    generate_cdl_frequency_response,
)


@pytest.mark.parametrize(
    "overrides",
    (
        {"num_rows_per_panel": True},
        {"num_cols_per_panel": 0},
        {"num_rows": 1.5},
        {"polarization": "triple"},
        {"polarization_type": "VH"},
        {"antenna_pattern": "invalid"},
        {"element_vertical_spacing": True},
        {"element_horizontal_spacing": float("inf")},
        {"panel_vertical_spacing": 0.0},
    ),
)
def test_panel_array_configuration_rejects_invalid_fields(overrides) -> None:
    with pytest.raises(ValueError):
        CDLArrayConfig(**overrides)


def test_panel_array_configuration_accepts_dual_polarization() -> None:
    config = CDLArrayConfig(polarization="dual", polarization_type="VH")

    assert config.polarization_type == "VH"


@pytest.mark.parametrize(
    ("overrides", "error"),
    (
        ({"polarization": "triple"}, ValueError),
        ({"antenna_locations": "unknown"}, ValueError),
        ({"antenna_locations": ()}, ValueError),
        ({"antenna_locations": (1, True)}, ValueError),
        ({"antenna_locations": (1, 1)}, ValueError),
        ({"antenna_locations": [1, 2]}, TypeError),
        ({"polarization": "dual", "polarization_type": "V"}, ValueError),
        ({"antenna_pattern": "invalid"}, ValueError),
        ({"device_depth": float("nan")}, ValueError),
        ({"port_power_offsets_db": [0.0]}, TypeError),
        ({"port_power_offsets_db": (0.0,)}, ValueError),
        ({"port_power_offsets_db": (float("nan"),) * 8}, ValueError),
    ),
)
def test_handheld_array_configuration_rejects_invalid_fields(overrides, error) -> None:
    with pytest.raises(error):
        CDLHandheldUTArrayConfig(**overrides)


def test_handheld_array_configuration_accepts_dual_polarized_candidate_roster() -> None:
    config = CDLHandheldUTArrayConfig(
        polarization="dual",
        polarization_type="cross",
        antenna_locations=(2, 4),
        port_power_offsets_db=(0.0, -1.0, 0.5, 0.25),
    )

    assert config.antenna_locations == (2, 4)


def _generate(**overrides):
    values = {
        "model": "A",
        "seed": 1,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30e3,
        "num_subcarriers": 64,
        "delay_spread_s": 100e-9,
    }
    values.update(overrides)
    return generate_cdl_frequency_response(**values)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    (
        ({"model": "Z"}, ValueError, "model must be one of"),
        ({"spec_version": "99.0"}, ValueError, "unsupported.*profile-table version"),
        ({"seed": True}, ValueError, "seed must be a nonnegative integer"),
        ({"seed": -1}, ValueError, "seed must be a nonnegative integer"),
        ({"num_subcarriers": True}, TypeError, "num_subcarriers must be an integer"),
        ({"num_subcarriers": 7}, ValueError, "num_subcarriers must be >= 8"),
        ({"batch_size": 0}, ValueError, "batch_size must be positive"),
        ({"num_time_steps": 0}, ValueError, "num_time_steps must be positive"),
        ({"direction": "sideways"}, ValueError, "direction must be"),
        ({"normalize_delays": 1}, TypeError, "normalize_delays must be a boolean"),
        ({"normalize_delays": False}, ValueError, "normalize_delays=False is unsupported"),
        ({"carrier_frequency_hz": float("inf")}, ValueError, "carrier_frequency_hz must be finite"),
        ({"delay_spread_s": -1.0}, ValueError, "delay_spread_s must be finite and nonnegative"),
        ({"min_speed_mps": -1.0}, ValueError, "min_speed_mps must be finite and nonnegative"),
        ({"max_speed_mps": -1.0}, ValueError, "max_speed_mps must be finite and >= min_speed_mps"),
        ({"min_speed_mps": 1.0}, ValueError, "max_speed_mps must be specified"),
        ({"ut_orientation_rad": (0.0, 1.0)}, ValueError, "must contain exactly three"),
        ({"ut_orientation_rad": (0.0, True, 0.0)}, ValueError, "components must be finite real"),
        ({"ut_orientation_rad": (0.0, float("nan"), 0.0)}, ValueError, "components must be finite"),
        (
            {"ut_velocity_mps": (1.0, 0.0, 0.0), "min_speed_mps": 0.5},
            ValueError,
            "set ut_velocity_mps or",
        ),
        ({"device": "tpu"}, ValueError, "device must be cpu or a CUDA device"),
    ),
)
def test_generation_preflight_rejects_invalid_requests(overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _generate(**overrides)


def test_generation_uses_mock_channel_for_axis_and_metadata_contract(monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    import sionna.phy.channel.tr38901 as tr38901
    import torch

    class Array:
        def __init__(self, **configuration):
            self.configuration = configuration

    class Channel:
        def __init__(self, **configuration):
            self.configuration = configuration

        def __call__(self, *, batch_size, num_time_steps, sampling_frequency):
            assert (batch_size, num_time_steps, sampling_frequency) == (1, 2, 1.0)
            taps = torch.ones((1, 1, 1, 1, 1, 2, 2), dtype=torch.complex64)
            delays = torch.tensor([[[[0.0, 1e-6]]]], dtype=torch.float32)
            return taps, delays

    class FakeHandheldArray(Array):
        pass

    monkeypatch.setattr(tr38901, "CDL", Channel)
    monkeypatch.setattr(tr38901, "PanelArray", Array)
    monkeypatch.setattr(tr38901, "HandheldUTArray", FakeHandheldArray)
    monkeypatch.setattr(tr38901, "models", object())
    monkeypatch.setattr(
        "integrations.sionna.cdl.call_with_sionna_seed",
        lambda _device, _seed, operation: operation(),
    )
    monkeypatch.setattr(
        "integrations.sionna.cdl._profile_metadata",
        lambda *_args: {
            "fixture": True,
            "parameter_table": "CDL-A.json",
            "parameter_table_source": "TR 38.901:19.2:CDL-A.json",
            "parameter_table_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        "integrations.sionna.cdl.sionna_phy_version",
        lambda: (_ for _ in ()).throw(PackageNotFoundError("sionna-phy")),
    )

    channel = _generate(
        num_time_steps=2,
        ut_velocity_mps=(3.0, 4.0, 0.0),
        ut_array=CDLHandheldUTArrayConfig(),
    )

    assert channel.h_path.shape == (1, 1, 1, 1, 1, 2, 2)
    assert channel.tau_s.shape == (1, 1, 1, 2)
    assert channel.channel_frequency_response.shape == (1, 1, 1, 1, 1, 2, 64)
    assert channel.metadata["sionna_version"] == "unknown"
    assert channel.metadata["geometric_los_label_provided"] is False
    assert channel.metadata["ut_array_kind"] == "handheld_ut"
    assert channel.metadata["maximum_doppler_hz"] > 0
    assert channel.metadata["temporal_evolution_observed"] is True

    panel_channel = _generate(num_time_steps=2)
    assert panel_channel.metadata["ut_array_kind"] == "panel"
    with pytest.raises(TypeError, match="ut_array must be"):
        _generate(ut_array=object())
    with pytest.raises(TypeError, match="bs_array must be"):
        _generate(bs_array=object())


def test_generation_reports_missing_optional_sionna_dependencies(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_sionna(name, *args, **kwargs):
        if name == "sionna.phy.channel.tr38901":
            raise ImportError("fixture hides optional Sionna PHY")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_sionna)
    with pytest.raises(RuntimeError, match="Install the optional Sionna 2.2 PHY dependencies"):
        _generate()
