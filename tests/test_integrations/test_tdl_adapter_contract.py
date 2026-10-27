"""Dependency-isolated contracts for the standardized TDL adapter."""

from __future__ import annotations

import hashlib
import json
import sys
import types

import pytest


@pytest.mark.parametrize(
    ("adapter_name", "overrides", "message"),
    [
        ("tdl", {"carrier_frequency_hz": True}, "carrier_frequency_hz"),
        ("tdl", {"subcarrier_spacing_hz": "30000"}, "subcarrier_spacing_hz"),
        ("tdl", {"delay_spread_s": True}, "delay_spread_s"),
        ("tdl", {"los_angle_of_arrival_rad": False}, "los_angle_of_arrival_rad"),
        ("tdl", {"min_speed_mps": "walking"}, "min_speed_mps"),
        ("cdl", {"sampling_frequency_hz": True}, "sampling_frequency_hz"),
        ("cdl", {"delay_spread_s": "1e-7"}, "delay_spread_s"),
        ("cdl", {"min_speed_mps": True}, "min_speed_mps"),
        ("cdl", {"max_speed_mps": "walking"}, "max_speed_mps"),
        ("cdl", {"ut_orientation_rad": (0.0, True, 0.0)}, "components must be finite real"),
    ],
)
def test_standardized_adapters_reject_boolean_and_nonreal_physical_inputs(
    adapter_name: str, overrides: dict[str, object], message: str
) -> None:
    from integrations.sionna.cdl import generate_cdl_frequency_response
    from integrations.sionna.tdl import generate_tdl_frequency_response

    if adapter_name == "tdl":
        arguments: dict[str, object] = {
            "model": "A",
            "seed": 1,
            "carrier_frequency_hz": 3.5e9,
            "subcarrier_spacing_hz": 30_000.0,
            "num_subcarriers": 8,
            "delay_spread_s": 100e-9,
        }
        generate = generate_tdl_frequency_response
    else:
        arguments = {
            "model": "A",
            "seed": 1,
            "carrier_frequency_hz": 3.5e9,
            "subcarrier_spacing_hz": 30_000.0,
            "num_subcarriers": 8,
            "delay_spread_s": 100e-9,
        }
        generate = generate_cdl_frequency_response

    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        generate(**arguments)  # type: ignore[arg-type]


def test_cdl_array_spacing_rejects_boolean_values() -> None:
    from integrations.sionna.cdl import CDLArrayConfig

    with pytest.raises(ValueError, match="element_horizontal_spacing"):
        CDLArrayConfig(element_horizontal_spacing=True)  # type: ignore[arg-type]


def test_tdl_adapter_forwards_nondefault_num_sinusoids(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    import integrations.sionna.tdl as tdl_adapter
    from datasets import StandardizedFamily, validate_standardized_channel_admission

    captured: dict[str, object] = {}

    class FakeTDL:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.los = False
            self.delay_spread = torch.tensor(100e-9)
            self.num_clusters = 1
            self.delays = torch.tensor([0.0])
            self.mean_powers = torch.tensor([1.0])

        def __call__(
            self,
            *,
            batch_size: int,
            num_time_steps: int,
            sampling_frequency: float,
        ) -> tuple[object, object]:
            del sampling_frequency
            h_path = torch.zeros(
                (batch_size, 1, 1, 1, 1, 1, num_time_steps),
                dtype=torch.complex64,
            )
            tau_s = torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32)
            return h_path, tau_s

    class FakeResource:
        def read_bytes(self) -> bytes:
            return b'{"model":"TDL-A","version":"19.2"}\n'

    modules = {
        module_name: types.ModuleType(module_name)
        for module_name in (
            "sionna",
            "sionna.phy",
            "sionna.phy.channel",
            "sionna.phy.channel.tr38901",
            "sionna.phy.channel.tr38901.models",
        )
    }
    for module_name, module in modules.items():
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, module)
    modules["sionna"].phy = modules["sionna.phy"]  # type: ignore[attr-defined]
    modules["sionna.phy"].channel = modules["sionna.phy.channel"]  # type: ignore[attr-defined]
    modules["sionna.phy.channel"].tr38901 = modules[  # type: ignore[attr-defined]
        "sionna.phy.channel.tr38901"
    ]
    modules["sionna.phy.channel.tr38901"].TDL = FakeTDL  # type: ignore[attr-defined]
    modules["sionna.phy.channel.tr38901"].models = modules[  # type: ignore[attr-defined]
        "sionna.phy.channel.tr38901.models"
    ]
    modules["sionna.phy.channel.tr38901.models"].parameter_file = (  # type: ignore[attr-defined]
        lambda filename, spec_version: FakeResource()
    )
    monkeypatch.setattr(tdl_adapter, "call_with_sionna_seed", lambda _device, _seed, call: call())

    channel = tdl_adapter.generate_tdl_frequency_response(
        model="A",
        seed=3,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        delay_spread_s=100e-9,
        num_sinusoids=7,
    )

    assert captured["num_sinusoids"] == 7
    assert channel.metadata["num_sinusoids"] == 7
    assert channel.metadata["profile_parameter_table_source"] == "TR 38.901:19.2:TDL-A.json"
    assert (
        channel.metadata["profile_parameter_table_sha256"]
        == hashlib.sha256(FakeResource().read_bytes()).hexdigest()
    )
    validate_standardized_channel_admission(
        StandardizedFamily.TDL, "A", channel.h_path[0], channel.metadata
    )
    tampered_metadata = dict(channel.metadata)
    tampered_metadata["profile_parameter_table_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="resolved TDL profile parameter digest is invalid"):
        validate_standardized_channel_admission(
            StandardizedFamily.TDL, "A", channel.h_path[0], tampered_metadata
        )
    tampered_metadata = dict(channel.metadata)
    tampered_metadata["profile_parameter_table_source"] = "TR 38.901:16.1:TDL-A.json"
    with pytest.raises(ValueError, match="parameter table source identity or digest is invalid"):
        validate_standardized_channel_admission(
            StandardizedFamily.TDL, "A", channel.h_path[0], tampered_metadata
        )
    tampered_metadata = dict(channel.metadata)
    tampered_metadata.pop("resolved_profile_parameters_sha256")
    with pytest.raises(ValueError, match="must be bound by the profile digest"):
        validate_standardized_channel_admission(
            StandardizedFamily.TDL, "A", channel.h_path[0], tampered_metadata
        )


def test_cdl_profile_metadata_hashes_importlib_resource_bytes() -> None:
    torch = pytest.importorskip("torch")

    from integrations.sionna.cdl import _profile_metadata

    table_bytes = json.dumps(
        {
            "powers": [0.0, -3.0, -6.0],
            "los": False,
            "delays": [0.0, 0.5, 1.0],
            "aod": [0.0, 1.0, 2.0],
            "aoa": [0.0, 1.0, 2.0],
            "zod": [0.0, 1.0, 2.0],
            "zoa": [0.0, 1.0, 2.0],
            "cASD": 5.0,
            "cASA": 5.0,
            "cZSD": 5.0,
            "cZSA": 5.0,
            "xpr": 10.0,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    class FakeResource:
        def read_bytes(self) -> bytes:
            return table_bytes

    class FakeResources:
        def parameter_file(self, filename: str, spec_version: str) -> FakeResource:
            assert filename == "CDL-A.json"
            assert spec_version == "19.2"
            return FakeResource()

    metadata = _profile_metadata("A", "19.2", 100e-9, FakeResources(), torch, "cpu", torch.float32)

    assert metadata["parameter_table"] == "CDL-A.json"
    assert metadata["parameter_table_source"] == "TR 38.901:19.2:CDL-A.json"
    assert metadata["parameter_table_sha256"] == hashlib.sha256(table_bytes).hexdigest()
