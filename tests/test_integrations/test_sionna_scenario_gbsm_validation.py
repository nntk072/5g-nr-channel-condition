"""Preflight validation for explicit-topology scenario GBSM requests."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from integrations.sionna.scenario_gbsm import (
    _native_array_layout,
    _preflight_numpy_topology,
    _topology_state,
    _topology_tensor,
    generate_scenario_gbsm_frequency_response,
)


def _topology() -> dict[str, np.ndarray]:
    return {
        "ut_loc": np.zeros((1, 1, 3), dtype=np.float32),
        "bs_loc": np.ones((1, 1, 3), dtype=np.float32),
        "ut_velocities": np.zeros((1, 1, 3), dtype=np.float32),
        "in_state": np.zeros((1, 1), dtype=np.bool_),
        "los": np.ones((1, 1, 1), dtype=np.bool_),
    }


def _call(**overrides):
    arguments = {
        "scenario": "UMi",
        "seed": 1,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30e3,
        "num_subcarriers": 12,
        "topology": _topology(),
        "scenario_options": {"o2i_model": "low"},
    }
    arguments.update(overrides)
    return generate_scenario_gbsm_frequency_response(**arguments)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"scenario": None}, ValueError, "scenario must be one of"),
        ({"scenario": "umi"}, ValueError, "scenario must be one of"),
        ({"seed": True}, ValueError, "seed must be a nonnegative integer"),
        ({"seed": 1.5}, ValueError, "seed must be a nonnegative integer"),
        ({"seed": -1}, ValueError, "seed must be a nonnegative integer"),
        ({"carrier_frequency_hz": True}, TypeError, "must be a real number"),
        ({"carrier_frequency_hz": float("inf")}, ValueError, "finite and positive"),
        ({"subcarrier_spacing_hz": 0}, ValueError, "finite and positive"),
        ({"num_subcarriers": True}, TypeError, "must be an integer"),
        ({"num_subcarriers": 7}, ValueError, "at least 8"),
        ({"num_time_steps": False}, TypeError, "must be an integer"),
        ({"num_time_steps": 0}, ValueError, "must be positive"),
        ({"direction": "sideways"}, ValueError, "direction must be downlink or uplink"),
        ({"enable_pathloss": 1}, TypeError, "controls must be Boolean"),
        ({"spec_version": "15.0"}, ValueError, "supported TR 38.901 version"),
        ({"device": "tpu"}, ValueError, "device must be cpu or a CUDA device"),
        ({"ut_array": object()}, TypeError, "must be CDLArrayConfig values"),
        ({"topology": []}, TypeError, "topology must be a mapping"),
        ({"topology": {"unexpected": True}}, ValueError, "topology fields must match"),
        ({"scenario_options": []}, TypeError, "scenario_options must be a mapping"),
        ({"scenario_options": {"o2i_model": float("nan")}}, ValueError, "JSON-compatible"),
        ({"scenario_options": {}}, ValueError, "must include"),
        (
            {"scenario_options": {"o2i_model": "low", "extra": 1}},
            ValueError,
            "must include",
        ),
        ({"scenario_options": {"o2i_model": "medium"}}, ValueError, "must be low or high"),
    ],
)
def test_scenario_request_preflight_rejects_invalid_values(overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _call(**overrides)


@pytest.mark.parametrize("scenario", ["RMa", "InH", "InF"])
def test_scenarios_without_required_options_accept_no_options(
    scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Block the runtime import after software preflight, even if it is installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch is required|Sionna 2.2 PHY is required"):
        _call(
            scenario=scenario,
            scenario_options=None,
            topology={**_topology(), "in_state": np.ones((1, 1), dtype=np.bool_)},
        )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("ut_loc", np.zeros((1, 2, 3)), ValueError, r"shape \[positive batch, 1, 3\]"),
        ("bs_loc", np.zeros((0, 1, 3)), ValueError, r"shape \[positive batch, 1, 3\]"),
        ("ut_velocities", np.zeros((1, 1, 2)), ValueError, r"shape \[positive batch, 1, 3\]"),
        ("in_state", np.zeros((1, 1), dtype=np.int8), TypeError, "Boolean dtype"),
        ("los", np.ones((1, 1), dtype=np.bool_), ValueError, r"shape \[1, 1, 1\]"),
        ("bs_orientations", np.full((1, 1, 3), np.nan), ValueError, "finite values"),
        ("ut_orientations", np.ones((1, 1, 2)), ValueError, r"shape \[batch, 1, 3\]"),
    ],
)
def test_numpy_topology_preflight_rejects_invalid_shapes_and_values(
    field: str, value: np.ndarray, error: type[Exception], message: str
) -> None:
    topology = _topology()
    topology[field] = value
    with pytest.raises(error, match=message):
        _call(topology=topology)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ut_loc", np.zeros((0, 1, 3)), r"shape \[positive batch, 1, 3\]"),
        ("bs_orientations", np.array([["bad"]]), "real numeric"),
    ],
)
def test_numpy_topology_preflight_rejects_empty_batch_and_bad_optional_values(
    field: str, value: np.ndarray, message: str
) -> None:
    topology = _topology()
    topology[field] = value
    with pytest.raises(ValueError if field == "ut_loc" else TypeError, match=message):
        _preflight_numpy_topology(topology)


def test_numpy_topology_preflight_skips_torch_tensor_inputs() -> None:
    topology = {name: torch.as_tensor(value) for name, value in _topology().items()}
    _preflight_numpy_topology(topology)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ut_loc", torch.zeros((0, 1, 3)), r"shape \[positive batch, 1, 3\]"),
        ("ut_orientations", torch.zeros((1, 2, 3)), r"shape \[batch, 1, 3\]"),
        ("in_state", torch.zeros((1, 1, 1), dtype=torch.bool), r"shape \[batch, 1\]"),
        ("los", torch.ones((1, 1), dtype=torch.bool), r"shape \[batch, 1, 1\]"),
    ],
)
def test_torch_topology_shapes_are_checked_before_native_import(
    field: str, value: torch.Tensor, message: str
) -> None:
    topology = {name: torch.as_tensor(item) for name, item in _topology().items()}
    topology[field] = value
    with pytest.raises(ValueError, match=message):
        _call(topology=topology)


def test_missing_sionna_runtime_fails_after_software_topology_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sionna.phy.channel.tr38901", None)
    with pytest.raises(RuntimeError, match="Sionna 2.2 PHY is required"):
        _call()


def test_missing_sionna_version_metadata_has_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import PackageNotFoundError

    import integrations.sionna.scenario_gbsm as scenario_gbsm

    monkeypatch.setattr(
        scenario_gbsm,
        "sionna_phy_version",
        lambda: (_ for _ in ()).throw(PackageNotFoundError("sionna")),
    )
    with pytest.raises(RuntimeError, match="version metadata is required"):
        _call()


def test_unsupported_sionna_version_stops_before_native_model_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import integrations.sionna.scenario_gbsm as scenario_gbsm

    monkeypatch.setattr(scenario_gbsm, "sionna_phy_version", lambda: "3.0")
    with pytest.raises(RuntimeError, match="requires Sionna 2.2.x"):
        _call()


def test_topology_preflight_requires_every_explicit_field() -> None:
    topology = _topology()
    del topology["los"]
    with pytest.raises(ValueError, match="topology fields must match"):
        _call(topology=topology)


def test_mapping_topology_option_is_accepted_by_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mapping, rather than dict, is the public contract. Stop at the runtime edge.
    monkeypatch.setitem(sys.modules, "torch", None)
    topology = MappingProxy(_topology())
    with pytest.raises(RuntimeError, match="PyTorch is required|Sionna 2.2 PHY is required"):
        _call(topology=topology)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (torch.tensor([True]), TypeError, "real numeric"),
        (torch.tensor([1 + 2j]), TypeError, "real numeric"),
        (np.array(["bad"]), TypeError, "real numeric"),
        (np.array([np.nan]), ValueError, "finite values"),
        (torch.tensor([float("inf")]), ValueError, "finite values"),
    ],
)
def test_topology_tensor_rejects_nonreal_or_nonfinite_values(value, error, message) -> None:
    with pytest.raises(error, match=message):
        _topology_tensor(torch, "field", value, device="cpu")


def test_topology_tensor_converts_numpy_and_torch_inputs() -> None:
    assert _topology_tensor(torch, "numpy", np.array([1]), device="cpu").dtype == torch.float32
    assert (
        _topology_tensor(torch, "torch", torch.tensor([1.0]), device="cpu").dtype == torch.float32
    )


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (torch.tensor([1]), TypeError, "Boolean dtype"),
        (np.array([1]), TypeError, "Boolean dtype"),
    ],
)
def test_topology_state_rejects_nonboolean_values(value, error, message) -> None:
    with pytest.raises(error, match=message):
        _topology_state(torch, "flag", value, device="cpu")


def test_topology_state_converts_boolean_numpy_and_torch_values() -> None:
    assert _topology_state(torch, "numpy", np.array([True]), device="cpu").dtype == torch.bool
    assert _topology_state(torch, "torch", torch.tensor([False]), device="cpu").dtype == torch.bool


def test_normalized_native_array_layout_records_pattern_major_ports() -> None:
    array = SimpleNamespace(
        normalized_positions=SimpleNamespace(
            x=np.array([0.0, 0.5]),
            y=np.array([0.0, 0.0]),
            z=np.array([0.0, 0.0]),
        ),
        array_size=2,
        num_ant=4,
        antenna_pattern=SimpleNamespace(patterns=(object(), object())),
    )
    layout = _native_array_layout(array, polarization="dual", polarization_type="VH")
    assert layout["pattern_labels_in_port_order"] == ["V", "V", "H", "H"]
    assert layout["port_order"] == "pattern_major_element_minor"


def test_normalized_native_array_layout_converts_torch_position_components() -> None:
    array = SimpleNamespace(
        normalized_positions=SimpleNamespace(
            x=torch.tensor([0.0, 0.5]),
            y=torch.tensor([0.0, 0.0]),
            z=torch.tensor([0.0, 0.0]),
        ),
        array_size=2,
        num_ant=2,
        antenna_pattern=SimpleNamespace(patterns=(object(),)),
    )
    layout = _native_array_layout(array, polarization="single", polarization_type="V")
    assert layout["normalized_element_positions_wavelengths"][1] == [0.5, 0.0, 0.0]


def test_normalized_native_array_layout_converts_numpy_method_components() -> None:
    array = SimpleNamespace(
        normalized_positions=SimpleNamespace(
            x=_NumpyValue(np.array([0.0])),
            y=_NumpyValue(np.array([0.0])),
            z=_NumpyValue(np.array([0.0])),
        ),
        array_size=1,
        num_ant=1,
        antenna_pattern=SimpleNamespace(patterns=(object(),)),
    )
    layout = _native_array_layout(array, polarization="single", polarization_type="V")
    assert layout["normalized_element_positions_wavelengths"] == [[0.0, 0.0, 0.0]]


class _NumpyValue:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


def test_native_array_layout_converts_numpy_method_objects() -> None:
    array = SimpleNamespace(
        ant_pos=_NumpyValue(np.array([[0.1, 0.0, 0.0]])),
        lambda_0=_NumpyValue(np.array([0.1])),
        num_ant=1,
    )
    layout = _native_array_layout(array, polarization="single", polarization_type="V")
    assert layout["normalized_element_positions_wavelengths"] == [[1.0, 0.0, 0.0]]


def test_dual_native_array_layout_converts_torch_polarization_indices() -> None:
    array = SimpleNamespace(
        ant_pos=np.zeros((2, 3)),
        lambda_0=0.5,
        num_ant=2,
        ant_ind_pol1=torch.tensor([0]),
        ant_ind_pol2=torch.tensor([1]),
    )
    layout = _native_array_layout(array, polarization="dual", polarization_type="VH")
    assert layout["pattern_labels_in_port_order"] == ["V", "H"]


@pytest.mark.parametrize(
    ("array", "polarization", "polarization_type", "error", "message"),
    [
        (
            SimpleNamespace(
                normalized_positions=SimpleNamespace(x=[0.0, 1.0], y=[0.0, 0.0], z=[0.0, 0.0]),
                array_size=2,
                num_ant=3,
                antenna_pattern=SimpleNamespace(patterns=(object(), object())),
            ),
            "dual",
            "VH",
            TypeError,
            "lacks the pinned normalized layout API",
        ),
        (
            SimpleNamespace(
                normalized_positions=SimpleNamespace(x=[0.0], y=[0.0], z=[0.0]),
                array_size=1,
                num_ant=1,
                antenna_pattern=SimpleNamespace(patterns=(object(),)),
            ),
            "dual",
            "VH",
            TypeError,
            "lacks the pinned normalized layout API",
        ),
        (
            SimpleNamespace(
                normalized_positions=SimpleNamespace(x=[0.0, 1.0], y=[0.0, 0.0], z=[0.0, 0.0]),
                array_size=1,
                num_ant=2,
                antenna_pattern=SimpleNamespace(patterns=(object(), object())),
            ),
            "dual",
            "VH",
            ValueError,
            "normalized positions have an invalid shape",
        ),
    ],
)
def test_normalized_array_layout_rejects_pattern_and_position_mismatches(
    array, polarization, polarization_type, error, message
) -> None:
    with pytest.raises(error, match=message):
        _native_array_layout(array, polarization=polarization, polarization_type=polarization_type)


def test_native_array_layout_supports_sionna_22_positions() -> None:
    array = SimpleNamespace(
        ant_pos=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        lambda_0=np.array([0.1]),
        num_ant=2,
    )
    layout = _native_array_layout(array, polarization="single", polarization_type="V")
    assert layout["normalized_element_positions_wavelengths"] == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert layout["pattern_labels_in_port_order"] == ["V", "V"]


def test_dual_native_array_layout_maps_explicit_polarization_indices() -> None:
    array = SimpleNamespace(
        ant_pos=np.zeros((4, 3)),
        _lambda_0=0.5,
        num_ant=4,
        ant_ind_pol1=np.array([0, 1]),
        ant_ind_pol2=np.array([2, 3]),
    )
    layout = _native_array_layout(array, polarization="dual", polarization_type="cross")
    assert layout["pattern_labels_in_port_order"] == ["+45", "+45", "-45", "-45"]


@pytest.mark.parametrize(
    "array",
    [
        SimpleNamespace(ant_pos=np.zeros((1, 3)), num_ant=1),
        SimpleNamespace(ant_pos=np.zeros((1, 3)), lambda_0=np.array([0.0]), num_ant=1),
        SimpleNamespace(ant_pos=np.zeros((1, 3)), lambda_0=np.array([0.2, 0.3]), num_ant=1),
        SimpleNamespace(ant_pos=np.zeros((1, 3)), lambda_0=np.array([0.2]), num_ant=2),
        SimpleNamespace(
            ant_pos=np.zeros((2, 3)),
            lambda_0=np.array([0.2]),
            num_ant=2,
            ant_ind_pol1=np.array([0]),
            ant_ind_pol2=np.array([0]),
        ),
    ],
)
def test_native_array_layout_rejects_incomplete_native_layout(array) -> None:
    with pytest.raises(TypeError, match="lacks the pinned normalized layout API"):
        _native_array_layout(array, polarization="dual", polarization_type="VH")


class MappingProxy(Mapping):
    """Minimal read-only Mapping wrapper for topology inputs."""

    def __init__(self, values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _install_fake_sionna(monkeypatch: pytest.MonkeyPatch, h_path, tau_s) -> None:
    import integrations.sionna.scenario_gbsm as scenario_gbsm

    class FakePanelArray:
        def __init__(self, **_kwargs):
            self.normalized_positions = SimpleNamespace(x=[0.0], y=[0.0], z=[0.0])
            self.array_size = 1
            self.num_ant = 1
            self.antenna_pattern = SimpleNamespace(patterns=(object(),))

    class FakeScenario:
        def __init__(self, **_kwargs):
            pass

        def set_topology(self, **_kwargs):
            pass

        def __call__(self, **_kwargs):
            return h_path, tau_s

    fake_module = ModuleType("sionna.phy.channel.tr38901")
    for name in ("InF", "InH", "RMa", "UMa", "UMi"):
        setattr(fake_module, name, FakeScenario)
    fake_module.PanelArray = FakePanelArray
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(scenario_gbsm, "sionna_phy_version", lambda: "2.2.0")


def _fake_model_output():
    return (
        torch.ones((1, 1, 1, 1, 1, 2, 1), dtype=torch.complex64),
        torch.full((1, 1, 1, 2), 1e-9, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("h_path", "tau_s", "message"),
    [
        (torch.ones((1, 1)), torch.ones((1, 1, 1, 1)), "unexpected path/delay ranks"),
        (
            torch.ones((2, 1, 1, 1, 1, 2, 1), dtype=torch.complex64),
            torch.ones((1, 1, 1, 2)),
            "changed the declared topology batch size",
        ),
        (
            torch.ones((1, 2, 1, 1, 1, 2, 1), dtype=torch.complex64),
            torch.ones((1, 1, 1, 2)),
            "more than one BS/UT link",
        ),
        (
            torch.ones((1, 1, 1, 1, 1, 3, 1), dtype=torch.complex64),
            torch.ones((1, 1, 1, 2)),
            "axes do not align",
        ),
        (
            torch.ones((1, 1, 1, 1, 1, 2, 1)),
            torch.ones((1, 1, 1, 2)),
            "non-finite paths or delays",
        ),
        (
            torch.ones((1, 1, 1, 1, 1, 2, 1), dtype=torch.complex64),
            torch.full((1, 1, 1, 2), -1.0),
            "non-finite paths or delays",
        ),
    ],
)
def test_scenario_model_output_contract_is_checked_with_fake_model(
    monkeypatch: pytest.MonkeyPatch, h_path: torch.Tensor, tau_s: torch.Tensor, message: str
) -> None:
    _install_fake_sionna(monkeypatch, h_path, tau_s)
    with pytest.raises(ValueError, match=message):
        _call()


def test_scenario_cfr_rejects_nonfinite_sum_from_finite_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h_path = torch.full((1, 1, 1, 1, 1, 2, 1), 1e308 + 0j, dtype=torch.complex128)
    tau_s = torch.zeros((1, 1, 1, 2), dtype=torch.float64)
    _install_fake_sionna(monkeypatch, h_path, tau_s)
    with pytest.raises(ValueError, match="scenario CFR contains non-finite values"):
        _call()
