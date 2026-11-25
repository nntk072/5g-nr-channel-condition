import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from datasets.path_oracle import PathSolveStatus
from integrations.sionna.path_oracle import sionna_paths_to_oracle_table


class _InteractionType:
    NONE = 0
    SPECULAR = 1
    DIFFUSE = 2
    REFRACTION = 4
    DIFFRACTION = 8
    SENSING = 16


def _install_interaction_constants(monkeypatch):
    sionna = ModuleType("sionna")
    rt = ModuleType("sionna.rt")
    constants = ModuleType("sionna.rt.constants")
    constants.InteractionType = _InteractionType
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", rt)
    monkeypatch.setitem(sys.modules, "sionna.rt.constants", constants)


def _paths():
    return SimpleNamespace(
        valid=np.array([[[True, True]]], dtype=bool),
        a=(np.array([[[[[1.0, 0.5]]]]]), np.array([[[[[0.25, -0.5]]]]])),
        tau=np.array([[[1e-8, 2e-8]]], dtype=np.float64),
        doppler=np.array([[[1.0, -2.0]]], dtype=np.float64),
        theta_t=np.array([[[0.2, 0.4]]], dtype=np.float64),
        phi_t=np.array([[[3.5, -3.5]]], dtype=np.float64),
        theta_r=np.array([[[0.6, 0.8]]], dtype=np.float64),
        phi_r=np.array([[[1.0, 2.0]]], dtype=np.float64),
        interactions=np.array([[[[0, 1]]], [[[0, 8]]]], dtype=np.int64),
        objects=np.array([[[[-1, 4]]], [[[-1, 5]]]], dtype=np.int64),
        primitives=np.array([[[[-1, 9]]], [[[-1, 10]]]], dtype=np.int64),
        vertices=np.zeros((2, 1, 1, 2, 3), dtype=np.float64),
    )


class _NumpyField:
    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array


def test_sionna_paths_adapter_converts_path_and_interaction_contract(monkeypatch):
    _install_interaction_constants(monkeypatch)
    table = sionna_paths_to_oracle_table(
        _paths(), link_id="link-1", frame_index=3, solve_status=PathSolveStatus.SOLVED
    )

    assert table.link_ids == ("link-1",)
    assert table.frame_indices.tolist() == [3]
    assert table.solve_status == (PathSolveStatus.SOLVED,)
    np.testing.assert_array_equal(table.delay_s, [1e-8, 2e-8])
    np.testing.assert_array_equal(table.complex_gain, [1 + 0.25j, 0.5 - 0.5j])
    np.testing.assert_array_equal(table.interaction_order, [0, 2])
    np.testing.assert_allclose(table.aoa_azimuth_elevation_rad[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(
        table.aoa_azimuth_elevation_rad[:, 1], np.pi / 2 - np.array([0.6, 0.8])
    )
    np.testing.assert_allclose(
        table.aod_azimuth_elevation_rad[:, 0], [3.5 - 2 * np.pi, -3.5 + 2 * np.pi]
    )
    assert table.path_kind == ("direct", "specular_reflection+diffraction")
    assert table.interaction_types.tolist() == [1, 8]
    assert table.object_ids.tolist() == [4, 5]
    assert table.primitive_ids.tolist() == [9, 10]
    assert table.interaction_vertices_m.shape == (2, 3)


def test_sionna_paths_adapter_materializes_numpy_fields_and_labels_unknown_bits(monkeypatch):
    _install_interaction_constants(monkeypatch)
    paths = _paths()
    paths.valid = _NumpyField(paths.valid)
    paths.interactions = paths.interactions.copy()
    paths.interactions[0, 0, 0, 1] = 64
    table = sionna_paths_to_oracle_table(paths, link_id="link-1")
    assert table.path_kind == ("direct", "interaction_mask_64+diffraction")


def test_sionna_paths_adapter_accepts_per_port_path_axes(monkeypatch):
    _install_interaction_constants(monkeypatch)
    paths = _paths()
    for name in ("valid", "tau", "doppler", "theta_t", "phi_t", "theta_r", "phi_r"):
        setattr(paths, name, getattr(paths, name)[:, None, :, None, :])
    table = sionna_paths_to_oracle_table(paths, link_id="per-port")
    assert table.group_count == 1
    assert table.path_count == 2


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"link_id": "  "}, ValueError, "link_id"),
        ({"frame_index": True}, ValueError, "frame_index"),
        ({"rx_device_index": -1}, ValueError, "rx_device_index"),
        ({"tx_device_index": 1}, IndexError, "device index"),
        ({"solve_status": "solved"}, TypeError, "PathSolveStatus"),
        ({"rx_port_map": ()}, ValueError, "rx_port_map"),
        ({"rx_port_map": (0, 0)}, ValueError, "duplicate"),
        ({"rx_port_map": (1,)}, IndexError, "outside"),
        ({"tx_port_map": (True,)}, ValueError, "tx_port_map"),
    ],
)
def test_sionna_paths_adapter_rejects_invalid_selection(changes, error, message, monkeypatch):
    _install_interaction_constants(monkeypatch)
    arguments = {"link_id": "link-1"}
    arguments.update(changes)
    with pytest.raises(error, match=message):
        sionna_paths_to_oracle_table(_paths(), **arguments)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("valid", np.ones((1, 1, 2), dtype=np.int8), TypeError, "Boolean dtype"),
        (
            "a",
            (np.ones((1, 1, 1, 1, 2), dtype=np.int64), np.ones((1, 1, 1, 1, 2))),
            TypeError,
            "real floating dtype",
        ),
        ("tau", np.ones((1, 1, 2), dtype=np.complex128), TypeError, "real floating dtype"),
        ("tau", np.ones((2, 3, 2), dtype=np.float64), ValueError, "cannot broadcast"),
        ("tau", np.ones((1, 1), dtype=np.float64), ValueError, "device-pair or per-port"),
        ("interactions", np.ones((1, 1, 1, 1), dtype=np.float64), TypeError, "integer dtype"),
        ("vertices", np.ones((1, 1, 1, 2, 2), dtype=np.float64), ValueError, "XYZ axes"),
        ("interactions", np.ones((1, 1, 1), dtype=np.int64), ValueError, "depth plus path axes"),
        ("interactions", np.ones((1, 2, 1, 2), dtype=np.int64), ValueError, "do not match"),
        (
            "vertices",
            np.ones((1, 2, 1, 1, 1, 2, 3), dtype=np.float64),
            ValueError,
            "do not match",
        ),
        ("a", (np.ones((1, 1, 1, 1, 2)),), ValueError, "real and imaginary"),
        (
            "a",
            (np.ones((1, 1, 1, 1, 2)), np.ones((1, 1, 1, 1, 1))),
            ValueError,
            "matching five-axis",
        ),
    ],
)
def test_sionna_paths_adapter_rejects_invalid_path_fields(
    field, value, error, message, monkeypatch
):
    _install_interaction_constants(monkeypatch)
    paths = _paths()
    setattr(paths, field, value)
    with pytest.raises(error, match=message):
        sionna_paths_to_oracle_table(paths, link_id="link-1")


def test_sionna_paths_adapter_keeps_empty_groups_and_infers_no_path(monkeypatch):
    _install_interaction_constants(monkeypatch)
    paths = _paths()
    paths.valid[:] = False
    table = sionna_paths_to_oracle_table(paths, link_id="empty")
    assert table.solve_status == (PathSolveStatus.NO_PATH,)
    assert table.path_offsets.tolist() == [0, 0]
    assert table.path_kind == ()
    assert table.interaction_vertices_m.shape == (0, 3)


def test_sionna_paths_adapter_requires_runtime_interaction_constants(monkeypatch):
    for name in ("sionna", "sionna.rt", "sionna.rt.constants"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(RuntimeError, match="Sionna RT is required"):
        sionna_paths_to_oracle_table(_paths(), link_id="link-1")
