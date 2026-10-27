from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from datasets.path_oracle import PathSolveStatus
from integrations.sionna.path_oracle import sionna_paths_to_oracle_table


def _install_interaction_constants(monkeypatch) -> None:
    sionna = ModuleType("sionna")
    rt = ModuleType("sionna.rt")
    constants = ModuleType("sionna.rt.constants")
    constants.InteractionType = SimpleNamespace(
        NONE=0,
        SPECULAR=1,
        DIFFUSE=2,
        REFRACTION=4,
        DIFFRACTION=8,
        SENSING=16,
    )
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", rt)
    monkeypatch.setitem(sys.modules, "sionna.rt.constants", constants)


def _paths(*, valid_dtype=np.bool_, interaction_dtype=np.int32):
    real = np.asarray([[[[[0.5, np.nan]]]]], dtype=np.float32)
    imag = np.asarray([[[[[0.25, np.nan]]]]], dtype=np.float32)
    return SimpleNamespace(
        valid=np.asarray([[[True, False]]], dtype=valid_dtype),
        a=(real, imag),
        tau=np.asarray([[[1e-8, np.nan]]], dtype=np.float64),
        doppler=np.asarray([[[0.0, np.nan]]], dtype=np.float32),
        theta_t=np.asarray([[[np.pi / 2, np.nan]]], dtype=np.float32),
        phi_t=np.asarray([[[0.0, np.nan]]], dtype=np.float32),
        theta_r=np.asarray([[[np.pi / 2, np.nan]]], dtype=np.float32),
        phi_r=np.asarray([[[0.0, np.nan]]], dtype=np.float32),
        interactions=np.asarray([[[[1, 0]]]], dtype=interaction_dtype),
        objects=np.asarray([[[[4, -1]]]], dtype=np.int32),
        primitives=np.asarray([[[[7, -1]]]], dtype=np.int32),
        vertices=np.asarray([[[[[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]]]]], dtype=np.float32),
    )


def test_paths_adapter_checks_field_dtypes_and_ignores_invalid_padding(monkeypatch):
    _install_interaction_constants(monkeypatch)

    table = sionna_paths_to_oracle_table(_paths(), link_id="tx-rx")

    assert table.path_count == 1
    assert table.group_count == 1
    assert table.solve_status == (PathSolveStatus.SOLVED,)
    assert table.path_kind == ("specular_reflection",)
    np.testing.assert_array_equal(table.complex_gain, np.asarray([0.5 + 0.25j]))
    np.testing.assert_array_equal(table.interaction_vertices_m, np.asarray([[1.0, 2.0, 3.0]]))


def test_paths_adapter_rejects_non_boolean_valid_mask(monkeypatch):
    _install_interaction_constants(monkeypatch)

    with pytest.raises(TypeError, match="Paths.valid must use a Boolean dtype"):
        sionna_paths_to_oracle_table(_paths(valid_dtype=np.uint8), link_id="tx-rx")


def test_paths_adapter_rejects_complex_delay_components(monkeypatch):
    _install_interaction_constants(monkeypatch)
    paths = _paths()
    paths.tau = paths.tau.astype(np.complex64)

    with pytest.raises(TypeError, match="Paths.tau must use a real floating dtype"):
        sionna_paths_to_oracle_table(paths, link_id="tx-rx")


def test_paths_adapter_rejects_noninteger_interaction_slots(monkeypatch):
    _install_interaction_constants(monkeypatch)
    paths = _paths(interaction_dtype=np.float32)

    with pytest.raises(TypeError, match="Paths.interactions must use an integer dtype"):
        sionna_paths_to_oracle_table(paths, link_id="tx-rx")
