import sys
import types

import numpy as np
import pytest
from integrations.sionna.adapter import (
    GridConfig,
    _phy_cir_to_cfr,
    count_valid_paths,
    direct_indirect_diagnostics,
    has_valid_path,
)
from propagation.physics import subcarrier_frequencies


def test_phy_shapes_to_cfr():
    a = np.ones((3, 1, 2, 1, 1, 2, 1), dtype=np.complex64)
    tau = np.zeros((3, 1, 1, 2))
    h = _phy_cir_to_cfr(a, tau, subcarrier_frequencies(32, 30e3))
    assert h.shape == (3, 2, 32)
    assert np.allclose(h, 2)


def test_phy_array_shape_rejected():
    with pytest.raises(ValueError):
        _phy_cir_to_cfr(np.ones((1, 2)), np.ones((1, 2)), np.arange(10))


class MockPaths:
    def __init__(self, valid):
        self.valid = np.asarray(valid)


def test_depth_zero_direct_oracle():
    assert has_valid_path(MockPaths([False, False, True]))
    assert not has_valid_path(MockPaths([False, False]))
    assert count_valid_paths(MockPaths([True, True, False])) == 2


def test_ray_oracle_missing_field_fails():
    with pytest.raises(ValueError):
        has_valid_path(object())


def test_grid_invalid():
    with pytest.raises(ValueError):
        GridConfig(n_subcarriers=4)


def test_rt_cfr_accepts_real_imag_tuple_and_preserves_all_links(monkeypatch):
    import integrations.sionna.adapter as adapter

    dr = types.ModuleType("drjit")
    dr.arange = lambda dtype, count: np.arange(count, dtype=np.float32)
    mi = types.ModuleType("mitsuba")
    mi.Float = np.float32
    monkeypatch.setitem(sys.modules, "drjit", dr)
    monkeypatch.setitem(sys.modules, "mitsuba", mi)

    class TuplePaths:
        def cfr(self, **kwargs):
            assert kwargs["normalize"] is False
            real = np.ones((1, 2, 1, 3, 1, 8), dtype=np.float32)
            return real, np.full_like(real, 0.25)

    result = adapter.rt_cfr(TuplePaths(), GridConfig(n_subcarriers=8))
    assert result.shape == (1, 6, 8)
    assert result.dtype == np.complex64
    np.testing.assert_allclose(result, 1.0 + 0.25j)


@pytest.mark.parametrize(
    ("real", "imaginary", "message"),
    [
        (np.ones((1, 8)), np.ones((1, 7)), "dimension mismatch"),
        (np.ones((1, 8)), None, "complex CFR"),
        (np.full((1, 8), np.nan + 1j), None, "Nonfinite"),
    ],
)
def test_rt_cfr_rejects_malformed_tuple_and_array_results(monkeypatch, real, imaginary, message):
    import integrations.sionna.adapter as adapter

    dr = types.ModuleType("drjit")
    dr.arange = lambda dtype, count: np.arange(count, dtype=np.float32)
    mi = types.ModuleType("mitsuba")
    mi.Float = np.float32
    monkeypatch.setitem(sys.modules, "drjit", dr)
    monkeypatch.setitem(sys.modules, "mitsuba", mi)

    class MalformedPaths:
        def cfr(self, **kwargs):
            if imaginary is None:
                return real
            return real, imaginary

    with pytest.raises(ValueError, match=message):
        adapter.rt_cfr(MalformedPaths(), GridConfig(n_subcarriers=8))


def test_path_validity_helpers_accept_tensor_like_objects():
    class TensorLike:
        def numpy(self):
            return np.asarray([[False, True], [False, False]])

    paths = MockPaths(TensorLike())
    assert has_valid_path(paths)
    assert count_valid_paths(paths) == 1


def test_path_count_rejects_missing_valid_field():
    with pytest.raises(ValueError, match="missing valid field"):
        count_valid_paths(object())


def test_direct_and_reflected_power_diagnostic():
    direct = np.ones((1, 1, 32), dtype=np.complex64)
    full = direct + np.ones_like(direct) * 0.5j
    diag = direct_indirect_diagnostics(full, direct)
    assert abs(diag["oracle_direct_power"] - 1.0) < 1e-6
    assert abs(diag["residual_indirect_power"] - 0.25) < 1e-6
    assert abs(diag["direct_to_indirect_power_db"] - 6.0206) < 1e-3


@pytest.mark.parametrize(
    ("full", "direct"),
    [
        (np.ones(4, dtype=np.complex64), np.ones(3, dtype=np.complex64)),
        (np.ones(4), np.ones(4)),
    ],
)
def test_power_diagnostic_requires_aligned_complex_cfr(full, direct):
    with pytest.raises(ValueError, match="aligned complex CFR"):
        direct_indirect_diagnostics(full, direct)


def test_decomposition_never_used_as_model_feature():
    from propagation.physics import observable_features

    h = np.ones((1, 1, 32), dtype=np.complex64)
    a = direct_indirect_diagnostics(h, h / 2)
    b = direct_indirect_diagnostics(h, h / 3)
    assert a != b
    assert np.array_equal(observable_features(h), observable_features(h))
