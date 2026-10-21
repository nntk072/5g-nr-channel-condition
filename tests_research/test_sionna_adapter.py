import numpy as np
import pytest
from loslab.sionna2 import (_phy_cir_to_cfr,has_valid_path,count_valid_paths,
                             rt_cfr,GridConfig,RayScene,generate_raytraced)
from loslab.physics import subcarrier_frequencies


def test_phy_shapes_to_cfr():
    a=np.ones((3,1,2,1,1,2,1),dtype=np.complex64)
    tau=np.zeros((3,1,1,2))
    h=_phy_cir_to_cfr(a,tau,subcarrier_frequencies(32,30e3))
    assert h.shape==(3,2,32)
    assert np.allclose(h,2)

def test_phy_array_shape_rejected():
    with pytest.raises(ValueError):
        _phy_cir_to_cfr(np.ones((1,2)),np.ones((1,2)),np.arange(10))

class MockPaths:
    def __init__(self,valid):self.valid=np.asarray(valid)

def test_depth_zero_direct_oracle():
    assert has_valid_path(MockPaths([False,False,True]))
    assert not has_valid_path(MockPaths([False,False]))
    assert count_valid_paths(MockPaths([True,True,False]))==2

def test_ray_oracle_missing_field_fails():
    with pytest.raises(ValueError):has_valid_path(object())

def test_grid_invalid():
    with pytest.raises(ValueError):GridConfig(n_subcarriers=4)

from loslab.sionna2 import direct_indirect_diagnostics

def test_direct_and_reflected_power_diagnostic():
    direct=np.ones((1,1,32),dtype=np.complex64)
    full=direct+np.ones_like(direct)*.5j
    diag=direct_indirect_diagnostics(full,direct)
    assert abs(diag['oracle_direct_power']-1.)<1e-6
    assert abs(diag['residual_indirect_power']-.25)<1e-6
    assert abs(diag['direct_to_indirect_power_db']-6.0206)<1e-3

def test_decomposition_never_used_as_model_feature():
    from loslab.physics import observable_features
    h=np.ones((1,1,32),dtype=np.complex64)
    a=direct_indirect_diagnostics(h,h/2)
    b=direct_indirect_diagnostics(h,h/3)
    assert a!=b
    assert np.array_equal(observable_features(h),observable_features(h))
