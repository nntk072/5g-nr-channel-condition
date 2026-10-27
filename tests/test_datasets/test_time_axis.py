import numpy as np
import pytest
from datasets.time_axis import canonical_time_axis


def test_time_axis_accepts_numpy_real_scalars_and_normalizes_to_float():
    assert canonical_time_axis((np.int64(0), np.float64(0.5)), name="times") == (0.0, 0.5)


@pytest.mark.parametrize("value", ["0.5", 0.5 + 0j, True])
def test_time_axis_rejects_coercible_nonreal_values(value):
    with pytest.raises(TypeError, match="real numeric timestamps"):
        canonical_time_axis((value,), name="times")


@pytest.mark.parametrize("values", [(), (0.0, 0.0), (0.0, float("inf"))])
def test_time_axis_requires_nonempty_finite_strictly_increasing_values(values):
    with pytest.raises(ValueError):
        canonical_time_axis(values, name="times")
