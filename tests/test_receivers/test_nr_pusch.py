import numpy as np
import pytest
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def test_pusch_config_accepts_valid_dmrs_configuration():
    config = NRPUSCHConfig(
        n_size_bwp=24,
        subcarrier_spacing_khz=30,
        dmrs_length=2,
        dmrs_additional_position=1,
        num_cdm_groups_without_data=2,
    )
    assert config.n_size_bwp == 24


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_size_bwp": 0},
        {"subcarrier_spacing_khz": 20},
        {"num_antenna_ports": 3},
        {"num_layers": 2},
        {"dmrs_config_type": 3},
        {"dmrs_length": 3},
        {"dmrs_length": 2, "dmrs_additional_position": 2},
        {"num_cdm_groups_without_data": 0},
        {"mapping_type": "C"},
        {"precoding": "unknown"},
        {"interpolation_type": "spline"},
        {"n_size_bwp": True},
    ],
)
def test_pusch_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        NRPUSCHConfig(**kwargs)


def test_nr_channel_estimate_preserves_named_axis_contract():
    h_hat = np.zeros((2, 1, 1, 1, 1, 14, 24), dtype=np.complex64)
    result = NRChannelEstimate(
        h_hat=h_hat,
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        pilot_mask=np.zeros((1, 1, 14, 24), dtype=np.int32),
        metadata={"sionna_version": "2.2.0"},
    )
    assert result.h_hat.shape == (2, 1, 1, 1, 1, 14, 24)


def test_nr_channel_estimate_rejects_non_broadcastable_variance():
    h_hat = np.zeros((2, 1, 1, 1, 1, 14, 24), dtype=np.complex64)
    with pytest.raises(ValueError, match="broadcastable"):
        NRChannelEstimate(
            h_hat=h_hat,
            error_variance=np.zeros((3, 1, 1, 1, 1, 14, 24), dtype=np.float32),
            pilot_mask=np.zeros((1, 1, 14, 24), dtype=np.int32),
            metadata={"sionna_version": "2.2.0"},
        )
