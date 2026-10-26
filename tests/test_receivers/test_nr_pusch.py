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
    assert config.n_size_grid == 24


def test_pusch_config_defaults_to_sionna_receiver_linear_interpolation():
    assert NRPUSCHConfig().interpolation_type == "lin"


def test_pusch_config_accepts_bwp_offset_inside_wider_carrier_grid():
    config = NRPUSCHConfig(
        n_size_bwp=6,
        n_size_grid=24,
        n_start_grid=2,
        n_start_bwp=7,
    )
    assert config.n_start_bwp - config.n_start_grid == 5


def test_pusch_config_keeps_legacy_positional_field_order():
    config = NRPUSCHConfig(6, 15)
    assert config.n_size_bwp == 6
    assert config.subcarrier_spacing_khz == 15
    assert config.carrier_grid_size == 6


def test_pusch_config_normalizes_json_symbol_allocation_to_immutable_tuple():
    config = NRPUSCHConfig(mapping_type="B", symbol_allocation=[3, 10])
    assert config.symbol_allocation == (3, 10)
    assert isinstance(config.symbol_allocation, tuple)


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
        {"n_cell_id": 1008},
        {"dmrs_n_id": (1,)},
        {"dmrs_n_id": (1, 65536)},
        {"dmrs_n_scid": 2},
        {"mapping_type": "C"},
        {"symbol_allocation": (3,)},
        {"symbol_allocation": (False, 10)},
        {"symbol_allocation": (-1, 10)},
        {"symbol_allocation": (3, 12)},
        {"mapping_type": "A", "symbol_allocation": (1, 13)},
        {"precoding": "unknown"},
        {"interpolation_type": "spline"},
        {"n_size_bwp": True},
        {"n_size_grid": 0},
        {"n_size_grid": 276},
        {"n_size_grid": True},
        {"n_start_grid": -1},
        {"n_start_grid": True},
        {"n_start_bwp": -1},
        {"n_start_bwp": 2474},
        {"n_start_bwp": 1, "n_start_grid": 2},
        {"n_size_bwp": 6, "n_size_grid": 24, "n_start_bwp": 19},
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
