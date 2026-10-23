"""Native Sionna checks for partial PUSCH DM-RS mapping."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from receivers.nr import NRPUSCHConfig  # noqa: E402


@pytest.mark.native_sionna
@pytest.mark.parametrize("dmrs_config_type", (1, 2))
@pytest.mark.parametrize(
    ("additional_position", "expected_symbols"),
    (
        (0, (0,)),
        (1, (0, 6)),
        (2, (0, 3, 6)),
        (3, (0, 3, 6)),
    ),
)
def test_mapping_b_duration_eight_dmrs_matches_ts_38211(
    dmrs_config_type: int,
    additional_position: int,
    expected_symbols: tuple[int, ...],
) -> None:
    """Check the duration-8 row of TS 38.211 V19.1.0 Table 6.4.1.1.3-3."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            subcarrier_spacing_khz=30,
            dmrs_config_type=dmrs_config_type,
            dmrs_additional_position=additional_position,
            mapping_type="B",
            symbol_allocation=(3, 8),
        ),
        device="cpu",
    )
    mask = receiver.pilot_mask[0, 0].bool()
    actual_symbols = tuple(
        int(index) for index in torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
    )
    comb_offsets = (0, 2, 4, 6, 8, 10) if dmrs_config_type == 1 else (0, 1, 6, 7)
    expected_subcarriers = tuple(12 * rb + offset for rb in range(2) for offset in comb_offsets)

    assert actual_symbols == expected_symbols
    assert receiver.metadata["symbol_allocation"] == (3, 8)
    assert receiver.resource_grid.num_ofdm_symbols == 8
    for symbol in expected_symbols:
        actual_subcarriers = tuple(
            int(index) for index in torch.nonzero(mask[symbol], as_tuple=False).flatten()
        )
        assert actual_subcarriers == expected_subcarriers

    transmitted_grid, _ = receiver.transmit(batch_size=1, seed=17)
    estimate = receiver.estimate(transmitted_grid, noise_variance=0.0)
    assert transmitted_grid.shape[-2] == 8
    assert estimate.h_hat.shape[-2] == 8
