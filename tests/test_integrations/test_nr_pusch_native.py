import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.noise import proper_complex_awgn  # noqa: E402
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from receivers.nr import NRPUSCHConfig  # noqa: E402


def _pusch_dmrs_gold_reference(
    *,
    cell_id: int,
    dmrs_n_id: tuple[int, int] | None,
    n_scid: int,
    slot: int,
    symbol: int,
    count: int,
) -> torch.Tensor:
    """Independent TS 38.211 5.2.1/6.4.1.1.1 Gold-sequence reference.

    For transform-precoding-disabled PUSCH, use
    ``c_init = 2^17(14*n_slot + l + 1)(2*N_ID^nSCID + 1)
    + 2*N_ID^nSCID + n_SCID (mod 2^31)`` and map consecutive Gold chips to
    ``(1-2*c(2n) + j*(1-2*c(2n+1)))/sqrt(2)``.
    """
    n_id = cell_id if dmrs_n_id is None else dmrs_n_id[n_scid]
    c_init = (2**17 * (14 * slot + symbol + 1) * (2 * n_id + 1) + 2 * n_id + n_scid) % 2**31
    n_c = 1600
    sequence_length = 2 * count
    state_length = n_c + sequence_length + 31
    x1 = [0] * state_length
    x2 = [0] * state_length
    x1[0] = 1
    for index in range(31):
        x2[index] = (c_init >> index) & 1
    for index in range(n_c + sequence_length):
        x1[index + 31] = (x1[index + 3] + x1[index]) & 1
        x2[index + 31] = (x2[index + 3] + x2[index + 2] + x2[index + 1] + x2[index]) & 1
    chips = [(x1[index + n_c] + x2[index + n_c]) & 1 for index in range(sequence_length)]
    scale = 1.0 / math.sqrt(2.0)
    return torch.tensor(
        [
            complex(scale * (1 - 2 * chips[2 * index]), scale * (1 - 2 * chips[2 * index + 1]))
            for index in range(count)
        ],
        dtype=torch.complex128,
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("subcarrier_spacing_khz", "n_size_bwp"),
    ((15, 2), (30, 6), (60, 12), (120, 24)),
)
def test_native_pusch_grid_axes_and_fft_mapping_across_numerologies(
    subcarrier_spacing_khz: int, n_size_bwp: int
) -> None:
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
        ),
        device="cpu",
    )
    grid = receiver.resource_grid
    effective_bins = torch.as_tensor(grid.effective_subcarrier_ind, dtype=torch.long)
    pilot_mask = receiver.pilot_mask

    assert grid.fft_size == 12 * n_size_bwp
    assert grid.num_effective_subcarriers == grid.fft_size
    torch.testing.assert_close(effective_bins, torch.arange(grid.fft_size))
    assert tuple(pilot_mask.shape) == (
        1,
        1,
        grid.num_ofdm_symbols,
        grid.num_effective_subcarriers,
    )
    assert bool(pilot_mask.any())
    assert bool((~pilot_mask.bool()).any())
    assert receiver.metadata["resource_grid"]["fft_size"] == 12 * n_size_bwp
    assert receiver.metadata["resource_grid"]["subcarrier_spacing_hz"] == (
        subcarrier_spacing_khz * 1_000
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("mapping_type", "additional_position", "expected_symbols"),
    (
        ("A", 0, (2,)),
        ("A", 1, (2, 11)),
        ("A", 2, (2, 7, 11)),
        ("A", 3, (2, 5, 8, 11)),
        ("B", 0, (0,)),
        ("B", 1, (0, 10)),
        ("B", 2, (0, 5, 10)),
        ("B", 3, (0, 3, 6, 9)),
    ),
)
def test_pusch_type1_dmrs_positions_match_ts_38211_full_slot_reference(
    mapping_type: str, additional_position: int, expected_symbols: tuple[int, ...]
) -> None:
    """Check TS 38.211 V19.1.0 Table 6.4.1.1.3-3 for mapping A and B."""
    n_size_bwp = 2
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=30,
            dmrs_config_type=1,
            dmrs_length=1,
            dmrs_additional_position=additional_position,
            num_cdm_groups_without_data=1,
            mapping_type=mapping_type,
        ),
        device="cpu",
    )
    mask = receiver.pilot_mask[0, 0].bool()
    actual_symbols = tuple(
        int(index) for index in torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
    )
    expected_subcarriers = tuple(
        12 * rb + offset for rb in range(n_size_bwp) for offset in (0, 2, 4, 6, 8, 10)
    )

    assert actual_symbols == expected_symbols
    for symbol in expected_symbols:
        actual_subcarriers = tuple(
            int(index) for index in torch.nonzero(mask[symbol], as_tuple=False).flatten()
        )
        assert actual_subcarriers == expected_subcarriers


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("mapping_type", "additional_position", "expected_symbols", "symbol_allocation"),
    (
        ("A", 0, (2,), (0, 10)),
        ("A", 1, (2, 9), (0, 10)),
        ("A", 2, (2, 6, 9), (0, 10)),
        ("A", 3, (2, 6, 9), (0, 10)),
        ("B", 0, (0,), (3, 10)),
        ("B", 1, (0, 8), (3, 10)),
        ("B", 2, (0, 4, 8), (3, 10)),
        ("B", 3, (0, 3, 6, 9), (3, 10)),
    ),
)
def test_pusch_type2_single_symbol_partial_allocation_matches_ts_38211(
    mapping_type: str,
    additional_position: int,
    expected_symbols: tuple[int, ...],
    symbol_allocation: tuple[int, int],
) -> None:
    """Check Type-2 DM-RS timing/comb for 10-symbol A/B allocations."""
    n_size_bwp = 2
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=30,
            dmrs_config_type=2,
            dmrs_length=1,
            dmrs_additional_position=additional_position,
            num_cdm_groups_without_data=1,
            mapping_type=mapping_type,
            symbol_allocation=symbol_allocation,
        ),
        device="cpu",
    )
    mask = receiver.pilot_mask[0, 0].bool()
    actual_symbols = tuple(
        int(index) for index in torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
    )
    expected_subcarriers = tuple(
        12 * rb + offset for rb in range(n_size_bwp) for offset in (0, 1, 6, 7)
    )

    assert actual_symbols == expected_symbols
    assert receiver.metadata["symbol_allocation"] == symbol_allocation
    for symbol in expected_symbols:
        actual_subcarriers = tuple(
            int(index) for index in torch.nonzero(mask[symbol], as_tuple=False).flatten()
        )
        assert actual_subcarriers == expected_subcarriers


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("mapping_type", "additional_position", "expected_symbols"),
    (
        ("A", 0, (2, 3)),
        ("A", 1, (2, 3, 10, 11)),
        ("B", 0, (0, 1)),
        ("B", 1, (0, 1, 9, 10)),
    ),
)
def test_pusch_double_symbol_dmrs_positions_match_ts_38211_full_slot_reference(
    mapping_type: str, additional_position: int, expected_symbols: tuple[int, ...]
) -> None:
    """Check double-symbol positions from TS 38.211 V19.1.0 Table 6.4.1.1.3-4."""
    n_size_bwp = 2
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=30,
            dmrs_config_type=1,
            dmrs_length=2,
            dmrs_additional_position=additional_position,
            num_cdm_groups_without_data=1,
            mapping_type=mapping_type,
        ),
        device="cpu",
    )
    mask = receiver.pilot_mask[0, 0].bool()
    actual_symbols = tuple(
        int(index) for index in torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
    )
    expected_subcarriers = tuple(
        12 * rb + offset for rb in range(n_size_bwp) for offset in (0, 2, 4, 6, 8, 10)
    )

    assert actual_symbols == expected_symbols
    for symbol in expected_symbols:
        actual_subcarriers = tuple(
            int(index) for index in torch.nonzero(mask[symbol], as_tuple=False).flatten()
        )
        assert actual_subcarriers == expected_subcarriers


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("additional_position", "expected_symbols"),
    ((0, (0,)), (1, (0, 8)), (2, (0, 4, 8)), (3, (0, 3, 6, 9))),
)
def test_pusch_mapping_b_partial_allocation_dmrs_positions_match_ts_38211(
    additional_position: int, expected_symbols: tuple[int, ...]
) -> None:
    """Check allocation-relative mapping-B DM-RS for a 10-symbol PUSCH."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            subcarrier_spacing_khz=30,
            dmrs_additional_position=additional_position,
            mapping_type="B",
            symbol_allocation=(3, 10),
        ),
        device="cpu",
    )
    mask = receiver.pilot_mask[0, 0].bool()
    actual_symbols = tuple(
        int(index) for index in torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
    )

    assert actual_symbols == expected_symbols
    assert receiver.metadata["symbol_allocation"] == (3, 10)
    transmitted_grid, _ = receiver.transmit(batch_size=1, seed=17)
    assert transmitted_grid.shape[-2] == 10
    estimate = receiver.estimate(transmitted_grid, noise_variance=0.0)
    assert estimate.h_hat.shape[-2] == 10


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("additional_position", "expected_symbols"),
    (
        (0, (2,)),
        (1, (2, 9)),
        (2, (2, 6, 9)),
        (3, (2, 6, 9)),
    ),
)
def test_pusch_mapping_a_short_allocation_dmrs_positions_match_ts_38211(
    additional_position: int, expected_symbols: tuple[int, ...]
) -> None:
    """Check mapping-A positions for l_d=10 in TS 38.211 Table 6.4.1.1.3-3."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            subcarrier_spacing_khz=30,
            dmrs_additional_position=additional_position,
            mapping_type="A",
            symbol_allocation=(0, 10),
        ),
        device="cpu",
    )
    mask = receiver.pilot_mask[0, 0].bool()
    actual_symbols = tuple(
        int(index) for index in torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
    )

    assert actual_symbols == expected_symbols
    assert receiver.metadata["symbol_allocation"] == (0, 10)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("mapping_type", "additional_position", "expected_symbols"),
    (
        ("A", 0, (2, 3)),
        ("A", 1, (2, 3, 10, 11)),
        ("B", 0, (0, 1)),
        ("B", 1, (0, 1, 9, 10)),
    ),
)
def test_pusch_type2_double_symbol_dmrs_time_positions_match_ts_38211(
    mapping_type: str, additional_position: int, expected_symbols: tuple[int, ...]
) -> None:
    """Check Type-2 double-symbol timing and frequency comb against TS 38.211."""
    n_size_bwp = 2
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=30,
            dmrs_config_type=2,
            dmrs_length=2,
            dmrs_additional_position=additional_position,
            num_cdm_groups_without_data=1,
            mapping_type=mapping_type,
        ),
        device="cpu",
    )
    mask = receiver.pilot_mask[0, 0].bool()
    actual_symbols = tuple(
        int(index) for index in torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
    )
    expected_subcarriers = tuple(
        12 * rb + offset for rb in range(n_size_bwp) for offset in (0, 1, 6, 7)
    )

    assert actual_symbols == expected_symbols
    for symbol in expected_symbols:
        actual_subcarriers = tuple(
            int(index) for index in torch.nonzero(mask[symbol], as_tuple=False).flatten()
        )
        assert actual_subcarriers == expected_subcarriers


@pytest.mark.native_sionna
def test_pusch_type1_two_port_dmrs_pilots_are_orthogonal() -> None:
    """Check Type-1 two-stream CDM orthogonality from TS 38.211 Table 6.4.1.1.3-1."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=4,
            subcarrier_spacing_khz=30,
            num_antenna_ports=2,
            num_layers=2,
            dmrs_config_type=1,
            num_cdm_groups_without_data=2,
        ),
        device="cpu",
    )
    pattern = receiver.resource_grid.pilot_pattern
    masks = receiver.pilot_mask[0].bool()
    pilots = pattern.pilots[0].to(torch.complex128)

    assert tuple(masks.shape) == (2, 14, 48)
    assert torch.equal(masks[0], masks[1])
    assert tuple(pilots.shape) == (2, 48)
    assert bool((torch.linalg.vector_norm(pilots, dim=-1) > 0).all())
    torch.testing.assert_close(
        torch.vdot(pilots[0], pilots[1]),
        torch.zeros((), dtype=torch.complex128),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize("num_layers", (2, 4))
def test_pusch_type2_multiport_dmrs_pilots_are_orthogonal(num_layers: int) -> None:
    """Check Type-2 multi-stream CDM pilots and frequency groups against TS 38.211."""
    n_size_bwp = 4
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=30,
            num_antenna_ports=num_layers,
            num_layers=num_layers,
            dmrs_config_type=2,
            dmrs_length=1,
            dmrs_additional_position=0,
            num_cdm_groups_without_data=2,
        ),
        device="cpu",
    )
    pattern = receiver.resource_grid.pilot_pattern
    masks = receiver.pilot_mask[0].bool()
    pilots = pattern.pilots[0].to(torch.complex128)

    assert tuple(masks.shape) == (num_layers, 14, 12 * n_size_bwp)
    assert torch.equal(masks, masks[0].expand_as(masks))
    expected_subcarriers = tuple(
        12 * rb + offset for rb in range(n_size_bwp) for offset in (0, 1, 2, 3, 6, 7, 8, 9)
    )
    actual_subcarriers = tuple(
        int(index) for index in torch.nonzero(masks[0, 2], as_tuple=False).flatten()
    )
    assert actual_subcarriers == expected_subcarriers
    assert tuple(pilots.shape) == (num_layers, len(expected_subcarriers))
    assert bool((torch.linalg.vector_norm(pilots, dim=-1) > 0).all())
    gram = pilots @ pilots.conj().T
    diagonal = torch.diag(torch.diagonal(gram))
    torch.testing.assert_close(
        gram - diagonal,
        torch.zeros_like(gram),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.native_sionna
def test_type2_four_layer_estimator_and_decoder_recover_known_flat_mimo_channel() -> None:
    """Estimate and decode four layers through a known flat 4x4 MIMO channel."""
    from sionna.phy.nr import PUSCHReceiver

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=4,
            subcarrier_spacing_khz=30,
            num_antenna_ports=4,
            num_layers=4,
            dmrs_config_type=2,
            dmrs_length=1,
            dmrs_additional_position=0,
            num_cdm_groups_without_data=2,
        ),
        device="cpu",
    )
    transmitted_grid, transmitted_bits = receiver.transmit(batch_size=1, seed=7)
    channel = torch.tensor(
        [
            [1.0 + 0.0j, 0.2 + 0.1j, 0.0 - 0.1j, 0.3 + 0.0j],
            [0.0 + 0.1j, 0.8 + 0.0j, 0.2 + 0.0j, 0.1 - 0.1j],
            [0.2 + 0.0j, 0.0 + 0.1j, 0.9 + 0.0j, 0.15 + 0.0j],
            [0.05 + 0.0j, 0.2 + 0.0j, 0.0 + 0.1j, 1.1 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    received_grid = torch.einsum("rl,btlsf->btrsf", channel, transmitted_grid)

    estimate = receiver.estimate(received_grid, noise_variance=0.0)
    estimated_channel = estimate.h_hat[0, 0, :, 0]

    assert tuple(estimated_channel.shape) == (4, 4, 14, 48)
    torch.testing.assert_close(
        estimated_channel,
        channel[:, :, None, None].expand_as(estimated_channel),
        rtol=0,
        atol=1e-6,
    )

    channel_truth = channel.reshape(1, 1, 4, 1, 4, 1, 1).expand(
        1,
        1,
        4,
        1,
        4,
        receiver.resource_grid.num_ofdm_symbols,
        receiver.resource_grid.fft_size,
    )
    decoder = PUSCHReceiver(
        receiver._transmitter,
        channel_estimator="perfect",
        return_tb_crc_status=True,
        input_domain="freq",
        device="cpu",
    )
    decoded_bits, crc_ok = decoder(
        received_grid,
        torch.tensor(1e-7),
        h=channel_truth,
    )
    torch.testing.assert_close(decoded_bits, transmitted_bits)
    assert bool(crc_ok.all())


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("num_cdm_groups", "offsets_per_rb"),
    (
        (1, (0, 1, 6, 7)),
        (2, (0, 1, 2, 3, 6, 7, 8, 9)),
        (3, tuple(range(12))),
    ),
)
def test_pusch_type2_dmrs_frequency_offsets_match_ts_38211_table(
    num_cdm_groups: int, offsets_per_rb: tuple[int, ...]
) -> None:
    """Check Type-2 CDM-group frequency offsets from TS 38.211 Table 6.4.1.1.3-2."""
    n_size_bwp = 3
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=30,
            dmrs_config_type=2,
            dmrs_length=1,
            dmrs_additional_position=0,
            num_cdm_groups_without_data=num_cdm_groups,
        ),
        device="cpu",
    )
    mask = receiver.pilot_mask[0, 0].bool()
    actual_symbols = tuple(
        int(index) for index in torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
    )
    expected_subcarriers = tuple(
        12 * rb + offset for rb in range(n_size_bwp) for offset in offsets_per_rb
    )

    assert actual_symbols == (2,)
    actual_subcarriers = tuple(
        int(index) for index in torch.nonzero(mask[2], as_tuple=False).flatten()
    )
    assert actual_subcarriers == expected_subcarriers


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("cell_id", "dmrs_n_id", "n_scid", "dmrs_config_type"),
    ((1, None, 0, 1), (37, (17, 29), 1, 1), (81, None, 0, 2), (91, (7, 13), 1, 2)),
)
def test_pusch_dmrs_values_match_independent_ts_38211_gold_sequence(
    cell_id: int, dmrs_n_id: tuple[int, int] | None, n_scid: int, dmrs_config_type: int
) -> None:
    """Validate actual Sionna DMRS pilots against TS 38.211 Gold equations."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            subcarrier_spacing_khz=30,
            n_cell_id=cell_id,
            dmrs_n_id=dmrs_n_id,
            dmrs_n_scid=n_scid,
            dmrs_config_type=dmrs_config_type,
        ),
        device="cpu",
    )
    pilots = receiver.resource_grid.pilot_pattern.pilots[0, 0]
    assert receiver.metadata["n_cell_id"] == cell_id
    assert receiver.metadata["dmrs_n_id"] == dmrs_n_id
    assert receiver.metadata["dmrs_n_scid"] == n_scid
    expected = _pusch_dmrs_gold_reference(
        cell_id=cell_id,
        dmrs_n_id=dmrs_n_id,
        n_scid=n_scid,
        slot=0,
        symbol=2,
        count=int(pilots.numel()),
    )

    torch.testing.assert_close(pilots.to(torch.complex128), expected, rtol=0, atol=1e-7)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("subcarrier_spacing_khz", "n_size_bwp"),
    ((15, 2), (30, 6), (60, 12), (120, 24)),
)
def test_pusch_dmrs_estimator_recovers_frequency_selective_noiseless_channel_on_cpu(
    subcarrier_spacing_khz: int, n_size_bwp: int
) -> None:
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
        ),
        device="cpu",
    )
    grid = receiver.resource_grid
    transmit_grid, _ = receiver.transmit(2, seed=5302)
    fft_bins = torch.arange(grid.fft_size)
    # Distinct adjacent FFT-bin gains expose a one-bin or DC/guard mapping shift.
    channel_on_fft_bins = torch.where(
        (fft_bins % 2) == 0,
        torch.tensor(0.25 + 0.4j, dtype=torch.complex64),
        torch.tensor(0.8 - 0.15j, dtype=torch.complex64),
    )
    received_grid = transmit_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel_on_fft_bins.reshape(
        1, 1, 1, 1, grid.fft_size
    )

    observation = receiver.observe("cpu-noiseless-frequency-selective", received_grid, 0.0)
    effective_bins = torch.as_tensor(grid.effective_subcarrier_ind, dtype=torch.long)
    expected = channel_on_fft_bins.index_select(0, effective_bins)
    expected = expected.reshape(1, 1, 1, 1, 1, 1, -1).expand_as(observation.channel_estimate.h_hat)
    pilot_mask = observation.channel_estimate.pilot_mask.to(dtype=torch.bool).reshape(
        1, 1, 1, 1, 1, grid.num_ofdm_symbols, grid.num_effective_subcarriers
    )
    pilot_mask = pilot_mask.expand_as(observation.channel_estimate.h_hat)

    assert observation.fft_bin_indices == tuple(int(index) for index in effective_bins)
    assert grid.num_effective_subcarriers == grid.fft_size
    assert observation.fft_bin_indices == tuple(range(grid.fft_size))
    assert bool(pilot_mask.any())
    torch.testing.assert_close(
        observation.channel_estimate.h_hat[pilot_mask], expected[pilot_mask], atol=1e-5, rtol=1e-5
    )


@pytest.mark.native_sionna
def test_pusch_ls_pilot_estimate_is_unbiased_with_analytical_awgn_variance_on_cpu():
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30), device="cpu"
    )
    batch_size = 1024
    transmit_grid, _ = receiver.transmit(batch_size, seed=4801)
    channel = torch.tensor(0.6 - 0.2j, dtype=torch.complex64)
    clean_grid = transmit_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel
    complex_noise_variance = 0.05
    noise_generator = torch.Generator().manual_seed(15)
    noise = (
        torch.complex(
            torch.randn(clean_grid.shape, generator=noise_generator),
            torch.randn(clean_grid.shape, generator=noise_generator),
        )
        * (complex_noise_variance / 2.0) ** 0.5
    )
    estimate = receiver.estimate(clean_grid + noise, complex_noise_variance)

    pilot_symbol, pilot_subcarrier = (
        int(index) for index in torch.nonzero(receiver.pilot_mask[0, 0].bool())[0]
    )
    samples = estimate.h_hat[:, 0, 0, 0, 0, pilot_symbol, pilot_subcarrier]
    predicted_variance = estimate.error_variance[0, 0, 0, 0, 0, pilot_symbol, pilot_subcarrier]
    empirical_bias = samples.mean() - channel
    empirical_variance = (samples - samples.mean()).abs().square().mean()

    assert float(empirical_bias.abs()) < 0.02
    assert float(predicted_variance) == pytest.approx(0.025, rel=1e-6)
    assert float(empirical_variance / predicted_variance) == pytest.approx(1.0, abs=0.15)


@pytest.mark.native_sionna
def test_sionna_lmmse_effective_noise_matches_analytical_correlated_noise_fixture():
    from sionna.phy.mimo import lmmse_equalizer

    num_samples = 32_768
    channel = torch.tensor(
        [[1.0 + 0.2j, 0.1 - 0.3j], [0.4 + 0.1j, 0.9 + 0.2j], [0.2 - 0.1j, -0.3 + 0.8j]],
        dtype=torch.complex64,
    )
    noise_covariance = torch.tensor(
        [
            [0.8 + 0j, 0.12 + 0.08j, 0.0 + 0j],
            [0.12 - 0.08j, 0.9 + 0j, 0.05 - 0.04j],
            [0.0 + 0j, 0.05 + 0.04j, 0.7 + 0j],
        ],
        dtype=torch.complex64,
    )
    cholesky = torch.linalg.cholesky(noise_covariance)
    generator = torch.Generator().manual_seed(6917)
    symbols = proper_complex_awgn(
        torch.zeros((num_samples, 2), dtype=torch.complex64), 1.0, generator=generator
    )
    white_noise = proper_complex_awgn(
        torch.zeros((num_samples, 3), dtype=torch.complex64), 1.0, generator=generator
    )
    noise = white_noise @ cholesky.transpose(-2, -1)
    received = symbols @ channel.transpose(-2, -1) + noise

    equalized, effective_variance = lmmse_equalizer(received, channel, noise_covariance)

    inverse_noise_covariance = torch.linalg.inv(noise_covariance)
    linear_mmse_filter = torch.linalg.solve(
        channel.mH @ inverse_noise_covariance @ channel + torch.eye(2, dtype=torch.complex64),
        channel.mH @ inverse_noise_covariance,
    )
    unbiased_filter = linear_mmse_filter / torch.diagonal(linear_mmse_filter @ channel).unsqueeze(
        -1
    )
    expected_equalized = received @ unbiased_filter.transpose(-2, -1)
    residual_channel = unbiased_filter @ channel - torch.eye(2, dtype=torch.complex64)
    analytical_error_covariance = (
        residual_channel @ residual_channel.mH
        + unbiased_filter @ noise_covariance @ unbiased_filter.mH
    )
    analytical_variance = torch.diagonal(analytical_error_covariance).real
    empirical_variance = (equalized - symbols).abs().square().mean(dim=0)

    torch.testing.assert_close(equalized, expected_equalized, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(effective_variance, analytical_variance, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        empirical_variance / analytical_variance,
        torch.ones_like(analytical_variance),
        atol=0.025,
        rtol=0,
    )


@pytest.mark.native_sionna
def test_pusch_dmrs_estimator_runs_on_cuda_with_named_axes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    grid = receiver.resource_grid
    pilot_mask = receiver.pilot_mask[0, 0].bool()
    pilot_symbols = grid.pilot_pattern.pilots[0, 0]
    expected_channel = torch.tensor(0.5 + 0.25j, device="cuda:0")
    received = torch.zeros(
        (2, 1, 2, grid.num_ofdm_symbols, grid.fft_size),
        dtype=torch.complex64,
        device="cuda:0",
    )
    received[..., pilot_mask] = pilot_symbols * expected_channel
    result = receiver.estimate(received, torch.tensor(1e-3, device="cuda:0"))
    observation = receiver.observe("cuda-flat-channel-001", received, torch.tensor(1e-3))

    assert result.h_hat.shape == (
        2,
        1,
        2,
        1,
        1,
        grid.num_ofdm_symbols,
        grid.num_effective_subcarriers,
    )
    assert result.h_hat.is_cuda
    assert result.error_variance.is_cuda
    assert result.pilot_mask.is_cuda
    assert observation.sample_id == "cuda-flat-channel-001"
    assert observation.fft_bin_indices == tuple(
        int(index) for index in grid.effective_subcarrier_ind
    )
    assert observation.model_inputs().received_grid.is_cuda
    assert bool(torch.isfinite(result.h_hat).all())
    assert bool(torch.isfinite(result.error_variance).all())
    assert result.error_variance.shape == (
        1,
        1,
        1,
        1,
        1,
        grid.num_ofdm_symbols,
        grid.num_effective_subcarriers,
    )
    torch.testing.assert_close(
        result.h_hat,
        expected_channel.expand_as(result.h_hat),
        atol=1e-5,
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="dimensions must be positive"):
        receiver.estimate(received[:, :0], torch.tensor(1e-3, device="cuda:0"))

    torch.manual_seed(7)
    clean_grid = torch.zeros(
        (16, 1, 2, grid.num_ofdm_symbols, grid.fft_size),
        dtype=torch.complex64,
        device="cuda:0",
    )
    clean_grid[..., pilot_mask] = pilot_symbols * expected_channel

    def estimate_nmse(noise_variance: float) -> torch.Tensor:
        variance = torch.tensor(noise_variance, device="cuda:0")
        real_noise = torch.randn(clean_grid.shape, device="cuda:0")
        imag_noise = torch.randn(clean_grid.shape, device="cuda:0")
        noisy_grid = clean_grid + torch.complex(real_noise, imag_noise) * torch.sqrt(variance / 2)
        estimate = receiver.estimate(noisy_grid, variance)
        return torch.mean(torch.abs(estimate.h_hat - expected_channel) ** 2)

    low_noise_nmse = estimate_nmse(1e-3)
    high_noise_nmse = estimate_nmse(1e-1)
    assert low_noise_nmse < high_noise_nmse


@pytest.mark.native_sionna
def test_pusch_transmitter_rayleigh_channel_and_estimator_close_the_loop():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from sionna.phy.channel import OFDMChannel, RayleighBlockFading

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    resource_grid = receiver.resource_grid
    transmit_grid, _ = receiver._transmitter(8)
    channel = OFDMChannel(
        RayleighBlockFading(num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=1),
        resource_grid,
        return_channel=True,
        device="cuda:0",
    )
    received_grid, channel_truth = channel(transmit_grid)
    channel_truth = channel_truth[:, :, :, :, 0, :, :].unsqueeze(4)
    noiseless_estimate = receiver.estimate(received_grid, torch.tensor(0.0, device="cuda:0"))

    assert noiseless_estimate.h_hat.shape == channel_truth.shape
    torch.testing.assert_close(noiseless_estimate.h_hat, channel_truth, atol=1e-5, rtol=1e-5)

    torch.manual_seed(11)

    def estimate_nmse(noise_variance: float) -> torch.Tensor:
        variance = torch.tensor(noise_variance, device="cuda:0")
        real_noise = torch.randn(received_grid.shape, device="cuda:0")
        imag_noise = torch.randn(received_grid.shape, device="cuda:0")
        noisy_grid = received_grid + torch.complex(real_noise, imag_noise) * torch.sqrt(
            variance / 2
        )
        estimate = receiver.estimate(noisy_grid, variance)
        return torch.mean(torch.abs(estimate.h_hat - channel_truth) ** 2) / torch.mean(
            torch.abs(channel_truth) ** 2
        )

    assert estimate_nmse(1e-3) < estimate_nmse(1e-1)
