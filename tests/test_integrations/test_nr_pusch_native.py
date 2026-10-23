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
def test_offset_bwp_maps_into_carrier_fft_bins_and_decodes() -> None:
    """Map a 6-RB BWP at CRB 7 into a grid starting at CRB 2, then decode."""
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import OFDMDemodulator

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=6,
            n_size_grid=24,
            n_start_grid=2,
            n_start_bwp=7,
            subcarrier_spacing_khz=30,
            mapping_type="B",
            symbol_allocation=(3, 10),
        ),
        device="cpu",
    )
    bwp_grid, transmitted_bits = receiver.transmit(batch_size=1, seed=60421)
    carrier_slot_grid = receiver.place_bwp_allocation_in_carrier_slot(bwp_grid)
    active_bins = torch.nonzero(
        carrier_slot_grid.abs().sum(dim=(0, 1, 2, 3)) > 0,
        as_tuple=False,
    ).flatten()
    expected_bins = torch.arange(5 * 12, (5 + 6) * 12)
    assert receiver.resource_grid.fft_size == 6 * 12
    assert carrier_slot_grid.shape[-1] == 24 * 12
    torch.testing.assert_close(active_bins, expected_bins)
    assert receiver.metadata["carrier_grid"]["bwp_start_rb_in_grid"] == 5

    waveform = receiver.modulate_carrier_slot_grid(carrier_slot_grid)
    cp_samples = round(
        int(receiver.resource_grid.cyclic_prefix_length)
        * receiver.config.carrier_grid_size
        / receiver.config.n_size_bwp
    )
    demodulator = OFDMDemodulator(
        fft_size=24 * 12,
        l_min=0,
        cyclic_prefix_length=cp_samples,
        device="cpu",
    )
    recovered_carrier_grid = demodulator(waveform)
    torch.testing.assert_close(recovered_carrier_grid, carrier_slot_grid, rtol=1e-5, atol=1e-6)
    recovered_bwp_grid = receiver.extract_bwp_allocation_from_carrier_slot(recovered_carrier_grid)

    decoder = PUSCHReceiver(
        receiver._transmitter,
        return_tb_crc_status=True,
        input_domain="freq",
        device="cpu",
    )
    decoded_bits, crc_ok = decoder(recovered_bwp_grid, torch.tensor(1e-7))
    torch.testing.assert_close(decoded_bits, transmitted_bits)
    assert bool(crc_ok.all())


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("n_size_bwp", "bwp_start_in_grid"),
    ((4, 0), (6, 5), (12, 12), (4, 20)),
)
@pytest.mark.parametrize("subcarrier_spacing_khz", (15, 30, 60, 120))
def test_bwp_allocation_mapping_roundtrips_at_carrier_grid_edges(
    n_size_bwp: int, bwp_start_in_grid: int, subcarrier_spacing_khz: int
) -> None:
    """Preserve allocation bins and OFDM round-trips across offsets and numerologies."""
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import OFDMDemodulator

    start_symbol, duration = 3, 10
    n_size_grid = 24
    n_start_grid = 11
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            n_size_grid=n_size_grid,
            n_start_grid=n_start_grid,
            n_start_bwp=n_start_grid + bwp_start_in_grid,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
            mapping_type="B",
            symbol_allocation=(start_symbol, duration),
        ),
        device="cpu",
    )
    allocation_grid, transmitted_bits = receiver.transmit(
        batch_size=1,
        seed=60421 + 1000 * subcarrier_spacing_khz + 100 * n_size_bwp + bwp_start_in_grid,
    )

    carrier_slot_grid = receiver.place_bwp_allocation_in_carrier_slot(allocation_grid)
    start_bin = 12 * bwp_start_in_grid
    stop_bin = start_bin + 12 * n_size_bwp
    active_bins = torch.nonzero(
        carrier_slot_grid.abs().sum(dim=(0, 1, 2, 3)) > 0,
        as_tuple=False,
    ).flatten()

    torch.testing.assert_close(active_bins, torch.arange(start_bin, stop_bin))
    torch.testing.assert_close(
        receiver.extract_bwp_allocation_from_carrier_slot(carrier_slot_grid),
        allocation_grid,
    )
    waveform = receiver.modulate_carrier_slot_grid(carrier_slot_grid)
    cp_samples = round(int(receiver.resource_grid.cyclic_prefix_length) * n_size_grid / n_size_bwp)
    demodulator = OFDMDemodulator(
        fft_size=12 * n_size_grid,
        l_min=0,
        cyclic_prefix_length=cp_samples,
        device="cpu",
    )
    recovered_carrier_grid = demodulator(waveform)
    torch.testing.assert_close(recovered_carrier_grid, carrier_slot_grid, rtol=1e-5, atol=1e-6)
    recovered_allocation = receiver.extract_bwp_allocation_from_carrier_slot(recovered_carrier_grid)
    decoder = PUSCHReceiver(
        receiver._transmitter,
        return_tb_crc_status=True,
        input_domain="freq",
        device="cpu",
    )
    decoded_bits, crc_ok = decoder(recovered_allocation, torch.tensor(1e-7))
    torch.testing.assert_close(decoded_bits, transmitted_bits)
    assert bool(crc_ok.all())
    assert torch.count_nonzero(carrier_slot_grid[..., :start_bin]) == 0
    assert torch.count_nonzero(carrier_slot_grid[..., stop_bin:]) == 0
    assert torch.count_nonzero(carrier_slot_grid[..., :start_symbol, :]) == 0
    assert torch.count_nonzero(carrier_slot_grid[..., start_symbol + duration :, :]) == 0


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
    ("mapping_type", "symbol_allocation", "additional_position", "expected_symbols"),
    (
        ("A", (0, 10), 0, (2, 3)),
        ("A", (0, 10), 1, (2, 3, 8, 9)),
        ("B", (3, 10), 0, (0, 1)),
        ("B", (3, 10), 1, (0, 1, 7, 8)),
    ),
)
def test_pusch_type2_double_symbol_partial_allocation_matches_ts_38211(
    mapping_type: str,
    symbol_allocation: tuple[int, int],
    additional_position: int,
    expected_symbols: tuple[int, ...],
) -> None:
    """Check partial double-symbol DM-RS positions from TS 38.211 Table 6.4.1.1.3-4."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            subcarrier_spacing_khz=30,
            dmrs_config_type=2,
            dmrs_length=2,
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
    expected_subcarriers = tuple(12 * rb + offset for rb in range(2) for offset in (0, 1, 6, 7))

    assert actual_symbols == expected_symbols
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
def test_partial_mapping_b_pusch_allocation_places_and_decodes_at_slot_offset() -> None:
    """Place, OFDM-modulate, demodulate, and decode a scheduled partial allocation."""
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import OFDMDemodulator

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=6,
            subcarrier_spacing_khz=30,
            dmrs_additional_position=1,
            mapping_type="B",
            symbol_allocation=(3, 10),
        ),
        device="cpu",
    )
    allocation_grid, transmitted_bits = receiver.transmit(batch_size=1, seed=9031)

    slot_grid = receiver.place_allocation_in_slot(allocation_grid)
    assert slot_grid.shape[-2] == 14
    assert not bool(slot_grid[..., :3, :].any())
    assert not bool(slot_grid[..., 13:, :].any())

    waveform = receiver.modulate_slot_grid(slot_grid)
    cyclic_prefix_length = int(receiver.resource_grid.cyclic_prefix_length)
    expected_samples = 14 * (receiver.resource_grid.fft_size + cyclic_prefix_length)
    assert waveform.shape[-1] == expected_samples
    demodulator = OFDMDemodulator(
        fft_size=receiver.resource_grid.fft_size,
        l_min=0,
        cyclic_prefix_length=cyclic_prefix_length,
        device="cpu",
    )
    recovered_slot_grid = demodulator(waveform)
    torch.testing.assert_close(recovered_slot_grid, slot_grid, rtol=1e-5, atol=1e-6)
    received_allocation = receiver.extract_allocation_from_slot(recovered_slot_grid)
    torch.testing.assert_close(received_allocation, allocation_grid, rtol=1e-5, atol=1e-6)

    decoder = PUSCHReceiver(
        receiver._transmitter,
        return_tb_crc_status=True,
        input_domain="freq",
        device="cpu",
    )
    decoded_bits, crc_ok = decoder(received_allocation, torch.tensor(1e-7))
    torch.testing.assert_close(decoded_bits, transmitted_bits)
    assert bool(crc_ok.all())


@pytest.mark.native_sionna
def test_partial_mapping_b_slot_waveform_decodes_static_in_cp_echo() -> None:
    """Decode a scheduled slot waveform after a seeded two-tap in-CP channel."""
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import OFDMDemodulator

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=6,
            subcarrier_spacing_khz=30,
            dmrs_additional_position=1,
            mapping_type="B",
            symbol_allocation=(3, 10),
        ),
        device="cpu",
    )
    allocation_grid, transmitted_bits = receiver.transmit(batch_size=1, seed=9031)
    slot_grid = receiver.place_allocation_in_slot(allocation_grid)
    transmit_waveform = receiver.modulate_slot_grid(slot_grid)
    received_waveform = transmit_waveform.clone()
    echo_delay_samples = 2
    echo_gain = 0.2 + 0.1j
    received_waveform[..., echo_delay_samples:] += (
        echo_gain * transmit_waveform[..., :-echo_delay_samples]
    )
    noise_variance = 1e-5
    generator = torch.Generator(device="cpu").manual_seed(5221)
    received_waveform += proper_complex_awgn(
        received_waveform,
        noise_variance,
        generator=generator,
    )

    demodulator = OFDMDemodulator(
        fft_size=receiver.resource_grid.fft_size,
        l_min=0,
        cyclic_prefix_length=int(receiver.resource_grid.cyclic_prefix_length),
        device="cpu",
    )
    received_slot_grid = demodulator(received_waveform)
    received_allocation = receiver.extract_allocation_from_slot(received_slot_grid)
    decoder = PUSCHReceiver(
        receiver._transmitter,
        return_tb_crc_status=True,
        input_domain="freq",
        device="cpu",
    )
    decoded_bits, crc_ok = decoder(received_allocation, torch.tensor(noise_variance))

    assert echo_delay_samples < receiver.resource_grid.cyclic_prefix_length
    torch.testing.assert_close(decoded_bits, transmitted_bits)
    assert bool(crc_ok.all())


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
def test_partial_mapping_b_dmrs_sequence_uses_absolute_slot_symbol_index() -> None:
    """A nonzero mapping-B allocation start must affect the TS 38.211 DM-RS seed."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            subcarrier_spacing_khz=30,
            n_cell_id=37,
            dmrs_n_id=(17, 29),
            dmrs_n_scid=1,
            dmrs_config_type=1,
            dmrs_additional_position=0,
            mapping_type="B",
            symbol_allocation=(3, 10),
        ),
        device="cpu",
    )
    pilots = receiver.resource_grid.pilot_pattern.pilots[0, 0].to(torch.complex128)
    expected_absolute_symbol = _pusch_dmrs_gold_reference(
        cell_id=37,
        dmrs_n_id=(17, 29),
        n_scid=1,
        slot=0,
        symbol=3,
        count=int(pilots.numel()),
    )
    incorrect_allocation_relative_symbol = _pusch_dmrs_gold_reference(
        cell_id=37,
        dmrs_n_id=(17, 29),
        n_scid=1,
        slot=0,
        symbol=0,
        count=int(pilots.numel()),
    )

    torch.testing.assert_close(pilots, expected_absolute_symbol, rtol=0, atol=1e-7)
    assert not torch.equal(pilots, incorrect_allocation_relative_symbol)
    assert receiver.resource_grid.num_ofdm_symbols == 10
    assert receiver.metadata["symbol_allocation"] == (3, 10)


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
def test_pusch_linear_dmrs_interpolation_tracks_analytical_two_tap_channel_on_all_res():
    """Check linear DM-RS interpolation against known delay-domain channel truth."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=4,
            subcarrier_spacing_khz=30,
            interpolation_type="lin",
        ),
        device="cpu",
    )
    grid = receiver.resource_grid
    transmitted_grid, _ = receiver.transmit(batch_size=1, seed=5_302)
    centered_bins = torch.arange(grid.fft_size, dtype=torch.float32) - grid.fft_size // 2
    direct = 0.9 + 0.1j
    echo = 0.25 - 0.18j
    delay_samples = 3
    channel = direct + echo * torch.exp(
        -2j * torch.pi * centered_bins * delay_samples / grid.fft_size
    )
    received_grid = transmitted_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel.reshape(
        1, 1, 1, 1, grid.fft_size
    )

    estimate = receiver.estimate(received_grid, noise_variance=0.0).h_hat
    expected = channel.reshape(1, 1, 1, 1, 1, 1, grid.fft_size).expand_as(estimate)
    normalized_mse = torch.mean(torch.abs(estimate - expected) ** 2) / torch.mean(
        torch.abs(expected) ** 2
    )
    nearest_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=4,
            subcarrier_spacing_khz=30,
            interpolation_type="nn",
        ),
        device="cpu",
    )
    nearest_estimate = nearest_receiver.estimate(received_grid, noise_variance=0.0).h_hat
    nearest_nmse = torch.mean(torch.abs(nearest_estimate - expected) ** 2) / torch.mean(
        torch.abs(expected) ** 2
    )

    assert tuple(estimate.shape[-2:]) == (grid.num_ofdm_symbols, grid.fft_size)
    assert float(normalized_mse) < 0.02
    assert float(normalized_mse) < float(nearest_nmse)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    (
        "dmrs_config_type",
        "dmrs_length",
        "dmrs_additional_position",
        "delay_samples",
    ),
    (
        (1, 1, 0, 3),
        (1, 1, 1, 3),
        (1, 2, 0, 5),
        (2, 1, 0, 5),
        (2, 1, 1, 5),
        (2, 2, 0, 5),
    ),
)
@pytest.mark.parametrize("complex_noise_variance", (0.01, 0.05, 0.2))
def test_pusch_interpolation_bias_variance_tradeoff_matches_monte_carlo(
    dmrs_config_type: int,
    dmrs_length: int,
    dmrs_additional_position: int,
    delay_samples: int,
    complex_noise_variance: float,
) -> None:
    """Compare interpolation bias and reported uncertainty on paired channels."""
    common_config = {
        "n_size_bwp": 4,
        "subcarrier_spacing_khz": 30,
        "dmrs_config_type": dmrs_config_type,
        "dmrs_length": dmrs_length,
        "dmrs_additional_position": dmrs_additional_position,
        "num_cdm_groups_without_data": 1,
    }
    linear_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(**common_config, interpolation_type="lin"), device="cpu"
    )
    nearest_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(**common_config, interpolation_type="nn"), device="cpu"
    )
    assert (
        linear_receiver.metadata["error_variance_semantics"]
        == "sionna_linear_interpolated_marginal_variance_not_exact_covariance"
    )
    assert (
        nearest_receiver.metadata["error_variance_semantics"]
        == "sionna_nearest_neighbor_pilot_error_variance"
    )
    batch_size = 1024
    transmitted_grid, _ = linear_receiver.transmit(batch_size, seed=73491)
    fft_size = linear_receiver.resource_grid.fft_size
    centered_bins = torch.arange(fft_size, dtype=torch.float32) - fft_size // 2
    channel = (
        0.9
        + 0.1j
        + (0.25 - 0.18j) * torch.exp(-2j * torch.pi * centered_bins * delay_samples / fft_size)
    )
    clean_grid = transmitted_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel.reshape(
        1, 1, 1, 1, fft_size
    )
    noise_generator = torch.Generator().manual_seed(23917)
    noise = (
        torch.complex(
            torch.randn(clean_grid.shape, generator=noise_generator),
            torch.randn(clean_grid.shape, generator=noise_generator),
        )
        * (complex_noise_variance / 2.0) ** 0.5
    )

    linear_noisy = linear_receiver.estimate(clean_grid + noise, complex_noise_variance)
    nearest_noisy = nearest_receiver.estimate(clean_grid + noise, complex_noise_variance)
    linear_reference = linear_receiver.estimate(clean_grid[:1], 0.0).h_hat[0, 0, 0, 0, 0]
    nearest_reference = nearest_receiver.estimate(clean_grid[:1], 0.0).h_hat[0, 0, 0, 0, 0]
    true_channel = channel.reshape(1, fft_size).expand_as(linear_reference)
    linear_bias_nmse = torch.mean(torch.abs(linear_reference - true_channel) ** 2) / torch.mean(
        torch.abs(true_channel) ** 2
    )
    nearest_bias_nmse = torch.mean(torch.abs(nearest_reference - true_channel) ** 2) / torch.mean(
        torch.abs(true_channel) ** 2
    )

    ratios = []
    for noisy_estimate in (linear_noisy, nearest_noisy):
        samples = noisy_estimate.h_hat[:, 0, 0, 0, 0]
        empirical_mean = samples.mean(dim=0)
        empirical_variance = (samples - empirical_mean).abs().square().mean(dim=0)
        predicted_variance = noisy_estimate.error_variance[0, 0, 0, 0, 0]
        valid = predicted_variance > 0
        reference = linear_reference if noisy_estimate is linear_noisy else nearest_reference
        mean_bias = (empirical_mean - reference).abs()
        bias_tolerance = 6.0 * torch.sqrt(predicted_variance / batch_size)
        assert bool(valid.any())
        assert bool((mean_bias[valid] < bias_tolerance[valid]).all())
        ratios.append(empirical_variance[valid].sum() / predicted_variance[valid].sum())

    # The linear estimate should reduce two-tap channel bias in this bounded
    # resolution regime; nearest-neighbor variance remains the calibrated mode.
    assert float(linear_bias_nmse) < float(nearest_bias_nmse)
    assert float(ratios[0]) <= 1.05
    assert float(ratios[0]) >= 0.6
    assert float(ratios[1]) == pytest.approx(1.0, abs=0.12)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("doppler_hz", "dmrs_additional_position"),
    ((35.0, 1), (350.0, 1), (35.0, 2), (350.0, 2)),
)
def test_pusch_interpolation_tracks_cp_timed_doppler_with_declared_variance(
    doppler_hz: float, dmrs_additional_position: int
) -> None:
    """Check temporal interpolation on a symbol-static Doppler CFR model."""
    from integrations.sionna.nr_timing import nr_normal_cp_symbol_start_times

    common_config = {
        "n_size_bwp": 4,
        "subcarrier_spacing_khz": 30,
        "dmrs_additional_position": dmrs_additional_position,
    }
    linear_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(**common_config, interpolation_type="lin"), device="cpu"
    )
    nearest_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(**common_config, interpolation_type="nn"), device="cpu"
    )
    batch_size = 1024
    noise_variance = 0.05
    transmitted_grid, _ = linear_receiver.transmit(batch_size, seed=88107)
    fft_size = linear_receiver.resource_grid.fft_size
    centered_bins = torch.arange(fft_size, dtype=torch.float32) - fft_size // 2
    symbol_times = torch.tensor(nr_normal_cp_symbol_start_times(30_000.0, 14), dtype=torch.float32)
    delayed_echo = (0.25 - 0.18j) * torch.exp(-2j * torch.pi * centered_bins * 3 / fft_size)
    channel = (0.9 + 0.1j) + torch.exp(2j * torch.pi * doppler_hz * symbol_times).reshape(
        14, 1
    ) * delayed_echo.reshape(1, fft_size)
    clean_grid = transmitted_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel.reshape(
        1, 1, 1, 14, fft_size
    )
    noise_generator = torch.Generator().manual_seed(19733)
    noise = (
        torch.complex(
            torch.randn(clean_grid.shape, generator=noise_generator),
            torch.randn(clean_grid.shape, generator=noise_generator),
        )
        * (noise_variance / 2.0) ** 0.5
    )

    linear_noisy = linear_receiver.estimate(clean_grid + noise, noise_variance)
    nearest_noisy = nearest_receiver.estimate(clean_grid + noise, noise_variance)
    linear_reference = linear_receiver.estimate(clean_grid[:1], 0.0).h_hat[0, 0, 0, 0, 0]
    nearest_reference = nearest_receiver.estimate(clean_grid[:1], 0.0).h_hat[0, 0, 0, 0, 0]
    linear_nmse = torch.mean(torch.abs(linear_reference - channel) ** 2) / torch.mean(
        torch.abs(channel) ** 2
    )
    nearest_nmse = torch.mean(torch.abs(nearest_reference - channel) ** 2) / torch.mean(
        torch.abs(channel) ** 2
    )

    ratios = []
    for noisy_estimate, reference in (
        (linear_noisy, linear_reference),
        (nearest_noisy, nearest_reference),
    ):
        samples = noisy_estimate.h_hat[:, 0, 0, 0, 0]
        empirical_mean = samples.mean(dim=0)
        empirical_variance = (samples - empirical_mean).abs().square().mean(dim=0)
        predicted_variance = noisy_estimate.error_variance[0, 0, 0, 0, 0]
        valid = predicted_variance > 0
        mean_bias = (empirical_mean - reference).abs()
        bias_tolerance = 6.0 * torch.sqrt(predicted_variance / batch_size)
        assert bool(valid.any())
        assert bool((mean_bias[valid] < bias_tolerance[valid]).all())
        ratios.append(empirical_variance[valid].sum() / predicted_variance[valid].sum())

    assert float(linear_nmse) < float(nearest_nmse)
    assert 0.6 <= float(ratios[0]) <= 1.05
    assert float(ratios[1]) == pytest.approx(1.0, abs=0.12)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("dmrs_config_type", "dmrs_length", "num_cdm_groups_without_data", "n_size_bwp"),
    (
        (1, 1, 1, 2),
        (1, 1, 2, 4),
        (2, 1, 1, 4),
        (2, 1, 2, 6),
        (1, 2, 1, 4),
        (2, 2, 1, 4),
    ),
)
@pytest.mark.parametrize("complex_noise_variance", (0.01, 0.05, 0.2))
def test_pusch_ls_pilot_bias_and_variance_match_analysis_across_dmrs_and_snr(
    dmrs_config_type: int,
    dmrs_length: int,
    num_cdm_groups_without_data: int,
    n_size_bwp: int,
    complex_noise_variance: float,
) -> None:
    """Check all pilot REs against their analytical LS variance across regimes."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=30,
            dmrs_config_type=dmrs_config_type,
            dmrs_length=dmrs_length,
            num_cdm_groups_without_data=num_cdm_groups_without_data,
        ),
        device="cpu",
    )
    batch_size = 1024
    transmit_grid, _ = receiver.transmit(batch_size, seed=4801)
    channel = torch.tensor(0.6 - 0.2j, dtype=torch.complex64)
    clean_grid = transmit_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel
    noise_generator = torch.Generator().manual_seed(15)
    noise = (
        torch.complex(
            torch.randn(clean_grid.shape, generator=noise_generator),
            torch.randn(clean_grid.shape, generator=noise_generator),
        )
        * (complex_noise_variance / 2.0) ** 0.5
    )
    estimate = receiver.estimate(clean_grid + noise, complex_noise_variance)
    pilot_mask = receiver.pilot_mask[0, 0].bool()
    samples = estimate.h_hat[:, 0, 0, 0, 0][:, pilot_mask]
    predicted_variance = estimate.error_variance[0, 0, 0, 0, 0][pilot_mask]
    empirical_bias = (samples.mean(dim=0) - channel).abs()
    empirical_variance = (samples - samples.mean(dim=0)).abs().square().mean()
    mean_predicted_variance = predicted_variance.mean()
    bias_tolerance = 5.0 * torch.sqrt(predicted_variance / batch_size)

    assert samples.shape[1] == int(pilot_mask.sum())
    assert bool(torch.isfinite(predicted_variance).all())
    assert bool((predicted_variance > 0).all())
    torch.testing.assert_close(
        predicted_variance,
        torch.full_like(
            predicted_variance,
            complex_noise_variance / (2 * dmrs_length * num_cdm_groups_without_data),
        ),
        rtol=1e-6,
        atol=1e-7,
    )
    assert bool((empirical_bias < bias_tolerance).all())
    assert float(empirical_variance / mean_predicted_variance) == pytest.approx(1.0, abs=0.12)


@pytest.mark.native_sionna
def test_pusch_dmrs_uncertainty_tracks_distinct_per_batch_noise_variances():
    """Check batch noise broadcasting against realized pilot-estimation errors."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30), device="cpu"
    )
    batch_per_level = 256
    noise_levels = torch.tensor((0.01, 0.05, 0.2), dtype=torch.float32)
    noise_variance = noise_levels.repeat_interleave(batch_per_level)
    batch_size = noise_variance.numel()
    transmit_grid, _ = receiver.transmit(batch_size, seed=58_311)
    channel = torch.tensor(0.63 - 0.27j, dtype=torch.complex64)
    clean_grid = transmit_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel
    generator = torch.Generator().manual_seed(91_227)
    white_noise = torch.complex(
        torch.randn(clean_grid.shape, generator=generator),
        torch.randn(clean_grid.shape, generator=generator),
    )
    received_grid = clean_grid + white_noise * torch.sqrt(
        noise_variance.reshape(batch_size, 1, 1, 1, 1) / 2
    )

    estimate = receiver.estimate(received_grid, noise_variance)
    pilot_mask = receiver.pilot_mask[0, 0].bool()
    channel_errors = estimate.h_hat[:, 0, 0, 0, 0][:, pilot_mask] - channel
    predicted_variance = estimate.error_variance[:, 0, 0, 0, 0][:, pilot_mask]

    assert estimate.error_variance.shape[0] == batch_size
    for level_index, expected_noise in enumerate(noise_levels):
        start = level_index * batch_per_level
        stop = start + batch_per_level
        empirical_variance = channel_errors[start:stop].abs().square().mean()
        predicted = predicted_variance[start:stop].mean()
        torch.testing.assert_close(
            predicted,
            expected_noise / 2,
            atol=1e-7,
            rtol=1e-6,
        )
        assert float(empirical_variance / predicted) == pytest.approx(1.0, abs=0.08)

    unsupported_per_re_variance = noise_variance[:1].reshape(1, 1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match=r"\[batch, rx, rx_ant\]"):
        receiver.estimate(received_grid[:1], unsupported_per_re_variance)


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
