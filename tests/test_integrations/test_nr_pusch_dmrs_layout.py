"""Native Sionna PUSCH DM-RS positions, sequences and orthogonality checks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.noise import proper_complex_awgn  # noqa: E402
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from receivers.nr import NRPUSCHConfig  # noqa: E402
from tests.test_integrations.nr_pusch_native_support import (  # noqa: E402
    _pusch_dmrs_gold_reference,
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
@pytest.mark.parametrize("duration", (10, 11, 12, 13, 14))
@pytest.mark.parametrize("mapping_type", ("A", "B"))
@pytest.mark.parametrize("additional_position", (0, 1, 2, 3))
def test_pusch_type2_single_symbol_partial_allocation_matches_ts_38211(
    duration: int, mapping_type: str, additional_position: int
) -> None:
    """Check partial Type-2 single-symbol timing against TS 38.211 Table 6.4.1.1.3-3."""
    expected_by_duration = {
        10: {
            "A": {0: (2,), 1: (2, 9), 2: (2, 6, 9), 3: (2, 6, 9)},
            "B": {0: (0,), 1: (0, 8), 2: (0, 4, 8), 3: (0, 3, 6, 9)},
        },
        11: {
            "A": {0: (2,), 1: (2, 9), 2: (2, 6, 9), 3: (2, 6, 9)},
            "B": {0: (0,), 1: (0, 8), 2: (0, 4, 8), 3: (0, 3, 6, 9)},
        },
        12: {
            "A": {0: (2,), 1: (2, 9), 2: (2, 6, 9), 3: (2, 5, 8, 11)},
            "B": {0: (0,), 1: (0, 10), 2: (0, 5, 10), 3: (0, 3, 6, 9)},
        },
        13: {
            "A": {0: (2,), 1: (2, 11), 2: (2, 7, 11), 3: (2, 5, 8, 11)},
            "B": {0: (0,), 1: (0, 10), 2: (0, 5, 10), 3: (0, 3, 6, 9)},
        },
        14: {
            "A": {0: (2,), 1: (2, 11), 2: (2, 7, 11), 3: (2, 5, 8, 11)},
            "B": {0: (0,), 1: (0, 10), 2: (0, 5, 10), 3: (0, 3, 6, 9)},
        },
    }
    expected_symbols = expected_by_duration[duration][mapping_type][additional_position]
    start_symbol = 0 if mapping_type == "A" else 14 - duration
    symbol_allocation = (start_symbol, duration)
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
@pytest.mark.parametrize("subcarrier_spacing_khz", (15, 30, 60, 120))
@pytest.mark.parametrize("duration", (10, 11, 12, 13, 14))
@pytest.mark.parametrize("mapping_type", ("A", "B"))
@pytest.mark.parametrize("additional_position", (0, 1))
def test_pusch_type2_double_symbol_partial_allocation_matches_ts_38211(
    mapping_type: str,
    duration: int,
    additional_position: int,
    subcarrier_spacing_khz: int,
) -> None:
    """Check partial Type-2 double-symbol timing across Table 6.4.1.1.3-4 rows."""
    expected_by_duration = {
        10: {"A": {0: (2, 3), 1: (2, 3, 8, 9)}, "B": {0: (0, 1), 1: (0, 1, 7, 8)}},
        11: {"A": {0: (2, 3), 1: (2, 3, 8, 9)}, "B": {0: (0, 1), 1: (0, 1, 7, 8)}},
        12: {"A": {0: (2, 3), 1: (2, 3, 8, 9)}, "B": {0: (0, 1), 1: (0, 1, 9, 10)}},
        13: {"A": {0: (2, 3), 1: (2, 3, 10, 11)}, "B": {0: (0, 1), 1: (0, 1, 9, 10)}},
        14: {"A": {0: (2, 3), 1: (2, 3, 10, 11)}, "B": {0: (0, 1), 1: (0, 1, 9, 10)}},
    }
    expected_symbols = expected_by_duration[duration][mapping_type][additional_position]
    start_symbol = 0 if mapping_type == "A" else 14 - duration
    symbol_allocation = (start_symbol, duration)
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
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
@pytest.mark.parametrize("duration", (10, 11, 12, 13, 14))
@pytest.mark.parametrize("mapping_type", ("A", "B"))
@pytest.mark.parametrize("additional_position", (0, 1))
def test_partial_type2_double_symbol_pusch_allocation_places_and_decodes(
    duration: int, mapping_type: str, additional_position: int
) -> None:
    """Schedule and decode Type-2 double-symbol DM-RS over partial slot allocations."""
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import OFDMDemodulator

    start_symbol = 0 if mapping_type == "A" else 14 - duration
    symbol_allocation = (start_symbol, duration)
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=6,
            subcarrier_spacing_khz=30,
            dmrs_config_type=2,
            dmrs_length=2,
            dmrs_additional_position=additional_position,
            mapping_type=mapping_type,
            symbol_allocation=symbol_allocation,
        ),
        device="cpu",
    )
    allocation_grid, transmitted_bits = receiver.transmit(
        batch_size=1,
        seed=9031 + duration + (0 if mapping_type == "A" else 20) + 40 * additional_position,
    )

    slot_grid = receiver.place_allocation_in_slot(allocation_grid)
    assert slot_grid.shape[-2] == 14
    assert not bool(slot_grid[..., :start_symbol, :].any())
    allocation_end = start_symbol + duration
    assert not bool(slot_grid[..., allocation_end:, :].any())

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
    assert receiver.metadata["symbol_allocation"] == symbol_allocation
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
@pytest.mark.parametrize("duration", (10, 11, 12, 13, 14))
@pytest.mark.parametrize("mapping_type", ("A", "B"))
@pytest.mark.parametrize("additional_position", (0, 1, 2, 3))
def test_partial_type2_single_symbol_pusch_allocation_places_and_decodes(
    duration: int, mapping_type: str, additional_position: int
) -> None:
    """Schedule and decode Type-2 single-symbol DM-RS over partial slot allocations."""
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import OFDMDemodulator

    start_symbol = 0 if mapping_type == "A" else 14 - duration
    symbol_allocation = (start_symbol, duration)
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=6,
            subcarrier_spacing_khz=30,
            dmrs_config_type=2,
            dmrs_length=1,
            dmrs_additional_position=additional_position,
            mapping_type=mapping_type,
            symbol_allocation=symbol_allocation,
        ),
        device="cpu",
    )
    allocation_grid, transmitted_bits = receiver.transmit(
        batch_size=1,
        seed=12031 + duration + (0 if mapping_type == "A" else 20) + 40 * additional_position,
    )

    slot_grid = receiver.place_allocation_in_slot(allocation_grid)
    assert slot_grid.shape[-2] == 14
    assert not bool(slot_grid[..., :start_symbol, :].any())
    allocation_end = start_symbol + duration
    assert not bool(slot_grid[..., allocation_end:, :].any())

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
    assert receiver.metadata["symbol_allocation"] == symbol_allocation
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
@pytest.mark.parametrize("num_ports", (2, 4))
def test_pusch_type1_multiport_dmrs_pilots_are_orthogonal(num_ports: int) -> None:
    """Check Type-1 multi-stream CDM orthogonality from TS 38.211 Table 6.4.1.1.3-1."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=4,
            subcarrier_spacing_khz=30,
            num_antenna_ports=num_ports,
            num_layers=num_ports,
            dmrs_config_type=1,
            num_cdm_groups_without_data=2,
        ),
        device="cpu",
    )
    pattern = receiver.resource_grid.pilot_pattern
    masks = receiver.pilot_mask[0].bool()
    pilots = pattern.pilots[0].to(torch.complex128)

    assert tuple(masks.shape) == (num_ports, 14, 48)
    assert torch.equal(masks, masks[0].expand_as(masks))
    assert tuple(pilots.shape) == (num_ports, 48)
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
