"""Native Sionna PUSCH allocation, carrier-grid mapping and decoding checks."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from receivers.nr import NRPUSCHConfig  # noqa: E402
from receivers.nr.pusch import compose_pusch_intra_slot_hops  # noqa: E402


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
    assert receiver.metadata["frequency_hopping"] == "neither"
    assert receiver.metadata["transform_precoding"] is False
    assert receiver.metadata["resource_grid"]["subcarrier_spacing_hz"] == (
        subcarrier_spacing_khz * 1_000
    )


@pytest.mark.native_sionna
def test_pusch_adapter_supports_isolated_perfect_csi_diagnostic_decode() -> None:
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30), device="cpu"
    )
    transmitted_grid, _ = receiver.transmit(batch_size=2, seed=922)
    grid = receiver.resource_grid
    channel_truth = torch.ones(
        2,
        1,
        1,
        1,
        1,
        grid.num_ofdm_symbols,
        grid.fft_size,
        dtype=torch.complex64,
    )

    _, crc_ok = receiver.decode_pusch_with_channel_truth(
        transmitted_grid, torch.tensor(1e-6), channel_truth
    )

    assert bool(crc_ok.all())
    assert receiver._perfect_csi_decoder is not None


@pytest.mark.native_sionna
def test_offset_bwp_maps_into_carrier_fft_bins_and_decodes() -> None:
    """Map a 6-RB BWP at CRB 7 into a grid starting at CRB 2, then decode."""
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
    recovered_carrier_grid = receiver.demodulate_carrier_slot_waveform(waveform)
    torch.testing.assert_close(recovered_carrier_grid, carrier_slot_grid, rtol=1e-5, atol=1e-6)
    decoded_bits, crc_ok = receiver.decode_carrier_slot_waveform(waveform, torch.tensor(1e-7))
    torch.testing.assert_close(decoded_bits, transmitted_bits)
    assert bool(crc_ok.all())
    with pytest.raises(ValueError, match="exactly one configured full-carrier slot"):
        receiver.demodulate_carrier_slot_waveform(waveform[..., :-1])


@pytest.mark.native_sionna
def test_caller_resolved_intra_slot_hops_dehop_into_bounded_pusch_receiver() -> None:
    """Rehome the two received segments and decode with the regular DM-RS path."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            n_size_grid=24,
            n_start_grid=10,
            n_start_bwp=12,
            subcarrier_spacing_khz=30,
            mapping_type="B",
            symbol_allocation=(3, 10),
        ),
        device="cpu",
    )
    allocation_grid, transmitted_bits = receiver.transmit(batch_size=1, seed=60422)
    hopped_grid = compose_pusch_intra_slot_hops(
        allocation_grid.detach().cpu().numpy(),
        carrier_grid_size_rb=24,
        carrier_grid_start_rb=10,
        bwp_start_rb=12,
        bwp_size_rb=12,
        allocation_start_symbol=3,
        first_hop_rb_start=0,
        second_hop_rb_start=7,
        first_hop_symbol_count=4,
    )

    decoded_bits, crc_ok, report = receiver.decode_intra_slot_hopped_carrier_grid(
        hopped_grid.frequency_grid,
        hopped_grid.metadata,
        torch.tensor(1e-7),
    )

    torch.testing.assert_close(decoded_bits, transmitted_bits)
    assert bool(crc_ok.all())
    assert report["schema"] == "loslab.nr_pusch_intra_slot_hopped_decode.v1"
    assert report["status"] == "DECODED_NOT_QUALIFIED"
    assert report["dehop_replay"]["status"] == "PASS_INPUT_SCOPED_REPLAY"
    assert report["receiver_hopping_qualification"] == "NOT_QUALIFIED"
    assert report["dmrs_source"] == "caller_supplied_hop_mapped_received_grid"
    assert report["noise_variance_shape"] == []
    assert len(report["noise_variance_sha256"]) == 64
    import jsonschema

    schema = json.loads(
        Path("datasets/schema/nr-pusch-intra-slot-hopped-decode.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(report)


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
