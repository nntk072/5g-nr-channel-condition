import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from datasets.nr_store import read_nr_record, write_nr_record  # noqa: E402
from experiments.link_curves import (  # noqa: E402
    run_tdl_pusch_bler_curve,
    run_tdl_pusch_rx_diversity_curve,
    run_tdl_pusch_time_bler_curve,
)
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from integrations.sionna.tdl_pusch import (  # noqa: E402
    SionnaTDLTimePUSCHLink,
    generate_tdl_pusch_batch,
    tdl_pusch_records,
)
from receivers.nr import NRPUSCHConfig  # noqa: E402


@pytest.mark.native_sionna
def test_tdl_pusch_link_produces_stored_receiver_records_with_separate_truth(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    values = {
        "model": "A",
        "batch_size": 2,
        "seed": 72,
        "carrier_frequency_hz": 3.5e9,
        "delay_spread_s": 100e-9,
        "noise_variance": 1e-3,
        "num_rx_ant": 2,
        "spec_version": "19.2",
    }
    link = generate_tdl_pusch_batch(receiver, **values)
    repeated = generate_tdl_pusch_batch(receiver, **values)
    observation = receiver.observe("tdl-link-72", link.received_grid, link.noise_variance)
    records = tdl_pusch_records(receiver, link, sample_ids=("tdl-link-72-0", "tdl-link-72-1"))

    assert link.transmit_grid.is_cuda
    assert link.transmit_bits.is_cuda
    assert link.received_grid.shape == (2, 1, 2, 14, 24)
    assert link.channel_truth.shape == (2, 1, 2, 1, 1, 14, 24)
    assert observation.model_inputs().received_grid.is_cuda
    assert observation.model_inputs().h_hat.shape == (2, 1, 2, 1, 1, 14, 24)
    assert len(records) == 2
    assert records[0].observation.model_inputs().received_grid.shape == (1, 1, 2, 14, 24)
    assert records[0].targets["channel_cfr"].shape == (1, 1, 2, 1, 1, 14, 24)
    assert records[0].target_masks["channel_cfr"].all()
    assert not hasattr(records[0].observation.model_inputs(), "transmit_bits")
    assert link.metadata["channel_model"].startswith("Sionna OFDMChannel")
    torch.testing.assert_close(link.received_grid, repeated.received_grid)
    stored = write_nr_record(tmp_path, records[0])
    restored = read_nr_record(tmp_path, stored)
    assert restored.observation.sample_id == "tdl-link-72-0"
    assert restored.targets["channel_cfr"].shape == (1, 1, 2, 1, 1, 14, 24)
    assert restored.provenance["channel_truth_source"] == "Sionna OFDMChannel return_channel"


@pytest.mark.native_sionna
def test_tdl_pusch_bler_curve_records_matched_fading_realizations():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    result = run_tdl_pusch_bler_curve(
        snr_db=(0.0, 12.0),
        frames_per_snr=16,
        batch_size=8,
        seed=8102,
        profile="A",
        delay_spread_s=100e-9,
        carrier_frequency_hz=3.5e9,
        speed_mps=3.0,
        n_size_bwp=2,
        subcarrier_spacing_khz=30,
    )

    assert result["evidence_grade"] == "SIMULATION_ONLY_UNVALIDATED"
    assert [point["snr_db"] for point in result["points"]] == [0.0, 12.0]
    assert all(point["frames"] == 16 for point in result["points"])
    assert all(0.0 <= point["bler"] <= 1.0 for point in result["points"])
    assert all(
        item["transmit_bits_match_across_snr"] and item["channel_truth_allclose_across_snr"]
        for item in result["configuration"]["matched_realization_checks"]
    )


@pytest.mark.native_sionna
def test_sample_domain_tdl_pusch_decodes_waveforms_and_records_matched_bler():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=12, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    link = SionnaTDLTimePUSCHLink(
        receiver,
        model="A",
        delay_spread_s=100e-9,
        carrier_frequency_hz=3.5e9,
        speed_mps=3.0,
    )
    waveform, bits, clean, channel_truth = link.transmit_channel(8, seed=99102)
    decoded, crc_ok = link.decode(clean, 1e-4)

    assert waveform.is_cuda and channel_truth.is_cuda
    assert waveform.ndim == 4 and clean.shape[-1] > waveform.shape[-1]
    assert channel_truth.shape[-2] == clean.shape[-1]
    assert link.metadata["channel_memory_samples"] <= link.metadata["cyclic_prefix_min_samples"]
    assert torch.count_nonzero(channel_truth[..., 1:, :] - channel_truth[..., :-1, :]) > 0
    assert torch.equal(decoded, bits)
    assert crc_ok.all()

    result = run_tdl_pusch_time_bler_curve(
        snr_db=(0.0, 18.0),
        frames_per_snr=8,
        batch_size=4,
        seed=99102,
        profile="A",
        delay_spread_s=100e-9,
        carrier_frequency_hz=3.5e9,
        speed_mps=3.0,
        n_size_bwp=12,
        subcarrier_spacing_khz=30,
    )
    assert result["evidence_grade"] == "SIMULATION_ONLY_UNVALIDATED"
    assert [point["snr_db"] for point in result["points"]] == [0.0, 18.0]
    assert [point["frames"] for point in result["points"]] == [8, 8]
    assert all(
        len(point["frame_error_mask"]) == point["frames"]
        and sum(point["frame_error_mask"]) == point["bler_errors"]
        and point["mean_transmitted_tb_bits_per_codeword"] == point["bits"] / point["frames"]
        and point["successfully_decoded_tb_bits_per_codeword"]
        <= point["mean_transmitted_tb_bits_per_codeword"]
        for point in result["points"]
    )
    assert result["points"][0]["bler"] > result["points"][1]["bler"]
    assert all(
        item["transmit_bits_match_across_snr"] and item["channel_truth_match_across_snr"]
        for item in result["configuration"]["matched_realization_checks"]
    )


@pytest.mark.native_sionna
def test_sample_domain_tdl_pusch_receiver_decodes_one_and_two_receive_antennas():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=12, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    link = SionnaTDLTimePUSCHLink(
        receiver,
        model="A",
        delay_spread_s=100e-9,
        carrier_frequency_hz=3.5e9,
        speed_mps=3.0,
        num_rx_ant=2,
    )
    waveform, bits, clean, channel_truth = link.transmit_channel(16, seed=99103)

    decoded_two, crc_two = link.decode(clean, 1e-6)
    decoded_one, crc_one = link.decode(clean[:, :, :1, :], 1e-6)

    assert waveform.is_cuda and channel_truth.is_cuda
    assert clean.shape[2] == 2
    assert channel_truth.shape[2] == 2
    assert torch.equal(decoded_two, bits)
    assert torch.equal(decoded_one, bits)
    assert crc_two.all() and crc_one.all()


@pytest.mark.native_sionna
def test_sample_domain_tdl_pusch_rx_diversity_uses_paired_channel_draws():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    result = run_tdl_pusch_rx_diversity_curve(
        snr_db=(8.0, 12.0),
        frames_per_snr=16,
        batch_size=8,
        seed=99104,
        profile="A",
        delay_spread_s=100e-9,
        carrier_frequency_hz=3.5e9,
        speed_mps=3.0,
        n_size_bwp=12,
        subcarrier_spacing_khz=30,
    )

    assert result["schema"] == "loslab.tdl_pusch_rx_diversity.v1"
    assert result["configuration"]["num_rx_ant"] == 2
    assert result["configuration"]["n_size_bwp_resource_blocks"] == 12
    assert result["configuration"]["subcarrier_spacing_khz"] == 30
    assert result["configuration"]["receive_spatial_correlation_assumption"].startswith(
        "uncorrelated"
    )
    assert len(result["configuration"]["clean_receive_power_by_antenna_per_batch"]) == 2
    assert all(
        check["single_rx_is_first_antenna_of_two_rx_draw"]
        for check in result["configuration"]["matched_realization_checks"]
    )
    assert [point["snr_db"] for point in result["points"]] == [8.0, 12.0]
    assert all(
        len(point["single_rx"]["frame_error_mask"]) == 16
        and len(point["two_rx"]["frame_error_mask"]) == 16
        and all(power > 0.0 for power in point["mean_clean_receive_power_by_antenna"])
        and "mcnemar_exact_two_sided_p_value_holm" in point["paired_comparison"]
        for point in result["points"]
    )


@pytest.mark.native_sionna
def test_sample_domain_link_records_explicit_channel_over_cyclic_prefix_stress():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=12, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    with pytest.raises(ValueError, match="allow_cp_overrun=True"):
        SionnaTDLTimePUSCHLink(
            receiver,
            model="A",
            delay_spread_s=300e-9,
            carrier_frequency_hz=3.5e9,
            speed_mps=3.0,
        )

    result = run_tdl_pusch_time_bler_curve(
        snr_db=(12.0, 24.0),
        frames_per_snr=16,
        batch_size=8,
        seed=99103,
        profile="A",
        delay_spread_s=300e-9,
        carrier_frequency_hz=3.5e9,
        speed_mps=3.0,
        n_size_bwp=12,
        subcarrier_spacing_khz=30,
        allow_cp_overrun=True,
    )
    config = result["configuration"]
    assert result["evidence_grade"] == "SIMULATION_ONLY_UNVALIDATED"
    assert config["allow_cp_overrun"] is True
    assert config["cp_covers_channel_memory"] is False
    assert config["channel_memory_samples"] > config["cyclic_prefix_min_samples"]
    assert [point["snr_db"] for point in result["points"]] == [12.0, 24.0]
    assert all(point["frames"] == 16 for point in result["points"])
    assert all(0 <= point["bler"] <= 1 for point in result["points"])
    assert all(
        check["transmit_bits_match_across_snr"] and check["channel_truth_match_across_snr"]
        for check in config["matched_realization_checks"]
    )
