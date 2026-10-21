import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from datasets.nr_store import read_nr_record, write_nr_record  # noqa: E402
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from integrations.sionna.tdl_pusch import (  # noqa: E402
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
