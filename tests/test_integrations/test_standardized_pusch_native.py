"""Native CPU PUSCH/DMRS tests over standardized TDL/CDL channel truth."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from datasets.nr_store import read_nr_record, write_nr_record  # noqa: E402
from datasets.standardized import (  # noqa: E402
    StandardizedFamily,
    standardized_run_from_channel,
)
from integrations.sionna.standardized_pusch import (  # noqa: E402
    build_standardized_pusch_dmrs_records,
)
from receivers.nr import NRPUSCHConfig  # noqa: E402


def _standardized_run(family: StandardizedFamily):
    if family is StandardizedFamily.TDL:
        from integrations.sionna.tdl import generate_tdl_frequency_response

        channel = generate_tdl_frequency_response(
            model="A",
            seed=711,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=128,
            delay_spread_s=100e-9,
            batch_size=2,
        )
    else:
        from integrations.sionna.cdl import generate_cdl_frequency_response

        channel = generate_cdl_frequency_response(
            model="D",
            seed=712,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=128,
            delay_spread_s=100e-9,
            batch_size=2,
        )
    return standardized_run_from_channel(channel, family=family)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("family", "profile_label"),
    [(StandardizedFamily.TDL, 0), (StandardizedFamily.CDL, 1)],
)
def test_standardized_tdl_cdl_to_pusch_dmrs_is_seeded_and_target_is_separate(
    tmp_path, family: StandardizedFamily, profile_label: int
) -> None:
    run = _standardized_run(family)
    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    first = build_standardized_pusch_dmrs_records(
        run, snr_db=50.0, seed=8129, receiver_config=config, device="cpu"
    )
    replay = build_standardized_pusch_dmrs_records(
        run, snr_db=50.0, seed=8129, receiver_config=config, device="cpu"
    )

    assert len(first) == 2
    for record, repeated in zip(first, replay, strict=True):
        observation = record.observation
        estimate = observation.channel_estimate
        truth = record.targets["channel_cfr"]
        assert observation.received_grid.shape == (1, 1, 1, 14, 72)
        assert estimate.h_hat.shape == (1, 1, 1, 1, 1, 14, 72)
        assert truth.shape == estimate.h_hat.shape
        assert record.targets["profile_los_proxy"].tolist() == [profile_label]
        assert record.target_masks["profile_los_proxy"].tolist() == [True]
        assert record.provenance["geometric_los_label"] is None
        assert record.provenance["pusch_transmit_seed"] == record.provenance["noise_seed"] - 1
        assert record.provenance["receiver_link_model"] == (
            "ideal frequency-domain channel; no ICI/ISI"
        )
        assert not hasattr(observation.model_inputs(), "targets")
        assert not hasattr(observation.model_inputs(), "channel_truth")
        torch.testing.assert_close(
            observation.received_grid, repeated.observation.received_grid, rtol=0, atol=0
        )
        torch.testing.assert_close(estimate.h_hat, repeated.observation.channel_estimate.h_hat)

        estimate_nmse = torch.mean(torch.abs(estimate.h_hat - truth) ** 2) / torch.mean(
            torch.abs(truth) ** 2
        )
        # DMRS CDM despreading/interpolation smooths neighboring frequency bins,
        # so this compares the full estimate to oracle CFR with a declared,
        # nonzero frequency-selective interpolation tolerance.
        assert float(estimate_nmse) < 0.01
        restored = read_nr_record(tmp_path, write_nr_record(tmp_path, record))
        np.testing.assert_array_equal(
            restored.targets["profile_los_proxy"], record.targets["profile_los_proxy"]
        )
        np.testing.assert_allclose(
            restored.observation.channel_estimate.h_hat, estimate.h_hat.cpu().numpy()
        )
        np.testing.assert_allclose(restored.targets["channel_cfr"], truth.cpu().numpy())


def test_standardized_dmrs_rejects_source_bandwidth_that_cannot_cover_pusch() -> None:
    from integrations.sionna.tdl import generate_tdl_frequency_response

    channel = generate_tdl_frequency_response(
        model="A",
        seed=32,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=16,
        delay_spread_s=100e-9,
    )
    run = standardized_run_from_channel(channel, family=StandardizedFamily.TDL)

    with pytest.raises(ValueError, match="inside the source CFR bandwidth"):
        build_standardized_pusch_dmrs_records(
            run,
            snr_db=20.0,
            seed=14,
            receiver_config=NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30),
            device="cpu",
        )


@pytest.mark.native_sionna
def test_standardized_tdl_simo_keeps_receive_antenna_axis() -> None:
    from integrations.sionna.tdl import generate_tdl_frequency_response

    channel = generate_tdl_frequency_response(
        model="A",
        seed=902,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=128,
        delay_spread_s=100e-9,
        batch_size=1,
        num_rx_ant=2,
    )
    run = standardized_run_from_channel(channel, family=StandardizedFamily.TDL)
    records = build_standardized_pusch_dmrs_records(
        run,
        snr_db=35.0,
        seed=1702,
        receiver_config=NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30),
        device="cpu",
    )

    record = records[0]
    assert record.observation.received_grid.shape == (1, 1, 2, 14, 72)
    assert record.observation.channel_estimate.h_hat.shape == (1, 1, 2, 1, 1, 14, 72)
    assert record.targets["channel_cfr"].shape == (1, 1, 2, 1, 1, 14, 72)
    assert record.target_masks["channel_cfr"].shape == record.targets["channel_cfr"].shape


@pytest.mark.native_sionna
def test_independent_noise_views_keep_waveform_parent_and_channel_truth(tmp_path) -> None:
    run = _standardized_run(StandardizedFamily.TDL)
    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    first = build_standardized_pusch_dmrs_records(
        run, snr_db=18.0, seed=1207, noise_seed=8101, receiver_config=config, device="cpu"
    )
    second = build_standardized_pusch_dmrs_records(
        run, snr_db=18.0, seed=1207, noise_seed=8102, receiver_config=config, device="cpu"
    )
    replay = build_standardized_pusch_dmrs_records(
        run, snr_db=18.0, seed=1207, noise_seed=8101, receiver_config=config, device="cpu"
    )

    assert len(first) == len(second) == len(replay) == 2
    assert len({record.observation.sample_id for record in (*first, *second)}) == 4
    for view_a, view_b, repeated in zip(first, second, replay, strict=True):
        assert view_a.observation.sample_id == repeated.observation.sample_id
        assert view_a.observation.sample_id != view_b.observation.sample_id
        assert view_a.provenance["parent_subject_id"] == view_b.provenance["parent_subject_id"]
        assert view_a.provenance["channel_sample_id"] == view_b.provenance["channel_sample_id"]
        assert view_a.provenance["receiver_view_kind"] == "standardized_pusch_noise_view"
        assert view_a.provenance["pusch_transmit_seed"] == view_b.provenance["pusch_transmit_seed"]
        assert view_a.provenance["noise_seed"] != view_b.provenance["noise_seed"]
        assert view_a.provenance["snr_reference"] == (
            "post-channel, pre-beamforming receive resource grid"
        )
        assert (
            "mean received PUSCH resource-grid complex-sample power"
            in view_a.provenance["snr_definition"]
        )
        assert view_a.provenance["noise_figure_db"] is None
        assert view_a.provenance["noise_variance_per_real_dimension"] == pytest.approx(
            view_a.provenance["noise_variance_per_complex_dimension"] / 2
        )
        assert (
            view_a.provenance["receiver_view_identity_sha256"]
            != view_b.provenance["receiver_view_identity_sha256"]
        )
        torch.testing.assert_close(
            view_a.targets["channel_cfr"], view_b.targets["channel_cfr"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            view_a.observation.received_grid,
            repeated.observation.received_grid,
            rtol=0,
            atol=0,
        )
        assert not torch.equal(view_a.observation.received_grid, view_b.observation.received_grid)
        for view in (view_a, view_b):
            restored = read_nr_record(tmp_path, write_nr_record(tmp_path, view))
            assert restored.observation.sample_id == view.observation.sample_id
            assert restored.provenance["parent_subject_id"] == view.provenance["parent_subject_id"]
            assert restored.provenance["receiver_view_kind"] == "standardized_pusch_noise_view"
            np.testing.assert_array_equal(
                restored.targets["channel_cfr"], view.targets["channel_cfr"].cpu().numpy()
            )
