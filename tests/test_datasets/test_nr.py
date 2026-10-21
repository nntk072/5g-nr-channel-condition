import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation
from datasets.nr_store import read_nr_record, write_nr_record
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def make_observation(**overrides):
    h_hat = np.zeros((2, 1, 2, 1, 1, 14, 24), dtype=np.complex64)
    estimate = NRChannelEstimate(
        h_hat=h_hat,
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        pilot_mask=np.zeros((1, 1, 14, 24), dtype=np.bool_),
        metadata={"receiver": "PUSCHLSChannelEstimator", "sionna_version": "2.2.0"},
    )
    values = {
        "sample_id": "sample-001",
        "received_grid": np.zeros((2, 1, 2, 14, 24), dtype=np.complex64),
        "channel_estimate": estimate,
        "fft_bin_indices": tuple(range(24)),
        "receiver_config": NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
    }
    values.update(overrides)
    return NRObservation(**values)


def test_nr_observation_builds_inference_payload_without_sample_oracle_ids():
    observation = make_observation()
    model_inputs = observation.model_inputs()

    assert model_inputs.received_grid.shape == (2, 1, 2, 14, 24)
    assert model_inputs.h_hat.shape == (2, 1, 2, 1, 1, 14, 24)
    assert model_inputs.fft_bin_indices == tuple(range(24))
    assert not hasattr(model_inputs, "sample_id")
    assert not hasattr(model_inputs, "targets")


def test_receiver_output_factory_preserves_the_authoritative_fft_bin_mapping():
    observation = NRObservation.from_receiver_output(
        sample_id="sample-from-grid",
        received_grid=np.zeros((2, 1, 2, 14, 24), dtype=np.complex64),
        channel_estimate=make_observation().channel_estimate,
        effective_subcarrier_indices=np.arange(24, dtype=np.int64),
        receiver_config=NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
    )

    assert observation.sample_id == "sample-from-grid"
    assert observation.fft_bin_indices == tuple(range(24))


def test_nr_training_record_keeps_targets_and_provenance_outside_model_inputs():
    observation = make_observation()
    record = NRChannelRecord(
        observation=observation,
        targets={"geometric_los": np.asarray(True), "channel_cfr": np.zeros((24,))},
        target_masks={"geometric_los": np.asarray(True)},
        provenance={"scene_hash": "sha256:abc", "split": "train"},
    )

    assert "geometric_los" in record.targets
    assert "scene_hash" in record.provenance
    assert not hasattr(record.observation.model_inputs(), "provenance")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sample_id": ""}, "sample_id"),
        ({"fft_bin_indices": (0, 1, 2)}, "map every estimated"),
        (
            {
                "fft_bin_indices": (
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    24,
                )
            },
            "valid zero-based",
        ),
        ({"received_grid": np.zeros((2, 1, 2, 13, 24), dtype=np.complex64)}, "axes differ"),
    ],
)
def test_nr_observation_rejects_invalid_or_privileged_input(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_observation(**overrides)


def test_nr_observation_requires_typed_receiver_config():
    with pytest.raises(TypeError, match="typed NRPUSCHConfig"):
        make_observation(receiver_config={"scene_id": "scene-7"})


def test_nr_channel_record_rejects_masks_without_targets():
    with pytest.raises(ValueError, match="correspond to declared targets"):
        NRChannelRecord(
            observation=make_observation(),
            targets={"geometric_los": True},
            target_masks={"path_delays": True},
            provenance={"source": "simulated"},
        )


def test_nr_record_round_trip_is_lossless_pickle_free_and_content_addressed(tmp_path):
    observation = make_observation()
    record = NRChannelRecord(
        observation=observation,
        targets={
            "geometric_los": np.asarray(True),
            "channel_cfr": (np.arange(24) + 1j * np.arange(24)).astype(np.complex64),
        },
        target_masks={"geometric_los": np.asarray(True)},
        provenance={"scene_hash": "sha256:abc", "simulator": "Sionna 2.2"},
    )

    stored = write_nr_record(tmp_path, record)
    assert write_nr_record(tmp_path, record) == stored
    restored = read_nr_record(tmp_path, stored)

    assert restored.observation.sample_id == observation.sample_id
    assert restored.observation.fft_bin_indices == observation.fft_bin_indices
    np.testing.assert_array_equal(restored.observation.received_grid, observation.received_grid)
    np.testing.assert_array_equal(
        restored.observation.channel_estimate.h_hat,
        observation.channel_estimate.h_hat,
    )
    np.testing.assert_array_equal(restored.targets["channel_cfr"], record.targets["channel_cfr"])
    assert restored.provenance == record.provenance


def test_nr_record_store_detects_tampering_and_sample_id_mutation(tmp_path):
    record = NRChannelRecord(
        observation=make_observation(),
        targets={"geometric_los": np.asarray(True)},
        target_masks={},
        provenance={"scene_hash": "sha256:abc"},
    )
    stored = write_nr_record(tmp_path, record)
    data_path = tmp_path / stored.relative_path
    data_path.write_bytes(data_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="content hash"):
        read_nr_record(tmp_path, stored)

    with pytest.raises(FileExistsError, match="File exists"):
        write_nr_record(
            tmp_path,
            NRChannelRecord(
                observation=make_observation(received_grid=np.ones((2, 1, 2, 14, 24))),
                targets={"geometric_los": np.asarray(True)},
                target_masks={},
                provenance={"scene_hash": "sha256:abc"},
            ),
        )
