import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation
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
