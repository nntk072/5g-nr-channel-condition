import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation, NRTrajectory, TargetDefinition
from datasets.nr_store import (
    read_nr_record,
    read_nr_trajectory,
    write_nr_record,
    write_nr_trajectory,
)
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


def make_single_record(sample_id, *, targets, target_masks):
    estimate = NRChannelEstimate(
        h_hat=np.zeros((1, 1, 2, 1, 1, 14, 24), dtype=np.complex64),
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        pilot_mask=np.zeros((1, 1, 14, 24), dtype=np.bool_),
        metadata={"receiver": "PUSCHLSChannelEstimator"},
    )
    observation = make_observation(
        sample_id=sample_id,
        received_grid=np.zeros((1, 1, 2, 14, 24), dtype=np.complex64),
        channel_estimate=estimate,
    )
    return NRChannelRecord(
        observation=observation,
        targets=targets,
        target_masks=target_masks,
        provenance={"source": "trajectory-fixture"},
    )


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


@pytest.mark.parametrize(
    ("target", "mask", "exception", "message"),
    [
        (np.zeros((2, 3)), np.ones((2, 3)), TypeError, "boolean dtype"),
        (np.zeros((2, 3)), np.ones((4,), dtype=np.bool_), ValueError, "broadcastable"),
    ],
)
def test_nr_channel_record_validates_target_mask_dtype_and_shape(target, mask, exception, message):
    with pytest.raises(exception, match=message):
        NRChannelRecord(
            observation=make_observation(),
            targets={"target": target},
            target_masks={"target": mask},
            provenance={"source": "unit-test"},
        )


def test_nr_trajectory_keeps_time_ordered_multitarget_masks_out_of_model_inputs():
    first = make_single_record(
        "trajectory-1-step-0",
        targets={"channel_cfr": np.ones(24, dtype=np.complex64), "los": np.asarray(True)},
        target_masks={
            "channel_cfr": np.ones(24, dtype=np.bool_),
            "los": np.asarray(True),
        },
    )
    second = make_single_record(
        "trajectory-1-step-1",
        targets={"channel_cfr": np.full(24, 2, dtype=np.complex64)},
        target_masks={"channel_cfr": np.asarray([True] * 23 + [False])},
    )
    trajectory = NRTrajectory(
        trajectory_id="route-1", sample_times_s=(0.0, 0.1), records=(first, second)
    )

    values, masks = trajectory.target_series("los")

    assert trajectory.sample_ids == ("trajectory-1-step-0", "trajectory-1-step-1")
    assert trajectory.target_names == ("channel_cfr", "los")
    assert len(trajectory.model_inputs()) == 2
    assert not hasattr(trajectory.model_inputs()[0], "targets")
    assert bool(values[0]) and masks[0]
    assert values[1] is None and masks[1] is None
    assert not bool(trajectory.target_series("channel_cfr")[1][1][-1])


def test_nr_trajectory_rejects_duplicate_samples_and_nonmonotone_times():
    record = make_single_record(
        "one",
        targets={"target": np.asarray(1.0)},
        target_masks={},
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        NRTrajectory(
            "route",
            (0.0, 0.0),
            (
                record,
                make_single_record(
                    "two",
                    targets={"target": np.asarray(2.0)},
                    target_masks={},
                ),
            ),
        )
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        NRTrajectory("route", (0.0, 1.0), (record, record))


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
        target_definitions={
            "geometric_los": TargetDefinition(
                semantic="depth-zero RT visibility oracle", units="boolean", source="sionna_rt"
            ),
            "channel_cfr": TargetDefinition(
                semantic="perfect synchronized CFR",
                units="complex channel gain",
                source="sionna_phy",
            ),
        },
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
    assert restored.target_definitions == record.target_definitions


def test_target_definition_contract_requires_meaning_units_and_source():
    with pytest.raises(ValueError, match="semantic"):
        TargetDefinition(semantic="", units="seconds", source="simulator")

    with pytest.raises(ValueError, match="describe every declared target"):
        NRChannelRecord(
            observation=make_observation(),
            targets={"los": np.asarray(True), "channel_cfr": np.zeros(24)},
            target_masks={},
            provenance={"source": "fixture"},
            target_definitions={
                "los": TargetDefinition("direct visibility", "boolean", "RT oracle")
            },
        )


def test_nr_trajectory_round_trip_preserves_order_masks_and_content_addressing(tmp_path):
    records = (
        make_single_record(
            "route-a-0",
            targets={"los": np.asarray(True), "channel_cfr": np.ones(24, np.complex64)},
            target_masks={"los": np.asarray(True), "channel_cfr": np.ones(24, np.bool_)},
        ),
        make_single_record(
            "route-a-1",
            targets={"channel_cfr": np.full(24, 2, np.complex64)},
            target_masks={"channel_cfr": np.asarray([True] * 12 + [False] * 12)},
        ),
    )
    trajectory = NRTrajectory("route-a", (2.0, 2.1), records)

    stored = write_nr_trajectory(tmp_path, trajectory)
    assert write_nr_trajectory(tmp_path, trajectory) == stored
    restored = read_nr_trajectory(tmp_path, stored.manifest_relative_path)

    assert restored.trajectory_id == trajectory.trajectory_id
    assert restored.sample_times_s == trajectory.sample_times_s
    assert restored.sample_ids == trajectory.sample_ids
    assert restored.target_names == trajectory.target_names
    np.testing.assert_array_equal(
        restored.target_series("channel_cfr")[1][1],
        trajectory.target_series("channel_cfr")[1][1],
    )


def test_nr_trajectory_store_detects_payload_tampering(tmp_path):
    record = make_single_record("route-b-0", targets={"label": np.asarray(1)}, target_masks={})
    stored = write_nr_trajectory(tmp_path, NRTrajectory("route-b", (0.0,), (record,)))
    payload = tmp_path / stored.relative_path
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="content hash"):
        read_nr_trajectory(tmp_path, stored)


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
