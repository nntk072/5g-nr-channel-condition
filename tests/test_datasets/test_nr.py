import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from datasets import audit_nr_trajectory_target_support
from datasets.nr import (
    NRChannelRecord,
    NRObservation,
    NRTrajectory,
    TargetDefinition,
)
from datasets.nr_store import (
    NRStoredRecord,
    read_nr_record,
    read_nr_trajectory,
    write_nr_record,
    write_nr_trajectory,
)
from datasets.targets import TargetValidity
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


def make_single_record(
    sample_id,
    *,
    targets,
    target_masks=None,
    target_reason_codes=None,
    reference_transport_block_bits=None,
    provenance_overrides=None,
):
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
    provenance = {"source": "trajectory-fixture"}
    if reference_transport_block_bits is not None:
        reference_array = np.asarray(reference_transport_block_bits, dtype=np.uint8)
        provenance.update(
            {
                "reference_transport_block_shape": list(reference_array.shape),
                "reference_transport_block_sha256": hashlib.sha256(
                    reference_array.tobytes(order="C")
                ).hexdigest(),
            }
        )
    if provenance_overrides is not None:
        provenance.update(provenance_overrides)
    return NRChannelRecord(
        observation=observation,
        targets=targets,
        target_masks={} if target_masks is None else target_masks,
        provenance=provenance,
        target_definitions={
            name: TargetDefinition.versioned(
                semantic=f"synthetic test target {name}",
                units="fixture unit",
                source="NR dataset test fixture",
                reference="deterministic in-memory fixture values",
                aggregation_rule="preserve each declared target element without pooling",
                definition_version="tests.nr.fixture.v1",
            )
            for name in targets
        },
        target_reason_codes=target_reason_codes or {},
        reference_transport_block_bits=reference_transport_block_bits,
    )


def test_nr_observation_builds_inference_payload_without_sample_oracle_ids():
    observation = make_observation()
    model_inputs = observation.model_inputs()

    assert model_inputs.received_grid.shape == (2, 1, 2, 14, 24)
    assert model_inputs.h_hat.shape == (2, 1, 2, 1, 1, 14, 24)
    assert model_inputs.fft_bin_indices == tuple(range(24))
    assert not hasattr(model_inputs, "sample_id")
    assert not hasattr(model_inputs, "targets")
    assert not hasattr(model_inputs, "target_definitions")


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


@pytest.mark.parametrize(
    ("indices", "exception", "message"),
    [
        (np.asarray([0.0, 1.5]), TypeError, "integer"),
        (np.asarray([False, True]), TypeError, "integer"),
        (np.asarray([[0, 1]]), ValueError, "one-dimensional"),
        ([0, 1.5], TypeError, "only integers"),
    ],
)
def test_receiver_output_factory_rejects_coerced_fft_bin_mappings(
    indices, exception, message
):
    with pytest.raises(exception, match=message):
        NRObservation.from_receiver_output(
            sample_id="invalid-index-types",
            received_grid=np.zeros((2, 1, 2, 14, 24), dtype=np.complex64),
            channel_estimate=make_observation().channel_estimate,
            effective_subcarrier_indices=indices,
            receiver_config=NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        )


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
    ("name", "target", "error", "message"),
    [
        ("nan_target", np.asarray([np.nan]), ValueError, "finite values"),
        ("infinite_target", np.asarray([np.inf]), ValueError, "finite values"),
        ("empty_target", np.asarray([], dtype=np.float32), ValueError, "non-empty"),
        ("", np.asarray([1.0]), ValueError, "target names"),
        ("text_target", np.asarray(["invalid"]), TypeError, "numeric or boolean"),
    ],
)
def test_nr_training_record_rejects_invalid_numeric_targets(
    name: str, target: np.ndarray, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        NRChannelRecord(
            observation=make_observation(),
            targets={name: target},
            target_masks={},
            provenance={"source": "target-validation-test"},
        )


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


def test_nr_channel_record_requires_zero_storage_for_unavailable_targets() -> None:
    with pytest.raises(ValueError, match="masked target elements.*zero storage"):
        NRChannelRecord(
            observation=make_observation(),
            targets={"target": np.asarray([1.0, 2.0])},
            target_masks={"target": np.asarray([True, False])},
            provenance={"source": "unit-test"},
        )

    record = NRChannelRecord(
        observation=make_observation(),
        targets={"target": np.asarray([1.0, 0.0])},
        target_masks={"target": np.asarray([True, False])},
        provenance={"source": "unit-test"},
        target_reason_codes={"target": (TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE)},
    )
    np.testing.assert_array_equal(record.targets["target"], [1.0, 0.0])

    with pytest.raises(ValueError, match="reason codes are required for unavailable"):
        NRChannelRecord(
            observation=make_observation(),
            targets={"target": np.asarray([1.0, 0.0])},
            target_masks={"target": np.asarray([True, False])},
            provenance={"source": "unit-test"},
        )

    with pytest.raises(ValueError, match="masks and reason codes disagree"):
        NRChannelRecord(
            observation=make_observation(),
            targets={"target": np.asarray([1.0, 0.0])},
            target_masks={"target": np.asarray([True, False])},
            provenance={"source": "unit-test"},
            target_reason_codes={"target": (TargetValidity.VALID, TargetValidity.VALID)},
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
        targets={"channel_cfr": np.asarray([2] * 23 + [0], dtype=np.complex64)},
        target_masks={"channel_cfr": np.asarray([True] * 23 + [False])},
        target_reason_codes={
            "channel_cfr": (TargetValidity.VALID,) * 23 + (TargetValidity.NOT_OBSERVABLE,)
        },
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
    assert trajectory.target_reason_series("los") == (
        (TargetValidity.VALID,),
        None,
    )
    assert trajectory.target_reason_series("channel_cfr")[1][-1] is TargetValidity.NOT_OBSERVABLE
    assert not bool(trajectory.target_series("channel_cfr")[1][1][-1])
    assert trajectory.target_support_counts("los") == {
        "record_count": 2,
        "present_record_count": 1,
        "missing_record_count": 1,
        "element_count": 1,
        "selected_element_count": 1,
        "masked_element_count": 0,
        "implicit_all_valid_mask_record_count": 0,
    }
    assert trajectory.target_support_counts("channel_cfr") == {
        "record_count": 2,
        "present_record_count": 2,
        "missing_record_count": 0,
        "element_count": 48,
        "selected_element_count": 47,
        "masked_element_count": 1,
        "implicit_all_valid_mask_record_count": 0,
    }


def test_trajectory_support_counts_report_implicit_all_valid_masks():
    record = make_single_record(
        "implicit-mask-step",
        targets={"scalar": np.asarray(3.0)},
        target_masks={},
    )
    trajectory = NRTrajectory("implicit-mask", (0.0,), (record,))

    assert trajectory.target_support_counts("scalar") == {
        "record_count": 1,
        "present_record_count": 1,
        "missing_record_count": 0,
        "element_count": 1,
        "selected_element_count": 1,
        "masked_element_count": 0,
        "implicit_all_valid_mask_record_count": 1,
    }


def test_nr_target_support_audit_reports_expected_split_family_cells():
    train = NRTrajectory(
        "train-route",
        (0.0, 0.1),
        (
            make_single_record("train-0", targets={"los": np.asarray(True)}, target_masks={}),
            make_single_record(
                "train-1",
                targets={"channel_cfr": np.asarray([1, 0, 1], dtype=np.complex64)},
                target_masks={"channel_cfr": np.asarray([True, False, True])},
                target_reason_codes={
                    "channel_cfr": (
                        TargetValidity.VALID,
                        TargetValidity.NOT_OBSERVABLE,
                        TargetValidity.VALID,
                    )
                },
            ),
        ),
    )
    test = NRTrajectory(
        "test-route",
        (1.0, 1.1),
        (
            make_single_record("test-0", targets={"channel_cfr": np.ones(3)}, target_masks={}),
            make_single_record("test-1", targets={"channel_cfr": np.ones(3)}, target_masks={}),
        ),
    )

    report = audit_nr_trajectory_target_support(
        (("train", "site-a", train), ("test", "site-a", test)),
        (
            ("train", "site-a", "los"),
            ("train", "site-a", "channel_cfr"),
            ("test", "site-a", "los"),
            ("test", "site-a", "channel_cfr"),
        ),
    )

    assert report["schema"] == "umcd.nr_target_support_audit.v1"
    assert report["status"] == "INPUT_SCOPED_NOT_RELEASE_QUALIFIED"
    assert report["trajectory_count"] == 2
    assert report["cells"] == [
        {
            "split_id": "test",
            "family_id": "site-a",
            "target_id": "channel_cfr",
            "trajectory_count": 1,
            "record_count": 2,
            "present_record_count": 2,
            "missing_record_count": 0,
            "element_count": 6,
            "selected_element_count": 6,
            "masked_element_count": 0,
            "implicit_all_valid_mask_record_count": 2,
        },
        {
            "split_id": "test",
            "family_id": "site-a",
            "target_id": "los",
            "trajectory_count": 1,
            "record_count": 2,
            "present_record_count": 0,
            "missing_record_count": 2,
            "element_count": 0,
            "selected_element_count": 0,
            "masked_element_count": 0,
            "implicit_all_valid_mask_record_count": 0,
        },
        {
            "split_id": "train",
            "family_id": "site-a",
            "target_id": "channel_cfr",
            "trajectory_count": 1,
            "record_count": 2,
            "present_record_count": 1,
            "missing_record_count": 1,
            "element_count": 3,
            "selected_element_count": 2,
            "masked_element_count": 1,
            "implicit_all_valid_mask_record_count": 0,
        },
        {
            "split_id": "train",
            "family_id": "site-a",
            "target_id": "los",
            "trajectory_count": 1,
            "record_count": 2,
            "present_record_count": 1,
            "missing_record_count": 1,
            "element_count": 1,
            "selected_element_count": 1,
            "masked_element_count": 0,
            "implicit_all_valid_mask_record_count": 1,
        },
    ]


def test_nr_target_support_audit_rejects_duplicate_trajectory_assignments():
    trajectory = NRTrajectory(
        "reused-route",
        (0.0,),
        (
            make_single_record(
                "reused-sample", targets={"target": np.asarray(1.0)}, target_masks={}
            ),
        ),
    )

    with pytest.raises(ValueError, match="duplicate trajectory assignment"):
        audit_nr_trajectory_target_support(
            (("train", "family", trajectory), ("test", "family", trajectory)),
            (("train", "family", "target"), ("test", "family", "target")),
        )


def test_nr_target_support_audit_rejects_reused_samples_across_trajectories():
    record = make_single_record(
        "shared-sample", targets={"target": np.asarray(1.0)}, target_masks={}
    )
    first = NRTrajectory("route-first", (0.0,), (record,))
    second = NRTrajectory("route-second", (1.0,), (record,))

    with pytest.raises(ValueError, match="sample IDs may not be counted more than once"):
        audit_nr_trajectory_target_support(
            (("train", "family", first), ("train", "family", second)),
            (("train", "family", "target"),),
        )


def test_nr_target_support_audit_keeps_expected_empty_cells_visible():
    report = audit_nr_trajectory_target_support((), (("test", "site-a", "los"),))

    assert report["trajectory_count"] == 0
    assert report["cells"] == [
        {
            "split_id": "test",
            "family_id": "site-a",
            "target_id": "los",
            "trajectory_count": 0,
            "record_count": 0,
            "present_record_count": 0,
            "missing_record_count": 0,
            "element_count": 0,
            "selected_element_count": 0,
            "masked_element_count": 0,
            "implicit_all_valid_mask_record_count": 0,
        }
    ]


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


@pytest.mark.parametrize("invalid_time", ["0.1", 0.1 + 0.0j, True])
def test_nr_trajectory_rejects_coercible_nonreal_timestamps(invalid_time):
    record = make_single_record("one", targets={"target": np.asarray(1.0)}, target_masks={})
    with pytest.raises(TypeError, match="real numeric timestamps"):
        NRTrajectory("route", (invalid_time,), (record,))


@pytest.mark.parametrize("trajectory_id", [None, 1, True])
def test_nr_trajectory_rejects_non_string_identity(trajectory_id):
    record = make_single_record(
        "one",
        targets={"target": np.asarray(1.0)},
        target_masks={},
    )

    with pytest.raises(TypeError, match="trajectory_id must be a string"):
        NRTrajectory(trajectory_id, (0.0,), (record,))


@pytest.mark.parametrize("trajectory_id", ["", " ", " route", "route "])
def test_nr_trajectory_rejects_noncanonical_identity(trajectory_id):
    record = make_single_record(
        "one",
        targets={"target": np.asarray(1.0)},
        target_masks={},
    )

    with pytest.raises(ValueError, match="non-empty canonical string"):
        NRTrajectory(trajectory_id, (0.0,), (record,))


def test_nr_record_round_trip_is_lossless_pickle_free_and_content_addressed(tmp_path):
    observation = make_observation()
    reference_bits = np.asarray([[1, 0, 1, 1, 0]], dtype=np.uint8)
    record = NRChannelRecord(
        observation=observation,
        targets={
            "geometric_los": np.asarray(True),
            "channel_cfr": (np.arange(24) + 1j * np.arange(24)).astype(np.complex64),
        },
        target_masks={"geometric_los": np.asarray(True)},
        provenance={
            "scene_hash": "sha256:abc",
            "simulator": "Sionna 2.2",
            "reference_transport_block_shape": list(reference_bits.shape),
            "reference_transport_block_sha256": hashlib.sha256(
                reference_bits.tobytes(order="C")
            ).hexdigest(),
        },
        reference_transport_block_bits=reference_bits,
        target_definitions={
            "geometric_los": TargetDefinition.versioned(
                semantic="depth-zero RT visibility oracle",
                units="boolean",
                source="sionna_rt",
                reference="test fixture visibility value",
                aggregation_rule="one visibility value per sample",
                definition_version="tests.nr.roundtrip-los.v1",
            ),
            "channel_cfr": TargetDefinition.versioned(
                semantic="perfect synchronized CFR",
                units="complex channel gain",
                source="sionna_phy",
                reference="test fixture complex CFR vector",
                aggregation_rule="retain each subcarrier value without pooling",
                definition_version="tests.nr.roundtrip-cfr.v1",
            ),
        },
    )

    stored = write_nr_record(tmp_path, record)
    assert stored.schema_version == 4
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
    assert restored.target_reason_codes == record.target_reason_codes
    np.testing.assert_array_equal(
        restored.reference_transport_block_bits, record.reference_transport_block_bits
    )
    assert record.provenance["reference_transport_block_shape"] == (1, 5)
    with pytest.raises(TypeError):
        record.provenance["reference_transport_block_shape"][0] = 2
    assert not hasattr(observation.model_inputs(), "reference_transport_block_bits")
    with np.load(tmp_path / stored.relative_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/nr-record-metadata.v4.schema.json"
    )
    import jsonschema

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(metadata)


@pytest.mark.parametrize(
    "reference_bits",
    (
        np.asarray([], dtype=np.uint8),
        np.asarray([[0, 2]], dtype=np.uint8),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    ),
)
def test_nr_record_rejects_invalid_reference_transport_bits(reference_bits):
    with pytest.raises(ValueError, match="reference_transport_block_bits"):
        make_single_record(
            "invalid-reference-payload",
            targets={"label": np.asarray(1)},
            reference_transport_block_bits=reference_bits,
        )


@pytest.mark.parametrize(
    "provenance_overrides",
    (
        {"reference_transport_block_shape": [1, 9]},
        {"reference_transport_block_sha256": "0" * 64},
    ),
)
def test_nr_record_rejects_reference_payload_provenance_drift(provenance_overrides):
    with pytest.raises(ValueError, match="provenance shape/digest"):
        make_single_record(
            "reference-payload-provenance-drift",
            targets={"label": np.asarray(1)},
            reference_transport_block_bits=np.asarray([[1, 0, 1]], dtype=np.uint8),
            provenance_overrides=provenance_overrides,
        )


def test_nr_v4_reader_rejects_rehashed_reference_payload_provenance_drift(tmp_path):
    record = make_single_record(
        "stored-reference-payload-provenance-drift",
        targets={"label": np.asarray(1)},
        reference_transport_block_bits=np.asarray([[1, 0, 1]], dtype=np.uint8),
    )
    stored = write_nr_record(tmp_path, record)
    with np.load(tmp_path / stored.relative_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
    metadata["provenance"]["reference_transport_block_sha256"] = "0" * 64
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
        dtype=np.uint8,
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = f"records/{digest[:2]}/{digest}.npz"
    object_path = tmp_path / relative_path
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(payload)

    forged = NRStoredRecord(
        sample_id=stored.sample_id,
        content_sha256=digest,
        relative_path=relative_path,
        schema_version=4,
    )
    with pytest.raises(ValueError, match="provenance shape/digest"):
        read_nr_record(tmp_path, forged)


def test_nr_writer_rejects_reference_payload_mutated_after_record_admission(tmp_path):
    record = make_single_record(
        "mutated-reference-payload",
        targets={"label": np.asarray(1)},
        reference_transport_block_bits=np.asarray([[1, 0, 1]], dtype=np.uint8),
    )
    record.reference_transport_block_bits.setflags(write=True)
    record.reference_transport_block_bits[0, 0] = 0

    with pytest.raises(ValueError, match="provenance shape/digest"):
        write_nr_record(tmp_path, record)
    assert not (tmp_path / "records").exists()


def test_nr_v2_record_without_target_definitions_remains_readable(tmp_path):
    record = make_single_record("historical-v2", targets={"label": np.asarray(1)})
    current = write_nr_record(tmp_path, record)
    with np.load(tmp_path / current.relative_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
    metadata["schema_version"] = 2
    metadata["target_definitions"] = {}
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
        dtype=np.uint8,
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = f"records/{digest[:2]}/{digest}.npz"
    object_path = tmp_path / relative_path
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(payload)

    restored = read_nr_record(
        tmp_path,
        NRStoredRecord(
            sample_id="historical-v2",
            content_sha256=digest,
            relative_path=relative_path,
            schema_version=2,
        ),
    )

    assert restored.targets["label"].item() == 1
    assert restored.target_definitions == {}


def test_nr_record_manifest_publication_failure_is_retryable(tmp_path, monkeypatch):
    import datasets.nr_store as nr_store

    record = make_single_record("retryable-record", targets={"label": np.asarray(1)})
    publish = nr_store._publish_immutable
    failed = False

    def fail_before_manifest(path, payload):
        nonlocal failed
        if not failed and "manifests" in path.parts:
            failed = True
            raise OSError("injected manifest publication failure")
        publish(path, payload)

    monkeypatch.setattr(nr_store, "_publish_immutable", fail_before_manifest)
    with pytest.raises(OSError, match="injected manifest publication failure"):
        write_nr_record(tmp_path, record)

    assert not (tmp_path / "manifests").exists()
    assert list((tmp_path / "records").rglob("*.npz"))

    stored = write_nr_record(tmp_path, record)
    assert read_nr_record(tmp_path, stored).observation.sample_id == "retryable-record"
    assert not [path for path in tmp_path.rglob(".*") if path.is_file()]


def test_nr_record_writer_requires_complete_versioned_target_semantics(tmp_path):
    missing = NRChannelRecord(
        observation=make_observation(),
        targets={"label": np.asarray(1)},
        target_masks={},
        provenance={"source": "test"},
    )
    with pytest.raises(ValueError, match="one target definition for every target"):
        write_nr_record(tmp_path, missing)
    assert not (tmp_path / "records").exists()

    legacy = NRChannelRecord(
        observation=make_observation(),
        targets={"label": np.asarray(1)},
        target_masks={},
        provenance={"source": "test"},
        target_definitions={"label": TargetDefinition("legacy label", "class", "test")},
    )
    with pytest.raises(ValueError, match="is not versioned"):
        write_nr_record(tmp_path, legacy)
    assert not (tmp_path / "records").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("masked_target", "masked target elements.*zero storage"),
        ("mask_reason", "masks and reason codes disagree"),
    ],
)
def test_nr_record_writer_revalidates_mutable_target_arrays(tmp_path, mutate, message):
    target = np.asarray([1.0, 0.0], dtype=np.float32)
    mask = np.asarray([True, False], dtype=bool)
    record = make_single_record(
        "mutated-target-record",
        targets={"label": target},
        target_masks={"label": mask},
        target_reason_codes={
            "label": (TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE)
        },
    )
    if mutate == "masked_target":
        target[1] = 4.0
    else:
        mask[1] = True

    with pytest.raises(ValueError, match=message):
        write_nr_record(tmp_path, record)
    assert not (tmp_path / "records").exists()


def test_nr_record_path_reader_bounds_and_stabilizes_manifest_snapshot(tmp_path, monkeypatch):
    import datasets.nr_store as nr_store

    record = make_single_record("bounded-record-manifest", targets={"label": np.asarray(1)})
    stored = write_nr_record(tmp_path, record)
    sample_key = hashlib.sha256(stored.sample_id.encode()).hexdigest()
    manifest_path = tmp_path / "manifests" / f"{sample_key}.json"
    monkeypatch.setattr(nr_store, "_MAX_NR_RECORD_MANIFEST_BYTES", 16)

    with pytest.raises(ValueError, match="NR record manifest exceeds the 16-byte limit"):
        read_nr_record(tmp_path, manifest_path)


def test_nr_v1_record_migrates_missing_reasons_as_explicit_legacy(tmp_path):
    record = make_single_record(
        "legacy-reason-record",
        targets={"target": np.asarray([3.0, 0.0], dtype=np.float32)},
        target_masks={"target": np.asarray([True, False])},
        target_reason_codes={"target": (TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE)},
    )
    original = write_nr_record(tmp_path, record)
    with np.load(tmp_path / original.relative_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
    metadata["schema_version"] = 1
    for reason_array_key in metadata.pop("target_reason_codes").values():
        arrays.pop(reason_array_key)
    metadata.pop("target_reason_codebook")
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
        dtype=np.uint8,
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = f"records/{digest[:2]}/{digest}.npz"
    target_path = tmp_path / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(payload)

    restored = read_nr_record(
        tmp_path,
        NRStoredRecord(
            sample_id=record.observation.sample_id,
            content_sha256=digest,
            relative_path=relative_path,
            schema_version=1,
        ),
    )

    assert restored.target_reason_codes["target"] == (
        TargetValidity.VALID,
        TargetValidity.LEGACY_REASON_UNRECORDED,
    )
    np.testing.assert_array_equal(restored.targets["target"], [3.0, 0.0])


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


def test_trajectory_rejects_target_definition_drift():
    first = make_single_record(
        "defined-0",
        targets={"channel_cfr": np.ones(24, np.complex64)},
        target_masks={"channel_cfr": np.ones(24, np.bool_)},
    )
    second = make_single_record(
        "defined-1",
        targets={"channel_cfr": np.ones(24, np.complex64)},
        target_masks={"channel_cfr": np.ones(24, np.bool_)},
    )
    definition_a = TargetDefinition("effective CFR", "dimensionless", "Sionna PHY")
    definition_b = TargetDefinition("absolute CFR", "dimensionless", "Sionna RT")
    first = NRChannelRecord(
        first.observation,
        first.targets,
        first.target_masks,
        first.provenance,
        {"channel_cfr": definition_a},
    )
    second = NRChannelRecord(
        second.observation,
        second.targets,
        second.target_masks,
        second.provenance,
        {"channel_cfr": definition_b},
    )

    with pytest.raises(ValueError, match="changes within one trajectory"):
        NRTrajectory("definition-drift", (0.0, 0.1), (first, second))


def test_nr_trajectory_round_trip_preserves_order_masks_and_content_addressing(tmp_path):
    records = (
        make_single_record(
            "route-a-0",
            targets={"los": np.asarray(True), "channel_cfr": np.ones(24, np.complex64)},
            target_masks={"los": np.asarray(True), "channel_cfr": np.ones(24, np.bool_)},
        ),
        make_single_record(
            "route-a-1",
            targets={"channel_cfr": np.asarray([2] * 12 + [0] * 12, np.complex64)},
            target_masks={"channel_cfr": np.asarray([True] * 12 + [False] * 12)},
            target_reason_codes={
                "channel_cfr": (TargetValidity.VALID,) * 12 + (TargetValidity.NOT_OBSERVABLE,) * 12
            },
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


def test_nr_trajectory_manifest_publication_failure_is_retryable(tmp_path, monkeypatch):
    import datasets.nr_store as nr_store

    trajectory = NRTrajectory(
        "retryable-route",
        (0.0,),
        (make_single_record("retryable-route-0", targets={"label": np.asarray(1)}),),
    )
    publish = nr_store._publish_immutable
    failed = False

    def fail_before_trajectory_manifest(path, payload):
        nonlocal failed
        if not failed and "trajectories" in path.parts and "manifests" in path.parts:
            failed = True
            raise OSError("injected trajectory manifest publication failure")
        publish(path, payload)

    monkeypatch.setattr(nr_store, "_publish_immutable", fail_before_trajectory_manifest)
    with pytest.raises(OSError, match="injected trajectory manifest publication failure"):
        write_nr_trajectory(tmp_path, trajectory)

    trajectory_manifest_dir = tmp_path / "manifests" / "trajectories"
    assert not trajectory_manifest_dir.exists()
    assert list((tmp_path / "manifests").glob("*.json"))

    stored = write_nr_trajectory(tmp_path, trajectory)
    restored = read_nr_trajectory(tmp_path, stored.manifest_relative_path)
    assert restored.trajectory_id == trajectory.trajectory_id
    assert restored.sample_ids == trajectory.sample_ids
    assert not [path for path in tmp_path.rglob(".*") if path.is_file()]


def test_nr_trajectory_partial_record_publication_is_retryable(tmp_path, monkeypatch):
    import datasets.nr_store as nr_store

    trajectory = NRTrajectory(
        "partial-record-route",
        (0.0, 0.1),
        (
            make_single_record(
                "partial-record-route-0",
                targets={"label": np.asarray(0)},
                target_masks={},
            ),
            make_single_record(
                "partial-record-route-1",
                targets={"label": np.asarray(1)},
                target_masks={},
            ),
        ),
    )
    publish = nr_store._publish_immutable
    record_publications = 0
    injected = False

    def fail_before_second_record_object(path, payload):
        nonlocal record_publications, injected
        if "records" in path.parts and path.suffix == ".npz":
            record_publications += 1
            if record_publications == 2 and not injected:
                injected = True
                raise OSError("injected second trajectory record publication failure")
        publish(path, payload)

    monkeypatch.setattr(nr_store, "_publish_immutable", fail_before_second_record_object)
    with pytest.raises(OSError, match="injected second trajectory record publication failure"):
        write_nr_trajectory(tmp_path, trajectory)

    assert len(list((tmp_path / "manifests").glob("*.json"))) == 1
    assert not (tmp_path / "manifests" / "trajectories").exists()

    stored = write_nr_trajectory(tmp_path, trajectory)
    restored = read_nr_trajectory(tmp_path, stored.manifest_relative_path)
    assert restored.trajectory_id == trajectory.trajectory_id
    assert restored.sample_ids == trajectory.sample_ids
    assert not [path for path in tmp_path.rglob(".*") if path.is_file()]


def test_nr_trajectory_store_detects_payload_tampering(tmp_path):
    record = make_single_record("route-b-0", targets={"label": np.asarray(1)}, target_masks={})
    stored = write_nr_trajectory(tmp_path, NRTrajectory("route-b", (0.0,), (record,)))
    payload = tmp_path / stored.relative_path
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="content hash"):
        read_nr_trajectory(tmp_path, stored)


def test_nr_trajectory_store_rejects_duplicate_manifest_keys(tmp_path):
    record = make_single_record("route-c-0", targets={"label": np.asarray(1)}, target_masks={})
    stored = write_nr_trajectory(tmp_path, NRTrajectory("route-c", (0.0,), (record,)))
    manifest_path = tmp_path / stored.manifest_relative_path
    manifest_path.write_text(
        '{"trajectory_id":"forged","trajectory_id":"route-c",'
        '"content_sha256":"' + stored.content_sha256 + '",'
        '"relative_path":"' + stored.relative_path + '",'
        '"sample_ids":["route-c-0"],"schema_version":1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        read_nr_trajectory(tmp_path, stored.manifest_relative_path)


def test_nr_trajectory_store_rejects_oversized_manifest_before_loading_content(
    tmp_path, monkeypatch
):
    record = make_single_record("route-large-manifest-0", targets={"label": np.asarray(1)})
    stored = write_nr_trajectory(tmp_path, NRTrajectory("route-large-manifest", (0.0,), (record,)))
    manifest_path = tmp_path / stored.manifest_relative_path
    manifest_path.write_bytes(b" " * 33)

    import datasets.nr_store as nr_store

    monkeypatch.setattr(nr_store, "_MAX_NR_TRAJECTORY_MANIFEST_BYTES", 32)
    real_open = nr_store._open_store_record

    def reject_content_open(base, relative_path):
        if relative_path == stored.relative_path:
            pytest.fail("oversized manifest must fail before opening trajectory content")
        return real_open(base, relative_path)

    monkeypatch.setattr(nr_store, "_open_store_record", reject_content_open)
    with pytest.raises(ValueError, match="NR trajectory manifest exceeds the 32-byte limit"):
        read_nr_trajectory(tmp_path, stored.manifest_relative_path)


def test_nr_trajectory_store_rejects_hardlinked_manifest(tmp_path):
    record = make_single_record("route-hardlink-manifest-0", targets={"label": np.asarray(1)})
    stored = write_nr_trajectory(
        tmp_path, NRTrajectory("route-hardlink-manifest", (0.0,), (record,))
    )
    manifest_path = tmp_path / stored.manifest_relative_path
    hardlink = tmp_path / "trajectory-manifest-copy.json"
    try:
        os.link(manifest_path, hardlink)
    except OSError as exc:
        pytest.skip(f"hard-link creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="single-link regular file"):
        read_nr_trajectory(tmp_path, stored.manifest_relative_path)


def test_nr_trajectory_store_rejects_oversized_content_before_hashing(tmp_path, monkeypatch):
    record = make_single_record("route-large-content-0", targets={"label": np.asarray(1)})
    stored = write_nr_trajectory(tmp_path, NRTrajectory("route-large-content", (0.0,), (record,)))
    (tmp_path / stored.relative_path).write_bytes(b"x" * 17)

    import datasets.nr_store as nr_store

    monkeypatch.setattr(nr_store, "_MAX_NR_TRAJECTORY_CONTENT_BYTES", 16)
    with pytest.raises(ValueError, match="NR trajectory content exceeds the 16-byte limit"):
        read_nr_trajectory(tmp_path, stored)


def test_nr_trajectory_writer_rejects_content_over_configured_limit(tmp_path, monkeypatch):
    import datasets.nr_store as nr_store

    monkeypatch.setattr(nr_store, "_MAX_NR_TRAJECTORY_CONTENT_BYTES", 16)
    record = make_single_record("route-write-limit-0", targets={"label": np.asarray(1)})
    with pytest.raises(ValueError, match="NR trajectory content exceeds the 16-byte limit"):
        write_nr_trajectory(tmp_path, NRTrajectory("route-write-limit", (0.0,), (record,)))


def test_nr_trajectory_writer_rejects_manifest_over_limit_before_content_publish(
    tmp_path, monkeypatch
):
    import datasets.nr_store as nr_store

    monkeypatch.setattr(nr_store, "_MAX_NR_TRAJECTORY_MANIFEST_BYTES", 16)
    record = make_single_record("route-pointer-limit-0", targets={"label": np.asarray(1)})
    with pytest.raises(ValueError, match="NR trajectory manifest exceeds the 16-byte limit"):
        write_nr_trajectory(tmp_path, NRTrajectory("route-pointer-limit", (0.0,), (record,)))
    assert not (tmp_path / "trajectories").exists()


def test_nr_trajectory_store_rejects_symlinked_content_object(tmp_path):
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform cannot open content objects without following symlinks")
    record = make_single_record("route-d-0", targets={"label": np.asarray(1)}, target_masks={})
    stored = write_nr_trajectory(tmp_path, NRTrajectory("route-d", (0.0,), (record,)))
    payload_path = tmp_path / stored.relative_path
    original_path = payload_path.with_name("original-trajectory.json")
    payload_path.rename(original_path)
    payload_path.symlink_to(original_path.name)

    with pytest.raises(OSError):
        read_nr_trajectory(tmp_path, stored)


def test_nr_record_store_detects_tampering_and_sample_id_mutation(tmp_path):
    record = NRChannelRecord(
        observation=make_observation(),
        targets={"geometric_los": np.asarray(True)},
        target_masks={},
        provenance={"scene_hash": "sha256:abc"},
        target_definitions={
            "geometric_los": TargetDefinition.versioned(
                semantic="test LOS label",
                units="boolean",
                source="test fixture",
                reference="fixed test value",
                aggregation_rule="one label per record",
                definition_version="tests.nr.tamper-los.v1",
            )
        },
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
                observation=make_observation(
                    received_grid=np.ones((2, 1, 2, 14, 24), dtype=np.complex64)
                ),
                targets={"geometric_los": np.asarray(True)},
                target_masks={},
                provenance={"scene_hash": "sha256:abc"},
                target_definitions=record.target_definitions,
            ),
        )
