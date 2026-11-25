import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets import audit_nr_trajectory_target_support
from datasets.nr import (
    LinkedNRObservation,
    NRChannelRecord,
    NRObservation,
    NRReceiverInferenceRecord,
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


def test_numpy_nr_record_construction_does_not_import_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def reject_eager_torch_import(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("NumPy-only NR record construction must not import Torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_eager_torch_import)
    record = make_single_record("numpy-only-record", targets={"label": np.asarray(1)})
    assert record.observation.received_grid.dtype == np.dtype(np.complex64)


def test_torch_nr_observation_and_target_validation_remains_supported():
    torch = pytest.importorskip("torch")
    estimate = NRChannelEstimate(
        h_hat=torch.zeros((1, 1, 2, 1, 1, 14, 24), dtype=torch.complex64),
        error_variance=torch.zeros((1, 1, 1, 1, 1, 14, 24), dtype=torch.float32),
        pilot_mask=torch.zeros((1, 1, 14, 24), dtype=torch.bool),
        metadata={"receiver": "torch-contract-test"},
    )
    observation = NRObservation(
        sample_id="torch-observation",
        received_grid=torch.zeros((1, 1, 2, 14, 24), dtype=torch.complex64),
        channel_estimate=estimate,
        fft_bin_indices=tuple(range(24)),
        receiver_config=NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
    )
    record = NRChannelRecord(
        observation=observation,
        targets={"label": torch.asarray(1.0)},
        target_masks={},
        provenance={"source": "torch-contract-test"},
    )

    assert torch.is_tensor(observation.received_grid_mask)
    assert record.targets["label"].shape == torch.Size([])


def test_nr_observation_builds_inference_payload_without_sample_oracle_ids():
    observation = make_observation()
    model_inputs = observation.model_inputs()

    assert model_inputs.received_grid.shape == (2, 1, 2, 14, 24)
    assert model_inputs.h_hat.shape == (2, 1, 2, 1, 1, 14, 24)
    assert model_inputs.fft_bin_indices == tuple(range(24))
    assert not hasattr(model_inputs, "sample_id")
    assert not hasattr(model_inputs, "targets")
    assert not hasattr(model_inputs, "target_definitions")


@pytest.mark.parametrize(
    ("grid", "exception", "message"),
    [
        (np.zeros((2, 1, 2, 14, 24), dtype=np.float32), TypeError, "must be complex-valued"),
        (np.full((2, 1, 2, 14, 24), np.nan + 0j, dtype=np.complex64), ValueError, "finite values"),
    ],
)
def test_nr_observation_audits_receiver_grid_at_construction(grid, exception, message):
    with pytest.raises(exception, match=message):
        make_observation(received_grid=grid)


def test_nr_observation_requires_canonical_sample_id_and_tuple_fft_mapping():
    with pytest.raises(ValueError, match="canonical string"):
        make_observation(sample_id=" padded-id ")
    with pytest.raises(TypeError, match="must be a tuple"):
        make_observation(fft_bin_indices=list(range(24)))


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
        (7, TypeError, "one-dimensional integer sequence"),
    ],
)
def test_receiver_output_factory_rejects_coerced_fft_bin_mappings(indices, exception, message):
    with pytest.raises(exception, match=message):
        NRObservation.from_receiver_output(
            sample_id="invalid-index-types",
            received_grid=np.zeros((2, 1, 2, 14, 24), dtype=np.complex64),
            channel_estimate=make_observation().channel_estimate,
            effective_subcarrier_indices=indices,
            receiver_config=NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        )


def test_receiver_output_factory_accepts_torch_integer_bins_and_rejects_other_dtypes():
    torch = pytest.importorskip("torch")
    observation = make_observation()
    kwargs = {
        "sample_id": "torch-index-types",
        "received_grid": observation.received_grid,
        "channel_estimate": observation.channel_estimate,
        "receiver_config": observation.receiver_config,
    }

    accepted = NRObservation.from_receiver_output(
        **kwargs, effective_subcarrier_indices=torch.arange(24, dtype=torch.int64)
    )
    assert accepted.fft_bin_indices == tuple(range(24))
    for values, error, message in (
        (torch.arange(24, dtype=torch.int64).reshape(2, 12), ValueError, "one-dimensional"),
        (torch.ones(24, dtype=torch.bool), TypeError, "integer dtype"),
        (torch.arange(24, dtype=torch.float32), TypeError, "integer dtype"),
        (torch.ones(24, dtype=torch.complex64), TypeError, "integer dtype"),
    ):
        with pytest.raises(error, match=message):
            NRObservation.from_receiver_output(**kwargs, effective_subcarrier_indices=values)


def test_torch_tensor_candidate_falls_back_when_torch_import_is_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    import datasets.nr as nr_module

    def unavailable(_name):
        raise ImportError("Torch intentionally hidden by test")

    monkeypatch.setattr(nr_module, "import_module", unavailable)
    assert nr_module._torch_module_for_tensor_candidate(torch.ones(1)) is None


def test_linked_observation_requires_one_sample_and_typed_observation():
    observation = make_observation()
    with pytest.raises(ValueError, match="exactly one batch item"):
        LinkedNRObservation("link-1", 0, observation)
    with pytest.raises(TypeError, match="must be an NRObservation"):
        LinkedNRObservation("link-1", 0, object())


def test_torch_target_masks_enforce_zero_storage_for_unavailable_values():
    torch = pytest.importorskip("torch")
    target = torch.asarray([1.0, 2.0])
    mask = torch.asarray([True, False])

    with pytest.raises(ValueError, match="masked target elements.*zero storage"):
        make_single_record(
            "torch-masked-invalid", targets={"target": target}, target_masks={"target": mask}
        )

    accepted = make_single_record(
        "torch-masked-valid",
        targets={"target": torch.asarray([1.0, 0.0])},
        target_masks={"target": mask},
        target_reason_codes={"target": (TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE)},
    )
    assert torch.equal(accepted.targets["target"], torch.asarray([1.0, 0.0]))


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


def test_nr_observation_rejects_duplicate_fft_bin_mapping():
    with pytest.raises(ValueError, match="fft_bin_indices must be unique"):
        make_observation(fft_bin_indices=(0,) * 24)


@pytest.mark.parametrize(
    ("link_id", "frame_index", "observation", "message"),
    [
        ("link", True, None, "frame_index must be an integer"),
        ("link", 0, object(), "observation must be an NRObservation"),
        ("link", 0, make_observation(), "exactly one batch item"),
    ],
)
def test_linked_nr_observation_rejects_invalid_frame_binding(
    link_id, frame_index, observation, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        LinkedNRObservation(link_id, frame_index, observation)


def test_linked_nr_observation_accepts_single_item_binding():
    observation = make_single_record(
        "single-linked-observation", targets={"target": np.asarray(1.0)}
    ).observation

    linked = LinkedNRObservation("link", 0, observation)

    assert linked.observation.sample_id == "single-linked-observation"


def test_nr_record_rejects_missing_provenance_and_bad_target_definitions():
    record = make_single_record("empty-provenance", targets={"target": np.asarray(1.0)})
    with pytest.raises(ValueError, match="provenance must not be empty"):
        replace(record, provenance={})

    record = make_single_record("bad-definition", targets={"target": np.asarray(1.0)})
    with pytest.raises(TypeError, match="TargetDefinition values"):
        replace(record, target_definitions={"target": object()})


def test_nr_record_rejects_non_boolean_scalar_mask():
    with pytest.raises(TypeError, match="target mask 'target' must have boolean dtype"):
        make_single_record(
            "integer-mask",
            targets={"target": np.asarray(1.0)},
            target_masks={"target": 1},
        )

    record = make_single_record(
        "boolean-mask",
        targets={"target": np.asarray(1.0)},
        target_masks={"target": True},
    )
    assert record.target_masks["target"] is True


def test_nr_target_definition_axes_must_match_target_rank():
    record = make_single_record("bad-axis-rank", targets={"metric": np.asarray(1.0)})
    definition = TargetDefinition.versioned(
        semantic="synthetic scalar metric",
        units="fixture unit",
        source="NR dataset test fixture",
        reference="deterministic in-memory fixture values",
        aggregation_rule="one scalar value",
        definition_version="tests.nr.fixture.scalar.v1",
        value_axes=("element",),
    )

    with pytest.raises(ValueError, match="rank must match its declared value axes"):
        replace(
            record, target_definitions={"metric": definition}
        ).require_versioned_target_definitions()


def test_trajectory_target_accessors_reject_unknown_target_names():
    record = make_single_record("unknown-target-step", targets={"known": np.asarray(1.0)})
    trajectory = NRTrajectory("unknown-target-route", (0.0,), (record,))

    with pytest.raises(KeyError, match="unknown trajectory target: missing"):
        trajectory.target_series("missing")
    with pytest.raises(KeyError, match="unknown trajectory target: missing"):
        trajectory.target_reason_series("missing")
    with pytest.raises(KeyError, match="unknown trajectory target: missing"):
        trajectory.target_support_counts("missing")


def test_trajectory_rejects_mixed_receiver_configurations():
    first = make_single_record("config-first", targets={"target": np.asarray(1.0)})
    second = make_single_record("config-second", targets={"target": np.asarray(1.0)})
    second_observation = replace(
        second.observation,
        receiver_config=NRPUSCHConfig(n_size_bwp=4, subcarrier_spacing_khz=30),
    )

    with pytest.raises(ValueError, match="same receiver config"):
        NRTrajectory(
            "mixed-config-route",
            (0.0, 1.0),
            (first, replace(second, observation=second_observation)),
        )


@pytest.mark.parametrize(
    ("assignments", "expected_cells", "error", "message"),
    [
        ((), (), ValueError, "expected_cells must declare"),
        ((), (("split", "family"),), ValueError, "each expected cell must be"),
        (((),), (("split", "family", "target"),), ValueError, "each assignment must be"),
        (
            (("split", "family", object()),),
            (("split", "family", "target"),),
            TypeError,
            "trajectory must be an NRTrajectory",
        ),
    ],
)
def test_nr_target_support_audit_rejects_malformed_rosters(
    assignments, expected_cells, error, message
):
    with pytest.raises(error, match=message):
        audit_nr_trajectory_target_support(assignments, expected_cells)


def test_nr_target_support_audit_requires_expected_cell_for_each_assignment():
    trajectory = NRTrajectory(
        "undeclared-route",
        (0.0,),
        (make_single_record("undeclared-sample", targets={"target": np.asarray(1.0)}),),
    )

    with pytest.raises(ValueError, match="has no expected coverage cells"):
        audit_nr_trajectory_target_support(
            (("split", "family", trajectory),), (("other", "family", "target"),)
        )


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
    assert stored.schema_version == 5
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
        Path(__file__).resolve().parents[2] / "datasets/schema/nr-record-metadata.v5.schema.json"
    )
    import jsonschema

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(metadata)


def test_nr_v5_round_trip_preserves_digest_bound_target_axes(tmp_path):
    record = make_single_record("axis-bound", targets={"metric": np.asarray([1.0])})
    definition = TargetDefinition.versioned(
        semantic="synthetic axis-bound metric",
        units="fixture unit",
        source="NR dataset test fixture",
        reference="deterministic in-memory fixture values",
        aggregation_rule="one value per declared element",
        definition_version="tests.nr.fixture.v2",
        value_axes=("element",),
    )
    record = replace(record, target_definitions={"metric": definition})
    stored = write_nr_record(tmp_path, record)
    restored = read_nr_record(tmp_path, stored)
    assert restored.target_definitions["metric"].value_axes == ("element",)


def test_nr_geometric_visibility_v2_scalar_roundtrips_in_schema_v5(tmp_path):
    record = make_single_record("geometric-visibility-v2", targets={"geometric_los": True})
    definition = TargetDefinition.versioned(
        semantic="digest-bound NumPy loaded-scene direct-visibility oracle",
        units="boolean",
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "loaded-scene triangle query cross-checked with Mitsuba segment intersection "
            "and depth-zero Sionna PathSolver for each recorded TX/RX pose and geometry digest"
        ),
        aggregation_rule="one independent direct-path visibility value per frame",
        definition_version="umcd.rt-trajectory.geometric-los.v2",
    )
    record = replace(record, target_definitions={"geometric_los": definition})

    stored = write_nr_record(tmp_path, record)
    restored = read_nr_record(tmp_path, stored)

    assert restored.targets["geometric_los"].shape == ()
    assert bool(restored.targets["geometric_los"])
    assert restored.target_definitions["geometric_los"] == definition


def test_nr_standardized_profile_proxy_roundtrips_fixed_singleton_schema_v5(tmp_path):
    record = make_single_record(
        "standardized-profile-proxy-v1",
        targets={"profile_los_proxy": np.asarray([1], dtype=np.uint8)},
    )
    definition = TargetDefinition.versioned(
        semantic="standardized_profile_los_category_proxy",
        units="binary_class",
        source="Sionna standardized TDL/CDL profile model",
        reference="profile_los_component false maps to 0 and true maps to 1; not geometric LOS truth",
        aggregation_rule="one profile-category proxy per generated channel record",
        definition_version="umcd.standardized.profile-los-proxy.v1",
    )
    record = replace(record, target_definitions={"profile_los_proxy": definition})

    stored = write_nr_record(tmp_path, record)
    restored = read_nr_record(tmp_path, stored)

    assert restored.targets["profile_los_proxy"].shape == (1,)
    np.testing.assert_array_equal(restored.targets["profile_los_proxy"], [1])
    assert restored.target_definitions["profile_los_proxy"] == definition


@pytest.mark.parametrize(
    "version",
    [
        "umcd.standardized.effective-cfr.v1",
        "umcd.tdl-pusch.effective-cfr.v1",
        "umcd.rt-trajectory.absolute-cfr.v1",
        "umcd.rt-trajectory.symbol-cfr.v1",
    ],
)
def test_nr_writer_keeps_axis_ambiguous_canonical_cfr_versions_read_only(tmp_path, version):
    record = make_single_record(
        "legacy-axis-cfr",
        targets={"channel_cfr": np.ones((24,), dtype=np.complex64)},
    )
    definition = TargetDefinition.versioned(
        semantic="legacy axis-ambiguous fixture",
        units="complex channel gain",
        source="legacy fixture",
        reference="historical unbound axis declaration",
        aggregation_rule="historical coefficient sequence",
        definition_version=version,
    )
    record = replace(record, target_definitions={"channel_cfr": definition})

    with pytest.raises(ValueError, match="axis-ambiguous NR target definitions are read-only"):
        write_nr_record(tmp_path, record)
    assert not (tmp_path / "records").exists()


def test_nr_trajectory_writer_preflights_axis_ambiguous_targets_before_any_publish(tmp_path):
    valid = make_single_record("route-valid-first", targets={"diagnostic": np.asarray([1.0])})
    legacy = make_single_record(
        "route-legacy-second",
        targets={"channel_cfr": np.ones((24,), dtype=np.complex64)},
    )
    legacy_definition = TargetDefinition.versioned(
        semantic="legacy axis-ambiguous fixture",
        units="complex channel gain",
        source="legacy fixture",
        reference="historical unbound axis declaration",
        aggregation_rule="historical coefficient sequence",
        definition_version="umcd.rt-trajectory.absolute-cfr.v1",
    )
    legacy = replace(legacy, target_definitions={"channel_cfr": legacy_definition})
    trajectory = NRTrajectory("axis-policy-route", (0.0, 1e-3), (valid, legacy))

    with pytest.raises(ValueError, match="axis-ambiguous NR target definitions are read-only"):
        write_nr_trajectory(tmp_path, trajectory)
    assert not (tmp_path / "records").exists()


@pytest.mark.parametrize(
    ("version", "semantic", "source", "reference", "aggregation"),
    [
        (
            "umcd.standardized.effective-cfr.v2",
            "perfect_synchronized_effective_subcarrier_cfr",
            "standardized Sionna TDL/CDL CFR adapter",
            "profile-relative excess-delay CFR resampled to this resource grid",
            "retain each effective subcarrier coefficient in receiver FFT-bin order; "
            "no antenna, symbol, or power averaging",
        ),
        (
            "umcd.tdl-pusch.effective-cfr.v2",
            "perfect synchronized effective-subcarrier CFR",
            "sionna_phy_ofdmchannel_return_channel",
            "OFDMChannel return_channel with the configured TDL profile, carrier, "
            "sample rate, and effective FFT-bin mapping",
            "retain each effective subcarrier coefficient; no antenna, symbol, or power averaging",
        ),
    ],
)
def test_nr_v5_round_trip_preserves_canonical_cfr_v2_axes(
    tmp_path, version, semantic, source, reference, aggregation
):
    axes = (
        "batch",
        "rx_antenna",
        "rx_stream",
        "ofdm_symbol",
        "tx_stream",
        "tx_antenna",
        "effective_subcarrier",
    )
    shape = (1, 1, 2, 1, 1, 14, 24)
    record = make_single_record(
        "canonical-cfr-axis-bound",
        targets={"channel_cfr": np.ones(shape, dtype=np.complex64)},
        target_masks={"channel_cfr": np.ones(shape, dtype=bool)},
    )
    definition = TargetDefinition.versioned(
        semantic=semantic,
        units="complex channel gain (dimensionless)",
        source=source,
        reference=reference,
        aggregation_rule=aggregation,
        definition_version=version,
        value_axes=axes,
    )
    record = replace(record, target_definitions={"channel_cfr": definition})
    stored = write_nr_record(tmp_path / version, record)
    restored = read_nr_record(tmp_path / version, stored)
    assert restored.target_definitions["channel_cfr"] == definition
    np.testing.assert_array_equal(restored.targets["channel_cfr"], record.targets["channel_cfr"])


@pytest.mark.parametrize(
    (
        "version",
        "target_name",
        "semantic",
        "axes",
        "shape",
        "source",
        "reference",
        "aggregation",
    ),
    [
        (
            "umcd.rt-trajectory.absolute-cfr.v2",
            "channel_cfr",
            "absolute-delay full-path complex CFR on effective PUSCH subcarriers",
            ("effective_subcarrier",),
            (24,),
            "sionna_rt_pathsolver",
            "Sionna RT full solver-valid path set evaluated with absolute propagation delays and configured carrier/subcarrier offsets",
            "coherent sum of all retained solver-valid paths per effective subcarrier and frame; no path-count or power normalization",
        ),
        (
            "umcd.rt-trajectory.symbol-cfr.v2",
            "channel_cfr_ofdm_symbols",
            "absolute-delay full-path CFR with path-Doppler phase evolution sampled at PUSCH OFDM symbols",
            ("ofdm_symbol", "effective_subcarrier"),
            (14, 24),
            "sionna_rt_paths_cfr_doppler",
            "absolute-delay RT path channel evaluated at the declared NR OFDM symbol start timestamps",
            "coherent sum of retained solver-valid paths per effective subcarrier and OFDM symbol; no path or symbol averaging",
        ),
    ],
)
def test_nr_v5_round_trip_preserves_rt_cfr_v2_axis_shapes(
    tmp_path, version, target_name, semantic, axes, shape, source, reference, aggregation
):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(np.complex64)
    record = make_single_record("rt-cfr-v2", targets={target_name: values})
    definition = TargetDefinition.versioned(
        semantic=semantic,
        units="complex channel gain (dimensionless)",
        source=source,
        reference=reference,
        aggregation_rule=aggregation,
        definition_version=version,
        value_axes=axes,
    )
    record = replace(record, target_definitions={target_name: definition})

    stored = write_nr_record(tmp_path / version, record)
    restored = read_nr_record(tmp_path / version, stored)

    assert restored.targets[target_name].shape == shape
    np.testing.assert_array_equal(restored.targets[target_name], values)
    assert restored.target_definitions[target_name] == definition


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
        schema_version=5,
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


def test_nr_record_integrity_rejects_removed_reference_payload():
    record = make_single_record(
        "removed-reference-payload",
        targets={"label": np.asarray(1)},
        reference_transport_block_bits=np.asarray([[1, 0, 1]], dtype=np.uint8),
    )
    object.__setattr__(record, "reference_transport_block_bits", None)

    with pytest.raises(ValueError, match="requires its payload array"):
        record.validate_reference_transport_block_integrity()


def test_nr_trajectory_rejects_multiframe_observation():
    from dataclasses import replace

    record = make_single_record("multiframe-observation", targets={"label": np.asarray(1)})
    record = replace(record, observation=make_observation(sample_id="multiframe-observation"))

    with pytest.raises(ValueError, match="exactly one observation"):
        NRTrajectory("multiframe-route", (0.0,), (record,))


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
        target_reason_codes={"label": (TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE)},
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


def test_nr_receiver_inference_record_keeps_identity_outside_single_sample_payload():
    batched_inputs = make_observation().model_inputs()
    with pytest.raises(ValueError, match="exactly one received-grid sample"):
        NRReceiverInferenceRecord("episode", "link", 0, "sample", batched_inputs)

    inputs = make_single_record(
        "single-input", targets={"label": np.asarray(1)}
    ).observation.model_inputs()
    record = NRReceiverInferenceRecord("episode", "link", 0, "sample", inputs)
    assert record.model_inputs is inputs
    with pytest.raises(ValueError, match="episode_id"):
        NRReceiverInferenceRecord(" episode", "link", 0, "sample", inputs)
    with pytest.raises(ValueError, match="frame_index"):
        NRReceiverInferenceRecord("episode", "link", -1, "sample", inputs)
    with pytest.raises(TypeError, match="model_inputs must be NRModelInputs"):
        NRReceiverInferenceRecord("episode", "link", 0, "sample", object())


def test_torch_received_grid_mask_must_share_the_grid_device():
    torch = pytest.importorskip("torch")
    base = make_observation()
    estimate = NRChannelEstimate(
        h_hat=torch.zeros((2, 1, 2, 1, 1, 14, 24), dtype=torch.complex64),
        error_variance=torch.zeros((1, 1, 1, 1, 1, 14, 24), dtype=torch.float32),
        pilot_mask=torch.zeros((1, 1, 14, 24), dtype=torch.bool),
        metadata={"receiver": "device-mismatch-test"},
    )

    with pytest.raises(ValueError, match="same device as received_grid"):
        NRObservation(
            sample_id="device-mismatch",
            received_grid=torch.zeros((2, 1, 2, 14, 24), dtype=torch.complex64),
            channel_estimate=estimate,
            fft_bin_indices=tuple(range(24)),
            receiver_config=base.receiver_config,
            received_grid_mask=torch.empty((2, 1, 2, 14, 24), dtype=torch.bool, device="meta"),
        )


def test_reference_transport_block_tensor_and_integrity_guards():
    torch = pytest.importorskip("torch")
    bits = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    record = make_single_record(
        "reference-bits",
        targets={"target": torch.asarray([1.0])},
        target_masks={},
        reference_transport_block_bits=bits,
    )
    tensor_provenance = dict(record.provenance)
    tensor_provenance["reference_transport_block_shape"] = list(bits.shape)
    tensor_record = NRChannelRecord(
        observation=record.observation,
        targets=record.targets,
        target_masks=record.target_masks,
        provenance=tensor_provenance,
        target_definitions=record.target_definitions,
        target_reason_codes=record.target_reason_codes,
        reference_transport_block_bits=torch.asarray(bits),
    )
    np.testing.assert_array_equal(tensor_record.reference_transport_block_bits, bits)
    tensor_record.validate_reference_transport_block_integrity()

    corrupted = tensor_record.reference_transport_block_bits
    corrupted.setflags(write=True)
    corrupted[0, 0] = 0
    with pytest.raises(ValueError, match="differs from its provenance shape/digest"):
        tensor_record.validate_reference_transport_block_integrity()

    with pytest.raises(ValueError, match="provenance requires its payload array"):
        make_single_record(
            "missing-reference-bits",
            targets={"target": np.asarray([1.0])},
            target_masks={},
            provenance_overrides={"reference_transport_block_shape": [2, 3]},
        )


def test_torch_target_support_and_unexpected_target_audit_paths():
    torch = pytest.importorskip("torch")
    record = make_single_record(
        "torch-support",
        targets={"scalar": torch.asarray([1.0, 0.0])},
        target_masks={"scalar": torch.asarray([True, False])},
        target_reason_codes={"scalar": (TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE)},
    )
    trajectory = NRTrajectory("torch-support-route", (0.0,), (record,))
    assert trajectory.target_support_counts("scalar")["selected_element_count"] == 1

    with pytest.raises(ValueError, match="contains undeclared targets"):
        audit_nr_trajectory_target_support(
            (("train", "site", trajectory),), (("train", "site", "other-target"),)
        )


def test_torch_target_must_be_nonempty_and_reason_codes_must_match_target_elements():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="must be non-empty"):
        make_single_record("empty-torch-target", targets={"scalar": torch.asarray([])})
    with pytest.raises(ValueError, match="one TargetValidity per target element"):
        make_single_record(
            "bad-reason-count",
            targets={"scalar": np.asarray([1.0, 0.0])},
            target_masks={"scalar": np.asarray([True, False])},
            target_reason_codes={"scalar": (TargetValidity.VALID,)},
        )
