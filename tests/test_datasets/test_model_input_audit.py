from dataclasses import dataclass

import numpy as np
import pytest
from datasets.analytical import AnalyticalModelInputs, generate_analytical_multipath
from datasets.channel import ChannelModelInputs
from datasets.model_input_audit import audit_model_input_payload
from datasets.nr import NRModelInputs
from datasets.observations import FrequencyDomainModelInputs
from receivers.nr import NRPUSCHConfig


def test_audits_known_model_input_contracts_and_time_aligned_rosters() -> None:
    analytical = AnalyticalModelInputs(
        frequencies_hz=np.arange(8, dtype=np.float64),
        channel_frequency_response=np.ones(8, dtype=np.complex64),
    )
    legacy = ChannelModelInputs(np.ones((2, 1, 8), dtype=np.complex64))
    standardized = FrequencyDomainModelInputs(
        observed_cfr=np.ones((1, 1, 8), dtype=np.complex64),
        noise_variance=0.01,
        snr_db=20.0,
        carrier_hz=3.5e9,
        subcarrier_offsets_hz=np.arange(-4, 4, dtype=np.float64),
    )
    nr = NRModelInputs(
        received_grid=np.ones((1, 1, 1, 1, 8), dtype=np.complex64),
        h_hat=np.ones((1, 1, 1, 1, 1, 1, 8), dtype=np.complex64),
        error_variance=np.ones((1, 1, 1, 1, 1, 1, 8), dtype=np.float32),
        pilot_mask=np.ones((1, 1, 1, 8), dtype=bool),
        fft_bin_indices=tuple(range(8)),
        receiver_config=NRPUSCHConfig(),
    )

    assert audit_model_input_payload(legacy) is legacy
    assert audit_model_input_payload(analytical) is analytical
    assert audit_model_input_payload(standardized) is standardized
    assert audit_model_input_payload(nr) is nr
    aligned = audit_model_input_payload((nr, nr))
    assert aligned[0] is nr and aligned[1] is nr


def test_analytical_sample_audits_observation_separately_from_path_oracle() -> None:
    sample = generate_analytical_multipath(seed=4, num_paths=2)

    assert audit_model_input_payload(sample.model_inputs) is sample.model_inputs
    with pytest.raises(TypeError, match="unrecognized model-input contract"):
        audit_model_input_payload(sample.oracle)


def test_rejects_unregistered_or_target_bearing_payloads() -> None:
    @dataclass(frozen=True)
    class LeakyInputs:
        received_grid: np.ndarray
        los_target: np.ndarray

    with pytest.raises(TypeError, match="unrecognized model-input contract"):
        audit_model_input_payload(LeakyInputs(np.ones(1), np.ones(1)))
    with pytest.raises(TypeError, match="unrecognized model-input contract"):
        audit_model_input_payload({"received_grid": np.ones(1), "target": np.ones(1)})


def test_rejects_dataclass_that_spoofs_a_registered_module_and_class_name() -> None:
    spoof_class = type(
        "AnalyticalModelInputs",
        (),
        {
            "__module__": "datasets.analytical",
            "__annotations__": {
                "frequencies_hz": np.ndarray,
                "channel_frequency_response": np.ndarray,
            },
        },
    )
    spoof_class = dataclass(frozen=True)(spoof_class)
    spoof = spoof_class(
        frequencies_hz=np.arange(8, dtype=np.float64),
        channel_frequency_response=np.ones(8, dtype=np.complex64),
    )

    with pytest.raises(TypeError, match="unrecognized model-input contract"):
        audit_model_input_payload(spoof)


def test_rejects_empty_or_unregistered_time_aligned_payloads() -> None:
    with pytest.raises(ValueError, match="must be nonempty"):
        audit_model_input_payload(())
    with pytest.raises(TypeError, match="unrecognized model-input contract"):
        audit_model_input_payload((object(),))


def test_rejects_objects_and_object_arrays_inside_registered_fields() -> None:
    hidden_oracle = AnalyticalModelInputs(
        frequencies_hz=np.arange(8, dtype=np.float64),
        channel_frequency_response=np.asarray([object()] * 8, dtype=object),
    )
    with pytest.raises(TypeError, match="numeric array"):
        audit_model_input_payload(hidden_oracle)

    hidden_metadata = AnalyticalModelInputs(
        frequencies_hz=np.arange(8, dtype=np.float64),
        channel_frequency_response={"cfr": np.ones(8, dtype=np.complex64)},
    )
    with pytest.raises(TypeError, match="NumPy array or dense PyTorch tensor"):
        audit_model_input_payload(hidden_metadata)


def test_rejects_boolean_scalar_and_nonfinite_registered_inputs() -> None:
    boolean_noise = FrequencyDomainModelInputs(
        observed_cfr=np.ones((1, 1, 8), dtype=np.complex64),
        noise_variance=True,
        snr_db=20.0,
        carrier_hz=3.5e9,
        subcarrier_offsets_hz=np.arange(-4, 4, dtype=np.float64),
    )
    with pytest.raises(TypeError, match="noise_variance must be a real numeric scalar"):
        audit_model_input_payload(boolean_noise)

    nonfinite_frequency = AnalyticalModelInputs(
        frequencies_hz=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.inf]),
        channel_frequency_response=np.ones(8, dtype=np.complex64),
    )
    with pytest.raises(ValueError, match="frequencies_hz must contain only finite"):
        audit_model_input_payload(nonfinite_frequency)
