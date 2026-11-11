from dataclasses import fields, replace

import numpy as np
import pytest
from datasets.model_input_audit import audit_model_input_payload
from experiments.observable_multipath import generate_grouped_two_path_dataset


def test_two_path_receiver_input_view_excludes_truth_and_split_metadata() -> None:
    dataset = generate_grouped_two_path_dataset(
        num_groups=40, samples_per_group=1, n_tones=8, seed=31
    )

    model_inputs = dataset.model_inputs()

    assert [item.name for item in fields(model_inputs)] == [
        "observed_cfr",
        "pilot_mask",
        "frequencies_hz",
        "noise_variance",
    ]
    assert audit_model_input_payload(model_inputs) is model_inputs
    assert model_inputs.observed_cfr.shape == (40, 8)
    assert model_inputs.pilot_mask.shape == model_inputs.observed_cfr.shape
    assert model_inputs.frequencies_hz.shape == model_inputs.observed_cfr.shape
    assert model_inputs.noise_variance.shape == (40,)
    for name in ("relative_delay_s", "gains", "group_id", "sample_id", "split", "stratum"):
        assert not hasattr(model_inputs, name)


def test_two_path_receiver_input_view_rejects_axis_and_value_domain_drift() -> None:
    dataset = generate_grouped_two_path_dataset(
        num_groups=40, samples_per_group=1, n_tones=8, seed=32
    )
    inputs = dataset.model_inputs()

    with pytest.raises(ValueError, match="masks and frequencies must match"):
        audit_model_input_payload(replace(inputs, frequencies_hz=inputs.frequencies_hz[:, :-1]))
    with pytest.raises(TypeError, match="pilot_mask must be Boolean-valued"):
        audit_model_input_payload(replace(inputs, pilot_mask=inputs.pilot_mask.astype(np.float32)))
    with pytest.raises(ValueError, match="noise_variance must be positive"):
        audit_model_input_payload(
            replace(inputs, noise_variance=np.zeros_like(inputs.noise_variance))
        )
