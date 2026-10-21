import numpy as np
import pytest
from datasets.analytical import generate_analytical_multipath
from propagation.physics import frequency_response


def test_seeded_one_to_three_path_cfr_matches_closed_form_and_keeps_oracle_private():
    sample = generate_analytical_multipath(
        seed=51,
        num_paths=3,
        num_subcarriers=96,
        subcarrier_spacing_hz=60_000,
        max_excess_delay_s=800e-9,
        receiver_timing_offset_s=120e-9,
    )
    repeated = generate_analytical_multipath(
        seed=51,
        num_paths=3,
        num_subcarriers=96,
        subcarrier_spacing_hz=60_000,
        max_excess_delay_s=800e-9,
        receiver_timing_offset_s=120e-9,
    )

    expected = frequency_response(
        sample.oracle.complex_gains,
        sample.oracle.excess_delays_s + sample.oracle.receiver_timing_offset_s,
        sample.model_inputs.frequencies_hz,
    )
    np.testing.assert_allclose(sample.model_inputs.channel_frequency_response, expected)
    np.testing.assert_array_equal(
        sample.model_inputs.channel_frequency_response,
        repeated.model_inputs.channel_frequency_response,
    )
    assert sample.sample_id == repeated.sample_id
    assert sample.oracle.direct_component_mask.tolist() == [True, False, False]
    assert not hasattr(sample.model_inputs, "sample_id")
    assert not hasattr(sample.model_inputs, "oracle")
    assert sample.metadata["generator"] == "analytical_multipath_debug"


def test_direct_component_flag_does_not_get_mislabeled_as_geometric_los():
    sample = generate_analytical_multipath(seed=2, num_paths=2, direct_component_present=False)

    assert sample.oracle.direct_component_present is False
    assert not sample.oracle.direct_component_mask.any()
    assert "geometric_los" not in sample.metadata


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"num_paths": 4}, "num_paths"),
        ({"num_subcarriers": 7}, "num_subcarriers"),
        ({"subcarrier_spacing_hz": 0}, "subcarrier_spacing_hz"),
        ({"max_excess_delay_s": -1}, "max_excess_delay_s"),
        ({"receiver_timing_offset_s": -1}, "receiver_timing_offset_s"),
    ],
)
def test_analytical_generator_rejects_invalid_physical_parameters(overrides, message):
    config = {"seed": 1, "num_paths": 2}
    config.update(overrides)
    with pytest.raises(ValueError, match=message):
        generate_analytical_multipath(**config)
