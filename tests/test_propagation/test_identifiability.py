import numpy as np
import pytest
from propagation.identifiability import path_steering_diagnostics


def test_single_path_is_well_conditioned_and_has_no_interpath_coherence():
    frequencies = np.linspace(-10e6, 10e6, 128)

    result = path_steering_diagnostics(np.asarray([300e-9]), frequencies)

    assert result.rank == 1
    assert result.condition_number == pytest.approx(1.0)
    assert result.mutual_coherence == 0.0
    np.testing.assert_allclose(result.normalized_gram, [[1.0]])


def test_coincident_path_delays_are_rank_deficient_and_unidentifiable():
    frequencies = np.linspace(-15e6, 15e6, 256)

    result = path_steering_diagnostics(np.asarray([100e-9, 100e-9]), frequencies)

    assert result.rank == 1
    assert np.isinf(result.condition_number)
    assert result.mutual_coherence == pytest.approx(1.0)


def test_delay_separation_tracks_finite_bandwidth_steering_coherence():
    num_tones = 256
    spacing_hz = 30_000.0
    frequencies = (np.arange(num_tones) - num_tones // 2) * spacing_hz
    first_null_delay_s = 1.0 / (num_tones * spacing_hz)
    close = path_steering_diagnostics(np.asarray([0.0, first_null_delay_s / 100]), frequencies)
    first_null = path_steering_diagnostics(np.asarray([0.0, first_null_delay_s]), frequencies)

    assert close.mutual_coherence > 0.99
    assert close.condition_number > 10
    assert first_null.mutual_coherence < 1e-12
    assert first_null.condition_number == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("delays", "frequencies", "message"),
    [
        (np.asarray([]), np.arange(8), "delays_s"),
        (np.asarray([0.0]), np.asarray([1.0]), "frequencies_hz"),
        (np.asarray([-1e-9]), np.arange(8), "nonnegative"),
        (np.asarray([0.0]), np.asarray([0.0, np.nan]), "frequencies_hz"),
    ],
)
def test_path_steering_diagnostics_reject_invalid_inputs(delays, frequencies, message):
    with pytest.raises(ValueError, match=message):
        path_steering_diagnostics(delays, frequencies)
