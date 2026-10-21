import numpy as np
import pytest
from models.multipath import fit_sparse_multipath


def _cfr(frequencies_hz, delays_s, gains):
    steering = np.exp(-2j * np.pi * frequencies_hz[:, None] * delays_s[None, :])
    return steering @ gains


def test_sparse_multipath_selects_one_path_and_estimates_noise_from_pilots():
    frequencies = np.linspace(-10e6, 10e6, 128)
    rng = np.random.default_rng(718)
    noise_variance = 2e-3
    truth = _cfr(frequencies, np.asarray([110e-9]), np.asarray([0.9 - 0.25j]))
    noise = np.sqrt(noise_variance / 2) * (
        rng.standard_normal(frequencies.size) + 1j * rng.standard_normal(frequencies.size)
    )
    observed = truth + noise
    mask = np.zeros(frequencies.size, dtype=np.bool_)
    mask[::2] = True

    fit = fit_sparse_multipath(
        observed,
        frequencies,
        mask,
        maximum_paths=4,
        maximum_delay_s=500e-9,
        delay_grid_size=513,
    )

    assert fit.model_order == 1
    assert fit.tested_orders == (0, 1, 2, 3, 4)
    assert fit.delays_s[0] == pytest.approx(110e-9, abs=0.5e-9)
    assert fit.estimated_noise_variance == pytest.approx(noise_variance, rel=0.25)
    assert fit.predicted_cfr.shape == observed.shape


def test_sparse_multipath_recovers_two_paths_without_receiving_noise_oracle():
    frequencies = np.linspace(-15e6, 15e6, 160)
    delays = np.asarray([70e-9, 245e-9])
    gains = np.asarray([1.0 + 0.2j, 0.55 - 0.7j])
    truth = _cfr(frequencies, delays, gains)
    rng = np.random.default_rng(92)
    noise_variance = 1e-4
    noise = np.sqrt(noise_variance / 2) * (
        rng.standard_normal(frequencies.size) + 1j * rng.standard_normal(frequencies.size)
    )
    observed = truth + noise
    mask = np.ones(frequencies.size, dtype=np.bool_)
    mask[::5] = False

    fit = fit_sparse_multipath(
        observed,
        frequencies,
        mask,
        maximum_paths=4,
        maximum_delay_s=500e-9,
        delay_grid_size=801,
    )

    assert fit.model_order == 2
    np.testing.assert_allclose(fit.delays_s, delays, atol=0.5e-9, rtol=0)
    assert fit.estimated_noise_variance == pytest.approx(noise_variance, rel=0.25)
    assert np.mean(np.abs(fit.predicted_cfr - truth) ** 2) < 1e-5


def test_sparse_multipath_never_uses_unobserved_tones_for_selection_or_fit():
    frequencies = np.linspace(-5e6, 5e6, 32)
    truth = _cfr(frequencies, np.asarray([75e-9]), np.asarray([1.0 + 0.3j]))
    mask = np.zeros(frequencies.size, dtype=np.bool_)
    mask[::2] = True
    observed = truth.copy()
    changed_unobserved = truth.copy()
    changed_unobserved[~mask] = 500.0 - 300j
    kwargs = {
        "maximum_paths": 3,
        "maximum_delay_s": 300e-9,
        "delay_grid_size": 257,
    }

    first = fit_sparse_multipath(observed, frequencies, mask, **kwargs)
    second = fit_sparse_multipath(changed_unobserved, frequencies, mask, **kwargs)

    assert first.model_order == second.model_order == 1
    np.testing.assert_allclose(first.delays_s, second.delays_s, atol=0, rtol=0)
    np.testing.assert_allclose(first.gains, second.gains, atol=0, rtol=0)
    np.testing.assert_allclose(first.predicted_cfr, second.predicted_cfr, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("observed", "frequencies", "mask", "options", "message"),
    [
        (
            np.ones(8, dtype=np.complex128),
            np.arange(8, dtype=np.float64),
            np.ones(8, dtype=np.bool_),
            {"maximum_paths": True, "maximum_delay_s": 1.0},
            "maximum_paths must be an integer",
        ),
        (
            np.ones(8, dtype=np.complex128),
            np.arange(8, dtype=np.float64),
            np.ones(8, dtype=np.bool_),
            {"maximum_paths": 2, "maximum_delay_s": 0.0},
            "maximum_delay_s must be finite and positive",
        ),
        (
            np.ones(8, dtype=np.complex128),
            np.arange(8, dtype=np.float64),
            np.ones(8, dtype=np.float64),
            {"maximum_paths": 2, "maximum_delay_s": 1.0},
            "pilot_mask must have boolean dtype",
        ),
    ],
)
def test_sparse_multipath_rejects_invalid_inputs(observed, frequencies, mask, options, message):
    with pytest.raises((TypeError, ValueError), match=message):
        fit_sparse_multipath(observed, frequencies, mask, **options)
