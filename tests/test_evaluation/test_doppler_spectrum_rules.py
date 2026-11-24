"""Input contracts and deterministic summaries for Doppler-spectrum helpers."""

import numpy as np
import pytest
from evaluation.covariance.doppler_spectrum import (
    cluster_spectrum_cdf,
    jakes_doppler_cdf,
    simultaneous_spectrum_cdf_bounds,
)


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ([True, False], TypeError, "real numeric values"),
        ([1.0 + 1.0j], TypeError, "real numeric values"),
        (["0.2"], TypeError, "real numeric values"),
        ([np.nan], ValueError, "must be finite"),
    ],
)
def test_jakes_cdf_rejects_nonreal_or_nonfinite_values(values, error, message) -> None:
    with pytest.raises(error, match=message):
        jakes_doppler_cdf(values)


def test_jakes_cdf_is_symmetric_and_clips_outside_support() -> None:
    actual = jakes_doppler_cdf(np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]))

    np.testing.assert_allclose(actual, [0.0, 0.0, 0.5, 2.0 / 3.0, 1.0, 1.0])
    np.testing.assert_allclose(jakes_doppler_cdf([-0.5, 0.5]).sum(), 1.0)


@pytest.mark.parametrize(
    ("channel", "error", "message"),
    [
        (np.ones((2, 16), dtype=np.complex128), ValueError, "draw, tap, time"),
        (np.ones((2, 1, 15), dtype=np.complex128), ValueError, "time>=16"),
        (np.ones((0, 1, 16), dtype=np.complex128), ValueError, "draw, tap, time"),
        (np.ones((1, 1, 16)), TypeError, "dense complex array"),
        (np.full((1, 1, 16), np.nan + 0j), ValueError, "must be finite"),
        (np.zeros((1, 1, 16), dtype=np.complex128), ValueError, "positive finite spectral power"),
    ],
)
def test_cluster_spectrum_rejects_invalid_channel_arrays(channel, error, message) -> None:
    with pytest.raises(error, match=message):
        cluster_spectrum_cdf(
            channel,
            sampling_frequency_hz=100.0,
            maximum_doppler_hz=10.0,
            normalized_frequency_cutoffs=(-1.0, 0.0, 1.0),
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"sampling_frequency_hz": True}, TypeError, "sampling_frequency_hz must be a real scalar"),
        ({"sampling_frequency_hz": 0.0}, ValueError, "must be finite and positive"),
        ({"maximum_doppler_hz": np.inf}, ValueError, "must be finite and positive"),
        ({"sampling_frequency_hz": 20.0}, ValueError, "must exceed twice maximum_doppler_hz"),
        ({"normalized_frequency_cutoffs": ()}, TypeError, "nonempty real sequence"),
        ({"normalized_frequency_cutoffs": (0.0, 0.0)}, ValueError, "strictly increasing"),
        ({"normalized_frequency_cutoffs": (np.nan,)}, ValueError, "strictly increasing"),
        ({"normalized_frequency_cutoffs": (0.0 + 1.0j,)}, TypeError, "nonempty real sequence"),
        ({"normalized_frequency_cutoffs": (5.1,)}, ValueError, "sampled Nyquist band"),
    ],
)
def test_cluster_spectrum_rejects_invalid_sampling_and_cutoffs(kwargs, error, message) -> None:
    arguments = {
        "sampling_frequency_hz": 100.0,
        "maximum_doppler_hz": 10.0,
        "normalized_frequency_cutoffs": (-1.0, 0.0, 1.0),
    }
    arguments.update(kwargs)
    channel = np.ones((2, 1, 32), dtype=np.complex128)
    channel[:, :, 1] = 1j
    with pytest.raises(error, match=message):
        cluster_spectrum_cdf(channel, **arguments)


def test_cluster_spectrum_returns_monotone_per_draw_two_sided_cdfs() -> None:
    rng = np.random.default_rng(812)
    channel = rng.normal(size=(19, 3, 64)) + 1j * rng.normal(size=(19, 3, 64))

    result = cluster_spectrum_cdf(
        channel,
        sampling_frequency_hz=100.0,
        maximum_doppler_hz=10.0,
        normalized_frequency_cutoffs=(-3.0, 0.0, 3.0),
    )

    assert result.shape == (19, 3)
    assert np.isfinite(result).all()
    assert ((result >= 0.0) & (result <= 1.0)).all()
    assert (np.diff(result, axis=1) >= 0.0).all()
    # A zero-mean centered channel has essentially all spectral power in band.
    assert (result[:, 0] < result[:, 1]).all()
    assert (result[:, 1] < result[:, 2]).all()


@pytest.mark.parametrize(
    ("values", "expected", "alpha", "error", "message"),
    [
        (np.ones((1, 2)), (1.0, 1.0), 0.05, ValueError, "draw>=2"),
        (np.ones((2, 0)), (), 0.05, ValueError, "draw>=2"),
        (np.ones((2, 2)), (1.0,), 0.05, ValueError, "one value per frequency cutoff"),
        (np.ones((2, 1), dtype=bool), (1.0,), 0.05, TypeError, "real numeric values"),
        (np.array([[np.nan], [0.2]]), (1.0,), 0.05, ValueError, "finite probabilities"),
        (np.array([[1.2], [0.2]]), (1.0,), 0.05, ValueError, "finite probabilities"),
        (np.array([[0.2], [0.4]]), (np.inf,), 0.05, ValueError, "finite probabilities"),
        (
            np.array([[0.2], [0.4]]),
            (0.3,),
            True,
            TypeError,
            "familywise_alpha must be a real scalar",
        ),
        (np.array([[0.2], [0.4]]), (0.3,), 0.0, ValueError, "familywise_alpha must be in"),
    ],
)
def test_simultaneous_bounds_reject_invalid_values(values, expected, alpha, error, message) -> None:
    with pytest.raises(error, match=message):
        simultaneous_spectrum_cdf_bounds(
            values,
            expected_cdf=expected,
            familywise_alpha=alpha,
            maximum_absolute_error=0.1,
        )


@pytest.mark.parametrize("kwargs", [{"family_claim_count": True}, {"family_claim_count": 0}])
def test_simultaneous_bounds_rejects_invalid_family_claim_count(kwargs) -> None:
    with pytest.raises(ValueError, match="family_claim_count must be an integer"):
        simultaneous_spectrum_cdf_bounds(
            np.array([[0.2], [0.4]]),
            expected_cdf=(0.3,),
            familywise_alpha=0.05,
            maximum_absolute_error=0.2,
            **kwargs,
        )


def test_simultaneous_bounds_reports_cluster_count_and_familywise_margin() -> None:
    per_draw = np.array([[0.2, 0.4], [0.4, 0.6], [0.3, 0.5]])

    result = simultaneous_spectrum_cdf_bounds(
        per_draw,
        expected_cdf=(0.3, 0.5),
        familywise_alpha=0.05,
        maximum_absolute_error=1.0,
        family_claim_count=4,
    )

    assert result["draw_count"] == 3
    assert result["claim_count"] == 4
    np.testing.assert_allclose(result["mean_cdf"], (0.3, 0.5))
    assert result["status"] is True


def test_simultaneous_bounds_rejects_infinite_tolerance() -> None:
    with pytest.raises(ValueError, match="maximum_absolute_error must be finite"):
        simultaneous_spectrum_cdf_bounds(
            np.array([[0.2], [0.4]]),
            expected_cdf=(0.3,),
            familywise_alpha=0.05,
            maximum_absolute_error=np.inf,
        )


def test_simultaneous_bounds_requires_positive_error_tolerance() -> None:
    with pytest.raises(ValueError, match="maximum_absolute_error must be positive"):
        simultaneous_spectrum_cdf_bounds(
            np.array([[0.2], [0.4]]),
            expected_cdf=(0.3,),
            familywise_alpha=0.05,
            maximum_absolute_error=0.0,
        )


def test_simultaneous_bounds_defaults_family_claim_count_to_cutoffs() -> None:
    result = simultaneous_spectrum_cdf_bounds(
        np.array([[0.2, 0.4], [0.4, 0.6]]),
        expected_cdf=(0.3, 0.5),
        familywise_alpha=0.05,
        maximum_absolute_error=1.0,
    )

    assert result["claim_count"] == 2
