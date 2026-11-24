"""Finite delay-domain DM-RS estimator: exact recovery of a finite-delay channel and input rules."""

from __future__ import annotations

import math
import types

import pytest

torch = pytest.importorskip("torch")

from experiments.cp_bounded_dft_estimator import (  # noqa: E402
    FiniteDelayDFTChannelEstimator,
    make_finite_delay_dft_decoder,
)

FFT = 16
SYMBOLS = 6
DMRS_SYMBOLS = (1, 4)
PILOT_STEP = 4


def _mask() -> torch.Tensor:
    mask = torch.zeros(1, 1, SYMBOLS, FFT, dtype=torch.bool)
    for symbol in DMRS_SYMBOLS:
        mask[0, 0, symbol, ::PILOT_STEP] = True
    return mask


def _pilot_values() -> torch.Tensor:
    count = len(DMRS_SYMBOLS) * (FFT // PILOT_STEP)
    phases = torch.linspace(0.0, 2.0, count, dtype=torch.float64)
    return torch.polar(torch.ones(count, dtype=torch.float64), phases).to(torch.complex64)[None, None, :]


class _FakeEstimator:
    """Reads pilot resource elements from a grid in the order the receiver uses."""

    def __init__(self, mask: torch.Tensor) -> None:
        self._mask = mask

    def _extract_pilots(self, grid: torch.Tensor) -> torch.Tensor:
        return grid[..., self._mask[0, 0]]


def _receiver(mask=None, pilots=None, fft=FFT, symbols=SYMBOLS):
    mask = _mask() if mask is None else mask
    pilots = _pilot_values() if pilots is None else pilots
    pattern = types.SimpleNamespace(pilots=pilots)
    grid = types.SimpleNamespace(fft_size=fft, num_ofdm_symbols=symbols, pilot_pattern=pattern)
    return types.SimpleNamespace(
        _torch=torch,
        pilot_mask=mask,
        resource_grid=grid,
        _estimator=_FakeEstimator(mask),
        make_decoder_with_estimator=lambda estimator: ("decoder", estimator),
    )


def _true_response(taps: torch.Tensor, delays: list[int]) -> torch.Tensor:
    """Frequency response on the centered bin order used by the estimator's output."""
    bins = torch.arange(FFT, dtype=torch.float64) - FFT // 2
    basis = torch.stack(
        [torch.exp(-2j * math.pi * bins * delay / FFT) for delay in delays], dim=1
    )
    return (basis @ taps.to(torch.complex128)).to(torch.complex64)


def _noise_free_grid(taps: torch.Tensor, delays: list[int]) -> torch.Tensor:
    """Received grid with every resource element carrying the same time-invariant channel."""
    response = _true_response(taps, delays)
    grid = response[None, None, None, None, :].expand(1, 1, 1, SYMBOLS, FFT).clone()
    # The pilots carry known values; the received pilot equals channel times pilot.
    pilot_values = _pilot_values()[0, 0]
    received = grid.clone()
    received[..., _mask()[0, 0]] = response[None, None, None, None, :][..., _mask()[0, 0].nonzero(as_tuple=True)[1]] * pilot_values
    return received


def test_noise_free_finite_delay_channel_is_recovered_exactly_at_every_symbol() -> None:
    delays = [0, 1, 2]
    taps = torch.tensor([1.0 + 0.5j, -0.3 + 0.2j, 0.1 - 0.4j], dtype=torch.complex128)
    estimator = FiniteDelayDFTChannelEstimator(_receiver(), max_delay_samples=2, min_delay_samples=0)
    received = _noise_free_grid(taps, delays)
    estimate, variance = estimator(received, torch.tensor(0.0))
    truth = _true_response(taps, delays)
    assert estimate.shape[-2] == SYMBOLS and estimate.shape[-1] == FFT
    assert torch.allclose(estimate.reshape(-1, SYMBOLS, FFT)[0, 2], truth, atol=1e-4)
    assert torch.allclose(variance, torch.zeros_like(variance), atol=1e-8)


def test_noise_variance_propagates_into_positive_error_variance() -> None:
    taps = torch.tensor([1.0, 0.2, 0.0], dtype=torch.complex128)
    estimator = FiniteDelayDFTChannelEstimator(_receiver(), max_delay_samples=2)
    received = _noise_free_grid(taps, [0, 1, 2])
    _, quiet = estimator(received, torch.tensor(1e-4))
    _, loud = estimator(received, torch.tensor(1e-2))
    assert torch.all(quiet >= 0)
    assert torch.all(loud > quiet)


def test_time_interpolation_between_dmrs_symbols_is_linear() -> None:
    taps = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)
    estimator = FiniteDelayDFTChannelEstimator(_receiver(), max_delay_samples=2)
    received = _noise_free_grid(taps, [0, 1, 2])
    received[..., 1, :] = received[..., 1, :] * 1.0
    estimate, _ = estimator(received, torch.tensor(0.0))
    # Symbols 2 and 3 lie between the DM-RS symbols 1 and 4 and must equal their convex combination.
    left, right = estimate[..., 1, :], estimate[..., 4, :]
    expected_2 = (2.0 / 3.0) * left + (1.0 / 3.0) * right
    expected_3 = (1.0 / 3.0) * left + (2.0 / 3.0) * right
    assert torch.allclose(estimate[..., 2, :], expected_2, atol=1e-5)
    assert torch.allclose(estimate[..., 3, :], expected_3, atol=1e-5)
    assert torch.allclose(estimate[..., 0, :], left, atol=1e-5)
    assert torch.allclose(estimate[..., 5, :], right, atol=1e-5)


def test_constructor_rejects_non_integer_and_inconsistent_delay_intervals() -> None:
    receiver = _receiver()
    with pytest.raises(TypeError, match="must be an integer"):
        FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=2.0)
    with pytest.raises(TypeError, match="must be an integer"):
        FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=2, min_delay_samples=True)
    with pytest.raises(ValueError, match="must not exceed"):
        FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=0, min_delay_samples=1)
    with pytest.raises(ValueError, match="inside one FFT period"):
        FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=FFT)
    with pytest.raises(ValueError, match="inside one FFT period"):
        FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=2, min_delay_samples=-FFT)
    with pytest.raises(ValueError, match="fewer taps"):
        FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=FFT - 1, min_delay_samples=0)


def test_constructor_rejects_unsupported_grids_and_pilot_patterns() -> None:
    stacked = torch.zeros(1, 2, SYMBOLS, FFT, dtype=torch.bool)
    stacked[0, 0] = _mask()[0, 0]
    with pytest.raises(ValueError, match="one TX stream"):
        FiniteDelayDFTChannelEstimator(_receiver(mask=stacked), max_delay_samples=2)
    no_pilots = torch.zeros(1, 1, SYMBOLS, FFT, dtype=torch.bool)
    with pytest.raises(ValueError, match="no DM-RS pilot symbols"):
        FiniteDelayDFTChannelEstimator(_receiver(mask=no_pilots), max_delay_samples=2)
    unequal = _mask().clone()
    unequal[0, 0, 4, 1] = True
    with pytest.raises(ValueError, match="equal pilot counts"):
        FiniteDelayDFTChannelEstimator(_receiver(mask=unequal), max_delay_samples=2)
    with pytest.raises(ValueError, match="one pilot sequence per DM-RS RE"):
        FiniteDelayDFTChannelEstimator(_receiver(pilots=torch.ones(1, 1, 3)), max_delay_samples=2)
    # Two pilots per DM-RS symbol cannot identify a three-tap span (rank 2 < 3).
    clustered = torch.zeros(1, 1, SYMBOLS, FFT, dtype=torch.bool)
    clustered[0, 0, 1, :2] = True
    clustered[0, 0, 4, :2] = True
    clustered_pilots = torch.ones(1, 1, 4, dtype=torch.complex64)
    with pytest.raises(ValueError, match="cannot identify the configured delay span"):
        FiniteDelayDFTChannelEstimator(_receiver(mask=clustered, pilots=clustered_pilots), max_delay_samples=2)


def test_estimation_rejects_zero_pilots_and_invalid_noise() -> None:
    zero_pilots = _pilot_values().clone()
    zero_pilots[0, 0, 0] = 0
    estimator = FiniteDelayDFTChannelEstimator(_receiver(), max_delay_samples=2)
    received = _noise_free_grid(torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128), [0, 1, 2])
    bad_receiver = _receiver(pilots=zero_pilots)
    zero_estimator = FiniteDelayDFTChannelEstimator(bad_receiver, max_delay_samples=2)
    with pytest.raises(ValueError, match="nonzero pilots"):
        zero_estimator(received, torch.tensor(0.0))
    with pytest.raises(ValueError, match="finite, nonnegative"):
        estimator(received, torch.tensor(-1.0))
    with pytest.raises(ValueError, match="finite, nonnegative"):
        estimator(received, torch.tensor(float("nan")))
    with pytest.raises(ValueError, match="broadcast"):
        estimator(received, torch.zeros(2, 2, 2, 2))
    with pytest.raises(ValueError, match="broadcast over"):
        estimator(received, torch.zeros(2))


def test_estimation_rejects_a_pilot_count_that_differs_from_the_grid() -> None:
    estimator = FiniteDelayDFTChannelEstimator(_receiver(), max_delay_samples=2)
    wrong = torch.zeros(1, 1, 1, SYMBOLS, FFT, dtype=torch.complex64)
    wrong_mask = torch.zeros(1, 1, SYMBOLS, FFT, dtype=torch.bool)
    wrong_mask[0, 0, 1, :2] = True
    estimator._base_estimator = _FakeEstimator(wrong_mask)
    with pytest.raises(ValueError, match="differs from the configured DM-RS grid"):
        estimator(wrong, torch.tensor(0.0))


def test_decoder_factory_builds_the_estimator_through_the_receiver() -> None:
    receiver = _receiver()
    decoder_name, estimator = make_finite_delay_dft_decoder(receiver, max_delay_samples=3, min_delay_samples=1)
    assert decoder_name == "decoder"
    assert isinstance(estimator, FiniteDelayDFTChannelEstimator)
    assert estimator._tap_count == 3
    # The default eight-tap span needs at least eight independent pilots per symbol; this
    # grid has four, so the default configuration must refuse rather than guess.
    with pytest.raises(ValueError, match="cannot identify the configured delay span"):
        make_finite_delay_dft_decoder(receiver)
