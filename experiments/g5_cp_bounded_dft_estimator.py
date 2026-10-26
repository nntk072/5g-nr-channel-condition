"""Experimental finite delay-domain DM-RS channel estimator for G5."""

from __future__ import annotations

import math
from typing import Any


class FiniteDelayDFTChannelEstimator:
    """Fit a finite impulse response to PUSCH pilot LS estimates and interpolate in time."""

    def __init__(
        self,
        receiver: Any,
        *,
        max_delay_samples: int,
        min_delay_samples: int = 0,
    ) -> None:
        torch = receiver._torch
        for name, value in (
            ("min_delay_samples", min_delay_samples),
            ("max_delay_samples", max_delay_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        mask = receiver.pilot_mask
        if mask.shape[:2] != (1, 1):
            raise ValueError("the experimental DFT estimator currently supports one TX stream")
        self._base_estimator = receiver._estimator
        self._fft_size = int(receiver.resource_grid.fft_size)
        self._num_symbols = int(receiver.resource_grid.num_ofdm_symbols)
        if min_delay_samples > max_delay_samples:
            raise ValueError("min_delay_samples must not exceed max_delay_samples")
        if min_delay_samples <= -self._fft_size or max_delay_samples >= self._fft_size:
            raise ValueError("delay interval must lie inside one FFT period")
        self._min_delay_samples = min_delay_samples
        self._max_delay_samples = max_delay_samples
        self._tap_count = max_delay_samples - min_delay_samples + 1
        if self._tap_count >= self._fft_size:
            raise ValueError("delay interval must contain fewer taps than the FFT size")
        self._dmrs_symbol_indices: list[int] = []
        self._pilot_subcarriers: list[Any] = []
        for symbol in range(self._num_symbols):
            indices = torch.nonzero(mask[0, 0, symbol], as_tuple=False).flatten()
            if indices.numel():
                frequencies = indices.to(dtype=torch.float32) - self._fft_size // 2
                basis = self._frequency_basis(
                    frequencies,
                    self._tap_count,
                    self._fft_size,
                    min_delay_samples=self._min_delay_samples,
                )
                if int(torch.linalg.matrix_rank(basis).item()) != self._tap_count:
                    raise ValueError(
                        "DM-RS pilot frequencies cannot identify the configured delay span"
                    )
                self._dmrs_symbol_indices.append(symbol)
                self._pilot_subcarriers.append(indices)
        if not self._dmrs_symbol_indices:
            raise ValueError("resource grid contains no DM-RS pilot symbols")
        counts = {int(indices.numel()) for indices in self._pilot_subcarriers}
        if len(counts) != 1:
            raise ValueError("DM-RS symbols must have equal pilot counts for this estimator")
        expected_pilot_count = len(self._dmrs_symbol_indices) * counts.pop()
        pilot_values = receiver.resource_grid.pilot_pattern.pilots
        if tuple(pilot_values.shape) != (1, 1, expected_pilot_count):
            raise ValueError("the experimental estimator requires one pilot sequence per DM-RS RE")
        self._pilot_values = pilot_values.reshape(1, 1, 1, 1, 1, len(self._dmrs_symbol_indices), -1)

        all_bins = torch.arange(self._fft_size, dtype=torch.float32) - self._fft_size // 2
        self._synthesis_basis = self._frequency_basis(
            all_bins,
            self._tap_count,
            self._fft_size,
            min_delay_samples=self._min_delay_samples,
        )

    @staticmethod
    def _frequency_basis(
        frequencies: Any,
        tap_count: int,
        fft_size: int,
        *,
        min_delay_samples: int = 0,
    ) -> Any:
        import torch

        delays = torch.arange(
            min_delay_samples,
            min_delay_samples + tap_count,
            dtype=frequencies.dtype,
            device=frequencies.device,
        )
        return torch.exp(-2j * math.pi * frequencies[:, None] * delays[None, :] / fft_size)

    def __call__(self, received_grid: Any, noise_variance: Any) -> tuple[Any, Any]:
        """Estimate CFR and its LS-propagated variance on the configured resource grid."""
        import torch

        pilot_observations = self._base_estimator._extract_pilots(received_grid)
        pilot_count = int(self._pilot_subcarriers[0].numel())
        expected_count = len(self._dmrs_symbol_indices) * pilot_count
        if pilot_observations.shape[-1] != expected_count:
            raise ValueError("Sionna pilot-estimate shape differs from the configured DM-RS grid")
        pilot_observations = pilot_observations.reshape(
            *pilot_observations.shape[:-1], len(self._dmrs_symbol_indices), pilot_count
        )
        pilot_values = self._pilot_values.to(
            device=pilot_observations.device, dtype=pilot_observations.dtype
        )
        if bool((pilot_values.abs() == 0).any()):
            raise ValueError("the experimental estimator requires nonzero pilots at every DM-RS RE")
        h_ls = pilot_observations / pilot_values

        noise = torch.as_tensor(noise_variance, dtype=torch.float32, device=received_grid.device)
        if noise.ndim > 3 or not bool(torch.isfinite(noise).all()) or bool((noise < 0).any()):
            raise ValueError("noise_variance must be finite, nonnegative, and broadcast over RX")
        leading_shape = tuple(int(value) for value in received_grid.shape[:3])
        noise_shape = (*noise.shape, *((1,) * (3 - noise.ndim)))
        noise = noise.reshape(noise_shape)
        if any(
            noise_size not in (1, grid_size)
            for noise_size, grid_size in zip(noise.shape, leading_shape, strict=True)
        ):
            raise ValueError("noise_variance must broadcast over [batch, rx, rx_ant]")
        raw_noise_variance = noise.reshape(*noise.shape, 1, 1, 1, 1) / pilot_values.abs().square()

        channel_by_pilot_symbol: list[Any] = []
        variance_by_pilot_symbol: list[Any] = []
        for pilot_index, subcarriers in enumerate(self._pilot_subcarriers):
            frequencies = subcarriers.to(device=h_ls.device, dtype=torch.float32)
            frequencies = frequencies - self._fft_size // 2
            pilot_basis = self._frequency_basis(
                frequencies,
                self._tap_count,
                self._fft_size,
                min_delay_samples=self._min_delay_samples,
            ).to(device=h_ls.device)
            synthesis_basis = self._synthesis_basis.to(device=h_ls.device)
            gram_inverse = torch.linalg.inv(pilot_basis.mH @ pilot_basis)
            pilot_estimates = h_ls[..., pilot_index, :]
            leading_shape = pilot_estimates.shape[:-1]
            flattened = pilot_estimates.reshape(-1, pilot_estimates.shape[-1]).mT
            taps = torch.linalg.lstsq(pilot_basis, flattened).solution.mT.reshape(
                *leading_shape, self._tap_count
            )
            response = torch.matmul(taps, synthesis_basis.mT)
            channel_by_pilot_symbol.append(response)

            response_variance_gain = torch.einsum(
                "kl,lm,km->k", synthesis_basis, gram_inverse, synthesis_basis.conj()
            ).real.clamp_min(0.0)
            pilot_variance = raw_noise_variance[..., pilot_index, :].mean(dim=-1)
            variance_by_pilot_symbol.append(pilot_variance[..., None] * response_variance_gain)

        channels = torch.stack(channel_by_pilot_symbol, dim=-2)
        variances = torch.stack(variance_by_pilot_symbol, dim=-2)
        output_channels: list[Any] = []
        output_variances: list[Any] = []
        for symbol in range(self._num_symbols):
            right = next(
                (
                    index
                    for index, pilot_symbol in enumerate(self._dmrs_symbol_indices)
                    if pilot_symbol >= symbol
                ),
                len(self._dmrs_symbol_indices) - 1,
            )
            if self._dmrs_symbol_indices[right] <= symbol or right == 0:
                left = right
                weight = 0.0
            else:
                left = right - 1
                weight = (symbol - self._dmrs_symbol_indices[left]) / (
                    self._dmrs_symbol_indices[right] - self._dmrs_symbol_indices[left]
                )
            left_channel = channels[..., left, :]
            right_channel = channels[..., right, :]
            left_variance = variances[..., left, :]
            right_variance = variances[..., right, :]
            output_channels.append((1.0 - weight) * left_channel + weight * right_channel)
            output_variances.append(
                (1.0 - weight) ** 2 * left_variance + weight**2 * right_variance
            )
        channel_estimate = torch.stack(output_channels, dim=-2)
        error_variance = torch.stack(output_variances, dim=-2).expand_as(channel_estimate)
        return channel_estimate, error_variance


def make_finite_delay_dft_decoder(
    receiver: Any,
    *,
    max_delay_samples: int = 7,
    min_delay_samples: int = 0,
) -> Any:
    """Build a diagnostic PUSCH decoder using the experimental delay-domain estimator."""
    estimator = FiniteDelayDFTChannelEstimator(
        receiver,
        min_delay_samples=min_delay_samples,
        max_delay_samples=max_delay_samples,
    )
    return receiver.make_decoder_with_estimator(estimator)
