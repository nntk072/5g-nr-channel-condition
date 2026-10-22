"""Map native Sionna RT paths onto a Sionna NR resource grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class NRGridChannel:
    """Complex RT channel with explicit OFDM axes and conversion metadata.

    Channel axes are [batch, rx, rx antenna, tx, tx antenna, OFDM symbol,
    FFT bin]. The channel is not normalized unless explicitly requested.
    """

    channel: Any
    frequencies_hz: Any
    metadata: Mapping[str, Any]


def paths_to_nr_ofdm_channel(
    paths: Any,
    resource_grid: Any,
    *,
    batch_size: int = 1,
    sampling_frequency_hz: float = 1.0,
    num_time_steps: int = 1,
    normalize_delays: bool = False,
    normalize: bool = False,
    device: str = "cpu",
) -> NRGridChannel:
    """Evaluate RT Paths on the baseband bins of an NR OFDM grid.

    One RT time step denotes a static snapshot repeated over the grid's OFDM
    symbols. For time evolution, supply one RT sample per OFDM symbol and set
    ``sampling_frequency_hz`` to the reciprocal physical sample interval.
    Absolute RT delays are preserved by default.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional PyTorch/Sionna integration") from exc

    if not callable(getattr(paths, "cfr", None)):
        raise TypeError("paths must provide Sionna RT Paths.cfr")
    for name, value in (("batch_size", batch_size), ("num_time_steps", num_time_steps)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    if not isinstance(normalize_delays, bool) or not isinstance(normalize, bool):
        raise TypeError("normalization options must be booleans")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty PyTorch device string")

    fft_size = getattr(resource_grid, "fft_size", None)
    num_symbols = getattr(resource_grid, "num_ofdm_symbols", None)
    subcarrier_spacing = getattr(resource_grid, "subcarrier_spacing", None)
    if isinstance(fft_size, bool) or not isinstance(fft_size, int) or fft_size <= 0:
        raise ValueError("resource_grid must define a positive integer FFT size")
    if isinstance(num_symbols, bool) or not isinstance(num_symbols, int) or num_symbols <= 0:
        raise ValueError("resource_grid must define positive integer FFT and symbol sizes")
    if (
        isinstance(subcarrier_spacing, bool)
        or not isinstance(subcarrier_spacing, (int, float))
        or not isfinite(subcarrier_spacing)
        or subcarrier_spacing <= 0
    ):
        raise ValueError("resource_grid subcarrier_spacing must be finite and positive")
    if num_time_steps not in (1, num_symbols):
        raise ValueError(
            "num_time_steps must be one static snapshot or match the OFDM symbol count"
        )

    first_bin = -(fft_size // 2)
    stop_bin = fft_size // 2 if fft_size % 2 == 0 else fft_size // 2 + 1
    frequencies = torch.arange(first_bin, stop_bin, dtype=torch.float32, device="cpu") * float(
        subcarrier_spacing
    )
    response = paths.cfr(
        frequencies=frequencies.numpy(),
        sampling_frequency=sampling_frequency_hz,
        num_time_steps=num_time_steps,
        normalize_delays=normalize_delays,
        normalize=normalize,
        out_type="torch",
    )
    if not torch.is_tensor(response) or not torch.is_complex(response):
        raise TypeError("Sionna Paths.cfr(out_type='torch') must return a complex tensor")
    if response.ndim != 6 or tuple(response.shape[-2:]) != (num_time_steps, fft_size):
        raise ValueError(
            "Sionna Paths.cfr returned an unexpected [rx, rx_ant, tx, tx_ant, time, fft_bin] shape"
        )
    if num_time_steps == 1 and num_symbols > 1:
        response = response.expand(*response.shape[:-2], num_symbols, fft_size)
    channel = response.unsqueeze(0).expand(batch_size, *response.shape).to(device=device)

    return NRGridChannel(
        channel=channel,
        frequencies_hz=frequencies.to(device=device),
        metadata={
            "source": "sionna.rt.Paths.cfr",
            "normalize_delays": normalize_delays,
            "normalize_channel_power": normalize,
            "sampling_frequency_hz": sampling_frequency_hz,
            "num_time_steps": num_time_steps,
            "static_snapshot_repeated": num_time_steps == 1,
            "frequency_reference": "NR baseband subcarrier offsets",
            "channel_axes": [
                "batch",
                "rx",
                "rx_antenna",
                "tx",
                "tx_antenna",
                "ofdm_symbol",
                "fft_bin",
            ],
            "output_device": str(channel.device),
        },
    )


def paths_to_nr_ofdm_channel_at_times(
    paths: Any,
    resource_grid: Any,
    sample_times_s: Sequence[float],
    *,
    batch_size: int = 1,
    device: str = "cpu",
) -> NRGridChannel:
    """Evaluate the path-resolved CFR at explicit, possibly nonuniform times.

    Sionna 2.2 ``Paths.cfr`` samples at uniformly spaced times. This adapter
    uses its public ``Paths.cir`` baseband path coefficients and path Doppler
    values to evaluate the same complex channel at arbitrary symbol-start
    times, preserving absolute path delays and the native antenna axes.
    """
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional PyTorch/Sionna integration") from exc

    if not callable(getattr(paths, "cir", None)):
        raise TypeError("paths must provide Sionna RT Paths.cir")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty PyTorch device string")
    times = np.asarray(sample_times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ValueError("sample_times_s must be a non-empty finite vector")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_times_s must be strictly increasing")

    fft_size = getattr(resource_grid, "fft_size", None)
    subcarrier_spacing = getattr(resource_grid, "subcarrier_spacing", None)
    num_symbols = getattr(resource_grid, "num_ofdm_symbols", None)
    if isinstance(fft_size, bool) or not isinstance(fft_size, int) or fft_size <= 0:
        raise ValueError("resource_grid must define a positive integer FFT size")
    if isinstance(num_symbols, bool) or not isinstance(num_symbols, int) or num_symbols <= 0:
        raise ValueError("resource_grid must define a positive OFDM symbol count")
    if times.size != num_symbols:
        raise ValueError("sample_times_s length must match the resource-grid OFDM symbol count")
    if (
        isinstance(subcarrier_spacing, bool)
        or not isinstance(subcarrier_spacing, (int, float))
        or not isfinite(subcarrier_spacing)
        or subcarrier_spacing <= 0
    ):
        raise ValueError("resource_grid subcarrier_spacing must be finite and positive")

    # The one-time-step CIR gives each path's absolute-delay baseband
    # coefficient at t=0. Doppler is then applied per path at each exact time.
    coefficients, delays_s = paths.cir(
        sampling_frequency=1.0,
        num_time_steps=1,
        normalize_delays=False,
        out_type="numpy",
    )
    coefficients = np.asarray(coefficients)[..., 0]
    delays_s = np.asarray(delays_s, dtype=np.float64)

    def to_numpy(value: Any) -> Any:
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            return numpy_method()
        detach_method = getattr(value, "detach", None)
        if callable(detach_method):
            return detach_method().cpu().numpy()
        return value

    doppler_hz = np.asarray(to_numpy(paths.doppler), dtype=np.float64)
    valid = np.asarray(to_numpy(paths.valid), dtype=bool)
    if coefficients.ndim != 5 or not np.iscomplexobj(coefficients):
        raise ValueError("Paths.cir returned unexpected path coefficient axes")

    # Synthetic arrays omit element axes in path state; expand them to match
    # Paths.cir's canonical [rx, rx_ant, tx, tx_ant, path] coefficient shape.
    def antenna_axes(value: Any) -> Any:
        if value.ndim == 3:
            return value[:, None, :, None, :]
        return value

    delays_s = antenna_axes(delays_s)
    doppler_hz = antenna_axes(doppler_hz)
    valid = antenna_axes(valid)
    try:
        delays_s = np.broadcast_to(delays_s, coefficients.shape)
        doppler_hz = np.broadcast_to(doppler_hz, coefficients.shape)
        valid = np.broadcast_to(valid, coefficients.shape)
    except ValueError as exc:
        raise ValueError("Paths CIR, delay, Doppler, and valid-path axes do not align") from exc
    if not np.isfinite(coefficients).all() or not np.isfinite(delays_s).all():
        raise ValueError("Paths.cir returned nonfinite coefficients or delays")
    if not np.isfinite(doppler_hz[valid]).all():
        raise ValueError("valid RT paths contain nonfinite Doppler values")

    first_bin = -(fft_size // 2)
    stop_bin = fft_size // 2 if fft_size % 2 == 0 else fft_size // 2 + 1
    frequencies_hz = np.arange(first_bin, stop_bin, dtype=np.float64) * float(subcarrier_spacing)
    masked_coefficients = np.where(valid, coefficients, 0.0)
    safe_delays_s = np.where(valid, delays_s, 0.0)
    safe_doppler_hz = np.where(valid, doppler_hz, 0.0)
    delay_phase = np.exp(-2j * np.pi * safe_delays_s[..., :, None] * frequencies_hz)
    time_channels = []
    for time_s in times:
        time_phase = np.exp(2j * np.pi * safe_doppler_hz * time_s)
        time_channels.append(
            np.sum(
                (masked_coefficients * time_phase)[..., :, None] * delay_phase,
                axis=-2,
            )
        )
    channel_array = np.stack(time_channels, axis=-2).astype(np.complex64, copy=False)
    channel = torch.as_tensor(channel_array, dtype=torch.complex64, device=device)
    channel = channel.unsqueeze(0).expand(batch_size, *channel.shape)
    frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float32, device=device)
    return NRGridChannel(
        channel=channel,
        frequencies_hz=frequencies,
        metadata={
            "source": "Sionna RT Paths.cir and per-path Doppler at explicit times",
            "normalize_delays": False,
            "normalize_channel_power": False,
            "sample_times_s": times.tolist(),
            "num_time_steps": int(times.size),
            "static_snapshot_repeated": False,
            "frequency_reference": "NR baseband subcarrier offsets",
            "channel_axes": [
                "batch",
                "rx",
                "rx_antenna",
                "tx",
                "tx_antenna",
                "ofdm_symbol",
                "fft_bin",
            ],
            "output_device": str(channel.device),
        },
    )
