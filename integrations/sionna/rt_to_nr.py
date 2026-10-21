"""Map native Sionna RT paths onto a Sionna NR resource grid."""

from __future__ import annotations

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
