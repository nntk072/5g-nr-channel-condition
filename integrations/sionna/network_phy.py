"""Aligned multi-transmitter NR resource-grid composition and noise models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class NetworkGridComposition:
    """Linear source contributions before receiver noise.

    ``source_contributions`` has axes ``[batch, rx, rx_port, tx, symbol,
    subcarrier]``; ``received_grid`` sums the TX axis. TX port precoding is
    already represented in ``transmit_grids``. Power is total per transmitter
    averaged over ports and the full resource grid, in watts per grid RE.
    """

    source_contributions: Any
    received_grid: Any
    transmit_power_w_per_re: Any


@dataclass(frozen=True, slots=True)
class NetworkSINR:
    """Average desired, cochannel-interference, noise power, and linear SINR."""

    desired_power_w_per_re: Any
    interference_power_w_per_re: Any
    noise_power_w_per_re: Any
    sinr_linear: Any
    valid: Any


@dataclass(frozen=True, slots=True)
class NoisyNetworkGrid:
    """Joint network grid with receiver-specific independent complex AWGN."""

    received_grid: Any
    noise_grid: Any
    noise_variance_w_per_re: Any
    seed: int


def compose_downlink_resource_grids(
    channel: Any,
    transmit_grids: Any,
    transmit_power_w_per_re: Sequence[float] | Any,
) -> NetworkGridComposition:
    """Superpose aligned transmitter grids through every RT cross-link.

    Args:
        channel: Complex channel with axes ``[batch, rx, rx_port, tx, tx_port,
            symbol, subcarrier]``. Each entry is a dimensionless channel gain.
        transmit_grids: Complex, precoded per-port grids with axes ``[batch,
            tx, tx_port, symbol, subcarrier]``.
        transmit_power_w_per_re: Total average power for each transmitter,
            summed over its ports and averaged over batch, symbols and all FFT
            bins (including unused bins). A zero disables that transmitter.

    This is a synchronized, same-numerology frequency-domain model. It does
    not model relative timing/CFO, adjacent-channel leakage, or RF
    nonlinearity; callers must not use it for those conditions.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to compose network resource grids") from exc

    if not torch.is_tensor(channel) or not torch.is_complex(channel):
        raise TypeError("channel must be a complex PyTorch tensor")
    if not torch.is_tensor(transmit_grids) or not torch.is_complex(transmit_grids):
        raise TypeError("transmit_grids must be a complex PyTorch tensor")
    if channel.ndim != 7:
        raise ValueError(
            "channel axes must be [batch, rx, rx_port, tx, tx_port, symbol, subcarrier]"
        )
    if transmit_grids.ndim != 5:
        raise ValueError("transmit_grids axes must be [batch, tx, tx_port, symbol, subcarrier]")
    if tuple(channel.shape[0:1]) != tuple(transmit_grids.shape[0:1]):
        raise ValueError("channel and transmit grid batch dimensions differ")
    if tuple(channel.shape[3:]) != tuple(transmit_grids.shape[1:]):
        raise ValueError("channel TX-port/symbol/subcarrier axes do not match transmit grids")
    if any(size <= 0 for size in channel.shape) or any(size <= 0 for size in transmit_grids.shape):
        raise ValueError("channel and transmit grid axes must be non-empty")
    if channel.device != transmit_grids.device:
        raise ValueError("channel and transmit grids must be on the same device")
    if channel.dtype != transmit_grids.dtype:
        raise ValueError("channel and transmit grids must use the same complex dtype")
    if not bool(torch.isfinite(channel).all()) or not bool(torch.isfinite(transmit_grids).all()):
        raise ValueError("channel and transmit grids must be finite")

    tx_count = transmit_grids.shape[1]
    power = torch.as_tensor(
        transmit_power_w_per_re, dtype=torch.float64, device=transmit_grids.device
    )
    if power.shape != (tx_count,):
        raise ValueError("transmit_power_w_per_re must provide one power value per transmitter")
    if not bool(torch.isfinite(power).all()) or bool((power < 0).any()):
        raise ValueError("transmit powers must be finite and nonnegative")

    grid_power = transmit_grids.abs().to(torch.float64).square().sum(dim=2).mean(dim=(0, 2, 3))
    enabled = power > 0
    if bool((enabled & (grid_power <= 0)).any()):
        raise ValueError("a positive-power transmitter must have nonzero grid energy")
    scale = torch.zeros_like(power)
    scale[enabled] = torch.sqrt(power[enabled] / grid_power[enabled])
    scaled_grids = transmit_grids * scale.to(transmit_grids.real.dtype)[None, :, None, None, None]

    contributions = torch.einsum("bratpsf,btpsf->bratsf", channel, scaled_grids)
    return NetworkGridComposition(
        source_contributions=contributions,
        received_grid=contributions.sum(dim=3),
        transmit_power_w_per_re=power,
    )


def network_sinr(
    composition: NetworkGridComposition,
    serving_tx_indices: Sequence[int],
    noise_variance_w_per_re: Sequence[float] | Any,
) -> NetworkSINR:
    """Measure desired-link SINR after coherently summing all non-serving TXs.

    Powers are averaged over batch, RX ports, symbols, and subcarriers. The
    interference waveform is summed before taking its power, so cross terms
    are retained for deterministic, aligned grids.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to measure network SINR") from exc

    contributions = composition.source_contributions
    if not torch.is_tensor(contributions) or not torch.is_complex(contributions):
        raise TypeError("composition must contain complex PyTorch source contributions")
    if contributions.ndim != 6:
        raise ValueError(
            "source contributions must have [batch, rx, rx_port, tx, symbol, subcarrier] axes"
        )
    rx_count, tx_count = contributions.shape[1], contributions.shape[3]
    if len(serving_tx_indices) != rx_count:
        raise ValueError("serving_tx_indices must provide one transmitter index per receiver")
    serving = tuple(serving_tx_indices)
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < tx_count
        for index in serving
    ):
        raise ValueError("each serving transmitter index must address an existing transmitter")
    noise = torch.as_tensor(
        noise_variance_w_per_re, dtype=torch.float64, device=contributions.device
    )
    if noise.shape != (rx_count,):
        raise ValueError("noise_variance_w_per_re must provide one variance per receiver")
    if not bool(torch.isfinite(noise).all()) or bool((noise < 0).any()):
        raise ValueError("noise variances must be finite and nonnegative")

    desired_powers = []
    interference_powers = []
    for rx_index, serving_index in enumerate(serving):
        per_tx = contributions[:, rx_index]
        desired = per_tx[:, :, serving_index]
        interferer_indices = [index for index in range(tx_count) if index != serving_index]
        if interferer_indices:
            interference = per_tx[:, :, interferer_indices].sum(dim=2)
            interference_power = interference.abs().square().mean(dtype=torch.float64)
        else:
            interference_power = torch.zeros((), dtype=torch.float64, device=contributions.device)
        desired_powers.append(desired.abs().square().mean(dtype=torch.float64))
        interference_powers.append(interference_power)
    desired_power = torch.stack(desired_powers)
    interference_power = torch.stack(interference_powers)
    denominator = interference_power + noise
    valid = (desired_power > 0) | (denominator > 0)
    sinr = desired_power / denominator
    sinr = torch.where(valid, sinr, torch.full_like(sinr, float("nan")))
    return NetworkSINR(desired_power, interference_power, noise, sinr, valid)


def add_independent_receiver_awgn(
    composition: NetworkGridComposition,
    noise_variance_w_per_re: Sequence[float] | Any,
    *,
    seed: int,
) -> NoisyNetworkGrid:
    """Add independent complex AWGN streams to each receiver node.

    A receiver's array ports share the declared thermal variance but receive
    independent samples. Seeds are derived deterministically as ``seed + rx``
    to guarantee replay without reusing one random stream across UEs.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to add network receiver noise") from exc

    received = composition.received_grid
    if not torch.is_tensor(received) or not torch.is_complex(received) or received.ndim != 5:
        raise ValueError(
            "composition received_grid must be complex [batch, rx, rx_port, symbol, subcarrier]"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rx_count = received.shape[1]
    if seed + rx_count - 1 > 2**63 - 1:
        raise ValueError("seed and receiver-derived seeds must fit in [0, 2**63 - 1]")
    noise_variance = torch.as_tensor(
        noise_variance_w_per_re, dtype=torch.float64, device=received.device
    )
    if noise_variance.shape != (rx_count,):
        raise ValueError("noise_variance_w_per_re must provide one variance per receiver")
    if not bool(torch.isfinite(noise_variance).all()) or bool((noise_variance < 0).any()):
        raise ValueError("noise variances must be finite and nonnegative")

    noise_parts = []
    real_dtype = received.real.dtype
    sample_shape = (received.shape[0], *received.shape[2:])
    for rx_index in range(rx_count):
        generator = torch.Generator(device=received.device).manual_seed(seed + rx_index)
        real = torch.randn(
            sample_shape,
            dtype=real_dtype,
            device=received.device,
            generator=generator,
        )
        imag = torch.randn(
            sample_shape,
            dtype=real_dtype,
            device=received.device,
            generator=generator,
        )
        scale = torch.sqrt(noise_variance[rx_index].to(real_dtype) / 2.0)
        noise_parts.append(torch.complex(real, imag) * scale)
    noise_grid = torch.stack(noise_parts, dim=1)
    return NoisyNetworkGrid(
        received_grid=received + noise_grid,
        noise_grid=noise_grid,
        noise_variance_w_per_re=noise_variance,
        seed=seed,
    )
