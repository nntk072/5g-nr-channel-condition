"""Network-level channel records, validators, and waveform propagation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_NETWORK_FRACTIONAL_DELAY_KERNEL_RADIUS = 16


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _validated_nonnegative_vector(
    torch: Any,
    value: Any,
    *,
    size: int,
    device: Any,
    name: str,
) -> Any:
    """Validate real physical values before converting them to float64."""
    if _contains_boolean(value):
        raise ValueError(f"{name} must contain real numeric values, not Boolean values")
    try:
        probe = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if probe.layout != torch.strided:
        raise ValueError(f"{name} must be a dense real-valued vector")
    if probe.dtype == torch.bool or probe.is_complex():
        raise ValueError(f"{name} must contain real numeric values, not Boolean or complex")
    if not (probe.is_floating_point() or probe.dtype in integer_dtypes):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        raw = torch.as_tensor(value, dtype=torch.float64, device=device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if raw.shape != (size,):
        expected = (
            "one power value per transmitter"
            if name == "transmit_power_w_per_re"
            else "one variance per receiver"
        )
        raise ValueError(f"{name} must provide {expected}")
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")
    return raw


def _validated_real_values(
    torch: Any,
    value: Any,
    *,
    shape: tuple[int, ...],
    device: Any,
    name: str,
    nonnegative: bool = False,
) -> Any:
    """Validate finite dense real inputs before converting to float64."""
    if _contains_boolean(value):
        raise ValueError(f"{name} must contain real numeric values, not Boolean values")
    try:
        probe = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if probe.layout != torch.strided:
        raise ValueError(f"{name} must be a dense real-valued array")
    if probe.dtype == torch.bool or probe.is_complex():
        raise ValueError(f"{name} must contain real numeric values, not Boolean or complex")
    if not (probe.is_floating_point() or probe.dtype in integer_dtypes):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        result = torch.as_tensor(value, dtype=torch.float64, device=device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if tuple(result.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    if nonnegative and bool((result < 0).any()):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_composition(torch: Any, composition: Any) -> tuple[Any, Any]:
    contributions = getattr(composition, "source_contributions", None)
    received = getattr(composition, "received_grid", None)
    powers = getattr(composition, "transmit_power_w_per_re", None)
    if (
        contributions is None
        or not torch.is_tensor(contributions)
        or not torch.is_complex(contributions)
    ):
        raise TypeError("composition must contain complex PyTorch source contributions")
    if contributions is None:
        raise TypeError("composition must contain complex PyTorch source contributions")
    if contributions.ndim != 6:
        raise ValueError(
            "source contributions must have [batch, rx, rx_port, tx, symbol, subcarrier] axes"
        )
    if received is None or not torch.is_tensor(received) or not torch.is_complex(received):
        raise ValueError(
            "composition received_grid must be complex [batch, rx, rx_port, symbol, subcarrier]"
        )
    if received.ndim != 5:
        raise ValueError(
            "composition received_grid must be complex [batch, rx, rx_port, symbol, subcarrier]"
        )
    expected_shape = (
        contributions.shape[0],
        contributions.shape[1],
        contributions.shape[2],
        contributions.shape[4],
        contributions.shape[5],
    )
    if tuple(received.shape) != expected_shape:
        raise ValueError("received_grid shape does not match source contributions")
    if received.device != contributions.device or received.dtype != contributions.dtype:
        raise ValueError("received_grid and source contributions must share device and dtype")
    if not bool(torch.isfinite(received).all()) or not bool(torch.isfinite(contributions).all()):
        raise ValueError("received_grid and source contributions must be finite")
    summed = contributions.sum(dim=3)
    if not torch.allclose(received, summed, rtol=1e-5, atol=1e-7):
        raise ValueError("received_grid must equal the sum of source contributions")
    if powers is None or not torch.is_tensor(powers):
        raise ValueError("transmit_power_w_per_re must provide one value per transmitter")
    if powers.shape != (contributions.shape[3],):
        raise ValueError("transmit_power_w_per_re must provide one value per transmitter")
    if powers.device != contributions.device or not powers.is_floating_point():
        raise ValueError("transmit powers must be a real tensor on the contribution device")
    if not bool(torch.isfinite(powers).all()) or bool((powers < 0).any()):
        raise ValueError("transmit powers must be finite and nonnegative")
    return contributions, received


def _mean_complex_power(torch: Any, samples: Any) -> Any:
    return (
        samples.real.to(dtype=torch.float64).square()
        + samples.imag.to(dtype=torch.float64).square()
    ).mean()


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
class NetworkLinkBudget:
    """Large-scale per-link budget applied to unit-mean small-scale fading.

    ``channel`` has the same seven axes as the input small-scale channel.
    Power gain is ``10 ** ((G_tx + G_rx + G_tx_beam + G_rx_beam - L_path) / 10)``.
    The reported mean received power assumes unit-mean small-scale fading; an
    instantaneous received power is available only after channel composition.
    """

    channel: Any
    path_loss_db: Any
    tx_antenna_gain_dbi: Any
    rx_antenna_gain_dbi: Any
    tx_beam_gain_dbi: Any
    rx_beam_gain_dbi: Any
    net_gain_db: Any
    power_gain_linear: Any
    transmit_power_w_per_re: Any
    mean_received_power_w_per_re: Any
    input_channel_semantics: str
    subcarrier_frequencies_hz: Any | None


@dataclass(frozen=True, slots=True)
class NetworkSINR:
    """Average desired, cochannel-interference, noise power, and linear SINR."""

    desired_power_w_per_re: Any
    interference_power_w_per_re: Any
    noise_power_w_per_re: Any
    sinr_linear: Any
    valid: Any


@dataclass(frozen=True, slots=True)
class NetworkResourceSINR:
    """Per-RE desired, interference, noise power, and linear SINR.

    Power arrays have ``[rx, symbol, subcarrier]`` axes. Averages are over
    independent batch draws and receive ports only, so resource-dependent
    scheduling, notches, and frequency-selective link budgets remain visible.
    """

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


@dataclass(frozen=True, slots=True)
class NetworkWaveformComposition:
    """Per-link channel-filtered waveforms aligned to a receiver capture.

    ``source_contributions`` has axes ``[batch, rx, rx_port, tx, sample]``;
    ``received_waveform`` sums the transmitter axis. Each input link waveform
    is assumed already scaled and channel-filtered. ``link_start_samples``
    gives the integer offset of each link's first sample relative to capture
    sample zero. All links must use the same sample rate and carrier/numerology.
    Per-link residual CFO, initial phase, fractional delay, and shared node-level
    phase-noise processes are applied on the capture timeline. Adjacent-channel
    leakage and nonlinear RF effects are not modeled.
    """

    source_contributions: Any
    received_waveform: Any
    link_start_samples: Any
    link_power_gain_linear: Any
    sample_rate_hz: float
    link_cfo_hz: Any
    link_initial_phase_rad: Any
    link_fractional_delay_samples: Any
    fractional_delay_kernel_radius: int
    tx_phase_noise_linewidth_hz: Any
    rx_phase_noise_linewidth_hz: Any
    phase_noise_seed: int | None
    rx_iq_gain_imbalance_db: Any
    rx_iq_phase_imbalance_rad: Any
    link_filter_taps: Any | None = None


@dataclass(frozen=True, slots=True)
class NetworkTransmitterWaveform:
    """Per-transmitter waveform after memoryless TX I/Q imbalance, before propagation."""

    waveform: Any
    tx_iq_gain_imbalance_db: Any
    tx_iq_phase_imbalance_rad: Any
    signal_stage: str = "transmitter_iq_imbalance_before_propagation"


@dataclass(frozen=True, slots=True)
class NetworkMIMOChannelPropagation:
    """Per-transmitter waveforms propagated through declared causal MIMO FIRs."""

    link_waveforms: Any
    channel_fir_taps: Any
    output_sample_count: int


@dataclass(frozen=True, slots=True)
class NoisyNetworkWaveform:
    """Receiver-specific independent complex AWGN added to a network capture."""

    received_waveform: Any
    noise_waveform: Any
    noise_variance_w_per_sample: Any
    seed: int


def propagate_network_transmit_waveforms(
    transmitter_waveform: NetworkTransmitterWaveform,
    channel_fir_taps: Any,
    *,
    max_output_elements: int = 64_000_000,
) -> NetworkMIMOChannelPropagation:
    """Apply declared causal MIMO channel FIRs to transmitter-stage waveforms.

    The FIR has ``[rx, rx_port, tx, tx_port, tap]`` axes. Full causal
    convolution is retained, including the ``tap_count - 1`` tail samples.
    Taps are caller-supplied channel truth; the function does not derive them
    from a scene or infer calibration from a link-budget profile.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install PyTorch to propagate network transmitter waveforms") from error

    waveform = getattr(transmitter_waveform, "waveform", None)
    if (
        waveform is None
        or not torch.is_tensor(waveform)
        or waveform.layout != torch.strided
        or not torch.is_complex(waveform)
        or waveform.ndim != 4
    ):
        raise TypeError(
            "transmitter_waveform must contain dense complex [batch, tx, tx_port, sample]"
        )
    try:
        taps = torch.as_tensor(channel_fir_taps, device=waveform.device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("channel_fir_taps must be a complex MIMO FIR tensor") from error
    if (
        taps.layout != torch.strided
        or not torch.is_complex(taps)
        or taps.ndim != 5
        or tuple(taps.shape[2:4]) != tuple(waveform.shape[1:3])
        or any(size <= 0 for size in taps.shape)
        or taps.shape[-1] > 128
    ):
        raise ValueError(
            "channel_fir_taps must have [rx, rx_port, tx, tx_port, tap] axes with 1–128 taps"
        )
    if taps.dtype != waveform.dtype:
        raise ValueError("channel_fir_taps and transmitter waveform must share complex dtype")
    if not bool(torch.isfinite(waveform).all()) or not bool(torch.isfinite(taps).all()):
        raise ValueError("transmitter waveform and channel FIR taps must be finite")
    if (
        isinstance(max_output_elements, bool)
        or not isinstance(max_output_elements, int)
        or max_output_elements < 1
    ):
        raise ValueError("max_output_elements must be a positive integer")

    batch, tx_count, _tx_ports, sample_count = (int(value) for value in waveform.shape)
    rx_count, rx_ports, _tx_count, _tx_ports, tap_count = (int(value) for value in taps.shape)
    output_samples = sample_count + tap_count - 1
    link_cells = batch * rx_count * tx_count * rx_ports * output_samples
    temporary_cells = batch * rx_count * tx_count * rx_ports * sample_count
    estimated_cells = waveform.numel() + taps.numel() + link_cells + temporary_cells
    if estimated_cells > max_output_elements:
        raise ValueError("MIMO propagation waveform and FIR work exceed max_output_elements")

    links = torch.zeros(
        (batch, rx_count, tx_count, rx_ports, output_samples),
        dtype=waveform.dtype,
        device=waveform.device,
    )
    for tap_index in range(tap_count):
        contribution = torch.einsum("rptq,btqs->brtps", taps[..., tap_index], waveform)
        if not bool(torch.isfinite(contribution).all()):
            raise ValueError("MIMO channel propagation produced non-finite samples")
        links[..., tap_index : tap_index + sample_count] += contribution
    return NetworkMIMOChannelPropagation(links, taps, output_samples)


def _fractional_delay_capture_slice(
    torch: Any,
    source_waveform: Any,
    *,
    output_indices: Any,
    delay_samples: float,
    kernel_radius: int,
) -> Any:
    """Interpolate only requested delayed samples using bounded scratch space."""
    source_positions = output_indices - delay_samples
    centers = torch.floor(source_positions).to(torch.int64)
    taps = range(-kernel_radius, kernel_radius + 1)
    weight_sum = torch.zeros_like(source_positions)

    def tap_weights(tap: int) -> tuple[Any, Any]:
        source_indices = centers + tap
        distance = source_indices.to(source_waveform.real.dtype) - source_positions
        normalized_distance = distance / kernel_radius
        window = (
            0.42
            + 0.5 * torch.cos(math.pi * normalized_distance)
            + 0.08 * torch.cos(2.0 * math.pi * normalized_distance)
        )
        window = torch.where(torch.abs(normalized_distance) <= 1.0, window, 0.0)
        weights = torch.sinc(distance) * window
        return source_indices, weights

    for tap in taps:
        _, weights = tap_weights(tap)
        weight_sum += weights
    if not bool(torch.isfinite(weight_sum).all()) or bool((weight_sum.abs() <= 1e-12).any()):
        raise ValueError("fractional-delay interpolation weights are not normalizable")

    output = torch.zeros(
        (*source_waveform.shape[:-1], output_indices.numel()),
        dtype=source_waveform.dtype,
        device=source_waveform.device,
    )
    source_sample_count = source_waveform.shape[-1]
    for tap in taps:
        source_indices, weights = tap_weights(tap)
        valid = (source_indices >= 0) & (source_indices < source_sample_count)
        safe_indices = source_indices.clamp(0, source_sample_count - 1)
        samples = source_waveform.index_select(-1, safe_indices)
        normalized_weights = (weights / weight_sum).to(source_waveform.real.dtype)
        output += samples * valid.to(source_waveform.real.dtype) * normalized_weights
    return output


def _shared_phase_noise_processes(
    torch: Any,
    *,
    batch_size: int,
    node_count: int,
    sample_count: int,
    linewidth_hz: Any,
    sample_rate_hz: float,
    generator: Any,
    device: Any,
) -> Any:
    """Generate one Wiener phase trajectory per batch draw and RF node."""
    phase = torch.zeros((batch_size, node_count, sample_count), dtype=torch.float64, device=device)
    if sample_count == 1:
        return phase
    linewidth_ratio = linewidth_hz / sample_rate_hz
    if not bool(torch.isfinite(linewidth_ratio).all()):
        raise ValueError("phase-noise linewidth/sample-rate ratio must be finite")
    increment_scale = torch.sqrt(linewidth_ratio) * math.sqrt(2.0 * math.pi)
    increments = torch.randn(
        (batch_size, node_count, sample_count - 1),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    increments *= increment_scale[None, :, None]
    torch.cumsum(increments, dim=-1, out=phase[..., 1:])
    if not bool(torch.isfinite(phase).all()):
        raise ValueError("generated phase-noise trajectory is not finite")
    return phase
