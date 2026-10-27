"""Aligned multi-transmitter NR grid/waveform composition and noise models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

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
    if not torch.is_tensor(contributions) or not torch.is_complex(contributions):
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
class NoisyNetworkWaveform:
    """Receiver-specific independent complex AWGN added to a network capture."""

    received_waveform: Any
    noise_waveform: Any
    noise_variance_w_per_sample: Any
    seed: int


def apply_network_link_budget(
    small_scale_channel: Any,
    *,
    path_loss_db: Any,
    tx_antenna_gain_dbi: Any,
    rx_antenna_gain_dbi: Any,
    tx_beam_gain_dbi: Any,
    rx_beam_gain_dbi: Any,
    transmit_power_w_per_re: Any,
    input_channel_semantics: str,
    subcarrier_frequencies_hz: Any | None = None,
) -> NetworkLinkBudget:
    """Apply declared large-scale gains/loss to a normalized fading channel.

    Link-budget inputs may be frequency-flat per ``[rx, tx]`` link or vary
    over the channel's final subcarrier axis. Frequency-varying path loss and
    beam gains use ``[rx, tx, subcarrier]``; frequency-varying antenna gains
    use ``[tx, subcarrier]`` and ``[rx, subcarrier]``. Any frequency-varying
    input requires the exact, strictly increasing absolute RF frequency grid
    in ``subcarrier_frequencies_hz``. This applies caller-supplied profiles;
    it does not derive propagation loss or antenna patterns. The required
    semantics declaration prevents accidentally applying this large-scale
    budget to a channel that already contains path loss.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to apply network link budgets") from exc

    if input_channel_semantics != "unit_mean_small_scale_fading":
        raise ValueError(
            "input_channel_semantics must be unit_mean_small_scale_fading; "
            "large-scale loss must not be applied twice"
        )
    if not torch.is_tensor(small_scale_channel) or not torch.is_complex(small_scale_channel):
        raise TypeError("small_scale_channel must be a complex PyTorch tensor")
    if small_scale_channel.ndim != 7:
        raise ValueError(
            "small_scale_channel axes must be [batch, rx, rx_port, tx, tx_port, symbol, subcarrier]"
        )
    if any(size <= 0 for size in small_scale_channel.shape):
        raise ValueError("small_scale_channel axes must be non-empty")
    if not bool(torch.isfinite(small_scale_channel).all()):
        raise ValueError("small_scale_channel must be finite")
    rx_count, tx_count = small_scale_channel.shape[1], small_scale_channel.shape[3]
    device = small_scale_channel.device
    subcarrier_count = small_scale_channel.shape[-1]
    profiles = (
        (
            "path_loss_db",
            path_loss_db,
            (rx_count, tx_count),
            (rx_count, tx_count, subcarrier_count),
        ),
        (
            "tx_antenna_gain_dbi",
            tx_antenna_gain_dbi,
            (tx_count,),
            (tx_count, subcarrier_count),
        ),
        (
            "rx_antenna_gain_dbi",
            rx_antenna_gain_dbi,
            (rx_count,),
            (rx_count, subcarrier_count),
        ),
        (
            "tx_beam_gain_dbi",
            tx_beam_gain_dbi,
            (rx_count, tx_count),
            (rx_count, tx_count, subcarrier_count),
        ),
        (
            "rx_beam_gain_dbi",
            rx_beam_gain_dbi,
            (rx_count, tx_count),
            (rx_count, tx_count, subcarrier_count),
        ),
    )
    validated: dict[str, Any] = {}
    frequency_dependent = False
    for name, value, flat_shape, varying_shape in profiles:
        try:
            probe = torch.as_tensor(value, device=device)
        except (TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError(f"{name} must contain real numeric values") from error
        shape = tuple(probe.shape)
        selected_shape: tuple[int, ...]
        if shape == flat_shape:
            selected_shape = flat_shape
        elif shape == varying_shape:
            selected_shape = varying_shape
            frequency_dependent = True
        else:
            raise ValueError(f"{name} must have shape {flat_shape} or {varying_shape}")
        validated[name] = _validated_real_values(
            torch,
            value,
            shape=selected_shape,
            device=device,
            name=name,
            nonnegative=name == "path_loss_db",
        )
    frequencies = None
    if frequency_dependent and subcarrier_frequencies_hz is None:
        raise ValueError(
            "subcarrier_frequencies_hz is required for frequency-dependent link budgets"
        )
    if subcarrier_frequencies_hz is not None:
        frequencies = _validated_real_values(
            torch,
            subcarrier_frequencies_hz,
            shape=(subcarrier_count,),
            device=device,
            name="subcarrier_frequencies_hz",
            nonnegative=True,
        )
        if bool((frequencies <= 0).any()):
            raise ValueError("subcarrier_frequencies_hz must be strictly positive")
        if subcarrier_count > 1 and bool((frequencies[1:] <= frequencies[:-1]).any()):
            raise ValueError("subcarrier_frequencies_hz must be strictly increasing")

    path_loss = validated["path_loss_db"]
    tx_gain = validated["tx_antenna_gain_dbi"]
    rx_gain = validated["rx_antenna_gain_dbi"]
    tx_beam = validated["tx_beam_gain_dbi"]
    rx_beam = validated["rx_beam_gain_dbi"]
    tx_power = _validated_nonnegative_vector(
        torch,
        transmit_power_w_per_re,
        size=tx_count,
        device=device,
        name="transmit_power_w_per_re",
    )
    if frequency_dependent:

        def link_profile(value: Any, flat_shape: tuple[int, ...], mode: str) -> Any:
            if tuple(value.shape) == flat_shape:
                value = value.unsqueeze(-1)
            if mode == "tx":
                return value[None, :, :]
            if mode == "rx":
                return value[:, None, :]
            return value

        tx_link_gain = link_profile(tx_gain, (tx_count,), "tx")
        rx_link_gain = link_profile(rx_gain, (rx_count,), "rx")
        tx_link_beam = link_profile(tx_beam, (rx_count, tx_count), "link")
        rx_link_beam = link_profile(rx_beam, (rx_count, tx_count), "link")
        path_loss_profile = link_profile(path_loss, (rx_count, tx_count), "link")
        net_gain_db = tx_link_gain + rx_link_gain + tx_link_beam + rx_link_beam - path_loss_profile
    else:
        net_gain_db = tx_gain[None, :] + rx_gain[:, None] + tx_beam + rx_beam - path_loss
    amplitude_gain = torch.pow(
        torch.tensor(10.0, dtype=torch.float64, device=device), net_gain_db / 20.0
    )
    power_gain = amplitude_gain.square()
    if (
        not bool(torch.isfinite(amplitude_gain).all())
        or not bool(torch.isfinite(power_gain).all())
        or bool(((amplitude_gain > 0) & (power_gain == 0)).any())
    ):
        raise ValueError("large-scale link-budget gain is not representable in float64")
    output_gain = amplitude_gain.to(dtype=small_scale_channel.real.dtype)
    if not bool(torch.isfinite(output_gain).all()) or bool(
        ((amplitude_gain > 0) & (output_gain == 0)).any()
    ):
        raise ValueError("large-scale link-budget gain is not representable in channel dtype")
    if frequency_dependent:
        scaled_channel = small_scale_channel * output_gain[None, :, None, :, None, None, :]
    else:
        scaled_channel = small_scale_channel * output_gain[None, :, None, :, None, None, None]
    if not bool(torch.isfinite(scaled_channel).all()):
        raise ValueError("link-budget-scaled channel is not finite in channel dtype")
    if frequency_dependent:
        mean_received_power = power_gain * tx_power[None, :, None]
    else:
        mean_received_power = power_gain * tx_power[None, :]
    if not bool(torch.isfinite(mean_received_power).all()):
        raise ValueError("mean received link power is not representable in float64")
    return NetworkLinkBudget(
        channel=scaled_channel,
        path_loss_db=path_loss,
        tx_antenna_gain_dbi=tx_gain,
        rx_antenna_gain_dbi=rx_gain,
        tx_beam_gain_dbi=tx_beam,
        rx_beam_gain_dbi=rx_beam,
        net_gain_db=net_gain_db,
        power_gain_linear=power_gain,
        transmit_power_w_per_re=tx_power,
        mean_received_power_w_per_re=mean_received_power,
        input_channel_semantics=input_channel_semantics,
        subcarrier_frequencies_hz=frequencies,
    )


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
    power = _validated_nonnegative_vector(
        torch,
        transmit_power_w_per_re,
        size=tx_count,
        device=transmit_grids.device,
        name="transmit_power_w_per_re",
    )

    real64 = transmit_grids.real.to(torch.float64)
    imag64 = transmit_grids.imag.to(torch.float64)
    grid_power = (real64.square() + imag64.square()).sum(dim=2).mean(dim=(0, 2, 3))
    if not bool(torch.isfinite(grid_power).all()):
        raise ValueError("transmit grid power is not representable in float64")
    enabled = power > 0
    if bool((enabled & (grid_power <= 0)).any()):
        raise ValueError("a positive-power transmitter must have nonzero grid energy")
    scale = torch.zeros_like(power)
    scale[enabled] = torch.sqrt(power[enabled] / grid_power[enabled])
    output_scale = scale.to(transmit_grids.real.dtype)
    if not bool(torch.isfinite(output_scale).all()) or bool(
        (enabled & (scale > 0) & (output_scale == 0)).any()
    ):
        raise ValueError("transmit power scaling is not representable in the waveform dtype")
    scaled_grids = transmit_grids * output_scale[None, :, None, None, None]
    if not bool(torch.isfinite(scaled_grids).all()):
        raise ValueError("scaled transmit grids are not finite in the waveform dtype")

    contributions = torch.einsum("bratpsf,btpsf->bratsf", channel, scaled_grids)
    if not bool(torch.isfinite(contributions).all()):
        raise ValueError("channel-filtered source contributions are not finite")
    received_grid = contributions.sum(dim=3)
    if not bool(torch.isfinite(received_grid).all()):
        raise ValueError("superposed received grid is not finite")
    return NetworkGridComposition(
        source_contributions=contributions,
        received_grid=received_grid,
        transmit_power_w_per_re=power,
    )


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


def compose_received_waveforms(
    link_waveforms: Any,
    *,
    link_start_samples: Any,
    capture_num_samples: int,
    sample_rate_hz: float,
    link_power_gain_linear: Any | None = None,
    link_filter_taps: Any | None = None,
    link_cfo_hz: Any | None = None,
    link_initial_phase_rad: Any | None = None,
    link_fractional_delay_samples: Any | None = None,
    tx_phase_noise_linewidth_hz: Any | None = None,
    rx_phase_noise_linewidth_hz: Any | None = None,
    phase_noise_seed: int | None = None,
    rx_iq_gain_imbalance_db: Any | None = None,
    rx_iq_phase_imbalance_rad: Any | None = None,
    max_output_elements: int = 64_000_000,
) -> NetworkWaveformComposition:
    """Align and sum already propagated transmitter waveforms at each receiver.

    ``link_waveforms`` has axes ``[batch, rx, tx, rx_port, sample]`` and holds
    one small-scale channel-filtered contribution per transmitter/receiver
    pair. Optional ``link_filter_taps`` supplies a causal, complex baseband
    FIR for each RX/TX pair, with axes ``[rx, tx, tap]``; it is applied after
    the supplied small-scale channel response and before oscillator and
    receiver impairments. The caller is responsible for deriving the taps from
    a calibrated or otherwise declared complex frequency response. The API
    does not infer phase from frequency-dependent power-only link-budget data.
    Optional ``link_power_gain_linear`` supplies a nonnegative flat power gain
    for each RX/TX pair (for example, ``NetworkLinkBudget`` gains); its square
    root is applied once to the waveform amplitude. Positive start
    offsets insert leading zeros; negative offsets clip samples that arrived
    before the capture window. Samples outside the capture are discarded.
    Optional CFO and initial phase arrays have ``[rx, tx]`` axes. They describe
    each link's residual oscillator relative to capture time zero and are
    applied after integer alignment, so delayed links accrue phase over their
    arrival time. Optional fractional delays use the same ``[rx, tx]`` axes,
    are bounded to less than one sample, and use a 33-tap Blackman-windowed
    sinc interpolator with zero extension outside the source waveform. This
    does not model PLL-shaped phase-noise spectra, adjacent-channel leakage, or
    nonlinear RF effects. Optional transmitter and receiver linewidth vectors
    apply independent Wiener oscillators per node and batch draw; each node's
    trajectory is shared across its incident links and antenna ports. A seed
    is required when either linewidth is positive. Optional receiver I/Q gain
    and quadrature-skew vectors are applied after channel filtering; transmitter
    I/Q imbalance must be applied before propagation. Do not pass an already
    path-loss-scaled waveform together with its gain matrix.

    Outputs and temporary filter/impairment work are bounded by
    ``max_output_elements`` before allocating capture tensors. FIRs are limited
    to 128 taps, and filter work accounts for every tap and fractional-delay
    interpolation step.
    This is a common-sample-rate baseband composition primitive, not a complete
    asynchronous or multi-numerology network simulator.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to compose received waveforms") from exc

    if not torch.is_tensor(link_waveforms) or not torch.is_complex(link_waveforms):
        raise TypeError("link_waveforms must be a complex PyTorch tensor")
    if link_waveforms.ndim != 5:
        raise ValueError("link_waveforms must have [batch, rx, tx, rx_port, sample] axes")
    if any(size <= 0 for size in link_waveforms.shape):
        raise ValueError("link_waveforms axes must be non-empty")
    if not bool(torch.isfinite(link_waveforms).all()):
        raise ValueError("link_waveforms must contain only finite samples")
    if (
        isinstance(capture_num_samples, bool)
        or not isinstance(capture_num_samples, int)
        or capture_num_samples <= 0
    ):
        raise ValueError("capture_num_samples must be a positive integer")
    if (
        isinstance(max_output_elements, bool)
        or not isinstance(max_output_elements, int)
        or max_output_elements <= 0
    ):
        raise ValueError("max_output_elements must be a positive integer")
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, (int, float))
        or not math.isfinite(sample_rate_hz)
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be finite and positive")

    batch, rx_count, tx_count, rx_ports, _input_samples = link_waveforms.shape
    try:
        offset_probe = torch.as_tensor(link_start_samples, device=link_waveforms.device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("link_start_samples must contain integer sample offsets") from error
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if (
        offset_probe.layout != torch.strided
        or offset_probe.dtype == torch.bool
        or offset_probe.is_floating_point()
        or offset_probe.is_complex()
        or offset_probe.dtype not in integer_dtypes
    ):
        raise ValueError("link_start_samples must contain integer sample offsets")
    if tuple(offset_probe.shape) != (rx_count, tx_count):
        raise ValueError(f"link_start_samples must have shape {(rx_count, tx_count)}")
    offsets = offset_probe.to(torch.int64)
    if link_power_gain_linear is None:
        power_gain = torch.ones(
            (rx_count, tx_count), dtype=torch.float64, device=link_waveforms.device
        )
    else:
        power_gain = _validated_real_values(
            torch,
            link_power_gain_linear,
            shape=(rx_count, tx_count),
            device=link_waveforms.device,
            name="link_power_gain_linear",
            nonnegative=True,
        )
    if link_filter_taps is None:
        filter_taps = torch.ones(
            (rx_count, tx_count, 1), dtype=link_waveforms.dtype, device=link_waveforms.device
        )
    else:
        try:
            filter_probe = torch.as_tensor(link_filter_taps, device=link_waveforms.device)
        except (TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError("link_filter_taps must contain complex numeric values") from error
        if (
            filter_probe.layout != torch.strided
            or not filter_probe.is_complex()
            or filter_probe.ndim != 3
            or tuple(filter_probe.shape[:2]) != (rx_count, tx_count)
            or filter_probe.shape[2] < 1
            or filter_probe.shape[2] > 128
        ):
            raise ValueError(
                "link_filter_taps must be a dense complex [rx, tx, tap] array with 1–128 taps"
            )
        if not bool(torch.isfinite(filter_probe).all()):
            raise ValueError("link_filter_taps must contain only finite coefficients")
        filter_taps = filter_probe.to(dtype=link_waveforms.dtype)
        if not bool(torch.isfinite(filter_taps).all()) or bool(
            ((filter_probe != 0) & (filter_taps == 0)).any()
        ):
            raise ValueError("link_filter_taps are not representable in waveform precision")
    filter_is_identity = filter_taps.shape[2] == 1 and bool((filter_taps[..., 0] == 1).all())
    if link_cfo_hz is None:
        cfo = torch.zeros((rx_count, tx_count), dtype=torch.float64, device=link_waveforms.device)
    else:
        cfo = _validated_real_values(
            torch,
            link_cfo_hz,
            shape=(rx_count, tx_count),
            device=link_waveforms.device,
            name="link_cfo_hz",
        )
    if link_initial_phase_rad is None:
        initial_phase = torch.zeros(
            (rx_count, tx_count), dtype=torch.float64, device=link_waveforms.device
        )
    else:
        initial_phase = _validated_real_values(
            torch,
            link_initial_phase_rad,
            shape=(rx_count, tx_count),
            device=link_waveforms.device,
            name="link_initial_phase_rad",
        )
    if link_fractional_delay_samples is None:
        fractional_delay = torch.zeros(
            (rx_count, tx_count), dtype=torch.float64, device=link_waveforms.device
        )
    else:
        fractional_delay = _validated_real_values(
            torch,
            link_fractional_delay_samples,
            shape=(rx_count, tx_count),
            device=link_waveforms.device,
            name="link_fractional_delay_samples",
        )
    if bool((fractional_delay.abs() >= 1.0).any()):
        raise ValueError("link fractional delays must have magnitude below one sample")
    if tx_phase_noise_linewidth_hz is None:
        tx_linewidth = torch.zeros(tx_count, dtype=torch.float64, device=link_waveforms.device)
    else:
        tx_linewidth = _validated_real_values(
            torch,
            tx_phase_noise_linewidth_hz,
            shape=(tx_count,),
            device=link_waveforms.device,
            name="tx_phase_noise_linewidth_hz",
            nonnegative=True,
        )
    if rx_phase_noise_linewidth_hz is None:
        rx_linewidth = torch.zeros(rx_count, dtype=torch.float64, device=link_waveforms.device)
    else:
        rx_linewidth = _validated_real_values(
            torch,
            rx_phase_noise_linewidth_hz,
            shape=(rx_count,),
            device=link_waveforms.device,
            name="rx_phase_noise_linewidth_hz",
            nonnegative=True,
        )
    phase_noise_enabled = bool((tx_linewidth > 0).any() or (rx_linewidth > 0).any())
    if phase_noise_seed is None:
        if phase_noise_enabled:
            raise ValueError("phase_noise_seed is required when oscillator linewidth is positive")
    elif (
        isinstance(phase_noise_seed, bool)
        or not isinstance(phase_noise_seed, int)
        or phase_noise_seed < 0
        or phase_noise_seed > 2**63 - 1
    ):
        raise ValueError("phase_noise_seed must be an integer in [0, 2**63 - 1]")
    if rx_iq_gain_imbalance_db is None:
        rx_iq_gain = torch.zeros(rx_count, dtype=torch.float64, device=link_waveforms.device)
    else:
        rx_iq_gain = _validated_real_values(
            torch,
            rx_iq_gain_imbalance_db,
            shape=(rx_count,),
            device=link_waveforms.device,
            name="rx_iq_gain_imbalance_db",
        )
    if rx_iq_phase_imbalance_rad is None:
        rx_iq_phase = torch.zeros(rx_count, dtype=torch.float64, device=link_waveforms.device)
    else:
        rx_iq_phase = _validated_real_values(
            torch,
            rx_iq_phase_imbalance_rad,
            shape=(rx_count,),
            device=link_waveforms.device,
            name="rx_iq_phase_imbalance_rad",
        )
    iq_active = bool((rx_iq_gain != 0).any() or (rx_iq_phase != 0).any())
    i_gain = torch.pow(10.0, rx_iq_gain / 40.0)
    q_gain = torch.pow(10.0, -rx_iq_gain / 40.0)
    phase_skew = torch.remainder(rx_iq_phase + math.pi, 2.0 * math.pi) - math.pi
    iq_real_scale = i_gain.to(dtype=link_waveforms.real.dtype)
    iq_q_scale = q_gain.to(dtype=link_waveforms.real.dtype)
    iq_sin = torch.sin(phase_skew).to(dtype=link_waveforms.real.dtype)
    iq_cos = torch.cos(phase_skew).to(dtype=link_waveforms.real.dtype)
    if (
        not bool(torch.isfinite(iq_real_scale).all())
        or not bool(torch.isfinite(iq_q_scale).all())
        or not bool(torch.isfinite(iq_sin).all())
        or not bool(torch.isfinite(iq_cos).all())
        or bool(((i_gain > 0) & (iq_real_scale == 0)).any())
        or bool(((q_gain > 0) & (iq_q_scale == 0)).any())
    ):
        raise ValueError("receiver I/Q imbalance is not representable in waveform precision")
    cycles_per_sample = cfo / float(sample_rate_hz)
    if not bool(torch.isfinite(cycles_per_sample).all()):
        raise ValueError("link CFO divided by sample rate must be representable in float64")
    # Reduce before multiplying by sample indices to avoid overflow for large
    # finite CFO values while preserving their sampled phase progression.
    cycles_per_sample = torch.remainder(cycles_per_sample, 1.0)
    amplitude_gain = torch.sqrt(power_gain)
    waveform_gain = amplitude_gain.to(dtype=link_waveforms.real.dtype)
    if not bool(torch.isfinite(waveform_gain).all()) or bool(
        ((amplitude_gain > 0) & (waveform_gain == 0)).any()
    ):
        raise ValueError("link power gain is not representable in the waveform precision")

    contribution_elements = batch * rx_count * rx_ports * tx_count * capture_num_samples
    receive_elements = batch * rx_count * rx_ports * capture_num_samples
    phase_state_elements = (
        batch * (tx_count + rx_count) * capture_num_samples if phase_noise_enabled else 0
    )
    phase_work_elements = (
        batch * (max(tx_count, rx_count) + 2) * capture_num_samples if phase_noise_enabled else 0
    )
    iq_work_elements = 5 * batch * rx_ports * capture_num_samples if iq_active else 0
    filter_work_elements = 0
    if not filter_is_identity:
        interpolation_cost = (
            2 * _NETWORK_FRACTIONAL_DELAY_KERNEL_RADIUS + 1
            if bool((fractional_delay != 0).any())
            else 1
        )
        filter_work_elements = (
            batch
            * rx_count
            * tx_count
            * rx_ports
            * capture_num_samples
            * filter_taps.shape[2]
            * interpolation_cost
        )
    if (
        contribution_elements
        + receive_elements
        + phase_state_elements
        + phase_work_elements
        + iq_work_elements
        + filter_work_elements
        > max_output_elements
    ):
        raise ValueError(
            "waveform outputs, oscillator state, and impairment work exceed max_output_elements; "
            "reduce the capture, batch, or impairment roster"
        )

    tx_phase_noise = None
    rx_phase_noise = None
    if phase_noise_enabled:
        assert phase_noise_seed is not None
        phase_generator = torch.Generator(device=link_waveforms.device).manual_seed(
            phase_noise_seed
        )
        tx_phase_noise = _shared_phase_noise_processes(
            torch,
            batch_size=batch,
            node_count=tx_count,
            sample_count=capture_num_samples,
            linewidth_hz=tx_linewidth,
            sample_rate_hz=float(sample_rate_hz),
            generator=phase_generator,
            device=link_waveforms.device,
        )
        rx_phase_noise = _shared_phase_noise_processes(
            torch,
            batch_size=batch,
            node_count=rx_count,
            sample_count=capture_num_samples,
            linewidth_hz=rx_linewidth,
            sample_rate_hz=float(sample_rate_hz),
            generator=phase_generator,
            device=link_waveforms.device,
        )

    contributions = torch.zeros(
        (batch, rx_count, rx_ports, tx_count, capture_num_samples),
        dtype=link_waveforms.dtype,
        device=link_waveforms.device,
    )
    filtered_waveform_samples = _input_samples + filter_taps.shape[2] - 1
    for rx_index in range(rx_count):
        for tx_index in range(tx_count):
            offset = int(offsets[rx_index, tx_index].item())
            source_start = max(0, -offset)
            destination_start = max(0, offset)
            sample_count = min(
                filtered_waveform_samples - source_start,
                capture_num_samples - destination_start,
            )
            if sample_count > 0:
                relative_output_indices = torch.arange(
                    destination_start - offset,
                    destination_start - offset + sample_count,
                    dtype=torch.int64,
                    device=link_waveforms.device,
                )
                if filter_is_identity and fractional_delay[rx_index, tx_index] == 0:
                    aligned = link_waveforms[
                        :, rx_index, tx_index, :, source_start : source_start + sample_count
                    ]
                elif filter_is_identity:
                    aligned = _fractional_delay_capture_slice(
                        torch,
                        link_waveforms[:, rx_index, tx_index],
                        output_indices=relative_output_indices.to(link_waveforms.real.dtype),
                        delay_samples=float(fractional_delay[rx_index, tx_index].item()),
                        kernel_radius=_NETWORK_FRACTIONAL_DELAY_KERNEL_RADIUS,
                    )
                else:
                    aligned = torch.zeros(
                        (batch, rx_ports, sample_count),
                        dtype=link_waveforms.dtype,
                        device=link_waveforms.device,
                    )
                    for tap_index in range(filter_taps.shape[2]):
                        source_indices = relative_output_indices - tap_index
                        valid = (source_indices >= 0) & (source_indices < _input_samples)
                        if fractional_delay[rx_index, tx_index] == 0:
                            safe_indices = source_indices.clamp(0, _input_samples - 1)
                            filtered_samples = link_waveforms[:, rx_index, tx_index].index_select(
                                -1, safe_indices
                            )
                            filtered_samples = filtered_samples * valid.to(
                                link_waveforms.real.dtype
                            )
                        else:
                            filtered_samples = _fractional_delay_capture_slice(
                                torch,
                                link_waveforms[:, rx_index, tx_index],
                                output_indices=source_indices.to(link_waveforms.real.dtype),
                                delay_samples=float(fractional_delay[rx_index, tx_index].item()),
                                kernel_radius=_NETWORK_FRACTIONAL_DELAY_KERNEL_RADIUS,
                            )
                            filtered_samples = filtered_samples * valid.to(
                                link_waveforms.real.dtype
                            )
                        aligned += filtered_samples * filter_taps[rx_index, tx_index, tap_index]
                destination_indices = torch.arange(
                    destination_start,
                    destination_start + sample_count,
                    dtype=torch.float64,
                    device=link_waveforms.device,
                )
                cycles = torch.remainder(
                    cycles_per_sample[rx_index, tx_index] * destination_indices,
                    1.0,
                )
                base_phase = (2.0 * math.pi * cycles + initial_phase[rx_index, tx_index]).to(
                    dtype=link_waveforms.real.dtype
                )
                if phase_noise_enabled:
                    assert tx_phase_noise is not None and rx_phase_noise is not None
                    node_phase = (
                        tx_phase_noise[
                            :, tx_index, destination_start : destination_start + sample_count
                        ]
                        - rx_phase_noise[
                            :, rx_index, destination_start : destination_start + sample_count
                        ]
                    ).to(dtype=link_waveforms.real.dtype)
                    phase = base_phase[None, :] + node_phase
                else:
                    phase = base_phase
                oscillator = torch.polar(torch.ones_like(phase), phase)
                if phase_noise_enabled:
                    oscillator = oscillator[:, None, :]
                link_output = aligned * waveform_gain[rx_index, tx_index] * oscillator
                if iq_active:
                    i_component = (
                        iq_real_scale[rx_index] * link_output.real
                        - iq_q_scale[rx_index] * iq_sin[rx_index] * link_output.imag
                    )
                    q_component = iq_q_scale[rx_index] * iq_cos[rx_index] * link_output.imag
                    link_output = torch.complex(i_component, q_component)
                contributions[
                    :, rx_index, :, tx_index, destination_start : destination_start + sample_count
                ] = link_output
    received = contributions.sum(dim=3)
    if not bool(torch.isfinite(contributions).all()) or not bool(torch.isfinite(received).all()):
        raise ValueError("composed receiver waveform is not finite")
    return NetworkWaveformComposition(
        source_contributions=contributions,
        received_waveform=received,
        link_start_samples=offsets,
        link_power_gain_linear=power_gain,
        sample_rate_hz=float(sample_rate_hz),
        link_cfo_hz=cfo,
        link_initial_phase_rad=initial_phase,
        link_fractional_delay_samples=fractional_delay,
        fractional_delay_kernel_radius=_NETWORK_FRACTIONAL_DELAY_KERNEL_RADIUS,
        tx_phase_noise_linewidth_hz=tx_linewidth,
        rx_phase_noise_linewidth_hz=rx_linewidth,
        phase_noise_seed=phase_noise_seed,
        rx_iq_gain_imbalance_db=rx_iq_gain,
        rx_iq_phase_imbalance_rad=rx_iq_phase,
        link_filter_taps=filter_taps,
    )


def add_independent_receiver_waveform_awgn(
    composition: NetworkWaveformComposition,
    noise_variance_w_per_sample: Sequence[float] | Any,
    *,
    seed: int,
) -> NoisyNetworkWaveform:
    """Add independent receiver-node AWGN to an aligned waveform composition."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to add receiver waveform noise") from exc

    received = getattr(composition, "received_waveform", None)
    contributions = getattr(composition, "source_contributions", None)
    offsets = getattr(composition, "link_start_samples", None)
    power_gain = getattr(composition, "link_power_gain_linear", None)
    if not torch.is_tensor(received) or not torch.is_complex(received) or received.ndim != 4:
        raise TypeError("composition must contain complex [batch, rx, rx_port, sample] waveforms")
    if not torch.is_tensor(contributions) or not torch.is_complex(contributions):
        raise TypeError("composition must contain complex per-link contributions")
    if contributions.ndim != 5 or tuple(contributions.shape[:2]) != tuple(received.shape[:2]):
        raise ValueError("composition link contributions have invalid axes")
    if any(size <= 0 for size in received.shape) or any(size <= 0 for size in contributions.shape):
        raise ValueError("composition waveform axes must be non-empty")
    if tuple(contributions.shape[2:3] + contributions.shape[4:5]) != tuple(
        received.shape[2:3] + received.shape[3:4]
    ):
        raise ValueError("composition link contributions do not match the receiver waveform")
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if (
        not torch.is_tensor(offsets)
        or offsets.layout != torch.strided
        or offsets.dtype not in integer_dtypes
        or offsets.shape != (received.shape[1], contributions.shape[3])
    ):
        raise ValueError("composition must contain one sample offset per receiver/transmitter")
    if (
        not torch.is_tensor(power_gain)
        or tuple(power_gain.shape) != (received.shape[1], contributions.shape[3])
        or power_gain.device != received.device
        or not power_gain.is_floating_point()
        or not bool(torch.isfinite(power_gain).all())
        or bool((power_gain < 0).any())
    ):
        raise ValueError("composition must contain finite nonnegative RX/TX power gains")
    if (
        isinstance(composition.sample_rate_hz, bool)
        or not isinstance(composition.sample_rate_hz, (int, float))
        or not math.isfinite(composition.sample_rate_hz)
        or composition.sample_rate_hz <= 0
    ):
        raise ValueError("composition sample rate must be finite and positive")
    if received.device != contributions.device or received.dtype != contributions.dtype:
        raise ValueError("received waveforms and link contributions must share dtype and device")
    if not torch.equal(contributions.sum(dim=3), received):
        raise ValueError("received_waveform must equal the sum of source_contributions")
    if not bool(torch.isfinite(received).all()) or not bool(torch.isfinite(contributions).all()):
        raise ValueError("composition waveforms must be finite")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rx_count = received.shape[1]
    if seed + rx_count - 1 > 2**63 - 1:
        raise ValueError("seed and receiver-derived seeds must fit in [0, 2**63 - 1]")
    variance = _validated_nonnegative_vector(
        torch,
        noise_variance_w_per_sample,
        size=rx_count,
        device=received.device,
        name="noise_variance_w_per_sample",
    )
    from .noise import proper_complex_awgn

    noise_parts = []
    for rx_index in range(rx_count):
        generator = torch.Generator(device=received.device).manual_seed(seed + rx_index)
        noise_parts.append(
            proper_complex_awgn(
                received[:, rx_index],
                variance[rx_index],
                generator=generator,
            )
        )
    noise = torch.stack(noise_parts, dim=1)
    noisy = received + noise
    if not bool(torch.isfinite(noise).all()) or not bool(torch.isfinite(noisy).all()):
        raise ValueError("receiver waveform plus noise is not finite")
    return NoisyNetworkWaveform(
        received_waveform=noisy,
        noise_waveform=noise,
        noise_variance_w_per_sample=variance,
        seed=seed,
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

    contributions, _received = _validate_composition(torch, composition)
    rx_count, tx_count = contributions.shape[1], contributions.shape[3]
    if len(serving_tx_indices) != rx_count:
        raise ValueError("serving_tx_indices must provide one transmitter index per receiver")
    serving = tuple(serving_tx_indices)
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < tx_count
        for index in serving
    ):
        raise ValueError("each serving transmitter index must address an existing transmitter")
    noise = _validated_nonnegative_vector(
        torch,
        noise_variance_w_per_re,
        size=rx_count,
        device=contributions.device,
        name="noise_variance_w_per_re",
    )

    desired_powers = []
    interference_powers = []
    for rx_index, serving_index in enumerate(serving):
        per_tx = contributions[:, rx_index]
        desired = per_tx[:, :, serving_index]
        interferer_indices = [index for index in range(tx_count) if index != serving_index]
        if interferer_indices:
            interference = per_tx[:, :, interferer_indices].sum(dim=2)
            interference_power = _mean_complex_power(torch, interference)
        else:
            interference_power = torch.zeros((), dtype=torch.float64, device=contributions.device)
        desired_powers.append(_mean_complex_power(torch, desired))
        interference_powers.append(interference_power)
    desired_power = torch.stack(desired_powers)
    interference_power = torch.stack(interference_powers)
    if not bool(torch.isfinite(desired_power).all()) or not bool(
        torch.isfinite(interference_power).all()
    ):
        raise ValueError("measured network powers are not representable in float64")
    denominator = interference_power + noise
    valid = (desired_power > 0) | (denominator > 0)
    sinr = desired_power / denominator
    if not bool(torch.isfinite(sinr[valid]).all()):
        raise ValueError("network SINR is not representable in float64")
    sinr = torch.where(valid, sinr, torch.full_like(sinr, float("nan")))
    return NetworkSINR(desired_power, interference_power, noise, sinr, valid)


def network_resource_sinr(
    composition: NetworkGridComposition,
    serving_tx_indices: Sequence[int] | Any,
    noise_variance_w_per_re: Sequence[float] | Any,
) -> NetworkResourceSINR:
    """Measure receiver SINR for every OFDM symbol and subcarrier.

    ``serving_tx_indices`` may provide one transmitter index per receiver or
    an integer map with ``[rx, symbol, subcarrier]`` axes. Every non-serving
    transmitter is summed coherently on each resource element before power is
    measured, matching the aligned-grid signal equation. Independent batch
    draws and receive ports are averaged; symbols and subcarriers are not.
    Use this when a single receiver-wide mean would conceal interference
    notches, occupied-resource differences, or frequency-selective link loss.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to measure per-resource network SINR") from exc

    contributions, _received = _validate_composition(torch, composition)
    rx_count, tx_count = contributions.shape[1], contributions.shape[3]
    symbol_count, subcarrier_count = contributions.shape[-2:]
    try:
        serving_probe = torch.as_tensor(serving_tx_indices, device=contributions.device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("serving_tx_indices must contain integer transmitter indices") from error
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if serving_probe.layout != torch.strided:
        raise ValueError("serving_tx_indices must be a dense integer array")
    if (
        serving_probe.dtype == torch.bool
        or serving_probe.is_floating_point()
        or serving_probe.is_complex()
        or serving_probe.dtype not in integer_dtypes
    ):
        raise ValueError("serving_tx_indices must contain integer transmitter indices")
    if tuple(serving_probe.shape) == (rx_count,):
        serving_map = serving_probe.to(torch.int64)[:, None, None].expand(
            rx_count, symbol_count, subcarrier_count
        )
    elif tuple(serving_probe.shape) == (rx_count, symbol_count, subcarrier_count):
        serving_map = serving_probe.to(torch.int64)
    else:
        raise ValueError("serving_tx_indices must have shape [rx] or [rx, symbol, subcarrier]")
    if bool(((serving_map < 0) | (serving_map >= tx_count)).any()):
        raise ValueError("each serving transmitter index must address an existing transmitter")
    noise = _validated_nonnegative_vector(
        torch,
        noise_variance_w_per_re,
        size=rx_count,
        device=contributions.device,
        name="noise_variance_w_per_re",
    )

    desired_rows = []
    interference_rows = []
    tx_indices = torch.arange(tx_count, device=contributions.device).reshape(1, 1, tx_count, 1, 1)
    for rx_index in range(rx_count):
        per_tx = contributions[:, rx_index]
        serving = serving_map[rx_index]
        desired = torch.gather(
            per_tx,
            dim=2,
            index=serving[None, None, None, :, :].expand(
                per_tx.shape[0], per_tx.shape[1], 1, symbol_count, subcarrier_count
            ),
        ).squeeze(2)
        interferer_mask = tx_indices != serving[None, None, None, :, :]
        interference = per_tx.masked_fill(~interferer_mask, 0).sum(dim=2)
        interference_power = (
            interference.real.to(torch.float64).square()
            + interference.imag.to(torch.float64).square()
        ).mean(dim=(0, 1))
        desired_power = (
            desired.real.to(torch.float64).square() + desired.imag.to(torch.float64).square()
        ).mean(dim=(0, 1))
        desired_rows.append(desired_power)
        interference_rows.append(interference_power)

    desired_power = torch.stack(desired_rows)
    interference_power = torch.stack(interference_rows)
    noise_power = noise[:, None, None].expand_as(desired_power)
    if not bool(torch.isfinite(desired_power).all()) or not bool(
        torch.isfinite(interference_power).all()
    ):
        raise ValueError("measured per-resource network powers are not representable in float64")
    denominator = interference_power + noise_power
    valid = (desired_power > 0) | (denominator > 0)
    sinr = desired_power / denominator
    if not bool(torch.isfinite(sinr[valid]).all()):
        raise ValueError("per-resource network SINR is not representable in float64")
    sinr = torch.where(valid, sinr, torch.full_like(sinr, float("nan")))
    return NetworkResourceSINR(desired_power, interference_power, noise_power, sinr, valid)


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

    from .noise import proper_complex_awgn

    _contributions, received = _validate_composition(torch, composition)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rx_count = received.shape[1]
    if seed + rx_count - 1 > 2**63 - 1:
        raise ValueError("seed and receiver-derived seeds must fit in [0, 2**63 - 1]")
    noise_variance = _validated_nonnegative_vector(
        torch,
        noise_variance_w_per_re,
        size=rx_count,
        device=received.device,
        name="noise_variance_w_per_re",
    )

    noise_parts = []
    for rx_index in range(rx_count):
        generator = torch.Generator(device=received.device).manual_seed(seed + rx_index)
        noise_parts.append(
            proper_complex_awgn(
                received[:, rx_index],
                noise_variance[rx_index],
                generator=generator,
            )
        )
    noise_grid = torch.stack(noise_parts, dim=1)
    return NoisyNetworkGrid(
        received_grid=received + noise_grid,
        noise_grid=noise_grid,
        noise_variance_w_per_re=noise_variance,
        seed=seed,
    )
