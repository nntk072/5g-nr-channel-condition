"""Bounded sample-domain network waveform handoff to the NR PUSCH receiver."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .network_phy import NetworkWaveformComposition, NoisyNetworkWaveform


@dataclass(frozen=True, slots=True)
class NetworkPUSCHDecode:
    """Decoded NR outcomes with exact inputs and receiver configuration bound."""

    received_grid: Any
    decoded_bits: Any
    crc_ok: Any
    noise_variance_w_per_re: Any
    report: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(torch: Any, value: Any) -> str:
    array = value.detach().contiguous().cpu()
    metadata = {
        "dtype": str(array.dtype),
        "shape": [int(size) for size in array.shape],
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raw = array.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _real_noise_profile(torch: Any, values: Any, *, rx_count: int, device: Any) -> Any:
    if isinstance(values, bool):
        raise ValueError("noise_variance_w_per_re must be real and non-Boolean")
    try:
        probe = torch.as_tensor(values, device=device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("noise_variance_w_per_re must be real numeric values") from error
    if probe.dtype == torch.bool:
        raise ValueError("noise_variance_w_per_re must be real and non-Boolean")
    if (
        probe.layout != torch.strided
        or probe.is_complex()
        or not (
            probe.is_floating_point()
            or probe.dtype
            in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        )
        or tuple(probe.shape) != (rx_count,)
    ):
        raise ValueError("noise_variance_w_per_re must be a real [rx] vector")
    profile = probe.to(dtype=torch.float64)
    if not bool(torch.isfinite(profile).all()) or bool((profile <= 0).any()):
        raise ValueError("noise_variance_w_per_re must be finite and positive")
    return profile


def _unitary_sionna_noise_conversion(
    torch: Any, sample_variance: Any, *, rx_count: int, fft_size: int, device: Any
) -> Any:
    """Map iid time-sample noise variance through Sionna 2.2's unitary FFT pair."""
    if (
        not torch.is_tensor(sample_variance)
        or sample_variance.layout != torch.strided
        or sample_variance.dtype == torch.bool
        or sample_variance.is_complex()
        or not sample_variance.is_floating_point()
        or tuple(sample_variance.shape) != (rx_count,)
        or isinstance(fft_size, bool)
        or not isinstance(fft_size, int)
        or fft_size < 1
    ):
        raise ValueError("sample-domain noise variance or FFT size violates the Sionna contract")
    variance = sample_variance.to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
        raise ValueError("sample-domain noise variance must be finite and nonnegative")
    # Sionna PHY 2.2 applies 1/sqrt(N) to both the FFT and IFFT. The OFDM
    # demodulator discards CP samples, applies that unitary FFT, multiplies by
    # a unit-magnitude phase correction, and reorders bins. Each operation
    # preserves the variance of iid proper-complex AWGN on the retained samples.
    return variance


def decode_noisy_network_pusch(
    composition: NetworkWaveformComposition,
    noisy: NoisyNetworkWaveform,
    receiver: Any,
    *,
    noise_variance_w_per_re: Any | None = None,
    max_tensor_elements: int = 16_000_000,
) -> NetworkPUSCHDecode:
    """Demodulate a composed network capture and decode it with NR PUSCH.

    For the pinned Sionna PHY 2.2 receiver, iid proper-complex sample AWGN maps
    to equal per-RE variance because its FFT/IFFT pair is unitary and CP removal,
    phase compensation, and bin reordering preserve variance. An explicit
    receiver-domain profile may be supplied for other declared conventions.
    The returned report is always ``OBSERVED_NOT_QUALIFIED`` and does not
    establish receiver qualification.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install PyTorch to decode network waveforms") from error

    if not isinstance(composition, NetworkWaveformComposition) or not isinstance(
        noisy, NoisyNetworkWaveform
    ):
        raise TypeError("composition and noisy must be network waveform result objects")
    if (
        isinstance(max_tensor_elements, bool)
        or not isinstance(max_tensor_elements, int)
        or max_tensor_elements < 1
    ):
        raise ValueError("max_tensor_elements must be a positive integer")
    received = composition.received_waveform
    contributions = composition.source_contributions
    noise = noisy.noise_waveform
    received_noisy = noisy.received_waveform
    if (
        not torch.is_tensor(received)
        or not torch.is_complex(received)
        or received.ndim != 4
        or not torch.is_tensor(contributions)
        or not torch.is_complex(contributions)
        or contributions.ndim != 5
        or tuple(contributions.shape[:3])
        != (received.shape[0], received.shape[1], received.shape[2])
        or contributions.shape[4] != received.shape[3]
        or contributions.shape[3] < 1
        or not torch.is_tensor(noise)
        or not torch.is_complex(noise)
        or not torch.is_tensor(received_noisy)
        or not torch.is_complex(received_noisy)
    ):
        raise ValueError("network waveforms must have [batch, rx, rx_port, sample] axes")
    if any(size <= 0 for size in received.shape) or tuple(noise.shape) != tuple(received.shape):
        raise ValueError("network waveform axes must be nonempty and noise must match the capture")
    if tuple(received_noisy.shape) != tuple(received.shape):
        raise ValueError("noisy waveform must match the source composition shape")
    tensors = (received, contributions, noise, received_noisy)
    if any(value.device != received.device or value.dtype != received.dtype for value in tensors):
        raise ValueError(
            "network source, noise, and received waveforms must share dtype and device"
        )
    if sum(value.numel() for value in tensors) > max_tensor_elements:
        raise ValueError("network waveform inputs exceed max_tensor_elements")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("network waveform inputs must be finite")
    if not torch.equal(contributions.sum(dim=3), received):
        raise ValueError("source contributions do not sum to the composed receive waveform")
    if not torch.equal(received + noise, received_noisy):
        raise ValueError("noisy receive waveform does not equal source waveform plus noise")
    if not torch.is_tensor(noisy.noise_variance_w_per_sample):
        raise ValueError("noisy waveform must retain its sample-domain variance vector")
    if tuple(noisy.noise_variance_w_per_sample.shape) != (received.shape[1],):
        raise ValueError("sample-domain noise variance must contain one value per receiver")
    if not torch.isfinite(noisy.noise_variance_w_per_sample).all() or bool(
        (noisy.noise_variance_w_per_sample < 0).any()
    ):
        raise ValueError("sample-domain noise variance must be finite and nonnegative")

    metadata = getattr(receiver, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("receiver must expose JSON-serializable metadata")
    try:
        metadata_bytes = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("receiver metadata must be finite JSON data") from error
    grid_metadata = metadata.get("resource_grid")
    if not isinstance(grid_metadata, dict):
        raise ValueError("receiver metadata must declare its resource grid")
    fft_size = grid_metadata.get("fft_size")
    symbol_count = grid_metadata.get("num_ofdm_symbols")
    cyclic_prefix_samples = grid_metadata.get("cyclic_prefix_length_samples")
    subcarrier_spacing_hz = grid_metadata.get("subcarrier_spacing_hz")
    if (
        isinstance(fft_size, bool)
        or not isinstance(fft_size, int)
        or fft_size <= 0
        or isinstance(symbol_count, bool)
        or not isinstance(symbol_count, int)
        or symbol_count <= 0
        or isinstance(cyclic_prefix_samples, bool)
        or not isinstance(cyclic_prefix_samples, int)
        or cyclic_prefix_samples < 0
        or isinstance(subcarrier_spacing_hz, bool)
        or not isinstance(subcarrier_spacing_hz, int)
        or subcarrier_spacing_hz <= 0
    ):
        raise ValueError("receiver resource-grid dimensions and spacing must be positive integers")
    expected_sample_rate = fft_size * subcarrier_spacing_hz
    if not math.isclose(composition.sample_rate_hz, expected_sample_rate, rel_tol=1e-12):
        raise ValueError("network waveform sample rate does not match the receiver FFT grid")
    device_name = getattr(receiver, "device", None)
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("receiver must declare an execution device")
    device = torch.device(device_name)
    batch, rx_count, rx_ports, sample_count = received.shape
    expected_sample_count = symbol_count * (fft_size + cyclic_prefix_samples)
    if sample_count != expected_sample_count:
        raise ValueError(
            "network waveform sample count must exactly match the configured full-slot OFDM grid"
        )

    # This is a narrow audit of the explicitly supplied causal FIRs. It does
    # not claim the total propagation delay spread or receiver timing offset:
    # link arrival offsets, fractional-delay interpolation, and acquisition
    # remain separate inputs to an end-to-end CP/ISI experiment.
    filter_taps = composition.link_filter_taps
    if filter_taps is None:
        fir_support = torch.ones(
            (rx_count, contributions.shape[3], 1), dtype=received.dtype, device=received.device
        )
    else:
        if (
            not torch.is_tensor(filter_taps)
            or not torch.is_complex(filter_taps)
            or filter_taps.ndim != 3
            or tuple(filter_taps.shape[:2]) != (rx_count, contributions.shape[3])
            or filter_taps.shape[2] < 1
            or not bool(torch.isfinite(filter_taps).all())
        ):
            raise ValueError("link FIR taps must be finite complex [rx, tx, tap] values")
        fir_support = filter_taps
    fir_support = fir_support.to(device=received.device)
    fir_support_by_link: list[list[dict[str, int]]] = []
    maximum_fir_memory_samples = 0
    for rx_index in range(rx_count):
        receiver_support: list[dict[str, int]] = []
        for tx_index in range(contributions.shape[3]):
            active_taps = torch.nonzero(fir_support[rx_index, tx_index] != 0).reshape(-1)
            if active_taps.numel() == 0:
                first_tap = last_tap = -1
                memory_samples = 0
            else:
                first_tap = int(active_taps[0].item())
                last_tap = int(active_taps[-1].item())
                memory_samples = last_tap - first_tap
            maximum_fir_memory_samples = max(maximum_fir_memory_samples, memory_samples)
            receiver_support.append(
                {
                    "first_nonzero_tap": first_tap,
                    "last_nonzero_tap": last_tap,
                    "memory_samples": memory_samples,
                }
            )
        fir_support_by_link.append(receiver_support)
    estimated_grid_elements = batch * rx_count * rx_ports * symbol_count * fft_size
    if estimated_grid_elements + sum(value.numel() for value in tensors) > max_tensor_elements:
        raise ValueError("network waveform and demodulated grid exceed max_tensor_elements")
    if noise_variance_w_per_re is None:
        if metadata.get("sionna_version") != "2.2.0":
            raise ValueError(
                "implicit sample-to-RE noise conversion requires the pinned Sionna PHY 2.2.0 receiver"
            )
        n0 = _unitary_sionna_noise_conversion(
            torch,
            noisy.noise_variance_w_per_sample,
            rx_count=rx_count,
            fft_size=fft_size,
            device=device,
        )
        noise_conversion = "sionna_phy_2_2_0_unitary_dft_preserves_variance_v1"
    else:
        n0 = _real_noise_profile(
            torch,
            noise_variance_w_per_re,
            rx_count=rx_count,
            device=device,
        )
        noise_conversion = "caller_supplied_per_re_variance_v1"

    waveform_on_device = received_noisy.to(device=device, dtype=torch.complex64)
    received_grid = receiver.demodulate_slot_waveform(waveform_on_device)
    if (
        not torch.is_tensor(received_grid)
        or not torch.is_complex(received_grid)
        or tuple(received_grid.shape) != (batch, rx_count, rx_ports, symbol_count, fft_size)
        or not bool(torch.isfinite(received_grid).all())
    ):
        raise ValueError("receiver demodulator returned an invalid NR resource grid")
    noise_for_receiver = n0.to(dtype=received_grid.real.dtype).reshape(1, rx_count, 1)
    decoded_bits, crc_ok = receiver.decode_pusch(received_grid, noise_for_receiver)
    if (
        not torch.is_tensor(decoded_bits)
        or decoded_bits.numel() < 1
        or not torch.is_tensor(crc_ok)
        or crc_ok.dtype != torch.bool
        or crc_ok.numel() < 1
        or decoded_bits.shape[0] != batch
        or crc_ok.shape[0] != batch
        or not bool(torch.isfinite(decoded_bits).all())
    ):
        raise ValueError("PUSCH receiver must return nonempty batch-aligned bits and Boolean CRCs")
    if decoded_bits.numel() + crc_ok.numel() > max_tensor_elements:
        raise ValueError("PUSCH receiver outputs exceed max_tensor_elements")

    report: dict[str, Any] = {
        "schema": "umcd.network_pusch_receiver_decode.v3",
        "status": "OBSERVED_NOT_QUALIFIED",
        "implementation_sha256": hashlib.sha256(
            inspect.getsource(decode_noisy_network_pusch).encode("utf-8")
        ).hexdigest(),
        "execution_device": str(received_grid.device),
        "receiver_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "sample_rate_hz": float(composition.sample_rate_hz),
        "receiver_noise_convention": "watts_per_resource_element",
        "noise_variance_conversion": noise_conversion,
        "declared_fir_cp_diagnostic": {
            "method": "declared_causal_fir_nonzero_support_v1",
            "interpretation": "DECLARED_FILTER_SUPPORT_ONLY_NOT_TOTAL_CHANNEL_MEMORY_OR_QUALIFICATION",
            "cyclic_prefix_samples": int(cyclic_prefix_samples),
            "per_receiver_link_support": fir_support_by_link,
            "maximum_fir_memory_samples": maximum_fir_memory_samples,
            "exceeds_cyclic_prefix": maximum_fir_memory_samples > cyclic_prefix_samples,
            "filter_taps_sha256": _tensor_sha256(torch, fir_support),
        },
        "limitations": [
            "synchronization and acquisition are caller-controlled",
            "declared FIR support excludes link arrival offsets and fractional-delay interpolation",
            "synthetic decoder outcomes are not receiver qualification",
        ],
        "input_sha256": {
            "source_contributions": _tensor_sha256(torch, contributions),
            "clean_received_waveform": _tensor_sha256(torch, received),
            "noise_waveform": _tensor_sha256(torch, noise),
            "noisy_received_waveform": _tensor_sha256(torch, received_noisy),
            "sample_noise_variance": _tensor_sha256(torch, noisy.noise_variance_w_per_sample),
            "receiver_noise_variance_per_re": _tensor_sha256(torch, n0),
        },
        "output_sha256": {
            "received_grid": _tensor_sha256(torch, received_grid),
            "decoded_bits": _tensor_sha256(torch, decoded_bits),
            "crc_ok": _tensor_sha256(torch, crc_ok),
        },
        "shape": {
            "batch": int(batch),
            "receivers": int(rx_count),
            "receive_ports": int(rx_ports),
            "samples": int(sample_count),
            "ofdm_symbols": int(symbol_count),
            "fft_size": int(fft_size),
            "transport_blocks": int(crc_ok.numel()),
            "crc_failures": int((~crc_ok).sum().item()),
        },
    }
    schema_path = Path(__file__).resolve().parents[2] / (
        "datasets/schema/network-pusch-receiver-decode.v3.schema.json"
    )
    schema_bytes = schema_path.read_bytes()
    schema = json.loads(schema_bytes.decode("utf-8"))
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(schema)
    report["schema_sha256"] = hashlib.sha256(schema_bytes).hexdigest()
    report["report_sha256"] = _canonical_sha256(report)
    validation_errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if validation_errors:
        raise ValueError(
            "network PUSCH decode report violates schema: "
            + "; ".join(error.message for error in validation_errors[:10])
        )
    return NetworkPUSCHDecode(
        received_grid=received_grid,
        decoded_bits=decoded_bits,
        crc_ok=crc_ok,
        noise_variance_w_per_re=n0,
        report=report,
    )
