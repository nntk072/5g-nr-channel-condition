"""Controlled Sionna TDL → NR PUSCH frequency-domain link generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from datasets.nr import NRChannelRecord, NRObservation, TargetDefinition
from receivers.nr import NRChannelEstimate

from .random import call_with_sionna_seed


def seeded_complex_awgn(
    shape: tuple[int, ...],
    *,
    device: str,
    noise_variance: float,
    real_seed: int,
    imag_seed: int,
) -> Any:
    """Generate complex AWGN from independent component streams.

    Separate real and imaginary generators preserve identical sample prefixes
    when matched links have different waveform lengths.
    """
    import torch

    if not shape or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape
    ):
        raise ValueError("shape must contain positive integer dimensions")
    if isinstance(noise_variance, bool) or not isinstance(noise_variance, Real):
        raise ValueError("noise_variance must be a real scalar")
    noise_variance = float(noise_variance)
    if not math.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    if noise_variance > float(np.finfo(np.float32).max):
        raise ValueError("noise_variance must be representable as finite float32")
    if 0 < noise_variance < float(np.nextafter(np.float32(0), np.float32(1))):
        raise ValueError("positive noise_variance must be representable as nonzero float32")
    for name, seed in (("real_seed", real_seed), ("imag_seed", imag_seed)):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    stream_count = math.prod(shape[:-1]) if len(shape) > 1 else 1

    def normal_samples(seed: int) -> np.ndarray:
        samples = np.empty(shape, dtype=np.float32)
        stream_samples = samples.reshape(stream_count, shape[-1])
        for stream_index, row in enumerate(stream_samples):
            sequence = np.random.SeedSequence([seed, stream_index])
            generator = np.random.Generator(np.random.Philox(sequence))
            row[:] = generator.standard_normal(shape[-1], dtype=np.float32)
        return samples

    scale = math.sqrt(noise_variance / 2.0)
    real = torch.as_tensor(normal_samples(real_seed), device=device)
    imag = torch.as_tensor(normal_samples(imag_seed), device=device)
    return torch.complex(real, imag) * scale


@dataclass(frozen=True, slots=True)
class SionnaTDLPUSCHBatch:
    """Waveform, receiver observation, and private channel-truth target."""

    transmit_grid: Any
    transmit_bits: Any
    received_grid: Any
    channel_truth: Any
    noise_variance: Any
    metadata: dict[str, Any]


class SionnaPUSCHLinkDecoder:
    """Sionna-backed CRC decoder kept behind the native integration boundary."""

    def __init__(self, receiver: Any) -> None:
        try:
            from sionna.phy.nr import PUSCHReceiver
        except ImportError as exc:
            raise RuntimeError("Install the optional Sionna 2.2 PHY dependencies") from exc
        self._decoder = PUSCHReceiver(
            receiver._transmitter,
            channel_estimator=receiver._estimator,
            return_tb_crc_status=True,
            input_domain="freq",
            device=receiver.device,
        )

    def __call__(self, received_grid: Any, noise_variance: Any) -> tuple[Any, Any]:
        """Return decoded bits and transport-block CRC status."""
        return self._decoder(received_grid, noise_variance)


class SionnaTDLTimePUSCHLink:
    """Sionna waveform-domain PUSCH link over a sampled TDL channel."""

    def __init__(
        self,
        receiver: Any,
        *,
        model: str,
        delay_spread_s: float,
        carrier_frequency_hz: float,
        speed_mps: float,
        num_rx_ant: int = 1,
        spec_version: str = "19.2",
        allow_cp_overrun: bool = False,
        channel_estimator: Any | None = None,
    ) -> None:
        try:
            import torch
            from sionna.phy.channel import TimeChannel, time_lag_discrete_time_channel
            from sionna.phy.channel.tr38901 import TDL
            from sionna.phy.nr import PUSCHReceiver, PUSCHTransmitter
        except ImportError as exc:
            raise RuntimeError("Install the optional Sionna 2.2 PHY dependencies") from exc

        if receiver.config.num_layers != 1 or receiver.config.num_antenna_ports != 1:
            raise ValueError("initial sample-domain link supports one layer and one transmit port")
        if num_rx_ant <= 0:
            raise ValueError("num_rx_ant must be positive")
        if not isinstance(allow_cp_overrun, bool):
            raise TypeError("allow_cp_overrun must be a boolean")
        self._torch = torch
        self._device = receiver.device
        config = receiver._transmitter._pusch_configs
        self.transmitter = PUSCHTransmitter(
            config,
            return_bits=True,
            output_domain="time",
            device=self._device,
        )
        grid = self.transmitter.resource_grid
        l_min, l_max = time_lag_discrete_time_channel(grid.bandwidth, delay_spread_s)
        min_cyclic_prefix = int(torch.as_tensor(grid.cyclic_prefix_length).min().item())
        cp_covers_channel_memory = l_max - l_min <= min_cyclic_prefix
        if not cp_covers_channel_memory and not allow_cp_overrun:
            raise ValueError(
                "sample-domain channel memory exceeds the shortest cyclic prefix; "
                "increase the PUSCH bandwidth/resource-block allocation or reduce delay spread; "
                "set allow_cp_overrun=True only for an explicit stress test"
            )
        self.l_min = l_min
        self.l_max = l_max
        estimator = receiver._estimator if channel_estimator is None else channel_estimator
        self.receiver = PUSCHReceiver(
            self.transmitter,
            channel_estimator=estimator,
            input_domain="time",
            l_min=l_min,
            return_tb_crc_status=True,
            device=self._device,
        )
        channel_model = TDL(
            model=model,
            delay_spread=delay_spread_s,
            carrier_frequency=carrier_frequency_hz,
            min_speed=speed_mps,
            max_speed=speed_mps,
            num_rx_ant=num_rx_ant,
            num_tx_ant=1,
            spec_version=spec_version,
            device=self._device,
        )
        self.channel = TimeChannel(
            channel_model,
            bandwidth=grid.bandwidth,
            num_time_samples=grid.num_time_samples,
            l_min=l_min,
            l_max=l_max,
            normalize_channel=True,
            return_channel=True,
            device=self._device,
        )
        self.metadata = {
            "channel_model": "Sionna TimeChannel with sampled waveform-domain TDL",
            "profile": f"TDL-{model.upper()}",
            "spec_version": spec_version,
            "delay_spread_s": delay_spread_s,
            "carrier_frequency_hz": carrier_frequency_hz,
            "speed_mps": speed_mps,
            "num_rx_ant": num_rx_ant,
            "sample_rate_hz": grid.bandwidth,
            "l_min": l_min,
            "l_max": l_max,
            "cyclic_prefix_min_samples": min_cyclic_prefix,
            "channel_memory_samples": l_max - l_min,
            "cp_covers_channel_memory": cp_covers_channel_memory,
            "allow_cp_overrun": allow_cp_overrun,
            "resource_grid": {
                "fft_size": grid.fft_size,
                "num_time_samples": grid.num_time_samples,
                "bandwidth_hz": grid.bandwidth,
            },
        }

    def transmit_channel(self, batch_size: int, *, seed: int) -> tuple[Any, Any, Any, Any]:
        """Return seeded waveform, bits, noiseless channel output, and channel truth."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if seed > 2**63 - 2:
            raise ValueError("seed must leave room for the channel seed")
        waveform, bits = call_with_sionna_seed(
            self._device, seed, lambda: self.transmitter(batch_size)
        )
        received, channel_truth = call_with_sionna_seed(
            self._device, seed + 1, lambda: self.channel(waveform)
        )
        return waveform, bits, received, channel_truth

    def decode(self, received_waveform: Any, noise_variance: Any) -> tuple[Any, Any]:
        """Return decoded transport-block bits and CRC status."""
        return self.receiver(received_waveform, noise_variance)


def tdl_pusch_records(
    receiver: Any,
    batch: SionnaTDLPUSCHBatch,
    *,
    sample_ids: tuple[str, ...],
) -> tuple[NRChannelRecord, ...]:
    """Split one simulated batch into receiver-only observations and CFR targets."""
    batch_size = int(batch.received_grid.shape[0])
    if len(sample_ids) != batch_size or len(set(sample_ids)) != batch_size:
        raise ValueError("sample_ids must uniquely identify every batch example")
    estimate = receiver.estimate(batch.received_grid, batch.noise_variance)
    torch = receiver._torch
    effective_bins = torch.as_tensor(
        receiver.resource_grid.effective_subcarrier_ind,
        dtype=torch.long,
        device=batch.channel_truth.device,
    )
    records: list[NRChannelRecord] = []
    for batch_index, sample_id in enumerate(sample_ids):
        h_hat = estimate.h_hat[batch_index : batch_index + 1]
        variance = (
            estimate.error_variance
            if estimate.error_variance.shape[0] == 1
            else estimate.error_variance[batch_index : batch_index + 1]
        )
        channel_estimate = NRChannelEstimate(
            h_hat=h_hat,
            error_variance=variance,
            pilot_mask=estimate.pilot_mask,
            metadata=estimate.metadata,
        )
        observation = NRObservation.from_receiver_output(
            sample_id=sample_id,
            received_grid=batch.received_grid[batch_index : batch_index + 1],
            channel_estimate=channel_estimate,
            effective_subcarrier_indices=receiver.resource_grid.effective_subcarrier_ind,
            receiver_config=receiver.config,
        )
        # The initial receiver contract is one transmit antenna/port. Convert
        # OFDMChannel's antenna axis into the estimator's one stream axis.
        channel_truth = batch.channel_truth[batch_index : batch_index + 1]
        channel_truth = channel_truth.squeeze(4).unsqueeze(4)
        channel_truth = torch.index_select(channel_truth, -1, effective_bins)
        records.append(
            NRChannelRecord(
                observation=observation,
                targets={"channel_cfr": channel_truth},
                target_masks={"channel_cfr": torch.ones_like(channel_truth, dtype=torch.bool)},
                target_definitions={
                    "channel_cfr": TargetDefinition.versioned(
                        semantic="perfect synchronized effective-subcarrier CFR",
                        units="complex channel gain (dimensionless)",
                        source="sionna_phy_ofdmchannel_return_channel",
                        reference=(
                            "OFDMChannel return_channel with the configured TDL profile, "
                            "carrier, sample rate, and effective FFT-bin mapping"
                        ),
                        aggregation_rule=(
                            "retain each effective subcarrier coefficient; no antenna, "
                            "symbol, or power averaging"
                        ),
                        definition_version="umcd.tdl-pusch.effective-cfr.v1",
                    )
                },
                provenance={
                    **batch.metadata,
                    "batch_index": batch_index,
                    "channel_truth_source": "Sionna OFDMChannel return_channel",
                    "target_semantics": "perfect synchronized effective-subcarrier CFR",
                },
            )
        )
    return tuple(records)


def generate_tdl_pusch_batch(
    receiver: Any,
    *,
    model: str,
    batch_size: int,
    seed: int,
    carrier_frequency_hz: float,
    delay_spread_s: float,
    noise_variance: float,
    num_rx_ant: int = 1,
    min_speed_mps: float = 0.0,
    max_speed_mps: float | None = None,
    spec_version: str = "19.2",
) -> SionnaTDLPUSCHBatch:
    """Create seeded PUSCH observations through Sionna's TDL and OFDM blocks.

    Uses an ideal frequency-domain OFDM link: synchronization is perfect and
    Sionna's ``OFDMChannel`` assumes no ICI/ISI (so the CP must cover the
    effective channel delay). This deliberately is not a timing/CFO-impaired
    waveform or a claim of full receiver conformance.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_rx_ant, bool) or not isinstance(num_rx_ant, int) or num_rx_ant <= 0:
        raise ValueError("num_rx_ant must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(noise_variance, bool) or not isinstance(noise_variance, Real):
        raise ValueError("noise_variance must be a real scalar")
    noise_variance = float(noise_variance)
    if not math.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    if noise_variance > float(np.finfo(np.float32).max):
        raise ValueError("noise_variance must be representable as finite float32")
    if 0 < noise_variance < float(np.nextafter(np.float32(0), np.float32(1))):
        raise ValueError("positive noise_variance must be representable as nonzero float32")
    if not math.isfinite(carrier_frequency_hz) or carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz must be finite and positive")
    if not math.isfinite(delay_spread_s) or delay_spread_s < 0:
        raise ValueError("delay_spread_s must be finite and nonnegative")
    if receiver.config.num_layers != 1 or receiver.config.num_antenna_ports != 1:
        raise ValueError("initial PUSCH link supports one layer and one transmit antenna port")

    try:
        import torch
        from sionna.phy.channel import OFDMChannel
        from sionna.phy.channel.tr38901 import TDL
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional Sionna 2.2 PHY dependencies to generate NR PUSCH batches"
        ) from exc

    device = receiver.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if seed > 2**63 - 3:
        raise ValueError("seed must leave room for derived channel and noise seeds")
    resource_grid = receiver.resource_grid
    tdl = TDL(
        model=model,
        delay_spread=delay_spread_s,
        carrier_frequency=carrier_frequency_hz,
        min_speed=min_speed_mps,
        max_speed=max_speed_mps,
        num_rx_ant=num_rx_ant,
        num_tx_ant=1,
        spec_version=spec_version,
        device=device,
    )
    ofdm_channel = OFDMChannel(
        tdl,
        resource_grid,
        normalize_channel=False,
        return_channel=True,
        device=device,
    )
    transmit_grid, transmit_bits = receiver.transmit(batch_size, seed=seed)
    received_clean, channel_truth = call_with_sionna_seed(
        device,
        seed + 1,
        lambda: ofdm_channel(transmit_grid),
    )
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(seed + 2)
    noise = torch.complex(
        torch.randn(received_clean.shape, device=device, generator=noise_generator),
        torch.randn(received_clean.shape, device=device, generator=noise_generator),
    ) * math.sqrt(noise_variance / 2.0)
    received_grid = received_clean + noise
    return SionnaTDLPUSCHBatch(
        transmit_grid=transmit_grid,
        transmit_bits=transmit_bits,
        received_grid=received_grid,
        channel_truth=channel_truth,
        noise_variance=torch.as_tensor(noise_variance, dtype=torch.float32, device=device),
        metadata={
            "profile_proxy": model,
            "geometric_los_label": None,
            "spec_version": spec_version,
            "carrier_frequency_hz": carrier_frequency_hz,
            "delay_spread_s": delay_spread_s,
            "noise_variance_per_complex_dimension": noise_variance,
            "num_rx_ant": num_rx_ant,
            "sampling_frequency_hz": 1.0 / resource_grid.ofdm_symbol_duration,
            "seed": seed,
            "device": device,
            "channel_model": "Sionna OFDMChannel; perfect synchronization; no ICI/ISI",
        },
    )
