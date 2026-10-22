"""PUSCH/DMRS observations over immutable standardized TDL/CDL channel draws."""

from __future__ import annotations

import math

import numpy as np
from datasets.nr import NRChannelRecord, TargetDefinition
from datasets.standardized import StandardizedChannelRun
from receivers.nr import NRPUSCHConfig

from .nr_pusch import SionnaPUSCHLSReceiver
from .rt_dmrs import resample_centered_cfr


def build_standardized_pusch_dmrs_records(
    run: StandardizedChannelRun,
    *,
    snr_db: float,
    seed: int,
    receiver_config: NRPUSCHConfig = NRPUSCHConfig(),
    time_index: int = 0,
    device: str = "cpu",
) -> tuple[NRChannelRecord, ...]:
    """Transmit PUSCH through static standardized CFR and estimate with DMRS.

    The perfect synchronized CFR target, noisy PUSCH resource-grid observation,
    LS estimate, and profile-proxy label remain separate fields in each typed
    training record. This is an ideal frequency-domain link: no ICI/ISI,
    synchronization error, RF impairment, absolute power calibration, or
    geometric LOS label is implied.

    The current link supports one transmit antenna/port and one or more receive
    antennas (SISO/SIMO), with a channel held constant over one PUSCH allocation.
    """
    if not isinstance(run, StandardizedChannelRun):
        raise TypeError("run must be a StandardizedChannelRun")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 2:
        raise ValueError("seed must be an integer in [0, 2**63 - 2]")
    if isinstance(time_index, bool) or not isinstance(time_index, int) or time_index < 0:
        raise ValueError("time_index must be a nonnegative integer")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")
    for record in run.records:
        if time_index >= record.cfr.shape[-2]:
            raise ValueError(f"time_index is outside sample {record.sample_id}'s channel")
        if record.cfr.shape[2] != 1 or record.cfr.shape[3] != 1:
            raise ValueError("standardized PUSCH link currently requires one TX and one TX antenna")
        if record.cfr.shape[0] != 1:
            raise ValueError("standardized PUSCH link currently requires one RX node")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and Sionna PHY for PUSCH observations") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    receiver = SionnaPUSCHLSReceiver(receiver_config, device=device)
    grid = receiver.resource_grid
    if grid.num_effective_subcarriers != grid.fft_size:
        raise ValueError("standardized PUSCH adapter currently requires a full-band grid")
    spacing_values = {float(record.metadata["subcarrier_spacing_hz"]) for record in run.records}
    carrier_values = {float(record.metadata["carrier_frequency_hz"]) for record in run.records}
    if len(spacing_values) != 1 or len(carrier_values) != 1:
        raise ValueError("one standardized run must have a common carrier and subcarrier spacing")
    source_spacing_hz = spacing_values.pop()
    carrier_hz = carrier_values.pop()
    records: list[NRChannelRecord] = []

    for record in run.records:
        source_cfr = np.asarray(
            record.cfr[0, :, 0, 0, time_index, :][None, :, :], dtype=np.complex64
        )
        cfr = resample_centered_cfr(
            source_cfr,
            source_spacing_hz=source_spacing_hz,
            target_subcarriers=int(grid.fft_size),
            target_spacing_hz=receiver_config.subcarrier_spacing_khz * 1_000.0,
        )
        rx_antennas = cfr.shape[1]
        child_seed = int(
            np.random.SeedSequence([seed, record.sample_index]).generate_state(1, dtype=np.uint32)[
                0
            ]
        )
        transmit_grid, _ = receiver.transmit(1, seed=child_seed)
        h = torch.as_tensor(cfr[0], dtype=torch.complex64, device=device)
        # [batch, rx antenna, OFDM symbol, FFT bin], followed by the RX-node axis.
        received_clean = transmit_grid[:, 0, 0].unsqueeze(1) * h[None, :, None, :]
        received_clean = received_clean.unsqueeze(1)
        if received_clean.shape[2] != rx_antennas:
            raise RuntimeError("receive-antenna axis changed during PUSCH channel application")
        signal_power = received_clean.abs().square().mean()
        if not bool(torch.isfinite(signal_power)) or float(signal_power) <= 0:
            raise ValueError(f"sample {record.sample_id} has zero or invalid PUSCH signal power")
        noise_variance = signal_power * (10.0 ** (-snr_db / 10.0))
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(child_seed + 1)
        noise = torch.complex(
            torch.randn(
                received_clean.shape,
                dtype=torch.float32,
                device=device,
                generator=noise_generator,
            ),
            torch.randn(
                received_clean.shape,
                dtype=torch.float32,
                device=device,
                generator=noise_generator,
            ),
        ) * torch.sqrt(noise_variance / 2.0)
        received_grid = received_clean + noise
        observation = receiver.observe(record.sample_id, received_grid, noise_variance)

        effective_bins = torch.as_tensor(
            grid.effective_subcarrier_ind, dtype=torch.long, device=device
        )
        target_cfr = h.reshape(1, 1, rx_antennas, 1, 1, 1, int(grid.fft_size))
        target_cfr = torch.index_select(target_cfr, -1, effective_bins)
        target_cfr = target_cfr.expand(
            1,
            1,
            rx_antennas,
            1,
            1,
            int(grid.num_ofdm_symbols),
            int(grid.num_effective_subcarriers),
        )
        profile_target = np.asarray([int(record.profile_los_component)], dtype=np.uint8)
        profile_definition = TargetDefinition(
            semantic="standardized_profile_los_category_proxy",
            units="binary_class",
            source="Sionna 3GPP TR 38.901 profile model",
            reference="A/B/C map to 0 and D/E map to 1; not geometric LOS truth",
        )
        channel_definition = TargetDefinition(
            semantic="perfect_synchronized_effective_subcarrier_cfr",
            units="complex channel gain (dimensionless)",
            source="standardized Sionna TDL/CDL CFR adapter",
            reference="profile-relative excess-delay CFR; static over this PUSCH allocation",
        )
        records.append(
            NRChannelRecord(
                observation=observation,
                targets={
                    "channel_cfr": target_cfr,
                    "profile_los_proxy": profile_target,
                },
                target_masks={
                    "channel_cfr": torch.ones_like(target_cfr, dtype=torch.bool),
                    "profile_los_proxy": np.ones(profile_target.shape, dtype=bool),
                },
                provenance={
                    "observation_source": "Sionna PUSCH transmitter through standardized CFR",
                    "channel_run_id": run.run_id,
                    "channel_sample_id": record.sample_id,
                    "channel_family": record.family.value,
                    "profile_model": record.profile_model,
                    "profile_los_component": record.profile_los_component,
                    "geometric_los_label": None,
                    "snr_db": snr_db,
                    "pusch_transmit_seed": child_seed,
                    "noise_variance_per_complex_dimension": float(noise_variance),
                    "noise_seed": child_seed + 1,
                    "subcarrier_spacing_hz": receiver_config.subcarrier_spacing_khz * 1_000,
                    "carrier_frequency_hz": carrier_hz,
                    "time_index": time_index,
                    "channel_time_variation": "frozen CFR across one PUSCH allocation",
                    "receiver_link_model": "ideal frequency-domain channel; no ICI/ISI",
                    "receiver_metadata": receiver.metadata,
                    "profile_parameter_table_sha256": record.metadata.get("parameter_table_sha256"),
                    "target_semantics": {
                        "channel_cfr": channel_definition.semantic,
                        "profile_los_proxy": profile_definition.semantic,
                    },
                },
                target_definitions={
                    "channel_cfr": channel_definition,
                    "profile_los_proxy": profile_definition,
                },
            )
        )
    return tuple(records)
