"""PUSCH/DMRS observations over immutable standardized TDL/CDL channel draws."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any

import numpy as np
from datasets.nr import NRChannelRecord, TargetDefinition
from datasets.standardized import StandardizedChannelRun
from receivers.nr import NRPUSCHConfig

from .noise import proper_complex_awgn
from .nr_pusch import SionnaPUSCHLSReceiver
from .ofdm_impairments import (
    NRWaveformImpairments,
    apply_fractional_timing_offset,
    apply_integer_timing_offset,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_residual_cfo,
    generate_wiener_phase_noise_rad,
)
from .rt_dmrs import resample_centered_cfr


def _derive_impairment_seed(base_seed: int, sample_index: int, oscillator: int) -> int:
    words = np.random.SeedSequence([base_seed, sample_index, oscillator]).generate_state(
        2, dtype=np.uint32
    )
    return ((int(words[0]) << 32) | int(words[1])) & (2**63 - 1)


def _apply_phase_noise(
    waveform: Any,
    *,
    linewidth_hz: float | None,
    seed: int,
    sampling_frequency_hz: float,
    device: str,
) -> Any:
    if linewidth_hz is None:
        return waveform
    import torch

    if waveform.ndim < 3:
        raise ValueError("PUSCH waveform must keep node and antenna axes before time")
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(seed)
    phase = generate_wiener_phase_noise_rad(
        waveform.shape[-1],
        sampling_frequency_hz=sampling_frequency_hz,
        linewidth_hz=linewidth_hz,
        generator=noise_generator,
        # One oscillator trajectory per node and batch draw, shared by that
        # node's antenna elements. The final waveform axis is continuous time.
        leading_shape=(*waveform.shape[:-2], 1),
        device=device,
    )
    return apply_phase_noise(waveform, phase_noise_rad=phase)


def build_standardized_pusch_dmrs_records(
    run: StandardizedChannelRun,
    *,
    snr_db: float,
    seed: int,
    noise_seed: int | None = None,
    impairments: NRWaveformImpairments | None = None,
    impairment_seed: int | None = None,
    receiver_config: NRPUSCHConfig = NRPUSCHConfig(),
    time_index: int = 0,
    device: str = "cpu",
) -> tuple[NRChannelRecord, ...]:
    """Transmit PUSCH through static standardized CFR and estimate with DMRS.

    Perfect synchronized CFR target, noisy PUSCH resource-grid observation,
    LS estimate, and profile-proxy label remain separate fields in each typed
    training record. Without ``impairments``, this is an ideal frequency-domain
    link with no ICI/ISI, synchronization error, RF impairment, absolute power
    calibration, or geometric LOS label implied.

    With ``impairments``, transmitter impairments act on continuous sample-time
    waveforms, the static CFR acts on the resulting frequency bins, and
    receiver impairments act on the reconstructed continuous waveform. This
    retains impairment-induced inter-carrier mixing for a static circular
    channel model. Channel-delay-induced CP spillover and inter-symbol
    interference are not modeled.

    The current link supports one transmit antenna/port and one or more receive
    antennas (SISO/SIMO), with a channel held constant over one PUSCH allocation.

    ``seed`` controls the transmitted PUSCH waveform. ``noise_seed`` optionally
    controls receiver noise independently, allowing multiple observations of
    identical transmitted symbols and propagation truth. If omitted, the
    historical paired-seed behavior is retained.
    """
    if not isinstance(run, StandardizedChannelRun):
        raise TypeError("run must be a StandardizedChannelRun")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 2:
        raise ValueError("seed must be an integer in [0, 2**63 - 2]")
    if noise_seed is not None and (
        isinstance(noise_seed, bool)
        or not isinstance(noise_seed, int)
        or not 0 <= noise_seed <= 2**63 - 2
    ):
        raise ValueError("noise_seed must be an integer in [0, 2**63 - 2]")
    if impairments is not None and not isinstance(impairments, NRWaveformImpairments):
        raise TypeError("impairments must be an NRWaveformImpairments value or None")
    if (impairments is None) != (impairment_seed is None):
        raise ValueError("impairments and impairment_seed must be provided together")
    if impairment_seed is not None and (
        isinstance(impairment_seed, bool)
        or not isinstance(impairment_seed, int)
        or not 0 <= impairment_seed <= 2**63 - 2
    ):
        raise ValueError("impairment_seed must be an integer in [0, 2**63 - 2]")
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
    if impairments is not None:
        from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

        modulator = OFDMModulator(
            cyclic_prefix_length=grid.cyclic_prefix_length,
            device=device,
        )
        demodulator = OFDMDemodulator(
            fft_size=grid.fft_size,
            l_min=0,
            cyclic_prefix_length=grid.cyclic_prefix_length,
            device=device,
        )
        sampling_frequency_hz = grid.fft_size * receiver_config.subcarrier_spacing_khz * 1_000.0
    else:
        modulator = demodulator = None
        sampling_frequency_hz = 0.0
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
        if noise_seed is None:
            child_noise_seed = child_seed + 1
        else:
            noise_seed_words = np.random.SeedSequence(
                [noise_seed, record.sample_index]
            ).generate_state(2, dtype=np.uint32)
            child_noise_seed = (int(noise_seed_words[0]) << 32) | int(noise_seed_words[1])
        transmit_grid, _ = receiver.transmit(1, seed=child_seed)
        tx_phase_noise_seed = None
        rx_phase_noise_seed = None
        if impairments is not None:
            assert impairment_seed is not None
            tx_phase_noise_seed = _derive_impairment_seed(impairment_seed, record.sample_index, 0)
            rx_phase_noise_seed = _derive_impairment_seed(impairment_seed, record.sample_index, 1)
            tx_waveform = modulator(transmit_grid)
            tx_waveform = apply_iq_imbalance(
                tx_waveform,
                gain_imbalance_db=impairments.tx_iq_gain_imbalance_db,
                phase_imbalance_rad=impairments.tx_iq_phase_imbalance_rad,
            )
            tx_waveform = _apply_phase_noise(
                tx_waveform,
                linewidth_hz=impairments.tx_phase_noise_linewidth_hz,
                seed=tx_phase_noise_seed,
                sampling_frequency_hz=sampling_frequency_hz,
                device=device,
            )
            transmit_grid = demodulator(tx_waveform)
        h = torch.as_tensor(cfr[0], dtype=torch.complex64, device=device)
        # [batch, rx antenna, OFDM symbol, FFT bin], followed by the RX-node axis.
        received_clean = transmit_grid[:, 0, 0].unsqueeze(1) * h[None, :, None, :]
        received_clean = received_clean.unsqueeze(1)
        if received_clean.shape[2] != rx_antennas:
            raise RuntimeError("receive-antenna axis changed during PUSCH channel application")
        if impairments is not None:
            assert rx_phase_noise_seed is not None
            received_waveform = modulator(received_clean)
            received_waveform = apply_residual_cfo(
                received_waveform,
                cfo_hz=impairments.cfo_hz,
                sampling_frequency_hz=sampling_frequency_hz,
            )
            received_waveform = _apply_phase_noise(
                received_waveform,
                linewidth_hz=impairments.rx_phase_noise_linewidth_hz,
                seed=rx_phase_noise_seed,
                sampling_frequency_hz=sampling_frequency_hz,
                device=device,
            )
            received_waveform = apply_iq_imbalance(
                received_waveform,
                gain_imbalance_db=impairments.rx_iq_gain_imbalance_db,
                phase_imbalance_rad=impairments.rx_iq_phase_imbalance_rad,
            )
            received_waveform = apply_integer_timing_offset(
                received_waveform,
                offset_samples=impairments.integer_timing_offset_samples,
            )
            received_waveform = apply_fractional_timing_offset(
                received_waveform,
                offset_samples=impairments.fractional_timing_offset_samples,
            )
            received_clean = demodulator(received_waveform)
        signal_power = received_clean.abs().square().mean()
        if not bool(torch.isfinite(signal_power)) or float(signal_power) <= 0:
            raise ValueError(f"sample {record.sample_id} has zero or invalid PUSCH signal power")
        noise_variance = signal_power * (10.0 ** (-snr_db / 10.0))
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(child_noise_seed)
        noise = proper_complex_awgn(received_clean, noise_variance, generator=noise_generator)
        received_grid = received_clean + noise
        receiver_view_kind = (
            "standardized_pusch_impairment_view"
            if impairments is not None
            else "standardized_pusch_noise_view"
        )
        view_identity = {
            "run_id": run.run_id,
            "parent_sample_id": record.sample_id,
            "parent_sample_index": record.sample_index,
            "receiver_view_kind": receiver_view_kind,
            "receiver_config": asdict(receiver_config),
            "snr_db": snr_db,
            "transmit_seed": child_seed,
            "noise_seed": child_noise_seed,
            "time_index": time_index,
        }
        if impairments is not None:
            view_identity.update(
                {
                    "impairments": asdict(impairments),
                    "impairment_seed": impairment_seed,
                    "tx_phase_noise_seed": tx_phase_noise_seed,
                    "rx_phase_noise_seed": rx_phase_noise_seed,
                }
            )
        identity_bytes = json.dumps(
            view_identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        view_digest = hashlib.sha256(identity_bytes).hexdigest()
        view_id = f"{record.sample_id}:pusch:{view_digest[:20]}"
        observation = receiver.observe(view_id, received_grid, noise_variance)

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
                    "parent_subject_id": record.sample_id,
                    "receiver_view_kind": receiver_view_kind,
                    "receiver_view_identity_sha256": view_digest,
                    "channel_family": record.family.value,
                    "profile_model": record.profile_model,
                    "profile_los_component": record.profile_los_component,
                    "geometric_los_label": None,
                    "snr_db": snr_db,
                    "snr_definition": (
                        "mean received PUSCH resource-grid complex-sample power averaged over "
                        "batch, RX nodes, RX antennas, OFDM symbols, and FFT bins divided by "
                        "complex AWGN variance; no noise figure or beamforming"
                        if impairments is not None
                        else "mean received PUSCH resource-grid complex-sample power averaged over "
                        "batch, RX nodes, RX antennas, OFDM symbols, and FFT bins divided by "
                        "complex AWGN variance; no noise figure, beamforming, or RF impairment"
                    ),
                    "snr_reference": (
                        "post-channel and post-RF-impairment, pre-AWGN, pre-beamforming receive grid"
                        if impairments is not None
                        else "post-channel, pre-beamforming receive resource grid"
                    ),
                    "noise_figure_db": None,
                    "noise_model": (
                        "proper complex AWGN with independent resource-grid samples; "
                        "E[|n|^2] is noise_variance_per_complex_dimension"
                    ),
                    "pusch_transmit_seed": child_seed,
                    "noise_variance_per_complex_dimension": float(noise_variance),
                    "noise_variance_per_real_dimension": float(noise_variance) / 2.0,
                    "noise_seed": child_noise_seed,
                    "subcarrier_spacing_hz": receiver_config.subcarrier_spacing_khz * 1_000,
                    "carrier_frequency_hz": carrier_hz,
                    "time_index": time_index,
                    "channel_time_variation": "frozen CFR across one PUSCH allocation",
                    "receiver_link_model": (
                        "static diagonal frequency-domain CFR with sample-domain RF impairments; "
                        "impairment-induced ICI is retained, channel-induced CP spillover and "
                        "ISI are not modeled"
                        if impairments is not None
                        else "ideal frequency-domain channel; no ICI/ISI"
                    ),
                    "waveform_impairments": (
                        asdict(impairments) if impairments is not None else None
                    ),
                    "impairment_seed": impairment_seed,
                    "tx_phase_noise_seed": tx_phase_noise_seed,
                    "rx_phase_noise_seed": rx_phase_noise_seed,
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
