"""PUSCH/DMRS observations over immutable standardized TDL/CDL channel draws."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

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


def _apply_sampled_time_channel(
    transmit_waveform: Any,
    record: Any,
    *,
    time_index: int,
    sampling_frequency_hz: float,
    resource_grid: Any,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Apply one frozen standardized CIR as a sinc-discretized time channel."""
    import torch
    from sionna.phy.channel import (
        ApplyTimeChannel,
        cir_to_time_channel,
        time_lag_discrete_time_channel,
        time_to_ofdm_channel,
    )

    coefficients = torch.as_tensor(
        np.array(record.h_path[..., time_index], copy=True),
        dtype=torch.complex64,
        device=device,
    )
    delays = torch.as_tensor(
        np.array(record.tau_s, copy=True)[None, ...], dtype=torch.float32, device=device
    )
    maximum_delay_s = float(delays.max().item())
    l_min, l_max = time_lag_discrete_time_channel(
        sampling_frequency_hz, maximum_delay_spread=maximum_delay_s
    )
    l_tot = l_max - l_min + 1
    num_time_samples = int(transmit_waveform.shape[-1])
    channel_time_samples = num_time_samples + l_tot - 1
    coefficients = coefficients[None, ..., None].expand(
        1, *coefficients.shape, channel_time_samples
    )
    h_time = cir_to_time_channel(
        sampling_frequency_hz,
        coefficients,
        delays,
        l_min=l_min,
        l_max=l_max,
        normalize=False,
    )
    received_waveform = ApplyTimeChannel(
        num_time_samples=num_time_samples,
        l_tot=l_tot,
        device=device,
    )(transmit_waveform, h_time)
    channel_cfr = time_to_ofdm_channel(h_time, resource_grid, l_min)[0, 0, :, 0, 0]
    cyclic_prefix_length = int(resource_grid.cyclic_prefix_length)
    return (
        received_waveform,
        channel_cfr,
        {
            "sampling_frequency_hz": sampling_frequency_hz,
            "maximum_path_delay_s": maximum_delay_s,
            "first_discrete_time_lag": l_min,
            "last_discrete_time_lag": l_max,
            "discrete_time_channel_tap_count": l_tot,
            "cyclic_prefix_length_samples": cyclic_prefix_length,
            "tap_span_fits_cyclic_prefix": (l_max - l_min) <= cyclic_prefix_length,
            "path_delay_reference": "standardized profile-relative excess delay",
            "fractional_delay_interpolation": "Sionna cir_to_time_channel sinc pulse shaping",
        },
    )


def _apply_receiver_impairments(
    waveform: Any,
    impairments: NRWaveformImpairments,
    *,
    phase_noise_seed: int,
    sampling_frequency_hz: float,
    device: str,
) -> Any:
    """Apply one declared receiver impairment chain on the final sample axis."""
    waveform = apply_residual_cfo(
        waveform,
        cfo_hz=impairments.cfo_hz,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    waveform = _apply_phase_noise(
        waveform,
        linewidth_hz=impairments.rx_phase_noise_linewidth_hz,
        seed=phase_noise_seed,
        sampling_frequency_hz=sampling_frequency_hz,
        device=device,
    )
    waveform = apply_iq_imbalance(
        waveform,
        gain_imbalance_db=impairments.rx_iq_gain_imbalance_db,
        phase_imbalance_rad=impairments.rx_iq_phase_imbalance_rad,
    )
    waveform = apply_integer_timing_offset(
        waveform,
        offset_samples=impairments.integer_timing_offset_samples,
    )
    return apply_fractional_timing_offset(
        waveform,
        offset_samples=impairments.fractional_timing_offset_samples,
    )


def build_standardized_pusch_dmrs_records(
    run: StandardizedChannelRun,
    *,
    snr_db: float,
    seed: int,
    noise_seed: int | None = None,
    impairments: NRWaveformImpairments | None = None,
    impairment_seed: int | None = None,
    channel_mode: Literal["time_domain", "ideal_cfr"] = "time_domain",
    receiver_config: NRPUSCHConfig = NRPUSCHConfig(),
    time_index: int = 0,
    device: str = "cpu",
) -> tuple[NRChannelRecord, ...]:
    """Transmit PUSCH through a frozen standardized channel and estimate with DMRS.

    Perfect synchronized CFR target, noisy PUSCH resource-grid observation,
    LS estimate, and profile-proxy label remain separate fields in each typed
    training record. By default, the standardized path coefficients and delays
    are sinc-discretized and applied to the sample-domain waveform, so channel
    delay can create CP spillover/ISI. ``channel_mode="ideal_cfr"`` selects the
    explicit frequency-domain diagnostic control, which has no channel-induced
    ICI/ISI.

    Transmitter impairments act on continuous sample-time waveforms before the
    selected propagation channel, and receiver impairments act after it. The
    ideal-CFR control retains impairment-induced inter-carrier mixing but not
    channel-induced CP spillover.

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
    if channel_mode not in {"time_domain", "ideal_cfr"}:
        raise ValueError("channel_mode must be 'time_domain' or 'ideal_cfr'")
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
    from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

    modulator = OFDMModulator(cyclic_prefix_length=grid.cyclic_prefix_length, device=device)
    sampling_frequency_hz = grid.fft_size * receiver_config.subcarrier_spacing_khz * 1_000.0
    receiver_l_min = -6 if channel_mode == "time_domain" else 0
    demodulator = OFDMDemodulator(
        fft_size=grid.fft_size,
        l_min=receiver_l_min,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device=device,
    )
    ideal_demodulator = (
        demodulator
        if receiver_l_min == 0
        else OFDMDemodulator(
            fft_size=grid.fft_size,
            l_min=0,
            cyclic_prefix_length=grid.cyclic_prefix_length,
            device=device,
        )
    )
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
        transmit_grid, transmitted_bits = receiver.transmit(1, seed=child_seed)
        reference_transport_block_bits = np.asarray(
            transmitted_bits.detach().cpu().numpy()[0], dtype=np.uint8
        )
        if reference_transport_block_bits.ndim != 2:
            raise RuntimeError("Sionna transmitter returned an unexpected transport-block shape")
        transport_block_sha256 = hashlib.sha256(
            reference_transport_block_bits.tobytes(order="C")
        ).hexdigest()
        tx_phase_noise_seed = None
        rx_phase_noise_seed = None
        transmit_waveform = None
        if channel_mode == "time_domain" or impairments is not None:
            transmit_waveform = modulator(transmit_grid)
        if impairments is not None:
            assert impairment_seed is not None
            tx_phase_noise_seed = _derive_impairment_seed(impairment_seed, record.sample_index, 0)
            rx_phase_noise_seed = _derive_impairment_seed(impairment_seed, record.sample_index, 1)
            assert transmit_waveform is not None
            transmit_waveform = apply_iq_imbalance(
                transmit_waveform,
                gain_imbalance_db=impairments.tx_iq_gain_imbalance_db,
                phase_imbalance_rad=impairments.tx_iq_phase_imbalance_rad,
            )
            transmit_waveform = _apply_phase_noise(
                transmit_waveform,
                linewidth_hz=impairments.tx_phase_noise_linewidth_hz,
                seed=tx_phase_noise_seed,
                sampling_frequency_hz=sampling_frequency_hz,
                device=device,
            )

        channel_metadata: dict[str, Any] = {"channel_mode": channel_mode}
        if channel_mode == "time_domain":
            assert transmit_waveform is not None
            received_waveform, channel_response, channel_metadata = _apply_sampled_time_channel(
                transmit_waveform,
                record,
                time_index=time_index,
                sampling_frequency_hz=sampling_frequency_hz,
                resource_grid=grid,
                device=device,
            )
            rx_antennas = int(channel_response.shape[0])
        else:
            cfr_array = np.asarray(
                record.cfr[0, :, 0, 0, time_index, :][None, :, :], dtype=np.complex64
            )
            cfr = resample_centered_cfr(
                cfr_array,
                source_spacing_hz=source_spacing_hz,
                target_subcarriers=int(grid.fft_size),
                target_spacing_hz=receiver_config.subcarrier_spacing_khz * 1_000.0,
            )
            h = torch.as_tensor(cfr[0], dtype=torch.complex64, device=device)
            rx_antennas = int(h.shape[0])
            if impairments is not None:
                assert transmit_waveform is not None
                transmit_grid = ideal_demodulator(transmit_waveform)
            received_clean = transmit_grid[:, 0, 0].unsqueeze(1) * h[None, :, None, :]
            received_clean = received_clean.unsqueeze(1)
            if received_clean.shape[2] != rx_antennas:
                raise RuntimeError("receive-antenna axis changed during PUSCH channel application")
            channel_response = h[:, None, :].expand(
                rx_antennas,
                int(grid.num_ofdm_symbols),
                int(grid.fft_size),
            )
            channel_metadata.update(
                {
                    "path_delay_reference": "frequency-domain CFR control",
                    "tap_span_fits_cyclic_prefix": None,
                }
            )

        if channel_mode == "time_domain":
            # The sampled channel already returns continuous receiver samples.
            if impairments is not None:
                assert rx_phase_noise_seed is not None
                received_waveform = _apply_receiver_impairments(
                    received_waveform,
                    impairments,
                    phase_noise_seed=rx_phase_noise_seed,
                    sampling_frequency_hz=sampling_frequency_hz,
                    device=device,
                )
            received_clean = demodulator(received_waveform)
        elif impairments is not None:
            received_waveform = modulator(received_clean)
            assert rx_phase_noise_seed is not None
            received_waveform = _apply_receiver_impairments(
                received_waveform,
                impairments,
                phase_noise_seed=rx_phase_noise_seed,
                sampling_frequency_hz=sampling_frequency_hz,
                device=device,
            )
            received_clean = demodulator(received_waveform)

        target_cfr = channel_response[None, None, :, None, None, :, :]
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
            "channel_mode": channel_mode,
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
        target_cfr = torch.index_select(target_cfr, -1, effective_bins)
        profile_target = np.asarray([int(record.profile_los_component)], dtype=np.uint8)
        profile_definition = TargetDefinition.versioned(
            semantic="standardized_profile_los_category_proxy",
            units="binary_class",
            source="Sionna standardized TDL/CDL profile model",
            reference=(
                "profile_los_component false maps to 0 and true maps to 1; not geometric LOS truth"
            ),
            aggregation_rule="one profile-category proxy per generated channel record",
            definition_version="umcd.standardized.profile-los-proxy.v1",
        )
        channel_definition = TargetDefinition.versioned(
            semantic="perfect_synchronized_effective_subcarrier_cfr",
            units="complex channel gain (dimensionless)",
            source="standardized Sionna TDL/CDL CFR adapter",
            reference=(
                "profile-relative excess-delay path truth sinc-discretized for this NR sample rate; "
                "the static per-subcarrier target does not encode time-domain ISI/ICI"
                if channel_mode == "time_domain"
                else "profile-relative excess-delay CFR resampled to this resource grid"
            ),
            aggregation_rule=(
                "retain each effective subcarrier coefficient in receiver FFT-bin order; "
                "no antenna, symbol, or power averaging"
            ),
            definition_version="umcd.standardized.effective-cfr.v1",
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
                    "channel_mode": channel_mode,
                    "sampled_time_channel": channel_metadata,
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
                    "reference_transport_block_sha256": transport_block_sha256,
                    "reference_transport_block_shape": list(reference_transport_block_bits.shape),
                    "noise_variance_per_complex_dimension": float(noise_variance),
                    "noise_variance_per_real_dimension": float(noise_variance) / 2.0,
                    "noise_seed": child_noise_seed,
                    "subcarrier_spacing_hz": receiver_config.subcarrier_spacing_khz * 1_000,
                    "carrier_frequency_hz": carrier_hz,
                    "time_index": time_index,
                    "channel_time_variation": "frozen CFR across one PUSCH allocation",
                    "receiver_link_model": (
                        "sampled time-domain standardized CIR; channel-induced CP spillover/ISI "
                        "and impairment-induced ICI from sample-domain RF impairments are retained"
                        if channel_mode == "time_domain" and impairments is not None
                        else "sampled time-domain standardized CIR; channel-induced CP spillover/ISI "
                        "is retained"
                        if channel_mode == "time_domain"
                        else "static circular frequency-domain CFR with sample-domain RF impairments; "
                        "impairment-induced ICI is retained, channel-induced CP spillover/ISI is not modeled"
                        if impairments is not None
                        else "ideal diagonal frequency-domain CFR; no ICI/ISI"
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
                reference_transport_block_bits=reference_transport_block_bits,
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class StandardizedPUSCHDecodeOutcome:
    """Per-record diagnostic decode, including exact reference-payload errors."""

    sample_id: str
    transport_block_crc_pass: bool
    payload_bit_errors: int
    payload_bit_count: int


def audit_standardized_pusch_records_for_diagnostic(
    records: Sequence[NRChannelRecord],
    *,
    receiver_config: NRPUSCHConfig,
    estimate_source: Literal["dmrs_ls", "diagonal_cfr_oracle"],
    device: str = "cpu",
) -> tuple[StandardizedPUSCHDecodeOutcome, ...]:
    """Decode records and compare bits with their retained exact TX reference.

    The reference payload is evaluation-side data, stored outside
    ``NRObservation`` and never exposed to ``NRModelInputs``. This diagnostic
    cross-checks the payload bit error count and transport-block CRC; it is not
    a preregistered receiver qualification result.

    ``diagonal_cfr_oracle`` is an analysis-only control for separating channel
    estimation error from off-diagonal ISI/ICI. It must not be exposed as a
    model input or interpreted as a general receiver configuration.
    """
    if not records:
        raise ValueError("records must not be empty")
    if estimate_source not in {"dmrs_ls", "diagonal_cfr_oracle"}:
        raise ValueError("unsupported diagnostic estimate source")

    import torch
    from sionna.phy.nr import PUSCHReceiver

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA decoding was requested but CUDA is unavailable")
    transmitter = SionnaPUSCHLSReceiver(receiver_config, device=device)
    decoder = PUSCHReceiver(
        transmitter._transmitter,
        channel_estimator="perfect",
        return_tb_crc_status=True,
        input_domain="freq",
        device=device,
    )
    outcomes: list[StandardizedPUSCHDecodeOutcome] = []
    for record in records:
        reference_bits = record.reference_transport_block_bits
        if reference_bits is None:
            raise ValueError(
                f"record {record.observation.sample_id!r} has no retained TX reference payload"
            )
        reference_array = np.asarray(reference_bits, dtype=np.uint8)
        expected_shape = record.provenance.get("reference_transport_block_shape")
        expected_digest = record.provenance.get("reference_transport_block_sha256")
        if (
            not isinstance(expected_shape, tuple)
            or expected_shape != tuple(reference_array.shape)
            or expected_digest != hashlib.sha256(reference_array.tobytes(order="C")).hexdigest()
        ):
            raise ValueError("retained TX reference payload differs from its provenance digest")
        estimate = (
            record.observation.channel_estimate.h_hat
            if estimate_source == "dmrs_ls"
            else record.targets["channel_cfr"]
        )
        received_grid = torch.as_tensor(record.observation.received_grid, device=torch_device)
        estimate_tensor = torch.as_tensor(estimate, device=torch_device)
        decoded_bits, crc = decoder(
            received_grid,
            torch.as_tensor(
                record.provenance["noise_variance_per_complex_dimension"], device=torch_device
            ),
            h=estimate_tensor,
        )
        reference_tensor = torch.as_tensor(reference_array, device=decoded_bits.device)
        if tuple(decoded_bits.shape) == (1, *tuple(reference_tensor.shape)):
            reference_tensor = reference_tensor.unsqueeze(0)
        if tuple(decoded_bits.shape) != tuple(reference_tensor.shape):
            raise ValueError("decoded payload shape differs from retained TX reference payload")
        outcomes.append(
            StandardizedPUSCHDecodeOutcome(
                sample_id=record.observation.sample_id,
                transport_block_crc_pass=bool(crc.all()),
                payload_bit_errors=int(torch.count_nonzero(decoded_bits != reference_tensor)),
                payload_bit_count=int(reference_tensor.numel()),
            )
        )
    return tuple(outcomes)


def decode_standardized_pusch_records_for_diagnostic(
    records: Sequence[NRChannelRecord],
    *,
    receiver_config: NRPUSCHConfig,
    estimate_source: Literal["dmrs_ls", "diagonal_cfr_oracle"],
    device: str = "cpu",
) -> tuple[bool, ...]:
    """Return TB CRC outcomes, preserving compatibility with records lacking TX bits."""
    if not records:
        raise ValueError("records must not be empty")
    if estimate_source not in {"dmrs_ls", "diagonal_cfr_oracle"}:
        raise ValueError("unsupported diagnostic estimate source")

    import torch
    from sionna.phy.nr import PUSCHReceiver

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA decoding was requested but CUDA is unavailable")
    transmitter = SionnaPUSCHLSReceiver(receiver_config, device=device)
    decoder = PUSCHReceiver(
        transmitter._transmitter,
        channel_estimator="perfect",
        return_tb_crc_status=True,
        input_domain="freq",
        device=device,
    )
    crc_pass: list[bool] = []
    for record in records:
        estimate = (
            record.observation.channel_estimate.h_hat
            if estimate_source == "dmrs_ls"
            else record.targets["channel_cfr"]
        )
        _, crc = decoder(
            torch.as_tensor(record.observation.received_grid, device=torch_device),
            torch.as_tensor(
                record.provenance["noise_variance_per_complex_dimension"], device=torch_device
            ),
            h=torch.as_tensor(estimate, device=torch_device),
        )
        crc_pass.append(bool(crc.all()))
    return tuple(crc_pass)
