"""Fixed-payload PUSCH checks for multipath delay relative to the cyclic prefix."""

from __future__ import annotations

import hashlib
import math
from typing import Any


def run_static_cp_margin_experiment(
    *,
    frames: int = 512,
    batch_size: int = 64,
    seed: int = 50_000,
    snr_db: float = 30.0,
    n_size_bwp: int = 12,
    subcarrier_spacing_khz: int = 30,
    in_cp_guard_samples: int = 2,
    beyond_cp_samples: int = 3,
) -> dict[str, Any]:
    """Compare a fixed PUSCH payload with an echo just inside and beyond its CP.

    Both conditions reuse each transmitted waveform and standardized AWGN draw.
    The only channel change is the integer delay of a deterministic second path.
    The burst has zero prehistory, which is stated in the returned provenance;
    this is a controlled ISI diagnostic, not a standardized propagation model.
    """
    for name, value in (("frames", frames), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if seed > 2**63 - 2:
        raise ValueError("seed must leave room for the derived noise seed")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if not -300.0 <= snr_db <= 300.0:
        raise ValueError("snr_db produces an invalid noise variance")
    for name, value in (
        ("n_size_bwp", n_size_bwp),
        ("subcarrier_spacing_khz", subcarrier_spacing_khz),
        ("in_cp_guard_samples", in_cp_guard_samples),
        ("beyond_cp_samples", beyond_cp_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if in_cp_guard_samples < 1 or beyond_cp_samples < 1:
        raise ValueError("CP guard and beyond-CP offsets must each be at least one sample")

    try:
        import torch
        from sionna.phy.nr import PUSCHReceiver
        from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator
    except ImportError as exc:
        raise RuntimeError("Install the optional Sionna 2.2 PHY dependencies") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("The native PUSCH CP-margin experiment requires CUDA")

    from receivers.nr import NRPUSCHConfig

    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver

    device = "cuda:0"
    phy = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
        ),
        device=device,
    )
    grid = phy.resource_grid
    cp = int(torch.as_tensor(grid.cyclic_prefix_length).min().item())
    if in_cp_guard_samples >= cp:
        raise ValueError("in_cp_guard_samples must be smaller than the shortest CP")
    in_delay = cp - in_cp_guard_samples
    out_delay = cp + beyond_cp_samples
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
    decoder = PUSCHReceiver(
        phy._transmitter,
        return_tb_crc_status=True,
        input_domain="freq",
        device=device,
    )

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = 1.0 / snr_linear
    direct_gain = 0.85 + 0.0j
    echo_gain = 0.48 + 0.12j
    errors = {"inside_cp": 0, "beyond_cp": 0}
    bit_errors = {"inside_cp": 0, "beyond_cp": 0}
    error_masks: dict[str, list[bool]] = {"inside_cp": [], "beyond_cp": []}
    receive_waveform_power: dict[str, list[float]] = {"inside_cp": [], "beyond_cp": []}
    transmit_waveform_power: list[float] = []
    bits_per_frame: int | None = None
    checks: list[dict[str, Any]] = []
    frames_left = frames
    batch_index = 0
    while frames_left:
        count = min(batch_size, frames_left)
        batch_seed = seed + batch_index
        transmit_grid, transmit_bits = phy.transmit(count, seed=batch_seed)
        waveform = modulator(transmit_grid)
        transmit_waveform_power.append(float(waveform.abs().square().mean().item()))
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(batch_seed + 1)
        noise = torch.complex(
            torch.randn(waveform.shape, device=device, generator=noise_generator),
            torch.randn(waveform.shape, device=device, generator=noise_generator),
        ) / math.sqrt(2.0)
        bits_cpu = transmit_bits.detach().cpu().contiguous()
        noise_cpu = noise.detach().cpu().contiguous()
        checks.append(
            {
                "seed": batch_seed,
                "noise_seed": batch_seed + 1,
                "transmit_bits_sha256": hashlib.sha256(bits_cpu.numpy().tobytes()).hexdigest(),
                "standardized_noise_sha256": hashlib.sha256(
                    noise_cpu.numpy().tobytes()
                ).hexdigest(),
            }
        )
        if bits_per_frame is None:
            bits_per_frame = int(transmit_bits[0].numel())
        elif bits_per_frame != int(transmit_bits[0].numel()):
            raise RuntimeError("PUSCH transport-block size changed between batches")

        for condition, delay in (("inside_cp", in_delay), ("beyond_cp", out_delay)):
            delayed = torch.cat(
                (torch.zeros_like(waveform[..., :delay]), waveform[..., :-delay]), dim=-1
            )
            channel_output = direct_gain * waveform + echo_gain * delayed
            receive_waveform_power[condition].append(
                float(channel_output.abs().square().mean().item())
            )
            received_grid = demodulator(channel_output + noise * math.sqrt(noise_variance))
            decoded, crc_ok = decoder(received_grid, noise_variance)
            crc_ok = crc_ok.reshape(-1).to(dtype=torch.bool)
            error_masks[condition].extend((~crc_ok).detach().cpu().tolist())
            errors[condition] += int(torch.count_nonzero(~crc_ok).item())
            bit_errors[condition] += int(torch.count_nonzero(decoded != transmit_bits).item())
        frames_left -= count
        batch_index += 1

    points = {
        condition: {
            "frames": frames,
            "bler_errors": errors[condition],
            "bit_errors": bit_errors[condition],
            "bits": frames * int(bits_per_frame or 0),
            "frame_error_mask": error_masks[condition],
        }
        for condition in ("inside_cp", "beyond_cp")
    }
    return {
        "configuration": {
            "resource_blocks": n_size_bwp,
            "subcarrier_spacing_khz": subcarrier_spacing_khz,
            "fft_size": int(grid.fft_size),
            "cyclic_prefix_min_samples": cp,
            "in_cp_delay_samples": in_delay,
            "in_cp_remaining_margin_samples": cp - in_delay,
            "beyond_cp_delay_samples": out_delay,
            "beyond_cp_excess_samples": out_delay - cp,
            "direct_path_gain": [direct_gain.real, direct_gain.imag],
            "echo_path_gain": [echo_gain.real, echo_gain.imag],
            "sampled_waveform_snr_db": snr_db,
            "noise_variance_per_complex_sample": noise_variance,
            "zero_prehistory_at_burst_start": True,
            "frames": frames,
            "batch_size": batch_size,
            "seed": seed,
            "matched_realization_checks": checks,
            "transmit_waveform_mean_power_by_batch": transmit_waveform_power,
            "same_pusch_bits_and_awgn_across_conditions": True,
        },
        "points": points,
        "receive_waveform_mean_power_by_batch": receive_waveform_power,
        "limitations": [
            "Static deterministic two-path discrete-time channel; not a TDL/CDL profile.",
            "Finite PUSCH bursts start with zero channel prehistory; continuous-slot boundary effects are not represented.",
            "The comparison isolates CP support at one fixed PUSCH allocation and SNR only.",
            "Codeword pairing is conditional on this shared waveform and noise realization, not independent-scene inference.",
            "Not standards certification or validation against external measurements.",
        ],
    }
