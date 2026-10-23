"""Paired CPU PUSCH check for ordered RF impairments over known multipath.

This is a controlled receiver qualification, not a field BLER result. The
known chain is TX phase noise -> static two-tap propagation -> RX CFO/phase
noise -> integer capture delay -> AWGN. Timing and CFO are acquired and
corrected before DMRS channel estimation and PUSCH decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
from integrations.sionna.ofdm_impairments import (
    apply_integer_timing_offset,
    apply_phase_noise,
    apply_residual_cfo,
    apply_time_varying_multipath,
    estimate_integer_timing_offset_from_cp,
    estimate_residual_cfo_from_cp,
    generate_wiener_phase_noise_rad,
)
from receivers.nr import NRPUSCHConfig

SCHEMA = "umcd.g5_ordered_impairments_receiver.v1"
RESULT_SCHEMA = "umcd.g5_ordered_impairments_result.v1"
PAYLOAD_SEEDS = tuple(range(73_501, 73_509))
BLOCKS_PER_SEED = 16
NOISE_SEED_OFFSET = 200_000
TX_PHASE_SEED_OFFSET = 300_000
RX_PHASE_SEED_OFFSET = 400_000
NOISE_VARIANCE = 1e-3
CFO_HZ = 600.0
TIMING_OFFSET_SAMPLES = 2
TX_LINEWIDTH_HZ = 10.0
RX_LINEWIDTH_HZ = 10.0
DIRECT_COEFFICIENT = 0.9 - 0.1j
ECHO_COEFFICIENT = 0.35 + 0.2j
ECHO_DELAY_SAMPLES = 2
MAX_TIMING_OFFSET_SAMPLES = 3
CP_GUARD_SAMPLES = ECHO_DELAY_SAMPLES

# Frozen before running the confirmatory seeds: timing acquisition must be
# exact; absolute CFO error must be below 250 Hz at P95 and 350 Hz maximum;
# clean reference BLER must be <=1%; corrected CRC retention must be >=90%.
MAX_CFO_ERROR_P95_HZ = 250.0
MAX_CFO_ERROR_HZ = 350.0
MAX_REFERENCE_BLER = 0.01
MAX_IMPAIRED_BLER = 0.10


def _proper_complex_noise(shape: Any, seed: int, variance: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float32)
    imag = torch.randn(shape, generator=generator, dtype=torch.float32)
    return torch.complex(real, imag) * math.sqrt(variance / 2.0)


def _phase_noise(shape: Any, seed: int, sample_rate_hz: float, linewidth_hz: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return generate_wiener_phase_noise_rad(
        shape[-1],
        sampling_frequency_hz=sample_rate_hz,
        linewidth_hz=linewidth_hz,
        generator=generator,
        leading_shape=tuple(shape[:-1]),
    )


def _known_chain(
    waveform: torch.Tensor,
    *,
    sample_rate_hz: float,
    tx_phase: torch.Tensor,
    rx_phase: torch.Tensor,
    capture_delay: int,
) -> torch.Tensor:
    """Apply the physical impairment sequence declared by this experiment."""
    tx_noisy = apply_phase_noise(waveform, phase_noise_rad=tx_phase)
    propagated = apply_time_varying_multipath(
        tx_noisy,
        path_coefficients=(DIRECT_COEFFICIENT, ECHO_COEFFICIENT),
        path_delays_samples=(0, ECHO_DELAY_SAMPLES),
        path_dopplers_hz=(0.0, 0.0),
        sampling_frequency_hz=sample_rate_hz,
    )
    cfo = apply_residual_cfo(propagated, cfo_hz=CFO_HZ, sampling_frequency_hz=sample_rate_hz)
    rx_noisy = apply_phase_noise(cfo, phase_noise_rad=rx_phase)
    return apply_integer_timing_offset(rx_noisy, offset_samples=capture_delay)


def _run_ordered_impairment_probe(
    *,
    payload_seeds: tuple[int, ...] = PAYLOAD_SEEDS,
    blocks_per_seed: int = BLOCKS_PER_SEED,
) -> dict[str, Any]:
    """Run paired clean/impaired CPU PUSCH with independent payload/noise seeds."""
    if not payload_seeds or len(set(payload_seeds)) != len(payload_seeds):
        raise ValueError("payload_seeds must be nonempty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in payload_seeds
    ):
        raise ValueError("payload_seeds must contain nonnegative integers")
    if (
        isinstance(blocks_per_seed, bool)
        or not isinstance(blocks_per_seed, int)
        or blocks_per_seed < 1
    ):
        raise ValueError("blocks_per_seed must be a positive integer")

    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    receiver = SionnaPUSCHLSReceiver(config, device="cpu")
    grid = receiver.resource_grid
    sample_rate_hz = float(grid.fft_size * config.subcarrier_spacing_khz * 1_000)
    cp_length = int(grid.cyclic_prefix_length)
    if MAX_TIMING_OFFSET_SAMPLES >= cp_length:
        raise RuntimeError("configured capture timing search must fit within the CP")

    outcomes: dict[str, list[bool]] = {
        "clean": [],
        "impaired_uncorrected": [],
        "impaired_corrected": [],
    }
    payload_bit_errors = {name: 0 for name in outcomes}
    cfo_errors: list[float] = []
    timing_errors: list[int] = []
    max_reference_error = 0.0
    min_wrong_order_difference = math.inf

    for payload_seed in payload_seeds:
        tx_grid, tx_bits = receiver.transmit(blocks_per_seed, seed=payload_seed)
        tx_waveform = receiver.modulate_slot_grid(tx_grid)
        shape = tx_waveform.shape
        tx_phase = _phase_noise(
            shape, payload_seed + TX_PHASE_SEED_OFFSET, sample_rate_hz, TX_LINEWIDTH_HZ
        )
        rx_phase = _phase_noise(
            shape, payload_seed + RX_PHASE_SEED_OFFSET, sample_rate_hz, RX_LINEWIDTH_HZ
        )

        # Independently spell out the declared sample-domain reference rather
        # than using _known_chain to generate its expected values.
        tx_reference = tx_waveform * torch.polar(torch.ones_like(tx_phase), tx_phase)
        sample_axis = torch.arange(tx_waveform.shape[-1], dtype=torch.float32)
        delayed_tx = torch.nn.functional.pad(tx_reference, (ECHO_DELAY_SAMPLES, 0))[
            ..., : tx_reference.shape[-1]
        ]
        channel_reference = DIRECT_COEFFICIENT * tx_reference + ECHO_COEFFICIENT * delayed_tx
        cfo_phase = 2.0 * math.pi * CFO_HZ * sample_axis / sample_rate_hz
        cfo_reference = channel_reference * torch.polar(torch.ones_like(cfo_phase), cfo_phase)
        rx_reference = cfo_reference * torch.polar(torch.ones_like(rx_phase), rx_phase)
        reference_capture = torch.cat(
            (
                torch.zeros_like(rx_reference[..., :TIMING_OFFSET_SAMPLES]),
                rx_reference[..., :-TIMING_OFFSET_SAMPLES],
            ),
            dim=-1,
        )
        actual_capture = _known_chain(
            tx_waveform,
            sample_rate_hz=sample_rate_hz,
            tx_phase=tx_phase,
            rx_phase=rx_phase,
            capture_delay=TIMING_OFFSET_SAMPLES,
        )
        max_reference_error = max(
            max_reference_error,
            float(torch.max(torch.abs(actual_capture - reference_capture))),
        )

        # An order-negative control moves receiver phase noise ahead of the
        # multipath operator. It must differ because delayed samples carry the
        # oscillator phase from their transmit time, not their receive time.
        rx_phase_before_channel = apply_phase_noise(tx_reference, phase_noise_rad=rx_phase)
        wrong_order = apply_time_varying_multipath(
            rx_phase_before_channel,
            path_coefficients=(DIRECT_COEFFICIENT, ECHO_COEFFICIENT),
            path_delays_samples=(0, ECHO_DELAY_SAMPLES),
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        wrong_order = apply_residual_cfo(
            wrong_order, cfo_hz=CFO_HZ, sampling_frequency_hz=sample_rate_hz
        )
        wrong_order = apply_integer_timing_offset(wrong_order, offset_samples=TIMING_OFFSET_SAMPLES)
        min_wrong_order_difference = min(
            min_wrong_order_difference,
            float(torch.sqrt(torch.mean(torch.abs(actual_capture - wrong_order) ** 2))),
        )

        noise_seed = payload_seed + NOISE_SEED_OFFSET
        noise = _proper_complex_noise(shape, noise_seed, NOISE_VARIANCE)
        clean_channel = apply_time_varying_multipath(
            tx_waveform,
            path_coefficients=(DIRECT_COEFFICIENT, ECHO_COEFFICIENT),
            path_delays_samples=(0, ECHO_DELAY_SAMPLES),
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        clean_capture = (
            apply_integer_timing_offset(clean_channel, offset_samples=TIMING_OFFSET_SAMPLES) + noise
        )
        clean_grid = receiver.demodulate_slot_waveform(
            apply_integer_timing_offset(clean_capture, offset_samples=-TIMING_OFFSET_SAMPLES)
        )
        clean_bits, clean_crc = receiver.decode_pusch(clean_grid, torch.tensor(NOISE_VARIANCE))
        outcomes["clean"].extend(bool(x) for x in clean_crc.reshape(-1).tolist())
        payload_bit_errors["clean"] += int((clean_bits != tx_bits).sum().item())

        noisy_capture = actual_capture + noise
        uncorrected_grid = receiver.demodulate_slot_waveform(noisy_capture)
        uncorrected_bits, uncorrected_crc = receiver.decode_pusch(
            uncorrected_grid, torch.tensor(NOISE_VARIANCE)
        )
        outcomes["impaired_uncorrected"].extend(
            bool(x) for x in uncorrected_crc.reshape(-1).tolist()
        )
        payload_bit_errors["impaired_uncorrected"] += int(
            (uncorrected_bits != tx_bits).sum().item()
        )
        acquired_timing = estimate_integer_timing_offset_from_cp(
            noisy_capture,
            fft_size=int(grid.fft_size),
            cyclic_prefix_length=cp_length,
            max_offset_samples=MAX_TIMING_OFFSET_SAMPLES,
            guard_samples=CP_GUARD_SAMPLES,
        )
        timing_error = acquired_timing - TIMING_OFFSET_SAMPLES
        timing_errors.extend(int(x) for x in timing_error.reshape(-1).tolist())
        aligned = apply_integer_timing_offset(
            noisy_capture, offset_samples=-int(acquired_timing[0].item())
        )[..., : tx_waveform.shape[-1]]
        estimated_cfo = estimate_residual_cfo_from_cp(
            aligned,
            fft_size=int(grid.fft_size),
            cyclic_prefix_length=cp_length,
            sampling_frequency_hz=sample_rate_hz,
            guard_samples=CP_GUARD_SAMPLES,
        )
        cfo_errors.extend(float(x) for x in (estimated_cfo - CFO_HZ).reshape(-1).tolist())
        corrected = apply_residual_cfo(
            aligned,
            cfo_hz=-estimated_cfo,
            sampling_frequency_hz=sample_rate_hz,
        )
        received_grid = receiver.demodulate_slot_waveform(corrected)
        decoded, crc = receiver.decode_pusch(received_grid, torch.tensor(NOISE_VARIANCE))
        outcomes["impaired_corrected"].extend(bool(x) for x in crc.reshape(-1).tolist())
        if decoded.shape != tx_bits.shape:
            raise RuntimeError("PUSCH decoded payload shape changed unexpectedly")
        payload_bit_errors["impaired_corrected"] += int((decoded != tx_bits).sum().item())

    total = len(payload_seeds) * blocks_per_seed
    failures = {name: total - sum(values) for name, values in outcomes.items()}
    sorted_cfo_errors = sorted(abs(value) for value in cfo_errors)
    cfo_p95 = sorted_cfo_errors[math.ceil(0.95 * len(sorted_cfo_errors)) - 1]
    summary = {
        name: {
            "blocks": total,
            "crc_failures": failures[name],
            "bler": failures[name] / total,
            "payload_bit_errors": payload_bit_errors[name],
        }
        for name in outcomes
    }
    acceptance: dict[str, Any] = {
        "criteria_frozen_before_run": {
            "max_abs_timing_error_samples": 0,
            "p95_abs_cfo_error_hz": MAX_CFO_ERROR_P95_HZ,
            "max_abs_cfo_error_hz": MAX_CFO_ERROR_HZ,
            "max_clean_reference_bler": MAX_REFERENCE_BLER,
            "max_impaired_corrected_bler": MAX_IMPAIRED_BLER,
            "max_samplewise_reference_absolute_error": 2e-5,
            "min_wrong_order_rmse": 1e-4,
        },
        "observed": {
            "max_abs_timing_error_samples": max((abs(x) for x in timing_errors), default=0),
            "p95_abs_cfo_error_hz": cfo_p95,
            "max_abs_cfo_error_hz": max((abs(x) for x in cfo_errors), default=0.0),
            "max_samplewise_reference_absolute_error": max_reference_error,
            "min_wrong_order_rmse": min_wrong_order_difference,
            "clean_bler": summary["clean"]["bler"],
            "impaired_corrected_bler": summary["impaired_corrected"]["bler"],
        },
    }
    observed = acceptance["observed"]
    acceptance["pass"] = (
        observed["max_abs_timing_error_samples"] == 0
        and observed["p95_abs_cfo_error_hz"] < MAX_CFO_ERROR_P95_HZ
        and observed["max_abs_cfo_error_hz"] < MAX_CFO_ERROR_HZ
        and observed["max_samplewise_reference_absolute_error"] < 2e-5
        and observed["min_wrong_order_rmse"] > 1e-4
        and observed["clean_bler"] <= MAX_REFERENCE_BLER
        and observed["impaired_corrected_bler"] <= MAX_IMPAIRED_BLER
    )
    return {
        "schema": SCHEMA,
        "scope": "one static synthetic two-tap channel, 6-RB/30-kHz CPU PUSCH, known Wiener oscillator model; conditional receiver check, not deployment BLER",
        "config": {
            "sample_rate_hz": sample_rate_hz,
            "fft_size": int(grid.fft_size),
            "cyclic_prefix_samples": cp_length,
            "path_coefficients": [
                [DIRECT_COEFFICIENT.real, DIRECT_COEFFICIENT.imag],
                [ECHO_COEFFICIENT.real, ECHO_COEFFICIENT.imag],
            ],
            "path_delays_samples": [0, ECHO_DELAY_SAMPLES],
            "cfo_hz": CFO_HZ,
            "integer_capture_delay_samples": TIMING_OFFSET_SAMPLES,
            "tx_phase_noise_linewidth_hz": TX_LINEWIDTH_HZ,
            "rx_phase_noise_linewidth_hz": RX_LINEWIDTH_HZ,
            "waveform_awgn_complex_variance": NOISE_VARIANCE,
            "payload_seeds": list(payload_seeds),
            "noise_seeds": [seed + NOISE_SEED_OFFSET for seed in payload_seeds],
            "tx_phase_noise_seeds": [seed + TX_PHASE_SEED_OFFSET for seed in payload_seeds],
            "rx_phase_noise_seeds": [seed + RX_PHASE_SEED_OFFSET for seed in payload_seeds],
            "blocks_per_seed": blocks_per_seed,
        },
        "paired_outcomes": summary,
        "timing_error_samples_by_block": timing_errors,
        "cfo_error_hz_by_block": cfo_errors,
        "acceptance": acceptance,
    }


def run_ordered_impairment_probe(
    *,
    payload_seeds: tuple[int, ...] = PAYLOAD_SEEDS,
    blocks_per_seed: int = BLOCKS_PER_SEED,
) -> dict[str, Any]:
    """Run with bounded CPU threads and always restore the caller's setting."""
    original_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run_ordered_impairment_probe(
            payload_seeds=payload_seeds, blocks_per_seed=blocks_per_seed
        )
    finally:
        torch.set_num_threads(original_torch_threads)


def _canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical byte representation used for result hashes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_revision(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _source_matches_git_revision(repo_root: Path, source_name: str, source: bytes) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{source_name}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return source == completed.stdout


def build_hash_bound_result(
    report: dict[str, Any], *, created_utc: str | None = None
) -> dict[str, Any]:
    """Bind a probe report to its exact source, protocol, and runtime provenance."""
    source_path = Path(__file__).resolve()
    repo_root = source_path.parent.parent
    protocol = {
        "experiment_schema": report["schema"],
        "config": report["config"],
        "acceptance_criteria": report["acceptance"]["criteria_frozen_before_run"],
    }
    source_bytes = source_path.read_bytes()
    try:
        source_name = source_path.relative_to(repo_root).as_posix()
    except ValueError:
        source_name = source_path.name
    packages: dict[str, str | None] = {}
    for distribution in ("torch", "sionna", "sionna-phy", "sionna-rt"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None

    bound: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "interpretation": {
            "criterion_scope": "This verdict applies only to the declared deterministic synthetic CPU fixture.",
            "deployment_qualification": False,
            "g5_09_closed": False,
            "status": "conditional_fixture_pass"
            if report["acceptance"]["pass"]
            else "conditional_fixture_fail",
        },
        "experiment": report,
        "provenance": {
            "created_utc": created_utc or datetime.now(UTC).isoformat(),
            "git_revision": _git_revision(repo_root),
            "source_path": source_name,
            "source_sha256": _sha256(source_bytes),
            "source_matches_git_revision": _source_matches_git_revision(
                repo_root, source_name, source_bytes
            ),
            "protocol": protocol,
            "protocol_sha256": _sha256(_canonical_json_bytes(protocol)),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": packages,
                "torch_device": "cpu",
            },
        },
    }
    bound["result_sha256"] = _sha256(_canonical_json_bytes(bound))
    return bound


def verify_hash_bound_result(result: dict[str, Any]) -> bool:
    """Check internal result/config digests without asserting scientific validity."""
    if result.get("schema") != RESULT_SCHEMA:
        return False
    provenance = result.get("provenance")
    experiment = result.get("experiment")
    if not isinstance(provenance, dict) or not isinstance(experiment, dict):
        return False
    protocol = provenance.get("protocol")
    if not isinstance(protocol, dict):
        return False
    expected_protocol = {
        "experiment_schema": experiment.get("schema"),
        "config": experiment.get("config"),
        "acceptance_criteria": experiment.get("acceptance", {}).get("criteria_frozen_before_run"),
    }
    if protocol != expected_protocol:
        return False
    try:
        if provenance.get("protocol_sha256") != _sha256(_canonical_json_bytes(protocol)):
            return False
        unhashed = dict(result)
        supplied_digest = unhashed.pop("result_sha256", None)
        return supplied_digest == _sha256(_canonical_json_bytes(unhashed))
    except (TypeError, ValueError):
        return False


def write_hash_bound_result(path: str | Path, result: dict[str, Any]) -> None:
    """Write a verified result atomically as canonical, human-readable JSON."""
    if not verify_hash_bound_result(result):
        raise ValueError("refusing to write a result with invalid hash bindings")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=len(PAYLOAD_SEEDS))
    parser.add_argument("--blocks-per-seed", type=int, default=BLOCKS_PER_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        help="write a hash-bound local result JSON to this path",
    )
    args = parser.parse_args()
    if not 1 <= args.seed_count <= len(PAYLOAD_SEEDS):
        parser.error(f"--seed-count must be in [1, {len(PAYLOAD_SEEDS)}]")
    report = run_ordered_impairment_probe(
        payload_seeds=PAYLOAD_SEEDS[: args.seed_count], blocks_per_seed=args.blocks_per_seed
    )
    if args.output is not None:
        result = build_hash_bound_result(report)
        write_hash_bound_result(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
