"""Small NR PSS/SSS acquisition reference for controlled receiver tests.

The implementation follows the PSS/SSS sequence definitions and SS/PBCH
resource mapping in 3GPP TS 38.211. It searches a configured numerology and a
finite CFO hypothesis set; it is not a complete cell-search receiver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.signal import correlate

ComplexArray = NDArray[np.complex128]


class NRSyncNotFoundError(ValueError):
    """Raised when acquisition scores do not meet configured detection gates."""


@dataclass(frozen=True, slots=True)
class PSSDetection:
    """Best PSS match with integer/fractional timing and CP-refined CFO."""

    sample_offset: int
    nid2: int
    cfo_hz: float
    normalized_score: float
    fractional_sample_offset: float = 0.0


@dataclass(frozen=True, slots=True)
class NRCellDetection:
    """PSS/SSS-derived physical cell identity and acquisition scores."""

    sample_offset: int
    nid1: int
    nid2: int
    physical_cell_id: int
    cfo_hz: float
    pss_score: float
    sss_score: float
    fractional_sample_offset: float = 0.0


def _validate_identity(nid1: int, nid2: int | None = None) -> None:
    if isinstance(nid1, bool) or not isinstance(nid1, int) or not 0 <= nid1 <= 335:
        raise ValueError("nid1 must be an integer in [0, 335]")
    if nid2 is not None and (
        isinstance(nid2, bool) or not isinstance(nid2, int) or not 0 <= nid2 <= 2
    ):
        raise ValueError("nid2 must be an integer in [0, 2]")


def nr_pss_sequence(nid2: int) -> NDArray[np.int8]:
    """Generate the length-127 NR PSS sequence for N_ID^(2)."""
    if isinstance(nid2, bool) or not isinstance(nid2, int) or not 0 <= nid2 <= 2:
        raise ValueError("nid2 must be an integer in [0, 2]")
    x = np.zeros(127 + 7, dtype=np.int8)
    x[:7] = (0, 1, 1, 0, 1, 1, 1)
    for index in range(127):
        x[index + 7] = (x[index + 4] + x[index]) % 2
    indices = (np.arange(127) + 43 * nid2) % 127
    return (1 - 2 * x[indices]).astype(np.int8)


def nr_sss_sequence(nid1: int, nid2: int) -> NDArray[np.int8]:
    """Generate the length-127 NR SSS sequence for N_ID^(1), N_ID^(2)."""
    _validate_identity(nid1, nid2)
    x0 = np.zeros(127 + 7, dtype=np.int8)
    x1 = np.zeros(127 + 7, dtype=np.int8)
    x0[0] = 1
    x1[0] = 1
    for index in range(127):
        x0[index + 7] = (x0[index + 4] + x0[index]) % 2
        x1[index + 7] = (x1[index + 1] + x1[index]) % 2
    m0 = 15 * (nid1 // 112) + 5 * nid2
    m1 = nid1 % 112
    indexes = np.arange(127)
    return ((1 - 2 * x0[(indexes + m0) % 127]) * (1 - 2 * x1[(indexes + m1) % 127])).astype(np.int8)


def _validate_ofdm(fft_size: int, cyclic_prefix_length: int) -> None:
    if isinstance(fft_size, bool) or not isinstance(fft_size, int) or fft_size < 240:
        raise ValueError("fft_size must be an integer of at least 240 samples")
    if isinstance(cyclic_prefix_length, bool) or not isinstance(cyclic_prefix_length, int):
        raise TypeError("cyclic_prefix_length must be an integer")
    if not 0 < cyclic_prefix_length < fft_size:
        raise ValueError("cyclic_prefix_length must be in [1, fft_size)")


def modulate_nr_sync_symbol(
    sequence: NDArray[np.int8],
    *,
    fft_size: int,
    cyclic_prefix_length: int,
    fractional_delay_samples: float = 0.0,
) -> ComplexArray:
    """Map a PSS/SSS sequence to SSB tones, optionally delay it fractionally, and add a CP."""
    _validate_ofdm(fft_size, cyclic_prefix_length)
    if not math.isfinite(fractional_delay_samples) or abs(fractional_delay_samples) >= 0.5:
        raise ValueError("fractional_delay_samples must be finite and in (-0.5, 0.5)")
    if sequence.shape != (127,) or not np.isin(sequence, (-1, 1)).all():
        raise ValueError("sequence must contain 127 BPSK values in {-1, +1}")
    frequency = np.zeros(fft_size, dtype=np.complex128)
    # TS 38.211 maps the 127 signal REs to SSB indices 56..182. Relative to
    # the 240-subcarrier SSB grid origin this is k=-64..62.
    subcarrier_offsets = np.arange(-64, 63)
    frequency[subcarrier_offsets % fft_size] = sequence
    frequency *= np.exp(
        -2j * np.pi * np.fft.fftfreq(fft_size) * fft_size * fractional_delay_samples / fft_size
    )
    useful = np.fft.ifft(frequency)
    return np.concatenate((useful[-cyclic_prefix_length:], useful))


def make_nr_sync_burst(
    nid1: int,
    nid2: int,
    *,
    fft_size: int = 256,
    cyclic_prefix_length: int = 32,
    fractional_delay_samples: float = 0.0,
) -> ComplexArray:
    """Create PSS/SSS-only symbols at SSB symbol indices 0 and 2.

    The other two symbols are zero; PBCH and its DM-RS are deliberately not
    synthesized, so this is an acquisition fixture rather than a full SS/PBCH
    block transmitter.
    """
    _validate_identity(nid1, nid2)
    pss = modulate_nr_sync_symbol(
        nr_pss_sequence(nid2),
        fft_size=fft_size,
        cyclic_prefix_length=cyclic_prefix_length,
        fractional_delay_samples=fractional_delay_samples,
    )
    sss = modulate_nr_sync_symbol(
        nr_sss_sequence(nid1, nid2),
        fft_size=fft_size,
        cyclic_prefix_length=cyclic_prefix_length,
        fractional_delay_samples=fractional_delay_samples,
    )
    empty = np.zeros_like(pss)
    return np.concatenate((pss, empty, sss, empty))


def detect_nr_pss(
    capture: NDArray[np.complexfloating],
    *,
    fft_size: int,
    cyclic_prefix_length: int,
    sampling_frequency_hz: float,
    noise_variance: float | Sequence[float] | NDArray[np.generic] = 1.0,
    cfo_hypotheses_hz: tuple[float, ...] = (0.0,),
    fractional_timing_hypotheses_samples: tuple[float, ...] = (0.0,),
    cfo_guard_samples: int = 0,
    minimum_score: float = 0.3,
) -> PSSDetection:
    """Search timing/CFO hypotheses and refine CFO using repeated PSS CP samples.

    A vector represents one receive chain; a matrix is shaped
    ``[receive_chain, sample]``. Matrix inputs combine inverse-noise-weighted
    matched-filter energy noncoherently across chains, then sum weighted CP
    correlations for CFO refinement. ``noise_variance`` is shared or contains
    one independent complex-sample variance per receive chain.

    Integer timing is searched across the capture and fractional timing across
    ``fractional_timing_hypotheses_samples`` in ``[-0.5, 0.5)``. The best
    coarse timing is refined by bounded scalar optimization of the normalized
    PSS matched-filter score over the continuous fractional-delay interval.
    The finite CFO grid supplies the PSS matched-filter hypothesis; the
    returned CFO is then refined from repeated samples in the PSS cyclic prefix
    and is unambiguous only within ``±sampling_frequency_hz/(2*fft_size)``.
    ``cfo_guard_samples`` skips CP samples contaminated by a known channel-delay
    upper bound.
    """
    _validate_ofdm(fft_size, cyclic_prefix_length)
    samples = np.asarray(capture, dtype=np.complex128)
    if samples.ndim not in (1, 2) or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("capture must be a finite complex vector or receive-chain matrix")
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.shape[0] == 0 or samples.shape[1] < fft_size + cyclic_prefix_length:
        raise ValueError("capture is shorter than one PSS OFDM symbol")
    variances = _noise_variance_by_chain(noise_variance, samples.shape[0])
    inverse_variances = 1.0 / variances
    if not math.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    if not math.isfinite(minimum_score) or not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be finite and in [0, 1]")
    if not cfo_hypotheses_hz:
        raise ValueError("at least one CFO hypothesis is required")
    if any(not math.isfinite(float(value)) for value in cfo_hypotheses_hz):
        raise ValueError("CFO hypotheses must be finite")
    if (
        not fractional_timing_hypotheses_samples
        or any(
            not math.isfinite(float(value)) or not -0.5 <= value < 0.5
            for value in fractional_timing_hypotheses_samples
        )
        or len(set(fractional_timing_hypotheses_samples))
        != len(fractional_timing_hypotheses_samples)
    ):
        raise ValueError("fractional timing hypotheses must be unique values in [-0.5, 0.5)")
    if isinstance(cfo_guard_samples, bool) or not isinstance(cfo_guard_samples, int):
        raise TypeError("cfo_guard_samples must be an integer")
    if not 0 <= cfo_guard_samples < cyclic_prefix_length:
        raise ValueError("cfo_guard_samples must be in [0, cyclic_prefix_length)")

    symbol_length = fft_size + cyclic_prefix_length
    capture_energy = np.stack(
        [
            np.convolve(np.abs(chain) ** 2, np.ones(symbol_length, dtype=np.float64), mode="valid")
            for chain in samples
        ],
        axis=0,
    )
    best: PSSDetection | None = None
    sample_axis = np.arange(symbol_length, dtype=np.float64)
    for nid2 in range(3):
        for fractional_delay in fractional_timing_hypotheses_samples:
            base = modulate_nr_sync_symbol(
                nr_pss_sequence(nid2),
                fft_size=fft_size,
                cyclic_prefix_length=cyclic_prefix_length,
                fractional_delay_samples=float(fractional_delay),
            )
            for cfo_hz in cfo_hypotheses_hz:
                template = base * np.exp(
                    2j * np.pi * float(cfo_hz) * sample_axis / sampling_frequency_hz
                )
                correlations = np.stack(
                    [correlate(chain, template, mode="valid", method="fft") for chain in samples],
                    axis=0,
                )
                correlation_power = np.sum(
                    np.square(np.abs(correlations)) * inverse_variances[:, None], axis=0
                )
                denominator = np.sqrt(
                    np.sum(capture_energy * inverse_variances[:, None], axis=0)
                    * np.vdot(template, template).real
                )
                scores = np.divide(
                    np.sqrt(correlation_power),
                    denominator,
                    out=np.zeros_like(capture_energy[0]),
                    where=denominator > np.finfo(np.float64).tiny,
                )
                location = int(np.argmax(scores))
                score = float(scores[location])
                if best is None or score > best.normalized_score:
                    best = PSSDetection(
                        location,
                        nid2,
                        float(cfo_hz),
                        score,
                        float(fractional_delay),
                    )
    assert best is not None
    if best.normalized_score < minimum_score:
        raise NRSyncNotFoundError(
            f"best PSS score {best.normalized_score:.3f} is below minimum {minimum_score:.3f}"
        )
    coarse_best = best
    matched_samples = samples[
        :, coarse_best.sample_offset : coarse_best.sample_offset + symbol_length
    ]
    matched_energy = float(np.sum(np.square(np.abs(matched_samples)) * inverse_variances[:, None]))

    def negative_fractional_score(fractional_delay: float) -> float:
        candidate = modulate_nr_sync_symbol(
            nr_pss_sequence(coarse_best.nid2),
            fft_size=fft_size,
            cyclic_prefix_length=cyclic_prefix_length,
            fractional_delay_samples=fractional_delay,
        )
        candidate *= np.exp(2j * np.pi * coarse_best.cfo_hz * sample_axis / sampling_frequency_hz)
        candidate_energy = float(np.vdot(candidate, candidate).real)
        denominator = math.sqrt(matched_energy * candidate_energy)
        if denominator <= np.finfo(np.float64).tiny:
            return 0.0
        correlations = np.asarray(
            [np.vdot(candidate, chain) for chain in matched_samples], dtype=np.complex128
        )
        combined_correlation = math.sqrt(
            float(np.sum(np.square(np.abs(correlations)) * inverse_variances))
        )
        return -float(combined_correlation / denominator)

    refined = minimize_scalar(
        negative_fractional_score,
        bounds=(-0.5, 0.5),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 100},
    )
    if refined.success and math.isfinite(float(refined.fun)):
        refined_fraction = float(refined.x)
        refined_score = -float(refined.fun)
        if refined_score >= coarse_best.normalized_score:
            best = PSSDetection(
                coarse_best.sample_offset,
                coarse_best.nid2,
                coarse_best.cfo_hz,
                refined_score,
                refined_fraction,
            )
    prefix_start = best.sample_offset + cfo_guard_samples
    prefix_end = best.sample_offset + cyclic_prefix_length
    tail_start = best.sample_offset + fft_size + cfo_guard_samples
    tail_end = best.sample_offset + fft_size + cyclic_prefix_length
    cp_correlation = sum(
        inverse_variances[chain_index]
        * np.vdot(chain[prefix_start:prefix_end], chain[tail_start:tail_end])
        for chain_index, chain in enumerate(samples)
    )
    if abs(cp_correlation) <= np.finfo(np.float64).tiny:
        raise NRSyncNotFoundError(
            "PSS cyclic-prefix correlation has insufficient energy for CFO refinement"
        )
    refined_cfo_hz = (
        float(np.angle(cp_correlation)) * sampling_frequency_hz / (2.0 * np.pi * fft_size)
    )
    return PSSDetection(
        best.sample_offset,
        best.nid2,
        refined_cfo_hz,
        best.normalized_score,
        best.fractional_sample_offset,
    )


def decode_nr_sss_symbol(
    symbol_with_cp: NDArray[np.complexfloating],
    *,
    nid2: int,
    fft_size: int,
    cyclic_prefix_length: int,
    noise_variance: float | Sequence[float] | NDArray[np.generic] = 1.0,
    channel_estimate: NDArray[np.complexfloating] | None = None,
) -> tuple[int, float]:
    """Identify N_ID^(1), optionally using channel response from the PSS symbol.

    Matrix inputs use ``[receive_chain, sample]`` and combine candidate
    correlation energy after inverse-noise weighting, without requiring a
    common complex phase. ``noise_variance`` is shared or per chain.
    """
    if isinstance(nid2, bool) or not isinstance(nid2, int) or not 0 <= nid2 <= 2:
        raise ValueError("nid2 must be an integer in [0, 2]")
    _validate_ofdm(fft_size, cyclic_prefix_length)
    samples = np.asarray(symbol_with_cp, dtype=np.complex128)
    expected_length = fft_size + cyclic_prefix_length
    if samples.ndim not in (1, 2) or not np.isfinite(samples).all():
        raise ValueError("symbol_with_cp must be one finite complex symbol or receive-chain matrix")
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.shape[0] == 0 or samples.shape[1] != expected_length:
        raise ValueError("symbol_with_cp must contain one OFDM symbol including its CP")
    variances = _noise_variance_by_chain(noise_variance, samples.shape[0])
    inverse_variances = 1.0 / variances
    spectrum = np.fft.fft(samples[:, cyclic_prefix_length:], axis=-1)
    offsets = np.arange(-64, 63)
    observed = spectrum[:, offsets % fft_size]
    energy_squared = float(np.sum(np.square(np.abs(observed)) * inverse_variances[:, None]))
    energy = math.sqrt(energy_squared)
    if energy <= np.finfo(np.float64).tiny:
        raise ValueError("SSS symbol has insufficient energy")
    channel = (
        np.ones((samples.shape[0], 127), dtype=np.complex128)
        if channel_estimate is None
        else np.asarray(channel_estimate, dtype=np.complex128)
    )
    if channel.ndim == 1:
        channel = channel[None, :]
    if channel.shape != (samples.shape[0], 127) or not np.isfinite(channel).all():
        raise ValueError("channel_estimate must have shape (127,) or (receive_chains, 127)")
    channel_power = np.square(np.abs(channel)).sum(axis=1)
    if np.any(channel_power <= np.finfo(np.float64).tiny):
        raise ValueError("channel_estimate has insufficient energy")
    hypotheses = np.stack([nr_sss_sequence(nid1, nid2) for nid1 in range(336)])
    templates = hypotheses[None, :, :] * channel[:, None, :]
    correlations = np.einsum("rck,rk->rc", np.conj(templates), observed)
    projected_energy = np.square(np.abs(correlations)) / channel_power[:, None]
    weighted_projection = np.sum(projected_energy * inverse_variances[:, None], axis=0)
    scores = np.sqrt(weighted_projection / energy_squared)
    nid1 = int(np.argmax(scores))
    return nid1, float(scores[nid1])


def acquire_nr_cell(
    capture: NDArray[np.complexfloating],
    *,
    fft_size: int,
    cyclic_prefix_length: int,
    sampling_frequency_hz: float,
    noise_variance: float | Sequence[float] | NDArray[np.generic] = 1.0,
    cfo_hypotheses_hz: tuple[float, ...] = (0.0,),
    fractional_timing_hypotheses_samples: tuple[float, ...] = (0.0,),
    cfo_guard_samples: int = 0,
    minimum_pss_score: float = 0.3,
    minimum_sss_score: float = 0.4,
) -> NRCellDetection:
    """Acquire integer PSS timing and the 1008-cell identity from PSS/SSS.

    The capture may be one vector or a ``[receive_chain, sample]`` matrix.
    ``noise_variance`` may be shared or specify one independent complex-sample
    variance per receive chain; PSS/SSS scores and CP-based CFO refinement use
    the corresponding inverse-noise weights.

    Numerology is configured, CFO is searched only on the supplied grid, and
    The SSS decoder reuses a PSS-derived per-subcarrier channel estimate,
    assuming a static channel over the PSS-to-SSS interval. This is a controlled
    cell-search reference, not a complete or standards-certified NR synchronizer.
    """
    _validate_ofdm(fft_size, cyclic_prefix_length)
    if not math.isfinite(minimum_sss_score) or not 0.0 <= minimum_sss_score <= 1.0:
        raise ValueError("minimum_sss_score must be finite and in [0, 1]")
    samples = np.asarray(capture, dtype=np.complex128)
    if samples.ndim not in (1, 2) or not np.isfinite(samples).all():
        raise ValueError("capture must be a finite complex vector or receive-chain matrix")
    if samples.ndim == 1:
        samples = samples[None, :]
    variances = _noise_variance_by_chain(noise_variance, samples.shape[0])
    symbol_length = fft_size + cyclic_prefix_length
    if samples.shape[0] == 0 or samples.shape[1] < 3 * symbol_length:
        raise ValueError("capture is too short to contain both the PSS and the later SSS symbol")
    pss = detect_nr_pss(
        # Exclude candidate PSS positions too late to leave room for SSS at
        # SS/PBCH symbol index 2. Searching the trailing SSS/PBCH symbols as
        # extra PSS timing hypotheses creates false late acquisitions.
        samples[:, : samples.shape[1] - 2 * symbol_length],
        fft_size=fft_size,
        cyclic_prefix_length=cyclic_prefix_length,
        sampling_frequency_hz=sampling_frequency_hz,
        noise_variance=variances,
        cfo_hypotheses_hz=cfo_hypotheses_hz,
        fractional_timing_hypotheses_samples=fractional_timing_hypotheses_samples,
        cfo_guard_samples=cfo_guard_samples,
        minimum_score=minimum_pss_score,
    )
    sss_start = pss.sample_offset + 2 * symbol_length
    sss_end = sss_start + symbol_length
    if sss_end > samples.shape[1]:
        raise ValueError("capture ends before the SSS symbol")
    pss_start = pss.sample_offset
    pss_end = pss_start + symbol_length
    phase = np.exp(-2j * np.pi * pss.cfo_hz * np.arange(symbol_length) / sampling_frequency_hz)
    corrected_pss = samples[:, pss_start:pss_end] * phase[None, :]
    pss_spectrum = np.fft.fft(corrected_pss[:, cyclic_prefix_length:], axis=-1)
    offsets = np.arange(-64, 63)
    pss_observed = pss_spectrum[:, offsets % fft_size]
    channel_estimate: NDArray[np.complex128] = np.asarray(
        pss_observed / nr_pss_sequence(pss.nid2)[None, :], dtype=np.complex128
    )
    corrected = samples[:, sss_start:sss_end] * phase[None, :]
    nid1, sss_score = decode_nr_sss_symbol(
        corrected,
        nid2=pss.nid2,
        fft_size=fft_size,
        cyclic_prefix_length=cyclic_prefix_length,
        noise_variance=variances,
        channel_estimate=channel_estimate,
    )

    if sss_score < minimum_sss_score:
        raise NRSyncNotFoundError(
            f"best SSS score {sss_score:.3f} is below minimum {minimum_sss_score:.3f}"
        )
    return NRCellDetection(
        sample_offset=pss.sample_offset,
        nid1=nid1,
        nid2=pss.nid2,
        physical_cell_id=3 * nid1 + pss.nid2,
        cfo_hz=pss.cfo_hz,
        pss_score=pss.normalized_score,
        sss_score=sss_score,
        fractional_sample_offset=pss.fractional_sample_offset,
    )


def _noise_variance_by_chain(value: object, num_receive_chains: int) -> NDArray[np.float64]:
    """Expand a shared complex-noise variance or validate per-chain values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("noise_variance must contain real numeric values, not booleans")
    if isinstance(value, Real):
        variances = np.full(num_receive_chains, float(value), dtype=np.float64)
    else:
        object_values = np.asarray(value, dtype=object)
        if any(isinstance(item, (bool, np.bool_)) for item in object_values.flat):
            raise TypeError("noise_variance must contain real numeric values, not booleans")
        variances = np.asarray(value)
        if variances.shape != (num_receive_chains,):
            raise ValueError("noise_variance must be scalar or contain one value per receive chain")
        if variances.dtype.kind not in "iuf":
            raise TypeError("noise_variance values must be real numbers")
        variances = np.asarray(variances, dtype=np.float64)
    if not np.isfinite(variances).all() or np.any(variances <= 0.0):
        raise ValueError("noise_variance values must be finite and positive")
    return variances
