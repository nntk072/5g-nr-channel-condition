"""Unprivileged observables and mathematical channel transformations."""

from __future__ import annotations

import numpy as np


def frequency_response(a: np.ndarray, tau: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """CFR H[k] = sum_p a[p] exp(-j2 pi f[k] tau[p]).

    a: [..., paths] complex; tau: [..., paths] broadcast compatible;
    frequencies: [F] baseband frequencies (Hz). No power normalization/labeling.
    """
    a = np.asarray(a)
    tau = np.asarray(tau, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if a.ndim < 1 or not np.iscomplexobj(a) or tau.shape != a.shape:
        raise ValueError("a and tau must be identically shaped, with complex a")
    if frequencies.ndim != 1 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies must be finite 1D")
    return np.sum(a[..., :, None] * np.exp(-2j * np.pi * tau[..., :, None] * frequencies), axis=-2)


def subcarrier_frequencies(n_subcarriers: int, spacing_hz: float) -> np.ndarray:
    if n_subcarriers < 8 or spacing_hz <= 0:
        raise ValueError("invalid OFDM grid")
    return (np.arange(n_subcarriers, dtype=np.float64) - n_subcarriers // 2) * spacing_hz


def add_estimation_noise(h: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex channel-estimation error at specified per-example SNR.

    This is *not* a standardized PUSCH/DMRS LS estimator or complete 5G receiver.
    """
    arr = np.asarray(h)
    if not np.iscomplexobj(arr):
        raise ValueError("complex CFR required")
    if not np.isfinite(snr_db):
        raise ValueError("finite SNR required")
    power = np.mean(np.abs(arr) ** 2, axis=(-1, -2), keepdims=True)
    noise_var = np.maximum(power, 1e-24) * 10 ** (-snr_db / 10)
    noise = (rng.standard_normal(arr.shape) + 1j * rng.standard_normal(arr.shape)) * np.sqrt(
        noise_var / 2
    )
    return (arr + noise).astype(np.complex64)


def observable_features(h: np.ndarray, *, include_phase: bool = False) -> np.ndarray:
    """Stable, per-example amplitude-invariant CFO-agnostic features.

    Channels are not identified by label, simulator name, path kind or location.
    The default uses normalized |H| + time-domain |IFFT(H)| PDP and moments.
    Complex phase is optional; raw phase is sensitive to synchronization.
    """
    h = np.asarray(h)
    if h.ndim != 3 or not np.iscomplexobj(h):
        raise ValueError("expected [samples, antenna links, subcarriers] complex")
    power = np.mean(np.abs(h) ** 2, axis=(1, 2), keepdims=True)
    h_norm = h / np.sqrt(np.maximum(power, 1e-24))
    amps = np.log1p(np.abs(h_norm))
    pdp = np.abs(np.fft.ifft(h_norm, axis=-1)) ** 2
    pdp /= np.maximum(pdp.sum(axis=-1, keepdims=True), 1e-24)
    bins = np.arange(h.shape[-1], dtype=np.float64)
    mean = (pdp * bins).sum(-1)
    spread = np.sqrt(((bins[None, None, :] - mean[:, :, None]) ** 2 * pdp).sum(-1))
    entropy = -(pdp * np.log(np.maximum(pdp, 1e-12))).sum(-1)
    peak = pdp.max(-1)
    moments = np.stack([mean, spread, entropy, peak], axis=-1)
    moments[:, :, 0:2] /= h.shape[-1]
    moments[:, :, 2] /= np.log(h.shape[-1])
    parts = [
        amps.reshape(len(h), -1),
        np.log1p(pdp * float(h.shape[-1])).reshape(len(h), -1),
        moments.reshape(len(h), -1),
    ]
    if include_phase:
        parts += [h_norm.real.reshape(len(h), -1), h_norm.imag.reshape(len(h), -1)]
    output = np.column_stack(parts).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("nonfinite observables")
    return output


def delay_k_proxy_features(h: np.ndarray) -> np.ndarray:
    """Return receiver-visible normalized delay/PDP proxies per antenna link.

    Delay moments are expressed in fractions of the CFR record length. The
    dominant-to-remainder power ratio is only a coarse K-like proxy: the
    strongest circular-IFFT bin is not guaranteed to be a physical LOS path,
    and no carrier, pilot-estimator, or path-oracle information is used.
    """
    observed = np.asarray(h)
    if observed.ndim != 3 or not np.iscomplexobj(observed):
        raise ValueError("expected [samples, antenna links, subcarriers] complex CFR")
    if observed.shape[0] < 1 or observed.shape[1] < 1 or observed.shape[2] < 8:
        raise ValueError("delay/K proxy requires nonempty CFR rows with at least eight tones")
    if not np.isfinite(observed).all():
        raise ValueError("delay/K proxy requires finite CFR values")
    observed = np.asarray(observed, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):
        power = np.mean(np.abs(observed) ** 2, axis=(1, 2), keepdims=True)
    if not np.isfinite(power).all():
        raise ValueError("delay/K proxy CFR power is not representable in float64")
    normalized = observed / np.sqrt(np.maximum(power, 1e-24))
    # Inputs follow the centered subcarrier grid from subcarrier_frequencies.
    # Undo that centering before IFFT and center the resulting delay bins.
    pdp = (
        np.abs(
            np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(normalized, axes=-1), axis=-1), axes=-1)
        )
        ** 2
    )
    pdp /= np.maximum(pdp.sum(axis=-1, keepdims=True), 1e-24)
    delay = (
        np.arange(observed.shape[-1], dtype=np.float64) - observed.shape[-1] // 2
    ) / observed.shape[-1]
    mean_delay = np.sum(pdp * delay[None, None, :], axis=-1)
    rms_delay = np.sqrt(np.sum((delay[None, None, :] - mean_delay[..., None]) ** 2 * pdp, axis=-1))
    peak_fraction = np.max(pdp, axis=-1)
    remainder_fraction = np.maximum(1.0 - peak_fraction, 1e-12)
    dominant_to_remainder_db = 10.0 * np.log10(
        np.maximum(peak_fraction, 1e-12) / remainder_fraction
    )
    entropy = -np.sum(pdp * np.log(np.maximum(pdp, 1e-12)), axis=-1)
    entropy /= np.log(observed.shape[-1])
    features = np.stack(
        [mean_delay, rms_delay, peak_fraction, dominant_to_remainder_db, entropy], axis=-1
    ).reshape(observed.shape[0], -1)
    result = features.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("delay/K proxy produced nonfinite features")
    return result
