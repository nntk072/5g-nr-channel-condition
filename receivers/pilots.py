"""Nonideal pilot-based CFR observation (NOT a standards-compliant NR DMRS receiver).

An imperfect estimator is substantially more defensible than feeding true CIR
path coefficients or geometry to the classifier. For publication claims about
PUSCH/DMRS, replace this diagnostic comb with a validated Sionna 2.2 NR receiver.
"""

from __future__ import annotations

import numpy as np


def comb_ls_estimate(
    h: np.ndarray, *, spacing: int = 4, snr_db: float = 12.0, seed: int = 27
) -> np.ndarray:
    """Transmit known QPSK pilots, AWGN, LS at pilots + linear interpolation.

    Input: true channel frequency response [N, links, F]. Output: noisy estimated
    complex CFR with exactly the same dimensions. Pilot pattern is NOT 38.211.
    """
    h = np.asarray(h)
    if h.ndim != 3 or not np.iscomplexobj(h) or spacing < 1 or spacing > h.shape[-1] // 2:
        raise ValueError("Invalid CFR or pilot spacing")
    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite")
    rng = np.random.default_rng(seed)
    indices = np.arange(0, h.shape[-1], spacing)
    symbols = (
        rng.choice([-1.0, 1.0], size=len(indices)) + 1j * rng.choice([-1.0, 1.0], size=len(indices))
    ) / np.sqrt(2)
    pilot_channel = h[..., indices]
    rx = pilot_channel * symbols[None, None, :]
    # The SNR parameter is defined against average received pilot power per link.
    var = np.maximum(np.mean(np.abs(rx) ** 2, axis=-1, keepdims=True), 1e-24) * 10 ** (-snr_db / 10)
    rx += np.sqrt(var / 2) * (rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape))
    measured = rx / symbols[None, None, :]
    t = np.arange(h.shape[-1])
    flat = measured.reshape(-1, len(indices))
    out = np.empty((len(flat), len(t)), dtype=np.complex64)
    for i, line in enumerate(flat):
        out[i] = np.interp(t, indices, line.real) + 1j * np.interp(t, indices, line.imag)
    return out.reshape(h.shape)
