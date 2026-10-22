"""Receiver-visible frequency-domain measurements outside a full NR waveform model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class FrequencyDomainModelInputs:
    """Inference-only complex CFR observations with an explicit AWGN model."""

    observed_cfr: np.ndarray
    noise_variance: float
    snr_db: float
    carrier_hz: float
    subcarrier_offsets_hz: np.ndarray

    def __post_init__(self) -> None:
        channel = np.array(self.observed_cfr, copy=True)
        if (
            channel.ndim != 3
            or not np.iscomplexobj(channel)
            or min(channel.shape) < 1
            or not np.isfinite(channel).all()
        ):
            raise ValueError("observed_cfr must be finite complex [rx_port, tx_port, subcarrier]")
        offsets = np.array(self.subcarrier_offsets_hz, dtype=np.float64, copy=True)
        if (
            offsets.shape != (channel.shape[-1],)
            or not np.isfinite(offsets).all()
            or np.any(np.diff(offsets) <= 0)
            or not np.any(offsets == 0.0)
        ):
            raise ValueError("subcarrier offsets must be finite, increasing, and include DC")
        if not math.isfinite(self.noise_variance) or self.noise_variance < 0:
            raise ValueError("noise_variance must be finite and nonnegative")
        if not math.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite")
        if not math.isfinite(self.carrier_hz) or self.carrier_hz <= 0:
            raise ValueError("carrier_hz must be finite and positive")
        channel.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "observed_cfr", channel)
        object.__setattr__(self, "subcarrier_offsets_hz", offsets)


@dataclass(frozen=True, slots=True, eq=False)
class FrequencyDomainObservation:
    """One direct noisy-CFR measurement; this is not a PUSCH/DMRS observation."""

    sample_id: str
    model_inputs: FrequencyDomainModelInputs

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.model_inputs, FrequencyDomainModelInputs):
            raise TypeError("model_inputs must be FrequencyDomainModelInputs")


@dataclass(frozen=True, slots=True)
class LinkedFrequencyDomainObservation:
    """Bind a frequency-domain measurement to one link and episode frame."""

    link_id: str
    frame_index: int
    observation: FrequencyDomainObservation

    def __post_init__(self) -> None:
        if not isinstance(self.link_id, str) or not self.link_id.strip():
            raise ValueError("link_id must be a non-empty string")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if not isinstance(self.observation, FrequencyDomainObservation):
            raise TypeError("observation must be a FrequencyDomainObservation")
