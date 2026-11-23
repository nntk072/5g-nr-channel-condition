"""Dataclass grids for NR PUSCH intra-slot hopping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class NRPUSCHFrequencyHopGrid:
    """Carrier-grid composition of two caller-mapped intra-slot hop segments."""

    frequency_grid: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHDehoppedGrid:
    """Received carrier grid rehomed onto the first-hop BWP allocation."""

    frequency_grid: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHIntraSlotHopDMRSPlan:
    """Table-derived single-symbol DM-RS positions for two PUSCH hops."""

    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHIntraSlotHopDMRSGrid:
    """Single-port DM-RS reference grid for a replayed two-hop timing plan."""

    pilot_mask: np.ndarray
    pilot_grid: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHHoppedAllocationGrid:
    """Single-layer local-BWP grid with hop-specific DM-RS and mapped data."""

    allocation_grid: np.ndarray
    data_mask: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHHoppedEstimateDemap:
    """Per-call hop-aware LS estimates and data REs in mapper order."""

    received_data_symbols: np.ndarray
    channel_estimate: np.ndarray
    error_variance: np.ndarray
    metadata: Mapping[str, Any]
