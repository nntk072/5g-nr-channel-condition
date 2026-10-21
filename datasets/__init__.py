"""Versioned channel dataset contracts and dataset generators."""

from .channel import ChannelDataset
from .nr import NRChannelRecord, NRModelInputs, NRObservation
from .synthetic import debug_multipath

__all__ = ["ChannelDataset", "NRChannelRecord", "NRModelInputs", "NRObservation", "debug_multipath"]
