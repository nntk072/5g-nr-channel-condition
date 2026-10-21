"""Versioned channel dataset contracts and dataset generators."""

from .channel import ChannelDataset
from .synthetic import debug_multipath

__all__ = ["ChannelDataset", "debug_multipath"]
