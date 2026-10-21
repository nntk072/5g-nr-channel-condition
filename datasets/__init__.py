"""Versioned channel dataset contracts and dataset generators."""

from .analytical import (
    AnalyticalModelInputs,
    AnalyticalMultipathSample,
    AnalyticalPathOracle,
    generate_analytical_multipath,
)
from .channel import ChannelDataset
from .nr import NRChannelRecord, NRModelInputs, NRObservation
from .nr_store import NRStoredRecord, read_nr_record, write_nr_record
from .synthetic import debug_multipath

__all__ = [
    "ChannelDataset",
    "AnalyticalMultipathSample",
    "AnalyticalModelInputs",
    "AnalyticalPathOracle",
    "NRChannelRecord",
    "NRModelInputs",
    "NRObservation",
    "NRStoredRecord",
    "debug_multipath",
    "generate_analytical_multipath",
    "read_nr_record",
    "write_nr_record",
]
