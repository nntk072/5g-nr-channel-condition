"""Versioned channel dataset contracts and dataset generators."""

from .analytical import (
    AnalyticalModelInputs,
    AnalyticalMultipathSample,
    AnalyticalPathOracle,
    generate_analytical_multipath,
)
from .channel import ChannelDataset
from .channel_view import ChannelDelayReference, ChannelPowerScale, PhysicalChannelView
from .episode_store import (
    EpisodeBundle,
    StoredEpisodeBundle,
    read_episode_bundle,
    write_episode_bundle,
)
from .network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodeRole,
)
from .nr import NRChannelRecord, NRModelInputs, NRObservation, NRTrajectory, TargetDefinition
from .nr_store import (
    NRStoredRecord,
    NRStoredTrajectory,
    read_nr_record,
    read_nr_trajectory,
    write_nr_record,
    write_nr_trajectory,
)
from .path_oracle import PathOracleTable, PathSolveStatus
from .synthetic import debug_multipath
from .targets import TargetValidity, TargetValue, definition_digest

__all__ = [
    "ChannelDataset",
    "ChannelDelayReference",
    "ChannelPowerScale",
    "PhysicalChannelView",
    "EpisodeBundle",
    "StoredEpisodeBundle",
    "AntennaArraySpec",
    "DirectedLink",
    "LinkDirection",
    "LinkRole",
    "NetworkEpisode",
    "NetworkNode",
    "NodeRole",
    "PathOracleTable",
    "PathSolveStatus",
    "AnalyticalMultipathSample",
    "AnalyticalModelInputs",
    "AnalyticalPathOracle",
    "NRChannelRecord",
    "NRModelInputs",
    "NRObservation",
    "NRStoredRecord",
    "NRTrajectory",
    "TargetDefinition",
    "TargetValidity",
    "TargetValue",
    "definition_digest",
    "read_episode_bundle",
    "write_episode_bundle",
    "NRStoredTrajectory",
    "debug_multipath",
    "generate_analytical_multipath",
    "read_nr_record",
    "read_nr_trajectory",
    "write_nr_record",
    "write_nr_trajectory",
]
