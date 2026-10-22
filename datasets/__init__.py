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
    read_frequency_model_inputs,
    read_receiver_model_inputs,
    write_episode_bundle,
)
from .network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodePoseTimeline,
    NodeRole,
)
from .nr import (
    LinkedNRObservation,
    NRChannelRecord,
    NRModelInputs,
    NRObservation,
    NRTrajectory,
    TargetDefinition,
)
from .nr_store import (
    NRStoredRecord,
    NRStoredTrajectory,
    read_nr_record,
    read_nr_trajectory,
    write_nr_record,
    write_nr_trajectory,
)
from .observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)
from .path_oracle import PathOracleTable, PathSolveStatus
from .provenance import UMCDRunProvenance
from .splits import (
    EpisodeSplitRegistry,
    audit_episode_split_registry,
    freeze_episode_split_registry,
    make_episode_split_registry,
    read_episode_split_registry,
)
from .synthetic import debug_multipath
from .targets import TargetValidity, TargetValue, definition_digest

__all__ = [
    "ChannelDataset",
    "ChannelDelayReference",
    "ChannelPowerScale",
    "PhysicalChannelView",
    "EpisodeBundle",
    "LinkedNRObservation",
    "StoredEpisodeBundle",
    "AntennaArraySpec",
    "DirectedLink",
    "FrequencyDomainModelInputs",
    "FrequencyDomainObservation",
    "LinkedFrequencyDomainObservation",
    "LinkDirection",
    "LinkRole",
    "NetworkEpisode",
    "NetworkNode",
    "NodePoseTimeline",
    "NodeRole",
    "PathOracleTable",
    "PathSolveStatus",
    "EpisodeSplitRegistry",
    "UMCDRunProvenance",
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
    "read_frequency_model_inputs",
    "read_receiver_model_inputs",
    "audit_episode_split_registry",
    "freeze_episode_split_registry",
    "make_episode_split_registry",
    "read_episode_split_registry",
    "write_episode_bundle",
    "NRStoredTrajectory",
    "debug_multipath",
    "generate_analytical_multipath",
    "read_nr_record",
    "read_nr_trajectory",
    "write_nr_record",
    "write_nr_trajectory",
]
