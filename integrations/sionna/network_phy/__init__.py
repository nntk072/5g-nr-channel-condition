"""Sionna network PHY: records, composition, and link operations."""
# ruff: noqa: F401, E402

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .operations import (
    add_independent_receiver_awgn,
    add_independent_receiver_waveform_awgn,
    apply_network_link_budget,
    apply_network_transmitter_iq_imbalance,
    compose_downlink_resource_grids,
    compose_received_waveforms,
    network_resource_sinr,
    network_sinr,
)
from .records import (
    _NETWORK_FRACTIONAL_DELAY_KERNEL_RADIUS,
    NetworkGridComposition,
    NetworkLinkBudget,
    NetworkMIMOChannelPropagation,
    NetworkResourceSINR,
    NetworkSINR,
    NetworkTransmitterWaveform,
    NetworkWaveformComposition,
    NoisyNetworkGrid,
    NoisyNetworkWaveform,
    _contains_boolean,
    _fractional_delay_capture_slice,
    _mean_complex_power,
    _shared_phase_noise_processes,
    _validate_composition,
    _validated_nonnegative_vector,
    _validated_real_values,
    propagate_network_transmit_waveforms,
)

__all__ = [
    "NetworkGridComposition",
    "NetworkLinkBudget",
    "NetworkMIMOChannelPropagation",
    "NetworkResourceSINR",
    "NetworkSINR",
    "NetworkTransmitterWaveform",
    "NetworkWaveformComposition",
    "NoisyNetworkGrid",
    "NoisyNetworkWaveform",
    "_NETWORK_FRACTIONAL_DELAY_KERNEL_RADIUS",
    "_contains_boolean",
    "_fractional_delay_capture_slice",
    "_mean_complex_power",
    "_shared_phase_noise_processes",
    "_validate_composition",
    "_validated_nonnegative_vector",
    "_validated_real_values",
    "propagate_network_transmit_waveforms",
    "add_independent_receiver_awgn",
    "add_independent_receiver_waveform_awgn",
    "apply_network_link_budget",
    "apply_network_transmitter_iq_imbalance",
    "compose_downlink_resource_grids",
    "compose_received_waveforms",
    "network_resource_sinr",
    "network_sinr",
]
