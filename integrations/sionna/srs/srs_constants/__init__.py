# ruff: noqa: F401
"""Public SRS constants and resource-grid types."""
# ruff: noqa: F401, E402

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from numbers import Integral, Real
from typing import cast

import numpy as np

from .channel_estimates import (
    NRSRSMultiportChannelEstimate,
    NRSRSSinglePortChannelEstimate,
)
from .constants import (
    _MAX_SRS_INTERCELL_AUDIT_COORDINATES,
    _MAX_SRS_INTERCELL_AUDIT_RESOURCE_ELEMENTS,
    _SRS_BANDWIDTH_TABLE,
    _SRS_COMB_SIZES,
    _SRS_CYCLIC_SHIFT_COUNTS,
    _SRS_PERIODICITY_EXT_OFFSETS,
    _SRS_PERIODICITY_OFFSETS,
    _SRS_POSITIONING_SYMBOL_OFFSETS,
    _SRS_REQUEST_DCI_FORMATS,
    _SRS_SYMBOL_COUNTS,
)
from .dci_0_3 import (
    _DCI_0_3_CONFIG_DERIVED_FIELDS,
    _DCI_0_3_FIELD_ORDER,
    _nr_srs_dci_0_3_antenna_ports_width,
    _nr_srs_dci_0_3_pdcch_monitoring_adaptation_width,
    _nr_srs_dci_0_3_precoding_width,
    _nr_srs_dci_0_3_scell_dormancy_width,
    _nr_srs_dci_0_3_sri_width,
    nr_srs_aperiodic_target_slot_index,
    nr_srs_dci_0_3_frequency_assignment_widths,
    nr_srs_dci_0_3_request_list_index_from_bits,
    nr_srs_dci_0_3_scheduled_cell_ids,
)
from .ports_and_bwp import (
    NRSRSBWPFrequencyBounds,
    NRSRSBWPSelection,
    NRSRSPortConfiguration,
)
from .resource_grids import (
    NRSRSHoppedMultiportResourceGrid,
    NRSRSHoppedResourceGrid,
    NRSRSMultiportResourceGrid,
    NRSRSPositioningMultiportResourceGrid,
    NRSRSPositioningResourceGrid,
    NRSRSResourceGrid,
)
from .sequences import (
    _LOW_PAPR_SHORT_PHASES,
    _largest_prime_below,
    _normalized_multiport_beta,
    _nr_gold_bits,
    _srs_gold_bit_offset,
    _validate_srs_hopping_position,
)

__all__ = [
    "_MAX_SRS_INTERCELL_AUDIT_COORDINATES",
    "_MAX_SRS_INTERCELL_AUDIT_RESOURCE_ELEMENTS",
    "_SRS_BANDWIDTH_TABLE",
    "_SRS_COMB_SIZES",
    "_SRS_CYCLIC_SHIFT_COUNTS",
    "_SRS_PERIODICITY_EXT_OFFSETS",
    "_SRS_PERIODICITY_OFFSETS",
    "_SRS_POSITIONING_SYMBOL_OFFSETS",
    "_SRS_REQUEST_DCI_FORMATS",
    "_SRS_SYMBOL_COUNTS",
    "_DCI_0_3_CONFIG_DERIVED_FIELDS",
    "_DCI_0_3_FIELD_ORDER",
    "_nr_srs_dci_0_3_antenna_ports_width",
    "_nr_srs_dci_0_3_pdcch_monitoring_adaptation_width",
    "_nr_srs_dci_0_3_precoding_width",
    "_nr_srs_dci_0_3_scell_dormancy_width",
    "_nr_srs_dci_0_3_sri_width",
    "nr_srs_aperiodic_target_slot_index",
    "nr_srs_dci_0_3_frequency_assignment_widths",
    "nr_srs_dci_0_3_request_list_index_from_bits",
    "nr_srs_dci_0_3_scheduled_cell_ids",
    "NRSRSBWPFrequencyBounds",
    "NRSRSBWPSelection",
    "NRSRSPortConfiguration",
    "_LOW_PAPR_SHORT_PHASES",
    "_largest_prime_below",
    "_normalized_multiport_beta",
    "_nr_gold_bits",
    "_srs_gold_bit_offset",
    "_validate_srs_hopping_position",
    "NRSRSHoppedMultiportResourceGrid",
    "NRSRSHoppedResourceGrid",
    "NRSRSMultiportResourceGrid",
    "NRSRSPositioningMultiportResourceGrid",
    "NRSRSPositioningResourceGrid",
    "NRSRSResourceGrid",
    "NRSRSMultiportChannelEstimate",
    "NRSRSSinglePortChannelEstimate",
]
