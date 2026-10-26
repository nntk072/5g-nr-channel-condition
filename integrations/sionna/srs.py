"""Bounded TS 38.211 SRS sequence and resource-mapping primitives.

This module generates supported type-1 low-PAPR sequences and maps one
single-port resource or caller-scheduled multiport resources into a slot. It
includes per-transmission frequency-tree hopping and positioning-SRS symbol
offsets. Periodic-resource timing can be supplied directly or derived from
the RRC periodicity-and-offset CHOICE. Configured-port cyclic shifts and the
Release 19 eight-port TDM activity pattern are supported. Complete RRC decoding
and unsupported DCI trigger variants remain separate contracts. Positioning
Hyper-SFN gating is applied when resolving periodic SRS transmission counters.
A bounded two-bit typeB DCI request resolver is provided
for selected formats, including decoded multi-cell request-list resolution.
Single- and multiport LS receive estimators preserve resource support; the
multiport estimator also reports conditional noise covariance when supplied
port signatures are full rank. A bounded same-grid audit reports cross-cell
resource-element overlap without inferring interference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

_SRS_SYMBOL_COUNTS = frozenset({1, 2, 4, 8, 10, 12, 14})
_MAX_SRS_INTERCELL_AUDIT_RESOURCE_ELEMENTS = 100_000
_MAX_SRS_INTERCELL_AUDIT_COORDINATES = 1_000
_SRS_COMB_SIZES = frozenset({2, 4, 8})
_SRS_CYCLIC_SHIFT_COUNTS = {2: 8, 4: 12, 8: 6}
_SRS_POSITIONING_SYMBOL_OFFSETS = {
    2: {
        1: (0,),
        2: (0, 1),
        4: (0, 1, 0, 1),
    },
    4: {
        2: (0, 2),
        4: (0, 2, 1, 3),
        8: (0, 2, 1, 3, 0, 2, 1, 3),
        12: (0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3),
    },
    8: {
        4: (0, 4, 2, 6),
        8: (0, 4, 2, 6, 1, 5, 3, 7),
        12: (0, 4, 2, 6, 1, 5, 3, 7, 0, 4, 2, 6),
    },
}

_DCI_0_3_FIELD_ORDER = (
    "identifier_for_dci_formats",
    "scheduled_cell_set_indicator",
    "scheduled_cells_indicator",
    "bandwidth_part_indicator",
    "frequency_domain_resource_assignment",
    "time_domain_resource_assignment",
    "frequency_hopping_flag",
    "modulation_and_coding_scheme",
    "new_data_indicator",
    "redundancy_version",
    "harq_process_number",
    "first_downlink_assignment_index",
    "second_downlink_assignment_index",
    "tpc_command_for_scheduled_pusch",
    "srs_resource_indicator",
    "precoding_information_and_number_of_layers",
    "antenna_ports",
    "srs_request",
    "srs_offset_indicator",
    "csi_request",
    "ptrs_dmrs_association",
    "beta_offset_indicator",
    "dmrs_sequence_initialization",
    "ul_sch_indicator",
    "channel_access_cpext_capc",
    "open_loop_power_control_parameter_set_indication",
    "priority_indicator",
    "minimum_applicable_scheduling_offset_indicator",
    "scell_dormancy_indication",
    "pdcch_monitoring_adaptation_indication",
    "measurement_gap_cancellation",
)
_DCI_0_3_CONFIG_DERIVED_FIELDS = frozenset(
    {
        "identifier_for_dci_formats",
        "scheduled_cell_set_indicator",
        "scheduled_cells_indicator",
        "bandwidth_part_indicator",
        "frequency_domain_resource_assignment",
        "time_domain_resource_assignment",
        "frequency_hopping_flag",
        "dmrs_sequence_initialization",
        "ul_sch_indicator",
        "priority_indicator",
        "minimum_applicable_scheduling_offset_indicator",
        "beta_offset_indicator",
        "channel_access_cpext_capc",
        "tpc_command_for_scheduled_pusch",
        "harq_process_number",
        "modulation_and_coding_scheme",
        "new_data_indicator",
        "redundancy_version",
        "first_downlink_assignment_index",
        "second_downlink_assignment_index",
        "srs_resource_indicator",
        "precoding_information_and_number_of_layers",
        "antenna_ports",
        "srs_request",
        "srs_offset_indicator",
        "csi_request",
        "ptrs_dmrs_association",
        "open_loop_power_control_parameter_set_indication",
        "scell_dormancy_indication",
        "pdcch_monitoring_adaptation_indication",
        "measurement_gap_cancellation",
    }
)
if set(_DCI_0_3_FIELD_ORDER) != _DCI_0_3_CONFIG_DERIVED_FIELDS:
    raise RuntimeError("every DCI 0_3 field must have a configuration-derived width")
_SRS_PERIODICITY_OFFSETS = {
    "sl1": 1,
    **{
        f"sl{period}": period
        for period in (
            2,
            4,
            5,
            8,
            10,
            16,
            20,
            32,
            40,
            64,
            80,
            160,
            320,
            640,
            1280,
            2560,
            5120,
            10240,
            40960,
            81920,
        )
    },
}
_SRS_PERIODICITY_EXT_OFFSETS = {
    f"sl{period}": period for period in (128, 256, 512, 20480)
}
_SRS_REQUEST_DCI_FORMATS = frozenset({"0_1", "0_2", "1_1", "1_2"})
# TS 38.211 V19.2.0 Table 6.4.1.4.3-1, each row is
# ((m_SRS,0, N_0), ..., (m_SRS,3, N_3)).
_SRS_BANDWIDTH_TABLE: tuple[tuple[tuple[int, int], ...], ...] = (
    ((4, 1), (4, 1), (4, 1), (4, 1)),
    ((8, 1), (4, 2), (4, 1), (4, 1)),
    ((12, 1), (4, 3), (4, 1), (4, 1)),
    ((16, 1), (4, 4), (4, 1), (4, 1)),
    ((16, 1), (8, 2), (4, 2), (4, 1)),
    ((20, 1), (4, 5), (4, 1), (4, 1)),
    ((24, 1), (4, 6), (4, 1), (4, 1)),
    ((24, 1), (12, 2), (4, 3), (4, 1)),
    ((28, 1), (4, 7), (4, 1), (4, 1)),
    ((32, 1), (16, 2), (8, 2), (4, 2)),
    ((36, 1), (12, 3), (4, 3), (4, 1)),
    ((40, 1), (20, 2), (4, 5), (4, 1)),
    ((48, 1), (16, 3), (8, 2), (4, 2)),
    ((48, 1), (24, 2), (12, 2), (4, 3)),
    ((52, 1), (4, 13), (4, 1), (4, 1)),
    ((56, 1), (28, 2), (4, 7), (4, 1)),
    ((60, 1), (20, 3), (4, 5), (4, 1)),
    ((64, 1), (32, 2), (16, 2), (4, 4)),
    ((72, 1), (24, 3), (12, 2), (4, 3)),
    ((72, 1), (36, 2), (12, 3), (4, 3)),
    ((76, 1), (4, 19), (4, 1), (4, 1)),
    ((80, 1), (40, 2), (20, 2), (4, 5)),
    ((88, 1), (44, 2), (4, 11), (4, 1)),
    ((96, 1), (32, 3), (16, 2), (4, 4)),
    ((96, 1), (48, 2), (24, 2), (4, 6)),
    ((104, 1), (52, 2), (4, 13), (4, 1)),
    ((112, 1), (56, 2), (28, 2), (4, 7)),
    ((120, 1), (60, 2), (20, 3), (4, 5)),
    ((120, 1), (40, 3), (8, 5), (4, 2)),
    ((120, 1), (24, 5), (12, 2), (4, 3)),
    ((128, 1), (64, 2), (32, 2), (4, 8)),
    ((128, 1), (64, 2), (16, 4), (4, 4)),
    ((128, 1), (16, 8), (8, 2), (4, 2)),
    ((132, 1), (44, 3), (4, 11), (4, 1)),
    ((136, 1), (68, 2), (4, 17), (4, 1)),
    ((144, 1), (72, 2), (36, 2), (4, 9)),
    ((144, 1), (48, 3), (24, 2), (12, 2)),
    ((144, 1), (48, 3), (16, 3), (4, 4)),
    ((144, 1), (16, 9), (8, 2), (4, 2)),
    ((152, 1), (76, 2), (4, 19), (4, 1)),
    ((160, 1), (80, 2), (40, 2), (4, 10)),
    ((160, 1), (80, 2), (20, 4), (4, 5)),
    ((160, 1), (32, 5), (16, 2), (4, 4)),
    ((168, 1), (84, 2), (28, 3), (4, 7)),
    ((176, 1), (88, 2), (44, 2), (4, 11)),
    ((184, 1), (92, 2), (4, 23), (4, 1)),
    ((192, 1), (96, 2), (48, 2), (4, 12)),
    ((192, 1), (96, 2), (24, 4), (4, 6)),
    ((192, 1), (64, 3), (16, 4), (4, 4)),
    ((192, 1), (24, 8), (8, 3), (4, 2)),
    ((208, 1), (104, 2), (52, 2), (4, 13)),
    ((216, 1), (108, 2), (36, 3), (4, 9)),
    ((224, 1), (112, 2), (56, 2), (4, 14)),
    ((240, 1), (120, 2), (60, 2), (4, 15)),
    ((240, 1), (80, 3), (20, 4), (4, 5)),
    ((240, 1), (48, 5), (16, 3), (8, 2)),
    ((240, 1), (24, 10), (12, 2), (4, 3)),
    ((256, 1), (128, 2), (64, 2), (4, 16)),
    ((256, 1), (128, 2), (32, 4), (4, 8)),
    ((256, 1), (16, 16), (8, 2), (4, 2)),
    ((264, 1), (132, 2), (44, 3), (4, 11)),
    ((272, 1), (136, 2), (68, 2), (4, 17)),
    ((272, 1), (68, 4), (4, 17), (4, 1)),
    ((272, 1), (16, 17), (8, 2), (4, 2)),
)
# TS 38.211 V19.2.0, Tables 5.2.2.2-1 through 5.2.2.2-4.
_LOW_PAPR_SHORT_PHASES: dict[int, tuple[tuple[int, ...], ...]] = {
    6: (
        (-3, -1, 3, 3, -1, -3), (-3, 3, -1, -1, 3, -3),
        (-3, -3, -3, 3, 1, -3), (1, 1, 1, 3, -1, -3),
        (1, 1, 1, -3, -1, 3), (-3, 1, -1, -3, -3, -3),
        (-3, 1, 3, -3, -3, -3), (-3, -1, 1, -3, 1, -1),
        (-3, -1, -3, 1, -3, -3), (-3, -3, 1, -3, 3, -3),
        (-3, 1, 3, 1, -3, -3), (-3, -1, -3, 1, 1, -3),
        (1, 1, 3, -1, -3, 3), (1, 1, 3, 3, -1, 3),
        (1, 1, 1, -3, 3, -1), (1, 1, 1, -1, 3, -3),
        (-3, -1, -1, -1, 3, -1), (-3, -3, -1, 1, -1, -3),
        (-3, -3, -3, 1, -3, -1), (-3, 1, 1, -3, -1, -3),
        (-3, 3, -3, 1, 1, -3), (-3, 1, -3, -3, -3, -1),
        (1, 1, -3, 3, 1, 3), (1, 1, -3, -3, 1, -3),
        (1, 1, 3, -1, 3, 3), (1, 1, -3, 1, 3, 3),
        (1, 1, -1, -1, 3, -1), (1, 1, -1, 3, -1, -1),
        (1, 1, -1, 3, -3, -1), (1, 1, -3, 1, -1, -1),
    ),
    12: (
        (-3, 1, -3, -3, -3, 3, -3, -1, 1, 1, 1, -3),
        (-3, 3, 1, -3, 1, 3, -1, -1, 1, 3, 3, 3),
        (-3, 3, 3, 1, -3, 3, -1, 1, 3, -3, 3, -3),
        (-3, -3, -1, 3, 3, 3, -3, 3, -3, 1, -1, -3),
        (-3, -1, -1, 1, 3, 1, 1, -1, 1, -1, -3, 1),
        (-3, -3, 3, 1, -3, -3, -3, -1, 3, -1, 1, 3),
        (1, -1, 3, -1, -1, -1, -3, -1, 1, 1, 1, -3),
        (-1, -3, 3, -1, -3, -3, -3, -1, 1, -1, 1, -3),
        (-3, -1, 3, 1, -3, -1, -3, 3, 1, 3, 3, 1),
        (-3, -1, -1, -3, -3, -1, -3, 3, 1, 3, -1, -3),
        (-3, 3, -3, 3, 3, -3, -1, -1, 3, 3, 1, -3),
        (-3, -1, -3, -1, -1, -3, 3, 3, -1, -1, 1, -3),
        (-3, -1, 3, -3, -3, -1, -3, 1, -1, -3, 3, 3),
        (-3, 1, -1, -1, 3, 3, -3, -1, -1, -3, -1, -3),
        (1, 3, -3, 1, 3, 3, 3, 1, -1, 1, -1, 3),
        (-3, 1, 3, -1, -1, -3, -3, -1, -1, 3, 1, -3),
        (-1, -1, -1, -1, 1, -3, -1, 3, 3, -1, -3, 1),
        (-1, 1, 1, -1, 1, 3, 3, -1, -1, -3, 1, -3),
        (-3, 1, 3, 3, -1, -1, -3, 3, 3, -3, 3, -3),
        (-3, -3, 3, -3, -1, 3, 3, 3, -1, -3, 1, -3),
        (3, 1, 3, 1, 3, -3, -1, 1, 3, 1, -1, -3),
        (-3, 3, 1, 3, -3, 1, 1, 1, 1, 3, -3, 3),
        (-3, 3, 3, 3, -1, -3, -3, -1, -3, 1, 3, -3),
        (3, -1, -3, 3, -3, -1, 3, 3, 3, -3, -1, -3),
        (-3, -1, 1, -3, 1, 3, 3, 3, -1, -3, 3, 3),
        (-3, 3, 1, -1, 3, 3, -3, 1, -1, 1, -1, 1),
        (-1, 1, 3, -3, 1, -1, 1, -1, -1, -3, 1, -1),
        (-3, -3, 3, 3, 3, -3, -1, 1, -3, 3, 1, -3),
        (1, -1, 3, 1, 1, -1, -1, -1, 1, 3, -3, 1),
        (-3, 3, -3, 3, -3, -3, 3, -1, -1, 1, 3, -3),
    ),
    18: (
        (-1, 3, -1, -3, 3, 1, -3, -1, 3, -3, -1, -1, 1, 1, 1, -1, -1, -1),
        (3, -3, 3, -1, 1, 3, -3, -1, -3, -3, -1, -3, 3, 1, -1, 3, -3, 3),
        (-3, 3, 1, -1, -1, 3, -3, -1, 1, 1, 1, 1, 1, -1, 3, -1, -3, -1),
        (-3, -3, 3, 3, 3, 1, -3, 1, 3, 3, 1, -3, -3, 3, -1, -3, -1, 1),
        (1, 1, -1, -1, -3, -1, 1, -3, -3, -3, 1, -3, -1, -1, 1, -1, 3, 1),
        (3, -3, 1, 1, 3, -1, 1, -1, -1, -3, 1, 1, -1, 3, 3, -3, 3, -1),
        (-3, 3, -1, 1, 3, 1, -3, -1, 1, 1, -3, 1, 3, 3, -1, -3, -3, -3),
        (1, 1, -3, 3, 3, 1, 3, -3, 3, -1, 1, 1, -1, 1, -3, -3, -1, 3),
        (-3, 1, -3, -3, 1, -3, -3, 3, 1, -3, -1, -3, -3, -3, -1, 1, 1, 3),
        (3, -1, 3, 1, -3, -3, -1, 1, -3, -3, 3, 3, 3, 1, 3, -3, 3, -3),
        (-3, -3, -3, 1, -3, 3, 1, 1, 3, -3, -3, 1, 3, -1, 3, -3, -3, 3),
        (-3, -3, 3, 3, 3, -1, -1, -3, -1, -1, -1, 3, 1, -3, -3, -1, 3, -1),
        (-3, -1, -3, -3, 1, 1, -1, -3, -1, -3, -1, -1, 3, 3, -1, 3, 1, 3),
        (1, 1, -3, -3, -3, -3, 1, 3, -3, 3, 3, 1, -3, -1, 3, -1, -3, 1),
        (-3, 3, -1, -3, -1, -3, 1, 1, -3, -3, -1, -1, 3, -3, 1, 3, 1, 1),
        (3, 1, -3, 1, -3, 3, 3, -1, -3, -3, -1, -3, -3, 3, -3, -1, 1, 3),
        (-3, -1, -3, -1, -3, 1, 3, -3, -1, 3, 3, 3, 1, -1, -3, 3, -1, -3),
        (-3, -1, 3, 3, -1, 3, -1, -3, -1, 1, -1, -3, -1, -1, -1, 3, 3, 1),
        (-3, 1, -3, -1, -1, 3, 1, -3, -3, -3, -1, -3, -3, 1, 1, 1, -1, -1),
        (3, 3, 3, -3, -1, -3, -1, 3, -1, 1, -1, -3, 1, -3, -3, -1, 3, 3),
        (-3, 1, 1, -3, 1, 1, 3, -3, -1, -3, -1, 3, -3, 3, -1, -1, -1, -3),
        (1, -3, -1, -3, 3, 3, -1, -3, 1, -3, -3, -1, -3, -1, 1, 3, 3, 3),
        (-3, -3, 1, -1, -1, 1, 1, -3, -1, 3, 3, 3, 3, -1, 3, 1, 3, 1),
        (3, -1, -3, 1, -3, -3, -3, 3, 3, -1, 1, -3, -1, 3, 1, 1, 3, 3),
        (3, -1, -1, 1, -3, -1, -3, -1, -3, -3, -1, -3, 1, 1, 1, -3, -3, 3),
        (-3, -3, 1, -3, 3, 3, 3, -1, 3, 1, 1, -3, -3, -3, 3, -3, -1, -1),
        (-3, -1, -1, -3, 1, -3, 3, -1, -1, -3, 3, 3, -3, -1, 3, -1, -1, -1),
        (-3, -3, 3, 3, -3, 1, 3, -1, -3, 1, -1, -3, 3, -3, -1, -1, -1, 3),
        (-1, -3, 1, -3, -3, -3, 1, 1, 3, 3, -3, 3, 3, -3, -1, 3, -3, 1),
        (-3, 3, 1, -1, -1, -1, -1, 1, -1, 3, 3, -3, -1, 1, 3, -1, 3, -1),
    ),
    24: (
        (-1, -3, 3, -1, 3, 1, 3, -1, 1, -3, -1, -3, -1, 1, 3, -3, -1, -3, 3, 3, 3, -3, -3, -3),
        (-1, -3, 3, 1, 1, -3, 1, -3, -3, 1, -3, -1, -1, 3, -3, 3, 3, 3, -3, 1, 3, 3, -3, -3),
        (-1, -3, -3, 1, -1, -1, -3, 1, 3, -1, -3, -1, -1, -3, 1, 1, 3, 1, -3, -1, -1, 3, -3, -3),
        (1, -3, 3, -1, -3, -1, 3, 3, 1, -1, 1, 1, 3, -3, -1, -3, -3, -3, -1, 3, -3, -1, -3, -3),
        (-1, 3, -3, -3, -1, 3, -1, -1, 1, 3, 1, 3, -1, -1, -3, 1, 3, 1, -1, -3, 1, -1, -3, -3),
        (-3, -1, 1, -3, -3, 1, 1, -3, 3, -1, -1, -3, 1, 3, 1, -1, -3, -1, -3, 1, -3, -3, -3, -3),
        (-3, 3, 1, 3, -1, 1, -3, 1, -3, 1, -1, -3, -1, -3, -3, -3, -3, -1, -1, -1, 1, 1, -3, -3),
        (-3, 1, 3, -1, 1, -1, 3, -3, 3, -1, -3, -1, -3, 3, -1, -1, -1, -3, -1, -1, -3, 3, 3, -3),
        (-3, 1, -3, 3, -1, -1, -1, -3, 3, 1, -1, -3, -1, 1, 3, -1, 1, -1, 1, -3, -3, -3, -3, -3),
        (1, 1, -1, -3, -1, 1, 1, -3, 1, -1, 1, -3, 3, -3, -3, 3, -1, -3, 1, 3, -3, 1, -3, -3),
        (-3, -3, -3, -1, 3, -3, 3, 1, 3, 1, -3, -1, -1, -3, 1, 1, 3, 1, -1, -3, 3, 1, 3, -3),
        (-3, 3, -1, 3, 1, -1, -1, -1, 3, 3, 1, 1, 1, 3, 3, 1, -3, -3, -1, 1, -3, 1, 3, -3),
        (3, -3, 3, -1, -3, 1, 3, 1, -1, -1, -3, -1, 3, -3, 3, -1, -1, 3, 3, -3, -3, 3, -3, -3),
        (-3, 3, -1, 3, -1, 3, 3, 1, 1, -3, 1, 3, -3, 3, -3, -3, -1, 1, 3, -3, -1, -1, -3, -3),
        (-3, 1, -3, -1, -1, 3, 1, 3, -3, 1, -1, 3, 3, -1, -3, 3, -3, -1, -1, -3, -3, -3, 3, -3),
        (-3, -1, -1, -3, 1, -3, -3, -1, -1, 3, -1, 1, -1, 3, 1, -3, -1, 3, 1, 1, -1, -1, -3, -3),
        (-3, -3, 1, -1, 3, 3, -3, -1, 1, -1, -1, 1, 1, -1, -1, 3, -3, 1, -3, 1, -1, -1, -1, -3),
        (3, -1, 3, -1, 1, -3, 1, 1, -3, -3, 3, -3, -1, -1, -1, -1, -1, -3, -3, -1, 1, 1, -3, -3),
        (-3, 1, -3, 1, -3, -3, 1, -3, 1, -3, -3, -3, -3, -3, 1, -3, -3, 1, 1, -3, 1, 1, -3, -3),
        (-3, -3, 3, 3, 1, -1, -1, -1, 1, -3, -1, 1, -1, 3, -3, -1, -3, -1, -1, 1, -3, 3, -1, -3),
        (-3, -3, -1, -1, -1, -3, 1, -1, -3, -1, 3, -3, 1, -3, 3, -3, 3, 3, 1, -1, -1, 1, -3, -3),
        (3, -1, 1, -1, 3, -3, 1, 1, 3, -1, -3, 3, 1, -3, 3, -1, -1, -1, -1, 1, -3, -3, -3, -3),
        (-3, 1, -3, 3, -3, 1, -3, 3, 1, -1, -3, -1, -3, -3, -3, -3, 1, 3, -1, 1, 3, 3, 3, -3),
        (-3, -1, 1, -3, -1, -1, 1, 1, 1, 3, 3, -1, 1, -1, 1, -1, -1, -3, -3, -3, 3, 1, -1, -3),
        (-3, 3, -1, -3, -1, -1, -1, 3, -1, -1, 3, -3, -1, 3, -3, 3, -3, -1, 3, 1, 1, -1, -3, -3),
        (-3, 1, -1, -3, -3, -1, 1, -3, -1, -3, 1, 1, -1, 1, 1, 3, 3, 3, -1, 1, -1, 1, -1, -3),
        (-1, 3, -1, -1, 3, 3, -1, -1, -1, 3, -1, -3, 1, 3, 1, 1, -3, -3, -3, -1, -3, -1, -3, -3),
        (3, -3, -3, -1, 3, 3, -3, -1, 3, 1, 1, 1, 3, -1, 3, -3, -1, 3, -1, 3, 1, -1, -3, -3),
        (-3, 1, -3, 1, -3, 1, 1, 3, 1, -3, -3, -1, 1, 3, -1, -3, 3, 1, -1, -3, -3, -3, -3, -3),
        (3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3),
    ),
}


def srs_cyclic_shift_count(transmission_comb: int) -> int:
    """Return TS 38.211's maximum SRS cyclic shifts for a transmission comb."""

    if isinstance(transmission_comb, bool) or not isinstance(transmission_comb, Integral):
        raise TypeError("transmission_comb must be an integer")
    try:
        return _SRS_CYCLIC_SHIFT_COUNTS[int(transmission_comb)]
    except KeyError as error:
        raise ValueError("transmission_comb must be 2, 4, or 8") from error


@dataclass(frozen=True, slots=True)
class NRSRSTransmissionCombConfiguration:
    transmission_comb: int
    comb_offset: int
    base_cyclic_shift: int

    def __post_init__(self) -> None:
        for name in ("transmission_comb", "comb_offset", "base_cyclic_shift"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        comb = int(self.transmission_comb)
        maximum = srs_cyclic_shift_count(comb)
        if not 0 <= self.comb_offset < comb:
            raise ValueError("comb_offset must lie inside the transmission comb")
        if not 0 <= self.base_cyclic_shift < maximum:
            raise ValueError(f"base_cyclic_shift must be in [0, {maximum})")
        object.__setattr__(self, "transmission_comb", comb)
        object.__setattr__(self, "comb_offset", int(self.comb_offset))
        object.__setattr__(self, "base_cyclic_shift", int(self.base_cyclic_shift))


@dataclass(frozen=True, slots=True)
class NRSRSPortConfiguration:
    antenna_ports: tuple[int, ...]
    ports8_tdm: bool

    def __post_init__(self) -> None:
        if not isinstance(self.antenna_ports, tuple) or not self.antenna_ports:
            raise TypeError("antenna_ports must be a nonempty tuple")
        if any(isinstance(port, bool) or not isinstance(port, Integral) for port in self.antenna_ports):
            raise TypeError("antenna_ports must contain integers")
        if not isinstance(self.ports8_tdm, bool):
            raise TypeError("ports8_tdm must be a boolean")
        ports = tuple(int(port) for port in self.antenna_ports)
        if ports not in {
            (1000,),
            (1000, 1001),
            (1000, 1001, 1002, 1003),
            tuple(range(1000, 1008)),
        }:
            raise ValueError("antenna_ports must be a complete ordered SRS port roster")
        if self.ports8_tdm and ports != tuple(range(1000, 1008)):
            raise ValueError("ports8_tdm requires the complete eight-port roster")
        object.__setattr__(self, "antenna_ports", ports)


def nr_srs_transmission_comb_from_rrc(
    decoded_choice: Mapping[str, object], *, positioning_resource: bool = False
) -> NRSRSTransmissionCombConfiguration:
    """Resolve a decoded TS 38.331 transmissionComb CHOICE."""

    if not isinstance(decoded_choice, Mapping) or len(decoded_choice) != 1:
        raise ValueError("transmissionComb must be a single-choice decoded mapping")
    if not isinstance(positioning_resource, bool):
        raise TypeError("positioning_resource must be a boolean")
    choice, raw_value = next(iter(decoded_choice.items()))
    suffix = "-r16" if positioning_resource else ""
    if choice not in {f"n{comb}{suffix}" for comb in (2, 4, 8)}:
        raise ValueError("transmissionComb choice must be n2, n4, or n8 for this resource")
    if not isinstance(raw_value, Mapping):
        raise ValueError("transmissionComb choice value must be a mapping")
    comb = int(choice[1:].removesuffix(suffix))
    fields = {f"combOffset-n{comb}{suffix}", f"cyclicShift-n{comb}{suffix}"}
    if set(raw_value) != fields:
        raise ValueError("transmissionComb choice has missing or unsupported fields")
    return NRSRSTransmissionCombConfiguration(
        transmission_comb=comb,
        comb_offset=raw_value[f"combOffset-n{comb}{suffix}"],  # type: ignore[arg-type]
        base_cyclic_shift=raw_value[f"cyclicShift-n{comb}{suffix}"],  # type: ignore[arg-type]
    )


def nr_srs_port_configuration_from_rrc(
    nrof_srs_ports: str | None,
    *,
    nrof_srs_ports_n8: str | None = None,
    cyclic_shift_hopping_enabled: bool = False,
    comb_offset_hopping_enabled: bool = False,
) -> NRSRSPortConfiguration:
    """Resolve Release 19 decoded SRS port enums and ports8tdm constraints."""

    if not isinstance(cyclic_shift_hopping_enabled, bool) or not isinstance(
        comb_offset_hopping_enabled, bool
    ):
        raise TypeError("RRC hopping-presence flags must be booleans")
    if nrof_srs_ports_n8 is not None:
        if not isinstance(nrof_srs_ports_n8, str):
            raise TypeError("nrofSRS-Ports-n8 must be a string enum")
        if nrof_srs_ports_n8 not in {"ports8", "ports8tdm"}:
            raise ValueError("nrofSRS-Ports-n8 must be ports8 or ports8tdm")
        ports8_tdm = nrof_srs_ports_n8 == "ports8tdm"
        if ports8_tdm and (cyclic_shift_hopping_enabled or comb_offset_hopping_enabled):
            raise ValueError("ports8tdm cannot be combined with cyclic-shift or comb-offset hopping")
        return NRSRSPortConfiguration(tuple(range(1000, 1008)), ports8_tdm)
    if nrof_srs_ports is None:
        nrof_srs_ports = "port1"
    if not isinstance(nrof_srs_ports, str):
        raise TypeError("nrofSRS-Ports must be a string enum")
    counts = {"port1": 1, "ports2": 2, "ports4": 4}
    try:
        port_count = counts[nrof_srs_ports]
    except (KeyError, TypeError) as error:
        raise ValueError("nrofSRS-Ports must be port1, ports2, or ports4") from error
    return NRSRSPortConfiguration(tuple(range(1000, 1000 + port_count)), False)


def nr_srs_port_cyclic_shift_indices(
    antenna_ports: tuple[int, ...],
    *,
    transmission_comb: int,
    base_cyclic_shift: int,
    ports8_tdm: bool = False,
) -> tuple[int, ...]:
    """Resolve TS 38.211 antenna-port cyclic shifts for an SRS resource.

    ``antenna_ports`` contains the configured SRS antenna-port numbers
    (1000 onward), in the desired output order. ``base_cyclic_shift`` is the
    ``n_SRS^cs`` value configured by ``transmissionComb``. For the eight-port
    TDM option, the standard's effective four-port roster and port remapping
    are applied before deriving the cyclic shifts.
    """

    if not isinstance(antenna_ports, tuple) or not antenna_ports:
        raise TypeError("antenna_ports must be a nonempty tuple of integers")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    for name, value in (
        ("transmission_comb", transmission_comb),
        ("base_cyclic_shift", base_cyclic_shift),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    maximum = srs_cyclic_shift_count(int(transmission_comb))
    base_cyclic_shift = int(base_cyclic_shift)
    if not 0 <= base_cyclic_shift < maximum:
        raise ValueError(f"base_cyclic_shift must be in [0, {maximum})")
    if any(
        isinstance(port, bool) or not isinstance(port, Integral)
        for port in antenna_ports
    ):
        raise TypeError("antenna_ports must contain integers")
    ports = tuple(int(port) for port in antenna_ports)
    port_count = 8 if ports8_tdm else len(ports)
    if port_count not in {1, 2, 4, 8}:
        raise ValueError("SRS antenna-port count must be 1, 2, 4, or 8")
    if ports != tuple(range(1000, 1000 + port_count)):
        raise ValueError("antenna_ports must list the complete configured SRS port roster")
    effective_count = 4 if ports8_tdm else port_count
    shifts: list[int] = []
    for port in ports:
        offset = port - 1000
        mapped_port = (
            1000 + offset % 2 + (2 if offset >= 4 else 0)
            if ports8_tdm
            else port
        )
        mapped_index = mapped_port - 1000
        if effective_count == 8 and maximum == 6:
            increment = maximum * (mapped_index // 4) // (effective_count // 4)
        elif (effective_count == 4 and maximum == 6) or (
            effective_count == 8 and maximum == 12
        ):
            increment = maximum * (mapped_index // 2) // (effective_count // 2)
        else:
            increment = maximum * mapped_index // effective_count
        shifts.append((base_cyclic_shift + increment) % maximum)
    return tuple(shifts)


def nr_srs_port_symbol_activity(
    antenna_ports: tuple[int, ...],
    *,
    srs_symbol_count: int,
    ports8_tdm: bool = False,
) -> tuple[tuple[bool, ...], ...]:
    """Return TS 38.211 activity by antenna port and resource-local symbol."""

    if not isinstance(antenna_ports, tuple) or not antenna_ports:
        raise TypeError("antenna_ports must be a nonempty tuple of integers")
    if isinstance(srs_symbol_count, bool) or not isinstance(srs_symbol_count, Integral):
        raise TypeError("srs_symbol_count must be an integer")
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS:
        raise ValueError("srs_symbol_count is unsupported")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    if any(
        isinstance(port, bool) or not isinstance(port, Integral) for port in antenna_ports
    ):
        raise TypeError("antenna_ports must contain integers")
    ports = tuple(int(port) for port in antenna_ports)
    if len(set(ports)) != len(ports) or any(
        port not in range(1000, 1008) for port in ports
    ):
        raise ValueError("antenna_ports must be unique SRS ports in [1000, 1008)")
    if ports8_tdm and ports != tuple(range(1000, 1008)):
        raise ValueError("ports8_tdm requires the complete ordered port roster 1000 through 1007")
    if not ports8_tdm:
        return tuple((True,) * int(srs_symbol_count) for _ in ports)
    even_symbol_ports = {1000, 1001, 1004, 1005}
    return tuple(
        tuple(
            (symbol_index % 2 == 0) == (port in even_symbol_ports)
            for symbol_index in range(int(srs_symbol_count))
        )
        for port in ports
    )


def nr_srs_cyclic_shift_hopping_offset(
    *,
    transmission_comb: int,
    cyclic_shift_hopping_id: int,
    radio_frame_index: int,
    slot_index: int,
    slots_per_frame: int,
    slot_symbol_count: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
) -> int:
    """Resolve TS 38.211 V19.2.0 ``f_csh`` for one SRS resource symbol.

    The slot index is local to its radio frame. A subset is supplied as the
    configured ``n_SRS^cs,max``-bit bitmap; finer granularity and a subset are
    mutually exclusive, as specified for the cyclicShiftHopping IE.
    """

    for name, value in (
        ("transmission_comb", transmission_comb),
        ("cyclic_shift_hopping_id", cyclic_shift_hopping_id),
        ("radio_frame_index", radio_frame_index),
        ("slot_index", slot_index),
        ("slots_per_frame", slots_per_frame),
        ("slot_symbol_count", slot_symbol_count),
        ("start_symbol_index", start_symbol_index),
        ("resource_symbol_index", resource_symbol_index),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if not isinstance(hopping_finer_granularity, bool):
        raise TypeError("hopping_finer_granularity must be a boolean")
    maximum = srs_cyclic_shift_count(int(transmission_comb))
    identity = int(cyclic_shift_hopping_id)
    frame = int(radio_frame_index)
    slot = int(slot_index)
    slots = int(slots_per_frame)
    symbols = int(slot_symbol_count)
    start = int(start_symbol_index)
    resource_symbol = int(resource_symbol_index)
    if not 0 <= identity <= 65535:
        raise ValueError("cyclic_shift_hopping_id must be in [0, 65535]")
    if (
        frame < 0
        or slots not in {10, 20, 40, 80, 160, 320}
        or not 0 <= slot < slots
    ):
        raise ValueError("radio frame and frame-local slot indices are invalid")
    if symbols not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if not 0 <= start < symbols or not 0 <= resource_symbol < symbols - start:
        raise ValueError("SRS symbol coordinates must lie inside the slot")

    granularity = 2 if hopping_finer_granularity else 1
    if hopping_subset is None:
        subset = tuple(range(granularity * maximum))
    else:
        if not isinstance(hopping_subset, tuple) or len(hopping_subset) != maximum:
            raise ValueError("hopping_subset must be a bitmap matching the comb shift count")
        if any(not isinstance(bit, (bool, np.bool_)) for bit in hopping_subset):
            raise TypeError("hopping_subset must contain boolean bitmap entries")
        if hopping_finer_granularity:
            raise ValueError("hopping_subset and finer granularity cannot both be configured")
        subset = tuple(index for index, enabled in enumerate(hopping_subset) if enabled)
        if not 1 < len(subset) <= maximum:
            raise ValueError("hopping_subset must enable between 2 and n_cs,max shifts")

    symbol_offset = (
        ((frame % 128) * slots + slot) * symbols + start + resource_symbol
    )
    bits = _nr_gold_bits(identity, 8 * symbol_offset + 8)[8 * symbol_offset :]
    hop_word = sum(int(bit) << bit_index for bit_index, bit in enumerate(bits))
    return subset[hop_word % len(subset)]


def nr_srs_cyclic_shift_hopping_offsets(
    *,
    transmission_comb: int,
    cyclic_shift_hopping_id: int,
    radio_frame_index: int,
    slot_index: int,
    slots_per_frame: int,
    slot_symbol_count: int,
    start_symbol_index: int,
    srs_symbol_count: int,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
) -> tuple[int, ...]:
    """Resolve ``f_csh`` for every resource-local SRS symbol in one slot."""
    for name, value in (
        ("slot_symbol_count", slot_symbol_count),
        ("start_symbol_index", start_symbol_index),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    slot_symbol_count = int(slot_symbol_count)
    start_symbol_index = int(start_symbol_index)
    if slot_symbol_count not in {12, 14} or not 0 <= start_symbol_index < slot_symbol_count:
        raise ValueError("slot symbol coordinates are invalid")
    if (
        isinstance(srs_symbol_count, bool)
        or not isinstance(srs_symbol_count, Integral)
        or int(srs_symbol_count) not in _SRS_SYMBOL_COUNTS
    ):
        raise ValueError("srs_symbol_count must be one of the supported SRS lengths")
    if int(srs_symbol_count) > slot_symbol_count - start_symbol_index:
        raise ValueError("SRS resource symbols extend past the slot")
    return tuple(
        nr_srs_cyclic_shift_hopping_offset(
            transmission_comb=transmission_comb,
            cyclic_shift_hopping_id=cyclic_shift_hopping_id,
            radio_frame_index=radio_frame_index,
            slot_index=slot_index,
            slots_per_frame=slots_per_frame,
            slot_symbol_count=slot_symbol_count,
            start_symbol_index=start_symbol_index,
            resource_symbol_index=resource_symbol_index,
            hopping_subset=hopping_subset,
            hopping_finer_granularity=hopping_finer_granularity,
        )
        for resource_symbol_index in range(int(srs_symbol_count))
    )


@dataclass(frozen=True, slots=True)
class NRSRSBandwidthConfiguration:
    """Resolved fixed SRS sequence extent for one standardized c-SRS/b-SRS pair."""

    c_srs: int
    b_srs: int
    transmission_comb: int
    bandwidth_rb: int
    subband_count: int
    sequence_length: int
    allocated_bandwidth_subcarriers: int

    def __post_init__(self) -> None:
        values = {
            "c_srs": self.c_srs,
            "b_srs": self.b_srs,
            "transmission_comb": self.transmission_comb,
            "bandwidth_rb": self.bandwidth_rb,
            "subband_count": self.subband_count,
            "sequence_length": self.sequence_length,
            "allocated_bandwidth_subcarriers": self.allocated_bandwidth_subcarriers,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        c_srs = int(self.c_srs)
        b_srs = int(self.b_srs)
        transmission_comb = int(self.transmission_comb)
        if not 0 <= c_srs < len(_SRS_BANDWIDTH_TABLE):
            raise ValueError("c_srs must be in [0, 64)")
        if not 0 <= b_srs <= 3:
            raise ValueError("b_srs must be in [0, 3]")
        srs_cyclic_shift_count(transmission_comb)
        bandwidth_rb, subband_count = _SRS_BANDWIDTH_TABLE[c_srs][b_srs]
        expected_subcarriers = bandwidth_rb * 12
        expected_sequence_length = expected_subcarriers // transmission_comb
        if self.bandwidth_rb != bandwidth_rb:
            raise ValueError("bandwidth_rb disagrees with the standardized c_srs row")
        if self.subband_count != subband_count:
            raise ValueError("subband_count disagrees with the standardized c_srs/b_srs row")
        if self.allocated_bandwidth_subcarriers != expected_subcarriers:
            raise ValueError("allocated bandwidth disagrees with the standardized c_srs row")
        if self.sequence_length != expected_sequence_length:
            raise ValueError("sequence_length disagrees with the bandwidth and transmission comb")
        for name, value in values.items():
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True, slots=True)
class NRSRSFixedFrequencyAllocation:
    """Resolved SRS allocation for one frequency-hop counter in a grid frame."""

    bandwidth: NRSRSBandwidthConfiguration
    position_indices: tuple[int, ...]
    frequency_domain_position: int
    reference_subcarrier: int
    frequency_domain_shift_rb: int
    comb_offset: int
    start_subcarrier: int
    b_hop: int = 3
    n_srs: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bandwidth, NRSRSBandwidthConfiguration):
            raise TypeError("bandwidth must be an NRSRSBandwidthConfiguration")
        indices = tuple(self.position_indices)
        level_count = self.bandwidth.b_srs + 1
        if len(indices) != level_count:
            raise ValueError("position_indices must have one entry per bandwidth level")
        for name in (
            "frequency_domain_position",
            "reference_subcarrier",
            "frequency_domain_shift_rb",
            "comb_offset",
            "start_subcarrier",
            "b_hop",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
                raise ValueError("position indices must be nonnegative integers")
        expected_indices = nr_srs_frequency_position_indices(
            self.bandwidth.c_srs,
            b_srs=self.bandwidth.b_srs,
            b_hop=int(self.b_hop),
            frequency_domain_position=int(self.frequency_domain_position),
            n_srs=self.n_srs,
        )
        if tuple(indices) != expected_indices:
            raise ValueError("position_indices disagree with the resolved SRS frequency position")
        if self.comb_offset >= self.bandwidth.transmission_comb:
            raise ValueError("comb_offset must be smaller than transmission_comb")
        expected_start = (
            self.reference_subcarrier
            + 12 * self.frequency_domain_shift_rb
            + sum(
                12 * _SRS_BANDWIDTH_TABLE[self.bandwidth.c_srs][level][0] * index
                for level, index in enumerate(indices)
            )
        )
        if self.start_subcarrier != expected_start:
            raise ValueError("start_subcarrier disagrees with the resolved SRS allocation")
        if self.n_srs is not None and (
            isinstance(self.n_srs, bool)
            or not isinstance(self.n_srs, Integral)
            or self.n_srs < 0
        ):
            raise ValueError("n_srs must be a nonnegative integer when provided")
        object.__setattr__(self, "position_indices", tuple(int(index) for index in indices))
        for name in (
            "frequency_domain_position",
            "reference_subcarrier",
            "frequency_domain_shift_rb",
            "comb_offset",
            "start_subcarrier",
            "b_hop",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.n_srs is not None:
            object.__setattr__(self, "n_srs", int(self.n_srs))


@dataclass(frozen=True, slots=True)
class NRSRSAperiodicResourceSchedule:
    """Resolved active slot and per-symbol allocations for one SRS resource."""

    target_slot_index: int
    allocations: tuple[NRSRSFixedFrequencyAllocation, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_slot_index, bool)
            or not isinstance(self.target_slot_index, Integral)
            or self.target_slot_index < 0
        ):
            raise ValueError("target_slot_index must be a nonnegative integer")
        allocations = tuple(self.allocations)
        if not allocations or any(
            not isinstance(item, NRSRSFixedFrequencyAllocation) for item in allocations
        ):
            raise ValueError("allocations must contain fixed SRS frequency allocations")
        object.__setattr__(self, "target_slot_index", int(self.target_slot_index))
        object.__setattr__(self, "allocations", allocations)


@dataclass(frozen=True, slots=True)
class NRSRSPeriodicResourceSetSchedule:
    """One active periodic SRS set with its ordered resource schedules."""

    resource_set_id: int
    target_slot_index: int
    resources: tuple[tuple[int, NRSRSAperiodicResourceSchedule], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.resource_set_id, bool)
            or not isinstance(self.resource_set_id, Integral)
            or self.resource_set_id < 0
        ):
            raise ValueError("resource_set_id must be a nonnegative integer")
        if (
            isinstance(self.target_slot_index, bool)
            or not isinstance(self.target_slot_index, Integral)
            or self.target_slot_index < 0
        ):
            raise ValueError("target_slot_index must be a nonnegative integer")
        resources = tuple(self.resources)
        if not resources:
            raise ValueError("resources must contain at least one scheduled SRS resource")
        resource_ids: list[int] = []
        normalized_resources: list[tuple[int, NRSRSAperiodicResourceSchedule]] = []
        for item in resources:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("each resource entry must be a (resource_id, schedule) pair")
            resource_id, schedule = item
            if (
                isinstance(resource_id, bool)
                or not isinstance(resource_id, Integral)
                or resource_id < 0
            ):
                raise ValueError("resource IDs must be nonnegative integers")
            if not isinstance(schedule, NRSRSAperiodicResourceSchedule):
                raise TypeError("each resource schedule must be a periodic SRS schedule")
            if schedule.target_slot_index != int(self.target_slot_index):
                raise ValueError("all resource schedules must target the resource-set slot")
            resource_ids.append(int(resource_id))
            normalized_resources.append((int(resource_id), schedule))
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("resource IDs must be unique within the SRS resource set")
        object.__setattr__(self, "resource_set_id", int(self.resource_set_id))
        object.__setattr__(self, "target_slot_index", int(self.target_slot_index))
        object.__setattr__(
            self,
            "resources",
            tuple(normalized_resources),
        )


@dataclass(frozen=True, slots=True)
class NRSRSResourceSetAperiodicSchedule:
    """Resolved SRS resource-set identity, target slot, and symbol allocations."""

    resource_set_id: int
    target_slot_index: int
    allocations: tuple[NRSRSFixedFrequencyAllocation, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.resource_set_id, bool)
            or not isinstance(self.resource_set_id, Integral)
            or self.resource_set_id < 0
        ):
            raise ValueError("resource_set_id must be a nonnegative integer")
        schedule = NRSRSAperiodicResourceSchedule(
            self.target_slot_index, tuple(self.allocations)
        )
        object.__setattr__(self, "resource_set_id", int(self.resource_set_id))
        object.__setattr__(self, "target_slot_index", schedule.target_slot_index)
        object.__setattr__(self, "allocations", schedule.allocations)


@dataclass(frozen=True, slots=True)
class NRSRSBWPSelection:
    """Per-cell active UL BWP result for a decoded DCI 0_3 indicator."""

    indicator_bit_width: int
    selected_bwp_id_by_cell: tuple[tuple[str, int], ...]
    outcome_by_cell: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.indicator_bit_width, bool)
            or not isinstance(self.indicator_bit_width, Integral)
            or not 0 <= self.indicator_bit_width <= 2
        ):
            raise ValueError("indicator_bit_width must be in [0, 2]")
        selected = tuple(self.selected_bwp_id_by_cell)
        outcomes = tuple(self.outcome_by_cell)
        if not selected or len(selected) != len(outcomes):
            raise ValueError("BWP selection and outcome rows must be nonempty and aligned")
        if tuple(cell for cell, _ in selected) != tuple(cell for cell, _ in outcomes):
            raise ValueError("BWP selection and outcome rows must have identical cell order")
        if any(
            not isinstance(cell, str)
            or not cell
            or isinstance(bwp_id, bool)
            or not isinstance(bwp_id, Integral)
            or not 0 <= bwp_id <= 4
            for cell, bwp_id in selected
        ):
            raise ValueError("BWP selection rows require cell IDs and BWP IDs in [0, 4]")
        valid_outcomes = {
            "field_absent",
            "single_bwp",
            "selected_by_indicator",
            "codepoint_not_configured_for_cell",
            "active_bwp_change_unsupported",
        }
        if any(
            not isinstance(cell, str)
            or not cell
            or not isinstance(outcome, str)
            or outcome not in valid_outcomes
            for cell, outcome in outcomes
        ):
            raise ValueError("BWP outcome rows contain an invalid status")
        object.__setattr__(self, "indicator_bit_width", int(self.indicator_bit_width))
        object.__setattr__(self, "selected_bwp_id_by_cell", tuple(
            (cell, int(bwp_id)) for cell, bwp_id in selected
        ))
        object.__setattr__(self, "outcome_by_cell", outcomes)


@dataclass(frozen=True, slots=True)
class NRSRSBWPFrequencyBounds:
    """UL BWP extent in the caller's common subcarrier-index coordinate system."""

    start_subcarrier: int
    size_subcarriers: int

    def __post_init__(self) -> None:
        for name in ("start_subcarrier", "size_subcarriers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.start_subcarrier < 0 or self.size_subcarriers <= 0:
            raise ValueError("BWP start must be nonnegative and size must be positive")
        object.__setattr__(self, "start_subcarrier", int(self.start_subcarrier))
        object.__setattr__(self, "size_subcarriers", int(self.size_subcarriers))

    @property
    def end_subcarrier_exclusive(self) -> int:
        return self.start_subcarrier + self.size_subcarriers


@dataclass(frozen=True, slots=True)
class NRSRSBWPCellResourceSetSchedules:
    """A cell's selected UL BWP, extent, and feasible aperiodic SRS schedules."""

    cell_id: str
    ul_bwp_id: int
    frequency_bounds: NRSRSBWPFrequencyBounds
    resource_set_schedules: tuple[NRSRSResourceSetAperiodicSchedule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("cell_id must be a nonempty string")
        if (
            isinstance(self.ul_bwp_id, bool)
            or not isinstance(self.ul_bwp_id, Integral)
            or not 0 <= self.ul_bwp_id <= 4
        ):
            raise ValueError("ul_bwp_id must be in [0, 4]")
        if not isinstance(self.frequency_bounds, NRSRSBWPFrequencyBounds):
            raise TypeError("frequency_bounds must be NRSRSBWPFrequencyBounds")
        schedules = tuple(self.resource_set_schedules)
        if any(not isinstance(item, NRSRSResourceSetAperiodicSchedule) for item in schedules):
            raise ValueError("resource_set_schedules must contain SRS set schedules")
        for schedule in schedules:
            for allocation in schedule.allocations:
                allocation_end = (
                    allocation.start_subcarrier
                    + allocation.bandwidth.allocated_bandwidth_subcarriers
                )
                if (
                    allocation.start_subcarrier < self.frequency_bounds.start_subcarrier
                    or allocation_end > self.frequency_bounds.end_subcarrier_exclusive
                ):
                    raise ValueError(
                        f"SRS allocation for {self.cell_id!r} resource set "
                        f"{schedule.resource_set_id} lies outside selected UL BWP "
                        f"{self.ul_bwp_id}"
                    )
        object.__setattr__(self, "ul_bwp_id", int(self.ul_bwp_id))
        object.__setattr__(self, "resource_set_schedules", schedules)


@dataclass(frozen=True, slots=True)
class NRSRSBWPPeriodicResourceSetSchedule:
    """An active periodic SRS set proven to fit inside one selected UL BWP."""

    cell_id: str
    ul_bwp_id: int
    frequency_bounds: NRSRSBWPFrequencyBounds
    resource_set_schedule: NRSRSPeriodicResourceSetSchedule

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonempty string")
        if (
            isinstance(self.ul_bwp_id, bool)
            or not isinstance(self.ul_bwp_id, Integral)
            or not 0 <= self.ul_bwp_id <= 4
        ):
            raise ValueError("ul_bwp_id must be in [0, 4]")
        if not isinstance(self.frequency_bounds, NRSRSBWPFrequencyBounds):
            raise TypeError("frequency_bounds must be NRSRSBWPFrequencyBounds")
        if not isinstance(
            self.resource_set_schedule, NRSRSPeriodicResourceSetSchedule
        ):
            raise TypeError(
                "resource_set_schedule must be NRSRSPeriodicResourceSetSchedule"
            )
        for resource_id, schedule in self.resource_set_schedule.resources:
            for allocation in schedule.allocations:
                allocation_end = (
                    allocation.start_subcarrier
                    + allocation.bandwidth.allocated_bandwidth_subcarriers
                )
                if (
                    allocation.start_subcarrier < self.frequency_bounds.start_subcarrier
                    or allocation_end > self.frequency_bounds.end_subcarrier_exclusive
                ):
                    raise ValueError(
                        f"SRS allocation for {self.cell_id!r} resource {resource_id} "
                        f"lies outside selected UL BWP {self.ul_bwp_id}"
                    )
        object.__setattr__(self, "cell_id", self.cell_id.strip())
        object.__setattr__(self, "ul_bwp_id", int(self.ul_bwp_id))


@dataclass(frozen=True, slots=True)
class NRSRSBWPCellPeriodicResourceSetSchedules:
    """A cell's active-BWP context and its active periodic SRS resource sets."""

    cell_id: str
    ul_bwp_id: int
    frequency_bounds: NRSRSBWPFrequencyBounds
    resource_set_schedules: tuple[NRSRSBWPPeriodicResourceSetSchedule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonempty string")
        if (
            isinstance(self.ul_bwp_id, bool)
            or not isinstance(self.ul_bwp_id, Integral)
            or not 0 <= self.ul_bwp_id <= 4
        ):
            raise ValueError("ul_bwp_id must be in [0, 4]")
        if not isinstance(self.frequency_bounds, NRSRSBWPFrequencyBounds):
            raise TypeError("frequency_bounds must be NRSRSBWPFrequencyBounds")
        schedules = tuple(self.resource_set_schedules)
        if any(
            not isinstance(item, NRSRSBWPPeriodicResourceSetSchedule)
            for item in schedules
        ):
            raise TypeError(
                "resource_set_schedules must contain BWP-validated periodic sets"
            )
        if any(
            item.cell_id != self.cell_id.strip()
            or item.ul_bwp_id != int(self.ul_bwp_id)
            or item.frequency_bounds != self.frequency_bounds
            for item in schedules
        ):
            raise ValueError("periodic resource sets must share the cell's active-BWP binding")
        resource_set_ids = tuple(
            item.resource_set_schedule.resource_set_id for item in schedules
        )
        if len(set(resource_set_ids)) != len(resource_set_ids):
            raise ValueError("periodic SRS resource-set IDs must be unique within a cell")
        object.__setattr__(self, "cell_id", self.cell_id.strip())
        object.__setattr__(self, "ul_bwp_id", int(self.ul_bwp_id))
        object.__setattr__(self, "resource_set_schedules", schedules)


@dataclass(frozen=True, slots=True)
class NRSRSScheduledResourceOccurrence:
    """One resolved SRS resource on a shared, caller-defined OFDM grid.

    ``start_subcarrier`` values in allocations must use the same absolute
    subcarrier coordinate for every occurrence with the same ``grid_id``.
    ``target_slot_index`` is the linear slot index in that common numerology.
    One allocation may be broadcast over all SRS symbols, or a distinct
    allocation may be supplied per symbol.
    """

    cell_id: str
    resource_set_id: int
    resource_id: int
    grid_id: str
    carrier_hz: float
    numerology: int
    subcarrier_spacing_hz: float
    target_slot_index: int
    slot_symbol_count: int
    start_symbol_index: int
    srs_symbol_count: int
    allocations: tuple[NRSRSFixedFrequencyAllocation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonempty string")
        if not isinstance(self.grid_id, str) or not self.grid_id.strip():
            raise ValueError("grid_id must be a nonempty string")
        for name in (
            "resource_set_id",
            "resource_id",
            "numerology",
            "target_slot_index",
            "slot_symbol_count",
            "start_symbol_index",
            "srs_symbol_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.resource_set_id < 0 or self.resource_id < 0:
            raise ValueError("SRS resource and resource-set IDs must be nonnegative")
        if not 0 <= self.numerology <= 6:
            raise ValueError("numerology must be between 0 and 6")
        if self.target_slot_index < 0:
            raise ValueError("target_slot_index must be nonnegative")
        if self.slot_symbol_count not in {12, 14}:
            raise ValueError("slot_symbol_count must be 12 or 14")
        if (
            self.srs_symbol_count not in _SRS_SYMBOL_COUNTS
            or self.srs_symbol_count > self.slot_symbol_count
        ):
            raise ValueError("srs_symbol_count is unsupported for the slot")
        allocations = tuple(self.allocations)
        if not allocations or any(
            not isinstance(item, NRSRSFixedFrequencyAllocation) for item in allocations
        ):
            raise ValueError("allocations must contain fixed SRS frequency allocations")
        if len(allocations) not in {1, self.srs_symbol_count}:
            raise ValueError("allocations must be shared or provided once per SRS symbol")
        if not 0 <= self.start_symbol_index < self.slot_symbol_count:
            raise ValueError("start_symbol_index must lie inside the slot")
        if self.start_symbol_index + self.srs_symbol_count > self.slot_symbol_count:
            raise ValueError("SRS symbols extend past the slot")
        if isinstance(self.carrier_hz, bool) or not isinstance(self.carrier_hz, Real):
            raise TypeError("carrier_hz must be a finite positive number")
        carrier_hz = float(self.carrier_hz)
        if not np.isfinite(carrier_hz) or carrier_hz <= 0.0:
            raise ValueError("carrier_hz must be a finite positive number")
        if isinstance(self.subcarrier_spacing_hz, bool) or not isinstance(
            self.subcarrier_spacing_hz, Real
        ):
            raise TypeError("subcarrier_spacing_hz must be a finite positive number")
        spacing_hz = float(self.subcarrier_spacing_hz)
        expected_spacing_hz = 15_000.0 * (2**int(self.numerology))
        if not np.isfinite(spacing_hz) or spacing_hz != expected_spacing_hz:
            raise ValueError("subcarrier_spacing_hz disagrees with numerology")
        for name in (
            "resource_set_id",
            "resource_id",
            "numerology",
            "target_slot_index",
            "slot_symbol_count",
            "start_symbol_index",
            "srs_symbol_count",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "cell_id", self.cell_id.strip())
        object.__setattr__(self, "grid_id", self.grid_id.strip())
        object.__setattr__(self, "carrier_hz", carrier_hz)
        object.__setattr__(self, "subcarrier_spacing_hz", spacing_hz)
        object.__setattr__(self, "allocations", allocations)


def audit_nr_srs_intercell_re_overlap(
    occurrences: Sequence[NRSRSScheduledResourceOccurrence],
) -> dict[str, Any]:
    """Report shared SRS RE occupancy on one common caller-defined OFDM grid.

    This is a scheduling diagnostic only. It does not evaluate cyclic-shift
    orthogonality, power, propagation, spectral leakage, or receiver impact.
    """
    if isinstance(occurrences, (str, bytes)) or not isinstance(occurrences, Sequence):
        raise ValueError("occurrences must be a sequence of scheduled SRS resources")
    normalized = tuple(occurrences)
    if not normalized or any(
        not isinstance(item, NRSRSScheduledResourceOccurrence) for item in normalized
    ):
        raise ValueError("occurrences must be a nonempty sequence of typed SRS resources")
    contexts = {
        (item.grid_id, item.carrier_hz, item.numerology, item.subcarrier_spacing_hz)
        for item in normalized
    }
    if len(contexts) != 1:
        raise ValueError("one audit requires a single common grid, carrier, and numerology")
    identities = [
        (item.cell_id, item.resource_set_id, item.resource_id, item.target_slot_index)
        for item in normalized
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("scheduled SRS occurrence identities must be unique")
    expanded_count = sum(
        sum(
            item.allocations[
                0 if len(item.allocations) == 1 else symbol
            ].bandwidth.sequence_length
            for symbol in range(item.srs_symbol_count)
        )
        for item in normalized
    )
    if expanded_count > _MAX_SRS_INTERCELL_AUDIT_RESOURCE_ELEMENTS:
        raise ValueError("SRS overlap audit exceeds its resource-element limit")

    occupancy: dict[tuple[int, int, int], list[int]] = {}
    for occurrence_index, item in enumerate(normalized):
        for symbol_offset in range(item.srs_symbol_count):
            allocation = item.allocations[0 if len(item.allocations) == 1 else symbol_offset]
            symbol_index = item.start_symbol_index + symbol_offset
            for sequence_index in range(allocation.bandwidth.sequence_length):
                subcarrier_index = (
                    allocation.start_subcarrier
                    + allocation.comb_offset
                    + sequence_index * allocation.bandwidth.transmission_comb
                )
                coordinate = (item.target_slot_index, symbol_index, subcarrier_index)
                occupancy.setdefault(coordinate, []).append(occurrence_index)

    collision_coordinates: list[dict[str, Any]] = []
    pair_counts: dict[tuple[int, int], int] = {}
    overlap_count = 0
    for (slot, symbol, subcarrier), occurrence_indices in sorted(occupancy.items()):
        if len({normalized[index].cell_id for index in occurrence_indices}) < 2:
            continue
        overlap_count += 1
        participants = sorted(
            set(occurrence_indices),
            key=lambda index: (
                normalized[index].cell_id,
                normalized[index].resource_set_id,
                normalized[index].resource_id,
            ),
        )
        for left_position, left in enumerate(participants):
            for right in participants[left_position + 1 :]:
                if normalized[left].cell_id == normalized[right].cell_id:
                    continue
                pair = (min(left, right), max(left, right))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(collision_coordinates) < _MAX_SRS_INTERCELL_AUDIT_COORDINATES:
            collision_coordinates.append(
                {
                    "slot_index": slot,
                    "symbol_index": symbol,
                    "subcarrier_index": subcarrier,
                    "participants": [
                        {
                            "cell_id": normalized[index].cell_id,
                            "resource_set_id": normalized[index].resource_set_id,
                            "resource_id": normalized[index].resource_id,
                        }
                        for index in participants
                    ],
                }
            )

    grid_id, carrier_hz, numerology, spacing_hz = next(iter(contexts))
    pairwise = [
        {
            "left": {
                "cell_id": normalized[left].cell_id,
                "resource_set_id": normalized[left].resource_set_id,
                "resource_id": normalized[left].resource_id,
            },
            "right": {
                "cell_id": normalized[right].cell_id,
                "resource_set_id": normalized[right].resource_set_id,
                "resource_id": normalized[right].resource_id,
            },
            "shared_resource_element_count": count,
        }
        for (left, right), count in sorted(
            pair_counts.items(),
            key=lambda entry: (
                normalized[entry[0][0]].cell_id,
                normalized[entry[0][1]].cell_id,
                normalized[entry[0][0]].resource_set_id,
                normalized[entry[0][0]].resource_id,
                normalized[entry[0][1]].resource_set_id,
                normalized[entry[0][1]].resource_id,
            ),
        )
    ]
    return {
        "schema": "loslab.nr_srs_intercell_re_overlap.v1",
        "status": "INTERCELL_RE_OVERLAP_DETECTED"
        if overlap_count
        else "NO_INTERCELL_RE_OVERLAP",
        "admission_possible": False,
        "grid": {
            "grid_id": grid_id,
            "carrier_hz": carrier_hz,
            "numerology": numerology,
            "subcarrier_spacing_hz": spacing_hz,
        },
        "scheduled_occurrence_count": len(normalized),
        "expanded_resource_element_count": expanded_count,
        "shared_intercell_resource_element_count": overlap_count,
        "pairwise_resource_overlaps": pairwise,
        "collision_coordinates": collision_coordinates,
        "collision_coordinates_truncated": overlap_count > len(collision_coordinates),
        "limitations": [
            "Shared RE occupancy is not proof of interference; cyclic-shift and port orthogonality are not evaluated.",
            "Power, propagation, synchronization error, spectral leakage, and receiver impact are not evaluated.",
            "Grid identity and absolute subcarrier coordinates are caller supplied and must be common across cells.",
            "Linear slot indices must share one time origin across all scheduled cells.",
        ],
    }


def nr_srs_type1_bandwidth_configuration(
    c_srs: int, *, transmission_comb: int, b_srs: int = 0
) -> NRSRSBandwidthConfiguration:
    """Resolve the TS 38.211 type-1 SRS bandwidth table entry.

    ``b_srs`` selects one of the four standardized bandwidth levels. The
    returned extent does not by itself derive frequency position or resource
    hopping; :func:`nr_srs_fixed_frequency_allocation` resolves one position,
    using an explicit SRS transmission counter when hopping is enabled.
    """

    if isinstance(c_srs, bool) or not isinstance(c_srs, Integral):
        raise TypeError("c_srs must be an integer")
    c_srs = int(c_srs)
    if not 0 <= c_srs < len(_SRS_BANDWIDTH_TABLE):
        raise ValueError("c_srs must be in [0, 64)")
    if isinstance(b_srs, bool) or not isinstance(b_srs, Integral):
        raise TypeError("b_srs must be an integer")
    b_srs = int(b_srs)
    if not 0 <= b_srs <= 3:
        raise ValueError("b_srs must be in [0, 3]")
    srs_cyclic_shift_count(transmission_comb)
    bandwidth_rb, subband_count = _SRS_BANDWIDTH_TABLE[c_srs][b_srs]
    allocated_bandwidth_subcarriers = bandwidth_rb * 12
    if allocated_bandwidth_subcarriers % transmission_comb:
        raise ValueError("standardized SRS bandwidth must contain an integer sequence length")
    return NRSRSBandwidthConfiguration(
        c_srs=c_srs,
        b_srs=b_srs,
        transmission_comb=int(transmission_comb),
        bandwidth_rb=bandwidth_rb,
        subband_count=subband_count,
        sequence_length=allocated_bandwidth_subcarriers // int(transmission_comb),
        allocated_bandwidth_subcarriers=allocated_bandwidth_subcarriers,
    )


def nr_srs_positioning_symbol_offset(
    transmission_comb: int,
    *,
    srs_symbol_count: int,
    resource_symbol_index: int,
) -> int:
    """Resolve ``k'_offset`` for one SRS-PosResource symbol.

    TS 38.211 Table 6.4.1.4.3-2 applies this offset to SRS-PosResource. An
    ordinary SRS-Resource uses zero and should not call this helper. Symbol
    counts unsupported by the table are rejected rather than approximated.
    """

    for name, value in (
        ("transmission_comb", transmission_comb),
        ("srs_symbol_count", srs_symbol_count),
        ("resource_symbol_index", resource_symbol_index),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    transmission_comb = int(transmission_comb)
    srs_symbol_count = int(srs_symbol_count)
    resource_symbol_index = int(resource_symbol_index)
    if transmission_comb not in _SRS_POSITIONING_SYMBOL_OFFSETS:
        raise ValueError("transmission_comb must be 2, 4, or 8")
    offsets = _SRS_POSITIONING_SYMBOL_OFFSETS[transmission_comb].get(srs_symbol_count)
    if offsets is None:
        raise ValueError("symbol count is unsupported for this SRS-PosResource comb")
    if not 0 <= resource_symbol_index < srs_symbol_count:
        raise ValueError("resource_symbol_index must be inside the SRS resource")
    return offsets[resource_symbol_index]


def nr_srs_frequency_position_indices(
    c_srs: int,
    *,
    b_srs: int,
    b_hop: int,
    frequency_domain_position: int = 0,
    n_srs: int | None = None,
) -> tuple[int, ...]:
    """Resolve type-1 SRS frequency-tree indices for one transmission.

    ``n_srs`` is the TS 38.211 transmission counter. Periodicity and
    higher-layer scheduling are intentionally not inferred by this primitive.
    """

    for name, value in (
        ("c_srs", c_srs),
        ("b_srs", b_srs),
        ("b_hop", b_hop),
        ("frequency_domain_position", frequency_domain_position),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    c_srs = int(c_srs)
    b_srs = int(b_srs)
    b_hop = int(b_hop)
    frequency_domain_position = int(frequency_domain_position)
    if not 0 <= c_srs < len(_SRS_BANDWIDTH_TABLE):
        raise ValueError("c_srs must be in [0, 64)")
    if not 0 <= b_srs <= 3:
        raise ValueError("b_srs must be in [0, 3]")
    if not 0 <= b_hop <= 3:
        raise ValueError("b_hop must be in [0, 3]")
    if frequency_domain_position < 0:
        raise ValueError("frequency_domain_position must be nonnegative")
    if n_srs is not None:
        if isinstance(n_srs, bool) or not isinstance(n_srs, Integral):
            raise TypeError("n_srs must be an integer")
        n_srs = int(n_srs)
        if n_srs < 0:
            raise ValueError("n_srs must be nonnegative")
    hopping = b_hop < b_srs
    if hopping and n_srs is None:
        raise ValueError("n_srs is required when SRS frequency hopping is enabled")

    table_row = _SRS_BANDWIDTH_TABLE[c_srs]
    radices = [table_row[level][1] for level in range(b_srs + 1)]
    if hopping:
        # TS 38.211 fixes N_(b_hop)=1 for the hopping recurrence.
        radices[b_hop] = 1
    indices: list[int] = []
    product_previous = 1
    for level in range(b_srs + 1):
        bandwidth_rb = table_row[level][0]
        radix = radices[level]
        base_position = (4 * frequency_domain_position // bandwidth_rb) % radix
        if not hopping or level <= b_hop:
            indices.append(base_position)
        else:
            assert n_srs is not None
            product_through_level = product_previous * radix
            if radix % 2 == 0:
                hop_position = (radix // 2) * (
                    (n_srs % product_through_level) // product_previous
                ) + (n_srs % product_through_level) // (2 * product_previous)
            else:
                hop_position = (radix // 2) * (n_srs // product_previous)
            indices.append((base_position + hop_position) % radix)
        product_previous *= radix
    return tuple(indices)


def nr_srs_transmission_counter(
    *,
    resource_type: str,
    srs_symbol_count: int,
    resource_symbol_index: int,
    repetition_factor: int | None = None,
    ports8_tdm: bool = False,
    absolute_slot_index: int | None = None,
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> int | None:
    """Resolve the TS 38.211 ``n_SRS`` counter from resolved scheduling data.

    ``absolute_slot_index`` is the linear slot number within the radio-frame
    sequence, while periodicity and offset are already resolved in slots by
    the caller. ``None`` means the periodic/semi-persistent resource does not
    transmit in the supplied slot. For periodic positioning SRS, the optional
    Hyper-SFN index applies the TS 38.211 long-period gate using the supplied
    numerology and hyper-frame number.
    """

    if resource_type not in {"aperiodic", "periodic", "semi-persistent"}:
        raise ValueError("resource_type must be aperiodic, periodic, or semi-persistent")
    for name, value in (
        ("srs_symbol_count", srs_symbol_count),
        ("resource_symbol_index", resource_symbol_index),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    srs_symbol_count = int(srs_symbol_count)
    resource_symbol_index = int(resource_symbol_index)
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS:
        raise ValueError("srs_symbol_count is unsupported")
    if not 0 <= resource_symbol_index < srs_symbol_count:
        raise ValueError("resource_symbol_index must be inside the SRS resource")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    if not isinstance(positioning_resource, bool):
        raise TypeError("positioning_resource must be a boolean")
    port_factor = 2 if ports8_tdm else 1
    if srs_symbol_count % port_factor:
        raise ValueError("SRS symbol count must be divisible by the port factor")
    symbols_per_port_group = srs_symbol_count // port_factor
    if repetition_factor is None:
        repetition_factor = symbols_per_port_group
    elif isinstance(repetition_factor, bool) or not isinstance(repetition_factor, Integral):
        raise TypeError("repetition_factor must be an integer")
    repetition_factor = int(repetition_factor)
    if (
        repetition_factor < 1
        or repetition_factor > symbols_per_port_group
        or symbols_per_port_group % repetition_factor
    ):
        raise ValueError("repetition_factor must divide the SRS symbols per port group")
    counter_per_resource = symbols_per_port_group // repetition_factor
    symbol_counter = resource_symbol_index // (port_factor * repetition_factor)
    if resource_type == "aperiodic":
        if any(value is not None for value in (absolute_slot_index, periodicity_slots, slot_offset)):
            raise ValueError("aperiodic SRS does not use periodic slot parameters")
        if any(
            value is not None for value in (numerology, hyper_frame_number, hyper_sfn_index)
        ):
            raise ValueError("aperiodic SRS does not use Hyper-SFN parameters")
        return symbol_counter

    for name, value in (
        ("absolute_slot_index", absolute_slot_index),
        ("periodicity_slots", periodicity_slots),
        ("slot_offset", slot_offset),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer for periodic SRS")
    assert absolute_slot_index is not None
    assert periodicity_slots is not None
    assert slot_offset is not None
    absolute_slot_index = int(absolute_slot_index)
    periodicity_slots = int(periodicity_slots)
    slot_offset = int(slot_offset)
    if (
        absolute_slot_index < 0
        or periodicity_slots < 1
        or not 0 <= slot_offset < periodicity_slots
    ):
        raise ValueError("periodic SRS requires nonnegative slots and offset inside its period")
    elapsed_slots = absolute_slot_index - slot_offset
    if hyper_sfn_index is not None:
        if not positioning_resource:
            raise ValueError("Hyper-SFN gating is only defined for positioning SRS")
        if isinstance(numerology, bool) or not isinstance(numerology, Integral):
            raise TypeError("numerology must be an integer when Hyper-SFN gating is configured")
        if not 0 <= int(numerology) <= 6:
            raise ValueError("numerology must be between 0 and 6")
        if periodicity_slots < (1 << int(numerology)) * 10240:
            raise ValueError(
                "Hyper-SFN index requires a period of at least 2^mu * 10240 slots"
            )
        if isinstance(hyper_frame_number, bool) or not isinstance(
            hyper_frame_number, Integral
        ):
            raise TypeError("hyper_frame_number must be a nonnegative integer")
        if not 0 <= int(hyper_frame_number) <= 1023:
            raise ValueError("hyper_frame_number must be in [0, 1023]")
        if isinstance(hyper_sfn_index, bool) or not isinstance(hyper_sfn_index, Integral):
            raise TypeError("hyper_sfn_index must be 0 or 1")
        if hyper_sfn_index not in {0, 1}:
            raise ValueError("hyper_sfn_index must be 0 or 1")
        if (int(hyper_frame_number) + int(hyper_sfn_index)) % 2:
            return None
    elif numerology is not None or hyper_frame_number is not None:
        raise ValueError("numerology and hyper_frame_number require hyper_sfn_index")
    if elapsed_slots < 0 or elapsed_slots % periodicity_slots:
        return None
    occasion_index = elapsed_slots // periodicity_slots
    return occasion_index * counter_per_resource + symbol_counter


def nr_srs_periodicity_and_offset_slots(
    periodicity_and_offset: Mapping[str, object] | None,
    *,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
) -> tuple[int, int]:
    """Decode an RRC SRS periodicity CHOICE into slot period and offset.

    The arguments are single-entry mappings as commonly produced by ASN.1
    decoders, for example ``{"sl20": 7}``; ``sl1`` uses the ASN.1 NULL value
    ``None``. Release 16's extension CHOICE is accepted separately and takes
    precedence over the base CHOICE, as specified for the paired RRC fields.
    This resolves only the time-domain CHOICE; it does not parse RRC messages,
    decide set-level triggering, or apply positioning Hyper-SFN gating.
    """

    selected = (
        (periodicity_and_offset_ext, _SRS_PERIODICITY_EXT_OFFSETS)
        if periodicity_and_offset_ext is not None
        else (periodicity_and_offset, _SRS_PERIODICITY_OFFSETS)
    )
    choice, valid_choices = selected
    if not isinstance(choice, Mapping) or len(choice) != 1:
        raise ValueError("SRS periodicity CHOICE must contain exactly one alternative")
    name, offset = next(iter(choice.items()))
    if not isinstance(name, str):
        raise TypeError("SRS periodicity alternative name must be a string")
    normalized_name = name.casefold()
    if normalized_name not in valid_choices:
        raise ValueError("unsupported SRS periodicity CHOICE alternative")
    period_slots = valid_choices[normalized_name]
    if normalized_name == "sl1":
        if offset is not None:
            raise ValueError("sl1 SRS periodicity uses the ASN.1 NULL value")
        return period_slots, 0
    if isinstance(offset, bool) or not isinstance(offset, Integral):
        raise TypeError("SRS periodicity offset must be an integer")
    offset_slots = int(offset)
    if not 0 <= offset_slots < period_slots:
        raise ValueError("SRS periodicity offset must be inside its period")
    return period_slots, offset_slots


def nr_srs_aperiodic_target_slot_index(
    triggering_dci_slot: int,
    *,
    slot_offset: int | None = None,
    available_slot_offsets: Sequence[int] | None = None,
    selected_available_offset_index: int | None = None,
) -> int:
    """Resolve the target slot for one triggered aperiodic SRS resource set.

    The optional list contains offsets measured from the reference slot
    ``triggering_dci_slot + slot_offset``. The selected index is supplied by
    the caller after interpreting the applicable DCI/resource-set trigger
    mapping; this function does not decode that mapping.
    """

    if isinstance(triggering_dci_slot, bool) or not isinstance(triggering_dci_slot, Integral):
        raise TypeError("triggering_dci_slot must be an integer")
    if triggering_dci_slot < 0:
        raise ValueError("triggering_dci_slot must be nonnegative")
    if slot_offset is None:
        slot_offset = 0
    elif isinstance(slot_offset, bool) or not isinstance(slot_offset, Integral):
        raise TypeError("slot_offset must be an integer when configured")
    elif not 1 <= slot_offset <= 32:
        raise ValueError("configured slot_offset must be in [1, 32]")
    if available_slot_offsets is None:
        if selected_available_offset_index is not None:
            raise ValueError("an available offset index requires an available-slot-offset list")
        selected_offset = 0
    else:
        if (
            not isinstance(available_slot_offsets, Sequence)
            or isinstance(available_slot_offsets, (str, bytes))
            or not 1 <= len(available_slot_offsets) <= 4
        ):
            raise ValueError("available_slot_offsets must contain one to four values")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in available_slot_offsets
        ):
            raise ValueError("available slot offsets must be integers in [0, 7]")
        if any(value > 7 for value in available_slot_offsets):
            raise ValueError("available slot offsets must be in [0, 7]")
        if len(set(available_slot_offsets)) != len(available_slot_offsets):
            raise ValueError("available slot offsets must be distinct")
        if (
            isinstance(selected_available_offset_index, bool)
            or not isinstance(selected_available_offset_index, Integral)
        ):
            raise TypeError("selected_available_offset_index must be an integer")
        if not 0 <= selected_available_offset_index < len(available_slot_offsets):
            raise ValueError("selected available offset index is outside the configured list")
        selected_offset = int(available_slot_offsets[int(selected_available_offset_index)])
    return int(triggering_dci_slot) + int(slot_offset) + selected_offset


def nr_srs_triggered_resource_set_ids(
    srs_request_codepoint: int,
    resource_sets: Sequence[Mapping[str, object]],
    *,
    dci_format: str,
    srs_tpc_pdcch_group: str = "typeB",
    srs_request_mode: int | None = None,
    supplementary_uplink: bool = False,
) -> tuple[int, ...]:
    """Resolve a DCI SRS-request bit field to configured aperiodic sets.

    For DCI 0_1/1_1, the two low bits select triggers 0..3. For DCI
    0_2/1_2, ``srs_request_mode`` is the configured ``srs-RequestDCI-*``
    value (1 selects the one-bit table; 2 selects the two-bit table). When
    supplementary uplink is configured, the most-significant field bit
    selects non-SUL (0) or SUL (1), and the remaining bits carry the request.
    Resource-set records may declare ``ul_carrier`` as ``non_sul`` or
    ``sul``; this field is required when supplementary uplink is enabled.
    Resource-set records also use ``resource_set_id``, ``resource_type``,
    ``aperiodic_trigger``, and optional ``aperiodic_trigger_list``. Complete
    DCI/RRC parsing, typeA handling, and decoded multi-cell request-list
    indirection are handled outside this single-cell resolver.
    """

    if not isinstance(dci_format, str) or dci_format not in _SRS_REQUEST_DCI_FORMATS:
        raise ValueError("unsupported DCI format for this SRS request resolver")
    if not isinstance(srs_tpc_pdcch_group, str) or srs_tpc_pdcch_group != "typeB":
        raise ValueError("only typeB SRS TPC PDCCH group is supported")
    if not isinstance(supplementary_uplink, bool):
        raise TypeError("supplementary_uplink must be a boolean")
    if dci_format in {"0_2", "1_2"}:
        if isinstance(srs_request_mode, bool) or not isinstance(srs_request_mode, Integral):
            raise ValueError("DCI 0_2/1_2 requires configured srs_request_mode 1 or 2")
        srs_request_mode = int(srs_request_mode)
        if srs_request_mode not in {1, 2}:
            raise ValueError("srs_request_mode must be 1 or 2 for DCI 0_2/1_2")
        request_bit_count = srs_request_mode
    else:
        if srs_request_mode is not None:
            raise ValueError("srs_request_mode applies only to DCI 0_2/1_2")
        request_bit_count = 2
    if isinstance(srs_request_codepoint, bool) or not isinstance(
        srs_request_codepoint, Integral
    ):
        raise TypeError("srs_request_codepoint must be an integer")
    codepoint = int(srs_request_codepoint)
    field_bit_count = request_bit_count + int(supplementary_uplink)
    maximum_codepoint = (1 << field_bit_count) - 1
    if not 0 <= codepoint <= maximum_codepoint:
        raise ValueError(
            f"{field_bit_count}-bit SRS request field must be in [0, {maximum_codepoint}] "
            f"for DCI format {dci_format}"
        )
    carrier = "non_sul"
    if supplementary_uplink:
        carrier = "sul" if codepoint >> request_bit_count else "non_sul"
    request = codepoint & ((1 << request_bit_count) - 1)
    if not isinstance(resource_sets, Sequence) or isinstance(resource_sets, (str, bytes)):
        raise TypeError("resource_sets must be a sequence of decoded resource-set mappings")

    seen_ids: set[int] = set()
    matched_ids: list[int] = []
    for index, resource_set in enumerate(resource_sets):
        if not isinstance(resource_set, Mapping):
            raise TypeError(f"resource_sets[{index}] must be a mapping")
        resource_set_id = resource_set.get("resource_set_id")
        if isinstance(resource_set_id, bool) or not isinstance(resource_set_id, Integral):
            raise TypeError(f"resource_sets[{index}].resource_set_id must be an integer")
        resource_set_id = int(resource_set_id)
        if resource_set_id < 0 or resource_set_id in seen_ids:
            raise ValueError("resource-set IDs must be unique nonnegative integers")
        seen_ids.add(resource_set_id)
        resource_type = resource_set.get("resource_type")
        if not isinstance(resource_type, str) or resource_type not in {
            "aperiodic",
            "periodic",
            "semi-persistent",
        }:
            raise ValueError(f"resource_sets[{index}].resource_type is unsupported")
        if resource_type != "aperiodic":
            continue
        configured_carrier = resource_set.get("ul_carrier")
        if supplementary_uplink and (
            not isinstance(configured_carrier, str)
            or configured_carrier not in {"non_sul", "sul"}
        ):
            raise ValueError(
                f"resource_sets[{index}].ul_carrier must identify non_sul or sul"
            )
        if configured_carrier is not None and (
            not isinstance(configured_carrier, str)
            or configured_carrier not in {"non_sul", "sul"}
        ):
            raise ValueError(f"resource_sets[{index}].ul_carrier is invalid")
        if configured_carrier is not None and configured_carrier != carrier:
            continue
        primary_trigger = resource_set.get("aperiodic_trigger")
        if isinstance(primary_trigger, bool) or not isinstance(primary_trigger, Integral):
            raise TypeError(f"resource_sets[{index}].aperiodic_trigger must be an integer")
        primary_trigger = int(primary_trigger)
        if not 1 <= primary_trigger <= 3:
            raise ValueError(f"resource_sets[{index}].aperiodic_trigger must be in [1, 3]")
        additional = resource_set.get("aperiodic_trigger_list", ())
        if additional is None:
            additional = ()
        if not isinstance(additional, Sequence) or isinstance(additional, (str, bytes)):
            raise TypeError(f"resource_sets[{index}].aperiodic_trigger_list must be a sequence")
        triggers = [primary_trigger]
        for value in additional:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("aperiodic trigger-list entries must be integers")
            value = int(value)
            if not 1 <= value <= 3:
                raise ValueError("aperiodic trigger-list entries must be in [1, 3]")
            triggers.append(value)
        if request in triggers:
            matched_ids.append(resource_set_id)
    return tuple(matched_ids)


def nr_srs_dci_0_3_ul_bwp_selection(
    scheduled_cell_ids: Sequence[str],
    configured_ul_bwp_ids_by_cell: Mapping[str, Sequence[int]],
    active_ul_bwp_id_by_cell: Mapping[str, int],
    *,
    bwp_indicator: int | None,
    supports_active_bwp_change: bool = True,
    initial_bwp_id: int = 0,
) -> NRSRSBWPSelection:
    """Resolve the decoded DCI 0_3 BWP indicator independently per cell.

    The configuration maps cover the full ``scheduledCellListDCI-0-3`` for
    the selected cell set, not just the combination selected by DCI. Each
    configured-ID tuple includes the initial UL BWP and is normalized in
    ascending BWP-ID order. For up to three noninitial BWPs, the indicator
    indexes that ordered BWP list. With four noninitial BWPs, TS 38.212
    Table 7.3.1.1.2-1 maps codepoints 0..3 to BWP IDs 1..4. A codepoint with
    no corresponding BWP in an individual scheduled cell leaves that cell on
    its active BWP. A UE that does not support active BWP changes via DCI also
    remains on each cell's active BWP.

    This resolves a decoded field only; it does not parse DCI bits or perform
    resource-set lookup for the selected BWP.
    """
    if not isinstance(scheduled_cell_ids, Sequence) or isinstance(
        scheduled_cell_ids, (str, bytes)
    ):
        raise TypeError("scheduled_cell_ids must be an ordered sequence")
    cells = tuple(scheduled_cell_ids)
    if not cells or any(not isinstance(cell, str) or not cell for cell in cells):
        raise ValueError("scheduled_cell_ids must contain nonempty cell IDs")
    if len(set(cells)) != len(cells):
        raise ValueError("scheduled_cell_ids must be unique")
    for name, value in (
        ("configured_ul_bwp_ids_by_cell", configured_ul_bwp_ids_by_cell),
        ("active_ul_bwp_id_by_cell", active_ul_bwp_id_by_cell),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        if set(value) != set(cells):
            raise ValueError(f"{name} keys must exactly match scheduled_cell_ids")
    if not isinstance(supports_active_bwp_change, bool):
        raise TypeError("supports_active_bwp_change must be a boolean")
    if isinstance(initial_bwp_id, bool) or not isinstance(initial_bwp_id, Integral):
        raise TypeError("initial_bwp_id must be an integer")
    initial_bwp_id = int(initial_bwp_id)
    if not 0 <= initial_bwp_id <= 4:
        raise ValueError("initial_bwp_id must be in [0, 4]")

    configured_by_cell: dict[str, tuple[int, ...]] = {}
    active_by_cell: dict[str, int] = {}
    for cell in cells:
        raw_ids = configured_ul_bwp_ids_by_cell[cell]
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise TypeError(f"configured UL BWP IDs for {cell!r} must be an ordered sequence")
        bwp_ids = tuple(raw_ids)
        if not bwp_ids or any(
            isinstance(bwp_id, bool)
            or not isinstance(bwp_id, Integral)
            or not 0 <= bwp_id <= 4
            for bwp_id in bwp_ids
        ):
            raise ValueError(f"configured UL BWP IDs for {cell!r} must lie in [0, 4]")
        normalized_ids = tuple(sorted(int(bwp_id) for bwp_id in bwp_ids))
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(f"configured UL BWP IDs for {cell!r} must be unique")
        if initial_bwp_id not in normalized_ids:
            raise ValueError(f"configured UL BWP IDs for {cell!r} must include initial BWP")
        if len(normalized_ids) > 5:
            raise ValueError(f"configured UL BWP IDs for {cell!r} exceed the TS limit")
        active_id = active_ul_bwp_id_by_cell[cell]
        if (
            isinstance(active_id, bool)
            or not isinstance(active_id, Integral)
            or int(active_id) not in normalized_ids
        ):
            raise ValueError(f"active UL BWP for {cell!r} must be configured")
        configured_by_cell[cell] = normalized_ids
        active_by_cell[cell] = int(active_id)

    max_noninitial = max(
        len(bwp_ids) - 1 for bwp_ids in configured_by_cell.values()
    )
    indicator_domain_size = (
        max_noninitial + 1 if max_noninitial <= 3 else max_noninitial
    )
    bit_width = (indicator_domain_size - 1).bit_length()
    if bit_width == 0:
        if bwp_indicator is not None:
            raise ValueError("bwp_indicator must be absent when the field has zero bits")
    else:
        if isinstance(bwp_indicator, bool) or not isinstance(bwp_indicator, Integral):
            raise TypeError("bwp_indicator must be a decoded integer codepoint")
        if not 0 <= int(bwp_indicator) < (1 << bit_width):
            raise ValueError("bwp_indicator does not fit the configured field width")

    indicator_value = None if bwp_indicator is None else int(bwp_indicator)
    selected: list[tuple[str, int]] = []
    outcomes: list[tuple[str, str]] = []
    for cell in cells:
        bwp_ids = configured_by_cell[cell]
        active_id = active_by_cell[cell]
        outcome: str
        selected_id = active_id
        if bit_width == 0:
            outcome = "field_absent"
        elif len(bwp_ids) == 1:
            outcome = "single_bwp"
        elif not supports_active_bwp_change:
            outcome = "active_bwp_change_unsupported"
        else:
            if max_noninitial > 3:
                assert indicator_value is not None
                indicated_id = indicator_value + 1
            else:
                assert indicator_value is not None
                indicated_id = (
                    bwp_ids[indicator_value]
                    if indicator_value < len(bwp_ids)
                    else -1
                )
            if indicated_id in bwp_ids:
                selected_id = indicated_id
                outcome = "selected_by_indicator"
            else:
                outcome = "codepoint_not_configured_for_cell"
        selected.append((cell, selected_id))
        outcomes.append((cell, outcome))

    return NRSRSBWPSelection(bit_width, tuple(selected), tuple(outcomes))


def nr_srs_dci_0_3_scheduled_cell_ids(
    cell_set_configs: Sequence[Mapping[str, object]],
    *,
    scheduled_cell_set_indicator: int | None = None,
    scheduled_cells_indicator: int | None = None,
    default_set_id: int | None = None,
) -> tuple[str, ...]:
    """Resolve decoded DCI 0_3 cell-set and scheduled-cells indicators.

    Each normalized, already-decoded RRC configuration has ``set_id``,
    ``scheduled_cell_ids`` (the ordered ``scheduledCellListDCI-0-3``), and
    optional ``scheduled_cell_combos`` (the rows of
    ``scheduledCellComboListDCI-0-3`` as zero-based indexes into that list).
    A present cell-set indicator indexes configurations ordered by set ID. If
    that field is absent, a single configured set is selected implicitly;
    multiple configured sets require the caller to supply the configured
    ``default_set_id`` for the PDCCH cell/context. A scheduled-cells indicator
    indexes the configured combination rows; when no table exists, or it has
    one row, that indicator is absent and the implied row/list is used.

    The returned cell identifiers follow the RRC list order, which must
    already represent ascending serving-cell indexes. This function consumes
    decoded values; it does not parse DCI bits or ASN.1 PER.
    """
    if not isinstance(cell_set_configs, Sequence) or isinstance(
        cell_set_configs, (str, bytes)
    ):
        raise TypeError("cell_set_configs must be an ordered sequence")
    if not 1 <= len(cell_set_configs) <= 4:
        raise ValueError("cell_set_configs must contain 1..4 configured sets")

    normalized: list[tuple[int, tuple[str, ...], object]] = []
    seen_set_ids: set[int] = set()
    seen_cells: set[str] = set()
    for config in cell_set_configs:
        if not isinstance(config, Mapping):
            raise TypeError("each cell-set config must be a mapping")
        set_id = config.get("set_id")
        if isinstance(set_id, bool) or not isinstance(set_id, Integral):
            raise TypeError("each set_id must be an integer")
        set_id = int(set_id)
        if not 0 <= set_id <= 3 or set_id in seen_set_ids:
            raise ValueError("set_id values must be unique integers in [0, 3]")
        seen_set_ids.add(set_id)

        raw_cells = config.get("scheduled_cell_ids")
        if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
            raise TypeError("scheduled_cell_ids must be an ordered sequence")
        cells = tuple(raw_cells)
        if not 2 <= len(cells) <= 4 or any(
            not isinstance(cell, str) or not cell for cell in cells
        ):
            raise ValueError("scheduled_cell_ids must contain 2..4 nonempty IDs")
        if len(set(cells)) != len(cells):
            raise ValueError("scheduled_cell_ids must be unique within each set")
        if seen_cells.intersection(cells):
            raise ValueError("a scheduled cell cannot appear in more than one set")
        seen_cells.update(cells)

        combos = config.get("scheduled_cell_combos")
        normalized.append((set_id, cells, combos))

    normalized.sort(key=lambda item: item[0])
    if scheduled_cell_set_indicator is None:
        if default_set_id is None:
            if len(normalized) != 1:
                raise ValueError(
                    "multiple cell sets require scheduled_cell_set_indicator or default_set_id"
                )
            selected_set = normalized[0]
        else:
            if isinstance(default_set_id, bool) or not isinstance(default_set_id, Integral):
                raise TypeError("default_set_id must be an integer")
            matches = [item for item in normalized if item[0] == int(default_set_id)]
            if not matches:
                raise ValueError("default_set_id does not identify a configured set")
            selected_set = matches[0]
    else:
        if isinstance(scheduled_cell_set_indicator, bool) or not isinstance(
            scheduled_cell_set_indicator, Integral
        ):
            raise TypeError("scheduled_cell_set_indicator must be an integer")
        if default_set_id is not None:
            raise ValueError("default_set_id must be omitted when the set indicator is present")
        index = int(scheduled_cell_set_indicator)
        if not 0 <= index < len(normalized):
            raise ValueError("scheduled_cell_set_indicator selects no configured set")
        selected_set = normalized[index]

    _, cells, combos_obj = selected_set
    if combos_obj is None:
        if scheduled_cells_indicator is not None:
            raise ValueError(
                "scheduled_cells_indicator must be absent when no combo list is configured"
            )
        return cells
    if not isinstance(combos_obj, Sequence) or isinstance(combos_obj, (str, bytes)):
        raise TypeError("scheduled_cell_combos must be an ordered sequence")
    combos = tuple(combos_obj)
    if not 1 <= len(combos) <= 16:
        raise ValueError("scheduled_cell_combos must contain 1..16 rows")
    normalized_combos: list[tuple[int, ...]] = []
    for combo in combos:
        if not isinstance(combo, Sequence) or isinstance(combo, (str, bytes)):
            raise TypeError("each scheduled-cell combo must be an ordered index sequence")
        indices = tuple(combo)
        if not 1 <= len(indices) <= len(cells):
            raise ValueError("each scheduled-cell combo must select 1..N configured cells")
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices):
            raise TypeError("scheduled-cell combo indexes must be integers")
        index_values = tuple(int(i) for i in indices)
        if any(not 0 <= i < len(cells) for i in index_values):
            raise ValueError("scheduled-cell combo index is outside scheduled_cell_ids")
        if tuple(sorted(set(index_values))) != index_values:
            raise ValueError("scheduled-cell combo indexes must be unique and ascending")
        normalized_combos.append(index_values)
    if len(combos) == 1:
        if scheduled_cells_indicator is not None:
            raise ValueError("scheduled_cells_indicator is absent for a single combo row")
        combo_index = 0
    else:
        if isinstance(scheduled_cells_indicator, bool) or not isinstance(
            scheduled_cells_indicator, Integral
        ):
            raise TypeError("scheduled_cells_indicator must select a configured combo row")
        combo_index = int(scheduled_cells_indicator)
        if not 0 <= combo_index < len(combos):
            raise ValueError("scheduled_cells_indicator selects no configured combo row")

    return tuple(cells[i] for i in normalized_combos[combo_index])


def nr_srs_dci_0_3_request_list_index_from_bits(
    payload_bits: Sequence[int | bool],
    *,
    field_offset: int,
    request_list_entry_count: int,
) -> int | None:
    """Decode the DCI 0_3 SRS-request selector from an MSB-first payload.

    ``field_offset`` is the zero-based bit position in the already-decoded
    DCI information payload; determining the full DCI field layout remains
    the caller's responsibility. The selector width is ``ceil(log2(N))``
    for the configured ``srs-RequestListDCI-0-3`` length ``N``. With no list,
    the field is absent and this returns ``None``. A one-entry list has a
    zero-bit selector and resolves to entry zero. Reserved codepoints are
    rejected instead of silently aliasing a configured row.

    Bit positions follow the serialized payload order: index zero is the
    first (most-significant) bit. This function does not decode CRC, polar
    coding, or ASN.1 RRC configuration.
    """
    if not isinstance(payload_bits, Sequence) or isinstance(
        payload_bits, (str, bytes)
    ):
        raise TypeError("payload_bits must be a sequence of decoded bits")
    if isinstance(field_offset, bool) or not isinstance(field_offset, Integral):
        raise TypeError("field_offset must be an integer")
    if field_offset < 0:
        raise ValueError("field_offset must be nonnegative")
    if isinstance(request_list_entry_count, bool) or not isinstance(
        request_list_entry_count, Integral
    ):
        raise TypeError("request_list_entry_count must be an integer")
    entry_count = int(request_list_entry_count)
    if not 0 <= entry_count <= 16:
        raise ValueError("request_list_entry_count must be in [0, 16]")
    for index, bit in enumerate(payload_bits):
        if isinstance(bit, (bool, np.bool_)):
            continue
        if isinstance(bit, Integral) and int(bit) in {0, 1}:
            continue
        raise ValueError(f"payload_bits[{index}] must be a Boolean or integer bit")
    if int(field_offset) > len(payload_bits):
        raise ValueError("field_offset lies beyond the DCI payload")

    if entry_count == 0:
        return None
    bit_width = (entry_count - 1).bit_length()
    if bit_width == 0:
        return 0
    end = int(field_offset) + bit_width
    if end > len(payload_bits):
        raise ValueError("DCI payload does not contain the complete SRS request selector")
    selector = 0
    for bit in payload_bits[int(field_offset) : end]:
        selector = (selector << 1) | int(bit)
    if selector >= entry_count:
        raise ValueError("SRS request selector uses an unconfigured reserved codepoint")
    return selector


def nr_dci_0_3_field_values_from_layout(
    payload_bits: Sequence[int | bool],
    *,
    field_widths_bits: Mapping[str, int],
) -> dict[str, int | None]:
    """Decode all bounded MSB-first DCI 0_3 fields from a width roster.

    Widths must cover the complete canonical clause 7.3.1.1.4 field order,
    including zero-width absent fields. A nonzero field is returned as its
    unsigned integer codepoint; a zero-width field is returned as ``None``.
    Zero padding after the declared layout is accepted, while nonzero trailing
    bits, truncated payloads, malformed bits, and resource-heavy layouts fail
    closed. This handles the already-decoded DCI information payload only; it
    does not perform CRC/polar decoding, RRC/PER parsing, or field-specific
    configuration interpretation.
    """
    if not isinstance(payload_bits, Sequence) or isinstance(payload_bits, (str, bytes)):
        raise TypeError("payload_bits must be a sequence of decoded bits")
    if not isinstance(field_widths_bits, Mapping):
        raise TypeError("field_widths_bits must be a mapping")
    if any(not isinstance(name, str) for name in field_widths_bits):
        raise TypeError("DCI 0_3 field names must be strings")
    expected_names = set(_DCI_0_3_FIELD_ORDER)
    if set(field_widths_bits) != expected_names:
        missing = sorted(expected_names - set(field_widths_bits))
        extra = sorted(set(field_widths_bits) - expected_names)
        raise ValueError(
            f"DCI 0_3 field-width roster mismatch; missing={missing}, extra={extra}"
        )
    normalized_widths: dict[str, int] = {}
    total_width = 0
    for field_name in _DCI_0_3_FIELD_ORDER:
        width = field_widths_bits[field_name]
        if isinstance(width, bool) or not isinstance(width, Integral):
            raise TypeError(f"field width for {field_name} must be an integer")
        width = int(width)
        if not 0 <= width <= 512:
            raise ValueError(f"field width for {field_name} must be in [0, 512]")
        normalized_widths[field_name] = width
        total_width += width
    if total_width > 4096:
        raise ValueError("DCI 0_3 decoded layout exceeds the 4096-bit resource bound")
    if len(payload_bits) < total_width:
        raise ValueError("DCI payload is shorter than the declared format 0_3 field layout")
    if len(payload_bits) > total_width and any(payload_bits[total_width:]):
        raise ValueError("DCI payload has nonzero trailing bits beyond the declared layout")

    for index, bit in enumerate(payload_bits):
        if isinstance(bit, (bool, np.bool_)):
            continue
        if isinstance(bit, Integral) and int(bit) in {0, 1}:
            continue
        raise ValueError(f"payload_bits[{index}] must be a Boolean or integer bit")

    decoded: dict[str, int | None] = {}
    offset = 0
    for field_name in _DCI_0_3_FIELD_ORDER:
        width = normalized_widths[field_name]
        if width == 0:
            decoded[field_name] = None
            continue
        value = 0
        for bit in payload_bits[offset : offset + width]:
            value = (value << 1) | int(bit)
        decoded[field_name] = value
        offset += width
    return decoded


def nr_srs_dci_0_3_request_list_index_from_layout(
    payload_bits: Sequence[int | bool],
    *,
    field_widths_bits: Mapping[str, int],
    request_list_entry_count: int,
) -> int | None:
    """Locate and decode the SRS request selector using the ordered 0_3 layout.

    ``field_widths_bits`` must contain every field in the TS 38.212 clause
    7.3.1.1.4 order, including fields absent in a particular configuration as
    zero-width entries. Width derivation from decoded RRC/higher-layer state is
    caller supplied; this function owns the normative ordering, cumulative
    MSB-first offset, and selector-width consistency checks. Extra payload bits
    are accepted only as zero padding after the complete layout.
    """
    if isinstance(request_list_entry_count, bool) or not isinstance(
        request_list_entry_count, Integral
    ):
        raise TypeError("request_list_entry_count must be an integer")
    entry_count = int(request_list_entry_count)
    if not 0 <= entry_count <= 16:
        raise ValueError("request_list_entry_count must be in [0, 16]")
    decoded = nr_dci_0_3_field_values_from_layout(
        payload_bits,
        field_widths_bits=field_widths_bits,
    )
    expected_srs_width = (entry_count - 1).bit_length() if entry_count else 0
    if field_widths_bits["srs_request"] != expected_srs_width:
        raise ValueError(
            "srs_request field width does not match the configured request-list length"
        )
    if entry_count == 0:
        return None
    selector = decoded["srs_request"]
    if selector is None:
        return 0
    if selector >= entry_count:
        raise ValueError("SRS request selector uses an unconfigured reserved codepoint")
    return selector


def _nr_srs_dci_0_3_antenna_ports_width(
    *,
    mode: str,
    cell_ids: Sequence[str],
    transform_precoding_enabled_by_cell: Mapping[str, bool],
    dmrs_port_configuration_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    full_cell_roster: Sequence[str],
) -> int:
    """Derive DCI 0_3 antenna-port width from PUSCH DMRS configuration."""
    if not isinstance(mode, str) or mode not in {"type1a", "type2"}:
        raise ValueError("antenna_ports_mode must be 'type1a' or 'type2'")
    if not isinstance(transform_precoding_enabled_by_cell, Mapping) or not isinstance(
        dmrs_port_configuration_by_cell, Mapping
    ):
        raise TypeError("antenna-port configuration inputs must be mappings")
    if set(transform_precoding_enabled_by_cell) != set(full_cell_roster):
        raise ValueError(
            "transform_precoding_enabled_by_cell must cover the full scheduled cell list"
        )
    if set(dmrs_port_configuration_by_cell) != set(full_cell_roster):
        raise ValueError(
            "dmrs_port_configuration_by_cell must cover the full scheduled cell list"
        )
    widths: dict[str, int] = {}
    for cell in full_cell_roster:
        transform_precoding = transform_precoding_enabled_by_cell[cell]
        if type(transform_precoding) is not bool:
            raise TypeError("transform-precoding configuration values must be booleans")
        configs = dmrs_port_configuration_by_cell[cell]
        if not isinstance(configs, Sequence) or isinstance(configs, (str, bytes)):
            raise TypeError(f"DMRS port configurations for {cell!r} must be a sequence")
        if not 1 <= len(configs) <= 2:
            raise ValueError(
                f"DMRS port configurations for {cell!r} must contain one or two mapping types"
            )
        mapping_widths: list[int] = []
        mapping_types: set[str] = set()
        for config in configs:
            if not isinstance(config, Mapping):
                raise TypeError("each DMRS port configuration must be a mapping")
            if set(config) != {
                "mapping_type",
                "dmrs_type",
                "dmrs_type_enhanced",
                "max_length",
            }:
                raise ValueError("DMRS port configuration has an unsupported field roster")
            mapping_type = config["mapping_type"]
            if not isinstance(mapping_type, str) or mapping_type not in {"A", "B"}:
                raise ValueError("DMRS mapping_type must be 'A' or 'B'")
            if mapping_type in mapping_types:
                raise ValueError("DMRS mapping types must be unique per cell")
            mapping_types.add(mapping_type)
            dmrs_type = config["dmrs_type"]
            enhanced = config["dmrs_type_enhanced"]
            max_length = config["max_length"]
            if isinstance(dmrs_type, bool) or not isinstance(dmrs_type, Integral):
                raise TypeError("dmrs_type must be integer 1 or 2")
            if int(dmrs_type) not in {1, 2}:
                raise ValueError("dmrs_type must be integer 1 or 2")
            if type(enhanced) is not bool:
                raise TypeError("dmrs_type_enhanced must be a boolean")
            if isinstance(max_length, bool) or not isinstance(max_length, Integral):
                raise TypeError("max_length must be integer 1 or 2")
            if int(max_length) not in {1, 2}:
                raise ValueError("max_length must be integer 1 or 2")
            dmrs_type_value = int(dmrs_type)
            max_length_value = int(max_length)
            if transform_precoding:
                if dmrs_type_value != 1:
                    raise ValueError(
                        "transform precoding requires DMRS type 1 for antenna-port width"
                    )
                width = 2 * max_length_value
            else:
                width = (
                    2
                    + max_length_value
                    + int(dmrs_type_value == 2)
                    + int(enhanced)
                )
            mapping_widths.append(width)
        widths[cell] = max(mapping_widths)
    if mode == "type1a":
        return max((widths[cell] for cell in cell_ids), default=0)
    return sum(widths[cell] for cell in cell_ids)


def _nr_srs_dci_0_3_precoding_width(
    *,
    mode: str,
    cell_ids: Sequence[str],
    tx_config_by_cell: Mapping[str, str],
    transform_precoding_enabled_by_cell: Mapping[str, bool],
    codebook_srs_antenna_port_counts_by_cell: Mapping[str, Sequence[int]],
    codebook_subset_by_cell: Mapping[str, str | None],
    max_rank_for_codebook_by_cell: Mapping[str, int | None],
    ul_full_power_transmission_by_cell: Mapping[str, str | None],
    full_cell_roster: Sequence[str],
) -> int:
    """Derive and aggregate codebook TPMI widths from decoded RRC state."""
    if not isinstance(mode, str) or mode not in {"type1a", "type2"}:
        raise ValueError("tpmi_mode must be 'type1a' or 'type2'")
    configuration_maps = (
        tx_config_by_cell,
        transform_precoding_enabled_by_cell,
        codebook_srs_antenna_port_counts_by_cell,
        codebook_subset_by_cell,
        max_rank_for_codebook_by_cell,
        ul_full_power_transmission_by_cell,
    )
    if any(not isinstance(roster, Mapping) for roster in configuration_maps):
        raise TypeError("precoding configuration inputs must be mappings")
    for name, roster in zip(
        (
            "tx_config_by_cell",
            "transform_precoding_enabled_by_cell",
            "codebook_srs_antenna_port_counts_by_cell",
            "codebook_subset_by_cell",
            "max_rank_for_codebook_by_cell",
            "ul_full_power_transmission_by_cell",
        ),
        configuration_maps,
        strict=True,
    ):
        if set(roster) != set(full_cell_roster):
            raise ValueError(f"{name} must cover the full scheduled cell list")
    allowed_subsets = {
        "fullyAndPartialAndNonCoherent",
        "partialAndNonCoherent",
        "nonCoherent",
    }
    allowed_full_power = {None, "fullpower", "fullpowerMode1", "fullpowerMode2"}
    widths: dict[str, int] = {}
    for cell in full_cell_roster:
        tx_config = tx_config_by_cell[cell]
        if not isinstance(tx_config, str) or tx_config not in {
            "codebook",
            "nonCodebook",
        }:
            raise ValueError(
                f"tx_config_by_cell[{cell!r}] must be 'codebook' or 'nonCodebook'"
            )
        transform_precoding = transform_precoding_enabled_by_cell[cell]
        if type(transform_precoding) is not bool:
            raise TypeError("transform-precoding configuration values must be booleans")
        raw_ports = codebook_srs_antenna_port_counts_by_cell[cell]
        if not isinstance(raw_ports, Sequence) or isinstance(raw_ports, (str, bytes)):
            raise TypeError(f"codebook SRS antenna-port counts for {cell!r} must be a sequence")
        if tx_config == "nonCodebook":
            if (
                len(raw_ports) != 0
                or codebook_subset_by_cell[cell] is not None
                or max_rank_for_codebook_by_cell[cell] is not None
                or ul_full_power_transmission_by_cell[cell] is not None
            ):
                raise ValueError(
                    f"non-codebook cell {cell!r} must not have codebook-only configuration"
                )
            widths[cell] = 0
            continue
        if not raw_ports:
            raise ValueError(f"codebook cell {cell!r} must configure an SRS resource")
        ports: list[int] = []
        for raw_port_count in raw_ports:
            if isinstance(raw_port_count, bool) or not isinstance(raw_port_count, Integral):
                raise TypeError("codebook SRS antenna-port counts must be integers")
            port_count = int(raw_port_count)
            if port_count not in {1, 2, 4}:
                raise ValueError("DCI 0_3 codebook SRS antenna-port counts must be 1, 2, or 4")
            ports.append(port_count)
        port_count = max(ports)
        if port_count == 1:
            if (
                codebook_subset_by_cell[cell] is not None
                or max_rank_for_codebook_by_cell[cell] is not None
                or ul_full_power_transmission_by_cell[cell] is not None
            ):
                raise ValueError(
                    f"single-port codebook cell {cell!r} must not supply rank/subset/power mode"
                )
            widths[cell] = 0
            continue
        subset = codebook_subset_by_cell[cell]
        if not isinstance(subset, str) or subset not in allowed_subsets:
            raise ValueError(
                f"codebook subset for {cell!r} must be one of the supported RRC choices"
            )
        raw_rank = max_rank_for_codebook_by_cell[cell]
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, Integral):
            raise TypeError(f"maximum codebook rank for {cell!r} must be an integer")
        rank = int(raw_rank)
        if not 1 <= rank <= port_count:
            raise ValueError(f"maximum codebook rank for {cell!r} must be in [1, {port_count}]")
        full_power = ul_full_power_transmission_by_cell[cell]
        if full_power not in allowed_full_power:
            raise ValueError(f"ul-FullPowerTransmission value for {cell!r} is invalid")
        mode1 = full_power == "fullpowerMode1"
        if port_count == 2:
            if mode1:
                if rank == 2 and not transform_precoding:
                    if subset != "nonCoherent":
                        raise ValueError(
                            "2-port fullpowerMode1 rank-2 codebook requires nonCoherent subset"
                        )
                    widths[cell] = 2
                elif rank == 1 or transform_precoding:
                    if subset != "nonCoherent":
                        raise ValueError(
                            "2-port fullpowerMode1 rank-1 codebook requires nonCoherent subset"
                        )
                    widths[cell] = 2
                else:
                    raise ValueError("unsupported 2-port fullpowerMode1 codebook configuration")
            elif rank == 1 or transform_precoding:
                if subset not in {"fullyAndPartialAndNonCoherent", "nonCoherent"}:
                    raise ValueError("2-port rank-1 codebook does not support partial-only subset")
                widths[cell] = 3 if subset == "fullyAndPartialAndNonCoherent" else 1
            else:
                if subset not in {"fullyAndPartialAndNonCoherent", "nonCoherent"}:
                    raise ValueError("2-port rank-2 codebook does not support partial-only subset")
                widths[cell] = 4 if subset == "fullyAndPartialAndNonCoherent" else 2
        else:
            if mode1:
                if rank == 1 or transform_precoding:
                    if subset not in {"partialAndNonCoherent", "nonCoherent"}:
                        raise ValueError("4-port fullpowerMode1 rank-1 subset is unsupported")
                    widths[cell] = 4 if subset == "partialAndNonCoherent" else 3
                elif rank == 2:
                    if subset not in {"partialAndNonCoherent", "nonCoherent"}:
                        raise ValueError("4-port fullpowerMode1 rank-2 subset is unsupported")
                    widths[cell] = 5 if subset == "partialAndNonCoherent" else 4
                else:
                    if subset not in {"partialAndNonCoherent", "nonCoherent"}:
                        raise ValueError("4-port fullpowerMode1 rank-3/4 subset is unsupported")
                    widths[cell] = 6 if subset == "partialAndNonCoherent" else 4
            elif rank == 1 or transform_precoding:
                widths[cell] = {
                    "fullyAndPartialAndNonCoherent": 5,
                    "partialAndNonCoherent": 4,
                    "nonCoherent": 2,
                }[subset]
            else:
                widths[cell] = {
                    "fullyAndPartialAndNonCoherent": 6,
                    "partialAndNonCoherent": 5,
                    "nonCoherent": 4,
                }[subset]
    if mode == "type1a":
        return max((widths[cell] for cell in cell_ids), default=0)
    return sum(widths[cell] for cell in cell_ids)


def _nr_srs_dci_0_3_sri_width(
    *,
    mode: str,
    cell_ids: Sequence[str],
    tx_config_by_cell: Mapping[str, str],
    srs_resource_count_by_cell: Mapping[str, int],
    max_noncodebook_layers_by_cell: Mapping[str, int | None],
    ue_supports_max_mimo_layers_by_cell: Mapping[str, bool],
    ue_max_noncodebook_layers_by_cell: Mapping[str, int | None] | None,
    full_cell_roster: Sequence[str],
) -> int:
    """Derive DCI 0_3 SRI width from configured PUSCH/SRS resource state.

    If a serving-cell ``maxMIMO-Layers`` value is absent for non-codebook
    operation, the corresponding UE-supported maximum must be supplied.
    """
    if not isinstance(mode, str) or mode not in {"type1a", "type2"}:
        raise ValueError("sri_mode must be 'type1a' or 'type2'")
    if any(
        not isinstance(roster, Mapping)
        for roster in (
            tx_config_by_cell,
            srs_resource_count_by_cell,
            max_noncodebook_layers_by_cell,
            ue_supports_max_mimo_layers_by_cell,
        )
    ):
        raise TypeError("SRI configuration inputs must be mappings")
    if ue_max_noncodebook_layers_by_cell is not None and not isinstance(
        ue_max_noncodebook_layers_by_cell, Mapping
    ):
        raise TypeError("UE non-codebook layer capabilities must be a mapping")
    widths: dict[str, int] = {}
    for name, roster in (
        ("tx_config_by_cell", tx_config_by_cell),
        ("srs_resource_count_by_cell", srs_resource_count_by_cell),
        ("max_noncodebook_layers_by_cell", max_noncodebook_layers_by_cell),
        ("ue_supports_max_mimo_layers_by_cell", ue_supports_max_mimo_layers_by_cell),
    ):
        if set(roster) != set(full_cell_roster):
            raise ValueError(f"{name} must cover the full scheduled cell list")
    if ue_max_noncodebook_layers_by_cell is not None and set(
        ue_max_noncodebook_layers_by_cell
    ) != set(full_cell_roster):
        raise ValueError(
            "ue_max_noncodebook_layers_by_cell must cover the full scheduled cell list"
        )
    for cell in full_cell_roster:
        tx_config = tx_config_by_cell[cell]
        if not isinstance(tx_config, str) or tx_config not in {
            "codebook",
            "nonCodebook",
        }:
            raise ValueError(
                f"tx_config_by_cell[{cell!r}] must be 'codebook' or 'nonCodebook'"
            )
        supports_configured_max = ue_supports_max_mimo_layers_by_cell[cell]
        if type(supports_configured_max) is not bool:
            raise TypeError(
                "ue_supports_max_mimo_layers_by_cell values must be booleans"
            )
        raw_count = srs_resource_count_by_cell[cell]
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
            raise TypeError(f"SRS resource count for {cell!r} must be an integer")
        resource_count = int(raw_count)
        if not 1 <= resource_count <= 64:
            raise ValueError("SRS resource counts must be in [1, 64]")
        if tx_config == "codebook":
            widths[cell] = (resource_count - 1).bit_length()
            continue
        raw_layers = max_noncodebook_layers_by_cell[cell]
        if raw_layers is None or not supports_configured_max:
            if ue_max_noncodebook_layers_by_cell is None:
                raise ValueError(
                    "UE non-codebook layer capability is required when "
                    "maxMIMO-Layers is not configured or supported for "
                    f"{cell!r}"
                )
            raw_layers = ue_max_noncodebook_layers_by_cell[cell]
        if isinstance(raw_layers, bool) or not isinstance(raw_layers, Integral):
            raise TypeError(
                f"maximum non-codebook PUSCH layers for {cell!r} must be an integer"
            )
        max_layers = int(raw_layers)
        if not 1 <= max_layers <= 8:
            raise ValueError("maximum non-codebook PUSCH layers must be in [1, 8]")
        selectable_combinations = sum(
            comb(resource_count, layer_count)
            for layer_count in range(1, min(max_layers, resource_count) + 1)
        )
        widths[cell] = (selectable_combinations - 1).bit_length()
    if mode == "type1a":
        return max((widths[cell] for cell in cell_ids), default=0)
    return sum(widths[cell] for cell in cell_ids)


def nr_srs_dci_0_3_frequency_assignment_widths(
    scheduled_cell_ids: Sequence[str],
    configured_bwp_ids_by_cell: Mapping[str, Sequence[int]],
    active_bwp_id_by_cell: Mapping[str, int],
    bwp_frequency_config_by_cell: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> dict[str, int]:
    """Derive per-cell DCI 0_3 frequency-assignment widths from active UL BWPs.

    Regular allocation supports type 0, type 1, and dynamic switching. The
    interlace allocation type is supported for 15/30 kHz numerology when the
    decoded active-BWP configuration supplies its RB-set count.
    """
    cells = tuple(scheduled_cell_ids)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("scheduled_cell_ids must be nonempty and unique")
    if not isinstance(configured_bwp_ids_by_cell, Mapping) or not configured_bwp_ids_by_cell:
        raise ValueError("configured_bwp_ids_by_cell must be a nonempty mapping")
    full_cell_roster = set(configured_bwp_ids_by_cell)
    if not set(cells).issubset(full_cell_roster):
        raise ValueError("scheduled cells must be covered by the full BWP configuration roster")
    for name, value in (
        ("active_bwp_id_by_cell", active_bwp_id_by_cell),
        ("bwp_frequency_config_by_cell", bwp_frequency_config_by_cell),
    ):
        if not isinstance(value, Mapping) or set(value) != full_cell_roster:
            raise ValueError(f"{name} must exactly cover the full scheduled cell roster")

    rbg_sizes = {
        1: ((36, 2), (72, 4), (144, 8), (275, 16)),
        2: ((36, 4), (72, 8), (144, 16), (275, 16)),
        3: ((36, 8), (72, 16), (144, 32), (275, 32)),
    }
    widths: dict[str, int] = {}
    for cell in cells:
        raw_bwp_ids = configured_bwp_ids_by_cell[cell]
        if not isinstance(raw_bwp_ids, Sequence) or isinstance(raw_bwp_ids, (str, bytes)):
            raise TypeError(f"configured BWP IDs for {cell!r} must be an ordered sequence")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= value <= 4
            for value in raw_bwp_ids
        ):
            raise ValueError(f"configured BWP IDs for {cell!r} must be integers in [0, 4]")
        bwp_ids = tuple(int(value) for value in raw_bwp_ids)
        if len(set(bwp_ids)) != len(bwp_ids) or not bwp_ids:
            raise ValueError(f"configured BWP IDs for {cell!r} must be nonempty and unique")
        raw_configs = bwp_frequency_config_by_cell[cell]
        if (
            not isinstance(raw_configs, Mapping)
            or any(
                isinstance(key, bool) or not isinstance(key, Integral)
                for key in raw_configs
            )
            or set(raw_configs) != set(bwp_ids)
        ):
            raise ValueError(
                f"BWP frequency configurations for {cell!r} must cover every BWP"
            )
        active_id = active_bwp_id_by_cell[cell]
        if isinstance(active_id, bool) or not isinstance(active_id, Integral):
            raise TypeError(f"active BWP ID for {cell!r} must be an integer")
        if int(active_id) not in raw_configs:
            raise ValueError(f"active BWP {active_id!r} for {cell!r} is not configured")
        config = raw_configs[int(active_id)]
        if not isinstance(config, Mapping):
            raise TypeError(
                f"active BWP frequency configuration for {cell!r} must be a mapping"
            )

        size_raw = config.get("size_prb")
        start_raw = config.get("start_prb")
        for label, value in (("size_prb", size_raw), ("start_prb", start_raw)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{label} for {cell!r} must be an integer")
        size_prb = int(cast(Integral, size_raw))
        start_prb = int(cast(Integral, start_raw))
        if (
            not 1 <= size_prb <= 275
            or not 0 <= start_prb <= 274
            or start_prb + size_prb > 275
        ):
            raise ValueError(f"active BWP PRB extent for {cell!r} is outside supported bounds")

        interlace = config.get("use_interlace", False)
        if type(interlace) is not bool:
            raise TypeError(f"use_interlace for {cell!r} must be a boolean")
        if interlace:
            spacing = config.get("subcarrier_spacing_khz")
            if isinstance(spacing, bool) or not isinstance(spacing, Integral):
                raise TypeError(f"subcarrier_spacing_khz for {cell!r} must be an integer")
            base_width = {15: 6, 30: 5}.get(int(spacing))
            if base_width is None:
                raise ValueError("interlace DCI 0_3 supports 15 or 30 kHz spacing")
            rb_set_count = config.get("rb_set_count")
            if isinstance(rb_set_count, bool) or not isinstance(rb_set_count, Integral):
                raise TypeError(f"rb_set_count for {cell!r} must be an integer")
            rb_set_count = int(rb_set_count)
            if not 1 <= rb_set_count <= 275:
                raise ValueError(f"rb_set_count for {cell!r} must be in [1, 275]")
            widths[cell] = base_width + (rb_set_count - 1).bit_length()
            continue

        allocation = config.get("resource_allocation")
        if not isinstance(allocation, str) or allocation not in {
            "type0",
            "type1",
            "dynamicSwitch",
        }:
            raise ValueError(f"unsupported resource allocation mode for {cell!r}")
        hopping = config.get("frequency_hopping", False)
        if type(hopping) is not bool:
            raise TypeError(f"frequency_hopping for {cell!r} must be a boolean")
        if hopping:
            offset_count = config.get("frequency_hopping_offset_count")
            if isinstance(offset_count, bool) or not isinstance(offset_count, Integral):
                raise TypeError(
                    f"frequency_hopping_offset_count for {cell!r} must be an integer"
                )
            offset_count = int(offset_count)
            if offset_count not in {2, 4}:
                raise ValueError("frequency hopping offset list must contain 2 or 4 values")
            hopping_width = (offset_count - 1).bit_length()
        else:
            hopping_width = 0

        rbg_config_raw = config.get("rbg_size_config", 1)
        if isinstance(rbg_config_raw, str) and rbg_config_raw.startswith("config"):
            rbg_config_raw = rbg_config_raw.removeprefix("config")
        if isinstance(rbg_config_raw, bool) or not isinstance(
            rbg_config_raw, (Integral, str)
        ):
            raise TypeError(
                f"rbg_size_config for {cell!r} must be config1, config2, or config3"
            )
        try:
            rbg_config = int(rbg_config_raw)
        except ValueError as exc:
            raise ValueError("rbg_size_config must be config1, config2, or config3") from exc
        if rbg_config not in rbg_sizes:
            raise ValueError("rbg_size_config must be config1, config2, or config3")
        type0_p = next(
            limit_size
            for limit, limit_size in rbg_sizes[rbg_config]
            if size_prb <= limit
        )
        type0_count = (size_prb + start_prb % type0_p + type0_p - 1) // type0_p
        type0_width = type0_count

        granularity_raw = config.get("type1_granularity_prb", 1)
        if isinstance(granularity_raw, bool) or not isinstance(
            granularity_raw, Integral
        ):
            raise TypeError(f"type1_granularity_prb for {cell!r} must be an integer")
        granularity = int(granularity_raw)
        if granularity not in {1, 2, 4, 8}:
            raise ValueError("type1_granularity_prb must be one of 1, 2, 4, or 8")
        type1_count = (
            size_prb + start_prb % granularity + granularity - 1
        ) // granularity
        type1_width = (type1_count * (type1_count + 1) // 2 - 1).bit_length()
        if allocation == "type0":
            widths[cell] = type0_width
        elif allocation == "type1":
            widths[cell] = type1_width + hopping_width
        else:
            widths[cell] = 1 + max(type0_width, type1_width + hopping_width)
    return widths


def _nr_srs_dci_0_3_pdcch_monitoring_adaptation_width(
    *,
    enabled: bool,
    skipping_duration_count: int | None,
    search_space_group_ids: Sequence[int] | None,
) -> int:
    """Derive the Release 19 DCI 0_3 PDCCH monitoring adaptation width."""
    if type(enabled) is not bool:
        raise TypeError("pdcch_monitoring_adaptation_enabled must be a boolean")
    if skipping_duration_count is not None:
        if isinstance(skipping_duration_count, bool) or not isinstance(
            skipping_duration_count, Integral
        ):
            raise TypeError("pdcch_skipping_duration_count must be an integer or None")
        skipping_duration_count = int(skipping_duration_count)
        if not 1 <= skipping_duration_count <= 64:
            raise ValueError("pdcch_skipping_duration_count must be in [1, 64]")
    if search_space_group_ids is not None:
        if not isinstance(search_space_group_ids, Sequence) or isinstance(
            search_space_group_ids, (str, bytes)
        ):
            raise TypeError("search_space_group_ids must be an ordered sequence or None")
        if any(
            isinstance(group_id, bool)
            or not isinstance(group_id, Integral)
            or int(group_id) not in {0, 1, 2}
            for group_id in search_space_group_ids
        ):
            raise ValueError("search_space_group_ids must contain only IDs 0, 1, and 2")
        group_ids = tuple(int(group_id) for group_id in search_space_group_ids)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("search_space_group_ids must not contain duplicates")
        search_space_group_ids = group_ids
    if not enabled:
        return 0
    if skipping_duration_count is not None and search_space_group_ids is not None:
        return 2
    if skipping_duration_count is not None:
        return 1 if skipping_duration_count == 1 else 2
    if search_space_group_ids is not None:
        groups = set(search_space_group_ids)
        if groups == {0, 1}:
            return 1
        if groups == {0, 1, 2}:
            return 2
        raise ValueError(
            "enabled search-space group adaptation requires group IDs {0, 1} or {0, 1, 2}"
        )
    raise ValueError(
        "enabled PDCCH monitoring adaptation requires a skipping-duration list, "
        "a search-space group list, or both"
    )


def _nr_srs_dci_0_3_scell_dormancy_width(
    *,
    dormancy_configured: bool,
    dormancy_group_ids: Sequence[int] | None,
    carried_on_primary_cell_within_drx_active_time: bool,
    scell_has_multiple_dl_bwps: bool,
) -> int:
    """Derive the contextual SCell dormancy bitmap width for DCI 0_3."""
    for name, value in (
        ("dormancy_configured", dormancy_configured),
        (
            "carried_on_primary_cell_within_drx_active_time",
            carried_on_primary_cell_within_drx_active_time,
        ),
        ("scell_has_multiple_dl_bwps", scell_has_multiple_dl_bwps),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
    if dormancy_group_ids is None:
        if (
            dormancy_configured
            and carried_on_primary_cell_within_drx_active_time
            and scell_has_multiple_dl_bwps
        ):
            raise ValueError(
                "applicable SCell dormancy indication requires configured dormancy group IDs"
            )
        return 0
    if not isinstance(dormancy_group_ids, Sequence) or isinstance(
        dormancy_group_ids, (str, bytes)
    ):
        raise TypeError("dormancy_group_ids must be an ordered sequence or None")
    if any(
        isinstance(group_id, bool)
        or not isinstance(group_id, Integral)
        or int(group_id) < 0
        for group_id in dormancy_group_ids
    ):
        raise ValueError("dormancy_group_ids must contain nonnegative integers")
    groups = tuple(int(group_id) for group_id in dormancy_group_ids)
    if len(set(groups)) != len(groups):
        raise ValueError("dormancy_group_ids must be unique")
    if not dormancy_configured:
        return 0
    if (
        not carried_on_primary_cell_within_drx_active_time
        or not scell_has_multiple_dl_bwps
    ):
        return 0
    if not 1 <= len(groups) <= 5:
        raise ValueError("applicable dormancy_group_ids must contain 1..5 groups")
    return len(groups)


def nr_srs_dci_0_3_field_widths_from_configuration(
    *,
    cell_set_configs: Sequence[Mapping[str, object]],
    configured_ul_bwp_ids_by_cell: Mapping[str, Sequence[int]],
    active_ul_bwp_id_by_cell: Mapping[str, int],
    ul_bwp_frequency_allocation_by_cell: Mapping[
        str, Mapping[int, Mapping[str, object]]
    ],
    request_list_entry_count: int,
    srs_offset_list: Sequence[Sequence[int]] | None,
    report_trigger_size: int,
    tdra_field_index_list_entry_count: int,
    frequency_hopping_enabled_by_cell: Mapping[str, bool],
    transform_precoding_enabled_by_cell: Mapping[str, bool],
    beta_offsets_semi_static_by_cell: Mapping[str, bool],
    channel_access_config_entry_count_by_cell: Mapping[str, int],
    harq_process_number_size_bits_by_cell: Mapping[str, int],
    max_schedulable_pusch_count_by_cell: Mapping[str, int],
    redundancy_version_bits_by_cell: Mapping[str, int],
    sri_mode: str,
    tpmi_mode: str,
    antenna_ports_mode: str,
    tx_config_by_cell: Mapping[str, str],
    srs_resource_count_by_cell: Mapping[str, int],
    max_noncodebook_layers_by_cell: Mapping[str, int | None],
    ue_supports_max_mimo_layers_by_cell: Mapping[str, bool],
    dmrs_port_configuration_by_cell: Mapping[
        str, Sequence[Mapping[str, object]]
    ],
    codebook_srs_antenna_port_counts_by_cell: Mapping[str, Sequence[int]],
    codebook_subset_by_cell: Mapping[str, str | None],
    max_rank_for_codebook_by_cell: Mapping[str, int | None],
    ul_full_power_transmission_by_cell: Mapping[str, str | None],
    harq_ack_codebook_types: Sequence[str],
    dynamic_harq_ack_two_sub_codebooks_by_codebook: Sequence[bool],
    priority_indicator_configured: bool,
    minimum_scheduling_offset_configured: bool,
    selected_cell_set_id: int | None = None,
    scheduled_cells_indicator_value: int | None = None,
    initial_bwp_id: int = 0,
    supports_active_bwp_change: bool = True,
    pdcch_monitoring_adaptation_enabled: bool = False,
    pdcch_skipping_duration_count: int | None = None,
    search_space_group_ids: Sequence[int] | None = None,
    dormancy_configured: bool = False,
    dormancy_group_ids: Sequence[int] | None = None,
    carried_on_primary_cell_within_drx_active_time: bool = False,
    scell_has_multiple_dl_bwps: bool = False,
    ptrs_uplink_configured_by_cell: Mapping[str, bool] | None = None,
    max_pusch_rank_by_cell: Mapping[str, int] | None = None,
    p0_pusch_set_list_configured_by_cell: Mapping[str, bool] | None = None,
    sri_field_present_by_cell: Mapping[str, bool] | None = None,
    olpc_parameter_set_dci_0_1_bits_by_cell: Mapping[str, int] | None = None,
    ue_max_noncodebook_layers_by_cell: Mapping[str, int | None] | None = None,
    measurement_gap_cancellation_configured: bool = False,
) -> dict[str, int]:
    """Derive the DCI 0_3 fixed/cell/BWP/SRS selector widths from decoded state.

    This derives the format identifier, scheduled-cell-set, scheduled-cells,
    BWP-indicator, frequency-domain resource assignment, time-domain assignment,
    frequency-hopping, DMRS sequence initialization, UL-SCH, priority,
    minimum-scheduling-offset, SCell dormancy, PDCCH monitoring adaptation,
    beta-offset,
    channel-access, scheduled-PUSCH TPC, HARQ-process-number, MCS, NDI, RV,
    first and second downlink assignment index, SRS-request, SRS-offset-indicator,
    SRS-resource-indicator, TPMI, antenna-port, and CSI-request widths. Antenna-port
    width is derived from the configured DMRS mapping-type roster, DMRS type,
    enhanced-DMRS state, maximum length, and transform-precoding state. The TDRA
    width uses the configured
    ``tdra-FieldIndexListDCI-0-3`` entry count; the hopping flag is present
    when any cell in the complete scheduled-cell list has frequency hopping
    configured; and the DMRS sequence initialization field is present when
    transform precoding is disabled for any cell in that list. The UL-SCH
    indicator is one bit. Channel-access entry counts are zero when the field
    is not applicable to a scheduled cell and otherwise count that cell's
    decoded access-configuration list entries; the resulting field width is
    the maximum per-cell ``ceil(log2(count))``. Frequency assignment is
    computed from each scheduled cell's active UL BWP and its decoded
    allocation/interlace configuration. Codebook TPMI width is derived from
    SRS antenna-port count, subset, maximum rank, full-power-transmission mode,
    and transform-precoding state for the supported one-, two-, and four-port
    branches, then aggregated by type1a/type2. Every field in the returned
    ordered layout is derived here; callers cannot override a field width.
    Beta-offset width is zero when every scheduled cell uses semi-static beta
    offsets and two bits otherwise. TPC width is two bits per cell in the
    decoded scheduled-cell combination. MCS is five bits per selected cell.
    HARQ-process width is the sum of each selected cell's decoded 0–5-bit
    configuration. NDI width sums the maximum schedulable PUSCH count per
    selected cell; RV width sums that count times the cell's configured 0–2-bit
    RV width. Those per-cell maps cover the full list; multi-PUSCH count is one
    when no multi-PUSCH TDRA list is configured. The first and second DAI widths
    are the maximum widths over the one or two configured HARQ-ACK codebooks,
    with each codebook's mode and
    dynamic sub-codebook state supplied in matching ordered sequences.
    SCell dormancy width depends on its unique configured group roster and is
    zero outside the required primary-cell/DRX-Active-Time/multiple-DL-BWP
    occasion. PDCCH monitoring adaptation width is derived from its configured
    skipping-duration and search-space-group lists.
    Open-loop power-control indication width depends on per-cell P0 set-list
    presence, SRI field presence, and the configured one/two-bit OLPC width.
    Non-codebook SRI width uses configured ``maxMIMO-Layers`` only when the
    UE supports that parameter; otherwise ``ue_max_noncodebook_layers_by_cell``
    must supply the UE's supported maximum for each non-codebook cell.
    The function rejects omitted and extra field names so its result can be
    passed directly to :func:`nr_srs_dci_0_3_request_list_index_from_layout`.
    """
    if not isinstance(cell_set_configs, Sequence) or isinstance(
        cell_set_configs, (str, bytes)
    ):
        raise TypeError("cell_set_configs must be an ordered sequence")
    # Reuse the cell-set adapter for complete roster and row validation.
    if selected_cell_set_id is None:
        if len(cell_set_configs) != 1:
            raise ValueError("multiple cell sets require selected_cell_set_id")
        selected_config = cell_set_configs[0]
        if not isinstance(selected_config, Mapping):
            raise TypeError("each cell-set config must be a mapping")
        selected_cell_set_id = selected_config.get("set_id")  # type: ignore[assignment]
    selected_configs = [
        config
        for config in cell_set_configs
        if isinstance(config, Mapping) and config.get("set_id") == selected_cell_set_id
    ]
    if len(selected_configs) != 1:
        raise ValueError("selected_cell_set_id must identify exactly one configured cell set")
    full_cell_roster = selected_configs[0]["scheduled_cell_ids"]
    if not isinstance(full_cell_roster, Sequence) or isinstance(
        full_cell_roster, (str, bytes)
    ):
        raise TypeError("scheduled_cell_ids must be an ordered sequence")
    combos = selected_configs[0].get("scheduled_cell_combos")
    if combos is None:
        combo_count = 1
    elif isinstance(combos, Sequence) and not isinstance(combos, (str, bytes)):
        combo_count = len(combos)
    else:
        raise TypeError("scheduled_cell_combos must be an ordered sequence")
    selected_cell_roster = nr_srs_dci_0_3_scheduled_cell_ids(
        cell_set_configs,
        scheduled_cells_indicator=scheduled_cells_indicator_value,
        default_set_id=selected_cell_set_id,
    )
    antenna_ports_width = _nr_srs_dci_0_3_antenna_ports_width(
        mode=antenna_ports_mode,
        cell_ids=selected_cell_roster,
        transform_precoding_enabled_by_cell=transform_precoding_enabled_by_cell,
        dmrs_port_configuration_by_cell=dmrs_port_configuration_by_cell,
        full_cell_roster=full_cell_roster,
    )
    precoding_width = _nr_srs_dci_0_3_precoding_width(
        mode=tpmi_mode,
        cell_ids=selected_cell_roster,
        tx_config_by_cell=tx_config_by_cell,
        transform_precoding_enabled_by_cell=transform_precoding_enabled_by_cell,
        codebook_srs_antenna_port_counts_by_cell=(
            codebook_srs_antenna_port_counts_by_cell
        ),
        codebook_subset_by_cell=codebook_subset_by_cell,
        max_rank_for_codebook_by_cell=max_rank_for_codebook_by_cell,
        ul_full_power_transmission_by_cell=ul_full_power_transmission_by_cell,
        full_cell_roster=full_cell_roster,
    )
    sri_width = _nr_srs_dci_0_3_sri_width(
        mode=sri_mode,
        cell_ids=selected_cell_roster,
        tx_config_by_cell=tx_config_by_cell,
        srs_resource_count_by_cell=srs_resource_count_by_cell,
        max_noncodebook_layers_by_cell=max_noncodebook_layers_by_cell,
        ue_supports_max_mimo_layers_by_cell=ue_supports_max_mimo_layers_by_cell,
        ue_max_noncodebook_layers_by_cell=ue_max_noncodebook_layers_by_cell,
        full_cell_roster=full_cell_roster,
    )
    if not isinstance(configured_ul_bwp_ids_by_cell, Mapping):
        raise TypeError("configured_ul_bwp_ids_by_cell must be a mapping")
    if set(configured_ul_bwp_ids_by_cell) != set(full_cell_roster):
        raise ValueError("configured_ul_bwp_ids_by_cell must cover the full scheduled cell list")
    for cell in full_cell_roster:
        ids = configured_ul_bwp_ids_by_cell[cell]
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise TypeError(f"configured UL BWP IDs for {cell!r} must be an ordered sequence")
    bwp_field_present = any(
        len(configured_ul_bwp_ids_by_cell[cell]) > 1 for cell in full_cell_roster
    )
    bwp_selection = nr_srs_dci_0_3_ul_bwp_selection(
        full_cell_roster,
        configured_ul_bwp_ids_by_cell,
        active_ul_bwp_id_by_cell,
        bwp_indicator=0 if bwp_field_present else None,
        initial_bwp_id=initial_bwp_id,
        supports_active_bwp_change=supports_active_bwp_change,
    )
    frequency_assignment_widths = nr_srs_dci_0_3_frequency_assignment_widths(
        selected_cell_roster,
        configured_ul_bwp_ids_by_cell,
        active_ul_bwp_id_by_cell,
        ul_bwp_frequency_allocation_by_cell,
    )
    if isinstance(tdra_field_index_list_entry_count, bool) or not isinstance(
        tdra_field_index_list_entry_count, Integral
    ):
        raise TypeError("tdra_field_index_list_entry_count must be an integer")
    tdra_entry_count = int(tdra_field_index_list_entry_count)
    if not 1 <= tdra_entry_count <= 4096:
        raise ValueError("tdra_field_index_list_entry_count must be in [1, 4096]")
    if not isinstance(frequency_hopping_enabled_by_cell, Mapping):
        raise TypeError("frequency_hopping_enabled_by_cell must be a mapping")
    if set(frequency_hopping_enabled_by_cell) != set(full_cell_roster):
        raise ValueError(
            "frequency_hopping_enabled_by_cell must cover the full scheduled cell list"
        )
    if any(type(value) is not bool for value in frequency_hopping_enabled_by_cell.values()):
        raise TypeError("frequency-hopping configuration values must be booleans")
    if not isinstance(transform_precoding_enabled_by_cell, Mapping):
        raise TypeError("transform_precoding_enabled_by_cell must be a mapping")
    if set(transform_precoding_enabled_by_cell) != set(full_cell_roster):
        raise ValueError(
            "transform_precoding_enabled_by_cell must cover the full scheduled cell list"
        )
    if any(
        type(value) is not bool for value in transform_precoding_enabled_by_cell.values()
    ):
        raise TypeError("transform-precoding configuration values must be booleans")
    if ptrs_uplink_configured_by_cell is None and max_pusch_rank_by_cell is None:
        ptrs_dmrs_width_by_cell = dict.fromkeys(full_cell_roster, 0)
    elif not isinstance(ptrs_uplink_configured_by_cell, Mapping) or not isinstance(
        max_pusch_rank_by_cell, Mapping
    ):
        raise TypeError(
            "PTRS configuration and maximum PUSCH rank must both be supplied as mappings"
        )
    else:
        if set(ptrs_uplink_configured_by_cell) != set(full_cell_roster):
            raise ValueError("PTRS configuration must cover the full scheduled cell list")
        if set(max_pusch_rank_by_cell) != set(full_cell_roster):
            raise ValueError("maximum PUSCH rank must cover the full scheduled cell list")
        if any(
            type(value) is not bool
            for value in ptrs_uplink_configured_by_cell.values()
        ):
            raise TypeError("PTRS-UplinkConfig states must be booleans")
        max_pusch_rank: dict[str, int] = {}
        for cell, raw_rank in max_pusch_rank_by_cell.items():
            if isinstance(raw_rank, bool) or not isinstance(raw_rank, Integral):
                raise TypeError(f"maximum PUSCH rank for {cell!r} must be an integer")
            rank = int(raw_rank)
            if not 1 <= rank <= 8:
                raise ValueError("maximum PUSCH rank must be in [1, 8]")
            max_pusch_rank[cell] = rank
        ptrs_dmrs_width_by_cell = {
            cell: 2
            if ptrs_uplink_configured_by_cell[cell]
            and not transform_precoding_enabled_by_cell[cell]
            and max_pusch_rank[cell] > 1
            else 0
            for cell in full_cell_roster
        }
    olpc_inputs = (
        p0_pusch_set_list_configured_by_cell,
        sri_field_present_by_cell,
        olpc_parameter_set_dci_0_1_bits_by_cell,
    )
    if all(value is None for value in olpc_inputs):
        olpc_width_by_cell = dict.fromkeys(full_cell_roster, 0)
    elif any(not isinstance(value, Mapping) for value in olpc_inputs):
        raise TypeError("all open-loop power-control configuration maps must be supplied")
    else:
        p0_configured = cast(Mapping[str, bool], p0_pusch_set_list_configured_by_cell)
        sri_present = cast(Mapping[str, bool], sri_field_present_by_cell)
        olpc_bits_config = cast(
            Mapping[str, int], olpc_parameter_set_dci_0_1_bits_by_cell
        )
        for name, roster in (
            ("p0_pusch_set_list_configured_by_cell", p0_configured),
            ("sri_field_present_by_cell", sri_present),
            ("olpc_parameter_set_dci_0_1_bits_by_cell", olpc_bits_config),
        ):
            if set(roster) != set(full_cell_roster):
                raise ValueError(f"{name} must cover the full scheduled cell list")
        if any(type(value) is not bool for value in p0_configured.values()):
            raise TypeError("P0-PUSCH set-list configuration values must be booleans")
        if any(type(value) is not bool for value in sri_present.values()):
            raise TypeError("SRI field-presence values must be booleans")
        olpc_bits: dict[str, int] = {}
        for cell, raw_width in olpc_bits_config.items():
            if isinstance(raw_width, bool) or not isinstance(raw_width, Integral):
                raise TypeError(f"OLPC indication width for {cell!r} must be an integer")
            width = int(raw_width)
            if width not in {1, 2}:
                raise ValueError("configured OLPC indication widths must be one or two bits")
            olpc_bits[cell] = width
        olpc_width_by_cell = {
            cell: (
                0
                if not p0_configured[cell]
                else 1
                if sri_present[cell]
                else olpc_bits[cell]
            )
            for cell in full_cell_roster
        }
    if not isinstance(beta_offsets_semi_static_by_cell, Mapping):
        raise TypeError("beta_offsets_semi_static_by_cell must be a mapping")
    if set(beta_offsets_semi_static_by_cell) != set(full_cell_roster):
        raise ValueError(
            "beta_offsets_semi_static_by_cell must cover the full scheduled cell list"
        )
    if any(type(value) is not bool for value in beta_offsets_semi_static_by_cell.values()):
        raise TypeError("beta-offset configuration values must be booleans")
    if not isinstance(harq_ack_codebook_types, Sequence) or isinstance(
        harq_ack_codebook_types, (str, bytes)
    ):
        raise TypeError("harq_ack_codebook_types must be an ordered sequence")
    if not 1 <= len(harq_ack_codebook_types) <= 2:
        raise ValueError("one or two HARQ-ACK codebook types must be supplied")
    if any(
        not isinstance(value, str) or value not in {"semi_static", "dynamic"}
        for value in harq_ack_codebook_types
    ):
        raise ValueError("HARQ-ACK codebook types must be 'semi_static' or 'dynamic'")
    if not isinstance(dynamic_harq_ack_two_sub_codebooks_by_codebook, Sequence) or isinstance(
        dynamic_harq_ack_two_sub_codebooks_by_codebook, (str, bytes)
    ):
        raise TypeError(
            "dynamic_harq_ack_two_sub_codebooks_by_codebook must be an ordered sequence"
        )
    if len(dynamic_harq_ack_two_sub_codebooks_by_codebook) != len(
        harq_ack_codebook_types
    ):
        raise ValueError("HARQ-ACK codebook state sequences must have equal lengths")
    if any(
        type(value) is not bool
        for value in dynamic_harq_ack_two_sub_codebooks_by_codebook
    ):
        raise TypeError("two-sub-codebook flags must be booleans")
    if any(
        codebook_type == "semi_static" and two_sub_codebooks
        for codebook_type, two_sub_codebooks in zip(
            harq_ack_codebook_types,
            dynamic_harq_ack_two_sub_codebooks_by_codebook,
            strict=True,
        )
    ):
        raise ValueError("semi-static HARQ-ACK cannot configure dynamic sub-codebooks")
    if not isinstance(channel_access_config_entry_count_by_cell, Mapping):
        raise TypeError("channel_access_config_entry_count_by_cell must be a mapping")
    if set(channel_access_config_entry_count_by_cell) != set(full_cell_roster):
        raise ValueError(
            "channel_access_config_entry_count_by_cell must cover the full scheduled cell list"
        )
    channel_access_counts: list[int] = []
    for cell, raw_count in channel_access_config_entry_count_by_cell.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
            raise TypeError(f"channel-access entry count for {cell!r} must be an integer")
        count = int(raw_count)
        if not 0 <= count <= 64:
            raise ValueError("channel-access entry counts must be in [0, 64]")
        channel_access_counts.append(count)
    if not isinstance(harq_process_number_size_bits_by_cell, Mapping):
        raise TypeError("harq_process_number_size_bits_by_cell must be a mapping")
    if set(harq_process_number_size_bits_by_cell) != set(full_cell_roster):
        raise ValueError(
            "harq_process_number_size_bits_by_cell must cover the full scheduled cell list"
        )
    harq_process_widths: dict[str, int] = {}
    for cell, raw_width in harq_process_number_size_bits_by_cell.items():
        if isinstance(raw_width, bool) or not isinstance(raw_width, Integral):
            raise TypeError(f"HARQ-process width for {cell!r} must be an integer")
        width = int(raw_width)
        if not 0 <= width <= 5:
            raise ValueError("HARQ-process widths must be in [0, 5]")
        harq_process_widths[cell] = width
    if not isinstance(max_schedulable_pusch_count_by_cell, Mapping):
        raise TypeError("max_schedulable_pusch_count_by_cell must be a mapping")
    if set(max_schedulable_pusch_count_by_cell) != set(full_cell_roster):
        raise ValueError(
            "max_schedulable_pusch_count_by_cell must cover the full scheduled cell list"
        )
    schedulable_pusch_counts: dict[str, int] = {}
    for cell, raw_count in max_schedulable_pusch_count_by_cell.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
            raise TypeError(f"schedulable PUSCH count for {cell!r} must be an integer")
        count = int(raw_count)
        if not 1 <= count <= 64:
            raise ValueError("schedulable PUSCH counts must be in [1, 64]")
        schedulable_pusch_counts[cell] = count
    if not isinstance(redundancy_version_bits_by_cell, Mapping):
        raise TypeError("redundancy_version_bits_by_cell must be a mapping")
    if set(redundancy_version_bits_by_cell) != set(full_cell_roster):
        raise ValueError(
            "redundancy_version_bits_by_cell must cover the full scheduled cell list"
        )
    redundancy_widths: dict[str, int] = {}
    for cell, raw_width in redundancy_version_bits_by_cell.items():
        if isinstance(raw_width, bool) or not isinstance(raw_width, Integral):
            raise TypeError(f"redundancy-version width for {cell!r} must be an integer")
        width = int(raw_width)
        if not 0 <= width <= 2:
            raise ValueError("redundancy-version widths must be in [0, 2]")
        redundancy_widths[cell] = width
    if type(priority_indicator_configured) is not bool:
        raise TypeError("priority_indicator_configured must be a boolean")
    if type(minimum_scheduling_offset_configured) is not bool:
        raise TypeError("minimum_scheduling_offset_configured must be a boolean")
    if type(measurement_gap_cancellation_configured) is not bool:
        raise TypeError("measurement_gap_cancellation_configured must be a boolean")
    if isinstance(request_list_entry_count, bool) or not isinstance(
        request_list_entry_count, Integral
    ):
        raise TypeError("request_list_entry_count must be an integer")
    entry_count = int(request_list_entry_count)
    if not 0 <= entry_count <= 16:
        raise ValueError("request_list_entry_count must be in [0, 16]")
    if srs_offset_list is None:
        offset_count = 0
    else:
        if not isinstance(srs_offset_list, Sequence) or isinstance(
            srs_offset_list, (str, bytes)
        ):
            raise TypeError("srs_offset_list must be an ordered sequence or None")
        offset_count = len(srs_offset_list)
        if not 1 <= offset_count <= 8:
            raise ValueError("srs_offset_list must contain 1..8 configured entries")
        for row_index, row in enumerate(srs_offset_list):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError(f"srs_offset_list[{row_index}] must be an ordered sequence")
            if len(row) != len(full_cell_roster):
                raise ValueError(
                    f"srs_offset_list[{row_index}] must contain one index per scheduled cell"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) <= 7
                for value in row
            ):
                raise ValueError("SRS offset-list indexes must be integers in [0, 7]")
    if isinstance(report_trigger_size, bool) or not isinstance(
        report_trigger_size, Integral
    ):
        raise TypeError("report_trigger_size must be an integer")
    csi_width = int(report_trigger_size)
    if not 0 <= csi_width <= 6:
        raise ValueError("report_trigger_size must be in [0, 6]")
    cell_set_count = len(cell_set_configs)
    return {
        "identifier_for_dci_formats": 1,
        "scheduled_cell_set_indicator": (cell_set_count - 1).bit_length(),
        "scheduled_cells_indicator": (combo_count - 1).bit_length(),
        "bandwidth_part_indicator": bwp_selection.indicator_bit_width,
        "frequency_domain_resource_assignment": sum(
            frequency_assignment_widths[cell] for cell in selected_cell_roster
        ),
        "time_domain_resource_assignment": (tdra_entry_count - 1).bit_length(),
        "frequency_hopping_flag": int(any(frequency_hopping_enabled_by_cell.values())),
        "dmrs_sequence_initialization": int(
            not all(transform_precoding_enabled_by_cell.values())
        ),
        "ul_sch_indicator": 1,
        "priority_indicator": int(priority_indicator_configured),
        "minimum_applicable_scheduling_offset_indicator": int(
            minimum_scheduling_offset_configured
        ),
        "beta_offset_indicator": 0
        if all(beta_offsets_semi_static_by_cell.values())
        else 2,
        "tpc_command_for_scheduled_pusch": 2 * len(selected_cell_roster),
        "harq_process_number": sum(
            harq_process_widths[cell] for cell in selected_cell_roster
        ),
        "modulation_and_coding_scheme": 5 * len(selected_cell_roster),
        "new_data_indicator": sum(
            schedulable_pusch_counts[cell] for cell in selected_cell_roster
        ),
        "redundancy_version": sum(
            schedulable_pusch_counts[cell] * redundancy_widths[cell]
            for cell in selected_cell_roster
        ),
        "first_downlink_assignment_index": max(
            1 if codebook_type == "semi_static" else 2
            for codebook_type in harq_ack_codebook_types
        ),
        "second_downlink_assignment_index": max(
            2 if codebook_type == "dynamic" and two_sub_codebooks else 0
            for codebook_type, two_sub_codebooks in zip(
                harq_ack_codebook_types,
                dynamic_harq_ack_two_sub_codebooks_by_codebook,
                strict=True,
            )
        ),
        "channel_access_cpext_capc": max(
            ((count - 1).bit_length() if count else 0 for count in channel_access_counts),
            default=0,
        ),
        "srs_request": (entry_count - 1).bit_length() if entry_count else 0,
        "srs_offset_indicator": (offset_count - 1).bit_length() if offset_count else 0,
        "srs_resource_indicator": sri_width,
        "precoding_information_and_number_of_layers": precoding_width,
        "antenna_ports": antenna_ports_width,
        "csi_request": csi_width,
        "ptrs_dmrs_association": sum(
            ptrs_dmrs_width_by_cell[cell] for cell in selected_cell_roster
        ),
        "open_loop_power_control_parameter_set_indication": max(
            (olpc_width_by_cell[cell] for cell in selected_cell_roster),
            default=0,
        ),
        "scell_dormancy_indication": _nr_srs_dci_0_3_scell_dormancy_width(
            dormancy_configured=dormancy_configured,
            dormancy_group_ids=dormancy_group_ids,
            carried_on_primary_cell_within_drx_active_time=(
                carried_on_primary_cell_within_drx_active_time
            ),
            scell_has_multiple_dl_bwps=scell_has_multiple_dl_bwps,
        ),
        "pdcch_monitoring_adaptation_indication": (
            _nr_srs_dci_0_3_pdcch_monitoring_adaptation_width(
                enabled=pdcch_monitoring_adaptation_enabled,
                skipping_duration_count=pdcch_skipping_duration_count,
                search_space_group_ids=search_space_group_ids,
            )
        ),
        "measurement_gap_cancellation": int(measurement_gap_cancellation_configured),
    }


def nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules(
    request_list_index: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    triggering_dci_slot_by_cell: Mapping[str, int],
    *,
    cell_set_configs: Sequence[Mapping[str, object]],
    scheduled_cell_set_indicator: int | None = None,
    scheduled_cells_indicator: int | None = None,
    default_set_id: int | None = None,
    offset_combo_index: int | None = None,
    offset_combos: Sequence[Sequence[int]] | None = None,
    offset_combo_cell_ids: Sequence[str] = (),
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[tuple[str, tuple[NRSRSResourceSetAperiodicSchedule, ...]], ...]:
    """Resolve DCI 0_3 cell indicators and compose per-cell SRS schedules.

    The request-list row and resource-set mapping are scoped to the selected
    RRC cell set, as required by ``SRS-RequestCombo``. ``serving_cell_ids`` may
    be the wider serving-cell roster; its order is used to verify that the
    selected set's RRC cell list is in ascending serving-cell order.
    """
    scheduled_cells = nr_srs_dci_0_3_scheduled_cell_ids(
        cell_set_configs,
        scheduled_cell_set_indicator=scheduled_cell_set_indicator,
        scheduled_cells_indicator=scheduled_cells_indicator,
        default_set_id=default_set_id,
    )
    ordered_configs = sorted(
        cell_set_configs, key=lambda item: int(cast(Integral, item["set_id"]))
    )
    if scheduled_cell_set_indicator is not None:
        set_index = int(scheduled_cell_set_indicator)
    elif default_set_id is not None:
        set_index = next(
            index
            for index, item in enumerate(ordered_configs)
            if int(cast(Integral, item["set_id"])) == int(default_set_id)
        )
    else:
        set_index = 0
    set_cells = tuple(
        cast(Sequence[str], ordered_configs[set_index]["scheduled_cell_ids"])
    )
    if not isinstance(serving_cell_ids, Sequence) or isinstance(
        serving_cell_ids, (str, bytes)
    ):
        raise TypeError("serving_cell_ids must be an ordered sequence")
    global_cells = tuple(serving_cell_ids)
    if not global_cells or any(not isinstance(cell, str) or not cell for cell in global_cells):
        raise ValueError("serving_cell_ids must contain nonempty cell IDs")
    if len(set(global_cells)) != len(global_cells):
        raise ValueError("serving_cell_ids must be unique")
    if tuple(cell for cell in global_cells if cell in set(set_cells)) != set_cells:
        raise ValueError(
            "selected set scheduled_cell_ids must follow serving_cell_ids order"
        )
    if not isinstance(supplementary_uplink_cells, Sequence) or isinstance(
        supplementary_uplink_cells, (str, bytes)
    ):
        raise TypeError("supplementary_uplink_cells must be an ordered sequence")
    sul_cells = tuple(supplementary_uplink_cells)
    if any(not isinstance(cell, str) or not cell for cell in sul_cells):
        raise ValueError("supplementary_uplink_cells must contain nonempty cell IDs")
    if len(set(sul_cells)) != len(sul_cells) or not set(sul_cells).issubset(global_cells):
        raise ValueError("supplementary_uplink_cells must be a unique serving-cell subset")
    if not isinstance(resource_sets_by_cell, Mapping):
        raise TypeError("resource_sets_by_cell must be a mapping")
    if not set(set_cells).issubset(resource_sets_by_cell):
        raise ValueError("resource_sets_by_cell must include every cell in selected set")
    set_resource_sets = {cell: resource_sets_by_cell[cell] for cell in set_cells}
    return nr_srs_multicell_aperiodic_resource_set_schedules(
        request_list_index,
        request_list,
        set_cells,
        set_resource_sets,
        triggering_dci_slot_by_cell,
        scheduled_cell_ids=scheduled_cells,
        offset_combo_index=offset_combo_index,
        offset_combos=offset_combos,
        offset_combo_cell_ids=offset_combo_cell_ids,
        supplementary_uplink_cells=tuple(cell for cell in sul_cells if cell in set(set_cells)),
    )


def nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits(
    payload_bits: Sequence[int | bool],
    *,
    field_offset: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    triggering_dci_slot_by_cell: Mapping[str, int],
    cell_set_configs: Sequence[Mapping[str, object]],
    scheduled_cell_set_indicator: int | None = None,
    scheduled_cells_indicator: int | None = None,
    default_set_id: int | None = None,
    offset_combo_index: int | None = None,
    offset_combos: Sequence[Sequence[int]] | None = None,
    offset_combo_cell_ids: Sequence[str] = (),
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[tuple[str, tuple[NRSRSResourceSetAperiodicSchedule, ...]], ...] | None:
    """Decode the DCI 0_3 request-list selector and compose its SRS schedules.

    ``field_offset`` is the caller-resolved bit offset in an already-decoded
    DCI payload. The complete DCI field layout and ASN.1 RRC configuration are
    outside this adapter. If no request-list field is configured, return
    ``None`` because there is no selector or aperiodic request to resolve.
    """
    if not isinstance(request_list, Sequence) or isinstance(request_list, (str, bytes)):
        raise TypeError("request_list must be a sequence of per-cell request entries")
    request_list_index = nr_srs_dci_0_3_request_list_index_from_bits(
        payload_bits,
        field_offset=field_offset,
        request_list_entry_count=len(request_list),
    )
    if request_list_index is None:
        return None
    return nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules(
        request_list_index,
        request_list,
        serving_cell_ids,
        resource_sets_by_cell,
        triggering_dci_slot_by_cell,
        cell_set_configs=cell_set_configs,
        scheduled_cell_set_indicator=scheduled_cell_set_indicator,
        scheduled_cells_indicator=scheduled_cells_indicator,
        default_set_id=default_set_id,
        offset_combo_index=offset_combo_index,
        offset_combos=offset_combos,
        offset_combo_cell_ids=offset_combo_cell_ids,
        supplementary_uplink_cells=supplementary_uplink_cells,
    )


def nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_by_bwp(
    request_list_index: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell_and_bwp: Mapping[
        str, Mapping[int, Sequence[Mapping[str, object]]]
    ],
    triggering_dci_slot_by_cell: Mapping[str, int],
    *,
    cell_set_configs: Sequence[Mapping[str, object]],
    configured_ul_bwp_ids_by_cell: Mapping[str, Sequence[int]],
    active_ul_bwp_id_by_cell: Mapping[str, int],
    ul_bwp_frequency_bounds_by_cell_and_bwp: Mapping[
        str, Mapping[int, NRSRSBWPFrequencyBounds]
    ],
    bwp_indicator: int | None,
    supports_active_bwp_change: bool = True,
    scheduled_cell_set_indicator: int | None = None,
    scheduled_cells_indicator: int | None = None,
    default_set_id: int | None = None,
    offset_combo_index: int | None = None,
    offset_combos: Sequence[Sequence[int]] | None = None,
    offset_combo_cell_ids: Sequence[str] = (),
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[NRSRSBWPCellResourceSetSchedules, ...]:
    """Resolve DCI 0_3 cell/BWP choices and schedule SRS on those BWPs.

    Per-cell BWP resource-set tables cover every configured UL BWP in the
    selected cell set. Offset-combo cell membership is derived across those
    BWPs from aperiodic resource sets with multiple available slot offsets;
    a supplied roster, when nonempty, must match that derived RRC order.
    Caller-supplied frequency bounds are required for every configured UL BWP;
    each returned allocation must fit entirely inside its selected BWP in the
    same subcarrier coordinate system. Returned rows retain the selected BWP
    and its bounds even when no SRS set was triggered.
    """
    ordered_configs = sorted(
        cell_set_configs, key=lambda item: int(cast(Integral, item["set_id"]))
    )
    if scheduled_cell_set_indicator is not None:
        set_index = int(scheduled_cell_set_indicator)
    elif default_set_id is not None:
        set_index = next(
            index
            for index, item in enumerate(ordered_configs)
            if int(cast(Integral, item["set_id"])) == int(default_set_id)
        )
    else:
        set_index = 0
    set_cells = tuple(
        cast(Sequence[str], ordered_configs[set_index]["scheduled_cell_ids"])
    )

    bwp_selection = nr_srs_dci_0_3_ul_bwp_selection(
        set_cells,
        configured_ul_bwp_ids_by_cell,
        active_ul_bwp_id_by_cell,
        bwp_indicator=bwp_indicator,
        supports_active_bwp_change=supports_active_bwp_change,
    )
    selected_bwp_by_cell = dict(bwp_selection.selected_bwp_id_by_cell)
    if not isinstance(ul_bwp_frequency_bounds_by_cell_and_bwp, Mapping):
        raise TypeError("ul_bwp_frequency_bounds_by_cell_and_bwp must be a mapping")
    if set(ul_bwp_frequency_bounds_by_cell_and_bwp) != set(set_cells):
        raise ValueError(
            "UL BWP frequency-bound keys must exactly match the selected cell set"
        )
    bounds_by_cell: dict[str, dict[int, NRSRSBWPFrequencyBounds]] = {}
    for cell in set_cells:
        raw_bounds = ul_bwp_frequency_bounds_by_cell_and_bwp[cell]
        if not isinstance(raw_bounds, Mapping):
            raise TypeError(f"UL BWP frequency bounds for {cell!r} must be a mapping")
        configured_bwps = tuple(
            sorted(int(bwp_id) for bwp_id in configured_ul_bwp_ids_by_cell[cell])
        )
        if any(
            isinstance(bwp_id, bool) or not isinstance(bwp_id, Integral)
            for bwp_id in raw_bounds
        ):
            raise TypeError(f"UL BWP frequency-bound IDs for {cell!r} must be integers")
        if set(raw_bounds) != set(configured_bwps):
            raise ValueError(
                f"UL BWP frequency-bound IDs for {cell!r} must match configured UL BWPs"
            )
        normalized_bounds: dict[int, NRSRSBWPFrequencyBounds] = {}
        for bwp_id, bounds in raw_bounds.items():
            if not isinstance(bounds, NRSRSBWPFrequencyBounds):
                raise TypeError(
                    f"UL BWP frequency bound for {cell!r}/BWP {bwp_id} has invalid type"
                )
            normalized_bounds[int(bwp_id)] = bounds
        bounds_by_cell[cell] = normalized_bounds
    if not isinstance(resource_sets_by_cell_and_bwp, Mapping):
        raise TypeError("resource_sets_by_cell_and_bwp must be a mapping")
    if set(resource_sets_by_cell_and_bwp) != set(set_cells):
        raise ValueError(
            "resource_sets_by_cell_and_bwp keys must exactly match the selected cell set"
        )

    selected_resource_sets: dict[str, Sequence[Mapping[str, object]]] = {}
    derived_offset_cells: list[str] = []
    for cell in set_cells:
        configured_bwps = tuple(
            sorted(int(bwp_id) for bwp_id in configured_ul_bwp_ids_by_cell[cell])
        )
        by_bwp = resource_sets_by_cell_and_bwp[cell]
        if not isinstance(by_bwp, Mapping):
            raise TypeError(f"per-BWP resource-set table for {cell!r} must be a mapping")
        if any(
            isinstance(bwp_id, bool) or not isinstance(bwp_id, Integral)
            for bwp_id in by_bwp
        ):
            raise TypeError(f"per-BWP resource-set keys for {cell!r} must be integer IDs")
        if set(by_bwp) != set(configured_bwps):
            raise ValueError(
                f"per-BWP resource-set keys for {cell!r} must match configured UL BWPs"
            )
        for bwp_id in configured_bwps:
            records = by_bwp[bwp_id]
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                raise TypeError(f"resource sets for {cell!r}/BWP {bwp_id} must be a sequence")
            for record in records:
                if not isinstance(record, Mapping):
                    raise TypeError("each per-BWP resource-set record must be a mapping")
                if record.get("resource_type") != "aperiodic":
                    continue
                allocation = record.get("allocation")
                if not isinstance(allocation, Mapping):
                    raise ValueError(
                        f"aperiodic resource set for {cell!r}/BWP {bwp_id} "
                        "requires an allocation mapping"
                    )
                available = allocation.get("available_slot_offsets")
                if available is None:
                    continue
                if not isinstance(available, Sequence) or isinstance(
                    available, (str, bytes)
                ):
                    raise TypeError("available_slot_offsets must be an ordered sequence")
                if not available:
                    raise ValueError("available_slot_offsets must not be empty")
                if len(available) > 1 and cell not in derived_offset_cells:
                    derived_offset_cells.append(cell)
        selected_resource_sets[cell] = by_bwp[selected_bwp_by_cell[cell]]

    derived_offset_roster = tuple(derived_offset_cells)
    if offset_combo_cell_ids and tuple(offset_combo_cell_ids) != derived_offset_roster:
        raise ValueError("offset_combo_cell_ids must match BWP-derived RRC offset eligibility")
    schedules = nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules(
        request_list_index,
        request_list,
        serving_cell_ids,
        selected_resource_sets,
        triggering_dci_slot_by_cell,
        cell_set_configs=cell_set_configs,
        scheduled_cell_set_indicator=scheduled_cell_set_indicator,
        scheduled_cells_indicator=scheduled_cells_indicator,
        default_set_id=default_set_id,
        offset_combo_index=offset_combo_index,
        offset_combos=offset_combos,
        offset_combo_cell_ids=derived_offset_roster,
        supplementary_uplink_cells=supplementary_uplink_cells,
    )
    return tuple(
        NRSRSBWPCellResourceSetSchedules(
            cell_id=cell,
            ul_bwp_id=selected_bwp_by_cell[cell],
            frequency_bounds=bounds_by_cell[cell][selected_bwp_by_cell[cell]],
            resource_set_schedules=cell_schedules,
        )
        for cell, cell_schedules in schedules
    )


def nr_srs_multicell_request_resource_set_ids(
    request_list_index: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    scheduled_cell_ids: Sequence[str] | None = None,
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Resolve a decoded DCI 0_3/1_3 request-list entry for each scheduled cell.

    ``request_list`` is the already-decoded ``srs-RequestListDCI-0-3`` or
    ``srs-RequestListDCI-1-3`` roster. Each entry contains one per-cell SRS
    request codepoint, ordered by ascending serving-cell index as specified by
    TS 38.212. ``serving_cell_ids`` must use that same order. Codepoints use
    the per-cell two-bit request, with the leading SUL selector included when
    that cell is listed in ``supplementary_uplink_cells``. The returned tuple
    preserves serving-cell order and contains only matching aperiodic set IDs.
    When ``scheduled_cell_ids`` is supplied, request entries for configured but
    unscheduled cells are returned as empty and are not interpreted.

    This resolves the decoded request-list indirection only; it does not parse
    DCI bits, decode ASN.1 RRC, resolve scheduled-cell combinations, or handle
    typeA antenna-switching semantics.
    """

    if isinstance(request_list_index, bool) or not isinstance(request_list_index, Integral):
        raise TypeError("request_list_index must be an integer")
    if not isinstance(request_list, Sequence) or isinstance(request_list, (str, bytes)):
        raise TypeError("request_list must be a sequence of per-cell request entries")
    if not request_list:
        raise ValueError("request_list must contain at least one configured entry")
    request_list_index = int(request_list_index)
    if not 0 <= request_list_index < len(request_list):
        raise ValueError("request_list_index does not select a configured entry")
    if not isinstance(serving_cell_ids, Sequence) or isinstance(
        serving_cell_ids, (str, bytes)
    ):
        raise TypeError("serving_cell_ids must be an ordered sequence")
    cell_ids = tuple(serving_cell_ids)
    if not cell_ids or any(not isinstance(cell, str) or not cell for cell in cell_ids):
        raise ValueError("serving_cell_ids must contain nonempty cell identifiers")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("serving_cell_ids must be unique")
    if scheduled_cell_ids is None:
        scheduled_cells = cell_ids
    else:
        if not isinstance(scheduled_cell_ids, Sequence) or isinstance(
            scheduled_cell_ids, (str, bytes)
        ):
            raise TypeError("scheduled_cell_ids must be an ordered sequence")
        scheduled_cells = tuple(scheduled_cell_ids)
        if not scheduled_cells or any(
            not isinstance(cell, str) or not cell for cell in scheduled_cells
        ):
            raise ValueError("scheduled_cell_ids must contain nonempty cell IDs")
        if len(set(scheduled_cells)) != len(scheduled_cells) or not set(
            scheduled_cells
        ).issubset(cell_ids):
            raise ValueError("scheduled_cell_ids must be a unique subset of serving_cell_ids")
        if tuple(cell for cell in cell_ids if cell in set(scheduled_cells)) != scheduled_cells:
            raise ValueError("scheduled_cell_ids must follow serving-cell order")
    if not isinstance(resource_sets_by_cell, Mapping):
        raise TypeError("resource_sets_by_cell must be a mapping")
    if set(resource_sets_by_cell) != set(cell_ids):
        raise ValueError("resource_sets_by_cell keys must exactly match serving_cell_ids")
    if not isinstance(supplementary_uplink_cells, Sequence) or isinstance(
        supplementary_uplink_cells, (str, bytes)
    ):
        raise TypeError("supplementary_uplink_cells must be a sequence")
    sul_cells = tuple(supplementary_uplink_cells)
    if any(not isinstance(cell, str) for cell in sul_cells):
        raise TypeError("supplementary_uplink_cells entries must be strings")
    if len(set(sul_cells)) != len(sul_cells) or not set(sul_cells).issubset(cell_ids):
        raise ValueError("supplementary_uplink_cells must be unique scheduled cell IDs")

    selected_entry = request_list[request_list_index]
    if not isinstance(selected_entry, Sequence) or isinstance(selected_entry, (str, bytes)):
        raise TypeError("selected request-list entry must be a per-cell sequence")
    if len(selected_entry) != len(cell_ids):
        raise ValueError("selected request-list entry must contain one codepoint per cell")
    scheduled_set = set(scheduled_cells)

    return tuple(
        (
            cell_id,
            (
                nr_srs_triggered_resource_set_ids(
                    codepoint,
                    resource_sets_by_cell[cell_id],
                    dci_format="0_1",
                    supplementary_uplink=cell_id in sul_cells,
                )
                if cell_id in scheduled_set
                else ()
            ),
        )
        for cell_id, codepoint in zip(cell_ids, selected_entry, strict=True)
    )


def nr_srs_multicell_aperiodic_resource_set_schedules(
    request_list_index: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    triggering_dci_slot_by_cell: Mapping[str, int],
    *,
    scheduled_cell_ids: Sequence[str],
    offset_combo_index: int | None = None,
    offset_combos: Sequence[Sequence[int]] | None = None,
    offset_combo_cell_ids: Sequence[str] = (),
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[tuple[str, tuple[NRSRSResourceSetAperiodicSchedule, ...]], ...]:
    """Compose decoded multi-cell SRS requests, offset combos, and allocations.

    Request/offset table rows, the full configured serving-cell roster, and the
    DCI-selected scheduled-cell subset are already decoded from DCI/RRC.
    ``offset_combo_cell_ids`` is the ordered subset represented in each RRC
    ``SRS-OffsetCombo`` row; it can include configured cells that are not
    co-scheduled in this DCI. Each resource set's allocation carries its own
    available-slot-offset list and base ``slot_offset``. The DCI slot is
    supplied for each scheduled cell because cells may use different
    numerologies; this adapter does not convert time between numerologies.
    """
    selected = nr_srs_multicell_request_resource_set_ids(
        request_list_index,
        request_list,
        serving_cell_ids,
        resource_sets_by_cell,
        scheduled_cell_ids=scheduled_cell_ids,
        supplementary_uplink_cells=supplementary_uplink_cells,
    )
    cells = tuple(serving_cell_ids)
    if not isinstance(scheduled_cell_ids, Sequence) or isinstance(
        scheduled_cell_ids, (str, bytes)
    ):
        raise TypeError("scheduled_cell_ids must be an ordered sequence")
    scheduled_cells = tuple(scheduled_cell_ids)
    if not scheduled_cells or any(
        not isinstance(cell, str) or not cell for cell in scheduled_cells
    ):
        raise ValueError("scheduled_cell_ids must contain nonempty cell IDs")
    if len(set(scheduled_cells)) != len(scheduled_cells) or not set(scheduled_cells).issubset(
        cells
    ):
        raise ValueError("scheduled_cell_ids must be a unique subset of serving_cell_ids")
    if tuple(cell for cell in cells if cell in set(scheduled_cells)) != scheduled_cells:
        raise ValueError("scheduled_cell_ids must follow serving-cell order")
    if not isinstance(triggering_dci_slot_by_cell, Mapping):
        raise TypeError("triggering_dci_slot_by_cell must be a mapping")
    if set(triggering_dci_slot_by_cell) != set(scheduled_cells):
        raise ValueError("triggering_dci_slot_by_cell keys must exactly match scheduled_cell_ids")
    for cell_id, slot in triggering_dci_slot_by_cell.items():
        if isinstance(slot, bool) or not isinstance(slot, Integral) or slot < 0:
            raise ValueError(f"triggering DCI slot for {cell_id!r} must be nonnegative")

    if not isinstance(offset_combo_cell_ids, Sequence) or isinstance(
        offset_combo_cell_ids, (str, bytes)
    ):
        raise TypeError("offset_combo_cell_ids must be an ordered sequence")
    combo_cells = tuple(offset_combo_cell_ids)
    if any(not isinstance(cell, str) or not cell for cell in combo_cells):
        raise ValueError("offset_combo_cell_ids must contain nonempty cell IDs")
    if len(set(combo_cells)) != len(combo_cells) or not set(combo_cells).issubset(cells):
        raise ValueError("offset_combo_cell_ids must be a unique subset of serving_cell_ids")
    if tuple(cell for cell in cells if cell in set(combo_cells)) != combo_cells:
        raise ValueError("offset_combo_cell_ids must follow serving-cell order")

    selected_offset_by_cell: dict[str, int] = {}
    if offset_combos is None:
        if offset_combo_index is not None or combo_cells:
            raise ValueError("offset combo index/cell roster requires offset_combos")
    else:
        if not isinstance(offset_combos, Sequence) or isinstance(offset_combos, (str, bytes)):
            raise TypeError("offset_combos must be an ordered sequence of decoded rows")
        if not 1 <= len(offset_combos) <= 8 or not combo_cells:
            raise ValueError("offset_combos needs 1..8 rows and a nonempty cell roster")
        if isinstance(offset_combo_index, bool) or not isinstance(offset_combo_index, Integral):
            raise TypeError("offset_combo_index must select a configured offset-combo row")
        offset_combo_index = int(offset_combo_index)
        if not 0 <= offset_combo_index < len(offset_combos):
            raise ValueError("offset_combo_index does not select a configured row")
        row = offset_combos[offset_combo_index]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError("selected offset-combo row must be a sequence")
        if len(row) != len(combo_cells):
            raise ValueError("selected offset-combo row must match offset_combo_cell_ids")
        for cell_id, value in zip(combo_cells, row, strict=True):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 3:
                raise ValueError("SRS offset-combo values must be integer indices in [0, 3]")
            selected_offset_by_cell[cell_id] = int(value)

    schedules_by_cell = dict(selected)
    result: list[tuple[str, tuple[NRSRSResourceSetAperiodicSchedule, ...]]] = []
    for cell_id in scheduled_cells:
        resource_sets = resource_sets_by_cell[cell_id]
        selected_ids = schedules_by_cell[cell_id]
        sets_by_id = {
            int(record["resource_set_id"]): record
            for record in resource_sets
            if isinstance(record, Mapping)
        }
        cell_schedules: list[NRSRSResourceSetAperiodicSchedule] = []
        for resource_set_id in selected_ids:
            resource_set = sets_by_id[resource_set_id]
            allocation = resource_set.get("allocation")
            if not isinstance(allocation, Mapping):
                raise ValueError(
                    f"selected resource set {resource_set_id} for {cell_id!r} "
                    "requires an allocation mapping"
                )
            resolved_allocation = dict(allocation)
            available_offsets = resolved_allocation.get("available_slot_offsets")
            if available_offsets is not None:
                if not isinstance(available_offsets, Sequence) or isinstance(
                    available_offsets, (str, bytes)
                ):
                    raise TypeError("available_slot_offsets must be an ordered sequence")
                if not available_offsets:
                    raise ValueError("available_slot_offsets must not be empty")
                if len(available_offsets) == 1:
                    offset_index = 0
                else:
                    try:
                        offset_index = selected_offset_by_cell[cell_id]
                    except KeyError as exc:
                        raise ValueError(
                            f"cell {cell_id!r} needs an SRS offset-combo entry for "
                            f"resource set {resource_set_id}"
                        ) from exc
                if offset_index >= len(available_offsets):
                    raise ValueError(
                        f"SRS offset-combo index {offset_index} is outside the available "
                        f"offset list for {cell_id!r} resource set {resource_set_id}"
                    )
                resolved_allocation["selected_available_offset_index"] = offset_index
            elif cell_id in selected_offset_by_cell:
                # This cell may be represented in a combo because another set
                # or BWP has multiple offsets; no list means the base offset applies.
                resolved_allocation.pop("selected_available_offset_index", None)

            schedule = nr_srs_aperiodic_resource_schedule(
                int(triggering_dci_slot_by_cell[cell_id]), **cast(Any, resolved_allocation)
            )
            cell_schedules.append(
                NRSRSResourceSetAperiodicSchedule(
                    resource_set_id=resource_set_id,
                    target_slot_index=schedule.target_slot_index,
                    allocations=schedule.allocations,
                )
            )
        result.append((cell_id, tuple(cell_schedules)))
    return tuple(result)


def _normalized_multiport_beta(
    beta_srs: float, port_count: int, four_port_srs_3tx: bool
) -> float:
    if (
        isinstance(beta_srs, bool)
        or not isinstance(beta_srs, Real)
        or not np.isfinite(float(beta_srs))
        or beta_srs <= 0
    ):
        raise ValueError("beta_srs must be a finite positive real number")
    if not isinstance(four_port_srs_3tx, bool):
        raise TypeError("four_port_srs_3tx must be a boolean")
    if four_port_srs_3tx and port_count != 4:
        raise ValueError("four_port_srs_3tx requires exactly four SRS ports")
    effective_port_count = port_count - 1 if four_port_srs_3tx else port_count
    return float(beta_srs) / float(np.sqrt(effective_port_count))


def nr_srs_fixed_frequency_allocation(
    c_srs: int,
    *,
    transmission_comb: int,
    b_srs: int = 0,
    b_hop: int = 0,
    frequency_domain_position: int = 0,
    frequency_domain_shift_rb: int = 0,
    comb_offset: int = 0,
    reference_subcarrier: int = 0,
    n_srs: int | None = None,
) -> NRSRSFixedFrequencyAllocation:
    """Resolve the type-1 SRS frequency allocation for one transmission.

    For active hopping, ``n_srs`` is the TS 38.211 transmission counter.
    Deriving that counter from periodic or semi-persistent scheduling is
    outside this resolver. ``reference_subcarrier`` selects the caller's
    CRB/BWP grid origin. SRS-PosResource symbol offsets remain a separate
    mapping step via :func:`nr_srs_positioning_symbol_offset`.
    """

    configuration = nr_srs_type1_bandwidth_configuration(
        c_srs, transmission_comb=transmission_comb, b_srs=b_srs
    )
    for name, value in (
        ("b_hop", b_hop),
        ("frequency_domain_position", frequency_domain_position),
        ("frequency_domain_shift_rb", frequency_domain_shift_rb),
        ("comb_offset", comb_offset),
        ("reference_subcarrier", reference_subcarrier),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    b_hop = int(b_hop)
    if b_hop > 3:
        raise ValueError("b_hop must be in [0, 3]")
    comb = configuration.transmission_comb
    if int(comb_offset) >= comb:
        raise ValueError("comb_offset must be smaller than transmission_comb")

    position_indices = nr_srs_frequency_position_indices(
        c_srs,
        b_srs=configuration.b_srs,
        b_hop=b_hop,
        frequency_domain_position=int(frequency_domain_position),
        n_srs=n_srs,
    )
    frequency_tree_offset = 0
    for level, position in enumerate(position_indices):
        bandwidth_rb = _SRS_BANDWIDTH_TABLE[configuration.c_srs][level][0]
        frequency_tree_offset += 12 * bandwidth_rb * position

    start_subcarrier = (
        int(reference_subcarrier) + 12 * int(frequency_domain_shift_rb) + frequency_tree_offset
    )
    return NRSRSFixedFrequencyAllocation(
        bandwidth=configuration,
        position_indices=position_indices,
        frequency_domain_position=int(frequency_domain_position),
        reference_subcarrier=int(reference_subcarrier),
        frequency_domain_shift_rb=int(frequency_domain_shift_rb),
        comb_offset=int(comb_offset),
        start_subcarrier=start_subcarrier,
        b_hop=b_hop,
        n_srs=n_srs,
    )


def nr_srs_resource_frequency_allocations(
    c_srs: int,
    *,
    transmission_comb: int,
    b_srs: int,
    b_hop: int,
    srs_symbol_count: int,
    resource_type: str,
    frequency_domain_position: int = 0,
    frequency_domain_shift_rb: int = 0,
    comb_offset: int = 0,
    reference_subcarrier: int = 0,
    repetition_factor: int | None = None,
    ports8_tdm: bool = False,
    absolute_slot_index: int | None = None,
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    periodicity_and_offset: Mapping[str, object] | None = None,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> tuple[NRSRSFixedFrequencyAllocation, ...] | None:
    """Resolve each resource-symbol allocation from slot timing or RRC CHOICE.

    ``None`` means the periodic or semi-persistent resource is not active in
    the supplied slot. Periodicity may be supplied as resolved slot values or
    as the RRC ``periodicityAndOffset`` CHOICE and optional Release 16
    extension. For periodic positioning resources, the optional Hyper-SFN
    configuration is checked in the transmission counter. This does not parse
    complete ASN.1 RRC messages or resolve resource-set triggering.
    """

    if periodicity_and_offset is not None or periodicity_and_offset_ext is not None:
        if periodicity_slots is not None or slot_offset is not None:
            raise ValueError("supply resolved periodicity or the RRC CHOICE, not both")
        if resource_type == "aperiodic":
            raise ValueError("aperiodic SRS does not use periodicityAndOffset")
        periodicity_slots, slot_offset = nr_srs_periodicity_and_offset_slots(
            periodicity_and_offset,
            periodicity_and_offset_ext=periodicity_and_offset_ext,
        )

    if isinstance(srs_symbol_count, bool) or not isinstance(srs_symbol_count, Integral):
        raise TypeError("srs_symbol_count must be an integer")
    srs_symbol_count = int(srs_symbol_count)
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS:
        raise ValueError("srs_symbol_count is unsupported")
    allocations: list[NRSRSFixedFrequencyAllocation] = []
    for resource_symbol_index in range(srs_symbol_count):
        counter = nr_srs_transmission_counter(
            resource_type=resource_type,
            srs_symbol_count=srs_symbol_count,
            resource_symbol_index=resource_symbol_index,
            repetition_factor=repetition_factor,
            ports8_tdm=ports8_tdm,
            absolute_slot_index=absolute_slot_index,
            periodicity_slots=periodicity_slots,
            slot_offset=slot_offset,
            positioning_resource=positioning_resource,
            numerology=numerology,
            hyper_frame_number=hyper_frame_number,
            hyper_sfn_index=hyper_sfn_index,
        )
        if counter is None:
            return None
        allocations.append(
            nr_srs_fixed_frequency_allocation(
                c_srs,
                transmission_comb=transmission_comb,
                b_srs=b_srs,
                b_hop=b_hop,
                frequency_domain_position=frequency_domain_position,
                frequency_domain_shift_rb=frequency_domain_shift_rb,
                comb_offset=comb_offset,
                reference_subcarrier=reference_subcarrier,
                n_srs=counter,
            )
        )
    return tuple(allocations)


def nr_srs_periodic_resource_schedule(
    absolute_slot_index: int,
    *,
    c_srs: int,
    transmission_comb: int,
    b_srs: int,
    b_hop: int,
    srs_symbol_count: int,
    resource_type: str,
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    periodicity_and_offset: Mapping[str, object] | None = None,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
    frequency_domain_position: int = 0,
    frequency_domain_shift_rb: int = 0,
    comb_offset: int = 0,
    reference_subcarrier: int = 0,
    repetition_factor: int | None = None,
    ports8_tdm: bool = False,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> NRSRSAperiodicResourceSchedule | None:
    """Resolve one periodic/semi-persistent resource at an absolute slot.

    Returns ``None`` for a non-occasion slot or a Hyper-SFN-gated positioning
    occasion. The caller still resolves resource-set selection, active-BWP
    feasibility, and collisions with other transmissions.
    """

    if isinstance(absolute_slot_index, bool) or not isinstance(
        absolute_slot_index, Integral
    ):
        raise TypeError("absolute_slot_index must be an integer")
    if absolute_slot_index < 0:
        raise ValueError("absolute_slot_index must be nonnegative")
    if resource_type not in {"periodic", "semi-persistent"}:
        raise ValueError("periodic schedule requires periodic or semi-persistent resource_type")
    allocations = nr_srs_resource_frequency_allocations(
        c_srs,
        transmission_comb=transmission_comb,
        b_srs=b_srs,
        b_hop=b_hop,
        srs_symbol_count=srs_symbol_count,
        resource_type=resource_type,
        frequency_domain_position=frequency_domain_position,
        frequency_domain_shift_rb=frequency_domain_shift_rb,
        comb_offset=comb_offset,
        reference_subcarrier=reference_subcarrier,
        repetition_factor=repetition_factor,
        ports8_tdm=ports8_tdm,
        absolute_slot_index=int(absolute_slot_index),
        periodicity_slots=periodicity_slots,
        slot_offset=slot_offset,
        periodicity_and_offset=periodicity_and_offset,
        periodicity_and_offset_ext=periodicity_and_offset_ext,
        positioning_resource=positioning_resource,
        numerology=numerology,
        hyper_frame_number=hyper_frame_number,
        hyper_sfn_index=hyper_sfn_index,
    )
    if allocations is None:
        return None
    return NRSRSAperiodicResourceSchedule(int(absolute_slot_index), allocations)


def nr_srs_periodic_resource_set_schedule(
    absolute_slot_index: int,
    resource_set_id: int,
    resources: Sequence[Mapping[str, object]],
    *,
    resource_type: str = "periodic",
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    periodicity_and_offset: Mapping[str, object] | None = None,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> NRSRSPeriodicResourceSetSchedule | None:
    """Resolve all resources in one periodic set against shared occasion timing.

    Each decoded resource row has a unique ``resource_id`` and an ``allocation``
    mapping containing ``c_srs``, ``transmission_comb``, ``b_srs``,
    ``b_hop``, and ``srs_symbol_count``, plus the per-resource frequency and
    port options accepted by :func:`nr_srs_periodic_resource_schedule`. An
    optional ``timing`` mapping can override the common occasion inputs for an
    individual resource, since periodicity and offset belong to the resource
    configuration. The returned set contains only resources active in this
    slot; ``None`` means none are active.
    """
    if isinstance(absolute_slot_index, bool) or not isinstance(
        absolute_slot_index, Integral
    ):
        raise TypeError("absolute_slot_index must be an integer")
    if absolute_slot_index < 0:
        raise ValueError("absolute_slot_index must be nonnegative")
    if not isinstance(resources, Sequence) or isinstance(resources, (str, bytes)):
        raise TypeError("resources must be an ordered sequence of decoded SRS resources")
    if not resources:
        raise ValueError("resources must contain at least one decoded SRS resource")
    preflight_resource_ids: set[int] = set()
    for index, resource in enumerate(resources):
        if not isinstance(resource, Mapping):
            raise TypeError(f"resources[{index}] must be a mapping")
        resource_id = resource.get("resource_id")
        if isinstance(resource_id, bool) or not isinstance(resource_id, Integral):
            raise TypeError(f"resources[{index}].resource_id must be an integer")
        resource_id = int(resource_id)
        if resource_id < 0 or resource_id in preflight_resource_ids:
            raise ValueError("resource IDs must be unique nonnegative integers")
        preflight_resource_ids.add(resource_id)
    if isinstance(resource_set_id, bool) or not isinstance(resource_set_id, Integral):
        raise TypeError("resource_set_id must be an integer")
    if resource_set_id < 0:
        raise ValueError("resource_set_id must be nonnegative")
    if not isinstance(resource_type, str) or resource_type not in {
        "periodic",
        "semi-persistent",
    }:
        raise ValueError("resource_type must be periodic or semi-persistent")
    if not isinstance(positioning_resource, bool):
        raise TypeError("positioning_resource must be a boolean")

    required = {"c_srs", "transmission_comb", "b_srs", "b_hop", "srs_symbol_count"}
    optional = {
        "frequency_domain_position",
        "frequency_domain_shift_rb",
        "comb_offset",
        "reference_subcarrier",
        "repetition_factor",
        "ports8_tdm",
    }
    timing = {
        "periodicity_slots": periodicity_slots,
        "slot_offset": slot_offset,
        "periodicity_and_offset": periodicity_and_offset,
        "periodicity_and_offset_ext": periodicity_and_offset_ext,
        "positioning_resource": positioning_resource,
        "numerology": numerology,
        "hyper_frame_number": hyper_frame_number,
        "hyper_sfn_index": hyper_sfn_index,
    }
    timing_keys = frozenset(timing)
    seen_resource_ids: set[int] = set()
    schedules: list[tuple[int, NRSRSAperiodicResourceSchedule | None]] = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, Mapping):
            raise TypeError(f"resources[{index}] must be a mapping")
        resource_id = resource.get("resource_id")
        if isinstance(resource_id, bool) or not isinstance(resource_id, Integral):
            raise TypeError(f"resources[{index}].resource_id must be an integer")
        resource_id = int(resource_id)
        if resource_id < 0 or resource_id in seen_resource_ids:
            raise ValueError("resource IDs must be unique nonnegative integers")
        seen_resource_ids.add(resource_id)
        unsupported_resource_fields = set(resource) - {"resource_id", "allocation", "timing"}
        if unsupported_resource_fields:
            raise ValueError(
                f"resources[{index}] has unsupported fields: "
                f"{sorted(map(str, unsupported_resource_fields))}"
            )
        allocation = resource.get("allocation")
        if not isinstance(allocation, Mapping):
            raise ValueError(f"resources[{index}].allocation must be a mapping")
        fields = set(allocation)
        if any(not isinstance(field, str) for field in fields):
            raise ValueError(f"resources[{index}].allocation keys must be strings")
        missing = required - fields
        unsupported = fields - required - optional
        if missing or unsupported:
            raise ValueError(
                f"resources[{index}].allocation fields are invalid; "
                f"missing={sorted(missing)}, unsupported={sorted(unsupported)}"
            )
        resource_timing = resource.get("timing", {})
        if not isinstance(resource_timing, Mapping):
            raise ValueError(f"resources[{index}].timing must be a mapping")
        if any(not isinstance(field, str) for field in resource_timing):
            raise ValueError(f"resources[{index}].timing keys must be strings")
        unsupported_timing = set(resource_timing) - timing_keys
        if unsupported_timing:
            raise ValueError(
                f"resources[{index}].timing has unsupported fields: "
                f"{sorted(unsupported_timing)}"
            )
        schedule = nr_srs_periodic_resource_schedule(
            absolute_slot_index,
            resource_type=resource_type,
            **(timing | dict(resource_timing)),
            **cast(Any, allocation),
        )
        schedules.append((resource_id, schedule))

    active = [(resource_id, schedule) for resource_id, schedule in schedules if schedule is not None]
    if not active:
        return None
    return NRSRSPeriodicResourceSetSchedule(
        resource_set_id=int(resource_set_id),
        target_slot_index=int(absolute_slot_index),
        resources=tuple(
            (resource_id, cast(NRSRSAperiodicResourceSchedule, schedule))
            for resource_id, schedule in active
        ),
    )


def nr_srs_periodic_resource_set_schedule_by_bwp(
    absolute_slot_index: int,
    resource_set_id: int,
    resources: Sequence[Mapping[str, object]],
    *,
    cell_id: str,
    ul_bwp_id: int,
    frequency_bounds: NRSRSBWPFrequencyBounds,
    resource_type: str = "periodic",
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    periodicity_and_offset: Mapping[str, object] | None = None,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> NRSRSBWPPeriodicResourceSetSchedule | None:
    """Resolve a periodic SRS set and enforce its selected UL BWP extent.

    The BWP is caller-selected from active higher-layer state. This function
    validates every active per-symbol frequency allocation against that
    BWP's absolute subcarrier bounds. An off-occasion set returns ``None``.
    """
    if not isinstance(frequency_bounds, NRSRSBWPFrequencyBounds):
        raise TypeError("frequency_bounds must be NRSRSBWPFrequencyBounds")
    schedule = nr_srs_periodic_resource_set_schedule(
        absolute_slot_index,
        resource_set_id,
        resources,
        resource_type=resource_type,
        periodicity_slots=periodicity_slots,
        slot_offset=slot_offset,
        periodicity_and_offset=periodicity_and_offset,
        periodicity_and_offset_ext=periodicity_and_offset_ext,
        positioning_resource=positioning_resource,
        numerology=numerology,
        hyper_frame_number=hyper_frame_number,
        hyper_sfn_index=hyper_sfn_index,
    )
    if schedule is None:
        return None
    return NRSRSBWPPeriodicResourceSetSchedule(
        cell_id=cell_id,
        ul_bwp_id=ul_bwp_id,
        frequency_bounds=frequency_bounds,
        resource_set_schedule=schedule,
    )


def nr_srs_multicell_periodic_resource_set_schedules_by_bwp(
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    absolute_slot_index_by_cell: Mapping[str, int],
    active_ul_bwp_id_by_cell: Mapping[str, int],
    frequency_bounds_by_cell: Mapping[str, NRSRSBWPFrequencyBounds],
) -> tuple[NRSRSBWPCellPeriodicResourceSetSchedules, ...]:
    """Resolve periodic SRS sets for a frozen multi-cell roster.

    Each cell input contains the resource sets configured on its already
    selected active UL BWP. Set records have ``resource_set_id``,
    ``resource_type``, and ``resources`` fields, with optional common
    ``timing`` defaults. Per-resource timing can override those defaults.
    The output preserves serving-cell order and active-BWP identity, including
    cells with no SRS occasion in the supplied slot.
    """
    if not isinstance(serving_cell_ids, Sequence) or isinstance(
        serving_cell_ids, (str, bytes)
    ):
        raise TypeError("serving_cell_ids must be an ordered sequence")
    cells = tuple(serving_cell_ids)
    if not cells or any(not isinstance(cell, str) or not cell.strip() for cell in cells):
        raise ValueError("serving_cell_ids must contain nonempty cell IDs")
    cells = tuple(cell.strip() for cell in cells)
    if len(set(cells)) != len(cells):
        raise ValueError("serving_cell_ids must be unique")
    input_maps = {
        "resource_sets_by_cell": resource_sets_by_cell,
        "absolute_slot_index_by_cell": absolute_slot_index_by_cell,
        "active_ul_bwp_id_by_cell": active_ul_bwp_id_by_cell,
        "frequency_bounds_by_cell": frequency_bounds_by_cell,
    }
    for name, values in input_maps.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping")
        if set(values) != set(cells):
            raise ValueError(f"{name} keys must exactly match serving_cell_ids")

    timing_fields = frozenset(
        {
            "periodicity_slots",
            "slot_offset",
            "periodicity_and_offset",
            "periodicity_and_offset_ext",
            "positioning_resource",
            "numerology",
            "hyper_frame_number",
            "hyper_sfn_index",
        }
    )
    output: list[NRSRSBWPCellPeriodicResourceSetSchedules] = []
    for cell in cells:
        raw_slot = absolute_slot_index_by_cell[cell]
        if isinstance(raw_slot, bool) or not isinstance(raw_slot, Integral):
            raise TypeError(f"absolute slot for {cell!r} must be an integer")
        if raw_slot < 0:
            raise ValueError(f"absolute slot for {cell!r} must be nonnegative")
        bwp_id = active_ul_bwp_id_by_cell[cell]
        if isinstance(bwp_id, bool) or not isinstance(bwp_id, Integral):
            raise TypeError(f"active UL BWP ID for {cell!r} must be an integer")
        if not 0 <= bwp_id <= 4:
            raise ValueError(f"active UL BWP ID for {cell!r} must be in [0, 4]")
        bounds = frequency_bounds_by_cell[cell]
        if not isinstance(bounds, NRSRSBWPFrequencyBounds):
            raise TypeError(f"frequency bounds for {cell!r} have an invalid type")
        raw_sets = resource_sets_by_cell[cell]
        if not isinstance(raw_sets, Sequence) or isinstance(raw_sets, (str, bytes)):
            raise TypeError(f"resource sets for {cell!r} must be an ordered sequence")
        seen_set_ids: set[int] = set()
        schedules: list[NRSRSBWPPeriodicResourceSetSchedule] = []
        for index, resource_set in enumerate(raw_sets):
            if not isinstance(resource_set, Mapping):
                raise TypeError(f"resource_sets_by_cell[{cell!r}][{index}] must be a mapping")
            if any(not isinstance(key, str) for key in resource_set):
                raise ValueError("SRS resource-set keys must be strings")
            allowed_set_fields = {"resource_set_id", "resource_type", "resources", "timing"}
            unsupported = set(resource_set) - allowed_set_fields
            missing = {"resource_set_id", "resource_type", "resources"} - set(resource_set)
            if unsupported or missing:
                raise ValueError(
                    f"SRS resource-set fields are invalid; missing={sorted(missing)}, "
                    f"unsupported={sorted(unsupported)}"
                )
            resource_set_id = resource_set["resource_set_id"]
            if isinstance(resource_set_id, bool) or not isinstance(resource_set_id, Integral):
                raise TypeError("resource_set_id must be an integer")
            if resource_set_id < 0 or int(resource_set_id) in seen_set_ids:
                raise ValueError("resource_set_ids must be unique nonnegative integers per cell")
            seen_set_ids.add(int(resource_set_id))
            timing = resource_set.get("timing", {})
            if not isinstance(timing, Mapping):
                raise TypeError("resource-set timing must be a mapping")
            if any(not isinstance(key, str) for key in timing):
                raise ValueError("resource-set timing keys must be strings")
            unsupported_timing = set(timing) - timing_fields
            if unsupported_timing:
                raise ValueError(
                    f"resource-set timing has unsupported fields: {sorted(unsupported_timing)}"
                )
            scheduled = nr_srs_periodic_resource_set_schedule_by_bwp(
                int(raw_slot),
                int(resource_set_id),
                cast(Sequence[Mapping[str, object]], resource_set["resources"]),
                cell_id=cell,
                ul_bwp_id=int(bwp_id),
                frequency_bounds=bounds,
                resource_type=cast(str, resource_set["resource_type"]),
                **cast(Any, timing),
            )
            if scheduled is not None:
                schedules.append(scheduled)
        output.append(
            NRSRSBWPCellPeriodicResourceSetSchedules(
                cell_id=cell,
                ul_bwp_id=int(bwp_id),
                frequency_bounds=bounds,
                resource_set_schedules=tuple(schedules),
            )
        )
    return tuple(output)


def nr_srs_aperiodic_resource_schedule(
    triggering_dci_slot: int,
    *,
    c_srs: int,
    transmission_comb: int,
    b_srs: int,
    b_hop: int,
    srs_symbol_count: int,
    slot_offset: int | None = None,
    available_slot_offsets: Sequence[int] | None = None,
    selected_available_offset_index: int | None = None,
    frequency_domain_position: int = 0,
    frequency_domain_shift_rb: int = 0,
    comb_offset: int = 0,
    reference_subcarrier: int = 0,
    repetition_factor: int | None = None,
    ports8_tdm: bool = False,
) -> NRSRSAperiodicResourceSchedule:
    """Resolve one caller-selected aperiodic SRS set to target slot and hops.

    The caller resolves DCI-to-resource-set selection and passes the selected
    Release 17 available-slot-offset index. This adapter then derives every
    resource-symbol frequency allocation. It does not parse DCI or ASN.1 RRC.
    """

    target_slot = nr_srs_aperiodic_target_slot_index(
        triggering_dci_slot,
        slot_offset=slot_offset,
        available_slot_offsets=available_slot_offsets,
        selected_available_offset_index=selected_available_offset_index,
    )
    allocations = nr_srs_resource_frequency_allocations(
        c_srs,
        transmission_comb=transmission_comb,
        b_srs=b_srs,
        b_hop=b_hop,
        srs_symbol_count=srs_symbol_count,
        resource_type="aperiodic",
        frequency_domain_position=frequency_domain_position,
        frequency_domain_shift_rb=frequency_domain_shift_rb,
        comb_offset=comb_offset,
        reference_subcarrier=reference_subcarrier,
        repetition_factor=repetition_factor,
        ports8_tdm=ports8_tdm,
    )
    if allocations is None:
        raise RuntimeError("aperiodic SRS allocation unexpectedly resolved inactive")
    return NRSRSAperiodicResourceSchedule(target_slot, allocations)


def nr_srs_aperiodic_resource_set_schedules(
    triggering_dci_slot: int,
    srs_request_codepoint: int,
    resource_sets: Sequence[Mapping[str, object]],
    *,
    dci_format: str,
    srs_tpc_pdcch_group: str = "typeB",
    srs_request_mode: int | None = None,
    supplementary_uplink: bool = False,
) -> tuple[NRSRSResourceSetAperiodicSchedule, ...]:
    """Resolve a supported DCI request across every matching configured set.

    Each selected aperiodic set has an ``allocation`` mapping containing
    ``c_srs``, ``transmission_comb``, ``b_srs``, ``b_hop``, and
    ``srs_symbol_count``, plus the optional fields accepted by
    :func:`nr_srs_aperiodic_resource_schedule`. The trigger resolver validates
    all set identities and trigger assignments; frequency allocation is
    required only for sets selected by this DCI request.
    """

    selected_ids = nr_srs_triggered_resource_set_ids(
        srs_request_codepoint,
        resource_sets,
        dci_format=dci_format,
        srs_tpc_pdcch_group=srs_tpc_pdcch_group,
        srs_request_mode=srs_request_mode,
        supplementary_uplink=supplementary_uplink,
    )
    records_by_id: dict[int, Mapping[str, object]] = {}
    for record in resource_sets:
        if isinstance(record, Mapping):
            records_by_id[cast(int, record["resource_set_id"])] = record
    required = {
        "c_srs",
        "transmission_comb",
        "b_srs",
        "b_hop",
        "srs_symbol_count",
    }
    optional = {
        "slot_offset",
        "available_slot_offsets",
        "selected_available_offset_index",
        "frequency_domain_position",
        "frequency_domain_shift_rb",
        "comb_offset",
        "reference_subcarrier",
        "repetition_factor",
        "ports8_tdm",
    }
    schedules: list[NRSRSResourceSetAperiodicSchedule] = []
    for resource_set_id in selected_ids:
        resource_set = records_by_id[resource_set_id]
        allocation = resource_set.get("allocation")
        if not isinstance(allocation, Mapping):
            raise ValueError(
                f"selected resource set {resource_set_id} requires an allocation mapping"
            )
        allocation_fields = set(allocation)
        missing = required - allocation_fields
        unsupported = allocation_fields - required - optional
        if missing or unsupported:
            raise ValueError(
                f"selected resource set {resource_set_id} allocation fields are invalid; "
                f"missing={sorted(missing)}, unsupported={sorted(unsupported)}"
            )
        schedule = nr_srs_aperiodic_resource_schedule(
            triggering_dci_slot,
            **cast(Any, allocation),
        )
        schedules.append(
            NRSRSResourceSetAperiodicSchedule(
                resource_set_id=resource_set_id,
                target_slot_index=schedule.target_slot_index,
                allocations=schedule.allocations,
            )
        )
    return tuple(schedules)


def _nr_gold_bits(c_init: int, count: int) -> np.ndarray:
    """Return ``count`` TS 38.211 Gold bits after the 1600-bit offset."""

    sequence_length = 1600 + count
    x1 = np.zeros(sequence_length + 31, dtype=np.uint8)
    x2 = np.zeros(sequence_length + 31, dtype=np.uint8)
    x1[0] = 1
    for bit_index in range(31):
        x2[bit_index] = (c_init >> bit_index) & 1
    for bit_index in range(sequence_length):
        x1[bit_index + 31] = x1[bit_index + 3] ^ x1[bit_index]
        x2[bit_index + 31] = (
            x2[bit_index + 3]
            ^ x2[bit_index + 2]
            ^ x2[bit_index + 1]
            ^ x2[bit_index]
        )
    return x1[1600 : 1600 + count] ^ x2[1600 : 1600 + count]


def _srs_gold_bit_offset(
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    slot_symbol_count: int,
) -> int:
    return 8 * (
        slot_symbol_count * slot_index + start_symbol_index + resource_symbol_index
    )


def _validate_srs_hopping_position(
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    slot_symbol_count: int,
) -> tuple[int, int, int, int, int]:
    values = {
        "sequence_id": sequence_id,
        "slot_index": slot_index,
        "start_symbol_index": start_symbol_index,
        "resource_symbol_index": resource_symbol_index,
        "slot_symbol_count": slot_symbol_count,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    sequence_id = int(sequence_id)
    slot_index = int(slot_index)
    start_symbol_index = int(start_symbol_index)
    resource_symbol_index = int(resource_symbol_index)
    slot_symbol_count = int(slot_symbol_count)
    if not 0 <= sequence_id <= 65535:
        raise ValueError("sequence_id must be in [0, 65535]")
    if slot_index < 0:
        raise ValueError("slot_index must be nonnegative")
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if not 0 <= start_symbol_index < slot_symbol_count:
        raise ValueError("start_symbol_index must be inside its slot")
    if not 0 <= resource_symbol_index < slot_symbol_count - start_symbol_index:
        raise ValueError("resource_symbol_index must be inside its slot")
    return (
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    )


def nr_srs_group_hopping_index(
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    *,
    slot_symbol_count: int = 14,
) -> int:
    """Return the TS 38.211 SRS group index for a resource symbol.

    The pseudo-random sequence restarts at each radio frame. ``slot_index`` is
    the zero-based slot within that frame; ``start_symbol_index`` is the
    absolute SRS start symbol ``l0`` within the slot; and
    ``resource_symbol_index`` is the resource-local symbol index ``l'``.
    Set ``slot_symbol_count=12`` for extended-CP slots.
    """

    (
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    ) = _validate_srs_hopping_position(
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    )
    first_bit = _srs_gold_bit_offset(
        slot_index, start_symbol_index, resource_symbol_index, slot_symbol_count
    )
    hopping_bits = _nr_gold_bits(sequence_id, first_bit + 8)
    group_hopping_value = sum(
        int(hopping_bits[first_bit + bit]) << bit for bit in range(8)
    )
    return (group_hopping_value + sequence_id) % 30


def nr_srs_sequence_hopping_index(
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    sequence_length: int,
    *,
    slot_symbol_count: int = 14,
) -> int:
    """Return the SRS base-sequence number under TS 38.211 sequence hopping.

    Sequence hopping is enabled only for ``M_ZC >= 6 N_sc^RB``; with the NR
    resource-block width ``N_sc^RB=12``, shorter sequences use base sequence 0.
    """

    (
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    ) = _validate_srs_hopping_position(
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    )
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, Integral):
        raise TypeError("sequence_length must be an integer")
    sequence_length = int(sequence_length)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if sequence_length < 72:
        return 0
    bit_index = slot_symbol_count * slot_index + start_symbol_index + resource_symbol_index
    return int(_nr_gold_bits(sequence_id, bit_index + 1)[bit_index])


def _largest_prime_below(value: int) -> int:
    for candidate in range(value - 1, 1, -1):
        if all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1)):
            return candidate
    raise ValueError("sequence length must exceed 3")


def nr_low_papr_sequence_type1(
    sequence_length: int,
    *,
    group_index: int,
    base_sequence_index: int = 0,
    cyclic_shift_index: int = 0,
    cyclic_shift_count: int = 1,
    cyclic_shift_hopping_offset: int = 0,
    hopping_finer_granularity: bool = False,
) -> np.ndarray:
    """Generate the type-1 low-PAPR sequence for supported TS 38.211 lengths.

    This implements TS 38.211 clause 5.2.2.1 for explicitly supplied group,
    base-sequence, and cyclic-shift indices. ``cyclic_shift_count`` is the
    denominator in the normalized cyclic shift; for SRS it is 8, 12, or 6 for
    transmission comb sizes 2, 4, or 8, respectively. Lengths 6, 12, 18,
    and 24 use standardized phase tables; length 30 and lengths at least 36
    use their respective closed-form definitions.
    """

    for name, value in (
        ("sequence_length", sequence_length),
        ("group_index", group_index),
        ("base_sequence_index", base_sequence_index),
        ("cyclic_shift_index", cyclic_shift_index),
        ("cyclic_shift_count", cyclic_shift_count),
        ("cyclic_shift_hopping_offset", cyclic_shift_hopping_offset),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    sequence_length = int(sequence_length)
    group_index = int(group_index)
    base_sequence_index = int(base_sequence_index)
    cyclic_shift_index = int(cyclic_shift_index)
    cyclic_shift_count = int(cyclic_shift_count)
    cyclic_shift_hopping_offset = int(cyclic_shift_hopping_offset)
    if not isinstance(hopping_finer_granularity, bool):
        raise TypeError("hopping_finer_granularity must be a boolean")
    if (
        sequence_length not in _LOW_PAPR_SHORT_PHASES
        and sequence_length != 30
        and sequence_length < 36
    ):
        raise ValueError("supported type-1 sequence lengths are 6, 12, 18, 24, 30, or >=36")
    if not 0 <= group_index <= 29:
        raise ValueError("group_index must be in [0, 29]")
    if base_sequence_index not in {0, 1}:
        raise ValueError("base_sequence_index must be 0 or 1")
    if sequence_length < 72 and base_sequence_index != 0:
        raise ValueError("base_sequence_index must be 0 for sequence lengths below 72")

    if cyclic_shift_count < 1:
        raise ValueError("cyclic_shift_count must be positive")
    if not 0 <= cyclic_shift_index < cyclic_shift_count:
        raise ValueError("cyclic_shift_index must be in [0, cyclic_shift_count)")
    hop_denominator = 2 if hopping_finer_granularity else 1
    if not 0 <= cyclic_shift_hopping_offset < hop_denominator * cyclic_shift_count:
        raise ValueError(
            "cyclic_shift_hopping_offset is outside the configured shift range"
        )
    indices = np.arange(sequence_length, dtype=np.float64)
    if sequence_length in _LOW_PAPR_SHORT_PHASES:
        phases = np.asarray(_LOW_PAPR_SHORT_PHASES[sequence_length][group_index])
        base = np.exp(1j * np.pi * phases / 4.0)
    elif sequence_length == 30:
        base = np.exp(
            -1j * np.pi * group_index * (indices + 1.0) * (indices + 2.0) / 31.0
        )
    else:
        zc_length = _largest_prime_below(sequence_length)
        q_numerator = zc_length * (group_index + 1)
        q_bar = (2 * q_numerator + 31) // 62
        sign_exponent = 2 * q_numerator // 31
        q = q_bar + base_sequence_index * (-1 if sign_exponent % 2 else 1)
        residues = np.remainder(indices, zc_length)
        base = np.exp(-1j * np.pi * q * residues * (residues + 1.0) / zc_length)
    normalized_shift = cyclic_shift_index / cyclic_shift_count + (
        cyclic_shift_hopping_offset / (hop_denominator * cyclic_shift_count)
    )
    shift = np.exp(2j * np.pi * normalized_shift * indices)
    return np.asarray(base * shift, dtype=np.complex128)


def nr_srs_sequence_type1(
    sequence_length: int,
    *,
    group_index: int,
    base_sequence_index: int = 0,
    cyclic_shift_index: int = 0,
    cyclic_shift_hopping_offset: int = 0,
    hopping_finer_granularity: bool = False,
    transmission_comb: int,
) -> np.ndarray:
    """Generate a bounded SRS type-1 sequence with comb-correct cyclic shift.

    Group/base-sequence selection and hopping remain caller-owned. The comb
    determines the TS 38.211 cyclic-shift count through
    :func:`srs_cyclic_shift_count`.
    """

    return nr_low_papr_sequence_type1(
        sequence_length,
        group_index=group_index,
        base_sequence_index=base_sequence_index,
        cyclic_shift_index=cyclic_shift_index,
        cyclic_shift_count=srs_cyclic_shift_count(transmission_comb),
        cyclic_shift_hopping_offset=cyclic_shift_hopping_offset,
        hopping_finer_granularity=hopping_finer_granularity,
    )


def nr_srs_hopped_sequence_type1(
    sequence_length: int,
    *,
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    hopping_mode: str,
    transmission_comb: int,
    cyclic_shift_index: int = 0,
    cyclic_shift_hopping_offset: int = 0,
    hopping_finer_granularity: bool = False,
    slot_symbol_count: int = 14,
) -> np.ndarray:
    """Generate a type-1 SRS sequence using the configured hopping mode.

    ``slot_index`` is frame-local, ``start_symbol_index`` is the absolute slot
    symbol ``l0``, and ``resource_symbol_index`` is the resource-local symbol
    ``l'``. Sequence hopping applies only where the configured sequence-length
    eligibility threshold is met.
    """

    (
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    ) = _validate_srs_hopping_position(
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    )
    if hopping_mode == "neither":
        group_index = sequence_id % 30
        base_sequence_index = 0
    elif hopping_mode == "groupHopping":
        group_index = nr_srs_group_hopping_index(
            sequence_id,
            slot_index,
            start_symbol_index,
            resource_symbol_index,
            slot_symbol_count=slot_symbol_count,
        )
        base_sequence_index = 0
    elif hopping_mode == "sequenceHopping":
        group_index = sequence_id % 30
        base_sequence_index = nr_srs_sequence_hopping_index(
            sequence_id,
            slot_index,
            start_symbol_index,
            resource_symbol_index,
            sequence_length,
            slot_symbol_count=slot_symbol_count,
        )
    else:
        raise ValueError("hopping_mode must be neither, groupHopping, or sequenceHopping")
    return nr_srs_sequence_type1(
        sequence_length,
        group_index=group_index,
        base_sequence_index=base_sequence_index,
        cyclic_shift_index=cyclic_shift_index,
        cyclic_shift_hopping_offset=cyclic_shift_hopping_offset,
        hopping_finer_granularity=hopping_finer_granularity,
        transmission_comb=transmission_comb,
    )


def nr_srs_resource_sequences_type1(
    sequence_length: int,
    *,
    srs_symbol_count: int,
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    hopping_mode: str,
    transmission_comb: int,
    cyclic_shift_index: int = 0,
    cyclic_shift_hopping_offsets: tuple[int, ...] | None = None,
    cyclic_shift_hopping_id: int | None = None,
    radio_frame_index: int | None = None,
    slots_per_frame: int | None = None,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
    slot_symbol_count: int = 14,
) -> np.ndarray:
    """Generate one type-1 sequence row for each symbol in an SRS resource.

    ``start_symbol_index`` is ``l0`` within the slot and ``slot_index`` is
    local to the 10-ms radio frame. Frequency allocation is intentionally
    excluded; the returned ``[symbols, occupied subcarriers]`` matrix can be
    passed directly to :func:`map_nr_srs_resource`. To derive cyclic-shift
    hopping, provide its identity, radio-frame index, and slots per frame;
    this derives one ``f_csh`` value per resource symbol. Derived offsets and
    caller-supplied explicit offsets are mutually exclusive.
    """
    for name, value in (
        ("sequence_length", sequence_length),
        ("srs_symbol_count", srs_symbol_count),
        ("slot_symbol_count", slot_symbol_count),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    sequence_length = int(sequence_length)
    srs_symbol_count = int(srs_symbol_count)
    slot_symbol_count = int(slot_symbol_count)
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS or srs_symbol_count > slot_symbol_count:
        raise ValueError("unsupported SRS symbol count for the declared slot")
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if not isinstance(hopping_finer_granularity, bool):
        raise TypeError("hopping_finer_granularity must be a boolean")
    if cyclic_shift_hopping_id is None:
        if radio_frame_index is not None or slots_per_frame is not None or hopping_subset is not None:
            raise ValueError("cyclic-shift hopping coordinates require cyclic_shift_hopping_id")
    else:
        if cyclic_shift_hopping_offsets is not None:
            raise ValueError("do not combine derived and explicit cyclic-shift hopping offsets")
        if radio_frame_index is None or slots_per_frame is None:
            raise ValueError("cyclic-shift hopping requires radio_frame_index and slots_per_frame")
        cyclic_shift_hopping_offsets = nr_srs_cyclic_shift_hopping_offsets(
            transmission_comb=transmission_comb,
            cyclic_shift_hopping_id=cyclic_shift_hopping_id,
            radio_frame_index=radio_frame_index,
            slot_index=slot_index,
            slots_per_frame=slots_per_frame,
            slot_symbol_count=slot_symbol_count,
            start_symbol_index=start_symbol_index,
            srs_symbol_count=srs_symbol_count,
            hopping_subset=hopping_subset,
            hopping_finer_granularity=hopping_finer_granularity,
        )
    if cyclic_shift_hopping_offsets is None:
        hopping_offsets = (0,) * srs_symbol_count
    else:
        if not isinstance(cyclic_shift_hopping_offsets, tuple):
            raise TypeError("cyclic_shift_hopping_offsets must be a tuple of integers")
        if len(cyclic_shift_hopping_offsets) != srs_symbol_count:
            raise ValueError("cyclic-shift hopping offsets must match the SRS symbol count")
        if any(
            isinstance(offset, bool) or not isinstance(offset, Integral)
            for offset in cyclic_shift_hopping_offsets
        ):
            raise TypeError("cyclic_shift_hopping_offsets must contain integers")
        hopping_offsets = tuple(int(offset) for offset in cyclic_shift_hopping_offsets)
        maximum_hopping_offset = srs_cyclic_shift_count(transmission_comb) * (
            2 if hopping_finer_granularity else 1
        )
        if any(not 0 <= offset < maximum_hopping_offset for offset in hopping_offsets):
            raise ValueError("cyclic-shift hopping offsets exceed the configured granularity")
    rows = [
        nr_srs_hopped_sequence_type1(
            sequence_length,
            sequence_id=sequence_id,
            slot_index=slot_index,
            start_symbol_index=start_symbol_index,
            resource_symbol_index=resource_symbol_index,
            hopping_mode=hopping_mode,
            transmission_comb=transmission_comb,
            cyclic_shift_index=cyclic_shift_index,
            cyclic_shift_hopping_offset=hopping_offsets[resource_symbol_index],
            hopping_finer_granularity=hopping_finer_granularity,
            slot_symbol_count=slot_symbol_count,
        )
        for resource_symbol_index in range(srs_symbol_count)
    ]
    return np.stack(rows, axis=0)


def nr_srs_port_sequences_type1(
    sequence_length: int,
    *,
    cyclic_shift_indices: tuple[int, ...],
    srs_symbol_count: int,
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    hopping_mode: str,
    transmission_comb: int,
    cyclic_shift_hopping_offsets: tuple[int, ...] | None = None,
    cyclic_shift_hopping_id: int | None = None,
    radio_frame_index: int | None = None,
    slots_per_frame: int | None = None,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
    slot_symbol_count: int = 14,
    allow_duplicate_cyclic_shifts: bool = False,
) -> np.ndarray:
    """Generate caller-scheduled SRS sequences for multiple transmit ports.

    The returned array has shape ``[ports, SRS symbols, occupied subcarriers]``.
    ``cyclic_shift_indices`` are explicit zero-based indices in the comb-specific
    cyclic-shift set. This helper applies the same group/sequence hopping state
    to every port; derivation of valid port-specific shifts and resource
    scheduling from higher-layer configuration remain caller-owned. Optional
    cyclic-shift hopping coordinates are forwarded unchanged to each port so
    all ports share the same per-symbol ``f_csh`` sequence.
    """

    if not isinstance(cyclic_shift_indices, tuple) or not cyclic_shift_indices:
        raise TypeError("cyclic_shift_indices must be a nonempty tuple of integers")
    shift_count = srs_cyclic_shift_count(transmission_comb)
    for index in cyclic_shift_indices:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("cyclic_shift_indices must contain integers")
        if not 0 <= index < shift_count:
            raise ValueError(
                f"cyclic-shift indices must be in [0, {shift_count}) for this comb"
            )
    if not isinstance(allow_duplicate_cyclic_shifts, bool):
        raise TypeError("allow_duplicate_cyclic_shifts must be a boolean")
    if not allow_duplicate_cyclic_shifts and len(set(cyclic_shift_indices)) != len(
        cyclic_shift_indices
    ):
        raise ValueError("cyclic_shift_indices must be unique across ports")
    return np.stack(
        [
            nr_srs_resource_sequences_type1(
                sequence_length,
                srs_symbol_count=srs_symbol_count,
                sequence_id=sequence_id,
                slot_index=slot_index,
                start_symbol_index=start_symbol_index,
                hopping_mode=hopping_mode,
                transmission_comb=transmission_comb,
                cyclic_shift_index=int(cyclic_shift_index),
                cyclic_shift_hopping_offsets=cyclic_shift_hopping_offsets,
                cyclic_shift_hopping_id=cyclic_shift_hopping_id,
                radio_frame_index=radio_frame_index,
                slots_per_frame=slots_per_frame,
                hopping_subset=hopping_subset,
                hopping_finer_granularity=hopping_finer_granularity,
                slot_symbol_count=slot_symbol_count,
            )
            for cyclic_shift_index in cyclic_shift_indices
        ],
        axis=0,
    )


def nr_srs_scheduled_port_sequences_type1(
    sequence_length: int,
    *,
    antenna_ports: tuple[int, ...],
    base_cyclic_shift: int,
    srs_symbol_count: int,
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    hopping_mode: str,
    transmission_comb: int,
    ports8_tdm: bool = False,
    cyclic_shift_hopping_offsets: tuple[int, ...] | None = None,
    cyclic_shift_hopping_id: int | None = None,
    radio_frame_index: int | None = None,
    slots_per_frame: int | None = None,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
    slot_symbol_count: int = 14,
) -> np.ndarray:
    """Resolve TS 38.211 shifts and generate configured ports.

    For eight-port TDM, also pass ``ports8_tdm=True`` to the multiport mapper
    so inactive port-symbol elements are omitted from its grid and mask.
    """

    shifts = nr_srs_port_cyclic_shift_indices(
        antenna_ports,
        transmission_comb=transmission_comb,
        base_cyclic_shift=base_cyclic_shift,
        ports8_tdm=ports8_tdm,
    )
    return nr_srs_port_sequences_type1(
        sequence_length,
        cyclic_shift_indices=shifts,
        srs_symbol_count=srs_symbol_count,
        sequence_id=sequence_id,
        slot_index=slot_index,
        start_symbol_index=start_symbol_index,
        hopping_mode=hopping_mode,
        transmission_comb=transmission_comb,
        cyclic_shift_hopping_offsets=cyclic_shift_hopping_offsets,
        cyclic_shift_hopping_id=cyclic_shift_hopping_id,
        radio_frame_index=radio_frame_index,
        slots_per_frame=slots_per_frame,
        hopping_subset=hopping_subset,
        hopping_finer_granularity=hopping_finer_granularity,
        slot_symbol_count=slot_symbol_count,
        allow_duplicate_cyclic_shifts=True,
    )


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSResourceGrid:
    """One mapped SRS resource on a slot-local symbol/subcarrier grid."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        if grid.ndim != 2 or not np.iscomplexobj(grid):
            raise ValueError("SRS grid must be complex [slot symbols, subcarriers]")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("SRS mask must be boolean and match the grid")
        if not np.isfinite(grid).all():
            raise ValueError("SRS grid must be finite")
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(self.subcarrier_indices)
        if not symbols or not subcarriers:
            raise ValueError("SRS resource indices must be nonempty")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in (*symbols, *subcarriers)
        ):
            raise TypeError("SRS resource indices must be integers")
        if (
            len(set(symbols)) != len(symbols)
            or len(set(subcarriers)) != len(subcarriers)
            or any(index < 0 or index >= grid.shape[0] for index in symbols)
            or any(index < 0 or index >= grid.shape[1] for index in subcarriers)
        ):
            raise ValueError("SRS resource indices must be unique and inside the grid")
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        expected_mask[np.ix_(symbols, subcarriers)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("SRS mask disagrees with its declared indices")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(index) for index in symbols))
        object.__setattr__(self, "subcarrier_indices", tuple(int(index) for index in subcarriers))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSHoppedResourceGrid:
    """Mapped SRS with per-symbol frequency-hop and resource-element rosters."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices_by_symbol: tuple[tuple[int, ...], ...]
    position_indices_by_symbol: tuple[tuple[int, ...], ...]
    n_srs_by_symbol: tuple[int | None, ...]
    positioning_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(tuple(row) for row in self.subcarrier_indices_by_symbol)
        positions = tuple(tuple(row) for row in self.position_indices_by_symbol)
        counters = tuple(self.n_srs_by_symbol)
        offsets = tuple(self.positioning_offsets)
        if grid.ndim != 2 or not np.iscomplexobj(grid):
            raise ValueError("hopped SRS grid must be complex [slot symbols, subcarriers]")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("hopped SRS mask must be boolean and match the grid")
        if not np.isfinite(grid).all():
            raise ValueError("hopped SRS grid must be finite")
        rosters = (subcarriers, positions, counters, offsets)
        if not symbols or any(len(roster) != len(symbols) for roster in rosters):
            raise ValueError("hopped SRS symbol and per-symbol rosters must align")
        if any(not row for row in (*subcarriers, *positions)):
            raise ValueError("hopped SRS per-symbol rosters must be nonempty")
        indices = (
            *symbols,
            *offsets,
            *(index for roster in (*subcarriers, *positions) for index in roster),
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in indices):
            raise TypeError("hopped SRS indices and counters must be integers")
        if any(
            value is not None and (isinstance(value, bool) or not isinstance(value, Integral))
            for value in counters
        ):
            raise TypeError("hopped SRS counters must be integers or None")
        if (
            len(set(symbols)) != len(symbols)
            or any(symbol < 0 or symbol >= grid.shape[0] for symbol in symbols)
            or any(
                len(set(row)) != len(row)
                or any(index < 0 or index >= grid.shape[1] for index in row)
                for row in subcarriers
            )
            or any(value < 0 for value in offsets)
            or any(value is not None and value < 0 for value in counters)
            or any(value < 0 for row in positions for value in row)
        ):
            raise ValueError("hopped SRS indices and counters must be nonnegative and in-grid")
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for symbol, row in zip(symbols, subcarriers, strict=True):
            expected_mask[symbol, np.asarray(row, dtype=np.intp)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("hopped SRS mask disagrees with its per-symbol RE roster")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(
            self,
            "subcarrier_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in subcarriers),
        )
        object.__setattr__(
            self,
            "position_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in positions),
        )
        object.__setattr__(
            self,
            "n_srs_by_symbol",
            tuple(None if value is None else int(value) for value in counters),
        )
        object.__setattr__(self, "positioning_offsets", tuple(int(value) for value in offsets))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSHoppedMultiportResourceGrid:
    """Per-port grids for a hopped SRS resource with shared RE rosters."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices_by_symbol: tuple[tuple[int, ...], ...]
    position_indices_by_symbol: tuple[tuple[int, ...], ...]
    n_srs_by_symbol: tuple[int | None, ...]
    positioning_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(tuple(row) for row in self.subcarrier_indices_by_symbol)
        positions = tuple(tuple(row) for row in self.position_indices_by_symbol)
        counters = tuple(self.n_srs_by_symbol)
        offsets = tuple(self.positioning_offsets)
        if grid.ndim != 3 or not np.iscomplexobj(grid):
            raise ValueError(
                "hopped multiport grid must be complex [ports, slot symbols, subcarriers]"
            )
        if grid.shape[0] < 1 or not np.isfinite(grid).all():
            raise ValueError("hopped multiport grid must have ports and finite values")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("hopped multiport mask must be boolean and match the grid")
        rosters = (subcarriers, positions, counters, offsets)
        if not symbols or any(len(roster) != len(symbols) for roster in rosters):
            raise ValueError("hopped multiport symbol and per-symbol rosters must align")
        if any(not row for row in (*subcarriers, *positions)):
            raise ValueError("hopped multiport per-symbol rosters must be nonempty")
        indices = (
            *symbols,
            *offsets,
            *(index for roster in (*subcarriers, *positions) for index in roster),
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in indices):
            raise TypeError("hopped multiport indices must be integers")
        if any(
            value is not None and (isinstance(value, bool) or not isinstance(value, Integral))
            for value in counters
        ):
            raise TypeError("hopped multiport counters must be integers or None")
        if (
            len(set(symbols)) != len(symbols)
            or any(symbol < 0 or symbol >= grid.shape[1] for symbol in symbols)
            or any(
                len(set(row)) != len(row)
                or any(index < 0 or index >= grid.shape[2] for index in row)
                for row in subcarriers
            )
            or any(value < 0 for value in offsets)
            or any(value is not None and value < 0 for value in counters)
            or any(value < 0 for row in positions for value in row)
        ):
            raise ValueError("hopped multiport indices must be nonnegative and in-grid")
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for symbol, row in zip(symbols, subcarriers, strict=True):
            expected_mask[:, symbol, np.asarray(row, dtype=np.intp)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("hopped multiport mask disagrees with its per-symbol RE roster")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(
            self,
            "subcarrier_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in subcarriers),
        )
        object.__setattr__(
            self,
            "position_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in positions),
        )
        object.__setattr__(
            self,
            "n_srs_by_symbol",
            tuple(None if value is None else int(value) for value in counters),
        )
        object.__setattr__(self, "positioning_offsets", tuple(int(value) for value in offsets))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSPositioningResourceGrid:
    """Mapped positioning SRS with its per-symbol comb locations recorded."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices_by_symbol: tuple[tuple[int, ...], ...]
    positioning_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(tuple(row) for row in self.subcarrier_indices_by_symbol)
        offsets = tuple(self.positioning_offsets)
        if grid.ndim != 2 or not np.iscomplexobj(grid):
            raise ValueError("positioning SRS grid must be complex [slot symbols, subcarriers]")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("positioning SRS mask must be boolean and match the grid")
        if not np.isfinite(grid).all():
            raise ValueError("positioning SRS grid must be finite")
        if not symbols or len(symbols) != len(subcarriers) or len(symbols) != len(offsets):
            raise ValueError("symbol, subcarrier, and positioning-offset rosters must align")
        if any(not row for row in subcarriers):
            raise ValueError("each positioning SRS symbol must occupy at least one subcarrier")
        values = (*symbols, *offsets, *(index for row in subcarriers for index in row))
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError("positioning SRS resource indices must be integers")
        if (
            len(set(symbols)) != len(symbols)
            or any(symbol < 0 or symbol >= grid.shape[0] for symbol in symbols)
            or any(
                len(set(row)) != len(row)
                or any(index < 0 or index >= grid.shape[1] for index in row)
                for row in subcarriers
            )
            or any(offset < 0 for offset in offsets)
        ):
            raise ValueError("positioning SRS resource indices must be unique and inside the grid")
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for symbol, row in zip(symbols, subcarriers, strict=True):
            expected_mask[symbol, np.asarray(row, dtype=np.intp)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("positioning SRS mask disagrees with its declared indices")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(
            self,
            "subcarrier_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in subcarriers),
        )
        object.__setattr__(self, "positioning_offsets", tuple(int(value) for value in offsets))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSPositioningMultiportResourceGrid:
    """Mapped positioning SRS with per-port grids and per-symbol RE rosters."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices_by_symbol: tuple[tuple[int, ...], ...]
    positioning_offsets: tuple[int, ...]
    active_port_symbols: tuple[tuple[bool, ...], ...] | None = None

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(tuple(row) for row in self.subcarrier_indices_by_symbol)
        offsets = tuple(self.positioning_offsets)
        if grid.ndim != 3 or not np.iscomplexobj(grid):
            raise ValueError(
                "positioning multiport grid must be complex [ports, slot symbols, subcarriers]"
            )
        if grid.shape[0] < 1 or not np.isfinite(grid).all():
            raise ValueError("positioning multiport grid must have ports and finite values")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("positioning multiport mask must be boolean and match the grid")
        if not symbols or len(symbols) != len(subcarriers) or len(symbols) != len(offsets):
            raise ValueError("symbol, subcarrier, and positioning-offset rosters must align")
        if any(not row for row in subcarriers):
            raise ValueError("each positioning SRS symbol must occupy at least one subcarrier")
        values = (*symbols, *offsets, *(index for row in subcarriers for index in row))
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError("positioning multiport resource indices must be integers")
        if (
            len(set(symbols)) != len(symbols)
            or any(symbol < 0 or symbol >= grid.shape[1] for symbol in symbols)
            or any(
                len(set(row)) != len(row)
                or any(index < 0 or index >= grid.shape[2] for index in row)
                for row in subcarriers
            )
            or any(offset < 0 for offset in offsets)
        ):
            raise ValueError("positioning multiport indices must be unique and inside the grid")
        active = self.active_port_symbols
        if active is None:
            active = tuple((True,) * len(symbols) for _ in range(grid.shape[0]))
        else:
            active = tuple(tuple(row) for row in active)
            if len(active) != grid.shape[0] or any(
                len(row) != len(symbols)
                or any(not isinstance(value, (bool, np.bool_)) for value in row)
                for row in active
            ):
                raise ValueError(
                    "active_port_symbols must be boolean [ports, resource symbols]"
                )
            active = tuple(tuple(bool(value) for value in row) for row in active)
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for port_index, port_activity in enumerate(active):
            for symbol, row, is_active in zip(
                symbols, subcarriers, port_activity, strict=True
            ):
                if is_active:
                    expected_mask[port_index, symbol, np.asarray(row, dtype=np.intp)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("positioning multiport mask disagrees with its declared indices")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(
            self,
            "subcarrier_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in subcarriers),
        )
        object.__setattr__(self, "positioning_offsets", tuple(int(value) for value in offsets))
        object.__setattr__(self, "active_port_symbols", active)


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSMultiportResourceGrid:
    """SRS resources mapped onto independent transmit-port slot grids."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices: tuple[int, ...]
    active_port_symbols: tuple[tuple[bool, ...], ...] | None = None

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        if grid.ndim != 3 or not np.iscomplexobj(grid):
            raise ValueError("multiport SRS grid must be complex [ports, symbols, subcarriers]")
        if grid.shape[0] < 1 or not np.isfinite(grid).all():
            raise ValueError("multiport SRS grid must have ports and finite values")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("multiport SRS mask must be boolean and match the grid")
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(self.subcarrier_indices)
        if not symbols or not subcarriers:
            raise ValueError("SRS resource indices must be nonempty")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in (*symbols, *subcarriers)
        ):
            raise TypeError("SRS resource indices must be integers")
        if (
            len(set(symbols)) != len(symbols)
            or len(set(subcarriers)) != len(subcarriers)
            or any(index < 0 or index >= grid.shape[1] for index in symbols)
            or any(index < 0 or index >= grid.shape[2] for index in subcarriers)
        ):
            raise ValueError("SRS resource indices must be unique and inside the grid")
        active = self.active_port_symbols
        if active is None:
            active = tuple((True,) * len(symbols) for _ in range(grid.shape[0]))
        else:
            active = tuple(tuple(row) for row in active)
            if len(active) != grid.shape[0] or any(
                len(row) != len(symbols)
                or any(not isinstance(value, (bool, np.bool_)) for value in row)
                for row in active
            ):
                raise ValueError(
                    "active_port_symbols must be boolean [ports, resource symbols]"
                )
            active = tuple(tuple(bool(value) for value in row) for row in active)
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for port_index, active_symbols in enumerate(active):
            for symbol_index, is_active in zip(symbols, active_symbols, strict=True):
                if is_active:
                    expected_mask[port_index, symbol_index, np.asarray(subcarriers)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("multiport SRS mask disagrees with its declared indices")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(index) for index in symbols))
        object.__setattr__(self, "subcarrier_indices", tuple(int(index) for index in subcarriers))
        object.__setattr__(self, "active_port_symbols", active)


def map_nr_srs_resource(
    sequence: np.ndarray,
    *,
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    comb_size: int,
    comb_offset: int,
    frequency_start_subcarrier: int = 0,
    allocated_bandwidth_subcarriers: int | None = None,
    beta_srs: float = 1.0,
) -> NRSRSResourceGrid:
    """Map one pre-generated single-port, single-hop SRS sequence into a slot.

    ``start_position_offset`` counts symbols backward from the last slot symbol,
    as in the TS 38.211 ``startPosition`` field. The caller supplies either one
    sequence row reused for every SRS symbol or one row per SRS symbol. A
    symbol row contains one value for each occupied comb subcarrier. When
    ``allocated_bandwidth_subcarriers`` is supplied, the comb is confined to
    that many subcarriers beginning at ``frequency_start_subcarrier``; without
    it, the allocation extends to the end of the grid.

    This function does not generate the sequence or derive the allocation from
    ``c-SRS``/``b-SRS``; ``frequency_start_subcarrier`` and ``comb_offset`` are
    already-resolved mapping parameters.
    """

    integer_values = {
        "slot_symbol_count": slot_symbol_count,
        "subcarrier_count": subcarrier_count,
        "start_position_offset": start_position_offset,
        "srs_symbol_count": srs_symbol_count,
        "comb_size": comb_size,
        "comb_offset": comb_offset,
        "frequency_start_subcarrier": frequency_start_subcarrier,
    }
    if allocated_bandwidth_subcarriers is not None:
        integer_values["allocated_bandwidth_subcarriers"] = allocated_bandwidth_subcarriers
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    slot_symbol_count = int(slot_symbol_count)
    subcarrier_count = int(subcarrier_count)
    start_position_offset = int(start_position_offset)
    srs_symbol_count = int(srs_symbol_count)
    comb_size = int(comb_size)
    comb_offset = int(comb_offset)
    frequency_start_subcarrier = int(frequency_start_subcarrier)
    if allocated_bandwidth_subcarriers is not None:
        allocated_bandwidth_subcarriers = int(allocated_bandwidth_subcarriers)
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if subcarrier_count < 1:
        raise ValueError("subcarrier_count must be positive")
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS or srs_symbol_count > slot_symbol_count:
        raise ValueError("unsupported SRS symbol count for the declared slot")
    if (
        start_position_offset < srs_symbol_count - 1
        or start_position_offset >= slot_symbol_count
    ):
        raise ValueError("start_position_offset must fit the SRS symbols in the slot")
    if comb_size not in _SRS_COMB_SIZES:
        raise ValueError("comb_size must be 2, 4, or 8")
    if not 0 <= comb_offset < comb_size:
        raise ValueError("comb_offset must be in [0, comb_size)")
    if frequency_start_subcarrier < 0:
        raise ValueError("frequency_start_subcarrier must be nonnegative")
    if allocated_bandwidth_subcarriers is not None and (
        allocated_bandwidth_subcarriers < 1
        or frequency_start_subcarrier + allocated_bandwidth_subcarriers > subcarrier_count
    ):
        raise ValueError("allocated bandwidth must be positive and fit inside the grid")
    if (
        isinstance(beta_srs, bool)
        or not isinstance(beta_srs, Real)
        or not np.isfinite(float(beta_srs))
        or beta_srs <= 0
    ):
        raise ValueError("beta_srs must be a finite positive real number")

    values = np.asarray(sequence)
    if not np.iscomplexobj(values):
        raise TypeError("sequence must be complex-valued")
    if values.ndim not in {1, 2}:
        raise ValueError(
            "sequence must have shape [occupied REs] or [SRS symbols, occupied REs]"
        )
    frequency_width = (
        subcarrier_count - frequency_start_subcarrier
        if allocated_bandwidth_subcarriers is None
        else allocated_bandwidth_subcarriers
    )
    remaining_width = frequency_width - comb_offset
    occupied_count = (remaining_width + comb_size - 1) // comb_size
    if occupied_count < 1:
        raise ValueError("SRS comb has no resource elements inside the subcarrier grid")
    if values.shape[-1] != occupied_count:
        raise ValueError("sequence length does not match the occupied SRS comb")
    if values.ndim == 2 and values.shape[0] != srs_symbol_count:
        raise ValueError("per-symbol SRS sequence must include exactly srs_symbol_count rows")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")

    first_symbol = slot_symbol_count - 1 - start_position_offset
    symbol_indices = tuple(range(first_symbol, first_symbol + srs_symbol_count))
    subcarrier_indices = tuple(
        frequency_start_subcarrier + comb_offset + comb_size * index
        for index in range(occupied_count)
    )
    grid = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.complex128)
    mask = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.bool_)
    rows = (
        values
        if values.ndim == 2
        else np.broadcast_to(values, (srs_symbol_count, occupied_count))
    )
    for symbol_offset, symbol_index in enumerate(symbol_indices):
        grid[symbol_index, list(subcarrier_indices)] = beta_srs * rows[symbol_offset]
        mask[symbol_index, list(subcarrier_indices)] = True

    return NRSRSResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=symbol_indices,
        subcarrier_indices=subcarrier_indices,
    )


def map_nr_srs_hopped_resource(
    sequence: np.ndarray,
    *,
    frequency_allocations: tuple[NRSRSFixedFrequencyAllocation, ...]
    | list[NRSRSFixedFrequencyAllocation],
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    beta_srs: float = 1.0,
    positioning_resource: bool = False,
) -> NRSRSHoppedResourceGrid:
    """Map an SRS resource whose frequency allocation can vary by symbol.

    Supply one allocation to reuse a fixed position, or one allocation per
    resource symbol for repetition-group hopping. Each allocation's ``n_srs``
    field is retained in the result. Positioning-resource comb offsets are
    applied per symbol when ``positioning_resource`` is true.
    """

    if not isinstance(positioning_resource, bool):
        raise TypeError("positioning_resource must be a boolean")
    for name, value in (
        ("slot_symbol_count", slot_symbol_count),
        ("subcarrier_count", subcarrier_count),
        ("start_position_offset", start_position_offset),
        ("srs_symbol_count", srs_symbol_count),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    slot_symbol_count = int(slot_symbol_count)
    subcarrier_count = int(subcarrier_count)
    start_position_offset = int(start_position_offset)
    srs_symbol_count = int(srs_symbol_count)
    allocations = tuple(frequency_allocations)
    if not allocations or any(
        not isinstance(item, NRSRSFixedFrequencyAllocation) for item in allocations
    ):
        raise TypeError("frequency_allocations must contain resolved SRS allocation objects")
    if len(allocations) not in {1, srs_symbol_count}:
        raise ValueError("frequency_allocations must contain one or one-per-symbol allocation")
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if subcarrier_count < 1:
        raise ValueError("subcarrier_count must be positive")
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS or srs_symbol_count > slot_symbol_count:
        raise ValueError("unsupported SRS symbol count for the declared slot")
    if (
        start_position_offset < srs_symbol_count - 1
        or start_position_offset >= slot_symbol_count
    ):
        raise ValueError("start_position_offset must fit the SRS symbols in the slot")
    if (
        isinstance(beta_srs, bool)
        or not isinstance(beta_srs, Real)
        or not np.isfinite(float(beta_srs))
        or beta_srs <= 0
    ):
        raise ValueError("beta_srs must be a finite positive real number")

    first = allocations[0]
    if any(
        allocation.bandwidth != first.bandwidth
        or allocation.comb_offset != first.comb_offset
        or allocation.frequency_domain_position != first.frequency_domain_position
        or allocation.frequency_domain_shift_rb != first.frequency_domain_shift_rb
        or allocation.reference_subcarrier != first.reference_subcarrier
        or allocation.b_hop != first.b_hop
        for allocation in allocations[1:]
    ):
        raise ValueError("per-symbol frequency allocations must share one SRS configuration")
    rows_allocations = (
        allocations * srs_symbol_count if len(allocations) == 1 else allocations
    )
    width = first.bandwidth.allocated_bandwidth_subcarriers
    comb_size = first.bandwidth.transmission_comb
    offsets = (
        tuple(
            nr_srs_positioning_symbol_offset(
                comb_size,
                srs_symbol_count=srs_symbol_count,
                resource_symbol_index=index,
            )
            for index in range(srs_symbol_count)
        )
        if positioning_resource
        else (0,) * srs_symbol_count
    )
    values = np.asarray(sequence)
    if not np.iscomplexobj(values):
        raise TypeError("sequence must be complex-valued")
    if values.ndim not in {1, 2}:
        raise ValueError("sequence must have shape [occupied REs] or [SRS symbols, occupied REs]")
    if values.shape[-1] != first.bandwidth.sequence_length:
        raise ValueError("sequence length does not match the configured SRS bandwidth and comb")
    if values.ndim == 2 and values.shape[0] != srs_symbol_count:
        raise ValueError("per-symbol SRS sequence must include exactly srs_symbol_count rows")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")

    first_symbol = slot_symbol_count - 1 - start_position_offset
    symbol_indices = tuple(range(first_symbol, first_symbol + srs_symbol_count))
    rows = (
        values
        if values.ndim == 2
        else np.broadcast_to(values, (srs_symbol_count, first.bandwidth.sequence_length))
    )
    grid = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.complex128)
    mask = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.bool_)
    subcarriers_by_symbol: list[tuple[int, ...]] = []
    for row_index, (symbol, allocation, positioning_offset) in enumerate(
        zip(symbol_indices, rows_allocations, offsets, strict=True)
    ):
        if allocation.start_subcarrier + width > subcarrier_count:
            raise ValueError("a hopped SRS allocation does not fit inside the subcarrier grid")
        effective_comb_offset = (allocation.comb_offset + positioning_offset) % comb_size
        subcarriers = tuple(
            allocation.start_subcarrier + effective_comb_offset + comb_size * index
            for index in range(first.bandwidth.sequence_length)
        )
        if subcarriers[-1] >= allocation.start_subcarrier + width:
            raise ValueError("SRS comb placement exceeds its resolved allocation bandwidth")
        subcarriers_by_symbol.append(subcarriers)
        grid[symbol, list(subcarriers)] = beta_srs * rows[row_index]
        mask[symbol, list(subcarriers)] = True

    return NRSRSHoppedResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=symbol_indices,
        subcarrier_indices_by_symbol=tuple(subcarriers_by_symbol),
        position_indices_by_symbol=tuple(item.position_indices for item in rows_allocations),
        n_srs_by_symbol=tuple(item.n_srs for item in rows_allocations),
        positioning_offsets=offsets,
    )


def map_nr_srs_hopped_multiport_resource(
    sequences: np.ndarray,
    *,
    frequency_allocations: tuple[NRSRSFixedFrequencyAllocation, ...]
    | list[NRSRSFixedFrequencyAllocation],
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    beta_srs: float = 1.0,
    positioning_resource: bool = False,
    four_port_srs_3tx: bool = False,
) -> NRSRSHoppedMultiportResourceGrid:
    """Map caller-scheduled multiport sequences over resolved SRS hops.

    Applies the TS 38.211 per-port ``1/sqrt(N_ap)`` amplitude factor; the
    four-port three-transmit configuration uses an effective count of three.
    """

    values = np.asarray(sequences)
    if not np.iscomplexobj(values):
        raise TypeError("sequences must be complex-valued")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError(
            "sequences must have shape [ports, SRS symbols, occupied REs]"
        )
    if values.shape[1] != srs_symbol_count:
        raise ValueError("each port must have exactly srs_symbol_count rows")
    normalized_beta = _normalized_multiport_beta(
        beta_srs, values.shape[0], four_port_srs_3tx
    )
    mapped = tuple(
        map_nr_srs_hopped_resource(
            port_sequences,
            frequency_allocations=frequency_allocations,
            slot_symbol_count=slot_symbol_count,
            subcarrier_count=subcarrier_count,
            start_position_offset=start_position_offset,
            srs_symbol_count=srs_symbol_count,
            beta_srs=normalized_beta,
            positioning_resource=positioning_resource,
        )
        for port_sequences in values
    )
    first = mapped[0]
    return NRSRSHoppedMultiportResourceGrid(
        grid=np.stack([item.grid for item in mapped], axis=0),
        mask=np.stack([item.mask for item in mapped], axis=0),
        symbol_indices=first.symbol_indices,
        subcarrier_indices_by_symbol=first.subcarrier_indices_by_symbol,
        position_indices_by_symbol=first.position_indices_by_symbol,
        n_srs_by_symbol=first.n_srs_by_symbol,
        positioning_offsets=first.positioning_offsets,
    )


def map_nr_srs_positioning_resource(
    sequence: np.ndarray,
    *,
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    comb_size: int,
    comb_offset: int,
    frequency_start_subcarrier: int,
    allocated_bandwidth_subcarriers: int,
    beta_srs: float = 1.0,
) -> NRSRSPositioningResourceGrid:
    """Map a single-port SRS-PosResource with the table offset per symbol.

    The sequence is one row reused or one row per resource symbol. The
    allocation start and width must already be resolved. Each resource-local
    symbol uses TS 38.211 Table 6.4.1.4.3-2 before applying the transmission
    comb, so the occupied subcarrier roster can vary by symbol.
    """

    integer_values = {
        "slot_symbol_count": slot_symbol_count,
        "subcarrier_count": subcarrier_count,
        "start_position_offset": start_position_offset,
        "srs_symbol_count": srs_symbol_count,
        "comb_size": comb_size,
        "comb_offset": comb_offset,
        "frequency_start_subcarrier": frequency_start_subcarrier,
        "allocated_bandwidth_subcarriers": allocated_bandwidth_subcarriers,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    slot_symbol_count = int(slot_symbol_count)
    subcarrier_count = int(subcarrier_count)
    start_position_offset = int(start_position_offset)
    srs_symbol_count = int(srs_symbol_count)
    comb_size = int(comb_size)
    comb_offset = int(comb_offset)
    frequency_start_subcarrier = int(frequency_start_subcarrier)
    allocated_bandwidth_subcarriers = int(allocated_bandwidth_subcarriers)
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if subcarrier_count < 1:
        raise ValueError("subcarrier_count must be positive")
    if not 1 <= srs_symbol_count <= slot_symbol_count:
        raise ValueError("symbol count is unsupported for this SRS-PosResource comb")
    offsets = tuple(
        nr_srs_positioning_symbol_offset(
            comb_size,
            srs_symbol_count=srs_symbol_count,
            resource_symbol_index=symbol_index,
        )
        for symbol_index in range(srs_symbol_count)
    )
    if (
        start_position_offset < srs_symbol_count - 1
        or start_position_offset >= slot_symbol_count
    ):
        raise ValueError("start_position_offset must fit the SRS symbols in the slot")
    if comb_size not in _SRS_COMB_SIZES:
        raise ValueError("comb_size must be 2, 4, or 8")
    if not 0 <= comb_offset < comb_size:
        raise ValueError("comb_offset must be in [0, comb_size)")
    if frequency_start_subcarrier < 0:
        raise ValueError("frequency_start_subcarrier must be nonnegative")
    if allocated_bandwidth_subcarriers < 1 or (
        allocated_bandwidth_subcarriers % comb_size
        or frequency_start_subcarrier + allocated_bandwidth_subcarriers > subcarrier_count
    ):
        raise ValueError("allocated bandwidth must be positive, comb-aligned, and fit the grid")
    if (
        isinstance(beta_srs, bool)
        or not isinstance(beta_srs, Real)
        or not np.isfinite(float(beta_srs))
        or beta_srs <= 0
    ):
        raise ValueError("beta_srs must be a finite positive real number")

    values = np.asarray(sequence)
    if not np.iscomplexobj(values):
        raise TypeError("sequence must be complex-valued")
    if values.ndim not in {1, 2}:
        raise ValueError(
            "sequence must have shape [occupied REs] or [SRS symbols, occupied REs]"
        )
    expected_length = allocated_bandwidth_subcarriers // comb_size
    if values.shape[-1] != expected_length:
        raise ValueError("sequence length does not match the positioning SRS comb")
    if values.ndim == 2 and values.shape[0] != srs_symbol_count:
        raise ValueError("per-symbol SRS sequence must include exactly srs_symbol_count rows")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")

    first_symbol = slot_symbol_count - 1 - start_position_offset
    symbol_indices = tuple(range(first_symbol, first_symbol + srs_symbol_count))
    subcarriers_by_symbol = tuple(
        tuple(
            frequency_start_subcarrier
            + (comb_offset + offset) % comb_size
            + comb_size * index
            for index in range(expected_length)
        )
        for offset in offsets
    )
    grid = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.complex128)
    mask = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.bool_)
    rows = (
        values
        if values.ndim == 2
        else np.broadcast_to(values, (srs_symbol_count, expected_length))
    )
    for row_index, (symbol, subcarriers) in enumerate(
        zip(symbol_indices, subcarriers_by_symbol, strict=True)
    ):
        grid[symbol, list(subcarriers)] = beta_srs * rows[row_index]
        mask[symbol, list(subcarriers)] = True
    return NRSRSPositioningResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=symbol_indices,
        subcarrier_indices_by_symbol=subcarriers_by_symbol,
        positioning_offsets=offsets,
    )


def map_nr_srs_positioning_multiport_resource(
    sequences: np.ndarray,
    *,
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    comb_size: int,
    comb_offset: int,
    frequency_start_subcarrier: int,
    allocated_bandwidth_subcarriers: int,
    beta_srs: float = 1.0,
    four_port_srs_3tx: bool = False,
    ports8_tdm: bool = False,
) -> NRSRSPositioningMultiportResourceGrid:
    """Map caller-scheduled positioning-SRS rows with TS per-port scaling."""

    values = np.asarray(sequences)
    if not np.iscomplexobj(values):
        raise TypeError("sequences must be complex-valued")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError(
            "sequences must have shape [ports, SRS symbols, occupied REs]"
        )
    if values.shape[1] != srs_symbol_count:
        raise ValueError("each port must have exactly srs_symbol_count rows")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    if ports8_tdm and (values.shape[0] != 8 or four_port_srs_3tx):
        raise ValueError("ports8_tdm requires eight ports and cannot combine with four-port 3Tx")
    normalized_beta = _normalized_multiport_beta(
        beta_srs, values.shape[0], four_port_srs_3tx
    )
    mapped = [
        map_nr_srs_positioning_resource(
            port_sequences,
            slot_symbol_count=slot_symbol_count,
            subcarrier_count=subcarrier_count,
            start_position_offset=start_position_offset,
            srs_symbol_count=srs_symbol_count,
            comb_size=comb_size,
            comb_offset=comb_offset,
            frequency_start_subcarrier=frequency_start_subcarrier,
            allocated_bandwidth_subcarriers=allocated_bandwidth_subcarriers,
            beta_srs=normalized_beta,
        )
        for port_sequences in values
    ]
    grid = np.stack([item.grid for item in mapped], axis=0)
    mask = np.stack([item.mask for item in mapped], axis=0)
    activity = (
        nr_srs_port_symbol_activity(
            tuple(range(1000, 1008)),
            srs_symbol_count=srs_symbol_count,
            ports8_tdm=True,
        )
        if ports8_tdm
        else None
    )
    if activity is not None:
        for port_index, active_symbols in enumerate(activity):
            for symbol_index, is_active in zip(
                mapped[0].symbol_indices, active_symbols, strict=True
            ):
                if not is_active:
                    grid[port_index, symbol_index, :] = 0
                    mask[port_index, symbol_index, :] = False
    return NRSRSPositioningMultiportResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=mapped[0].symbol_indices,
        subcarrier_indices_by_symbol=mapped[0].subcarrier_indices_by_symbol,
        positioning_offsets=mapped[0].positioning_offsets,
        active_port_symbols=activity,
    )


def map_nr_srs_multiport_resource(
    sequences: np.ndarray,
    *,
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    comb_size: int,
    comb_offset: int,
    frequency_start_subcarrier: int = 0,
    allocated_bandwidth_subcarriers: int | None = None,
    beta_srs: float = 1.0,
    four_port_srs_3tx: bool = False,
    ports8_tdm: bool = False,
) -> NRSRSMultiportResourceGrid:
    """Map per-port SRS rows with TS 38.211 amplitude normalization."""

    values = np.asarray(sequences)
    if not np.iscomplexobj(values):
        raise TypeError("sequences must be complex-valued")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("sequences must have shape [ports, SRS symbols, occupied REs]")
    if values.shape[1] != srs_symbol_count:
        raise ValueError("each port must have exactly srs_symbol_count rows")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    if ports8_tdm and (values.shape[0] != 8 or four_port_srs_3tx):
        raise ValueError("ports8_tdm requires eight ports and cannot combine with four-port 3Tx")
    normalized_beta = _normalized_multiport_beta(
        beta_srs, values.shape[0], four_port_srs_3tx
    )
    mapped = [
        map_nr_srs_resource(
            port_sequences,
            slot_symbol_count=slot_symbol_count,
            subcarrier_count=subcarrier_count,
            start_position_offset=start_position_offset,
            srs_symbol_count=srs_symbol_count,
            comb_size=comb_size,
            comb_offset=comb_offset,
            frequency_start_subcarrier=frequency_start_subcarrier,
            allocated_bandwidth_subcarriers=allocated_bandwidth_subcarriers,
            beta_srs=normalized_beta,
        )
        for port_sequences in values
    ]
    grid = np.stack([item.grid for item in mapped], axis=0)
    activity = (
        nr_srs_port_symbol_activity(
            tuple(range(1000, 1008)),
            srs_symbol_count=srs_symbol_count,
            ports8_tdm=True,
        )
        if ports8_tdm
        else None
    )
    mask = np.stack([item.mask for item in mapped], axis=0)
    if activity is not None:
        for port_index, active_symbols in enumerate(activity):
            for symbol_index, is_active in zip(
                mapped[0].symbol_indices, active_symbols, strict=True
            ):
                if not is_active:
                    grid[port_index, symbol_index, :] = 0
                    mask[port_index, symbol_index, :] = False
    return NRSRSMultiportResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=mapped[0].symbol_indices,
        subcarrier_indices=mapped[0].subcarrier_indices,
        active_port_symbols=activity,
    )


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSSinglePortChannelEstimate:
    """Per-resource-element LS channel estimates from one mapped SRS port.

    Entries outside ``mask`` are zero. No averaging or interpolation is
    applied, so the result preserves the observed time-frequency locations.
    """

    channel_by_re: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        channel = np.asarray(self.channel_by_re)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        if channel.ndim != 2 or not np.iscomplexobj(channel):
            raise ValueError("channel_by_re must be complex [slot symbols, subcarriers]")
        if mask.shape != channel.shape or mask.dtype != np.bool_:
            raise ValueError("channel estimate mask must be boolean and match channel_by_re")
        if not np.isfinite(channel).all():
            raise ValueError("channel_by_re must contain only finite values")
        if np.any(channel[~mask] != 0):
            raise ValueError("channel_by_re must be zero outside the SRS resource mask")
        if (
            not symbols
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in symbols)
            or len(set(symbols)) != len(symbols)
            or any(i < 0 or i >= channel.shape[0] for i in symbols)
        ):
            raise ValueError("symbol_indices must be unique valid slot-symbol indices")
        channel_view = channel.view()
        mask_view = mask.view()
        channel_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "channel_by_re", channel_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(i) for i in symbols))


def nr_srs_single_port_ls_channel_estimate(
    mapped_resource: NRSRSResourceGrid | NRSRSHoppedResourceGrid | NRSRSPositioningResourceGrid,
    received_grid: np.ndarray,
) -> NRSRSSinglePortChannelEstimate:
    """Estimate the effective channel on mapped single-port SRS REs.

    The estimator computes ``Y/X`` only where the transmitted resource mask
    is true. It requires the receiver grid to use the same slot/subcarrier
    coordinates and the mapped transmit values to be nonzero on every active
    RE. The output is an un-interpolated per-RE estimate; synchronization,
    receive combining, multiport separation, and noise-variance estimation
    remain receiver-level responsibilities.
    """

    if not isinstance(
        mapped_resource,
        (NRSRSResourceGrid, NRSRSHoppedResourceGrid, NRSRSPositioningResourceGrid),
    ):
        raise TypeError("mapped_resource must be a mapped single-port SRS resource")
    received = np.asarray(received_grid)
    if received.ndim != 2 or not np.iscomplexobj(received):
        raise ValueError("received_grid must be complex [slot symbols, subcarriers]")
    if received.shape != mapped_resource.grid.shape:
        raise ValueError("received_grid shape must match the mapped SRS grid")
    if not np.isfinite(received).all():
        raise ValueError("received_grid must contain only finite values")
    transmit = mapped_resource.grid
    mask = mapped_resource.mask
    if not mask.any():
        raise ValueError("mapped SRS resource must occupy at least one resource element")
    if np.any(transmit[mask] == 0):
        raise ValueError("mapped SRS transmit values must be nonzero on active resource elements")
    estimate = np.zeros(transmit.shape, dtype=np.complex128)
    estimate[mask] = received[mask] / transmit[mask]
    return NRSRSSinglePortChannelEstimate(
        channel_by_re=estimate,
        mask=mask,
        symbol_indices=mapped_resource.symbol_indices,
    )


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSMultiportChannelEstimate:
    """Multiport LS estimates and conditional noise covariance by subcarrier.

    ``channel_by_rx_port_subcarrier`` has shape ``[rx antennas, TX ports,
    subcarriers]``. ``noise_covariance_by_rx_subcarrier`` has shape
    ``[rx antennas, subcarriers, TX ports, TX ports]``. Unavailable
    subcarriers are zero-filled and identified by a false validity mask plus a
    reason code; no frequency interpolation is performed. The LS model assumes
    the channel is constant over this SRS resource's symbols. Noise may use one
    variance per RX antenna or a per-RX/per-symbol variance profile; the latter
    drives weighted LS and is retained in the result. In profile mode,
    ``noise_variance_by_rx_ant`` is its per-RX slot-symbol mean. Synchronization
    errors and channel evolution remain unmodeled.
    """

    channel_by_rx_port_subcarrier: np.ndarray
    noise_covariance_by_rx_subcarrier: np.ndarray
    valid_subcarrier_mask: np.ndarray
    rank_by_subcarrier: np.ndarray
    condition_number_by_subcarrier: tuple[float | None, ...]
    reason_codes: tuple[str | None, ...]
    noise_variance_by_rx_ant: np.ndarray
    symbol_indices: tuple[int, ...]
    noise_variance_by_rx_symbol: np.ndarray | None = None

    def __post_init__(self) -> None:
        channel = np.asarray(self.channel_by_rx_port_subcarrier)
        covariance = np.asarray(self.noise_covariance_by_rx_subcarrier)
        valid = np.asarray(self.valid_subcarrier_mask)
        rank = np.asarray(self.rank_by_subcarrier)
        noise = np.asarray(self.noise_variance_by_rx_ant)
        symbol_noise = (
            None
            if self.noise_variance_by_rx_symbol is None
            else np.asarray(self.noise_variance_by_rx_symbol)
        )
        conditions = tuple(self.condition_number_by_subcarrier)
        reasons = tuple(self.reason_codes)
        symbols = tuple(self.symbol_indices)
        if channel.ndim != 3 or not np.iscomplexobj(channel):
            raise ValueError("channel estimate must be complex [rx antennas, TX ports, subcarriers]")
        rx_count, port_count, subcarrier_count = channel.shape
        if min(rx_count, port_count, subcarrier_count) < 1:
            raise ValueError("channel estimate dimensions must be nonempty")
        if covariance.shape != (rx_count, subcarrier_count, port_count, port_count):
            raise ValueError("noise covariance shape must be [rx antennas, subcarriers, ports, ports]")
        if not np.iscomplexobj(covariance):
            raise TypeError("noise covariance must be complex-valued")
        if valid.shape != (subcarrier_count,) or valid.dtype != np.bool_:
            raise ValueError("valid_subcarrier_mask must be boolean [subcarriers]")
        if rank.shape != (subcarrier_count,) or rank.dtype.kind not in "iu":
            raise ValueError("rank_by_subcarrier must be an integer [subcarriers] vector")
        if len(conditions) != subcarrier_count or len(reasons) != subcarrier_count:
            raise ValueError("condition and reason rosters must match subcarrier count")
        if noise.shape != (rx_count,) or noise.dtype.kind not in "iuf":
            raise ValueError("noise_variance_by_rx_ant must be real [rx antennas]")
        if not symbols or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in symbols
        ) or len(set(symbols)) != len(symbols):
            raise ValueError("symbol_indices must be unique nonnegative integers")
        if symbol_noise is not None and (
            symbol_noise.ndim != 2
            or symbol_noise.shape[0] != rx_count
            or symbol_noise.shape[1] <= max(symbols, default=-1)
            or symbol_noise.dtype.kind not in "iuf"
            or not np.isfinite(symbol_noise).all()
            or np.any(symbol_noise <= 0)
        ):
            raise ValueError("noise_variance_by_rx_symbol must be positive real [rx, symbols]")
        if (
            not np.isfinite(channel).all()
            or not np.isfinite(covariance).all()
            or not np.isfinite(noise).all()
            or np.any(noise <= 0)
            or np.any(rank < 0)
            or np.any(rank > port_count)
        ):
            raise ValueError("estimates, covariance, noise, or rank contain invalid values")
        for index, (is_valid, condition, reason) in enumerate(
            zip(valid, conditions, reasons, strict=True)
        ):
            if is_valid:
                if rank[index] != port_count or condition is None or not np.isfinite(condition):
                    raise ValueError("valid subcarriers require full rank and finite conditioning")
                if reason is not None:
                    raise ValueError("valid subcarriers cannot have an unavailable reason")
            elif not isinstance(reason, str) or not reason:
                raise ValueError("unavailable subcarriers require a reason code")
            if not is_valid and (
                np.any(channel[:, :, index] != 0) or np.any(covariance[:, index] != 0)
            ):
                raise ValueError("unavailable subcarriers must have zero estimates and covariance")
        channel_view = channel.view()
        covariance_view = covariance.view()
        valid_view = valid.view()
        rank_view = rank.view()
        noise_view = noise.view()
        symbol_noise_view = None if symbol_noise is None else symbol_noise.view()
        arrays_to_freeze = [channel_view, covariance_view, valid_view, rank_view, noise_view]
        if symbol_noise_view is not None:
            arrays_to_freeze.append(symbol_noise_view)
        for value in arrays_to_freeze:
            value.setflags(write=False)
        object.__setattr__(self, "channel_by_rx_port_subcarrier", channel_view)
        object.__setattr__(self, "noise_covariance_by_rx_subcarrier", covariance_view)
        object.__setattr__(self, "valid_subcarrier_mask", valid_view)
        object.__setattr__(self, "rank_by_subcarrier", rank_view)
        object.__setattr__(self, "condition_number_by_subcarrier", conditions)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "noise_variance_by_rx_ant", noise_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(self, "noise_variance_by_rx_symbol", symbol_noise_view)


def nr_srs_multiport_ls_channel_estimate(
    mapped_resource: (
        NRSRSMultiportResourceGrid
        | NRSRSHoppedMultiportResourceGrid
        | NRSRSPositioningMultiportResourceGrid
    ),
    received_grid: np.ndarray,
    *,
    noise_variance_by_rx_ant: float | Sequence[float] | None = None,
    noise_variance_by_rx_symbol: np.ndarray | None = None,
) -> NRSRSMultiportChannelEstimate:
    """Separate mapped SRS transmit ports with per-subcarrier least squares.

    The receiver grid is ``[rx antennas, slot symbols, subcarriers]`` and the
    mapped transmit grid is ``[TX ports, slot symbols, subcarriers]``. For each
    subcarrier this solves ``Y = X H`` over the mapped SRS symbols. A rank
    deficient port signature is retained as an unavailable subcarrier rather
    than producing an underdetermined estimate. Per-RX covariance is computed
    from the conditional weighted-LS equations. The optional variance profile
    has shape ``[rx antennas, slot symbols]`` and weights each observed SRS
    symbol independently; it can represent a caller-calibrated changing noise
    or interference level. No variance is inferred from the received grid.
    """
    supported = (
        NRSRSMultiportResourceGrid,
        NRSRSHoppedMultiportResourceGrid,
        NRSRSPositioningMultiportResourceGrid,
    )
    if not isinstance(mapped_resource, supported):
        raise TypeError("mapped_resource must be a mapped multiport SRS resource")
    received = np.asarray(received_grid)
    transmit = mapped_resource.grid
    transmit_mask = mapped_resource.mask
    if received.ndim != 3 or not np.iscomplexobj(received):
        raise ValueError("received_grid must be complex [rx antennas, slot symbols, subcarriers]")
    if received.shape[0] < 1:
        raise ValueError("received_grid must contain at least one RX antenna")
    if received.shape[1:] != transmit.shape[1:]:
        raise ValueError("received_grid symbol/subcarrier axes must match the mapped SRS grid")
    if not np.isfinite(received).all():
        raise ValueError("received_grid must contain only finite values")
    if not transmit_mask.any():
        raise ValueError("mapped SRS resource must occupy at least one resource element")
    if np.any(transmit[transmit_mask] == 0):
        raise ValueError("mapped SRS transmit values must be nonzero on active resource elements")

    rx_count, symbol_count, _ = received.shape
    if noise_variance_by_rx_symbol is not None and noise_variance_by_rx_ant is not None:
        raise ValueError("provide per-RX or per-RX/per-symbol noise variances, not both")
    symbol_noise: np.ndarray | None
    symbol_weights: np.ndarray | None = None
    noise_weights: np.ndarray | None = None
    if noise_variance_by_rx_symbol is not None:
        raw_symbol_noise = np.asarray(noise_variance_by_rx_symbol)
        if raw_symbol_noise.dtype.kind not in "iuf" or raw_symbol_noise.dtype.kind == "b":
            raise TypeError("noise_variance_by_rx_symbol must be real-valued")
        if raw_symbol_noise.shape != (rx_count, symbol_count):
            raise ValueError("noise_variance_by_rx_symbol must have shape [rx, slot symbols]")
        symbol_noise = np.array(raw_symbol_noise, dtype=np.float64, copy=True)
        if not np.isfinite(symbol_noise).all() or np.any(symbol_noise <= 0):
            raise ValueError("per-symbol noise variances must be finite and positive")
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            symbol_weights = np.reciprocal(symbol_noise)
        if not np.isfinite(symbol_weights).all() or np.any(symbol_weights <= 0):
            raise ValueError("per-symbol noise variance reciprocals must be representable")
        noise = np.mean(symbol_noise, axis=1)
        if not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError("mean per-RX noise variance must be finite and positive")
    else:
        if noise_variance_by_rx_ant is None:
            raise ValueError("per-RX noise variance is required")
        raw_noise = np.asarray(noise_variance_by_rx_ant)
        if raw_noise.dtype.kind not in "iuf" or raw_noise.dtype.kind == "b":
            raise TypeError("noise_variance_by_rx_ant must be real-valued")
        if raw_noise.ndim == 0:
            noise = np.full(rx_count, float(raw_noise), dtype=np.float64)
        elif raw_noise.shape == (rx_count,):
            noise = np.array(raw_noise, dtype=np.float64, copy=True)
        else:
            raise ValueError("noise_variance_by_rx_ant must be scalar or one value per RX antenna")
        if not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError("noise variances must be finite and positive")
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            noise_weights = np.reciprocal(noise)
        if not np.isfinite(noise_weights).all() or np.any(noise_weights <= 0):
            raise ValueError("noise variance reciprocals must be representable")
        symbol_noise = None

    port_count, _, subcarrier_count = transmit.shape
    channel = np.zeros((rx_count, port_count, subcarrier_count), dtype=np.complex128)
    covariance = np.zeros(
        (rx_count, subcarrier_count, port_count, port_count), dtype=np.complex128
    )
    valid = np.zeros(subcarrier_count, dtype=np.bool_)
    ranks = np.zeros(subcarrier_count, dtype=np.int64)
    condition_numbers: list[float | None] = [None] * subcarrier_count
    reasons: list[str | None] = ["not_occupied"] * subcarrier_count
    symbols = tuple(mapped_resource.symbol_indices)

    for subcarrier in range(subcarrier_count):
        design = transmit[:, symbols, subcarrier].T
        occupied_rows = np.any(transmit_mask[:, symbols, subcarrier], axis=0)
        design = design[occupied_rows]
        if design.shape[0] == 0:
            continue
        rank = int(np.linalg.matrix_rank(design))
        ranks[subcarrier] = rank
        if rank < port_count:
            reasons[subcarrier] = "rank_deficient_port_signature"
            continue
        condition = float(np.linalg.cond(design))
        if not np.isfinite(condition):
            reasons[subcarrier] = "numerically_singular_port_signature"
            continue
        condition_numbers[subcarrier] = condition
        observed = received[:, np.asarray(symbols)[occupied_rows], subcarrier].T
        selected_symbols = np.asarray(symbols)[occupied_rows]
        for rx_index in range(rx_count):
            if symbol_noise is None:
                assert noise_weights is not None
                weights = np.full(design.shape[0], noise_weights[rx_index])
            else:
                assert symbol_weights is not None
                weights = symbol_weights[rx_index, selected_symbols]
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_observed = observed[:, rx_index] * np.sqrt(weights)
            try:
                coefficients, _, solved_rank, _ = np.linalg.lstsq(
                    weighted_design, weighted_observed, rcond=None
                )
            except np.linalg.LinAlgError:
                channel[:, :, subcarrier] = 0
                covariance[:, subcarrier] = 0
                reasons[subcarrier] = "weighted_solver_failure"
                condition_numbers[subcarrier] = None
                break
            if solved_rank != port_count:
                channel[:, :, subcarrier] = 0
                covariance[:, subcarrier] = 0
                ranks[subcarrier] = int(solved_rank)
                reasons[subcarrier] = "rank_deficient_port_signature"
                condition_numbers[subcarrier] = None
                break
            gram = design.conj().T @ (weights[:, None] * design)
            try:
                coefficient_covariance = np.linalg.inv(gram)
            except np.linalg.LinAlgError:
                channel[:, :, subcarrier] = 0
                covariance[:, subcarrier] = 0
                reasons[subcarrier] = "singular_weighted_port_signature"
                condition_numbers[subcarrier] = None
                break
            if not np.isfinite(coefficients).all() or not np.isfinite(
                coefficient_covariance
            ).all():
                channel[:, :, subcarrier] = 0
                covariance[:, subcarrier] = 0
                reasons[subcarrier] = "nonfinite_weighted_estimate"
                condition_numbers[subcarrier] = None
                break
            covariance[rx_index, subcarrier] = coefficient_covariance
            channel[rx_index, :, subcarrier] = coefficients
        else:
            valid[subcarrier] = True
            reasons[subcarrier] = None

    return NRSRSMultiportChannelEstimate(
        channel_by_rx_port_subcarrier=channel,
        noise_covariance_by_rx_subcarrier=covariance,
        valid_subcarrier_mask=valid,
        rank_by_subcarrier=ranks,
        condition_number_by_subcarrier=tuple(condition_numbers),
        reason_codes=tuple(reasons),
        noise_variance_by_rx_ant=noise,
        symbol_indices=symbols,
        noise_variance_by_rx_symbol=symbol_noise,
    )
