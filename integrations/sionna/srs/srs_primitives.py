"""Constants and primitive lookups for NR SRS configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from numbers import Integral, Real
from typing import cast

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


_SRS_PERIODICITY_EXT_OFFSETS = {f"sl{period}": period for period in (128, 256, 512, 20480)}


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
        (-3, -1, 3, 3, -1, -3),
        (-3, 3, -1, -1, 3, -3),
        (-3, -3, -3, 3, 1, -3),
        (1, 1, 1, 3, -1, -3),
        (1, 1, 1, -3, -1, 3),
        (-3, 1, -1, -3, -3, -3),
        (-3, 1, 3, -3, -3, -3),
        (-3, -1, 1, -3, 1, -1),
        (-3, -1, -3, 1, -3, -3),
        (-3, -3, 1, -3, 3, -3),
        (-3, 1, 3, 1, -3, -3),
        (-3, -1, -3, 1, 1, -3),
        (1, 1, 3, -1, -3, 3),
        (1, 1, 3, 3, -1, 3),
        (1, 1, 1, -3, 3, -1),
        (1, 1, 1, -1, 3, -3),
        (-3, -1, -1, -1, 3, -1),
        (-3, -3, -1, 1, -1, -3),
        (-3, -3, -3, 1, -3, -1),
        (-3, 1, 1, -3, -1, -3),
        (-3, 3, -3, 1, 1, -3),
        (-3, 1, -3, -3, -3, -1),
        (1, 1, -3, 3, 1, 3),
        (1, 1, -3, -3, 1, -3),
        (1, 1, 3, -1, 3, 3),
        (1, 1, -3, 1, 3, 3),
        (1, 1, -1, -1, 3, -1),
        (1, 1, -1, 3, -1, -1),
        (1, 1, -1, 3, -3, -1),
        (1, 1, -3, 1, -1, -1),
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
class NRSRSPortConfiguration:
    antenna_ports: tuple[int, ...]
    ports8_tdm: bool

    def __post_init__(self) -> None:
        if not isinstance(self.antenna_ports, tuple) or not self.antenna_ports:
            raise TypeError("antenna_ports must be a nonempty tuple")
        if any(
            isinstance(port, bool) or not isinstance(port, Integral) for port in self.antenna_ports
        ):
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
            raise ValueError(
                "ports8tdm cannot be combined with cyclic-shift or comb-offset hopping"
            )
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
    if any(isinstance(port, bool) or not isinstance(port, Integral) for port in antenna_ports):
        raise TypeError("antenna_ports must contain integers")
    ports = tuple(int(port) for port in antenna_ports)
    if len(set(ports)) != len(ports) or any(port not in range(1000, 1008) for port in ports):
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
        object.__setattr__(
            self, "selected_bwp_id_by_cell", tuple((cell, int(bwp_id)) for cell, bwp_id in selected)
        )
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
        if any(
            value is not None for value in (absolute_slot_index, periodicity_slots, slot_offset)
        ):
            raise ValueError("aperiodic SRS does not use periodic slot parameters")
        if any(value is not None for value in (numerology, hyper_frame_number, hyper_sfn_index)):
            raise ValueError("aperiodic SRS does not use Hyper-SFN parameters")
        return symbol_counter

    for name, periodic_value in (
        ("absolute_slot_index", absolute_slot_index),
        ("periodicity_slots", periodicity_slots),
        ("slot_offset", slot_offset),
    ):
        if isinstance(periodic_value, bool) or not isinstance(periodic_value, Integral):
            raise TypeError(f"{name} must be an integer for periodic SRS")
    assert absolute_slot_index is not None
    assert periodicity_slots is not None
    assert slot_offset is not None
    absolute_slot_index = int(absolute_slot_index)
    periodicity_slots = int(periodicity_slots)
    slot_offset = int(slot_offset)
    if absolute_slot_index < 0 or periodicity_slots < 1 or not 0 <= slot_offset < periodicity_slots:
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
            raise ValueError("Hyper-SFN index requires a period of at least 2^mu * 10240 slots")
        if isinstance(hyper_frame_number, bool) or not isinstance(hyper_frame_number, Integral):
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
        if isinstance(selected_available_offset_index, bool) or not isinstance(
            selected_available_offset_index, Integral
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
    if isinstance(srs_request_codepoint, bool) or not isinstance(srs_request_codepoint, Integral):
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
            not isinstance(configured_carrier, str) or configured_carrier not in {"non_sul", "sul"}
        ):
            raise ValueError(f"resource_sets[{index}].ul_carrier must identify non_sul or sul")
        if configured_carrier is not None and (
            not isinstance(configured_carrier, str) or configured_carrier not in {"non_sul", "sul"}
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
    if not isinstance(scheduled_cell_ids, Sequence) or isinstance(scheduled_cell_ids, (str, bytes)):
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
            isinstance(bwp_id, bool) or not isinstance(bwp_id, Integral) or not 0 <= bwp_id <= 4
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

    max_noninitial = max(len(bwp_ids) - 1 for bwp_ids in configured_by_cell.values())
    indicator_domain_size = max_noninitial + 1 if max_noninitial <= 3 else max_noninitial
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
                indicated_id = bwp_ids[indicator_value] if indicator_value < len(bwp_ids) else -1
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
    if not isinstance(cell_set_configs, Sequence) or isinstance(cell_set_configs, (str, bytes)):
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
        if not 2 <= len(cells) <= 4 or any(not isinstance(cell, str) or not cell for cell in cells):
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
    if not isinstance(payload_bits, Sequence) or isinstance(payload_bits, (str, bytes)):
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
        raise ValueError(f"DCI 0_3 field-width roster mismatch; missing={missing}, extra={extra}")
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
        raise ValueError("dmrs_port_configuration_by_cell must cover the full scheduled cell list")
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
                width = 2 + max_length_value + int(dmrs_type_value == 2) + int(enhanced)
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
            raise ValueError(f"tx_config_by_cell[{cell!r}] must be 'codebook' or 'nonCodebook'")
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
            raise ValueError(f"tx_config_by_cell[{cell!r}] must be 'codebook' or 'nonCodebook'")
        supports_configured_max = ue_supports_max_mimo_layers_by_cell[cell]
        if type(supports_configured_max) is not bool:
            raise TypeError("ue_supports_max_mimo_layers_by_cell values must be booleans")
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
            raise TypeError(f"maximum non-codebook PUSCH layers for {cell!r} must be an integer")
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
            isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 4
            for value in raw_bwp_ids
        ):
            raise ValueError(f"configured BWP IDs for {cell!r} must be integers in [0, 4]")
        bwp_ids = tuple(int(value) for value in raw_bwp_ids)
        if len(set(bwp_ids)) != len(bwp_ids) or not bwp_ids:
            raise ValueError(f"configured BWP IDs for {cell!r} must be nonempty and unique")
        raw_configs = bwp_frequency_config_by_cell[cell]
        if (
            not isinstance(raw_configs, Mapping)
            or any(isinstance(key, bool) or not isinstance(key, Integral) for key in raw_configs)
            or set(raw_configs) != set(bwp_ids)
        ):
            raise ValueError(f"BWP frequency configurations for {cell!r} must cover every BWP")
        active_id = active_bwp_id_by_cell[cell]
        if isinstance(active_id, bool) or not isinstance(active_id, Integral):
            raise TypeError(f"active BWP ID for {cell!r} must be an integer")
        if int(active_id) not in raw_configs:
            raise ValueError(f"active BWP {active_id!r} for {cell!r} is not configured")
        config = raw_configs[int(active_id)]
        if not isinstance(config, Mapping):
            raise TypeError(f"active BWP frequency configuration for {cell!r} must be a mapping")

        size_raw = config.get("size_prb")
        start_raw = config.get("start_prb")
        for label, value in (("size_prb", size_raw), ("start_prb", start_raw)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{label} for {cell!r} must be an integer")
        size_prb = int(cast(Integral, size_raw))
        start_prb = int(cast(Integral, start_raw))
        if not 1 <= size_prb <= 275 or not 0 <= start_prb <= 274 or start_prb + size_prb > 275:
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
                raise TypeError(f"frequency_hopping_offset_count for {cell!r} must be an integer")
            offset_count = int(offset_count)
            if offset_count not in {2, 4}:
                raise ValueError("frequency hopping offset list must contain 2 or 4 values")
            hopping_width = (offset_count - 1).bit_length()
        else:
            hopping_width = 0

        rbg_config_raw = config.get("rbg_size_config", 1)
        if isinstance(rbg_config_raw, str) and rbg_config_raw.startswith("config"):
            rbg_config_raw = rbg_config_raw.removeprefix("config")
        if isinstance(rbg_config_raw, bool) or not isinstance(rbg_config_raw, (Integral, str)):
            raise TypeError(f"rbg_size_config for {cell!r} must be config1, config2, or config3")
        try:
            rbg_config = int(rbg_config_raw)
        except ValueError as exc:
            raise ValueError("rbg_size_config must be config1, config2, or config3") from exc
        if rbg_config not in rbg_sizes:
            raise ValueError("rbg_size_config must be config1, config2, or config3")
        type0_p = next(
            limit_size for limit, limit_size in rbg_sizes[rbg_config] if size_prb <= limit
        )
        type0_count = (size_prb + start_prb % type0_p + type0_p - 1) // type0_p
        type0_width = type0_count

        granularity_raw = config.get("type1_granularity_prb", 1)
        if isinstance(granularity_raw, bool) or not isinstance(granularity_raw, Integral):
            raise TypeError(f"type1_granularity_prb for {cell!r} must be an integer")
        granularity = int(granularity_raw)
        if granularity not in {1, 2, 4, 8}:
            raise ValueError("type1_granularity_prb must be one of 1, 2, 4, or 8")
        type1_count = (size_prb + start_prb % granularity + granularity - 1) // granularity
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
    if not isinstance(dormancy_group_ids, Sequence) or isinstance(dormancy_group_ids, (str, bytes)):
        raise TypeError("dormancy_group_ids must be an ordered sequence or None")
    if any(
        isinstance(group_id, bool) or not isinstance(group_id, Integral) or int(group_id) < 0
        for group_id in dormancy_group_ids
    ):
        raise ValueError("dormancy_group_ids must contain nonnegative integers")
    groups = tuple(int(group_id) for group_id in dormancy_group_ids)
    if len(set(groups)) != len(groups):
        raise ValueError("dormancy_group_ids must be unique")
    if not dormancy_configured:
        return 0
    if not carried_on_primary_cell_within_drx_active_time or not scell_has_multiple_dl_bwps:
        return 0
    if not 1 <= len(groups) <= 5:
        raise ValueError("applicable dormancy_group_ids must contain 1..5 groups")
    return len(groups)


def _normalized_multiport_beta(beta_srs: float, port_count: int, four_port_srs_3tx: bool) -> float:
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
            x2[bit_index + 3] ^ x2[bit_index + 2] ^ x2[bit_index + 1] ^ x2[bit_index]
        )
    return x1[1600 : 1600 + count] ^ x2[1600 : 1600 + count]


def _srs_gold_bit_offset(
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    slot_symbol_count: int,
) -> int:
    return 8 * (slot_symbol_count * slot_index + start_symbol_index + resource_symbol_index)


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
    group_hopping_value = sum(int(hopping_bits[first_bit + bit]) << bit for bit in range(8))
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
        raise ValueError("cyclic_shift_hopping_offset is outside the configured shift range")
    indices = np.arange(sequence_length, dtype=np.float64)
    if sequence_length in _LOW_PAPR_SHORT_PHASES:
        phases = np.asarray(_LOW_PAPR_SHORT_PHASES[sequence_length][group_index])
        base = np.exp(1j * np.pi * phases / 4.0)
    elif sequence_length == 30:
        base = np.exp(-1j * np.pi * group_index * (indices + 1.0) * (indices + 2.0) / 31.0)
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
                raise ValueError("active_port_symbols must be boolean [ports, resource symbols]")
            active = tuple(tuple(bool(value) for value in row) for row in active)
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for port_index, port_activity in enumerate(active):
            for symbol, row, is_active in zip(symbols, subcarriers, port_activity, strict=True):
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
                raise ValueError("active_port_symbols must be boolean [ports, resource symbols]")
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
    if start_position_offset < srs_symbol_count - 1 or start_position_offset >= slot_symbol_count:
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
        raise ValueError("sequence must have shape [occupied REs] or [SRS symbols, occupied REs]")
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
        values if values.ndim == 2 else np.broadcast_to(values, (srs_symbol_count, occupied_count))
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
            raise ValueError(
                "channel estimate must be complex [rx antennas, TX ports, subcarriers]"
            )
        rx_count, port_count, subcarrier_count = channel.shape
        if min(rx_count, port_count, subcarrier_count) < 1:
            raise ValueError("channel estimate dimensions must be nonempty")
        if covariance.shape != (rx_count, subcarrier_count, port_count, port_count):
            raise ValueError(
                "noise covariance shape must be [rx antennas, subcarriers, ports, ports]"
            )
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
        if (
            not symbols
            or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in symbols
            )
            or len(set(symbols)) != len(symbols)
        ):
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
    covariance = np.zeros((rx_count, subcarrier_count, port_count, port_count), dtype=np.complex128)
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
            if not np.isfinite(coefficients).all() or not np.isfinite(coefficient_covariance).all():
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
