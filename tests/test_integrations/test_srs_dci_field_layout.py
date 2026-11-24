"""DCI 0_3 field layout and width derivation for SRS requests."""

import numpy as np
import pytest
from integrations.sionna.srs import (
    map_nr_srs_multiport_resource,
    nr_dci_0_3_field_values_from_layout,
    nr_srs_dci_0_3_field_widths_from_configuration,
    nr_srs_dci_0_3_frequency_assignment_widths,
    nr_srs_dci_0_3_request_list_index_from_layout,
)
from integrations.sionna.srs.srs_constants.dci_0_3 import (
    _DCI_0_3_FIELD_ORDER,
    _nr_srs_dci_0_3_antenna_ports_width,
    _nr_srs_dci_0_3_precoding_width,
    _nr_srs_dci_0_3_sri_width,
)


def _dmrs(mapping_type="A", dmrs_type=1, enhanced=False, max_length=1):
    return {
        "mapping_type": mapping_type,
        "dmrs_type": dmrs_type,
        "dmrs_type_enhanced": enhanced,
        "max_length": max_length,
    }


def _antenna_width(*, mode="type1a", cells=("cell-a",), transform=None, configs=None):
    return _nr_srs_dci_0_3_antenna_ports_width(
        mode=mode,
        cell_ids=cells,
        transform_precoding_enabled_by_cell=(
            {cell: False for cell in cells} if transform is None else transform
        ),
        dmrs_port_configuration_by_cell=(
            {cell: (_dmrs(),) for cell in cells} if configs is None else configs
        ),
        full_cell_roster=cells,
    )


def test_antenna_port_width_uses_dmrs_mapping_aggregation_and_transform_precoding() -> None:
    configs = {
        "cell-a": (_dmrs("A", 1, max_length=1), _dmrs("B", 2, True, 2)),
        "cell-b": (_dmrs("A", 1, max_length=2),),
    }
    assert _antenna_width(cells=("cell-a", "cell-b"), configs=configs) == 6
    assert _antenna_width(mode="type2", cells=("cell-a", "cell-b"), configs=configs) == 10
    assert (
        _antenna_width(transform={"cell-a": True}, configs={"cell-a": (_dmrs(max_length=2),)}) == 4
    )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"mode": "type3"}, ValueError, "antenna_ports_mode"),
        ({"transform": {}}, ValueError, "must cover"),
        ({"configs": {"cell-a": ({"mapping_type": "A"},)}}, ValueError, "field roster"),
        ({"configs": {"cell-a": (_dmrs("C"),)}}, ValueError, "mapping_type"),
        ({"configs": {"cell-a": (_dmrs(), _dmrs("A", 2))}}, ValueError, "must be unique"),
        ({"configs": {"cell-a": (_dmrs(dmrs_type=True),)}}, TypeError, "dmrs_type"),
        ({"configs": {"cell-a": (_dmrs(enhanced=1),)}}, TypeError, "enhanced"),
        ({"configs": {"cell-a": (_dmrs(max_length=3),)}}, ValueError, "max_length"),
        (
            {"transform": {"cell-a": True}, "configs": {"cell-a": (_dmrs(dmrs_type=2),)}},
            ValueError,
            "requires DMRS type 1",
        ),
    ],
)
def test_antenna_port_width_rejects_invalid_dmrs_configuration(overrides, error, message) -> None:
    arguments = {"cells": ("cell-a",)}
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        _antenna_width(**arguments)


def _precoding_width(
    *,
    ports=(2,),
    subset="fullyAndPartialAndNonCoherent",
    rank=1,
    full_power=None,
    transform=False,
    mode="type1a",
):
    cells = ("cell-a",)
    return _nr_srs_dci_0_3_precoding_width(
        mode=mode,
        cell_ids=cells,
        tx_config_by_cell={"cell-a": "codebook"},
        transform_precoding_enabled_by_cell={"cell-a": transform},
        codebook_srs_antenna_port_counts_by_cell={"cell-a": ports},
        codebook_subset_by_cell={"cell-a": subset},
        max_rank_for_codebook_by_cell={"cell-a": rank},
        ul_full_power_transmission_by_cell={"cell-a": full_power},
        full_cell_roster=cells,
    )


@pytest.mark.parametrize(
    ("ports", "subset", "rank", "full_power", "transform", "expected"),
    [
        ((1,), None, None, None, False, 0),
        ((2,), "fullyAndPartialAndNonCoherent", 1, None, False, 3),
        ((2,), "fullyAndPartialAndNonCoherent", 2, None, False, 4),
        ((2,), "nonCoherent", 2, None, False, 2),
        ((2,), "nonCoherent", 1, "fullpowerMode1", False, 2),
        ((2,), "nonCoherent", 2, "fullpowerMode1", False, 2),
        ((4,), "fullyAndPartialAndNonCoherent", 1, None, False, 5),
        ((4,), "partialAndNonCoherent", 2, None, False, 5),
        ((4,), "nonCoherent", 3, None, False, 4),
        ((4,), "partialAndNonCoherent", 2, "fullpowerMode1", False, 5),
        ((4,), "nonCoherent", 3, "fullpowerMode1", False, 4),
        ((4,), "nonCoherent", 1, "fullpowerMode1", True, 3),
    ],
)
def test_precoding_width_follows_srs_ports_rank_subset_and_full_power(
    ports, subset, rank, full_power, transform, expected
) -> None:
    assert (
        _precoding_width(
            ports=ports,
            subset=subset,
            rank=rank,
            full_power=full_power,
            transform=transform,
        )
        == expected
    )


def test_precoding_width_aggregates_per_cell_and_allows_noncodebook_cells() -> None:
    common = {
        "tx_config_by_cell": {"cell-a": "codebook", "cell-b": "codebook"},
        "transform_precoding_enabled_by_cell": {"cell-a": False, "cell-b": False},
        "codebook_srs_antenna_port_counts_by_cell": {"cell-a": (2,), "cell-b": (2,)},
        "codebook_subset_by_cell": {
            "cell-a": "fullyAndPartialAndNonCoherent",
            "cell-b": "nonCoherent",
        },
        "max_rank_for_codebook_by_cell": {"cell-a": 1, "cell-b": 2},
        "ul_full_power_transmission_by_cell": {"cell-a": None, "cell-b": None},
        "full_cell_roster": ("cell-a", "cell-b"),
        "cell_ids": ("cell-a", "cell-b"),
    }
    assert _nr_srs_dci_0_3_precoding_width(mode="type1a", **common) == 3
    assert _nr_srs_dci_0_3_precoding_width(mode="type2", **common) == 5
    noncodebook = {
        **common,
        "tx_config_by_cell": {"cell-a": "nonCodebook", "cell-b": "codebook"},
        "codebook_srs_antenna_port_counts_by_cell": {"cell-a": (), "cell-b": (2,)},
        "codebook_subset_by_cell": {"cell-a": None, "cell-b": "nonCoherent"},
        "max_rank_for_codebook_by_cell": {"cell-a": None, "cell-b": 2},
        "ul_full_power_transmission_by_cell": {"cell-a": None, "cell-b": None},
    }
    assert _nr_srs_dci_0_3_precoding_width(mode="type2", **noncodebook) == 2


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"mode": "type3"}, ValueError, "tpmi_mode"),
        ({"tx_config_by_cell": {}}, ValueError, "must cover"),
        ({"tx_config_by_cell": {"cell-a": "unsupported"}}, ValueError, "must be 'codebook'"),
        ({"transform_precoding_enabled_by_cell": {"cell-a": 1}}, TypeError, "must be booleans"),
        (
            {"codebook_srs_antenna_port_counts_by_cell": {"cell-a": ()}},
            ValueError,
            "must configure",
        ),
        (
            {"codebook_srs_antenna_port_counts_by_cell": {"cell-a": (True,)}},
            TypeError,
            "must be integers",
        ),
        (
            {"codebook_srs_antenna_port_counts_by_cell": {"cell-a": (3,)}},
            ValueError,
            "must be 1, 2, or 4",
        ),
        ({"codebook_subset_by_cell": {"cell-a": "partialAndNonCoherent"}}, ValueError, "subset"),
        ({"max_rank_for_codebook_by_cell": {"cell-a": True}}, TypeError, "must be an integer"),
        ({"max_rank_for_codebook_by_cell": {"cell-a": 3}}, ValueError, "must be in"),
        ({"ul_full_power_transmission_by_cell": {"cell-a": "invalid"}}, ValueError, "is invalid"),
    ],
)
def test_precoding_width_rejects_invalid_srs_codebook_configuration(
    overrides, error, message
) -> None:
    arguments = {}
    for key, value in {
        "mode": "type1a",
        "cell_ids": ("cell-a",),
        "tx_config_by_cell": {"cell-a": "codebook"},
        "transform_precoding_enabled_by_cell": {"cell-a": False},
        "codebook_srs_antenna_port_counts_by_cell": {"cell-a": (2,)},
        "codebook_subset_by_cell": {"cell-a": "nonCoherent"},
        "max_rank_for_codebook_by_cell": {"cell-a": 1},
        "ul_full_power_transmission_by_cell": {"cell-a": None},
        "full_cell_roster": ("cell-a",),
    }.items():
        arguments[key] = value
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        _nr_srs_dci_0_3_precoding_width(**arguments)


def _sri_width(
    *,
    mode="type1a",
    tx_config="nonCodebook",
    resource_count=4,
    max_layers=2,
    supports_configured_max=True,
    ue_max_layers=None,
):
    cells = ("cell-a",)
    return _nr_srs_dci_0_3_sri_width(
        mode=mode,
        cell_ids=cells,
        tx_config_by_cell={"cell-a": tx_config},
        srs_resource_count_by_cell={"cell-a": resource_count},
        max_noncodebook_layers_by_cell={"cell-a": max_layers},
        ue_supports_max_mimo_layers_by_cell={"cell-a": supports_configured_max},
        ue_max_noncodebook_layers_by_cell=(
            None if ue_max_layers is None else {"cell-a": ue_max_layers}
        ),
        full_cell_roster=cells,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"tx_config": "codebook", "resource_count": 1}, 0),
        ({"tx_config": "codebook", "resource_count": 4}, 2),
        ({"resource_count": 4, "max_layers": 2}, 4),
        ({"resource_count": 2, "max_layers": 8}, 2),
        ({"max_layers": None, "supports_configured_max": False, "ue_max_layers": 1}, 2),
        ({"max_layers": 3, "supports_configured_max": False, "ue_max_layers": 2}, 4),
    ],
)
def test_sri_width_counts_configured_codebook_and_noncodebook_selections(kwargs, expected) -> None:
    assert _sri_width(**kwargs) == expected


def test_sri_width_uses_max_or_sum_for_the_selected_cells() -> None:
    common = {
        "tx_config_by_cell": {"cell-a": "codebook", "cell-b": "nonCodebook"},
        "srs_resource_count_by_cell": {"cell-a": 4, "cell-b": 4},
        "max_noncodebook_layers_by_cell": {"cell-a": None, "cell-b": 2},
        "ue_supports_max_mimo_layers_by_cell": {"cell-a": True, "cell-b": True},
        "ue_max_noncodebook_layers_by_cell": None,
        "full_cell_roster": ("cell-a", "cell-b"),
        "cell_ids": ("cell-a", "cell-b"),
    }
    assert _nr_srs_dci_0_3_sri_width(mode="type1a", **common) == 4
    assert _nr_srs_dci_0_3_sri_width(mode="type2", **common) == 6


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"mode": "type3"}, ValueError, "sri_mode"),
        ({"tx_config_by_cell": {}}, ValueError, "must cover"),
        ({"tx_config_by_cell": {"cell-a": "unsupported"}}, ValueError, "must be 'codebook'"),
        ({"ue_supports_max_mimo_layers_by_cell": {"cell-a": 1}}, TypeError, "must be booleans"),
        ({"srs_resource_count_by_cell": {"cell-a": True}}, TypeError, "must be an integer"),
        ({"srs_resource_count_by_cell": {"cell-a": 65}}, ValueError, "in \\[1, 64\\]"),
        (
            {"max_noncodebook_layers_by_cell": {"cell-a": None}},
            ValueError,
            "capability is required",
        ),
        ({"max_noncodebook_layers_by_cell": {"cell-a": True}}, TypeError, "must be an integer"),
        ({"max_noncodebook_layers_by_cell": {"cell-a": 9}}, ValueError, "in \\[1, 8\\]"),
        (
            {
                "supports_configured_max": False,
                "max_noncodebook_layers_by_cell": {"cell-a": None},
                "ue_max_noncodebook_layers_by_cell": {},
            },
            ValueError,
            "must cover",
        ),
    ],
)
def test_sri_width_rejects_invalid_rrc_or_ue_layer_configuration(overrides, error, message) -> None:
    arguments = {
        "mode": "type1a",
        "cell_ids": ("cell-a",),
        "tx_config_by_cell": {"cell-a": "nonCodebook"},
        "srs_resource_count_by_cell": {"cell-a": 4},
        "max_noncodebook_layers_by_cell": {"cell-a": 2},
        "ue_supports_max_mimo_layers_by_cell": {"cell-a": True},
        "ue_max_noncodebook_layers_by_cell": None,
        "full_cell_roster": ("cell-a",),
    }
    if "supports_configured_max" in overrides:
        arguments["ue_supports_max_mimo_layers_by_cell"] = {
            "cell-a": overrides.pop("supports_configured_max")
        }
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        _nr_srs_dci_0_3_sri_width(**arguments)


def test_dci_0_3_decoder_rejects_malformed_bit_rosters_and_resource_overflow() -> None:
    zero_widths = dict.fromkeys(_DCI_0_3_FIELD_ORDER, 0)
    invalid_cases = (
        ("not-bits", zero_widths, TypeError, "sequence of decoded bits"),
        ((), None, TypeError, "field_widths_bits must be a mapping"),
        ((), {**zero_widths, 7: 0}, TypeError, "field names must be strings"),
        ((), {**zero_widths, "not_a_field": 0}, ValueError, "roster mismatch"),
        ((), {**zero_widths, "srs_request": True}, TypeError, "must be an integer"),
        ((), {**zero_widths, "srs_request": -1}, ValueError, "must be in"),
        ((), {**zero_widths, "srs_request": 513}, ValueError, "must be in"),
        (
            (),
            {**zero_widths, **dict.fromkeys(_DCI_0_3_FIELD_ORDER[:9], 512)},
            ValueError,
            "exceeds the 4096-bit resource bound",
        ),
        ((1,), {**zero_widths, "srs_request": 2}, ValueError, "shorter than the declared"),
        ((1,), zero_widths, ValueError, "nonzero trailing bits"),
        ((2,), {**zero_widths, "srs_request": 1}, ValueError, r"payload_bits\[0\]"),
    )

    for payload, widths, error, message in invalid_cases:
        with pytest.raises(error, match=message):
            nr_dci_0_3_field_values_from_layout(payload, field_widths_bits=widths)


def test_dci_0_3_layout_parser_decodes_complete_field_roster_msb_first() -> None:
    field_names = (
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
    widths = dict.fromkeys(field_names, 0)
    widths.update(
        identifier_for_dci_formats=2,
        scheduled_cells_indicator=2,
        modulation_and_coding_scheme=3,
        srs_request=2,
        ul_sch_indicator=1,
    )
    field_values = {
        "identifier_for_dci_formats": 2,
        "scheduled_cells_indicator": 1,
        "modulation_and_coding_scheme": 5,
        "srs_request": 3,
        "ul_sch_indicator": 1,
    }
    bits = [
        (field_values.get(name, 0) >> shift) & 1
        for name in field_names
        for shift in range(widths[name] - 1, -1, -1)
    ]

    decoded = nr_dci_0_3_field_values_from_layout(bits + [False, False], field_widths_bits=widths)

    assert decoded["identifier_for_dci_formats"] == 2
    assert decoded["scheduled_cells_indicator"] == 1
    assert decoded["modulation_and_coding_scheme"] == 5
    assert decoded["srs_request"] == 3
    assert decoded["ul_sch_indicator"] == 1
    assert decoded["bandwidth_part_indicator"] is None


@pytest.mark.parametrize(
    ("bits", "widths", "message"),
    [
        ([], None, "field_widths_bits must be a mapping"),
        ([0], {"wrong": 1}, "field-width roster mismatch"),
    ],
)
def test_dci_0_3_layout_parser_rejects_malformed_contracts(
    bits: list[int], widths: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        nr_dci_0_3_field_values_from_layout(bits, field_widths_bits=widths)  # type: ignore[arg-type]


def test_dci_0_3_layout_parser_enforces_width_and_payload_bounds() -> None:
    field_names = (
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
    widths = dict.fromkeys(field_names, 0)
    widths["identifier_for_dci_formats"] = 1

    with pytest.raises(ValueError, match="shorter than the declared"):
        nr_dci_0_3_field_values_from_layout([], field_widths_bits=widths)
    with pytest.raises(ValueError, match="nonzero trailing bits"):
        nr_dci_0_3_field_values_from_layout([0, 1], field_widths_bits=widths)
    with pytest.raises(ValueError, match=r"payload_bits\[0\]"):
        nr_dci_0_3_field_values_from_layout([2], field_widths_bits=widths)

    widths["identifier_for_dci_formats"] = 4097
    with pytest.raises(ValueError, match=r"width for identifier_for_dci_formats.*\[0, 512\]"):
        nr_dci_0_3_field_values_from_layout([], field_widths_bits=widths)


def test_dci_0_3_frequency_assignment_widths_follow_active_bwp_configuration() -> None:
    cells = ("type0", "type1", "switch", "hopping", "interlace")
    configured = {cell: (0,) for cell in cells}
    active = {cell: 0 for cell in cells}
    configs = {
        "type0": {
            0: {
                "size_prb": 37,
                "start_prb": 1,
                "resource_allocation": "type0",
                "rbg_size_config": "config1",
            }
        },
        "type1": {
            0: {
                "size_prb": 20,
                "start_prb": 1,
                "resource_allocation": "type1",
                "type1_granularity_prb": 4,
            }
        },
        "switch": {
            0: {
                "size_prb": 10,
                "start_prb": 0,
                "resource_allocation": "dynamicSwitch",
                "rbg_size_config": 1,
                "type1_granularity_prb": 2,
            }
        },
        "hopping": {
            0: {
                "size_prb": 20,
                "start_prb": 0,
                "resource_allocation": "type1",
                "type1_granularity_prb": 4,
                "frequency_hopping": True,
                "frequency_hopping_offset_count": 4,
            }
        },
        "interlace": {
            0: {
                "size_prb": 50,
                "start_prb": 0,
                "use_interlace": True,
                "subcarrier_spacing_khz": 30,
                "rb_set_count": 4,
            }
        },
    }

    widths = nr_srs_dci_0_3_frequency_assignment_widths(cells, configured, active, configs)

    assert widths == {
        "type0": 10,
        "type1": 5,
        "switch": 6,
        "hopping": 6,
        "interlace": 7,
    }


@pytest.mark.parametrize(
    ("configuration", "error"),
    [
        ({"size_prb": 10, "start_prb": 266, "resource_allocation": "type0"}, "extent"),
        ({"size_prb": 10, "start_prb": 0, "resource_allocation": "unsupported"}, "mode"),
    ],
)
def test_dci_0_3_frequency_assignment_rejects_invalid_active_bwp_configuration(
    configuration, error
) -> None:
    with pytest.raises(ValueError, match=error):
        nr_srs_dci_0_3_frequency_assignment_widths(
            ("cell",),
            {"cell": (0,)},
            {"cell": 0},
            {"cell": {0: configuration}},
        )


def _frequency_width(configuration):
    return nr_srs_dci_0_3_frequency_assignment_widths(
        ("cell",), {"cell": (0,)}, {"cell": 0}, {"cell": {0: configuration}}
    )["cell"]


@pytest.mark.parametrize(
    ("configuration", "expected"),
    [
        ({"size_prb": 20, "start_prb": 1, "resource_allocation": "type1"}, 8),
        (
            {
                "size_prb": 20,
                "start_prb": 1,
                "resource_allocation": "type1",
                "frequency_hopping": True,
                "frequency_hopping_offset_count": 2,
            },
            9,
        ),
        (
            {
                "size_prb": 20,
                "start_prb": 1,
                "resource_allocation": "type1",
                "frequency_hopping": True,
                "frequency_hopping_offset_count": 4,
            },
            10,
        ),
        (
            {
                "size_prb": 20,
                "start_prb": 1,
                "resource_allocation": "dynamicSwitch",
                "rbg_size_config": "config2",
                "type1_granularity_prb": 4,
            },
            7,
        ),
        (
            {
                "size_prb": 50,
                "start_prb": 0,
                "use_interlace": True,
                "subcarrier_spacing_khz": 15,
                "rb_set_count": 1,
            },
            6,
        ),
        (
            {
                "size_prb": 50,
                "start_prb": 0,
                "use_interlace": True,
                "subcarrier_spacing_khz": 30,
                "rb_set_count": 4,
            },
            7,
        ),
    ],
)
def test_frequency_assignment_width_matches_allocation_mode(configuration, expected) -> None:
    assert _frequency_width(configuration) == expected


@pytest.mark.parametrize(
    ("configuration", "error", "message"),
    [
        ({"size_prb": True, "start_prb": 0}, TypeError, "size_prb"),
        ({"size_prb": 1, "start_prb": False}, TypeError, "start_prb"),
        (
            {"size_prb": 1, "start_prb": 0, "resource_allocation": "type1", "frequency_hopping": 1},
            TypeError,
            "frequency_hopping.*boolean",
        ),
        (
            {
                "size_prb": 1,
                "start_prb": 0,
                "resource_allocation": "type1",
                "frequency_hopping": True,
                "frequency_hopping_offset_count": 3,
            },
            ValueError,
            "offset list must contain 2 or 4",
        ),
        (
            {
                "size_prb": 1,
                "start_prb": 0,
                "resource_allocation": "type1",
                "rbg_size_config": "bad",
            },
            ValueError,
            "rbg_size_config must be config1",
        ),
        (
            {
                "size_prb": 1,
                "start_prb": 0,
                "resource_allocation": "type1",
                "type1_granularity_prb": 3,
            },
            ValueError,
            "type1_granularity_prb must be one of",
        ),
        (
            {
                "size_prb": 1,
                "start_prb": 0,
                "use_interlace": True,
                "subcarrier_spacing_khz": 60,
                "rb_set_count": 1,
            },
            ValueError,
            "supports 15 or 30 kHz",
        ),
        (
            {
                "size_prb": 1,
                "start_prb": 0,
                "use_interlace": True,
                "subcarrier_spacing_khz": 30,
                "rb_set_count": 276,
            },
            ValueError,
            "rb_set_count.*1, 275",
        ),
    ],
)
def test_frequency_assignment_width_rejects_invalid_active_bwp_fields(
    configuration, error, message
) -> None:
    with pytest.raises(error, match=message):
        _frequency_width(configuration)


def test_dci_0_3_known_widths_derive_from_decoded_cell_bwp_and_srs_configuration() -> None:
    field_names = (
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
    derived_names = {
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
    assert set(derived_names) == set(field_names)
    cell_set_configs = (
        {
            "set_id": 0,
            "scheduled_cell_ids": ("cell-1", "cell-2"),
            "scheduled_cell_combos": ((0, 1),),
        },
        {
            "set_id": 2,
            "scheduled_cell_ids": ("cell-3", "cell-4"),
            "scheduled_cell_combos": ((0,), (1,), (0, 1)),
        },
    )

    resolver_inputs = dict(
        cell_set_configs=cell_set_configs,
        selected_cell_set_id=2,
        configured_ul_bwp_ids_by_cell={"cell-3": (0, 1, 2), "cell-4": (0, 1, 2)},
        active_ul_bwp_id_by_cell={"cell-3": 0, "cell-4": 0},
        ul_bwp_frequency_allocation_by_cell={
            "cell-3": {
                0: {"size_prb": 10, "start_prb": 0, "resource_allocation": "type0"},
                1: {"size_prb": 25, "start_prb": 0, "resource_allocation": "type1"},
                2: {"size_prb": 50, "start_prb": 0, "resource_allocation": "dynamicSwitch"},
            },
            "cell-4": {
                0: {"size_prb": 10, "start_prb": 0, "resource_allocation": "type0"},
                1: {"size_prb": 25, "start_prb": 0, "resource_allocation": "type1"},
                2: {"size_prb": 50, "start_prb": 0, "resource_allocation": "dynamicSwitch"},
            },
        },
        request_list_entry_count=3,
        srs_offset_list=((0, 0), (1, 0), (2, 1), (3, 2), (4, 3)),
        report_trigger_size=3,
        tdra_field_index_list_entry_count=5,
        frequency_hopping_enabled_by_cell={"cell-3": True, "cell-4": False},
        transform_precoding_enabled_by_cell={"cell-3": True, "cell-4": False},
        beta_offsets_semi_static_by_cell={"cell-3": True, "cell-4": False},
        channel_access_config_entry_count_by_cell={"cell-3": 4, "cell-4": 0},
        harq_process_number_size_bits_by_cell={"cell-3": 3, "cell-4": 5},
        max_schedulable_pusch_count_by_cell={"cell-3": 2, "cell-4": 3},
        redundancy_version_bits_by_cell={"cell-3": 2, "cell-4": 1},
        sri_mode="type2",
        tpmi_mode="type2",
        antenna_ports_mode="type2",
        tx_config_by_cell={"cell-3": "nonCodebook", "cell-4": "codebook"},
        srs_resource_count_by_cell={"cell-3": 2, "cell-4": 4},
        max_noncodebook_layers_by_cell={"cell-3": 2, "cell-4": None},
        ue_supports_max_mimo_layers_by_cell={"cell-3": True, "cell-4": True},
        dmrs_port_configuration_by_cell={
            "cell-3": (
                {"mapping_type": "A", "dmrs_type": 1, "dmrs_type_enhanced": False, "max_length": 1},
            ),
            "cell-4": (
                {"mapping_type": "B", "dmrs_type": 2, "dmrs_type_enhanced": True, "max_length": 2},
                {"mapping_type": "A", "dmrs_type": 1, "dmrs_type_enhanced": False, "max_length": 1},
            ),
        },
        codebook_srs_antenna_port_counts_by_cell={"cell-3": (), "cell-4": (4,)},
        codebook_subset_by_cell={"cell-3": None, "cell-4": "nonCoherent"},
        max_rank_for_codebook_by_cell={"cell-3": None, "cell-4": 3},
        ul_full_power_transmission_by_cell={"cell-3": None, "cell-4": None},
        ptrs_uplink_configured_by_cell={"cell-3": True, "cell-4": True},
        max_pusch_rank_by_cell={"cell-3": 4, "cell-4": 3},
        p0_pusch_set_list_configured_by_cell={"cell-3": True, "cell-4": True},
        sri_field_present_by_cell={"cell-3": True, "cell-4": False},
        olpc_parameter_set_dci_0_1_bits_by_cell={"cell-3": 2, "cell-4": 2},
        harq_ack_codebook_types=("semi_static", "dynamic"),
        dynamic_harq_ack_two_sub_codebooks_by_codebook=(False, True),
        priority_indicator_configured=True,
        minimum_scheduling_offset_configured=False,
    )
    widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs, scheduled_cells_indicator_value=2
    )
    assert widths["scell_dormancy_indication"] == 0
    assert widths["ptrs_dmrs_association"] == 2
    assert widths["open_loop_power_control_parameter_set_indication"] == 2
    no_p0_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "p0_pusch_set_list_configured_by_cell": {
                "cell-3": False,
                "cell-4": False,
            },
        },
        scheduled_cells_indicator_value=2,
    )
    assert no_p0_widths["open_loop_power_control_parameter_set_indication"] == 0
    with pytest.raises(TypeError, match="all open-loop power-control configuration maps"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                key: value
                for key, value in resolver_inputs.items()
                if key != "olpc_parameter_set_dci_0_1_bits_by_cell"
            },
            scheduled_cells_indicator_value=2,
        )
    with pytest.raises(ValueError, match="one or two bits"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "olpc_parameter_set_dci_0_1_bits_by_cell": {
                    "cell-3": 3,
                    "cell-4": 2,
                },
            },
            scheduled_cells_indicator_value=2,
        )
    assert widths["pdcch_monitoring_adaptation_indication"] == 0
    assert widths["measurement_gap_cancellation"] == 0
    assert (
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            measurement_gap_cancellation_configured=True,
        )["measurement_gap_cancellation"]
        == 1
    )
    with pytest.raises(TypeError, match="must be a boolean"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            measurement_gap_cancellation_configured=1,
        )
    assert (
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
            pdcch_skipping_duration_count=1,
        )["pdcch_monitoring_adaptation_indication"]
        == 1
    )
    assert (
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
            pdcch_skipping_duration_count=3,
        )["pdcch_monitoring_adaptation_indication"]
        == 2
    )
    assert (
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
            search_space_group_ids=(0, 1),
        )["pdcch_monitoring_adaptation_indication"]
        == 1
    )
    assert (
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
            search_space_group_ids=(0, 1, 2),
        )["pdcch_monitoring_adaptation_indication"]
        == 2
    )
    assert (
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
            pdcch_skipping_duration_count=1,
            search_space_group_ids=(0, 1),
        )["pdcch_monitoring_adaptation_indication"]
        == 2
    )
    with pytest.raises(ValueError, match="requires a skipping-duration list"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
            search_space_group_ids=(0, 1, 1),
        )
    for group_ids in ((4,), (2, 4), (1, 3, 5), (0, 2, 4, 6)):
        assert nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            dormancy_configured=True,
            dormancy_group_ids=group_ids,
            carried_on_primary_cell_within_drx_active_time=True,
            scell_has_multiple_dl_bwps=True,
        )["scell_dormancy_indication"] == len(group_ids)
    assert (
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            dormancy_configured=True,
            dormancy_group_ids=(0, 1),
            carried_on_primary_cell_within_drx_active_time=False,
            scell_has_multiple_dl_bwps=True,
        )["scell_dormancy_indication"]
        == 0
    )
    with pytest.raises(ValueError, match="requires configured dormancy group IDs"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            dormancy_configured=True,
            carried_on_primary_cell_within_drx_active_time=True,
            scell_has_multiple_dl_bwps=True,
        )

    assert widths["identifier_for_dci_formats"] == 1
    assert widths["scheduled_cell_set_indicator"] == 1
    assert widths["scheduled_cells_indicator"] == 2
    assert widths["bandwidth_part_indicator"] == 2
    assert widths["frequency_domain_resource_assignment"] == 10
    assert widths["time_domain_resource_assignment"] == 3
    assert widths["frequency_hopping_flag"] == 1
    assert widths["dmrs_sequence_initialization"] == 1
    assert widths["ul_sch_indicator"] == 1
    assert widths["priority_indicator"] == 1
    assert widths["minimum_applicable_scheduling_offset_indicator"] == 0
    assert widths["beta_offset_indicator"] == 2
    assert widths["channel_access_cpext_capc"] == 2
    assert widths["tpc_command_for_scheduled_pusch"] == 4
    assert widths["harq_process_number"] == 8
    assert widths["modulation_and_coding_scheme"] == 10
    assert widths["new_data_indicator"] == 5
    assert widths["redundancy_version"] == 7
    assert widths["srs_resource_indicator"] == 4
    assert widths["precoding_information_and_number_of_layers"] == 4
    assert widths["antenna_ports"] == 8
    type1a_sri_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "scheduled_cells_indicator_value": 2,
            "sri_mode": "type1a",
        }
    )
    assert type1a_sri_widths["srs_resource_indicator"] == 2
    type1a_antenna_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "scheduled_cells_indicator_value": 2,
            "antenna_ports_mode": "type1a",
        }
    )
    assert type1a_antenna_widths["antenna_ports"] == 6
    type1a_precoding_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "scheduled_cells_indicator_value": 2,
            "tpmi_mode": "type1a",
        }
    )
    assert type1a_precoding_widths["precoding_information_and_number_of_layers"] == 4
    full_power_mode1_inputs = {
        **resolver_inputs,
        "codebook_subset_by_cell": {
            "cell-3": None,
            "cell-4": "partialAndNonCoherent",
        },
        "max_rank_for_codebook_by_cell": {"cell-3": None, "cell-4": 2},
        "ul_full_power_transmission_by_cell": {
            "cell-3": None,
            "cell-4": "fullpowerMode1",
        },
    }
    full_power_mode1_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **full_power_mode1_inputs, scheduled_cells_indicator_value=2
    )
    assert full_power_mode1_widths["precoding_information_and_number_of_layers"] == 5
    two_port_full_power_inputs = {
        **full_power_mode1_inputs,
        "codebook_srs_antenna_port_counts_by_cell": {
            "cell-3": (),
            "cell-4": (2,),
        },
        "codebook_subset_by_cell": {"cell-3": None, "cell-4": "nonCoherent"},
    }
    two_port_full_power_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **two_port_full_power_inputs, scheduled_cells_indicator_value=2
    )
    assert two_port_full_power_widths["precoding_information_and_number_of_layers"] == 2
    with pytest.raises(ValueError, match="mapping types must be unique"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "dmrs_port_configuration_by_cell": {
                    **resolver_inputs["dmrs_port_configuration_by_cell"],
                    "cell-4": (
                        {
                            "mapping_type": "B",
                            "dmrs_type": 2,
                            "dmrs_type_enhanced": True,
                            "max_length": 2,
                        },
                        {
                            "mapping_type": "B",
                            "dmrs_type": 1,
                            "dmrs_type_enhanced": False,
                            "max_length": 1,
                        },
                    ),
                },
            },
            scheduled_cells_indicator_value=2,
        )
    capability_fallback_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "max_noncodebook_layers_by_cell": {"cell-3": None, "cell-4": None},
            "ue_max_noncodebook_layers_by_cell": {"cell-3": 1, "cell-4": None},
        },
        scheduled_cells_indicator_value=2,
    )
    assert capability_fallback_widths["srs_resource_indicator"] == 3
    unsupported_config_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "ue_supports_max_mimo_layers_by_cell": {
                "cell-3": False,
                "cell-4": True,
            },
            "ue_max_noncodebook_layers_by_cell": {"cell-3": 1, "cell-4": None},
        },
        scheduled_cells_indicator_value=2,
    )
    assert unsupported_config_widths["srs_resource_indicator"] == 3
    with pytest.raises(ValueError, match="UE non-codebook layer capability is required"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "max_noncodebook_layers_by_cell": {
                    "cell-3": None,
                    "cell-4": None,
                },
            },
            scheduled_cells_indicator_value=2,
        )
    with pytest.raises(ValueError, match="sri_mode"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "scheduled_cells_indicator_value": 2,
                "sri_mode": "type3",
            }
        )
    with pytest.raises(ValueError, match="cover the full scheduled cell list"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "srs_resource_count_by_cell": {"cell-3": 2},
            },
            scheduled_cells_indicator_value=2,
        )
    assert widths["first_downlink_assignment_index"] == 2
    assert widths["second_downlink_assignment_index"] == 2
    assert widths["srs_request"] == 2
    assert widths["srs_offset_indicator"] == 3
    assert widths["csi_request"] == 3
    srs_offset = sum(widths[name] for name in field_names[: field_names.index("srs_request")])
    payload = [0] * sum(widths.values())
    payload[srs_offset : srs_offset + 2] = [1, 0]
    assert (
        nr_srs_dci_0_3_request_list_index_from_layout(
            payload, field_widths_bits=widths, request_list_entry_count=3
        )
        == 2
    )

    single_cell_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs, scheduled_cells_indicator_value=0
    )
    assert single_cell_widths["tpc_command_for_scheduled_pusch"] == 2
    assert single_cell_widths["harq_process_number"] == 3
    assert single_cell_widths["modulation_and_coding_scheme"] == 5
    assert single_cell_widths["new_data_indicator"] == 2
    assert single_cell_widths["redundancy_version"] == 4
    assert single_cell_widths["frequency_domain_resource_assignment"] == 5
    assert single_cell_widths["srs_resource_indicator"] == 2
    assert single_cell_widths["ptrs_dmrs_association"] == 0
    cell_four_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs, scheduled_cells_indicator_value=1
    )
    assert cell_four_widths["ptrs_dmrs_association"] == 2
    assert cell_four_widths["open_loop_power_control_parameter_set_indication"] == 2
    assert single_cell_widths["open_loop_power_control_parameter_set_indication"] == 1
    rank_one_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "max_pusch_rank_by_cell": {"cell-3": 1, "cell-4": 1},
        },
        scheduled_cells_indicator_value=2,
    )
    assert rank_one_widths["ptrs_dmrs_association"] == 0
    with pytest.raises(TypeError, match="must both be supplied"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                key: value
                for key, value in resolver_inputs.items()
                if key != "max_pusch_rank_by_cell"
            },
            scheduled_cells_indicator_value=2,
        )

    validation_cases = (
        ({"cell_set_configs": "not-a-roster"}, TypeError, "ordered sequence"),
        ({"selected_cell_set_id": None}, ValueError, "multiple cell sets require"),
        ({"selected_cell_set_id": 8}, ValueError, "identify exactly one configured cell set"),
        ({"tdra_field_index_list_entry_count": True}, TypeError, "must be an integer"),
        ({"tdra_field_index_list_entry_count": 0}, ValueError, "must be in \\[1, 4096\\]"),
        ({"frequency_hopping_enabled_by_cell": "bad"}, TypeError, "must be a mapping"),
        (
            {"frequency_hopping_enabled_by_cell": {"cell-3": True}},
            ValueError,
            "must cover the full scheduled cell list",
        ),
        (
            {"frequency_hopping_enabled_by_cell": {"cell-3": 1, "cell-4": False}},
            TypeError,
            "values must be booleans",
        ),
        (
            {"transform_precoding_enabled_by_cell": "bad"},
            TypeError,
            "antenna-port configuration inputs must be mappings",
        ),
        (
            {"transform_precoding_enabled_by_cell": {"cell-3": True}},
            ValueError,
            "must cover the full scheduled cell list",
        ),
        (
            {
                "ptrs_uplink_configured_by_cell": {"cell-3": True, "cell-4": True},
                "max_pusch_rank_by_cell": None,
            },
            TypeError,
            "must both be supplied",
        ),
        (
            {"max_pusch_rank_by_cell": {"cell-3": 0, "cell-4": 3}},
            ValueError,
            "maximum PUSCH rank must be in \\[1, 8\\]",
        ),
        (
            {"p0_pusch_set_list_configured_by_cell": {"cell-3": 1, "cell-4": False}},
            TypeError,
            "P0-PUSCH set-list configuration values must be booleans",
        ),
        (
            {"sri_field_present_by_cell": {"cell-3": True, "cell-4": 0}},
            TypeError,
            "SRI field-presence values must be booleans",
        ),
        (
            {"olpc_parameter_set_dci_0_1_bits_by_cell": {"cell-3": True, "cell-4": 2}},
            TypeError,
            "OLPC indication width.*must be an integer",
        ),
        (
            {"beta_offsets_semi_static_by_cell": {"cell-3": True, "cell-4": 1}},
            TypeError,
            "beta-offset configuration values must be booleans",
        ),
        ({"harq_ack_codebook_types": ()}, ValueError, "one or two HARQ-ACK codebook types"),
        (
            {"harq_ack_codebook_types": ("semi_static", "unknown")},
            ValueError,
            "types must be 'semi_static' or 'dynamic'",
        ),
        (
            {"dynamic_harq_ack_two_sub_codebooks_by_codebook": (False,)},
            ValueError,
            "state sequences must have equal lengths",
        ),
        (
            {"dynamic_harq_ack_two_sub_codebooks_by_codebook": (False, 1)},
            TypeError,
            "two-sub-codebook flags must be booleans",
        ),
        (
            {
                "harq_ack_codebook_types": ("semi_static", "dynamic"),
                "dynamic_harq_ack_two_sub_codebooks_by_codebook": (True, False),
            },
            ValueError,
            "semi-static HARQ-ACK cannot configure dynamic sub-codebooks",
        ),
        (
            {"channel_access_config_entry_count_by_cell": {"cell-3": True, "cell-4": 0}},
            TypeError,
            "entry count.*must be an integer",
        ),
        (
            {"channel_access_config_entry_count_by_cell": {"cell-3": 65, "cell-4": 0}},
            ValueError,
            "entry counts must be in \\[0, 64\\]",
        ),
        (
            {"harq_process_number_size_bits_by_cell": {"cell-3": 3, "cell-4": 6}},
            ValueError,
            "HARQ-process widths must be in \\[0, 5\\]",
        ),
        (
            {"max_schedulable_pusch_count_by_cell": {"cell-3": 2, "cell-4": 0}},
            ValueError,
            "schedulable PUSCH counts must be in \\[1, 64\\]",
        ),
        (
            {"redundancy_version_bits_by_cell": {"cell-3": 2, "cell-4": 3}},
            ValueError,
            "redundancy-version widths must be in \\[0, 2\\]",
        ),
        (
            {"priority_indicator_configured": 1},
            TypeError,
            "priority_indicator_configured must be a boolean",
        ),
        (
            {"minimum_scheduling_offset_configured": 1},
            TypeError,
            "minimum_scheduling_offset_configured must be a boolean",
        ),
        (
            {"measurement_gap_cancellation_configured": 1},
            TypeError,
            "measurement_gap_cancellation_configured must be a boolean",
        ),
        (
            {"request_list_entry_count": True},
            TypeError,
            "request_list_entry_count must be an integer",
        ),
        (
            {"request_list_entry_count": 17},
            ValueError,
            "request_list_entry_count must be in \\[0, 16\\]",
        ),
        ({"srs_offset_list": "bad"}, TypeError, "must be an ordered sequence or None"),
        ({"srs_offset_list": ()}, ValueError, "must contain 1..8 configured entries"),
        ({"srs_offset_list": ((0,),)}, ValueError, "one index per scheduled cell"),
        ({"srs_offset_list": ((0, True),)}, ValueError, "indexes must be integers in \\[0, 7\\]"),
        ({"report_trigger_size": True}, TypeError, "report_trigger_size must be an integer"),
        ({"report_trigger_size": 7}, ValueError, "report_trigger_size must be in \\[0, 6\\]"),
    )
    for changes, error, message in validation_cases:
        arguments = {**resolver_inputs, **changes}
        with pytest.raises(error, match=message):
            nr_srs_dci_0_3_field_widths_from_configuration(
                **arguments, scheduled_cells_indicator_value=2
            )


def test_dci_0_3_field_widths_derive_absent_frequency_hopping_and_single_tdra_entry() -> None:
    field_names = (
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
    derived_fields = {
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
    assert set(derived_fields) == set(field_names)
    widths = nr_srs_dci_0_3_field_widths_from_configuration(
        cell_set_configs=(
            {
                "set_id": 0,
                "scheduled_cell_ids": ("cell-1", "cell-2"),
                "scheduled_cell_combos": ((0, 1),),
            },
        ),
        configured_ul_bwp_ids_by_cell={"cell-1": (0,), "cell-2": (0,)},
        active_ul_bwp_id_by_cell={"cell-1": 0, "cell-2": 0},
        ul_bwp_frequency_allocation_by_cell={
            "cell-1": {0: {"size_prb": 10, "start_prb": 0, "resource_allocation": "type0"}},
            "cell-2": {0: {"size_prb": 10, "start_prb": 0, "resource_allocation": "type0"}},
        },
        request_list_entry_count=0,
        srs_offset_list=None,
        report_trigger_size=0,
        tdra_field_index_list_entry_count=1,
        frequency_hopping_enabled_by_cell={"cell-1": False, "cell-2": False},
        transform_precoding_enabled_by_cell={"cell-1": True, "cell-2": True},
        beta_offsets_semi_static_by_cell={"cell-1": True, "cell-2": True},
        channel_access_config_entry_count_by_cell={"cell-1": 0, "cell-2": 0},
        harq_process_number_size_bits_by_cell={"cell-1": 0, "cell-2": 0},
        max_schedulable_pusch_count_by_cell={"cell-1": 1, "cell-2": 1},
        redundancy_version_bits_by_cell={"cell-1": 0, "cell-2": 0},
        sri_mode="type1a",
        tpmi_mode="type1a",
        antenna_ports_mode="type1a",
        tx_config_by_cell={"cell-1": "codebook", "cell-2": "nonCodebook"},
        srs_resource_count_by_cell={"cell-1": 1, "cell-2": 3},
        max_noncodebook_layers_by_cell={"cell-1": None, "cell-2": 1},
        ue_supports_max_mimo_layers_by_cell={"cell-1": True, "cell-2": True},
        dmrs_port_configuration_by_cell={
            "cell-1": (
                {"mapping_type": "A", "dmrs_type": 1, "dmrs_type_enhanced": False, "max_length": 1},
            ),
            "cell-2": (
                {"mapping_type": "B", "dmrs_type": 1, "dmrs_type_enhanced": False, "max_length": 1},
            ),
        },
        codebook_srs_antenna_port_counts_by_cell={"cell-1": (1,), "cell-2": ()},
        codebook_subset_by_cell={"cell-1": None, "cell-2": None},
        max_rank_for_codebook_by_cell={"cell-1": None, "cell-2": None},
        ul_full_power_transmission_by_cell={"cell-1": None, "cell-2": None},
        harq_ack_codebook_types=("semi_static",),
        dynamic_harq_ack_two_sub_codebooks_by_codebook=(False,),
        priority_indicator_configured=False,
        minimum_scheduling_offset_configured=True,
    )
    assert widths["time_domain_resource_assignment"] == 0
    assert widths["frequency_hopping_flag"] == 0
    assert widths["dmrs_sequence_initialization"] == 0
    assert widths["ul_sch_indicator"] == 1
    assert widths["priority_indicator"] == 0
    assert widths["minimum_applicable_scheduling_offset_indicator"] == 1
    assert widths["beta_offset_indicator"] == 0
    assert widths["channel_access_cpext_capc"] == 0
    assert widths["tpc_command_for_scheduled_pusch"] == 4
    assert widths["harq_process_number"] == 0
    assert widths["modulation_and_coding_scheme"] == 10
    assert widths["new_data_indicator"] == 2
    assert widths["redundancy_version"] == 0
    assert widths["srs_resource_indicator"] == 2
    assert widths["first_downlink_assignment_index"] == 1
    assert widths["second_downlink_assignment_index"] == 0


@pytest.mark.parametrize(
    ("entry_count", "payload", "expected"),
    [(0, (), None), (1, (), 0)],
)
def test_dci_0_3_layout_request_selector_handles_absent_and_single_entry(
    entry_count, payload, expected
) -> None:
    widths = dict.fromkeys(_DCI_0_3_FIELD_ORDER, 0)

    assert (
        nr_srs_dci_0_3_request_list_index_from_layout(
            payload, field_widths_bits=widths, request_list_entry_count=entry_count
        )
        == expected
    )


def test_dci_0_3_layout_request_selector_rejects_reserved_selector() -> None:
    widths = dict.fromkeys(_DCI_0_3_FIELD_ORDER, 0)
    widths["srs_request"] = 2

    with pytest.raises(ValueError, match="reserved codepoint"):
        nr_srs_dci_0_3_request_list_index_from_layout(
            (1, 1), field_widths_bits=widths, request_list_entry_count=3
        )


@pytest.mark.parametrize(
    ("sequences", "ports8_tdm", "four_port_srs_3tx", "error", "message"),
    [
        (np.ones((1, 1, 24)), False, False, TypeError, "complex-valued"),
        (
            np.ones((1, 1, 24), dtype=np.complex128),
            1,
            False,
            TypeError,
            "ports8_tdm must be a boolean",
        ),
        (np.ones((1, 1, 24), dtype=np.complex128), True, False, ValueError, "requires eight ports"),
        (np.ones((8, 1, 24), dtype=np.complex128), True, True, ValueError, "cannot combine"),
    ],
)
def test_srs_multiport_mapper_rejects_invalid_port_mode_combinations(
    sequences, ports8_tdm, four_port_srs_3tx, error, message
) -> None:
    with pytest.raises(error, match=message):
        map_nr_srs_multiport_resource(
            sequences,
            slot_symbol_count=14,
            subcarrier_count=48,
            start_position_offset=1,
            srs_symbol_count=1,
            comb_size=2,
            comb_offset=0,
            ports8_tdm=ports8_tdm,
            four_port_srs_3tx=four_port_srs_3tx,
        )
