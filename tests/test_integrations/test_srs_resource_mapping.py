"""Regression definitions for bounded TS 38.211 SRS resource mapping."""

import json
from pathlib import Path

import numpy as np
import pytest
from integrations.sionna.srs import (
    NRSRSFixedFrequencyAllocation,
    NRSRSScheduledResourceOccurrence,
    audit_nr_srs_intercell_re_overlap,
    map_nr_srs_resource,
    nr_srs_fixed_frequency_allocation,
)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"slot_symbol_count": 13}, ValueError, "slot_symbol_count"),
        ({"srs_symbol_count": 3}, ValueError, "SRS symbol count"),
        ({"start_position_offset": 1}, ValueError, "start_position_offset"),
        ({"comb_size": 3}, ValueError, "comb_size"),
        ({"comb_offset": 2}, ValueError, "comb_offset"),
        ({"frequency_start_subcarrier": 48}, ValueError, "no resource elements"),
        ({"allocated_bandwidth_subcarriers": 49}, ValueError, "allocated bandwidth"),
        ({"allocated_bandwidth_subcarriers": True}, TypeError, "must be an integer"),
        ({"beta_srs": float("nan")}, ValueError, "beta_srs"),
    ],
)
def test_srs_mapper_rejects_invalid_resource_configuration(overrides, error, message) -> None:
    config = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 3,
        "srs_symbol_count": 4,
        "comb_size": 2,
        "comb_offset": 1,
    }
    config.update(overrides)
    with pytest.raises(error, match=message):
        map_nr_srs_resource(np.ones(24, dtype=np.complex128), **config)


def test_srs_intercell_overlap_audit_counts_shared_resource_elements_only() -> None:
    common = nr_srs_fixed_frequency_allocation(
        0,
        transmission_comb=2,
        b_srs=0,
        b_hop=0,
        comb_offset=0,
        reference_subcarrier=24,
    )
    disjoint_comb = nr_srs_fixed_frequency_allocation(
        0,
        transmission_comb=2,
        b_srs=0,
        b_hop=0,
        comb_offset=1,
        reference_subcarrier=24,
    )

    def occurrence(
        cell_id: str,
        allocation: NRSRSFixedFrequencyAllocation,
        *,
        slot: int = 4,
        symbol: int = 12,
    ) -> NRSRSScheduledResourceOccurrence:
        return NRSRSScheduledResourceOccurrence(
            cell_id=cell_id,
            resource_set_id=0,
            resource_id=0,
            grid_id="carrier-a-grid",
            carrier_hz=3.5e9,
            numerology=1,
            subcarrier_spacing_hz=30e3,
            target_slot_index=slot,
            slot_symbol_count=14,
            start_symbol_index=symbol,
            srs_symbol_count=1,
            allocations=(allocation,),
        )

    report = audit_nr_srs_intercell_re_overlap(
        [occurrence("cell-a", common), occurrence("cell-b", common)]
    )
    assert report["status"] == "INTERCELL_RE_OVERLAP_DETECTED"
    assert report["shared_intercell_resource_element_count"] == common.bandwidth.sequence_length
    assert report["pairwise_resource_overlaps"][0]["shared_resource_element_count"] == (
        common.bandwidth.sequence_length
    )
    assert "not proof of interference" in report["limitations"][0]
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/nr-srs-intercell-re-overlap.v1.schema.json"
    )
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
        report
    )

    no_overlap = audit_nr_srs_intercell_re_overlap(
        [
            occurrence("cell-a", common),
            occurrence("cell-b", disjoint_comb),
            occurrence("cell-c", common, slot=5),
        ]
    )
    assert no_overlap["status"] == "NO_INTERCELL_RE_OVERLAP"
    assert no_overlap["collision_coordinates"] == []
