"""Path-oracle layer metadata: the v1 storage contract accepts one shape and rejects every other."""

from __future__ import annotations

import copy

import pytest
from datasets.episode_store import storage
from datasets.episode_store.storage import (
    _PATH_ORACLE_METADATA_FIELDS,
    _PATH_ORACLE_OPTIONAL_METADATA_FIELDS,
    _PATH_ORACLE_REQUIRED_ARRAYS,
    _validate_path_oracle_layer,
)


def _metadata(**overrides) -> dict:
    metadata = {field: None for field in _PATH_ORACLE_METADATA_FIELDS}
    metadata.update(
        {
            "schema": "umcd.path_oracle.v1",
            "link_ids": ["gNB-1-to-UE-1"],
            "solve_status": ["solved"],
            "path_kind": None,
            "coefficient_convention": storage.PATH_COEFFICIENT_CONVENTION,
            "optional_fields": {
                field: False for field in _PATH_ORACLE_OPTIONAL_METADATA_FIELDS
            },
        }
    )
    metadata["optional_fields"]["interaction_vertex_frame"] = None
    metadata.update(overrides)
    return metadata


def _arrays():
    return set(_PATH_ORACLE_REQUIRED_ARRAYS)


def test_the_v1_contract_accepts_a_minimal_layer() -> None:
    _validate_path_oracle_layer(_metadata(), _arrays())


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("not-a-dict", "metadata fields do not match"),
        ({"schema": "umcd.path_oracle.v1"}, "metadata fields do not match"),
        (_metadata(schema="umcd.path_oracle.v9"), "unsupported path-oracle layer schema"),
        (_metadata(link_ids=[]), "link_ids must be non-empty strings"),
        (_metadata(link_ids=["  "]), "link_ids must be non-empty strings"),
        (_metadata(link_ids=[1]), "link_ids must be non-empty strings"),
        (_metadata(solve_status=["solved", "solved"]), "solve_status must match link_ids"),
        (_metadata(solve_status=["bogus"]), "solve_status contains an unknown value"),
        (_metadata(solve_status=[7]), "solve_status contains an unknown value"),
        (_metadata(path_kind=[""]), "path_kind must be null or an array"),
        (_metadata(path_kind="specular"), "path_kind must be null or an array"),
        (_metadata(coefficient_convention="other"), "coefficient_convention is unsupported"),
    ],
)
def test_top_level_metadata_is_validated(metadata, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_path_oracle_layer(metadata, _arrays())


def test_optional_field_block_must_match_the_contract() -> None:
    metadata = _metadata()
    metadata["optional_fields"] = {"doppler_hz": False}
    with pytest.raises(ValueError, match="optional_fields do not match the v1 contract"):
        _validate_path_oracle_layer(metadata, _arrays())
    metadata = _metadata()
    metadata["optional_fields"]["doppler_hz"] = 0
    with pytest.raises(ValueError, match="optional_fields.doppler_hz must be boolean"):
        _validate_path_oracle_layer(metadata, _arrays())


def test_interaction_geometry_needs_a_named_vertex_frame() -> None:
    metadata = _metadata()
    metadata["optional_fields"]["interaction_geometry"] = True
    with pytest.raises(ValueError, match="requires a named vertex frame"):
        _validate_path_oracle_layer(metadata, _arrays())
    metadata["optional_fields"]["interaction_vertex_frame"] = "  "
    with pytest.raises(ValueError, match="requires a named vertex frame"):
        _validate_path_oracle_layer(metadata, _arrays())


def test_vertex_frame_without_interaction_geometry_is_refused() -> None:
    metadata = _metadata()
    metadata["optional_fields"]["interaction_vertex_frame"] = "scene_xyz_m"
    with pytest.raises(ValueError, match="vertex frame requires interaction geometry"):
        _validate_path_oracle_layer(metadata, _arrays())


def test_optional_arrays_are_required_only_when_their_flag_is_set() -> None:
    metadata = _metadata()
    metadata["optional_fields"]["doppler_hz"] = True
    with pytest.raises(ValueError):
        _validate_path_oracle_layer(metadata, _arrays())
    assert copy.deepcopy(metadata)
