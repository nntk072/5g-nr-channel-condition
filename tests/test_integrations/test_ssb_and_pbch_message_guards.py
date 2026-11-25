"""SS/PBCH block scaling and BCCH-BCH message dispatch: shape, dtype, finiteness and alternative checks."""

from __future__ import annotations

import integrations.sionna.ssb_grid as ssb_grid_module
import numpy as np
import pytest
from integrations.sionna import pbch_dmrs
from integrations.sionna.pbch_dmrs import PBCHDMRSResourceMap
from integrations.sionna.pbch_mib import (
    PBCHMIB,
    PBCHMessageClassExtension,
    nr_bcch_bch_message_parse,
    nr_bcch_bch_message_serialize,
    nr_pbch_mib_serialize,
)
from integrations.sionna.ssb_grid import (
    map_nr_ssb_resource_grid,
    scale_nr_ssb_resource_grid_epre,
    scale_nr_ssb_resource_grid_power,
)


def test_pbch_dmrs_coordinate_count_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pbch_dmrs, "_DMRS_SYMBOL_COUNT", 145)

    with pytest.raises(RuntimeError, match="coordinate count is inconsistent"):
        pbch_dmrs._pbch_dmrs_coordinates(0)


def test_bcch_bch_message_serializer_accepts_mib_alternative() -> None:
    message = PBCHMIB(
        system_frame_number_msb=3,
        subcarrier_spacing_common="scs30or120",
        ssb_subcarrier_offset=7,
        dmrs_type_a_position="pos3",
        control_resource_set_zero=11,
        search_space_zero=14,
        cell_barred="barred",
        intra_freq_reselection="notAllowed",
        spare=1,
    )

    encoded = nr_bcch_bch_message_serialize(message)

    assert encoded.shape == (24,)
    assert encoded[0] == 0
    assert nr_bcch_bch_message_parse(encoded) == message


def test_epre_ratio_must_be_zero_or_three_db() -> None:
    grid = np.ones((4, 240), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be 0 or 3 dB"):
        scale_nr_ssb_resource_grid_epre(grid, pss_to_sss_epre_db=1.5)
    with pytest.raises(TypeError, match="must be a real number"):
        scale_nr_ssb_resource_grid_epre(grid, pss_to_sss_epre_db=True)


@pytest.mark.parametrize(
    ("grid", "error", "message"),
    [
        pytest.param(np.ones((4, 239)), ValueError, "must have shape", id="shape"),
        pytest.param(np.zeros((4, 240), dtype=bool), TypeError, "real or complex", id="boolean"),
        pytest.param(
            np.where(np.arange(4 * 240).reshape(4, 240) == 3, np.nan, 1.0),
            ValueError,
            "must be finite",
            id="nan",
        ),
    ],
)
def test_ssb_grid_shape_dtype_and_finiteness(grid, error, message) -> None:
    with pytest.raises(error, match=message):
        scale_nr_ssb_resource_grid_epre(grid, pss_to_sss_epre_db=0.0)


@pytest.mark.parametrize(
    ("grid", "ratio", "error", "message"),
    [
        (np.ones((4, 240)), True, TypeError, "must be a real number"),
        (np.ones((4, 239)), 0.0, ValueError, "must have shape"),
        (np.zeros((4, 240), dtype=bool), 0.0, TypeError, "real or complex"),
        (
            np.where(np.arange(4 * 240).reshape(4, 240) == 3, np.nan, 1.0),
            0.0,
            ValueError,
            "must be finite",
        ),
    ],
)
def test_ssb_power_scaling_validates_grid_and_ratio(grid, ratio, error, message) -> None:
    with pytest.raises(error, match=message):
        scale_nr_ssb_resource_grid_power(
            grid, ss_pbch_block_power_dbm=-30.0, pss_to_sss_epre_db=ratio
        )


def test_pbch_data_mapping_checks_the_internal_resource_count(monkeypatch) -> None:
    invalid_dmrs = PBCHDMRSResourceMap(
        physical_cell_id=0,
        ssb_index_in_half_frame=0,
        half_frame_index=0,
        max_candidate_blocks_per_half_frame=4,
        c_init=0,
        subcarriers=np.asarray([0], dtype=np.int16),
        ofdm_symbols=np.asarray([1], dtype=np.int8),
        sequence=np.asarray([1.0 + 0.0j]),
    )
    monkeypatch.setattr(ssb_grid_module, "nr_pbch_dmrs_ssb", lambda *_args, **_kwargs: invalid_dmrs)

    with pytest.raises(RuntimeError, match="resource count is inconsistent"):
        map_nr_ssb_resource_grid(
            0,
            0,
            np.ones(432, dtype=np.complex128),
            ssb_index_in_half_frame=0,
            half_frame_index=0,
            max_candidate_blocks_per_half_frame=4,
        )


def test_bcch_bch_alternatives_round_trip_through_their_serializer() -> None:
    extension = PBCHMessageClassExtension()
    bits = nr_bcch_bch_message_serialize(extension)
    assert bits.shape == (24,)
    assert bits[0] == 1
    restored = nr_bcch_bch_message_parse(bits)
    assert isinstance(restored, PBCHMessageClassExtension)


def test_unknown_message_alternatives_are_refused() -> None:
    with pytest.raises(TypeError, match="must be PBCHMIB or PBCHMessageClassExtension"):
        nr_bcch_bch_message_serialize("not-a-message")
    with pytest.raises(TypeError, match="mib must be a PBCHMIB"):
        nr_pbch_mib_serialize("not-a-mib")
